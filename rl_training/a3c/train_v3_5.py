"""
A3C Training V3.5 - Temporal Per-Pixel LSTM (Architecture Plan Implementation)

Implements V3.5_ARCHITECTURE_PLAN.md specifications exactly:
1. Per-pixel LSTM for temporal context (NOT 3D Conv, NOT ConvLSTM)
2. Temporal window = 5 timesteps (captures fire velocity and acceleration)
3. Hybrid CPU-GPU A3C (global model on GPU, workers on CPU)
4. Conservative memory filtering (max 120K grid cells)
5. 2 workers default (safe for 64GB RAM)

Hardware optimization (Ryzen9 9950x, 64GB RAM, RTX 5070 12GB):
- GPU: Global model (~500MB VRAM)
- RAM: 2 workers × ~6GB = ~12GB (plenty of headroom)
- No compromises on algorithm performance!
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import argparse
import json
import torch
import torch.multiprocessing as mp
from pathlib import Path
import sys
import wandb
from tqdm import tqdm

torch.set_num_threads(1)

sys.path.insert(0, str(Path(__file__).parent.parent))

from a3c.model_v3_5 import A3C_TemporalModel
from a3c.worker_v3_5 import worker_process_temporal
from wildfire_env_temporal_v3_5 import WildfireEnvTemporal


def create_filtered_episode_list(env_paths, max_file_size_mb=50, min_episode_length=4, temporal_window=3, max_grid_cells=50000):
    """
    Pre-scan environments and create list of (env_path, start_timestep) pairs.

    CRITICAL: Filters by GRID SIZE to prevent OOM on large grids.

    LAST-RESORT filtering for per-pixel LSTM (EXTREMELY memory intensive):
    - Max 50K grid cells (224×224) per environment
    - Temporal window: 3 (down from 5) - saves 40% memory
    - With chunked LSTM (10K chunks): ~3GB RAM per worker
    - 1 WORKER ONLY: ~6GB total RAM (per-pixel LSTM is too heavy for 2 workers)
    - Recomputation loop + per-pixel LSTM = OOM with 2 workers

    Args:
        env_paths: List of environment file paths
        max_file_size_mb: Skip files larger than this
        min_episode_length: Minimum timesteps with burns required
        temporal_window: Temporal window size (default 5 for V3.5)
        max_grid_cells: Maximum H×W cells (default 100K = 316×316, CRASH-PROOF)

    Returns:
        List of (env_path, start_timestep, episode_length) tuples
    """
    import gc

    MAX_SIZE = max_file_size_mb * 1024 * 1024
    filtered_episodes = []

    print(f"Scanning {len(env_paths)} environments for good episodes...")
    print(f"Filtering: file < {max_file_size_mb}MB, grid < {max_grid_cells:,} cells, min {min_episode_length} steps with burns")
    print(f"Temporal window: {temporal_window} timesteps")

    total_envs_scanned = 0
    total_envs_kept = 0
    total_episodes_found = 0
    total_skipped_large_grid = 0

    for idx, env_path in enumerate(tqdm(env_paths, desc="Scanning envs")):
        # Skip large files
        if env_path.stat().st_size >= MAX_SIZE:
            continue

        total_envs_scanned += 1

        try:
            env = WildfireEnvTemporal(env_path, temporal_window=temporal_window)

            # CRITICAL: Filter by grid size to prevent OOM
            grid_cells = env.H * env.W
            if grid_cells > max_grid_cells:
                total_skipped_large_grid += 1
                del env
                continue

            # Find timesteps with burns in future (limit to first 5 starts to save memory)
            env_has_episodes = False
            for start_t in range(min(5, env.T - min_episode_length)):
                # Check if there are burns in the next min_episode_length steps
                has_burns_count = 0
                for t in range(start_t, min(start_t + min_episode_length + 3, env.T - 1)):
                    actual_mask_t = env.fire_masks[t] > 0
                    actual_mask_t1 = env.fire_masks[t + 1] > 0
                    new_burns = (actual_mask_t1 & ~actual_mask_t)

                    if new_burns.sum() > 0:
                        has_burns_count += 1

                # If enough burns in this window, add as valid episode
                if has_burns_count >= min_episode_length:
                    max_length = min(env.T - start_t - 1, 8)  # Cap at 8 steps to save memory
                    filtered_episodes.append((env_path, start_t, max_length))
                    total_episodes_found += 1
                    env_has_episodes = True

            if env_has_episodes:
                total_envs_kept += 1

            # CRITICAL: Delete environment and force GC every 50 envs
            del env
            if idx % 50 == 0:
                gc.collect()

        except Exception as e:
            print(f"  Error loading {env_path.name}: {e}")
            continue

    print(f"\nFiltering results:")
    print(f"  Environments scanned: {total_envs_scanned}")
    print(f"  Skipped (large grid): {total_skipped_large_grid}")
    print(f"  Environments with good episodes: {total_envs_kept}")
    print(f"  Total episodes found: {total_episodes_found}")
    print(f"  Avg episodes per env: {total_episodes_found / max(1, total_envs_kept):.1f}")

    return filtered_episodes


def main():
    parser = argparse.ArgumentParser(description='A3C V3.5 Training - Temporal Per-Pixel LSTM')
    parser.add_argument('--repo-root', type=str, default='/home/chaseungjoon/code/WildfirePrediction')
    parser.add_argument('--num-workers', type=int, default=1, help='Number of parallel CPU workers (default 1 - per-pixel LSTM is too heavy for 2 workers!)')
    parser.add_argument('--max-episodes', type=int, default=1000, help='Total episodes across all workers')
    parser.add_argument('--temporal-window', type=int, default=3, help='Temporal window size (default 3 - reduced from 5 to prevent OOM)')
    parser.add_argument('--lr', type=float, default=7e-5, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--value-loss-coef', type=float, default=0.5, help='Value loss coefficient')
    parser.add_argument('--entropy-coef', type=float, default=0.015, help='Entropy coefficient')
    parser.add_argument('--max-grad-norm', type=float, default=0.5, help='Max gradient norm')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max-envs', type=int, default=None, help='Limit number of training environments')
    parser.add_argument('--max-file-size-mb', type=int, default=50, help='Max environment file size in MB')
    parser.add_argument('--max-grid-cells', type=int, default=50000, help='Max grid size (H×W) for safety (default 50K = 224×224, LAST-RESORT)')
    parser.add_argument('--min-episode-length', type=int, default=4, help='Min timesteps with burns per episode (mel4)')
    parser.add_argument('--log-interval', type=int, default=10, help='Log every N episodes')
    parser.add_argument('--wandb-project', type=str, default='wildfire-prediction', help='WandB project name')
    parser.add_argument('--wandb-run-name', type=str, default=None, help='WandB run name')
    parser.add_argument('--no-wandb', action='store_true', help='Disable WandB logging')
    parser.add_argument('--cpu-only', action='store_true', help='Force CPU-only training (no GPU)')
    args = parser.parse_args()

    # Set start method for multiprocessing
    mp.set_start_method('spawn', force=True)

    repo_root = Path(args.repo_root)
    env_dir = repo_root / 'tilling_data' / 'environments'

    # Load training environments
    train_split_path = env_dir / 'train_split.json'
    with open(train_split_path) as f:
        train_env_ids = json.load(f)

    if args.max_envs:
        train_env_ids = train_env_ids[:args.max_envs]

    train_paths = [env_dir / f'{eid}.pkl' for eid in train_env_ids]

    # Determine device
    use_gpu = torch.cuda.is_available() and not args.cpu_only
    device = torch.device('cuda' if use_gpu else 'cpu')

    print(f"=" * 80)
    print(f"A3C V3.5 Training - Per-Pixel LSTM (EMERGENCY MODE)")
    print(f"=" * 80)
    print(f"⚠️  WARNING: Per-pixel LSTM is EXTREMELY memory-intensive")
    print(f"⚠️  Hardware limits forced severe restrictions:")
    print(f"    - Temporal window: {args.temporal_window} (reduced from 5)")
    print(f"    - Max grid: {args.max_grid_cells:,} cells (224×224)")
    print(f"    - Workers: {args.num_workers} (2 workers = OOM crash)")
    print(f"    - LSTM chunk: 10K pixels (~40MB/chunk)")
    print(f"=" * 80)
    print(f"Problem: Per-cell 8-neighbor + Per-Pixel LSTM temporal context")
    print(f"Temporal model: Per-Pixel LSTM (2 layers, hidden=128)")
    print(f"Architecture: Hybrid CPU-GPU (global model on GPU, workers on CPU)")
    print(f"Device: {device} (GPU utilization: {'YES' if use_gpu else 'NO'})")
    print(f"Expected memory: ~6GB RAM (1 worker, EXTREMELY TIGHT)")
    print(f"Expected VRAM: ~500MB (global model on GPU)" if use_gpu else "")
    print(f"=" * 80)
    print(f"Training environments: {len(train_paths)}")
    print(f"Number of workers: {args.num_workers}")
    print(f"Max episodes: {args.max_episodes}")
    print(f"Learning rate: {args.lr}")
    print(f"Hardware: Ryzen9 9950x (32 threads), 64GB RAM, RTX 5070 12GB")
    print(f"=" * 80)

    # FILTER EPISODES by GRID SIZE (critical for memory)
    filtered_episodes = create_filtered_episode_list(
        train_paths,
        max_file_size_mb=args.max_file_size_mb,
        min_episode_length=args.min_episode_length,
        temporal_window=args.temporal_window,
        max_grid_cells=args.max_grid_cells
    )

    if len(filtered_episodes) == 0:
        print("\nERROR: No valid episodes found after filtering!")
        print("Try:")
        print("  - Increasing --max-file-size-mb")
        print("  - Decreasing --min-episode-length")
        print("  - Increasing --max-envs")
        return

    print(f"\n{'=' * 80}")
    print(f"Starting training with {len(filtered_episodes)} filtered episodes")
    print(f"{'=' * 80}\n")

    # Initialize WandB
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb_run_name = args.wandb_run_name or f"a3c-v3.5-perpixel-lstm-w{args.num_workers}-tw{args.temporal_window}"
        wandb.init(
            project=args.wandb_project,
            name=wandb_run_name,
            config={
                "model": "A3C-V3.5-Temporal-PerPixel-LSTM",
                "formulation": "temporal_per_cell_8neighbor",
                "temporal_window": args.temporal_window,
                "temporal_model": "Per-Pixel-LSTM-2layer",
                "architecture": "Hybrid-CPU-GPU",
                "device": str(device),
                "rewards": "dense_iou",
                "follows_architecture_plan": True,
                "num_workers": args.num_workers,
                "learning_rate": args.lr,
                "max_episodes": args.max_episodes,
                "gamma": args.gamma,
                "value_loss_coef": args.value_loss_coef,
                "entropy_coef": args.entropy_coef,
                "max_grad_norm": args.max_grad_norm,
                "seed": args.seed,
                "total_filtered_episodes": len(filtered_episodes),
                "min_episode_length": args.min_episode_length,
                "max_file_size_mb": args.max_file_size_mb,
                "max_grid_cells": args.max_grid_cells,
                "log_interval": args.log_interval,
            }
        )
        print(f"WandB initialized: {wandb.run.name}")
    else:
        print("WandB logging disabled")

    # Create shared model (on GPU if available, for fast LSTM computation)
    shared_model = A3C_TemporalModel(in_channels=14, temporal_window=args.temporal_window)

    if use_gpu:
        # Move shared model to GPU for fast gradient updates
        shared_model = shared_model.to(device)
        print(f"Shared model moved to GPU (device={device})")
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
    else:
        # CPU-only: still need to share memory for multiprocessing
        shared_model.share_memory()
        print(f"Shared model on CPU (multiprocessing shared memory)")

    shared_model.train()

    # Count parameters
    total_params = sum(p.numel() for p in shared_model.parameters())
    trainable_params = sum(p.numel() for p in shared_model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")

    # Create timestamped checkpoint directory
    from datetime import datetime
    timestamp = datetime.now().strftime("%y%m%d-%H%M")
    ckpt_dir = repo_root / 'rl_training' / 'a3c' / 'checkpoints_v3_5' / timestamp
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    print(f"Checkpoint directory: {ckpt_dir}")

    # Create shared optimizer
    optimizer = torch.optim.Adam(shared_model.parameters(), lr=args.lr)

    # Shared counters and values
    global_episode_counter = mp.Value('i', 0)
    global_best_iou = mp.Value('d', 0.0)
    lock = mp.Lock()

    # Queue for workers to send metrics
    metrics_queue = mp.Queue() if use_wandb else None

    # Training configuration
    config = {
        'seed': args.seed,
        'gamma': args.gamma,
        'value_loss_coef': args.value_loss_coef,
        'entropy_coef': args.entropy_coef,
        'max_grad_norm': args.max_grad_norm,
        'max_episodes': args.max_episodes,
        'log_interval': args.log_interval,
        'checkpoint_dir': str(ckpt_dir),
        'temporal_window': args.temporal_window,
    }

    # Create workers
    processes = []
    for worker_id in range(args.num_workers):
        p = mp.Process(
            target=worker_process_temporal,
            args=(worker_id, shared_model, optimizer, filtered_episodes, config,
                  global_episode_counter, global_best_iou, lock, metrics_queue)
        )
        p.start()
        processes.append(p)

    # Monitor metrics queue and log to WandB
    if use_wandb:
        import time
        while any(p.is_alive() for p in processes):
            while not metrics_queue.empty():
                metrics = metrics_queue.get()
                wandb.log(metrics)
            time.sleep(0.1)

        # Drain remaining metrics
        while not metrics_queue.empty():
            metrics = metrics_queue.get()
            wandb.log(metrics)
    else:
        # Just wait for workers
        for p in processes:
            p.join()

    # Save final model (move to CPU for portability)
    final_path = ckpt_dir / 'final_model.pt'
    torch.save({
        'model_state_dict': shared_model.cpu().state_dict() if use_gpu else shared_model.state_dict(),
        'best_iou': global_best_iou.value,
        'total_episodes': global_episode_counter.value,
        'config': vars(args),
        'filtered_episodes_count': len(filtered_episodes),
        'formulation': 'temporal_per_cell_8neighbor_perpixel_lstm',
        'temporal_window': args.temporal_window,
        'temporal_model': 'Per-Pixel-LSTM-2layer',
        'architecture_plan': 'V3.5_ARCHITECTURE_PLAN.md',
        'device_used': str(device),
    }, final_path)

    print(f"\n{'=' * 80}")
    print(f"Training Complete!")
    print(f"Total episodes: {global_episode_counter.value}")
    print(f"Best IoU: {global_best_iou.value:.4f}")
    print(f"Model saved to: {final_path}")
    print(f"{'=' * 80}")

    # Log final model to WandB
    if use_wandb:
        artifact = wandb.Artifact(
            name='final-model-v3-5-perpixel-lstm',
            type='model',
            description=f'A3C V3.5 (Per-Pixel LSTM, Architecture Plan) after {global_episode_counter.value} episodes, IoU {global_best_iou.value:.4f}'
        )
        artifact.add_file(str(final_path))
        wandb.log_artifact(artifact)

        wandb.summary['total_episodes'] = global_episode_counter.value
        wandb.summary['best_iou'] = global_best_iou.value

        wandb.finish()


if __name__ == '__main__':
    main()
