"""
A3C Training V3.5 - COLAB VERSION - Temporal Per-Pixel LSTM (FULL POWER)

Optimized for Google Colab with HIGH RAM + GPU:
1. Full 5 timestep window (as per architecture plan)
2. Multiple workers (4-8) for fast parallel training
3. Larger grids (150K cells) - no artificial limits
4. Mixed precision training for speed
5. Aggressive GPU utilization
6. Full architecture plan implementation - NO COMPROMISES

Expected Colab resources:
- RAM: 25GB+ (High-RAM runtime)
- GPU: V100/A100 (16GB+ VRAM)
- Workers: 4-8 parallel
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
from a3c.worker_v3_5_colab import worker_process_temporal_colab
from wildfire_env_temporal_v3_5 import WildfireEnvTemporal


def create_filtered_episode_list(env_paths, max_file_size_mb=50, min_episode_length=4, temporal_window=5, max_grid_cells=150000):
    """
    Pre-scan environments and create list of (env_path, start_timestep) pairs.

    COLAB VERSION - FULL POWER:
    - Max 150K grid cells (387×387) - Colab can handle it!
    - Full 5 timestep window (architecture plan spec)
    - With chunked LSTM (10K chunks): ~4GB RAM per worker
    - With 4-8 workers: 16-32GB total RAM (Colab High-RAM: 25GB+)
    - NO ARTIFICIAL LIMITS - let Colab's resources shine!

    Args:
        env_paths: List of environment file paths
        max_file_size_mb: Skip files larger than this
        min_episode_length: Minimum timesteps with burns required
        temporal_window: Temporal window size (default 5 for V3.5)
        max_grid_cells: Maximum H×W cells (default 150K for Colab)

    Returns:
        List of (env_path, start_timestep, episode_length) tuples
    """
    import gc

    MAX_SIZE = max_file_size_mb * 1024 * 1024
    filtered_episodes = []

    print(f"Scanning {len(env_paths)} environments for good episodes...")
    print(f"Filtering: file < {max_file_size_mb}MB, grid < {max_grid_cells:,} cells, min {min_episode_length} steps with burns")
    print(f"Temporal window: {temporal_window} timesteps (FULL ARCHITECTURE PLAN)")

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

            # COLAB: More generous grid filtering
            grid_cells = env.H * env.W
            if grid_cells > max_grid_cells:
                total_skipped_large_grid += 1
                del env
                continue

            # Find timesteps with burns in future
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
                    max_length = min(env.T - start_t - 1, 8)
                    filtered_episodes.append((env_path, start_t, max_length))
                    total_episodes_found += 1
                    env_has_episodes = True

            if env_has_episodes:
                total_envs_kept += 1

            # Delete environment and force GC every 50 envs
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
    parser = argparse.ArgumentParser(description='A3C V3.5 Training - Colab FULL POWER')
    parser.add_argument('--repo-root', type=str, default='/content/WildfirePrediction')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of parallel workers (4-8 for Colab)')
    parser.add_argument('--max-episodes', type=int, default=1000, help='Total episodes across all workers')
    parser.add_argument('--temporal-window', type=int, default=5, help='Temporal window size (5 per architecture plan)')
    parser.add_argument('--lr', type=float, default=7e-5, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--value-loss-coef', type=float, default=0.5, help='Value loss coefficient')
    parser.add_argument('--entropy-coef', type=float, default=0.015, help='Entropy coefficient')
    parser.add_argument('--max-grad-norm', type=float, default=0.5, help='Max gradient norm')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max-envs', type=int, default=None, help='Limit number of training environments')
    parser.add_argument('--max-file-size-mb', type=int, default=50, help='Max environment file size in MB')
    parser.add_argument('--max-grid-cells', type=int, default=150000, help='Max grid size (default 150K for Colab)')
    parser.add_argument('--min-episode-length', type=int, default=4, help='Min timesteps with burns per episode')
    parser.add_argument('--log-interval', type=int, default=10, help='Log every N episodes')
    parser.add_argument('--wandb-project', type=str, default='wildfire-prediction-colab', help='WandB project name')
    parser.add_argument('--wandb-run-name', type=str, default=None, help='WandB run name')
    parser.add_argument('--no-wandb', action='store_true', help='Disable WandB logging')
    parser.add_argument('--mixed-precision', action='store_true', help='Use mixed precision training (faster)')
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

    # Determine device (Colab has GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"=" * 80)
    print(f"A3C V3.5 COLAB - Temporal Per-Pixel LSTM (FULL ARCHITECTURE PLAN)")
    print(f"=" * 80)
    print(f"🚀 COLAB UNLIMITED POWER MODE")
    print(f"=" * 80)
    print(f"Problem: Per-cell 8-neighbor + Per-Pixel LSTM temporal context")
    print(f"Temporal window: {args.temporal_window} timesteps (FULL 5 as per plan!)")
    print(f"Temporal model: Per-Pixel LSTM (2 layers, hidden=128)")
    print(f"LSTM chunks: 10K pixels (~40MB/chunk)")
    print(f"Architecture: Hybrid CPU-GPU (global model on GPU, workers on CPU)")
    print(f"Device: {device}")
    print(f"Grid filter: max {args.max_grid_cells:,} cells (387×387, FULL POWER)")
    print(f"Workers: {args.num_workers} parallel (4-8x faster than local!)")
    print(f"Mixed precision: {'ENABLED' if args.mixed_precision else 'DISABLED'}")
    print(f"Expected memory: ~{args.num_workers * 4}GB RAM ({args.num_workers} workers × 4GB)")
    print(f"Expected VRAM: ~1GB (global model on GPU)")
    print(f"=" * 80)
    print(f"Training environments: {len(train_paths)}")
    print(f"Max episodes: {args.max_episodes}")
    print(f"Learning rate: {args.lr}")
    print(f"Colab Resources: HIGH-RAM (25GB+), GPU (V100/A100 16GB+)")
    print(f"=" * 80)

    # FILTER EPISODES (generous for Colab)
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
    print(f"Starting COLAB training with {len(filtered_episodes)} filtered episodes")
    print(f"{'=' * 80}\n")

    # Initialize WandB
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb_run_name = args.wandb_run_name or f"a3c-v3.5-COLAB-w{args.num_workers}-tw{args.temporal_window}"
        wandb.init(
            project=args.wandb_project,
            name=wandb_run_name,
            config={
                "model": "A3C-V3.5-Temporal-PerPixel-LSTM-COLAB",
                "formulation": "temporal_per_cell_8neighbor",
                "temporal_window": args.temporal_window,
                "temporal_model": "Per-Pixel-LSTM-2layer",
                "architecture": "Hybrid-CPU-GPU-COLAB",
                "device": str(device),
                "rewards": "dense_iou",
                "follows_architecture_plan": True,
                "colab_version": True,
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
                "max_grid_cells": args.max_grid_cells,
                "mixed_precision": args.mixed_precision,
            }
        )
        print(f"WandB initialized: {wandb.run.name}")
    else:
        print("WandB logging disabled")

    # Create shared model (on GPU for Colab)
    shared_model = A3C_TemporalModel(in_channels=14, temporal_window=args.temporal_window)
    shared_model = shared_model.to(device)
    shared_model.train()

    print(f"Shared model moved to GPU (device={device})")
    print(f"GPU Memory allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")

    # Count parameters
    total_params = sum(p.numel() for p in shared_model.parameters())
    trainable_params = sum(p.numel() for p in shared_model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")

    # Create timestamped checkpoint directory
    from datetime import datetime
    timestamp = datetime.now().strftime("%y%m%d-%H%M")
    ckpt_dir = repo_root / 'rl_training' / 'a3c' / 'checkpoints_v3_5_colab' / timestamp
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
        'mixed_precision': args.mixed_precision,
    }

    # Create workers
    processes = []
    for worker_id in range(args.num_workers):
        p = mp.Process(
            target=worker_process_temporal_colab,
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
        'model_state_dict': shared_model.cpu().state_dict(),
        'best_iou': global_best_iou.value,
        'total_episodes': global_episode_counter.value,
        'config': vars(args),
        'filtered_episodes_count': len(filtered_episodes),
        'formulation': 'temporal_per_cell_8neighbor_perpixel_lstm',
        'temporal_window': args.temporal_window,
        'temporal_model': 'Per-Pixel-LSTM-2layer',
        'architecture_plan': 'V3.5_ARCHITECTURE_PLAN.md',
        'colab_version': True,
        'device_used': str(device),
    }, final_path)

    print(f"\n{'=' * 80}")
    print(f"COLAB Training Complete!")
    print(f"Total episodes: {global_episode_counter.value}")
    print(f"Best IoU: {global_best_iou.value:.4f}")
    print(f"Model saved to: {final_path}")
    print(f"{'=' * 80}")

    # Log final model to WandB
    if use_wandb:
        artifact = wandb.Artifact(
            name='final-model-v3-5-perpixel-lstm-colab',
            type='model',
            description=f'A3C V3.5 COLAB (Per-Pixel LSTM, Full Power) after {global_episode_counter.value} episodes, IoU {global_best_iou.value:.4f}'
        )
        artifact.add_file(str(final_path))
        wandb.log_artifact(artifact)

        wandb.summary['total_episodes'] = global_episode_counter.value
        wandb.summary['best_iou'] = global_best_iou.value

        wandb.finish()


if __name__ == '__main__':
    main()
