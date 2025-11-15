"""
A3C Training V7 - Temporal LSTM

Per-cell 8-neighbor prediction with temporal context (last 3 timesteps).
Dense rewards at every timestep.
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
import gc

torch.set_num_threads(1)

sys.path.insert(0, str(Path(__file__).parent.parent))

from a3c.model_v7 import A3C_TemporalModel
from a3c.worker_v7 import worker_process_temporal
from wildfire_env_temporal import WildfireEnvTemporal


def create_filtered_episode_list(env_paths, max_file_size_mb=10, min_episode_length=3, window_size=3, max_envs_to_scan=None):
    """
    Pre-scan environments and create list of (env_path, start_timestep) pairs
    where episodes have actual fire spread.

    Args:
        env_paths: List of environment file paths
        max_file_size_mb: Skip files larger than this
        min_episode_length: Minimum timesteps with burns required
        window_size: Temporal window size (for temporal context)
        max_envs_to_scan: Maximum number of environments to scan (for memory safety)

    Returns:
        List of (env_path, start_timestep, episode_length) tuples
    """
    MAX_SIZE = max_file_size_mb * 1024 * 1024
    filtered_episodes = []

    # Limit environments to scan for memory safety
    if max_envs_to_scan:
        env_paths = env_paths[:max_envs_to_scan]

    print(f"Scanning {len(env_paths)} environments for good episodes...")
    print(f"Filtering: file size < {max_file_size_mb}MB, min {min_episode_length} steps with burns")

    total_envs_scanned = 0
    total_envs_kept = 0
    total_episodes_found = 0

    for idx, env_path in enumerate(tqdm(env_paths, desc="Scanning envs")):
        # Skip large files
        if env_path.stat().st_size >= MAX_SIZE:
            continue

        total_envs_scanned += 1

        try:
            env = WildfireEnvTemporal(env_path, window_size=window_size, max_cache_size=10)

            # Find timesteps with burns in future
            env_has_episodes = False
            for start_t in range(env.T - min_episode_length):
                # Check if there are burns in the next min_episode_length steps
                has_burns_count = 0
                for t in range(start_t, min(start_t + min_episode_length + 5, env.T - 1)):
                    actual_mask_t = env.fire_masks[t] > 0
                    actual_mask_t1 = env.fire_masks[t + 1] > 0
                    new_burns = (actual_mask_t1 & ~actual_mask_t)

                    if new_burns.sum() > 0:
                        has_burns_count += 1

                # If enough burns in this window, add as valid episode
                if has_burns_count >= min_episode_length:
                    max_length = min(env.T - start_t - 1, 20)  # Cap at 20 steps
                    filtered_episodes.append((env_path, start_t, max_length))
                    total_episodes_found += 1
                    env_has_episodes = True

            if env_has_episodes:
                total_envs_kept += 1

            # CRITICAL: Clean up environment data to prevent memory leak
            del env

            # Aggressive garbage collection during scanning (every 10 envs)
            if idx % 10 == 0:
                gc.collect()

        except Exception as e:
            print(f"  Error loading {env_path.name}: {e}")
            continue

    # Final garbage collection after scanning
    gc.collect()

    print(f"\nFiltering results:")
    print(f"  Environments scanned: {total_envs_scanned}")
    print(f"  Environments with good episodes: {total_envs_kept}")
    print(f"  Total episodes found: {total_episodes_found}")
    print(f"  Avg episodes per env: {total_episodes_found / max(1, total_envs_kept):.1f}")

    return filtered_episodes


def main():
    parser = argparse.ArgumentParser(description='A3C V7 Training - Temporal LSTM')
    parser.add_argument('--repo-root', type=str, default='/home/chaseungjoon/code/WildfirePrediction')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of parallel CPU workers (4 is optimal)')
    parser.add_argument('--max-episodes', type=int, default=1000, help='Total episodes across all workers')
    parser.add_argument('--lr', type=float, default=7e-5, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--value-loss-coef', type=float, default=0.5, help='Value loss coefficient')
    parser.add_argument('--entropy-coef', type=float, default=0.015, help='Entropy coefficient')
    parser.add_argument('--max-grad-norm', type=float, default=0.5, help='Max gradient norm')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max-envs', type=int, default=None, help='Limit number of training environments')
    parser.add_argument('--max-file-size-mb', type=int, default=50, help='Max environment file size in MB')
    parser.add_argument('--min-episode-length', type=int, default=4, help='Min timesteps with burns per episode')
    parser.add_argument('--window-size', type=int, default=3, help='Temporal window size (number of timesteps)')
    parser.add_argument('--lstm-hidden-dim', type=int, default=128, help='LSTM hidden dimension')
    parser.add_argument('--lstm-num-layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--log-interval', type=int, default=10, help='Log every N episodes')
    parser.add_argument('--max-envs-to-scan', type=int, default=None, help='Limit environments to scan (for memory safety)')
    parser.add_argument('--wandb-project', type=str, default='wildfire-prediction', help='WandB project name')
    parser.add_argument('--wandb-run-name', type=str, default=None, help='WandB run name')
    parser.add_argument('--no-wandb', action='store_true', help='Disable WandB logging')
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

    print(f"=" * 80)
    print(f"A3C V7 Training - Temporal LSTM")
    print(f"=" * 80)
    print(f"Problem: Per-cell 8-neighbor prediction with temporal context")
    print(f"Temporal window: {args.window_size} timesteps")
    print(f"LSTM: {args.lstm_num_layers} layers, hidden_dim={args.lstm_hidden_dim}")
    print(f"Rewards: DENSE (IoU at every timestep)")
    print(f"=" * 80)
    print(f"Training environments: {len(train_paths)}")
    print(f"Number of workers: {args.num_workers}")
    print(f"Max episodes: {args.max_episodes}")
    print(f"Learning rate: {args.lr}")
    print(f"=" * 80)

    # FILTER EPISODES
    filtered_episodes = create_filtered_episode_list(
        train_paths,
        max_file_size_mb=args.max_file_size_mb,
        min_episode_length=args.min_episode_length,
        window_size=args.window_size,
        max_envs_to_scan=args.max_envs_to_scan
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
        wandb_run_name = args.wandb_run_name or f"a3c-v7-lstm-w{args.num_workers}-lr{args.lr}"
        wandb.init(
            project=args.wandb_project,
            name=wandb_run_name,
            config={
                "model": "A3C-V7-Temporal-LSTM",
                "formulation": "per_cell_8neighbor_temporal",
                "rewards": "dense_per_timestep",
                "window_size": args.window_size,
                "lstm_hidden_dim": args.lstm_hidden_dim,
                "lstm_num_layers": args.lstm_num_layers,
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
                "log_interval": args.log_interval,
            }
        )
        print(f"WandB initialized: {wandb.run.name}")
    else:
        print("WandB logging disabled")

    # Create shared model on CPU
    shared_model = A3C_TemporalModel(
        in_channels=14,
        window_size=args.window_size,
        lstm_hidden_dim=args.lstm_hidden_dim,
        lstm_num_layers=args.lstm_num_layers
    )
    shared_model.share_memory()
    shared_model.train()

    # Count parameters
    total_params = sum(p.numel() for p in shared_model.parameters())
    trainable_params = sum(p.numel() for p in shared_model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")

    # Create timestamped checkpoint directory
    from datetime import datetime
    timestamp = datetime.now().strftime("%y%m%d-%H%M")
    ckpt_dir = repo_root / 'rl_training' / 'a3c' / 'checkpoints_v7' / timestamp
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
        'window_size': args.window_size,
        'lstm_hidden_dim': args.lstm_hidden_dim,
        'lstm_num_layers': args.lstm_num_layers,
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

    # Save final model (use existing timestamped ckpt_dir)
    final_path = ckpt_dir / 'final_model.pt'
    torch.save({
        'model_state_dict': shared_model.state_dict(),
        'best_iou': global_best_iou.value,
        'total_episodes': global_episode_counter.value,
        'config': vars(args),
        'filtered_episodes_count': len(filtered_episodes),
        'formulation': 'per_cell_8_neighbor_temporal_lstm'
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
            name='final-model-v7-temporal',
            type='model',
            description=f'A3C V7 (temporal LSTM) after {global_episode_counter.value} episodes, IoU {global_best_iou.value:.4f}'
        )
        artifact.add_file(str(final_path))
        wandb.log_artifact(artifact)

        wandb.summary['total_episodes'] = global_episode_counter.value
        wandb.summary['best_iou'] = global_best_iou.value

        wandb.finish()


if __name__ == '__main__':
    main()
