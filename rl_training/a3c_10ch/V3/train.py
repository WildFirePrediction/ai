"""
A3C Training V3 - 10 Channels (Wind-Focused)
CORRECT Formulation

Per-cell 8-neighbor prediction with dense rewards at every timestep.
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

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from a3c_10ch.V3.model import A3C_PerCellModel_Deep
from a3c_10ch.V3.worker import worker_process_correct


def load_episode_data(min_timesteps=4):
    """Load fire episode data from embedded_data/fire_episodes_10ch_wind"""
    import numpy as np

    episode_dir = Path("embedded_data/fire_episodes_10ch_wind")
    episode_files = sorted(episode_dir.glob("episode_*.npz"))

    episodes = []
    for ep_file in episode_files:
        data = np.load(ep_file)
        n_timesteps = len(data['states'])
        if n_timesteps >= min_timesteps:
            episodes.append(str(ep_file))

    return episodes


def create_filtered_episode_list(env_paths, max_file_size_mb=10, min_episode_length=4):
    """
    Load fire episode NPZ files with minimum timesteps.

    Args:
        env_paths: Ignored (for compatibility)
        max_file_size_mb: Ignored (for compatibility)
        min_episode_length: Minimum timesteps required

    Returns:
        List of episode file paths
    """
    episodes = load_episode_data(min_timesteps=min_episode_length)
    print(f"Loaded {len(episodes)} episodes with >= {min_episode_length} timesteps")
    return episodes


def main():
    parser = argparse.ArgumentParser(description='A3C V3 Training - 10 Channels')
    parser.add_argument('--repo-root', type=str, default='/home/chaseungjoon/code/WildfirePrediction')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of parallel CPU workers')
    parser.add_argument('--max-episodes', type=int, default=1000, help='Total episodes across all workers')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate (increased for stronger learning)')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--value-loss-coef', type=float, default=0.5, help='Value loss coefficient')
    parser.add_argument('--entropy-coef', type=float, default=0.05, help='Entropy coefficient (increased to encourage exploration)')
    parser.add_argument('--max-grad-norm', type=float, default=2.0, help='Max gradient norm (increased for stronger updates)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max-envs', type=int, default=None, help='Limit number of training environments')
    parser.add_argument('--max-file-size-mb', type=int, default=50, help='Max environment file size in MB')
    parser.add_argument('--min-episode-length', type=int, default=4, help='Min timesteps with burns per episode')
    parser.add_argument('--log-interval', type=int, default=10, help='Log every N episodes')
    parser.add_argument('--wandb-project', type=str, default='wildfire-prediction', help='WandB project name')
    parser.add_argument('--wandb-run-name', type=str, default=None, help='WandB run name')
    parser.add_argument('--no-wandb', action='store_true', help='Disable WandB logging')
    args = parser.parse_args()

    mp.set_start_method('spawn', force=True)

    repo_root = Path(args.repo_root)

    print(f"=" * 80)
    print(f"A3C V3 Training - 10 Channels (Wind-Focused)")
    print(f"=" * 80)
    print(f"Problem: Per-cell 8-neighbor prediction")
    print(f"Rewards: DENSE (IoU at every timestep)")
    print(f"Input: 10 channels (2 DEM + 3 Wind + 1 NDVI + 4 FSM)")
    print(f"=" * 80)
    print(f"Number of workers: {args.num_workers}")
    print(f"Max episodes: {args.max_episodes}")
    print(f"Learning rate: {args.lr}")
    print(f"=" * 80)

    filtered_episodes = create_filtered_episode_list(
        None,
        max_file_size_mb=args.max_file_size_mb,
        min_episode_length=args.min_episode_length
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

    use_wandb = not args.no_wandb
    if use_wandb:
        wandb_run_name = args.wandb_run_name or f"a3c-v3-10ch-w{args.num_workers}-lr{args.lr}"
        wandb.init(
            project=args.wandb_project,
            name=wandb_run_name,
            config={
                "model": "A3C-V3-10ch-PerCell-8Neighbor",
                "formulation": "correct_per_cell",
                "rewards": "dense_per_timestep",
                "input_channels": 10,
                "channels_info": "2 DEM + 3 Wind + 1 NDVI + 4 FSM",
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

    shared_model = A3C_PerCellModel_Deep(in_channels=10)
    shared_model.share_memory()
    shared_model.train()

    total_params = sum(p.numel() for p in shared_model.parameters())
    print(f"Model parameters: {total_params:,}")

    from datetime import datetime
    timestamp = datetime.now().strftime("%y%m%d-%H%M")
    ckpt_dir = repo_root / 'rl_training' / 'a3c_10ch' / 'V3' / 'checkpoints' / timestamp
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    print(f"Checkpoint directory: {ckpt_dir}")

    optimizer = torch.optim.Adam(shared_model.parameters(), lr=args.lr)

    global_episode_counter = mp.Value('i', 0)
    global_best_iou = mp.Value('d', 0.0)
    lock = mp.Lock()

    metrics_queue = mp.Queue() if use_wandb else None

    config = {
        'seed': args.seed,
        'gamma': args.gamma,
        'value_loss_coef': args.value_loss_coef,
        'entropy_coef': args.entropy_coef,
        'max_grad_norm': args.max_grad_norm,
        'max_episodes': args.max_episodes,
        'log_interval': args.log_interval,
        'checkpoint_dir': str(ckpt_dir),
    }

    processes = []
    for worker_id in range(args.num_workers):
        p = mp.Process(
            target=worker_process_correct,
            args=(worker_id, shared_model, optimizer, filtered_episodes, config,
                  global_episode_counter, global_best_iou, lock, metrics_queue)
        )
        p.start()
        processes.append(p)

    if use_wandb:
        import time
        while any(p.is_alive() for p in processes):
            while not metrics_queue.empty():
                metrics = metrics_queue.get()
                wandb.log(metrics)
            time.sleep(0.1)

        while not metrics_queue.empty():
            metrics = metrics_queue.get()
            wandb.log(metrics)
    else:
        for p in processes:
            p.join()

    final_path = ckpt_dir / 'final_model.pt'
    torch.save({
        'model_state_dict': shared_model.state_dict(),
        'best_iou': global_best_iou.value,
        'total_episodes': global_episode_counter.value,
        'config': vars(args),
        'filtered_episodes_count': len(filtered_episodes),
        'formulation': 'per_cell_8_neighbor_dense_rewards',
        'input_channels': 10
    }, final_path)

    print(f"\n{'=' * 80}")
    print(f"Training Complete!")
    print(f"Total episodes: {global_episode_counter.value}")
    print(f"Best IoU: {global_best_iou.value:.4f}")
    print(f"Model saved to: {final_path}")
    print(f"{'=' * 80}")

    if use_wandb:
        artifact = wandb.Artifact(
            name='final-model-v3-10ch',
            type='model',
            description=f'A3C V3 10ch after {global_episode_counter.value} episodes, IoU {global_best_iou.value:.4f}'
        )
        artifact.add_file(str(final_path))
        wandb.log_artifact(artifact)

        wandb.summary['total_episodes'] = global_episode_counter.value
        wandb.summary['best_iou'] = global_best_iou.value

        wandb.finish()


if __name__ == '__main__':
    main()
