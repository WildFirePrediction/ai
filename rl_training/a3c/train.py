"""
A3C Training with Parallel CPU Workers

Asynchronous Advantage Actor-Critic for spatial fire spread prediction.
Multiple workers run in parallel on CPU, sharing gradients through a global model.
"""
import os
# CRITICAL: Set threading limits BEFORE importing torch/numpy
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

# Also set PyTorch threads
torch.set_num_threads(1)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from a3c.model import A3C_SpatialFireModel
from a3c.worker import worker_process


def main():
    parser = argparse.ArgumentParser(description='A3C Training for Wildfire Prediction')
    parser.add_argument('--repo-root', type=str, default='/home/chaseungjoon/code/WildfirePrediction-SSD')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of parallel CPU workers')
    parser.add_argument('--max-episodes', type=int, default=1000, help='Total episodes across all workers')
    parser.add_argument('--max-steps', type=int, default=50, help='Max steps per episode')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--value-loss-coef', type=float, default=0.5, help='Value loss coefficient')
    parser.add_argument('--entropy-coef', type=float, default=0.01, help='Entropy coefficient')
    parser.add_argument('--max-grad-norm', type=float, default=0.5, help='Max gradient norm')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max-envs', type=int, default=None, help='Limit number of training environments')
    parser.add_argument('--log-interval', type=int, default=10, help='Log every N episodes')
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
    print(f"A3C Training with Parallel CPU Workers")
    print(f"=" * 80)
    print(f"Training environments: {len(train_paths)}")
    print(f"Number of workers: {args.num_workers}")
    print(f"Max episodes: {args.max_episodes}")
    print(f"Learning rate: {args.lr}")
    print(f"=" * 80)

    # Create shared model on CPU
    shared_model = A3C_SpatialFireModel(in_channels=14)
    shared_model.share_memory()  # Enable sharing across processes
    shared_model.train()

    # Create shared optimizer
    optimizer = torch.optim.Adam(shared_model.parameters(), lr=args.lr)

    # Shared counters and values
    global_episode_counter = mp.Value('i', 0)  # Episode counter
    global_best_iou = mp.Value('d', 0.0)  # Best IoU
    lock = mp.Lock()  # Lock for synchronization

    # Training configuration
    config = {
        'seed': args.seed,
        'gamma': args.gamma,
        'value_loss_coef': args.value_loss_coef,
        'entropy_coef': args.entropy_coef,
        'max_grad_norm': args.max_grad_norm,
        'max_episodes': args.max_episodes,
        'max_steps_per_episode': args.max_steps,
        'log_interval': args.log_interval,
    }

    # Create workers
    processes = []
    for worker_id in range(args.num_workers):
        p = mp.Process(
            target=worker_process,
            args=(worker_id, shared_model, optimizer, train_paths, config,
                  global_episode_counter, global_best_iou, lock)
        )
        p.start()
        processes.append(p)

    # Wait for all workers to finish
    for p in processes:
        p.join()

    # Save final model
    ckpt_dir = repo_root / 'rl_training' / 'a3c' / 'checkpoints'
    ckpt_dir.mkdir(exist_ok=True, parents=True)

    final_path = ckpt_dir / 'final_model.pt'
    torch.save({
        'model_state_dict': shared_model.state_dict(),
        'best_iou': global_best_iou.value,
        'total_episodes': global_episode_counter.value,
        'config': vars(args)
    }, final_path)

    print(f"\n" + "=" * 80)
    print(f"Training Complete!")
    print(f"Total episodes: {global_episode_counter.value}")
    print(f"Best IoU: {global_best_iou.value:.4f}")
    print(f"Model saved to: {final_path}")
    print(f"=" * 80)


if __name__ == '__main__':
    main()
