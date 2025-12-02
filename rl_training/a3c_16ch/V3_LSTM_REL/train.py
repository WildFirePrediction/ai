"""
A3C Training Script V3 LSTM REL - 16 Channels with Relaxed IoU (8-neighbor tolerance)
Same architecture as V3_LSTM but with dilated ground truth for reward computation
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import torch
import torch.multiprocessing as mp
import numpy as np
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from a3c_16ch.V3_LSTM_REL.model import A3C_PerCellModel_LSTM
from a3c_16ch.V3_LSTM_REL.worker import worker_process_lstm


def get_filtered_episodes(data_dir, mel_threshold=4):
    """
    Get filtered episodes with MEL >= threshold.
    MEL = Maximum Extent of Loss = number of timesteps - 1

    Args:
        data_dir: Directory containing embedded episodes
        mel_threshold: Minimum MEL value

    Returns:
        List of episode file paths
    """
    data_dir = Path(data_dir)
    all_episodes = sorted(data_dir.glob('episode_*.npz'))

    filtered = []
    for ep_file in all_episodes:
        try:
            data = np.load(ep_file)
            # MEL is computed as number of timesteps - 1
            num_timesteps = len(data['states'])
            mel = num_timesteps - 1

            if mel >= mel_threshold:
                filtered.append(ep_file)
        except Exception as e:
            print(f"Error loading {ep_file}: {e}")
            continue

    print(f"Found {len(filtered)} episodes with MEL >= {mel_threshold} (out of {len(all_episodes)} total)")
    return filtered


def main():
    parser = argparse.ArgumentParser(description='A3C V3 LSTM REL Training - Relaxed IoU (8-neighbor)')
    parser.add_argument('--data-dir', type=str,
                       default='/home/chaseungjoon/code/WildfirePrediction/embedded_data/fire_episodes_16ch_normalized',
                       help='Directory with embedded episodes')
    parser.add_argument('--checkpoint-dir', type=str,
                       default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--mel-threshold', type=int, default=4,
                       help='Minimum MEL threshold')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of A3C workers')
    parser.add_argument('--max-episodes', type=int, default=10000,
                       help='Maximum training episodes')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--sequence-length', type=int, default=3,
                       help='LSTM sequence length')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"A3C V3 LSTM REL Training - Relaxed IoU (8-neighbor tolerance)")
    print(f"=" * 70)
    print(f"Data directory: {args.data_dir}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"MEL threshold: {args.mel_threshold}")
    print(f"Number of workers: {args.num_workers}")
    print(f"Max episodes: {args.max_episodes}")
    print(f"Learning rate: {args.lr}")
    print(f"Gamma: {args.gamma}")
    print(f"Sequence length: {args.sequence_length}")
    print(f"Seed: {args.seed}")
    print(f"=" * 70)
    print(f"Features:")
    print(f"  - LSTM temporal context (sequence_length={args.sequence_length})")
    print(f"  - Data augmentation (rotation, flip)")
    print(f"  - Strict penalties for non-movement")
    print(f"  - RELAXED IoU: 8-neighbor tolerance (3x3 dilation)")
    print(f"=" * 70)

    # Get filtered episodes
    filtered_episodes = get_filtered_episodes(args.data_dir, args.mel_threshold)

    if len(filtered_episodes) == 0:
        print("ERROR: No episodes found!")
        return

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create shared model
    shared_model = A3C_PerCellModel_LSTM(in_channels=16, sequence_length=args.sequence_length)
    shared_model.share_memory()
    shared_model.train()

    # Count parameters
    total_params = sum(p.numel() for p in shared_model.parameters())
    trainable_params = sum(p.numel() for p in shared_model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    # Create optimizer
    optimizer = torch.optim.Adam(shared_model.parameters(), lr=args.lr)

    # Training configuration
    config = {
        'seed': args.seed,
        'gamma': args.gamma,
        'max_episodes': args.max_episodes,
        'checkpoint_dir': str(checkpoint_dir),
        'value_loss_coef': 0.5,
        'entropy_coef': 0.01,
        'max_grad_norm': 0.5,
        'log_interval': 10,
        'sequence_length': args.sequence_length,
    }

    # Shared counters
    global_episode_counter = mp.Value('i', 0)
    global_best_iou = mp.Value('d', 0.0)
    lock = mp.Lock()

    print(f"\nStarting training with {args.num_workers} workers...")
    print(f"Training on {len(filtered_episodes)} episodes (MEL >= {args.mel_threshold})")
    print(f"Temporal context: LSTM with sequence length {args.sequence_length}")
    print(f"Data augmentation: Enabled")
    print(f"Strict penalties: Enabled")
    print(f"Relaxed IoU: 3x3 dilation (8-neighbor tolerance)")
    print(f"=" * 70)

    # Start workers
    processes = []
    for worker_id in range(args.num_workers):
        p = mp.Process(
            target=worker_process_lstm,
            args=(worker_id, shared_model, optimizer, filtered_episodes, config,
                  global_episode_counter, global_best_iou, lock, None)
        )
        p.start()
        processes.append(p)
        print(f"Started worker {worker_id}")

    # Wait for all workers to finish
    for p in processes:
        p.join()

    print(f"\n" + "=" * 70)
    print(f"Training completed!")
    print(f"Final episodes: {global_episode_counter.value}")
    print(f"Best Relaxed IoU: {global_best_iou.value:.4f}")

    # Save final model
    final_model_path = checkpoint_dir / 'final_model.pt'
    torch.save({
        'episode': global_episode_counter.value,
        'model_state_dict': shared_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_iou': global_best_iou.value,
    }, final_model_path)
    print(f"Final model saved to: {final_model_path}")
    print(f"Best model saved to: {checkpoint_dir / 'best_model.pt'}")
    print(f"=" * 70)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
