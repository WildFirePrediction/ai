"""
Validation Script for A3C V6 LSTM RECALL - RECALL-FIRST SUPERAGGRO - 16 Channels with Relaxed IoU
Evaluates trained model on validation set with 8-neighbor tolerance
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import torch
import numpy as np
from pathlib import Path
import argparse
import sys
from scipy.ndimage import binary_dilation

torch.set_num_threads(1)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from a3c_16ch.V6_LSTM_RECALL.model import A3C_PerCellModel_LSTM


def compute_relaxed_iou(pred, target):
    """
    Compute IoU with 8-neighbor tolerance (dilated ground truth).

    Args:
        pred: (H, W) predicted mask
        target: (H, W) ground truth mask

    Returns:
        IoU with dilated target (relaxed matching)
    """
    # 3x3 dilation kernel for 8-neighbor tolerance
    structure = np.ones((3, 3), dtype=bool)

    # Dilate target
    target_dilated = binary_dilation(target > 0.5, structure=structure)
    pred_binary = pred > 0.5

    intersection = (pred_binary & target_dilated).sum()
    union = (pred_binary | target_dilated).sum()

    return float(intersection / (union + 1e-8))


def validate_episode(model, episode_file, sequence_length=3):
    """
    Validate model on a single episode with temporal sequences.

    Args:
        model: Trained LSTM model
        episode_file: Path to episode file
        sequence_length: LSTM sequence length

    Returns:
        mean_relaxed_iou: Average relaxed IoU across all timesteps
        episode_name: Name of the episode
    """
    try:
        data = np.load(episode_file)
        states_np = data['states']  # (T, 16, 30, 30)
        fire_masks = data['fire_masks']  # (T, 30, 30)

        T = len(states_np)
        episode_name = episode_file.stem

        if T < sequence_length + 1:
            return None, episode_name

        ious = []

        # Start from sequence_length to have enough history
        for t in range(sequence_length, T - 1):
            seq_start = t - sequence_length + 1
            sequence_t = states_np[seq_start:t+1]  # (seq_len, 16, 30, 30)

            current_fire = fire_masks[t]
            next_fire = fire_masks[t+1]

            # Prepare tensors
            sequence_tensor = torch.from_numpy(sequence_t).unsqueeze(0).float()  # (1, seq_len, 16, 30, 30)
            current_fire_tensor = torch.from_numpy(current_fire).unsqueeze(0).float()

            with torch.no_grad():
                action_grid, _, _, _, _ = model.get_action_and_value(
                    sequence_tensor, current_fire_tensor
                )

            prediction = action_grid.numpy()

            # Compute new burns
            actual_mask_t = current_fire > 0
            actual_mask_t1 = next_fire > 0
            new_burns = (actual_mask_t1 & ~actual_mask_t).astype(np.float32)

            # Compute RELAXED IoU
            iou = compute_relaxed_iou(prediction, new_burns)
            ious.append(iou)

        if len(ious) == 0:
            return None, episode_name

        return np.mean(ious), episode_name

    except Exception as e:
        print(f"Error validating {episode_file}: {e}")
        return None, episode_file.stem


def main():
    parser = argparse.ArgumentParser(description='A3C V6 LSTM RECALL - RECALL-FIRST SUPERAGGRO Validation - Relaxed IoU')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str,
                       default='/home/chaseungjoon/code/WildfirePrediction/embedded_data/fire_episodes_16ch_normalized',
                       help='Directory with embedded episodes')
    parser.add_argument('--split', type=str, default='val',
                       choices=['train', 'val', 'all'],
                       help='Which split to validate on')
    parser.add_argument('--sequence-length', type=int, default=3,
                       help='LSTM sequence length')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for validation results')

    args = parser.parse_args()

    print(f"A3C V6 LSTM RECALL - RECALL-FIRST SUPERAGGRO Validation - Relaxed IoU (8-neighbor tolerance)")
    print(f"=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data directory: {args.data_dir}")
    print(f"Split: {args.split}")
    print(f"Sequence length: {args.sequence_length}")
    print(f"Relaxed IoU: 3x3 dilation (8-neighbor tolerance)")
    print(f"=" * 70)

    # Load model
    print("\nLoading model...")
    model = A3C_PerCellModel_LSTM(in_channels=16, sequence_length=args.sequence_length)

    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded checkpoint from episode {checkpoint.get('episode', 'unknown')}")
    print(f"Checkpoint best relaxed IoU: {checkpoint.get('best_iou', 'unknown'):.4f}")

    # Load episodes
    print(f"\nLoading episodes from {args.data_dir}...")
    episode_dir = Path(args.data_dir)
    all_episodes = sorted(episode_dir.glob('episode_*.npz'))

    # Split into train/val (80/20 split)
    n_total = len(all_episodes)
    n_train = int(0.8 * n_total)

    if args.split == 'train':
        episodes = all_episodes[:n_train]
    elif args.split == 'val':
        episodes = all_episodes[n_train:]
    else:
        episodes = all_episodes

    print(f"Total episodes: {len(all_episodes)}")
    print(f"Train episodes: {n_train}")
    print(f"Val episodes: {n_total - n_train}")
    print(f"Validating on: {len(episodes)} episodes ({args.split} split)")

    # Validate
    print("\nValidating...")
    ious = []
    episode_names = []

    for i, episode_file in enumerate(episodes):
        mean_iou, name = validate_episode(model, episode_file, args.sequence_length)

        if mean_iou is not None:
            ious.append(mean_iou)
            episode_names.append(name)

            if (i + 1) % 100 == 0 or (i + 1) == len(episodes):
                print(f"Progress: {i+1}/{len(episodes)} | Current mean relaxed IoU: {np.mean(ious):.4f}")

    # Results
    ious = np.array(ious)
    mean_iou = np.mean(ious)
    median_iou = np.median(ious)
    std_iou = np.std(ious)
    min_iou = np.min(ious)
    max_iou = np.max(ious)

    print(f"\n" + "=" * 70)
    print(f"VALIDATION RESULTS (16ch LSTM REL - RELAXED IoU)")
    print(f"=" * 70)
    print(f"Episodes validated: {len(ious)}")
    print(f"Mean Relaxed IoU: {mean_iou:.4f}")
    print(f"Median Relaxed IoU: {median_iou:.4f}")
    print(f"Std IoU: {std_iou:.4f}")
    print(f"Min IoU: {min_iou:.4f}")
    print(f"Max IoU: {max_iou:.4f}")
    print(f"=" * 70)

    # Distribution
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, _ = np.histogram(ious, bins=bins)
    print(f"\nRelaxed IoU Distribution:")
    for i in range(len(bins) - 1):
        percentage = (hist[i] / len(ious)) * 100
        print(f"  {bins[i]:.1f}-{bins[i+1]:.1f}: {hist[i]:4d} ({percentage:5.1f}%)")

    # Top and bottom episodes
    sorted_indices = np.argsort(ious)
    print(f"\nTop 10 Episodes:")
    for idx in sorted_indices[-10:][::-1]:
        print(f"  {episode_names[idx]}: {ious[idx]:.4f}")

    print(f"\nBottom 10 Episodes:")
    for idx in sorted_indices[:10]:
        print(f"  {episode_names[idx]}: {ious[idx]:.4f}")

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        checkpoint_dir = Path(args.checkpoint).parent
        output_path = checkpoint_dir / f'validation_results_relaxed_{args.split}.npz'

    np.savez(
        output_path,
        mean_ious=ious,
        episode_names=np.array(episode_names),
        checkpoint_path=args.checkpoint,
        split=args.split
    )
    print(f"\nResults saved to: {output_path}")
    print(f"=" * 70)


if __name__ == '__main__':
    main()
