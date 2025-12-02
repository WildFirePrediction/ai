"""
Validation script for A3C V3 10-channel model
Evaluates model on validation episodes and computes IoU statistics
"""
import torch
import numpy as np
from pathlib import Path
import argparse
import sys
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from a3c_10ch.V3.model import A3C_PerCellModel_Deep


def compute_iou(pred, target):
    """Compute IoU between predicted and target masks"""
    pred_flat = pred.flatten()
    target_flat = target.flatten()

    intersection = (pred_flat * target_flat).sum()
    union = np.clip(pred_flat + target_flat, 0, 1).sum()

    return float(intersection / (union + 1e-8))


def validate_episode(model, episode_file, device='cpu'):
    """
    Validate model on a single episode

    Returns:
        dict with episode statistics
    """
    try:
        data = np.load(episode_file)
        states_np = data['states']  # (T, 10, 30, 30)
        fire_masks_np = data['fire_masks']  # (T, 30, 30)
        T = len(states_np)

        if T < 2:
            return None

        ious = []

        # Predict each timestep
        for t in range(T - 1):
            features_t = states_np[t]  # (10, 30, 30)
            current_fire = fire_masks_np[t]  # (30, 30)
            next_fire = fire_masks_np[t+1]  # (30, 30)

            # Convert to tensors
            state_tensor = torch.from_numpy(features_t).unsqueeze(0).float().to(device)
            current_fire_tensor = torch.from_numpy(current_fire).unsqueeze(0).float().to(device)

            # Get prediction
            with torch.no_grad():
                action_grid, _, _, _, _ = model.get_action_and_value(
                    state_tensor, current_fire_tensor
                )

            pred_mask = action_grid.cpu().numpy()

            # Compute IoU for new burns only
            actual_mask_t = current_fire > 0
            actual_mask_t1 = next_fire > 0
            new_burns = (actual_mask_t1 & ~actual_mask_t)
            predicted_mask = pred_mask > 0.5

            intersection = (predicted_mask & new_burns).sum()
            union = (predicted_mask | new_burns).sum()
            step_iou = intersection / (union + 1e-8) if union > 0 else 0.0

            ious.append(step_iou)

        return {
            'episode_name': episode_file.name,
            'timesteps': T,
            'mean_iou': np.mean(ious),
            'max_iou': np.max(ious),
            'min_iou': np.min(ious),
            'std_iou': np.std(ious),
            'ious': ious
        }

    except Exception as e:
        print(f"Error validating {episode_file.name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Validate A3C V3 10-channel model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file (best_model.pt)')
    parser.add_argument('--data-dir', type=str,
                       default='embedded_data/fire_episodes_10ch_wind',
                       help='Directory with episode data')
    parser.add_argument('--split', type=str, default='val',
                       choices=['train', 'val', 'all'],
                       help='Which split to validate on')
    parser.add_argument('--max-episodes', type=int, default=None,
                       help='Maximum episodes to validate (for quick test)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run on')
    args = parser.parse_args()

    print("="*80)
    print("A3C V3 10-CHANNEL MODEL VALIDATION")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data directory: {args.data_dir}")
    print(f"Device: {args.device}")
    print("="*80)

    # Load model
    print("\nLoading model...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)

    model = A3C_PerCellModel_Deep(in_channels=10)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)
    model.eval()

    print(f"Model loaded successfully")
    print(f"  Training episodes: {checkpoint.get('episode', 'unknown')}")
    print(f"  Best training IoU: {checkpoint.get('best_iou', 'unknown'):.4f}")

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

    if args.max_episodes:
        episodes = episodes[:args.max_episodes]

    print(f"Total episodes: {len(all_episodes)}")
    print(f"Train episodes: {n_train}")
    print(f"Val episodes: {n_total - n_train}")
    print(f"Validating on: {len(episodes)} episodes ({args.split} split)")

    # Validate
    print(f"\n{'='*80}")
    print("Running validation...")
    print("="*80)

    results = []
    for idx, ep_file in enumerate(episodes):
        result = validate_episode(model, ep_file, device=args.device)
        if result:
            results.append(result)

        if (idx + 1) % 100 == 0 or (idx + 1) == len(episodes):
            print(f"  Progress: {idx+1}/{len(episodes)}", flush=True)

    # Aggregate statistics
    print(f"\n{'='*80}")
    print("VALIDATION RESULTS")
    print("="*80)

    if not results:
        print("No valid results!")
        return

    all_ious = [r['mean_iou'] for r in results]
    all_ious_arr = np.array(all_ious)

    print(f"\nOverall Statistics:")
    print(f"  Episodes validated: {len(results)}")
    print(f"  Mean IoU: {all_ious_arr.mean():.4f}")
    print(f"  Median IoU: {np.median(all_ious_arr):.4f}")
    print(f"  Std IoU: {all_ious_arr.std():.4f}")
    print(f"  Min IoU: {all_ious_arr.min():.4f}")
    print(f"  Max IoU: {all_ious_arr.max():.4f}")

    # IoU distribution
    print(f"\nIoU Distribution:")
    bins = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 1.0)]
    for low, high in bins:
        count = ((all_ious_arr >= low) & (all_ious_arr < high)).sum()
        pct = 100 * count / len(all_ious_arr)
        print(f"  {low:.1f}-{high:.1f}: {count:4d} episodes ({pct:5.1f}%)")

    # Top and bottom episodes
    print(f"\nTop 5 Episodes:")
    top_results = sorted(results, key=lambda x: x['mean_iou'], reverse=True)[:5]
    for i, r in enumerate(top_results, 1):
        print(f"  {i}. {r['episode_name']}: {r['mean_iou']:.4f} (T={r['timesteps']})")

    print(f"\nBottom 5 Episodes:")
    bottom_results = sorted(results, key=lambda x: x['mean_iou'])[:5]
    for i, r in enumerate(bottom_results, 1):
        print(f"  {i}. {r['episode_name']}: {r['mean_iou']:.4f} (T={r['timesteps']})")

    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)

    # Save results
    output_file = Path(args.checkpoint).parent / f'validation_results_{args.split}.npz'
    np.savez_compressed(
        output_file,
        mean_ious=[r['mean_iou'] for r in results],
        episode_names=[r['episode_name'] for r in results],
        checkpoint_path=args.checkpoint,
        split=args.split
    )
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
