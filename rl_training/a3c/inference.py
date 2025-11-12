"""
A3C Inference Script - Run Trained Model on Test Data

Usage:
    python inference.py --checkpoint path/to/best_model.pt --env-path path/to/env.pkl
"""
import argparse
import torch
import numpy as np
import pickle
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from model_v2 import A3C_PerCellModel
from model_v3_medium import A3C_PerCellModel_Medium
from model_v5 import A3C_PerCellModel_4Neighbor
from wildfire_env_spatial import WildfireEnvSpatial


def load_model(checkpoint_path, model_type='v2'):
    """
    Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file
        model_type: 'v2', 'medium', or 'v5'

    Returns:
        Loaded model in eval mode
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Create model architecture
    if model_type == 'v2':
        model = A3C_PerCellModel(in_channels=14)
    elif model_type == 'medium':
        model = A3C_PerCellModel_Medium(in_channels=14, use_groupnorm=True)
    elif model_type == 'v5':
        model = A3C_PerCellModel_4Neighbor(in_channels=14, hidden_dim=128, use_groupnorm=True)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode

    print(f"Loaded model from episode {checkpoint.get('episode', 'unknown')}")
    if 'best_iou' in checkpoint:
        print(f"Best IoU: {checkpoint['best_iou']:.4f}")
    if 'best_f1' in checkpoint:
        print(f"Best F1: {checkpoint['best_f1']:.4f}")

    return model


def compute_metrics(pred_mask, actual_new_burns):
    """Compute comprehensive metrics."""
    pred_flat = (pred_mask > 0.5).astype(np.float32).flatten()
    target_flat = actual_new_burns.astype(np.float32).flatten()

    TP = (pred_flat * target_flat).sum()
    FP = (pred_flat * (1 - target_flat)).sum()
    FN = ((1 - pred_flat) * target_flat).sum()
    TN = ((1 - pred_flat) * (1 - target_flat)).sum()

    metrics = {
        'iou': float(TP / (TP + FP + FN + 1e-8)),
        'precision': float(TP / (TP + FP + 1e-8)),
        'recall': float(TP / (TP + FN + 1e-8)),
        'f1': float(2 * TP / (2 * TP + FP + FN + 1e-8)),
        'accuracy': float((TP + TN) / (TP + FP + FN + TN + 1e-8)),
        'tp': int(TP),
        'fp': int(FP),
        'fn': int(FN),
        'tn': int(TN),
    }

    return metrics


def run_inference(model, env_path, start_t=0, max_steps=20, verbose=True):
    """
    Run inference on a single environment.

    Args:
        model: Trained model
        env_path: Path to environment .pkl file
        start_t: Starting timestep
        max_steps: Maximum steps to simulate
        verbose: Print step-by-step results

    Returns:
        Dictionary with predictions and metrics
    """
    # Load environment
    env = WildfireEnvSpatial(env_path)

    # Fast-forward to start_t
    obs, info = env.reset()
    for _ in range(start_t):
        obs, _, _, _ = env.step(np.zeros((env.H, env.W)))

    print(f"\nEnvironment: {Path(env_path).name}")
    print(f"Grid size: {env.H}x{env.W}")
    print(f"Starting at timestep: {start_t}")
    print(f"Max steps: {max_steps}")
    print("=" * 80)

    # Run inference
    results = {
        'predictions': [],
        'actuals': [],
        'metrics_per_step': [],
    }

    done = False
    step = 0

    while not done and step < max_steps:
        # Convert obs to tensor
        state_tensor = torch.from_numpy(obs).unsqueeze(0).float()
        fire_mask = state_tensor[0, 5]  # Fire mask channel

        # Get prediction from model
        with torch.no_grad():
            action_grid, log_prob, entropy, value, info = model.get_action_and_value(
                state_tensor, fire_mask
            )

        # Convert to numpy
        prediction = action_grid.numpy()

        # Step environment
        next_obs, reward, done, _ = env.step(prediction)

        # Get actual new burns
        actual_mask_t = env.fire_masks[env.t - 1] > 0
        actual_mask_t1 = env.fire_masks[env.t] > 0
        new_burns = (actual_mask_t1 & ~actual_mask_t)

        # Compute metrics
        metrics = compute_metrics(prediction, new_burns)

        # Store results
        results['predictions'].append(prediction)
        results['actuals'].append(new_burns)
        results['metrics_per_step'].append(metrics)

        if verbose:
            print(f"Step {step:2d} | "
                  f"IoU: {metrics['iou']:.4f} | "
                  f"F1: {metrics['f1']:.4f} | "
                  f"P: {metrics['precision']:.4f} | "
                  f"R: {metrics['recall']:.4f} | "
                  f"TP: {metrics['tp']:3d} | "
                  f"FP: {metrics['fp']:3d} | "
                  f"FN: {metrics['fn']:3d}")

        obs = next_obs
        step += 1

    # Compute average metrics
    avg_metrics = {}
    for key in ['iou', 'precision', 'recall', 'f1', 'accuracy']:
        values = [m[key] for m in results['metrics_per_step']]
        avg_metrics[key] = np.mean(values)

    print("=" * 80)
    print(f"Average Metrics over {step} steps:")
    print(f"  IoU:       {avg_metrics['iou']:.4f}")
    print(f"  F1:        {avg_metrics['f1']:.4f}")
    print(f"  Precision: {avg_metrics['precision']:.4f}")
    print(f"  Recall:    {avg_metrics['recall']:.4f}")
    print(f"  Accuracy:  {avg_metrics['accuracy']:.4f}")

    results['avg_metrics'] = avg_metrics
    results['num_steps'] = step

    return results


def main():
    parser = argparse.ArgumentParser(description='Run inference with trained A3C model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (.pt file)')
    parser.add_argument('--env-path', type=str, required=True, help='Path to environment file (.pkl)')
    parser.add_argument('--model-type', type=str, default='v2', choices=['v2', 'medium', 'v5'],
                       help='Model architecture type')
    parser.add_argument('--start-t', type=int, default=0, help='Starting timestep')
    parser.add_argument('--max-steps', type=int, default=20, help='Maximum simulation steps')
    parser.add_argument('--quiet', action='store_true', help='Suppress step-by-step output')
    args = parser.parse_args()

    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model = load_model(args.checkpoint, model_type=args.model_type)

    # Run inference
    results = run_inference(
        model,
        args.env_path,
        start_t=args.start_t,
        max_steps=args.max_steps,
        verbose=not args.quiet
    )

    return results


if __name__ == '__main__':
    main()
