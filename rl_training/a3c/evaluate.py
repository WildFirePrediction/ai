"""
Batch Evaluation Script - Test on Validation Set

Usage:
    python evaluate.py --checkpoint path/to/best_model.pt --model-type v2
"""
import argparse
import torch
import numpy as np
import json
from pathlib import Path
import sys
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference import load_model, run_inference


def evaluate_on_validation_set(model, val_env_paths, max_envs=None, max_steps=20):
    """
    Evaluate model on validation set.

    Args:
        model: Trained model
        val_env_paths: List of validation environment paths
        max_envs: Limit number of environments to test (None = all)
        max_steps: Max steps per environment

    Returns:
        Dictionary with aggregated results
    """
    if max_envs:
        val_env_paths = val_env_paths[:max_envs]

    print(f"Evaluating on {len(val_env_paths)} validation environments")
    print(f"Max steps per environment: {max_steps}")
    print("=" * 80)

    all_metrics = []
    successful_envs = 0
    failed_envs = 0

    for env_path in tqdm(val_env_paths, desc="Evaluating"):
        try:
            # Run inference (quiet mode)
            results = run_inference(
                model,
                env_path,
                start_t=0,
                max_steps=max_steps,
                verbose=False
            )

            all_metrics.append(results['avg_metrics'])
            successful_envs += 1

        except Exception as e:
            print(f"\nError on {Path(env_path).name}: {e}")
            failed_envs += 1
            continue

    # Aggregate metrics across all environments
    aggregated = {}
    for key in ['iou', 'precision', 'recall', 'f1', 'accuracy']:
        values = [m[key] for m in all_metrics]
        aggregated[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
        }

    print("\n" + "=" * 80)
    print("VALIDATION SET RESULTS")
    print("=" * 80)
    print(f"Successfully evaluated: {successful_envs}/{len(val_env_paths)}")
    if failed_envs > 0:
        print(f"Failed: {failed_envs}")
    print()

    for metric_name, stats in aggregated.items():
        print(f"{metric_name.upper():10s} | "
              f"Mean: {stats['mean']:.4f} ± {stats['std']:.4f} | "
              f"Min: {stats['min']:.4f} | "
              f"Max: {stats['max']:.4f} | "
              f"Median: {stats['median']:.4f}")

    return {
        'aggregated_metrics': aggregated,
        'num_successful': successful_envs,
        'num_failed': failed_envs,
        'individual_metrics': all_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate A3C model on validation set')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--model-type', type=str, default='v2', choices=['v2', 'medium', 'v5'])
    parser.add_argument('--repo-root', type=str, default='/home/chaseungjoon/code/WildfirePrediction')
    parser.add_argument('--max-envs', type=int, default=None, help='Limit number of validation envs')
    parser.add_argument('--max-steps', type=int, default=20, help='Max steps per environment')
    parser.add_argument('--max-file-size-mb', type=int, default=50, help='Skip large files')
    parser.add_argument('--output', type=str, default=None, help='Save results to JSON file')
    args = parser.parse_args()

    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model = load_model(args.checkpoint, model_type=args.model_type)
    print()

    # Load validation split
    repo_root = Path(args.repo_root)
    val_split_path = repo_root / 'tilling_data' / 'environments' / 'val_split.json'

    with open(val_split_path) as f:
        val_env_ids = json.load(f)

    env_dir = repo_root / 'tilling_data' / 'environments'
    val_paths = [env_dir / f'{eid}.pkl' for eid in val_env_ids]

    # Filter by file size
    max_size = args.max_file_size_mb * 1024 * 1024
    val_paths_filtered = [p for p in val_paths if p.stat().st_size < max_size]

    print(f"Validation environments: {len(val_paths)} total")
    print(f"After size filtering (<{args.max_file_size_mb}MB): {len(val_paths_filtered)}")

    # Run evaluation
    results = evaluate_on_validation_set(
        model,
        val_paths_filtered,
        max_envs=args.max_envs,
        max_steps=args.max_steps
    )

    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_data = {
            'checkpoint': str(args.checkpoint),
            'model_type': args.model_type,
            'num_val_envs': len(val_paths_filtered),
            'aggregated_metrics': results['aggregated_metrics'],
            'num_successful': results['num_successful'],
            'num_failed': results['num_failed'],
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\nResults saved to: {output_path}")

    return results


if __name__ == '__main__':
    main()
