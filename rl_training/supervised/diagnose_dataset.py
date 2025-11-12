"""
Diagnostic script to analyze dataset and identify accuracy issues.
"""
import json
import torch
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
import pickle

sys.path.insert(0, str(Path(__file__).parent.parent))
from wildfire_env_spatial import WildfireEnvSpatial


def analyze_dataset(env_paths, max_envs=50, max_samples_per_env=10):
    """Analyze dataset to identify issues."""

    print("=" * 80)
    print("DATASET DIAGNOSTIC ANALYSIS")
    print("=" * 80)

    # Statistics to collect
    stats = {
        'total_samples': 0,
        'positive_pixels': [],
        'negative_pixels': [],
        'positive_ratio': [],
        'spatial_sizes': [],
        'has_burns': 0,
        'no_burns': 0,
        'env_sizes': [],
        'timesteps': [],
    }

    print(f"\nAnalyzing {min(len(env_paths), max_envs)} environments...")

    for env_idx, env_path in enumerate(tqdm(env_paths[:max_envs], desc="Loading envs")):
        # Check file size
        file_size_mb = env_path.stat().st_size / (1024 * 1024)
        stats['env_sizes'].append(file_size_mb)

        # Load environment
        env = WildfireEnvSpatial(env_path)
        stats['timesteps'].append(env.T)
        stats['spatial_sizes'].append((env.H, env.W))

        # Analyze samples from this environment
        num_samples = min(env.T - 1, max_samples_per_env)
        for t in range(num_samples):
            # Get target: new burns at t+1
            actual_mask_t = env.fire_masks[t] > 0
            actual_mask_t1 = env.fire_masks[t + 1] > 0
            target = (actual_mask_t1 & ~actual_mask_t).astype(np.float32)

            # Count pixels
            positive = target.sum()
            negative = target.size - positive
            total = target.size

            stats['total_samples'] += 1
            stats['positive_pixels'].append(positive)
            stats['negative_pixels'].append(negative)
            stats['positive_ratio'].append(positive / total if total > 0 else 0)

            if positive > 0:
                stats['has_burns'] += 1
            else:
                stats['no_burns'] += 1

    # Compute summary statistics
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print(f"\n1. DATASET SIZE")
    print(f"   Total samples analyzed: {stats['total_samples']}")
    print(f"   Samples with burns: {stats['has_burns']} ({100*stats['has_burns']/stats['total_samples']:.1f}%)")
    print(f"   Samples without burns: {stats['no_burns']} ({100*stats['no_burns']/stats['total_samples']:.1f}%)")

    print(f"\n2. CLASS IMBALANCE (Critical!)")
    pos_ratio = np.array(stats['positive_ratio'])
    print(f"   Average positive ratio: {pos_ratio.mean()*100:.4f}%")
    print(f"   Median positive ratio: {np.median(pos_ratio)*100:.4f}%")
    print(f"   Max positive ratio: {pos_ratio.max()*100:.4f}%")
    print(f"   Min positive ratio: {pos_ratio.min()*100:.4f}%")

    avg_pos = np.mean(stats['positive_pixels'])
    avg_neg = np.mean(stats['negative_pixels'])
    print(f"\n   Average positive pixels per sample: {avg_pos:.1f}")
    print(f"   Average negative pixels per sample: {avg_neg:.1f}")
    print(f"   Imbalance ratio (neg:pos): {avg_neg/max(avg_pos,1):.1f}:1")

    print(f"\n3. SPATIAL DIMENSIONS")
    sizes = np.array([(h*w) for h, w in stats['spatial_sizes']])
    print(f"   Average spatial size: {sizes.mean():.0f} pixels")
    print(f"   Min spatial size: {sizes.min():.0f} pixels")
    print(f"   Max spatial size: {sizes.max():.0f} pixels")

    print(f"\n4. FILE SIZES")
    env_sizes = np.array(stats['env_sizes'])
    print(f"   Average file size: {env_sizes.mean():.2f} MB")
    print(f"   Median file size: {np.median(env_sizes):.2f} MB")
    print(f"   Max file size: {env_sizes.max():.2f} MB")

    print(f"\n5. TIMESTEPS")
    timesteps = np.array(stats['timesteps'])
    print(f"   Average timesteps: {timesteps.mean():.1f}")
    print(f"   Min timesteps: {timesteps.min()}")
    print(f"   Max timesteps: {timesteps.max()}")

    # Compute expected "always predict zero" performance
    avg_pos_ratio = pos_ratio.mean()
    always_zero_accuracy = 1 - avg_pos_ratio

    print("\n" + "=" * 80)
    print("CRITICAL INSIGHT: Baseline Performance")
    print("=" * 80)
    print(f"If model predicts 'no burn' everywhere (all zeros):")
    print(f"  Accuracy: {always_zero_accuracy*100:.2f}%")
    print(f"  IoU: undefined (no predictions, no targets in many samples)")
    print("\nThis is likely why your model performance is poor!")
    print("The model learns to predict 'no burn' to minimize BCE loss.")

    # Analyze a few specific samples in detail
    print("\n" + "=" * 80)
    print("SAMPLE ANALYSIS (First 5 samples with burns)")
    print("=" * 80)

    sample_count = 0
    for env_idx, env_path in enumerate(env_paths[:20]):
        env = WildfireEnvSpatial(env_path)
        for t in range(min(5, env.T - 1)):
            actual_mask_t = env.fire_masks[t] > 0
            actual_mask_t1 = env.fire_masks[t + 1] > 0
            target = (actual_mask_t1 & ~actual_mask_t).astype(np.float32)

            if target.sum() > 0 and sample_count < 5:
                print(f"\nSample {sample_count + 1}:")
                print(f"  Env: {env_path.name}")
                print(f"  Timestep: {t} -> {t+1}")
                print(f"  Spatial size: {env.H}x{env.W} = {env.H*env.W} pixels")
                print(f"  Cells burning at t: {actual_mask_t.sum()}")
                print(f"  Cells burning at t+1: {actual_mask_t1.sum()}")
                print(f"  NEW burns (target): {target.sum()}")
                print(f"  Positive ratio: {target.sum()/(env.H*env.W)*100:.4f}%")
                sample_count += 1

        if sample_count >= 5:
            break

    print("\n" + "=" * 80)

    return stats


def main():
    repo_root = Path('/home/chaseungjoon/code/WildfirePrediction')
    env_dir = repo_root / 'tilling_data' / 'environments'

    # Load train split
    train_split_path = env_dir / 'train_split.json'
    with open(train_split_path) as f:
        train_env_ids = json.load(f)

    train_paths = [env_dir / f'{eid}.pkl' for eid in train_env_ids]

    # Filter out large files (>10MB) to avoid hanging
    print("Filtering out large files (>10MB)...")
    MAX_SIZE = 10 * 1024 * 1024  # 10 MB
    filtered_paths = [p for p in train_paths if p.stat().st_size < MAX_SIZE]
    print(f"Kept {len(filtered_paths)}/{len(train_paths)} environments")

    # Analyze dataset
    stats = analyze_dataset(filtered_paths, max_envs=100, max_samples_per_env=10)


if __name__ == '__main__':
    main()
