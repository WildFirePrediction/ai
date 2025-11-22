"""
A3C V3 Full Validation Script

Evaluates the best V3 model on validation set and computes comprehensive metrics.
"""
import os
import sys
from pathlib import Path
import pickle
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from wildfire_env_spatial import WildfireEnvSpatial
from a3c.model_v2 import A3C_PerCellModel  # V3 uses model_v2 architecture

# WandB integration (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WandB not available. Metrics will only be saved locally.")


def compute_iou(pred, target):
    """Compute Intersection over Union."""
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    intersection = (pred_flat * target_flat).sum()
    union = np.clip(pred_flat + target_flat, 0, 1).sum()
    
    if union == 0:
        return 0.0
    return float(intersection / union)


def compute_precision_recall_f1(pred, target):
    """Compute precision, recall, and F1 score."""
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    tp = (pred_flat * target_flat).sum()
    fp = (pred_flat * (1 - target_flat)).sum()
    fn = ((1 - pred_flat) * target_flat).sum()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return float(precision), float(recall), float(f1)


def convert_15ch_to_14ch(obs):
    """
    Convert 15-channel observation to 14-channel for old V3 models.
    
    Old (14 channels):
      0-2: static continuous (3)
      3-4: lcm, fsm (2)
      5-8: fire_mask, fire_intensity, fire_temp, fire_age (4)
      9-13: weather - temp, humidity, wind_speed, wind_x, wind_y (5)
    
    New (15 channels):
      0-2: static continuous (3)
      3-4: lcm, fsm (2)
      5-8: fire_mask, fire_intensity, fire_temp, fire_age (4)
      9-14: weather - temp, humidity, wind_speed, wind_x, wind_y, rainfall (6)
    
    Solution: Remove rainfall channel (index 14)
    """
    if obs.shape[0] == 15:
        # Remove rainfall channel (last weather channel)
        return obs[:14]  # Keep channels 0-13, drop 14 (rainfall)
    return obs


def evaluate_episode(env_path, model, device, min_episode_length=4):
    """
    Evaluate a single episode.
    
    Args:
        env_path: Path to environment pickle file
        model: Trained A3C model
        device: torch device
        min_episode_length: Minimum episode length filter
        
    Returns:
        metrics: Dictionary of metrics for this episode
        None if episode too short
    """
    env = WildfireEnvSpatial(env_path)
    
    # Check episode length
    if env.T < min_episode_length:
        return None
    
    obs, info = env.reset()
    
    # Convert 15-channel obs to 14-channel for old models
    obs = convert_15ch_to_14ch(obs)
    
    episode_ious = []
    episode_precisions = []
    episode_recalls = []
    episode_f1s = []
    total_burning_cells = []
    total_predicted_cells = []
    total_actual_new_burns = []
    
    done = False
    t = 0
    
    with torch.no_grad():
        while not done:
            # Convert observation to tensor
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)  # (1, 14, H, W)
            
            # Extract fire mask (channel 5 in 14-ch format after lcm/fsm)
            fire_mask = obs[5:6]  # Fire mask channel
            fire_mask_tensor = torch.FloatTensor(fire_mask).to(device)  # (1, H, W)
            
            # Get action from model
            action_grid, log_prob, entropy, value, burning_info = model.get_action_and_value(
                obs_tensor, fire_mask_tensor
            )
            
            # Convert to numpy
            action_np = action_grid.cpu().numpy()
            
            # Step environment
            next_obs, reward, done, info = env.step(action_np)
            
            # Convert next obs to 14-channel
            next_obs = convert_15ch_to_14ch(next_obs)
            
            # Get actual new burns (compare t+1 with t)
            actual_new_burns = env.fire_masks[t + 1] - env.fire_masks[t]
            actual_new_burns = np.clip(actual_new_burns, 0, 1)
            
            # Compute metrics
            iou = compute_iou(action_np, actual_new_burns)
            precision, recall, f1 = compute_precision_recall_f1(action_np, actual_new_burns)
            
            episode_ious.append(iou)
            episode_precisions.append(precision)
            episode_recalls.append(recall)
            episode_f1s.append(f1)
            total_burning_cells.append(int(fire_mask.sum()))
            total_predicted_cells.append(int(action_np.sum()))
            total_actual_new_burns.append(int(actual_new_burns.sum()))
            
            obs = next_obs
            t += 1
    
    # Aggregate episode metrics
    metrics = {
        'env_path': str(env_path),
        'episode_length': env.T,
        'mean_iou': float(np.mean(episode_ious)),
        'max_iou': float(np.max(episode_ious)),
        'min_iou': float(np.min(episode_ious)),
        'std_iou': float(np.std(episode_ious)),
        'mean_precision': float(np.mean(episode_precisions)),
        'mean_recall': float(np.mean(episode_recalls)),
        'mean_f1': float(np.mean(episode_f1s)),
        'avg_burning_cells': float(np.mean(total_burning_cells)),
        'avg_predicted_cells': float(np.mean(total_predicted_cells)),
        'avg_actual_new_burns': float(np.mean(total_actual_new_burns)),
        'timestep_ious': episode_ious,
        'timestep_precisions': episode_precisions,
        'timestep_recalls': episode_recalls,
        'timestep_f1s': episode_f1s,
    }
    
    return metrics


def validate_model(checkpoint_path, data_dir, split='val', min_episode_length=4, 
                   max_episodes=None, use_wandb=True, wandb_project='wildfire-a3c-v3'):
    """
    Run full validation on the model.
    
    Args:
        checkpoint_path: Path to model checkpoint
        data_dir: Directory containing environment pickle files
        split: 'val' or 'train' (for validation or train set)
        min_episode_length: Minimum episode length filter
        max_episodes: Maximum number of episodes to evaluate (None = all)
        use_wandb: Whether to use WandB logging
        wandb_project: WandB project name
    """
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {checkpoint_path}")
    model = A3C_PerCellModel(in_channels=14)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize WandB
    if use_wandb and WANDB_AVAILABLE:
        run = wandb.init(
            project=wandb_project,
            name=f"validation_{split}_{datetime.now().strftime('%y%m%d-%H%M')}",
            config={
                'checkpoint': str(checkpoint_path),
                'split': split,
                'min_episode_length': min_episode_length,
                'max_episodes': max_episodes,
            }
        )
    
    # Load environment manifest to get train/val split
    manifest_path = Path(data_dir).parent / 'environment_manifest.parquet'
    if manifest_path.exists():
        import pandas as pd
        manifest = pd.read_parquet(manifest_path)
        
        if split == 'val':
            env_paths = manifest[manifest['split'] == 'val']['file_path'].tolist()
        else:
            env_paths = manifest[manifest['split'] == 'train']['file_path'].tolist()
        
        # Extract just the filename from file_path (remove 'environments/' prefix)
        env_files = [Path(p).name for p in env_paths]
    else:
        # Fallback: use all environments
        print(f"Warning: No manifest found at {manifest_path}. Using all environments.")
        env_files = sorted(Path(data_dir).glob('*.pkl'))
        env_files = [f.name for f in env_files]
    
    print(f"Found {len(env_files)} {split} episodes")
    
    # Filter by episode length and limit
    filtered_env_files = []
    for env_file in tqdm(env_files, desc="Filtering episodes"):
        env_path = Path(data_dir) / env_file
        try:
            with open(env_path, 'rb') as f:
                data = pickle.load(f)
                T = int(data['metadata']['num_timesteps'])
                if T >= min_episode_length:
                    filtered_env_files.append(env_file)
        except Exception as e:
            print(f"Error loading {env_file}: {e}")
            continue
    
    print(f"After filtering (min_length={min_episode_length}): {len(filtered_env_files)} episodes")
    
    if max_episodes:
        filtered_env_files = filtered_env_files[:max_episodes]
        print(f"Limiting to {max_episodes} episodes")
    
    # Evaluate all episodes
    all_metrics = []
    for env_file in tqdm(filtered_env_files, desc="Evaluating episodes"):
        env_path = Path(data_dir) / env_file
        
        try:
            metrics = evaluate_episode(env_path, model, device, min_episode_length)
            if metrics:
                all_metrics.append(metrics)
        except Exception as e:
            print(f"Error evaluating {env_file}: {e}")
            continue
    
    print(f"\nSuccessfully evaluated {len(all_metrics)} episodes")
    
    # Aggregate results
    mean_ious = [m['mean_iou'] for m in all_metrics]
    max_ious = [m['max_iou'] for m in all_metrics]
    mean_precisions = [m['mean_precision'] for m in all_metrics]
    mean_recalls = [m['mean_recall'] for m in all_metrics]
    mean_f1s = [m['mean_f1'] for m in all_metrics]
    
    aggregate_results = {
        'num_episodes': len(all_metrics),
        'split': split,
        'min_episode_length': min_episode_length,
        'checkpoint': str(checkpoint_path),
        
        # IoU statistics
        'overall_mean_iou': float(np.mean(mean_ious)),
        'overall_median_iou': float(np.median(mean_ious)),
        'overall_std_iou': float(np.std(mean_ious)),
        'overall_max_iou': float(np.max(max_ious)),
        'overall_min_iou': float(np.min(mean_ious)),
        
        # Precision/Recall/F1 statistics
        'overall_mean_precision': float(np.mean(mean_precisions)),
        'overall_mean_recall': float(np.mean(mean_recalls)),
        'overall_mean_f1': float(np.mean(mean_f1s)),
        
        # Distribution statistics
        'iou_percentile_25': float(np.percentile(mean_ious, 25)),
        'iou_percentile_50': float(np.percentile(mean_ious, 50)),
        'iou_percentile_75': float(np.percentile(mean_ious, 75)),
        'iou_percentile_90': float(np.percentile(mean_ious, 90)),
        'iou_percentile_95': float(np.percentile(mean_ious, 95)),
    }
    
    # Print results
    print("\n" + "="*80)
    print(f"VALIDATION RESULTS - {split.upper()} SET")
    print("="*80)
    print(f"Episodes evaluated: {aggregate_results['num_episodes']}")
    print(f"Min episode length: {min_episode_length}")
    print("\nIoU Statistics:")
    print(f"  Mean IoU:   {aggregate_results['overall_mean_iou']:.4f} ({aggregate_results['overall_mean_iou']*100:.2f}%)")
    print(f"  Median IoU: {aggregate_results['overall_median_iou']:.4f} ({aggregate_results['overall_median_iou']*100:.2f}%)")
    print(f"  Std IoU:    {aggregate_results['overall_std_iou']:.4f}")
    print(f"  Max IoU:    {aggregate_results['overall_max_iou']:.4f} ({aggregate_results['overall_max_iou']*100:.2f}%)")
    print(f"  Min IoU:    {aggregate_results['overall_min_iou']:.4f} ({aggregate_results['overall_min_iou']*100:.2f}%)")
    print("\nPercentiles:")
    print(f"  25th: {aggregate_results['iou_percentile_25']:.4f} ({aggregate_results['iou_percentile_25']*100:.2f}%)")
    print(f"  50th: {aggregate_results['iou_percentile_50']:.4f} ({aggregate_results['iou_percentile_50']*100:.2f}%)")
    print(f"  75th: {aggregate_results['iou_percentile_75']:.4f} ({aggregate_results['iou_percentile_75']*100:.2f}%)")
    print(f"  90th: {aggregate_results['iou_percentile_90']:.4f} ({aggregate_results['iou_percentile_90']*100:.2f}%)")
    print(f"  95th: {aggregate_results['iou_percentile_95']:.4f} ({aggregate_results['iou_percentile_95']*100:.2f}%)")
    print("\nOther Metrics:")
    print(f"  Precision: {aggregate_results['overall_mean_precision']:.4f}")
    print(f"  Recall:    {aggregate_results['overall_mean_recall']:.4f}")
    print(f"  F1 Score:  {aggregate_results['overall_mean_f1']:.4f}")
    print("="*80)
    
    # Save results
    output_dir = Path(checkpoint_path).parent / 'validation_results'
    output_dir.mkdir(exist_ok=True)
    
    # Save aggregate results
    aggregate_path = output_dir / f'aggregate_{split}_{datetime.now().strftime("%y%m%d-%H%M")}.json'
    with open(aggregate_path, 'w') as f:
        json.dump(aggregate_results, f, indent=2)
    print(f"\nAggregate results saved to: {aggregate_path}")
    
    # Save detailed results
    detailed_path = output_dir / f'detailed_{split}_{datetime.now().strftime("%y%m%d-%H%M")}.json'
    with open(detailed_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Detailed results saved to: {detailed_path}")
    
    # Log to WandB
    if use_wandb and WANDB_AVAILABLE:
        wandb.log(aggregate_results)
        
        # Create histogram of IoUs
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.hist(mean_ious, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Mean IoU')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of Mean IoU - {split.upper()} Set')
        plt.axvline(aggregate_results['overall_mean_iou'], color='r', linestyle='--', 
                   label=f"Mean: {aggregate_results['overall_mean_iou']:.4f}")
        plt.axvline(aggregate_results['overall_median_iou'], color='g', linestyle='--',
                   label=f"Median: {aggregate_results['overall_median_iou']:.4f}")
        plt.legend()
        wandb.log({"iou_distribution": wandb.Image(plt)})
        plt.close()
        
        wandb.finish()
    
    return aggregate_results, all_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate A3C V3 model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, 
                       default='/home/chaseungjoon/code/WildfirePrediction/tilling_data/environments',
                       help='Directory containing environment pickle files')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'],
                       help='Dataset split to validate on')
    parser.add_argument('--min-episode-length', type=int, default=4,
                       help='Minimum episode length filter')
    parser.add_argument('--max-episodes', type=int, default=None,
                       help='Maximum number of episodes to evaluate (None = all)')
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable WandB logging')
    parser.add_argument('--wandb-project', type=str, default='wildfire-a3c-v3',
                       help='WandB project name')
    
    args = parser.parse_args()
    
    validate_model(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        split=args.split,
        min_episode_length=args.min_episode_length,
        max_episodes=args.max_episodes,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
    )
