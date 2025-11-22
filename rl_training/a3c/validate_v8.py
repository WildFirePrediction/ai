"""
A3C V8 Validation Script - Spatial + Channel Attention

Evaluates V8 attention models on validation set with comprehensive metrics.

Author: Wildfire Prediction Team
Version: 8.0
Date: 2025-11-22
"""

import os
import sys
from pathlib import Path
import pickle
import argparse
import torch
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from wildfire_env_spatial import WildfireEnvSpatial
from a3c.model_v8 import A3C_PerCellModel_V8

# WandB integration (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def compute_iou(pred, target):
    """Compute Intersection over Union."""
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    intersection = (pred_flat * target_flat).sum()
    union = np.clip(pred_flat + target_flat, 0, 1).sum()
    
    return 0.0 if union == 0 else float(intersection / union)


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


def evaluate_episode(env_path, model, device, min_episode_length=4):
    """Evaluate a single episode."""
    env = WildfireEnvSpatial(env_path)
    
    if env.T < min_episode_length:
        return None
    
    obs, info = env.reset()
    assert obs.shape[0] == 15, f"Expected 15 channels, got {obs.shape[0]}"
    
    episode_ious = []
    episode_precisions = []
    episode_recalls = []
    episode_f1s = []
    
    done = False
    t = 0
    
    with torch.no_grad():
        while not done:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            fire_mask = obs[5:6]
            fire_mask_tensor = torch.FloatTensor(fire_mask).to(device)
            
            action_grid, _, _, _, _ = model.get_action_and_value(
                obs_tensor, fire_mask_tensor
            )
            
            action_np = action_grid.cpu().numpy()
            next_obs, reward, done, info = env.step(action_np)
            
            actual_new_burns = env.fire_masks[t + 1] - env.fire_masks[t]
            actual_new_burns = np.clip(actual_new_burns, 0, 1)
            
            iou = compute_iou(action_np, actual_new_burns)
            precision, recall, f1 = compute_precision_recall_f1(action_np, actual_new_burns)
            
            episode_ious.append(iou)
            episode_precisions.append(precision)
            episode_recalls.append(recall)
            episode_f1s.append(f1)
            
            obs = next_obs
            t += 1
    
    return {
        'env_path': str(env_path),
        'episode_length': env.T,
        'mean_iou': float(np.mean(episode_ious)),
        'max_iou': float(np.max(episode_ious)),
        'min_iou': float(np.min(episode_ious)),
        'std_iou': float(np.std(episode_ious)),
        'mean_precision': float(np.mean(episode_precisions)),
        'mean_recall': float(np.mean(episode_recalls)),
        'mean_f1': float(np.mean(episode_f1s)),
    }


def validate_model(checkpoint_path, data_dir, split='val', min_episode_length=4, 
                   max_episodes=None, use_wandb=False, wandb_project='wildfire-a3c-v8'):
    """Run full validation."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading V8 model from {checkpoint_path}")
    model = A3C_PerCellModel_V8(in_channels=15)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'best_iou' in checkpoint:
            print(f"Training IoU: {checkpoint['best_iou']:.4f}")
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load manifest
    manifest_path = Path(data_dir).parent / 'environment_manifest.parquet'
    if manifest_path.exists():
        import pandas as pd
        manifest = pd.read_parquet(manifest_path)
        env_paths = manifest[manifest['split'] == split]['file_path'].tolist()
        env_files = [Path(p).name for p in env_paths]
    else:
        env_files = [f.name for f in sorted(Path(data_dir).glob('*.pkl'))]
    
    print(f"Found {len(env_files)} {split} episodes")
    
    # Filter episodes
    filtered_env_files = []
    for env_file in tqdm(env_files, desc="Filtering"):
        env_path = Path(data_dir) / env_file
        try:
            with open(env_path, 'rb') as f:
                data = pickle.load(f)
                if int(data['metadata']['num_timesteps']) >= min_episode_length:
                    filtered_env_files.append(env_file)
        except:
            continue
    
    if max_episodes:
        filtered_env_files = filtered_env_files[:max_episodes]
    
    print(f"Evaluating {len(filtered_env_files)} episodes")
    
    # Evaluate
    all_metrics = []
    for env_file in tqdm(filtered_env_files, desc="Evaluating"):
        env_path = Path(data_dir) / env_file
        try:
            metrics = evaluate_episode(env_path, model, device, min_episode_length)
            if metrics:
                all_metrics.append(metrics)
        except Exception as e:
            print(f"Error: {env_file}: {e}")
    
    # Aggregate
    mean_ious = [m['mean_iou'] for m in all_metrics]
    
    results = {
        'num_episodes': len(all_metrics),
        'split': split,
        'overall_mean_iou': float(np.mean(mean_ious)),
        'overall_median_iou': float(np.median(mean_ious)),
        'overall_std_iou': float(np.std(mean_ious)),
        'overall_max_iou': float(np.max([m['max_iou'] for m in all_metrics])),
    }
    
    # Print
    print(f"\n{'='*80}")
    print(f"VALIDATION RESULTS - V8 ATTENTION MODEL")
    print(f"{'='*80}")
    print(f"Episodes:     {results['num_episodes']}")
    print(f"Mean IoU:     {results['overall_mean_iou']:.4f} ({results['overall_mean_iou']*100:.2f}%)")
    print(f"Median IoU:   {results['overall_median_iou']:.4f}")
    print(f"Max IoU:      {results['overall_max_iou']:.4f}")
    print(f"{'='*80}")
    
    # Save
    output_dir = Path(checkpoint_path).parent / 'validation_results'
    output_dir.mkdir(exist_ok=True)
    
    results_path = output_dir / f'aggregate_{split}_{datetime.now().strftime("%y%m%d-%H%M")}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate A3C V8 model')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data-dir', type=str, 
                       default='/home/chaseungjoon/code/WildfirePrediction/tilling_data/environments')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'])
    parser.add_argument('--min-episode-length', type=int, default=4)
    parser.add_argument('--max-episodes', type=int, default=None)
    
    args = parser.parse_args()
    
    validate_model(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        split=args.split,
        min_episode_length=args.min_episode_length,
        max_episodes=args.max_episodes,
    )
