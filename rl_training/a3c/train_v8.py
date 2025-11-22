"""
A3C Training V8 - Spatial + Channel Attention

Main training script for V8 model with dual attention mechanisms.

Architecture: V3 Encoder + Spatial Attention + Channel Attention
Expected Performance: 40-50% IoU (baseline 31.82% + 8-18% improvement)

Key Features:
- Attention-enhanced feature extraction
- Dense rewards (IoU at every timestep)
- Per-cell 8-neighbor prediction
- Distributed training with 4 workers (optimal for hardware)
- WandB logging and checkpointing

Hardware Target: RTX 5070 12GB VRAM, 64GB RAM, Ryzen9 9950x 16C/32T

Author: Wildfire Prediction Team
Version: 8.0
Date: 2025-11-22
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
from datetime import datetime

torch.set_num_threads(1)

sys.path.insert(0, str(Path(__file__).parent.parent))

from a3c.model_v8 import A3C_PerCellModel_V8
from a3c.worker_v8 import worker_process_v8
from wildfire_env_spatial import WildfireEnvSpatial


def create_filtered_episode_list(
    env_paths: list,
    max_file_size_mb: int = 50,
    min_episode_length: int = 4
) -> list:
    """
    Pre-scan environments and filter for high-quality training episodes.
    
    Quality criteria:
    - File size < max_file_size_mb (exclude overly complex environments)
    - At least min_episode_length timesteps with actual fire spread
    - Episodes must have new burns (not just static fire)
    
    Args:
        env_paths: List of environment file paths
        max_file_size_mb: Maximum file size in MB
        min_episode_length: Minimum timesteps with new burns
    
    Returns:
        List of (env_path, start_timestep, episode_length) tuples
    """
    MAX_SIZE = max_file_size_mb * 1024 * 1024
    filtered_episodes = []
    
    print(f"\n{'='*80}")
    print(f"EPISODE FILTERING")
    print(f"{'='*80}")
    print(f"Scanning {len(env_paths)} environments...")
    print(f"Criteria:")
    print(f"  - File size < {max_file_size_mb} MB")
    print(f"  - Min {min_episode_length} timesteps with fire spread")
    print(f"{'='*80}\n")
    
    total_envs_scanned = 0
    total_envs_kept = 0
    total_episodes_found = 0
    
    for env_path in tqdm(env_paths, desc="Scanning environments"):
        # Skip large files (may be too complex or corrupted)
        if env_path.stat().st_size >= MAX_SIZE:
            continue
        
        total_envs_scanned += 1
        
        try:
            env = WildfireEnvSpatial(env_path)
            
            # Find valid starting points with sufficient fire spread
            env_has_episodes = False
            for start_t in range(env.T - min_episode_length):
                # Count timesteps with new burns in this window
                burns_count = 0
                for t in range(start_t, min(start_t + min_episode_length + 5, env.T - 1)):
                    actual_mask_t = env.fire_masks[t] > 0
                    actual_mask_t1 = env.fire_masks[t + 1] > 0
                    new_burns = (actual_mask_t1 & ~actual_mask_t)
                    
                    if new_burns.sum() > 0:
                        burns_count += 1
                
                # Add if quality threshold met
                if burns_count >= min_episode_length:
                    max_length = min(env.T - start_t - 1, 20)  # Cap at 20 timesteps
                    filtered_episodes.append((env_path, start_t, max_length))
                    total_episodes_found += 1
                    env_has_episodes = True
            
            if env_has_episodes:
                total_envs_kept += 1
        
        except Exception as e:
            print(f"  Warning: Error loading {env_path.name}: {e}")
            continue
    
    print(f"\n{'='*80}")
    print(f"FILTERING RESULTS")
    print(f"{'='*80}")
    print(f"Environments scanned:     {total_envs_scanned}")
    print(f"Environments kept:        {total_envs_kept} ({100*total_envs_kept/max(1,total_envs_scanned):.1f}%)")
    print(f"Total episodes found:     {total_episodes_found}")
    print(f"Avg episodes per env:     {total_episodes_found / max(1, total_envs_kept):.1f}")
    print(f"{'='*80}\n")
    
    return filtered_episodes


def main():
    """Main training function."""
    
    parser = argparse.ArgumentParser(
        description='A3C V8 Training - Spatial + Channel Attention',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Environment settings
    parser.add_argument('--repo-root', type=str, 
                       default='/home/chaseungjoon/code/WildfirePrediction',
                       help='Repository root directory')
    
    # Training settings
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of parallel workers (optimal: 4 for hardware)')
    parser.add_argument('--max-episodes', type=int, default=2000,
                       help='Total training episodes across all workers')
    parser.add_argument('--min-episode-length', type=int, default=4,
                       help='Minimum timesteps with burns per episode')
    parser.add_argument('--max-file-size-mb', type=int, default=50,
                       help='Maximum environment file size in MB')
    parser.add_argument('--max-envs', type=int, default=None,
                       help='Limit number of training environments (for debugging)')
    
    # Model hyperparameters
    parser.add_argument('--lr', type=float, default=7e-5,
                       help='Learning rate (proven effective for V3)')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor for future rewards')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                       help='GAE lambda for advantage estimation')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                       help='Value loss coefficient')
    parser.add_argument('--entropy-coef', type=float, default=0.015,
                       help='Entropy coefficient (proven effective for V3)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                       help='Maximum gradient norm for clipping')
    
    # Logging settings
    parser.add_argument('--log-interval', type=int, default=10,
                       help='Log every N episodes')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # WandB settings
    parser.add_argument('--wandb-project', type=str, default='wildfire-prediction',
                       help='WandB project name')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                       help='WandB run name (auto-generated if not provided)')
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable WandB logging')
    
    args = parser.parse_args()
    
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    # Setup paths
    repo_root = Path(args.repo_root)
    env_dir = repo_root / 'tilling_data' / 'environments'
    
    # Load training split
    train_split_path = env_dir / 'train_split.json'
    if not train_split_path.exists():
        raise FileNotFoundError(f"Training split not found: {train_split_path}")
    
    with open(train_split_path) as f:
        train_env_ids = json.load(f)
    
    if args.max_envs:
        train_env_ids = train_env_ids[:args.max_envs]
    
    train_paths = [env_dir / f'{eid}.pkl' for eid in train_env_ids]
    
    # Print header
    print(f"\n{'='*80}")
    print(f"A3C V8 TRAINING - SPATIAL + CHANNEL ATTENTION")
    print(f"{'='*80}")
    print(f"Architecture:     V3 Encoder + Spatial Attention + Channel Attention")
    print(f"Problem:          Per-cell 8-neighbor fire spread prediction")
    print(f"Rewards:          Dense IoU at every timestep")
    print(f"Input Channels:   15 (terrain + fire state + weather w/ rainfall)")
    print(f"Expected Gain:    +8-18% IoU over V3 baseline")
    print(f"{'='*80}")
    print(f"Training Setup:")
    print(f"  Total environments:  {len(train_paths)}")
    print(f"  Workers:             {args.num_workers}")
    print(f"  Max episodes:        {args.max_episodes}")
    print(f"  Learning rate:       {args.lr}")
    print(f"  Entropy coef:        {args.entropy_coef}")
    print(f"  Min episode length:  {args.min_episode_length}")
    print(f"{'='*80}\n")
    
    # Filter episodes
    filtered_episodes = create_filtered_episode_list(
        train_paths,
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
    
    # Initialize WandB
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb_run_name = args.wandb_run_name or f"v8-attention-w{args.num_workers}-mel{args.min_episode_length}"
        wandb.init(
            project=args.wandb_project,
            name=wandb_run_name,
            config={
                'model': 'A3C-V8-Attention',
                'architecture': 'V3-Encoder + Spatial-Attention + Channel-Attention',
                'formulation': 'per_cell_8_neighbor',
                'rewards': 'dense_iou_per_timestep',
                'input_channels': 15,
                'num_workers': args.num_workers,
                'learning_rate': args.lr,
                'max_episodes': args.max_episodes,
                'gamma': args.gamma,
                'gae_lambda': args.gae_lambda,
                'value_loss_coef': args.value_loss_coef,
                'entropy_coef': args.entropy_coef,
                'max_grad_norm': args.max_grad_norm,
                'seed': args.seed,
                'total_filtered_episodes': len(filtered_episodes),
                'min_episode_length': args.min_episode_length,
                'max_file_size_mb': args.max_file_size_mb,
            }
        )
        print(f"WandB initialized: {wandb.run.name}\n")
    else:
        print("WandB logging disabled\n")
    
    # Create shared model
    shared_model = A3C_PerCellModel_V8(in_channels=15)
    shared_model.share_memory()
    shared_model.train()
    
    # Count parameters
    total_params = sum(p.numel() for p in shared_model.parameters())
    print(f"Model created:")
    print(f"  Total parameters:    {total_params:,}")
    print(f"  Memory per worker:   ~25 MB")
    print(f"  Total GPU memory:    ~5 GB / 12 GB (safe)\n")
    
    # Create timestamped checkpoint directory
    timestamp = datetime.now().strftime("%y%m%d-%H%M")
    ckpt_dir = repo_root / 'rl_training' / 'a3c' / 're_checkpoints_v8' / timestamp
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    print(f"Checkpoint directory: {ckpt_dir}\n")
    
    # Create shared optimizer
    optimizer = torch.optim.Adam(shared_model.parameters(), lr=args.lr)
    
    # Shared counters
    global_episode_counter = mp.Value('i', 0)
    global_best_iou = mp.Value('d', 0.0)
    lock = mp.Lock()
    
    # Metrics queue for WandB
    metrics_queue = mp.Queue() if use_wandb else None
    
    # Training configuration
    config = {
        'seed': args.seed,
        'gamma': args.gamma,
        'gae_lambda': args.gae_lambda,
        'value_loss_coef': args.value_loss_coef,
        'entropy_coef': args.entropy_coef,
        'max_grad_norm': args.max_grad_norm,
        'max_episodes': args.max_episodes,
        'log_interval': args.log_interval,
        'checkpoint_dir': str(ckpt_dir),
    }
    
    # Launch workers
    print(f"{'='*80}")
    print(f"LAUNCHING {args.num_workers} WORKERS")
    print(f"{'='*80}\n")
    
    processes = []
    for worker_id in range(args.num_workers):
        p = mp.Process(
            target=worker_process_v8,
            args=(worker_id, shared_model, optimizer, filtered_episodes, config,
                  global_episode_counter, global_best_iou, lock, metrics_queue)
        )
        p.start()
        processes.append(p)
        print(f"Worker {worker_id} started (PID: {p.pid})")
    
    print(f"\nAll workers launched. Training in progress...\n")
    
    # Monitor metrics queue and log to WandB
    if use_wandb:
        import time
        while any(p.is_alive() for p in processes):
            while not metrics_queue.empty():
                metrics = metrics_queue.get()
                wandb.log(metrics)
            time.sleep(0.1)
        
        # Drain remaining metrics
        while not metrics_queue.empty():
            metrics = metrics_queue.get()
            wandb.log(metrics)
    else:
        # Just wait for workers
        for p in processes:
            p.join()
    
    # Save final model
    final_path = ckpt_dir / 'final_model.pt'
    torch.save({
        'model_state_dict': shared_model.state_dict(),
        'best_iou': global_best_iou.value,
        'total_episodes': global_episode_counter.value,
        'config': vars(args),
        'filtered_episodes_count': len(filtered_episodes),
        'architecture': 'V8-Spatial-Channel-Attention',
        'input_channels': 15,
    }, final_path)
    
    # Print final summary
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Total episodes:       {global_episode_counter.value}")
    print(f"Best validation IoU:  {global_best_iou.value:.4f} ({global_best_iou.value*100:.2f}%)")
    print(f"Checkpoints saved to: {ckpt_dir}")
    print(f"  - best_model.pt:    Best model (IoU {global_best_iou.value:.4f})")
    print(f"  - final_model.pt:   Final model")
    print(f"{'='*80}\n")
    
    # Log final model to WandB
    if use_wandb:
        artifact = wandb.Artifact(
            name='v8-attention-model',
            type='model',
            description=f'A3C V8 Attention after {global_episode_counter.value} episodes, IoU {global_best_iou.value:.4f}'
        )
        artifact.add_file(str(final_path))
        wandb.log_artifact(artifact)
        
        wandb.summary['total_episodes'] = global_episode_counter.value
        wandb.summary['best_iou'] = global_best_iou.value
        wandb.summary['final_checkpoint'] = str(final_path)
        
        wandb.finish()
    
    print("Training completed successfully!")


if __name__ == '__main__':
    main()
