"""
U-Net Training V2 - Fixed for Class Imbalance

Major improvements:
1. Filter out zero-burn samples
2. Weighted BCE loss with pos_weight
3. Fixed per-sample IoU computation
4. Added precision/recall/F1 metrics
"""
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import sys
from tqdm import tqdm
import wandb

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from wildfire_env_spatial import WildfireEnvSpatial
from supervised.unet_model import UNet


class FireSpreadDatasetFiltered(Dataset):
    """Dataset for fire spread prediction - FILTERS OUT ZERO-BURN SAMPLES."""

    def __init__(self, env_paths, max_file_size_mb=10):
        self.env_paths = []
        self.samples = []

        # Filter out large files first
        MAX_SIZE = max_file_size_mb * 1024 * 1024
        print(f"Filtering environment files (max size: {max_file_size_mb}MB)...")
        for path in env_paths:
            if path.stat().st_size < MAX_SIZE:
                self.env_paths.append(path)

        print(f"Kept {len(self.env_paths)}/{len(env_paths)} environments after size filtering")

        # Pre-compute all (env_idx, timestep) pairs - ONLY KEEP SAMPLES WITH BURNS
        print(f"Loading {len(self.env_paths)} environments and filtering zero-burn samples...")
        total_samples_before = 0
        total_samples_after = 0

        for env_idx, env_path in enumerate(tqdm(self.env_paths, desc="Scanning envs")):
            env = WildfireEnvSpatial(env_path)
            # Can predict for timesteps 0 to T-2 (predict t+1)
            for t in range(env.T - 1):
                total_samples_before += 1

                # Check if this sample has any burns
                actual_mask_t = env.fire_masks[t] > 0
                actual_mask_t1 = env.fire_masks[t + 1] > 0
                target = (actual_mask_t1 & ~actual_mask_t)

                # ONLY include samples with at least 1 burn
                if target.sum() > 0:
                    self.samples.append((env_idx, t))
                    total_samples_after += 1

        print(f"Filtered samples: {total_samples_before} → {total_samples_after}")
        print(f"Kept {100*total_samples_after/max(total_samples_before,1):.1f}% of samples")
        print(f"Total training samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        env_idx, t = self.samples[idx]
        env = WildfireEnvSpatial(self.env_paths[env_idx])

        # Get observation at time t
        obs, _ = env.reset()
        for _ in range(t):
            env.step(np.zeros((env.H, env.W)))  # Dummy action to advance time

        obs = env._get_obs(t)

        # Get target: new burns at t+1
        actual_mask_t = env.fire_masks[t] > 0
        actual_mask_t1 = env.fire_masks[t + 1] > 0
        target = (actual_mask_t1 & ~actual_mask_t).astype(np.float32)

        return torch.from_numpy(obs).float(), torch.from_numpy(target).float().unsqueeze(0)


def collate_fn(batch):
    """Custom collate function to pad variable-sized spatial dimensions."""
    obs_list, target_list = zip(*batch)

    # Find max dimensions in this batch
    max_h = max(obs.shape[1] for obs in obs_list)
    max_w = max(obs.shape[2] for obs in obs_list)

    # Pad all observations and targets to max dimensions
    obs_padded = []
    target_padded = []

    for obs, target in zip(obs_list, target_list):
        c, h, w = obs.shape
        # Pad observation: (C, H, W) -> (C, max_h, max_w)
        pad_h = max_h - h
        pad_w = max_w - w
        obs_pad = F.pad(obs, (0, pad_w, 0, pad_h), mode='constant', value=0)
        obs_padded.append(obs_pad)

        # Pad target: (1, H, W) -> (1, max_h, max_w)
        target_pad = F.pad(target, (0, pad_w, 0, pad_h), mode='constant', value=0)
        target_padded.append(target_pad)

    # Stack into batch
    obs_batch = torch.stack(obs_padded, dim=0)
    target_batch = torch.stack(target_padded, dim=0)

    return obs_batch, target_batch


def compute_metrics(pred, target):
    """
    Compute comprehensive metrics per sample, then average.

    Returns:
        dict with IoU, precision, recall, F1
    """
    pred = (pred > 0.5).float()
    batch_size = pred.shape[0]

    metrics = {
        'iou': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    for i in range(batch_size):
        p = pred[i].flatten()
        t = target[i].flatten()

        tp = (p * t).sum().item()
        fp = (p * (1 - t)).sum().item()
        fn = ((1 - p) * t).sum().item()
        tn = ((1 - p) * (1 - t)).sum().item()

        # IoU
        intersection = tp
        union = tp + fp + fn
        if union > 0:
            metrics['iou'].append(intersection / union)

        # Precision
        if tp + fp > 0:
            metrics['precision'].append(tp / (tp + fp))

        # Recall
        if tp + fn > 0:
            metrics['recall'].append(tp / (tp + fn))

        # F1
        if tp + fp > 0 and tp + fn > 0:
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
            if prec + rec > 0:
                metrics['f1'].append(2 * prec * rec / (prec + rec))

    # Average metrics (handle empty lists)
    return {
        'iou': np.mean(metrics['iou']) if metrics['iou'] else 0.0,
        'precision': np.mean(metrics['precision']) if metrics['precision'] else 0.0,
        'recall': np.mean(metrics['recall']) if metrics['recall'] else 0.0,
        'f1': np.mean(metrics['f1']) if metrics['f1'] else 0.0,
    }


def train_epoch(model, dataloader, optimizer, device, epoch, pos_weight):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_metrics = {'iou': 0, 'precision': 0, 'recall': 0, 'f1': 0}

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (obs, target) in enumerate(pbar):
        obs, target = obs.to(device), target.to(device)

        # Forward pass
        logits = model(obs)

        # Compute weighted BCE loss
        loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        with torch.no_grad():
            pred = torch.sigmoid(logits)
            batch_metrics = compute_metrics(pred, target)

        total_loss += loss.item()
        for k in all_metrics:
            all_metrics[k] += batch_metrics[k]

        # Update progress bar
        if batch_idx % 10 == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'iou': f'{batch_metrics["iou"]:.4f}',
                'f1': f'{batch_metrics["f1"]:.4f}'
            })

    n = len(dataloader)
    avg_loss = total_loss / n
    avg_metrics = {k: v / n for k, v in all_metrics.items()}

    return avg_loss, avg_metrics


def evaluate(model, dataloader, device, pos_weight):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_metrics = {'iou': 0, 'precision': 0, 'recall': 0, 'f1': 0}

    with torch.no_grad():
        for obs, target in tqdm(dataloader, desc="Evaluating"):
            obs, target = obs.to(device), target.to(device)

            logits = model(obs)
            loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)

            pred = torch.sigmoid(logits)
            batch_metrics = compute_metrics(pred, target)

            total_loss += loss.item()
            for k in all_metrics:
                all_metrics[k] += batch_metrics[k]

    n = len(dataloader)
    avg_loss = total_loss / n
    avg_metrics = {k: v / n for k, v in all_metrics.items()}

    return avg_loss, avg_metrics


def main():
    parser = argparse.ArgumentParser(description='U-Net Training V2 (Class Imbalance Fixed)')
    parser.add_argument('--repo-root', type=str, default='/home/chaseungjoon/code/WildfirePrediction')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max-envs', type=int, default=None, help='Limit number of environments')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--pos-weight', type=float, default=None, help='Positive class weight (auto-computed if None)')
    parser.add_argument('--max-file-size-mb', type=int, default=10, help='Max environment file size in MB')
    parser.add_argument('--wandb-project', type=str, default='wildfire-prediction', help='WandB project name')
    parser.add_argument('--wandb-run-name', type=str, default=None, help='WandB run name (auto-generated if not provided)')
    parser.add_argument('--no-wandb', action='store_true', help='Disable WandB logging')
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    env_dir = repo_root / 'tilling_data' / 'environments'

    # Load train/val splits
    train_split_path = env_dir / 'train_split.json'
    val_split_path = env_dir / 'val_split.json'

    with open(train_split_path) as f:
        train_env_ids = json.load(f)
    with open(val_split_path) as f:
        val_env_ids = json.load(f)

    if args.max_envs:
        train_env_ids = train_env_ids[:args.max_envs]
        val_env_ids = val_env_ids[:args.max_envs // 5]

    train_paths = [env_dir / f'{eid}.pkl' for eid in train_env_ids]
    val_paths = [env_dir / f'{eid}.pkl' for eid in val_env_ids]

    print(f"=" * 80)
    print(f"U-Net Training V2 (Class Imbalance Fixed)")
    print(f"=" * 80)
    print(f"Improvements:")
    print(f"  - Filtered out zero-burn samples")
    print(f"  - Weighted BCE loss with pos_weight")
    print(f"  - Per-sample IoU computation")
    print(f"  - Precision/Recall/F1 metrics")
    print(f"=" * 80)
    print(f"Training environments: {len(train_paths)}")
    print(f"Validation environments: {len(val_paths)}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"=" * 80)

    # Initialize WandB
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb_run_name = args.wandb_run_name or f"unet-v2-lr{args.lr}-bs{args.batch_size}-e{args.epochs}"
        wandb.init(
            project=args.wandb_project,
            name=wandb_run_name,
            config={
                "model": "UNet-V2",
                "learning_rate": args.lr,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "device": args.device,
                "train_envs": len(train_paths),
                "val_envs": len(val_paths),
                "num_workers": args.num_workers,
                "max_envs": args.max_envs,
                "filtered_zero_burns": True,
                "weighted_loss": True,
            }
        )
        print(f"WandB initialized: {wandb.run.name}")
    else:
        print("WandB logging disabled")

    # Create datasets
    print("\nCreating training dataset...")
    train_dataset = FireSpreadDatasetFiltered(train_paths, max_file_size_mb=args.max_file_size_mb)
    print("\nCreating validation dataset...")
    val_dataset = FireSpreadDatasetFiltered(val_paths, max_file_size_mb=args.max_file_size_mb)

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("\nERROR: No samples with burns found! Dataset is too filtered.")
        print("Try increasing --max-file-size-mb or checking your data.")
        return

    # Compute pos_weight if not provided
    if args.pos_weight is None:
        # Sample some batches to estimate class imbalance
        print("\nEstimating class imbalance...")
        temp_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
        total_pos = 0
        total_neg = 0
        for i, (_, target) in enumerate(temp_loader):
            if i >= 10:  # Sample first 10 batches
                break
            total_pos += target.sum().item()
            total_neg += (1 - target).sum().item()

        if total_pos > 0:
            imbalance_ratio = total_neg / total_pos
            args.pos_weight = min(imbalance_ratio, 1000.0)  # Cap at 1000
            print(f"Computed pos_weight: {args.pos_weight:.1f} (neg:pos ratio = {imbalance_ratio:.1f}:1)")
        else:
            args.pos_weight = 100.0
            print(f"Could not estimate imbalance, using default pos_weight: {args.pos_weight}")
    else:
        print(f"Using provided pos_weight: {args.pos_weight}")

    pos_weight = torch.tensor([args.pos_weight]).to(args.device)

    if use_wandb:
        wandb.config.update({"pos_weight": args.pos_weight})

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False,
        collate_fn=collate_fn
    )

    # Create model
    model = UNet(in_channels=14, out_channels=1).to(args.device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    if use_wandb:
        wandb.config.update({"model_parameters": total_params})

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    best_val_f1 = 0.0
    checkpoint_dir = repo_root / 'rl_training' / 'supervised' / 'checkpoints_unet_v2'
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 80)

        # Train
        train_loss, train_metrics = train_epoch(model, train_loader, optimizer, args.device, epoch, pos_weight)
        print(f"Train - Loss: {train_loss:.4f} | IoU: {train_metrics['iou']:.4f} | "
              f"P: {train_metrics['precision']:.4f} | R: {train_metrics['recall']:.4f} | "
              f"F1: {train_metrics['f1']:.4f}")

        # Validate
        val_loss, val_metrics = evaluate(model, val_loader, args.device, pos_weight)
        print(f"Val   - Loss: {val_loss:.4f} | IoU: {val_metrics['iou']:.4f} | "
              f"P: {val_metrics['precision']:.4f} | R: {val_metrics['recall']:.4f} | "
              f"F1: {val_metrics['f1']:.4f}")

        # Log to WandB
        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "train/iou": train_metrics['iou'],
                "train/precision": train_metrics['precision'],
                "train/recall": train_metrics['recall'],
                "train/f1": train_metrics['f1'],
                "val/loss": val_loss,
                "val/iou": val_metrics['iou'],
                "val/precision": val_metrics['precision'],
                "val/recall": val_metrics['recall'],
                "val/f1": val_metrics['f1'],
            })

        # Save best model (based on F1 score)
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_model_path = checkpoint_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'val_loss': val_loss
            }, best_model_path)
            print(f"✓ Saved best model (F1: {val_metrics['f1']:.4f})")

            # Log best model to WandB as artifact
            if use_wandb:
                artifact = wandb.Artifact(
                    name='best-model-v2',
                    type='model',
                    description=f'Best U-Net V2 at epoch {epoch} with F1 {val_metrics["f1"]:.4f}'
                )
                artifact.add_file(str(best_model_path))
                wandb.log_artifact(artifact)

        # Save checkpoint EVERY epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_metrics': val_metrics,
            'val_loss': val_loss
        }, checkpoint_dir / f'checkpoint_epoch{epoch}.pt')
        print(f"✓ Saved checkpoint for epoch {epoch}")

    print(f"\n" + "=" * 80)
    print(f"Training Complete!")
    print(f"Best Validation F1: {best_val_f1:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"=" * 80)

    # Finish WandB run
    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
