"""
U-Net Training for Spatial Fire Spread Prediction

Supervised learning with U-Net architecture to predict which cells will burn next.
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

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from wildfire_env_spatial import WildfireEnvSpatial
from supervised.unet_model import UNet


class FireSpreadDataset(Dataset):
    """Dataset for fire spread prediction."""

    def __init__(self, env_paths):
        self.env_paths = env_paths
        # Pre-compute all (env_idx, timestep) pairs
        self.samples = []

        print(f"Loading {len(env_paths)} environments...")
        for env_idx, env_path in enumerate(tqdm(env_paths, desc="Scanning envs")):
            env = WildfireEnvSpatial(env_path)
            # Can predict for timesteps 0 to T-2 (predict t+1)
            for t in range(env.T - 1):
                self.samples.append((env_idx, t))

        print(f"Total samples: {len(self.samples)}")

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


def compute_iou(pred, target):
    """Compute IoU metric."""
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = (pred + target).clamp(0, 1).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return (intersection / union).item()


def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_iou = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (obs, target) in enumerate(pbar):
        obs, target = obs.to(device), target.to(device)

        # Forward pass
        logits = model(obs)

        # Compute loss (BCE with logits)
        loss = F.binary_cross_entropy_with_logits(logits, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        with torch.no_grad():
            pred = torch.sigmoid(logits)
            iou = compute_iou(pred, target)

        total_loss += loss.item()
        total_iou += iou

        # Update progress bar
        if batch_idx % 10 == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'iou': f'{iou:.4f}'
            })

    avg_loss = total_loss / len(dataloader)
    avg_iou = total_iou / len(dataloader)

    return avg_loss, avg_iou


def evaluate(model, dataloader, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    total_iou = 0

    with torch.no_grad():
        for obs, target in tqdm(dataloader, desc="Evaluating"):
            obs, target = obs.to(device), target.to(device)

            logits = model(obs)
            loss = F.binary_cross_entropy_with_logits(logits, target)

            pred = torch.sigmoid(logits)
            iou = compute_iou(pred, target)

            total_loss += loss.item()
            total_iou += iou

    avg_loss = total_loss / len(dataloader)
    avg_iou = total_iou / len(dataloader)

    return avg_loss, avg_iou


def main():
    parser = argparse.ArgumentParser(description='U-Net Training for Fire Spread')
    parser.add_argument('--repo-root', type=str, default='/home/chaseungjoon/code/WildfirePrediction')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max-envs', type=int, default=None, help='Limit number of environments')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers')
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
    print(f"U-Net Training for Fire Spread Prediction")
    print(f"=" * 80)
    print(f"Training environments: {len(train_paths)}")
    print(f"Validation environments: {len(val_paths)}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"=" * 80)

    # Create datasets
    train_dataset = FireSpreadDataset(train_paths)
    val_dataset = FireSpreadDataset(val_paths)

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

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    best_val_iou = 0.0
    checkpoint_dir = repo_root / 'rl_training' / 'supervised' / 'checkpoints_unet'
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 80)

        # Train
        train_loss, train_iou = train_epoch(model, train_loader, optimizer, args.device, epoch)
        print(f"Train - Loss: {train_loss:.4f} | IoU: {train_iou:.4f}")

        # Validate
        val_loss, val_iou = evaluate(model, val_loader, args.device)
        print(f"Val   - Loss: {val_loss:.4f} | IoU: {val_iou:.4f}")

        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
                'val_loss': val_loss
            }, checkpoint_dir / 'best_model.pt')
            print(f"✓ Saved best model (IoU: {val_iou:.4f})")

        # Save checkpoint EVERY epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_iou': val_iou,
            'val_loss': val_loss
        }, checkpoint_dir / f'checkpoint_epoch{epoch}.pt')
        print(f"✓ Saved checkpoint for epoch {epoch}")

    print(f"\n" + "=" * 80)
    print(f"Training Complete!")
    print(f"Best Validation IoU: {best_val_iou:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"=" * 80)


if __name__ == '__main__':
    main()
