"""
Training script for U-Net wildfire prediction - 16 channels
Supervised learning with direct pixel-level supervision
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import argparse
import sys
from tqdm import tqdm
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sl_training.unet_16ch.model import UNet, combined_loss, dice_loss
from sl_training.unet_16ch.dataset import get_dataloaders, compute_iou


def train_epoch(model, train_loader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()

    total_loss = 0
    total_iou = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs = inputs.to(device)  # (B, 17, 30, 30)
        targets = targets.to(device)  # (B, 1, 30, 30)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)  # (B, 1, 30, 30) logits

        # Compute loss
        loss = combined_loss(outputs, targets, dice_weight=0.7, bce_weight=0.3)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Compute IoU
        with torch.no_grad():
            pred_probs = torch.sigmoid(outputs)
            batch_iou = compute_iou(pred_probs, targets)

        total_loss += loss.item()
        total_iou += batch_iou
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'iou': f'{batch_iou:.4f}'
        })

    avg_loss = total_loss / num_batches
    avg_iou = total_iou / num_batches

    return avg_loss, avg_iou


def validate(model, val_loader, device, epoch):
    """Validate on validation set"""
    model.eval()

    total_loss = 0
    total_iou = 0
    num_batches = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
        for inputs, targets in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            loss = combined_loss(outputs, targets, dice_weight=0.7, bce_weight=0.3)

            pred_probs = torch.sigmoid(outputs)
            batch_iou = compute_iou(pred_probs, targets)

            total_loss += loss.item()
            total_iou += batch_iou
            num_batches += 1

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'iou': f'{batch_iou:.4f}'
            })

    avg_loss = total_loss / num_batches
    avg_iou = total_iou / num_batches

    return avg_loss, avg_iou


def main():
    parser = argparse.ArgumentParser(description='U-Net Training - 16 Channels')
    parser.add_argument('--data-dir', type=str,
                       default='/home/chaseungjoon/code/WildfirePrediction/embedded_data/fire_episodes_16ch_full',
                       help='Directory with embedded episodes')
    parser.add_argument('--checkpoint-dir', type=str,
                       default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--min-mel', type=int, default=4,
                       help='Minimum MEL threshold')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of dataloader workers')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to train on')

    args = parser.parse_args()

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("U-NET WILDFIRE PREDICTION - 16 CHANNELS")
    print("="*70)
    print(f"Data directory: {args.data_dir}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"MEL threshold: {args.min_mel}")
    print(f"Device: {args.device}")
    print("="*70)

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = get_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        min_mel=args.min_mel
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Create model
    print("\nCreating model...")
    model = UNet(n_channels=17, n_classes=1, bilinear=True)
    model = model.to(args.device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    # Tensorboard writer
    writer = SummaryWriter(log_dir=checkpoint_dir / 'runs')

    # Training loop
    print("\n" + "="*70)
    print("Starting training...")
    print("="*70)

    best_val_iou = 0.0

    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_iou = train_epoch(model, train_loader, optimizer, args.device, epoch)

        # Validate
        val_loss, val_iou = validate(model, val_loader, args.device, epoch)

        # Learning rate scheduler
        scheduler.step(val_iou)

        # Log to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('IoU/train', train_iou, epoch)
        writer.add_scalar('IoU/val', val_iou, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        # Print epoch summary
        print(f"\nEpoch {epoch}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f} | Train IoU: {train_iou:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'train_iou': train_iou,
            'val_loss': val_loss,
            'val_iou': val_iou,
        }

        # Save latest
        torch.save(checkpoint, checkpoint_dir / 'latest.pt')

        # Save best
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(checkpoint, checkpoint_dir / 'best.pt')
            print(f"  ★ New best validation IoU: {val_iou:.4f}")

    print("\n" + "="*70)
    print("Training completed!")
    print(f"Best validation IoU: {best_val_iou:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print("="*70)

    writer.close()


if __name__ == '__main__':
    main()
