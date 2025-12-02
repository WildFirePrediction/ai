"""
Training script for U-Net wildfire prediction - V2 Multi-Timestep + Focal Loss
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

from sl_training.unet_16ch_v2.model import UNetMultiTimestep, multi_timestep_loss
from sl_training.unet_16ch_v2.dataset import get_dataloaders, compute_iou


def train_epoch(model, train_loader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()

    total_loss = 0
    total_iou_t1 = 0
    total_iou_t2 = 0
    total_iou_t3 = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs = inputs.to(device)  # (B, 17, 30, 30)
        targets = targets.to(device)  # (B, 3, 30, 30)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)  # (B, 3, 30, 30) logits

        # Compute loss
        loss = multi_timestep_loss(outputs, targets, focal_weight=0.7, dice_weight=0.3)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Compute IoU for each timestep
        with torch.no_grad():
            pred_probs = torch.sigmoid(outputs)
            ious = compute_iou(pred_probs, targets)

        total_loss += loss.item()
        total_iou_t1 += ious[0]
        total_iou_t2 += ious[1]
        total_iou_t3 += ious[2]
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            't+1': f'{ious[0]:.4f}',
            't+2': f'{ious[1]:.4f}',
            't+3': f'{ious[2]:.4f}'
        })

    avg_loss = total_loss / num_batches
    avg_iou_t1 = total_iou_t1 / num_batches
    avg_iou_t2 = total_iou_t2 / num_batches
    avg_iou_t3 = total_iou_t3 / num_batches

    return avg_loss, avg_iou_t1, avg_iou_t2, avg_iou_t3


def validate(model, val_loader, device, epoch):
    """Validate on validation set"""
    model.eval()

    total_loss = 0
    total_iou_t1 = 0
    total_iou_t2 = 0
    total_iou_t3 = 0
    num_batches = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
        for inputs, targets in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            loss = multi_timestep_loss(outputs, targets, focal_weight=0.7, dice_weight=0.3)

            pred_probs = torch.sigmoid(outputs)
            ious = compute_iou(pred_probs, targets)

            total_loss += loss.item()
            total_iou_t1 += ious[0]
            total_iou_t2 += ious[1]
            total_iou_t3 += ious[2]
            num_batches += 1

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                't+1': f'{ious[0]:.4f}',
                't+2': f'{ious[1]:.4f}',
                't+3': f'{ious[2]:.4f}'
            })

    avg_loss = total_loss / num_batches
    avg_iou_t1 = total_iou_t1 / num_batches
    avg_iou_t2 = total_iou_t2 / num_batches
    avg_iou_t3 = total_iou_t3 / num_batches

    return avg_loss, avg_iou_t1, avg_iou_t2, avg_iou_t3


def main():
    parser = argparse.ArgumentParser(description='U-Net V2 Training - Multi-Timestep + Focal Loss')
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
                       help='Initial learning rate')
    parser.add_argument('--min-mel', type=int, default=4,
                       help='Minimum MEL threshold')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of dataloader workers')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to train on')
    parser.add_argument('--n-timesteps', type=int, default=3,
                       help='Number of future timesteps to predict')

    args = parser.parse_args()

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("U-NET V2 WILDFIRE PREDICTION - MULTI-TIMESTEP + FOCAL LOSS")
    print("="*70)
    print(f"Data directory: {args.data_dir}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"MEL threshold: {args.min_mel}")
    print(f"Device: {args.device}")
    print(f"Predicting {args.n_timesteps} timesteps: t+1, t+2, t+3")
    print("="*70)

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = get_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        min_mel=args.min_mel,
        n_timesteps=args.n_timesteps
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Create model
    print("\nCreating model...")
    model = UNetMultiTimestep(n_channels=17, n_timesteps=args.n_timesteps, bilinear=True)
    model = model.to(args.device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Cosine annealing learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # Tensorboard writer
    writer = SummaryWriter(log_dir=checkpoint_dir / 'runs')

    # Training loop
    print("\n" + "="*70)
    print("Starting training...")
    print("="*70)

    best_val_iou_t1 = 0.0

    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_iou_t1, train_iou_t2, train_iou_t3 = train_epoch(
            model, train_loader, optimizer, args.device, epoch
        )

        # Validate
        val_loss, val_iou_t1, val_iou_t2, val_iou_t3 = validate(
            model, val_loader, args.device, epoch
        )

        # Learning rate scheduler
        scheduler.step()

        # Log to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('IoU_t1/train', train_iou_t1, epoch)
        writer.add_scalar('IoU_t1/val', val_iou_t1, epoch)
        writer.add_scalar('IoU_t2/train', train_iou_t2, epoch)
        writer.add_scalar('IoU_t2/val', val_iou_t2, epoch)
        writer.add_scalar('IoU_t3/train', train_iou_t3, epoch)
        writer.add_scalar('IoU_t3/val', val_iou_t3, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        # Print epoch summary
        print(f"\nEpoch {epoch}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f} | IoU t+1: {train_iou_t1:.4f} t+2: {train_iou_t2:.4f} t+3: {train_iou_t3:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | IoU t+1: {val_iou_t1:.4f} t+2: {val_iou_t2:.4f} t+3: {val_iou_t3:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'train_iou_t1': train_iou_t1,
            'train_iou_t2': train_iou_t2,
            'train_iou_t3': train_iou_t3,
            'val_loss': val_loss,
            'val_iou_t1': val_iou_t1,
            'val_iou_t2': val_iou_t2,
            'val_iou_t3': val_iou_t3,
        }

        # Save latest
        torch.save(checkpoint, checkpoint_dir / 'latest.pt')

        # Save best (based on t+1 IoU, most important)
        if val_iou_t1 > best_val_iou_t1:
            best_val_iou_t1 = val_iou_t1
            torch.save(checkpoint, checkpoint_dir / 'best.pt')
            print(f"  ★ New best validation IoU (t+1): {val_iou_t1:.4f}")

    print("\n" + "="*70)
    print("Training completed!")
    print(f"Best validation IoU (t+1): {best_val_iou_t1:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print("="*70)

    writer.close()


if __name__ == '__main__':
    main()
