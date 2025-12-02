"""Quick test to verify V3 setup works correctly"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from sl_training.unet_16ch_v3.model import UNetMultiTimestep, multi_timestep_loss
from sl_training.unet_16ch_v3.dataset import get_dataloaders, compute_iou

print("="*70)
print("U-NET V3 SETUP TEST")
print("="*70)

# Test 1: Create model
print("\n[Test 1] Creating model...")
model = UNetMultiTimestep(n_channels=17, n_timesteps=3)
total_params = sum(p.numel() for p in model.parameters())
print(f"  Model created successfully")
print(f"  Total parameters: {total_params:,}")

# Test 2: Create dataloaders
print("\n[Test 2] Creating dataloaders...")
data_dir = '/home/chaseungjoon/code/WildfirePrediction/embedded_data/fire_episodes_16ch_normalized'
train_loader, val_loader = get_dataloaders(
    data_dir, batch_size=4, num_workers=2, min_mel=4, n_timesteps=3
)
print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches: {len(val_loader)}")

# Test 3: Load a batch
print("\n[Test 3] Loading a batch...")
inputs, targets_dilated, targets_strict = next(iter(train_loader))
print(f"  Input shape: {inputs.shape}")
print(f"  Target dilated shape: {targets_dilated.shape}")
print(f"  Target strict shape: {targets_strict.shape}")

# Test 4: Forward pass
print("\n[Test 4] Running forward pass...")
model.eval()
with torch.no_grad():
    outputs = model(inputs)
    print(f"  Output shape: {outputs.shape}")

# Test 5: Loss computation
print("\n[Test 5] Computing loss...")
loss = multi_timestep_loss(outputs, targets_dilated)
print(f"  Loss (dilated): {loss.item():.4f}")

# Test 6: IoU computation
print("\n[Test 6] Computing IoU...")
pred_probs = torch.sigmoid(outputs)
ious_relaxed = compute_iou(pred_probs, targets_dilated)
ious_strict = compute_iou(pred_probs, targets_strict)
print(f"  Relaxed IoU: t+1={ious_relaxed[0]:.4f}, t+2={ious_relaxed[1]:.4f}, t+3={ious_relaxed[2]:.4f}")
print(f"  Strict IoU:  t+1={ious_strict[0]:.4f}, t+2={ious_strict[1]:.4f}, t+3={ious_strict[2]:.4f}")

# Test 7: Verify dilation effect
print("\n[Test 7] Verifying dilation effect...")
import numpy as np
strict_positive = (targets_strict.numpy() > 0.5).sum()
dilated_positive = (targets_dilated.numpy() > 0.5).sum()
dilation_ratio = dilated_positive / max(strict_positive, 1)
print(f"  Strict positive cells: {strict_positive}")
print(f"  Dilated positive cells: {dilated_positive}")
print(f"  Dilation ratio: {dilation_ratio:.2f}x")
print(f"  Expected ratio: 3-5x (depending on cell clustering)")

# Test 8: Backward pass
print("\n[Test 8] Testing backward pass...")
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
outputs = model(inputs)
loss = multi_timestep_loss(outputs, targets_dilated)
loss.backward()
optimizer.step()
print(f"  Backward pass successful")

print("\n" + "="*70)
print("ALL TESTS PASSED!")
print("Ready for full training")
print("="*70)
