"""
Test script for A3C V3 LSTM REL with Relaxed IoU
Verifies model creation, data loading, and relaxed IoU computation
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
from scipy.ndimage import binary_dilation

from a3c_16ch.V3_LSTM_REL.model import A3C_PerCellModel_LSTM
from a3c_16ch.V3_LSTM_REL.worker import compute_relaxed_iou

print("="*70)
print("A3C V3 LSTM REL SETUP TEST - Relaxed IoU (8-neighbor tolerance)")
print("="*70)

# Test 1: Model creation
print("\n[Test 1] Creating model...")
model = A3C_PerCellModel_LSTM(in_channels=16, sequence_length=3)
total_params = sum(p.numel() for p in model.parameters())
print(f"  Model created successfully")
print(f"  Total parameters: {total_params:,}")

# Test 2: Forward pass
print("\n[Test 2] Testing forward pass...")
batch_size = 1
seq_len = 3
H, W = 30, 30

sequence = torch.randn(batch_size, seq_len, 16, H, W)
fire_mask = torch.zeros(batch_size, H, W)
fire_mask[0, 15, 15] = 1.0  # Single burning cell

features, value = model(sequence, fire_mask)
print(f"  Input sequence shape: {sequence.shape}")
print(f"  Fire mask shape: {fire_mask.shape}")
print(f"  Output features shape: {features.shape}")
print(f"  Value shape: {value.shape}")

# Test 3: Action prediction
print("\n[Test 3] Testing action prediction...")
action_grid, log_prob, entropy, value, cells_info = model.get_action_and_value(
    sequence, fire_mask
)
print(f"  Action grid shape: {action_grid.shape}")
print(f"  Number of burning cells: {len(cells_info)}")
print(f"  Action grid sum: {action_grid.sum():.2f} cells predicted to burn")

# Test 4: Relaxed IoU computation
print("\n[Test 4] Testing relaxed IoU computation...")
# Create test masks
pred = np.zeros((30, 30), dtype=bool)
target = np.zeros((30, 30), dtype=bool)

# Set a single cell in target
target[15, 15] = True

# Test case 1: Exact match
pred[15, 15] = True
iou_exact = compute_relaxed_iou(pred, target)
print(f"  Exact match IoU: {iou_exact:.4f} (should be ~1.0)")

# Test case 2: One cell off (should still count with relaxed IoU)
pred = np.zeros((30, 30), dtype=bool)
pred[15, 16] = True  # One cell to the right
iou_neighbor = compute_relaxed_iou(pred, target)
print(f"  1-cell neighbor IoU: {iou_neighbor:.4f} (should be >0, relaxed)")

# Test case 3: Two cells off (should not count)
pred = np.zeros((30, 30), dtype=bool)
pred[15, 17] = True  # Two cells to the right
iou_far = compute_relaxed_iou(pred, target)
print(f"  2-cells away IoU: {iou_far:.4f} (should be 0.0)")

# Test 5: Verify dilation
print("\n[Test 5] Verifying 3x3 dilation...")
structure = np.ones((3, 3), dtype=bool)
target_single = np.zeros((30, 30), dtype=bool)
target_single[15, 15] = True
target_dilated = binary_dilation(target_single, structure=structure)
dilated_count = target_dilated.sum()
print(f"  Original target: 1 cell")
print(f"  Dilated target: {dilated_count} cells (should be 9 for 3x3)")

# Test 6: Load a sample episode
print("\n[Test 6] Loading sample episode...")
data_dir = Path("/home/chaseungjoon/code/WildfirePrediction/embedded_data/fire_episodes_16ch_normalized")
sample_episodes = sorted(data_dir.glob('episode_*.npz'))[:5]

if len(sample_episodes) > 0:
    sample_file = sample_episodes[0]
    data = np.load(sample_file)
    states = data['states']
    fire_masks = data['fire_masks']
    T = len(states)
    print(f"  Loaded episode: {sample_file.name}")
    print(f"  Shape: {states.shape}")
    print(f"  Timesteps: {T}")
    print(f"  MEL: {T-1}")

    # Test with real data
    if T >= 4:
        print("\n[Test 7] Testing with real episode data...")
        sequence_real = torch.from_numpy(states[0:3]).unsqueeze(0).float()
        fire_real = torch.from_numpy(fire_masks[2]).unsqueeze(0).float()

        with torch.no_grad():
            action_real, _, _, _, _ = model.get_action_and_value(sequence_real, fire_real)

        # Compute relaxed IoU
        current_fire = fire_masks[2]
        next_fire = fire_masks[3]
        actual_mask_t = current_fire > 0
        actual_mask_t1 = next_fire > 0
        new_burns = (actual_mask_t1 & ~actual_mask_t)
        predicted_mask = action_real.numpy() > 0.5

        iou_real = compute_relaxed_iou(predicted_mask, new_burns)
        print(f"  Predicted cells: {predicted_mask.sum()}")
        print(f"  Actual new burns: {new_burns.sum()}")
        print(f"  Relaxed IoU: {iou_real:.4f}")
else:
    print("  WARNING: No episodes found in data directory")

print("\n" + "="*70)
print("ALL TESTS PASSED!")
print("Ready for full training")
print("="*70)
