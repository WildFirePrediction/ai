"""
Test script for A3C Model V3 (Deeper Encoder)
Verifies the model works correctly before training.
"""
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from a3c.model_v3 import A3C_PerCellModel_Deep

def test_model():
    print("=" * 80)
    print("Testing A3C Model V3 (Deeper Encoder)")
    print("=" * 80)

    # Create model
    model = A3C_PerCellModel_Deep(in_channels=14, use_groupnorm=True)
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    # Compare to V2
    from a3c.model_v2 import A3C_PerCellModel
    model_v2 = A3C_PerCellModel(in_channels=14)
    v2_params = sum(p.numel() for p in model_v2.parameters())
    print(f"  V2 had: {v2_params:,}")
    print(f"  Increase: {(total_params / v2_params):.2f}x")

    # Test forward pass
    print("\n" + "=" * 80)
    print("Test 1: Forward Pass")
    print("=" * 80)

    # Create dummy input
    H, W = 50, 50
    x = torch.randn(1, 14, H, W)
    fire_mask = torch.zeros(1, H, W)

    # Add some burning cells
    fire_mask[0, 10:15, 10:15] = 1.0
    fire_mask[0, 30:35, 30:35] = 1.0

    num_burning = (fire_mask > 0.5).sum().item()
    print(f"Input shape: {x.shape}")
    print(f"Fire mask shape: {fire_mask.shape}")
    print(f"Number of burning cells: {num_burning}")

    try:
        with torch.no_grad():
            features, value = model(x, fire_mask)
        print(f"✓ Forward pass successful")
        print(f"  Features shape: {features.shape}")
        print(f"  Value shape: {value.shape}")
        print(f"  Value: {value.item():.4f}")
        assert features.shape == (1, 512, H, W), f"Expected (1, 512, {H}, {W}), got {features.shape}"
        assert value.shape == (1, 1), f"Expected (1, 1), got {value.shape}"
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False

    # Test get_action_and_value
    print("\n" + "=" * 80)
    print("Test 2: Get Action and Value")
    print("=" * 80)

    try:
        with torch.no_grad():
            action_grid, log_prob, entropy, value, info = model.get_action_and_value(x, fire_mask)

        print(f"✓ get_action_and_value successful")
        print(f"  Action grid shape: {action_grid.shape}")
        print(f"  Predicted burns: {(action_grid > 0.5).sum().item()}")
        print(f"  Log prob: {log_prob.item():.4f}")
        print(f"  Entropy: {entropy.item():.4f}")
        print(f"  Value: {value.item():.4f}")
        print(f"  Burning cells processed: {len(info)}")

        assert action_grid.shape == (H, W), f"Expected ({H}, {W}), got {action_grid.shape}"
        assert len(info) == num_burning, f"Expected {num_burning} cells, got {len(info)}"
    except Exception as e:
        print(f"✗ get_action_and_value failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test edge cases
    print("\n" + "=" * 80)
    print("Test 3: Edge Cases")
    print("=" * 80)

    # Test with no burning cells
    fire_mask_empty = torch.zeros(1, H, W)
    try:
        with torch.no_grad():
            action_grid, log_prob, entropy, value, info = model.get_action_and_value(x, fire_mask_empty)
        print(f"✓ Empty fire mask handled correctly")
        print(f"  Predicted burns: {(action_grid > 0.5).sum().item()}")
        assert (action_grid == 0).all(), "Action grid should be all zeros for empty fire mask"
    except Exception as e:
        print(f"✗ Empty fire mask test failed: {e}")
        return False

    # Test with fire at boundaries
    fire_mask_boundary = torch.zeros(1, H, W)
    fire_mask_boundary[0, 0, 0] = 1.0  # Top-left corner
    fire_mask_boundary[0, H-1, W-1] = 1.0  # Bottom-right corner
    try:
        with torch.no_grad():
            action_grid, log_prob, entropy, value, info = model.get_action_and_value(x, fire_mask_boundary)
        print(f"✓ Boundary fire mask handled correctly")
        print(f"  Burning cells at boundaries: {len(info)}")
    except Exception as e:
        print(f"✗ Boundary fire mask test failed: {e}")
        return False

    # Test backward pass (gradient flow)
    print("\n" + "=" * 80)
    print("Test 4: Backward Pass (Gradient Flow)")
    print("=" * 80)

    model.train()
    x_grad = torch.randn(1, 14, H, W, requires_grad=True)
    fire_mask_grad = fire_mask.clone()

    try:
        features, value = model(x_grad, fire_mask_grad)
        loss = value.sum()  # Dummy loss
        loss.backward()

        print(f"✓ Backward pass successful")
        print(f"  Input gradient exists: {x_grad.grad is not None}")
        if x_grad.grad is not None:
            print(f"  Input gradient norm: {x_grad.grad.norm().item():.6f}")

        # Check if encoder and value head parameters have gradients
        # (policy head won't have gradients since we only tested value, this is expected)
        encoder_params_with_grad = sum(1 for p in model.encoder.parameters() if p.grad is not None)
        encoder_total_params = sum(1 for p in model.encoder.parameters())
        value_params_with_grad = sum(1 for p in model.value_head.parameters() if p.grad is not None)
        value_total_params = sum(1 for p in model.value_head.parameters())

        print(f"  Encoder params with gradients: {encoder_params_with_grad}/{encoder_total_params}")
        print(f"  Value head params with gradients: {value_params_with_grad}/{value_total_params}")

        assert encoder_params_with_grad == encoder_total_params, "Not all encoder parameters received gradients"
        assert value_params_with_grad == value_total_params, "Not all value head parameters received gradients"
        print(f"  Note: Policy head not tested (only used during action sampling in training)")
    except Exception as e:
        print(f"✗ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)
    print("\nModel V3 is ready for training.")
    print(f"Architecture: 5-layer encoder (14→32→64→128→256→512)")
    print(f"Parameters: {total_params:,} ({(total_params / v2_params):.2f}x increase from V2)")
    print(f"GroupNorm: Enabled for training stability")
    print(f"Receptive field: ~11x11 pixels (vs ~7x7 in V2)")

    return True


if __name__ == '__main__':
    success = test_model()
    sys.exit(0 if success else 1)
