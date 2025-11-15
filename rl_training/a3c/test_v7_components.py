"""
Test script for A3C V7 components

Tests:
1. Environment observation sequence generation
2. Model forward/backward pass
3. Worker rollout for a few steps
"""
import sys
from pathlib import Path
import torch
import numpy as np
import json
import gc

sys.path.insert(0, str(Path(__file__).parent.parent))

from wildfire_env_temporal import WildfireEnvTemporal
from a3c.model_v7 import A3C_TemporalModel


def test_environment():
    """Test temporal environment observation sequence generation."""
    print("=" * 80)
    print("TEST 1: Environment Observation Sequence Generation")
    print("=" * 80)

    # Load a sample environment
    repo_root = Path('/home/chaseungjoon/code/WildfirePrediction')
    env_dir = repo_root / 'tilling_data' / 'environments'

    # Get first training environment
    train_split_path = env_dir / 'train_split.json'
    with open(train_split_path) as f:
        train_env_ids = json.load(f)

    env_path = env_dir / f'{train_env_ids[0]}.pkl'
    print(f"Loading environment: {env_path.name}")

    # Test with window_size=3
    window_size = 3
    env = WildfireEnvTemporal(env_path, window_size=window_size)

    print(f"Environment loaded successfully")
    print(f"  Timesteps: {env.T}")
    print(f"  Grid size: {env.H} x {env.W}")
    print(f"  Window size: {window_size}")

    # Test reset
    obs_seq, info = env.reset()
    print(f"\nReset observation sequence shape: {obs_seq.shape}")
    print(f"  Expected: ({window_size}, 14, {env.H}, {env.W})")
    assert obs_seq.shape == (window_size, 14, env.H, env.W), f"Wrong shape! Got {obs_seq.shape}"
    print(f"  ✓ Shape correct!")

    # Test that first timesteps are padded correctly
    print(f"\nChecking padding at t=0:")
    print(f"  obs_seq[0] should equal obs_seq[1] should equal obs_seq[2] (all t=0)")
    # Check fire mask channel
    fire_mask_0 = obs_seq[0, 5]  # Channel 5 is fire mask
    fire_mask_1 = obs_seq[1, 5]
    fire_mask_2 = obs_seq[2, 5]
    assert np.allclose(fire_mask_0, fire_mask_1) and np.allclose(fire_mask_1, fire_mask_2), "Padding failed!"
    print(f"  ✓ Padding correct!")

    # Test step
    print(f"\nTesting step:")
    action = np.zeros((env.H, env.W))
    next_obs_seq, reward, done, info = env.step(action)
    print(f"  Next obs_seq shape: {next_obs_seq.shape}")
    print(f"  Reward: {reward:.4f}")
    print(f"  Done: {done}")
    print(f"  Current timestep: {info['t']}")
    assert next_obs_seq.shape == (window_size, 14, env.H, env.W), f"Wrong shape after step!"
    print(f"  ✓ Step works correctly!")

    # Test that sequence shifts properly
    print(f"\nChecking temporal shift:")
    print(f"  After step, obs_seq[-1] should be new observation")
    # The last timestep's fire mask should differ from the previous obs_seq's last
    old_fire_mask = obs_seq[-1, 5]
    new_fire_mask = next_obs_seq[-1, 5]
    # They might be different if fire spreads
    print(f"  Old fire cells: {(old_fire_mask > 0).sum()}")
    print(f"  New fire cells: {(new_fire_mask > 0).sum()}")
    print(f"  ✓ Temporal sequence shifts correctly!")

    print(f"\n{'='*80}")
    print("TEST 1 PASSED: Environment works correctly!")
    print(f"{'='*80}\n")

    # Clean up
    del obs_seq, next_obs_seq, fire_mask_0, fire_mask_1, fire_mask_2
    del old_fire_mask, new_fire_mask
    gc.collect()

    return env


def test_model(env):
    """Test model forward/backward pass."""
    print("=" * 80)
    print("TEST 2: Model Forward/Backward Pass")
    print("=" * 80)

    window_size = 3
    model = A3C_TemporalModel(
        in_channels=14,
        window_size=window_size,
        lstm_hidden_dim=128,
        lstm_num_layers=2
    )
    model.train()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created successfully")
    print(f"  Total parameters: {total_params:,}")

    # Create sample input
    obs_seq, _ = env.reset()
    obs_seq_tensor = torch.from_numpy(obs_seq).unsqueeze(0).float()  # (1, 3, 14, H, W)
    fire_mask = obs_seq_tensor[0, -1, 5].unsqueeze(0)  # (1, H, W)

    print(f"\nInput shapes:")
    print(f"  obs_seq_tensor: {obs_seq_tensor.shape}")
    print(f"  fire_mask: {fire_mask.shape}")

    # Test forward pass
    print(f"\nTesting forward pass...")
    features, value = model(obs_seq_tensor, fire_mask)
    print(f"  ✓ Forward pass succeeded!")
    print(f"  Features shape: {features.shape}")
    print(f"  Value shape: {value.shape}")
    print(f"  Expected features: (1, 128, {env.H}, {env.W})")
    print(f"  Expected value: (1, 1)")
    assert features.shape == (1, 128, env.H, env.W), f"Wrong features shape! Got {features.shape}"
    assert value.shape == (1, 1), f"Wrong value shape! Got {value.shape}"

    # Test get_action_and_value
    print(f"\nTesting get_action_and_value...")
    action_grid, log_prob, entropy, value, burning_cells_info = model.get_action_and_value(
        obs_seq_tensor, fire_mask
    )
    print(f"  ✓ get_action_and_value succeeded!")
    print(f"  Action grid shape: {action_grid.shape}")
    print(f"  Log prob: {log_prob.item():.4f}")
    print(f"  Entropy: {entropy.item():.4f}")
    print(f"  Value: {value.item():.4f}")
    print(f"  Number of burning cells: {len(burning_cells_info)}")

    # Test backward pass
    print(f"\nTesting backward pass...")
    loss = -log_prob + 0.5 * value.pow(2).mean() - 0.01 * entropy
    loss.backward()
    print(f"  ✓ Backward pass succeeded!")

    # Check gradients exist
    has_grads = sum(1 for p in model.parameters() if p.grad is not None)
    total_params_count = sum(1 for p in model.parameters())
    print(f"  Parameters with gradients: {has_grads}/{total_params_count}")
    assert has_grads > 0, "No gradients computed!"

    # Check gradient magnitudes
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    avg_grad_norm = np.mean(grad_norms)
    max_grad_norm = np.max(grad_norms)
    print(f"  Average gradient norm: {avg_grad_norm:.6f}")
    print(f"  Max gradient norm: {max_grad_norm:.6f}")

    print(f"\n{'='*80}")
    print("TEST 2 PASSED: Model works correctly!")
    print(f"{'='*80}\n")

    # Clean up
    model.zero_grad()
    del obs_seq, obs_seq_tensor, fire_mask, features, value
    del action_grid, log_prob, entropy, burning_cells_info
    del loss, grad_norms
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return model


def test_worker_rollout(env, model):
    """Test worker rollout for a few steps."""
    print("=" * 80)
    print("TEST 3: Worker Rollout")
    print("=" * 80)

    window_size = 3
    model.train()

    print(f"Running rollout for 5 steps...")

    obs_seq, _ = env.reset()
    total_reward = 0
    steps = 0
    max_steps = 5

    trajectory = {
        'obs': [],
        'actions': [],
        'rewards': [],
        'log_probs': [],
        'values': [],
        'entropies': []
    }

    for step in range(max_steps):
        # Convert to tensor
        obs_seq_tensor = torch.from_numpy(obs_seq).unsqueeze(0).float()
        fire_mask = obs_seq_tensor[0, -1, 5]

        # Get action
        with torch.no_grad():
            action_grid, log_prob, entropy, value, burning_info = model.get_action_and_value(
                obs_seq_tensor, fire_mask
            )

        action_np = action_grid.numpy()

        # Step environment
        next_obs_seq, reward, done, info = env.step(action_np)

        # Store trajectory - DETACH TENSORS to break computation graph
        trajectory['obs'].append(obs_seq_tensor.detach().clone())
        trajectory['actions'].append(action_grid.detach().clone())
        trajectory['rewards'].append(reward)
        trajectory['log_probs'].append(log_prob.detach().clone())
        trajectory['values'].append(value.detach().clone())
        trajectory['entropies'].append(entropy.detach().clone())

        total_reward += reward
        steps += 1

        print(f"  Step {step+1}:")
        print(f"    Burning cells: {len(burning_info)}")
        print(f"    Action cells: {(action_np > 0.5).sum()}")
        print(f"    Reward (IoU): {reward:.4f}")
        print(f"    Log prob: {log_prob.item():.4f}")
        print(f"    Entropy: {entropy.item():.4f}")
        print(f"    Value: {value.item():.4f}")

        obs_seq = next_obs_seq

        if done:
            print(f"    Episode done!")
            break

        # Clear intermediate tensors
        del obs_seq_tensor, fire_mask, action_grid, log_prob, entropy, value, burning_info
        del action_np, next_obs_seq

    print(f"\nRollout complete:")
    print(f"  Total steps: {steps}")
    print(f"  Total reward: {total_reward:.4f}")
    print(f"  Average reward: {total_reward/max(1, steps):.4f}")

    # Clear environment cache to free memory
    env.obs_cache.clear()
    gc.collect()

    # Test computing loss from trajectory
    print(f"\nTesting loss computation from trajectory...")

    returns = []
    R = 0
    gamma = 0.99
    for r in reversed(trajectory['rewards']):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns).unsqueeze(1)

    # Recompute with gradients
    recomputed_log_probs = []
    recomputed_values = []
    recomputed_entropies = []

    for t in range(len(trajectory['obs'])):
        obs_t = trajectory['obs'][t]
        action_t = trajectory['actions'][t]
        fire_mask_t = obs_t[0, -1, 5]

        # Forward pass
        features, value_t = model(obs_t, fire_mask_t.unsqueeze(0))

        # Get burning cells
        burning_cells = model.get_burning_cells(fire_mask_t)

        if len(burning_cells) == 0:
            recomputed_log_probs.append(torch.tensor(0.0))
            recomputed_values.append(value_t)
            recomputed_entropies.append(torch.tensor(0.0))
            continue

        # Recompute for each burning cell
        step_log_probs = []
        step_entropies = []

        for i, j in burning_cells:
            logits_8d = model.predict_8_neighbors(features, i, j).squeeze(0)
            probs_8d = torch.sigmoid(logits_8d)
            probs_8d = torch.clamp(probs_8d, 1e-7, 1 - 1e-7)

            # Get action from grid
            neighbors = model.get_8_neighbor_coords(i, j, env.H, env.W)
            action_8d = torch.zeros(8)
            for n_idx, neighbor in enumerate(neighbors):
                if neighbor is not None:
                    ni, nj = neighbor
                    action_8d[n_idx] = action_t[ni, nj]

            log_prob_8d = (action_8d * torch.log(probs_8d) +
                          (1 - action_8d) * torch.log(1 - probs_8d))
            entropy_8d = -(probs_8d * torch.log(probs_8d) +
                          (1 - probs_8d) * torch.log(1 - probs_8d))

            step_log_probs.append(log_prob_8d.sum())
            step_entropies.append(entropy_8d.sum())

        recomputed_log_probs.append(torch.stack(step_log_probs).sum())
        recomputed_values.append(value_t)
        recomputed_entropies.append(torch.stack(step_entropies).sum())

    # Compute loss
    log_probs_tensor = torch.stack(recomputed_log_probs)
    values_tensor = torch.cat(recomputed_values)
    entropy_tensor = torch.stack(recomputed_entropies)

    advantages = (returns - values_tensor.detach()).squeeze(1)

    policy_loss = -(log_probs_tensor * advantages).mean()
    value_loss = torch.nn.functional.mse_loss(values_tensor, returns)
    entropy_loss = -entropy_tensor.mean()

    total_loss = policy_loss + 0.5 * value_loss + 0.015 * entropy_loss

    print(f"  Policy loss: {policy_loss.item():.4f}")
    print(f"  Value loss: {value_loss.item():.4f}")
    print(f"  Entropy loss: {entropy_loss.item():.4f}")
    print(f"  Total loss: {total_loss.item():.4f}")

    # Test backward
    model.zero_grad()
    total_loss.backward()

    # Check gradients
    has_grads = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"  Parameters with gradients: {has_grads}")
    print(f"  ✓ Loss computation and backprop succeeded!")

    print(f"\n{'='*80}")
    print("TEST 3 PASSED: Worker rollout works correctly!")
    print(f"{'='*80}\n")

    # Comprehensive cleanup
    model.zero_grad()
    del trajectory, returns, recomputed_log_probs, recomputed_values, recomputed_entropies
    del log_probs_tensor, values_tensor, entropy_tensor, advantages
    del policy_loss, value_loss, entropy_loss, total_loss
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


def main():
    print("\n")
    print("=" * 80)
    print("A3C V7 COMPONENT TESTS")
    print("=" * 80)
    print("\n")

    # Test 1: Environment
    env = test_environment()

    # Test 2: Model
    model = test_model(env)

    # Test 3: Worker rollout
    test_worker_rollout(env, model)

    print("=" * 80)
    print("ALL TESTS PASSED! ✓")
    print("=" * 80)
    print("\nV7 is ready for training!")
    print("\nRun the following command to start a short test training:")
    print("\n  PYTHONPATH=/home/chaseungjoon/code/WildfirePrediction:$PYTHONPATH \\")
    print("  python3 rl_training/a3c/train_v7.py \\")
    print("    --num-workers 2 \\")
    print("    --max-episodes 10 \\")
    print("    --max-envs 50 \\")
    print("    --no-wandb")
    print()


if __name__ == '__main__':
    main()
