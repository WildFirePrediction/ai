"""
A3C Worker Process

Each worker runs on CPU, interacts with environment, and computes gradients.
Gradients are sent to shared model for asynchronous updates.
"""
import os
# Limit threading to avoid contention between workers
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys

# Set PyTorch to use single thread per worker
torch.set_num_threads(1)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from wildfire_env_spatial import WildfireEnvSpatial
from a3c.model import A3C_SpatialFireModel


def compute_iou(pred, target):
    """Compute IoU between predicted and target burn masks."""
    pred_flat = pred.flatten()
    target_flat = target.flatten()

    intersection = (pred_flat * target_flat).sum()
    union = np.clip(pred_flat + target_flat, 0, 1).sum()

    return float(intersection / (union + 1e-8))


def worker_process(worker_id, shared_model, optimizer, env_paths, config, global_episode_counter, global_best_iou, lock):
    """
    Worker process for A3C training.

    Args:
        worker_id: Worker ID
        shared_model: Shared global model (on CPU)
        optimizer: Shared optimizer
        env_paths: List of environment paths to sample from
        config: Training configuration dict
        global_episode_counter: Shared counter for episodes
        global_best_iou: Shared best IoU value
        lock: Lock for synchronization
    """
    torch.manual_seed(config['seed'] + worker_id)
    np.random.seed(config['seed'] + worker_id)

    # Create local model (worker's copy)
    local_model = A3C_SpatialFireModel(in_channels=14)
    local_model.train()

    episode_count = 0
    max_episodes = config.get('max_episodes', 1000)

    print(f"[Worker {worker_id}] Starting training loop", flush=True)
    sys.stdout.flush()

    while True:
        # Check if training should stop
        with lock:
            if global_episode_counter.value >= max_episodes:
                break

        # Sync local model with shared model
        local_model.load_state_dict(shared_model.state_dict())

        # Sample random environment
        env_path = np.random.choice(env_paths)
        print(f"[Worker {worker_id}] Loading environment: {env_path.name}", flush=True)
        env = WildfireEnvSpatial(env_path)
        print(f"[Worker {worker_id}] Environment loaded, starting episode", flush=True)

        # Collect trajectory
        states, actions, rewards, log_probs, values, entropies = [], [], [], [], [], []

        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_iou = 0
        steps = 0

        while not done and steps < config.get('max_steps_per_episode', 50):
            # Convert obs to tensor
            state_tensor = torch.from_numpy(obs).unsqueeze(0).float()  # (1, 14, H, W)

            # Get action from local model
            with torch.no_grad():
                action, log_prob, entropy, value = local_model.get_action_and_value(state_tensor)

            # Convert action to numpy
            action_np = action.squeeze(0).numpy()  # (H, W)

            # Step environment
            next_obs, reward, done, info = env.step(action_np)

            # Compute IoU for this step
            if 't' in info and info['t'] < env.T - 1:
                target_mask = (env.fire_masks[info['t']+1] > 0).astype(np.float32)
                step_iou = compute_iou(action_np, target_mask)
                episode_iou += step_iou

            # Store trajectory
            states.append(state_tensor)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            entropies.append(entropy)

            episode_reward += reward
            steps += 1
            obs = next_obs

        # Compute returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + config['gamma'] * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32).unsqueeze(1)  # (T, 1)

        # Recompute log_probs, values, entropies WITH gradients
        # Stack states into batch
        states_batch = torch.cat(states, dim=0)  # (T, 14, H, W)
        actions_batch = torch.stack(actions, dim=0).squeeze(1)  # (T, H, W) - remove batch dim

        # Forward pass with gradients
        policy_logits, values_new = local_model(states_batch)  # (T, H, W), (T, 1)

        # Ensure policy_logits has 3 dimensions (T, H, W)
        if policy_logits.dim() == 2:
            # If somehow flattened to (T, H*W), we need to unflatten
            # For now, just flatten everything and sum
            T = len(states)
            policy_logits = policy_logits.view(T, -1)  # (T, H*W)
            actions_batch = actions_batch.view(T, -1)  # (T, H*W)

        # Compute log probabilities
        probs = torch.sigmoid(policy_logits)
        probs_clamped = torch.clamp(probs, 1e-7, 1 - 1e-7)

        # Compute log_probs for taken actions
        log_probs_per_cell = (actions_batch * torch.log(probs_clamped) +
                              (1 - actions_batch) * torch.log(1 - probs_clamped))

        # Sum over spatial dimensions (handle both 3D and 2D cases)
        if log_probs_per_cell.dim() == 3:
            log_probs_new = log_probs_per_cell.sum(dim=(1, 2))  # (T,)
        else:
            log_probs_new = log_probs_per_cell.sum(dim=1)  # (T,)

        # Compute entropy
        entropy_per_cell = -(probs_clamped * torch.log(probs_clamped) +
                            (1 - probs_clamped) * torch.log(1 - probs_clamped))
        if entropy_per_cell.dim() == 3:
            entropy_new = entropy_per_cell.sum(dim=(1, 2))  # (T,)
        else:
            entropy_new = entropy_per_cell.sum(dim=1)  # (T,)

        # Compute advantages
        advantages = (returns - values_new.detach()).squeeze(1)  # (T,)

        # Compute losses
        policy_loss = -(log_probs_new * advantages).mean()
        value_loss = F.mse_loss(values_new, returns)
        entropy_loss = -entropy_new.mean()

        total_loss = (policy_loss +
                     config['value_loss_coef'] * value_loss +
                     config['entropy_coef'] * entropy_loss)

        # Backprop to compute gradients on local model
        total_loss.backward()

        # Clip gradients on local model
        torch.nn.utils.clip_grad_norm_(local_model.parameters(), config['max_grad_norm'])

        # Update shared model (with lock to prevent race conditions)
        with lock:
            # Copy local gradients to shared model
            for shared_param, local_param in zip(shared_model.parameters(), local_model.parameters()):
                if local_param.grad is not None:
                    shared_param.grad = local_param.grad.clone()

            # Update shared model
            optimizer.step()
            optimizer.zero_grad()  # Zero shared gradients for next worker

            # Update global counters
            global_episode_counter.value += 1
            episode_count = global_episode_counter.value

            # Update best IoU
            avg_iou = episode_iou / max(1, steps)
            if avg_iou > global_best_iou.value:
                global_best_iou.value = avg_iou

        # Log progress
        if episode_count % config.get('log_interval', 10) == 0:
            avg_iou = episode_iou / max(1, steps)
            print(f"Worker {worker_id} | Episode {episode_count} | "
                  f"Reward: {episode_reward:.2f} | IoU: {avg_iou:.4f} | "
                  f"Steps: {steps} | Loss: {total_loss.item():.4f}")

    print(f"Worker {worker_id} finished after {episode_count} episodes")
