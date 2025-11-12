"""
A3C Worker V2 - Filtered Episodes

KEY IMPROVEMENT: Only samples from pre-filtered episodes with actual fire spread!
Each episode is guaranteed to have burns, providing better learning signal.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys

torch.set_num_threads(1)

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


def worker_process_filtered(worker_id, shared_model, optimizer, filtered_episodes, config,
                            global_episode_counter, global_best_iou, lock, metrics_queue=None):
    """
    Worker process for A3C V2 with filtered episodes.

    Args:
        worker_id: Worker ID
        shared_model: Shared global model
        optimizer: Shared optimizer
        filtered_episodes: List of (env_path, start_timestep, max_length) tuples
        config: Training configuration
        global_episode_counter: Shared counter
        global_best_iou: Shared best IoU
        lock: Synchronization lock
        metrics_queue: Queue for WandB metrics
    """
    torch.manual_seed(config['seed'] + worker_id)
    np.random.seed(config['seed'] + worker_id)

    # Create local model
    local_model = A3C_SpatialFireModel(in_channels=14)
    local_model.train()

    episode_count = 0
    max_episodes = config.get('max_episodes', 1000)

    print(f"[Worker {worker_id}] Starting with {len(filtered_episodes)} filtered episodes", flush=True)

    while True:
        # Check if training should stop
        with lock:
            if global_episode_counter.value >= max_episodes:
                break

        # Sync local model with shared model
        local_model.load_state_dict(shared_model.state_dict())

        # Sample random FILTERED episode
        env_path, start_t, max_length = filtered_episodes[np.random.randint(len(filtered_episodes))]

        # Load environment
        env = WildfireEnvSpatial(env_path)

        # Fast-forward to start_t
        obs, info = env.reset()
        for _ in range(start_t):
            obs, _, _, _ = env.step(np.zeros((env.H, env.W)))

        # Collect trajectory
        states, actions, rewards, log_probs, values, entropies = [], [], [], [], [], []

        done = False
        episode_reward = 0
        episode_iou = 0
        steps = 0

        while not done and steps < max_length:
            # Convert obs to tensor
            state_tensor = torch.from_numpy(obs).unsqueeze(0).float()

            # Get action from local model
            with torch.no_grad():
                action, log_prob, entropy, value = local_model.get_action_and_value(state_tensor)

            # Convert action to numpy
            action_np = action.squeeze(0).numpy()

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

        # Skip if episode too short
        if steps < 2:
            continue

        # Compute returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + config['gamma'] * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32).unsqueeze(1)

        # Recompute log_probs, values, entropies with gradients
        states_batch = torch.cat(states, dim=0)
        actions_batch = torch.stack(actions, dim=0).squeeze(1)

        # Forward pass
        policy_logits, values_new = local_model(states_batch)

        # Handle dimensions
        if policy_logits.dim() == 2:
            T = len(states)
            policy_logits = policy_logits.view(T, -1)
            actions_batch = actions_batch.view(T, -1)

        # Compute probabilities
        probs = torch.sigmoid(policy_logits)
        probs_clamped = torch.clamp(probs, 1e-7, 1 - 1e-7)

        # Compute log probabilities
        log_probs_per_cell = (actions_batch * torch.log(probs_clamped) +
                              (1 - actions_batch) * torch.log(1 - probs_clamped))

        if log_probs_per_cell.dim() == 3:
            log_probs_new = log_probs_per_cell.sum(dim=(1, 2))
        else:
            log_probs_new = log_probs_per_cell.sum(dim=1)

        # Compute entropy
        entropy_per_cell = -(probs_clamped * torch.log(probs_clamped) +
                            (1 - probs_clamped) * torch.log(1 - probs_clamped))
        if entropy_per_cell.dim() == 3:
            entropy_new = entropy_per_cell.sum(dim=(1, 2))
        else:
            entropy_new = entropy_per_cell.sum(dim=1)

        # Compute advantages
        advantages = (returns - values_new.detach()).squeeze(1)

        # Compute losses
        policy_loss = -(log_probs_new * advantages).mean()
        value_loss = F.mse_loss(values_new, returns)
        entropy_loss = -entropy_new.mean()

        total_loss = (policy_loss +
                     config['value_loss_coef'] * value_loss +
                     config['entropy_coef'] * entropy_loss)

        # Backprop
        total_loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(local_model.parameters(), config['max_grad_norm'])

        # Update shared model
        with lock:
            # Copy gradients to shared model
            for shared_param, local_param in zip(shared_model.parameters(), local_model.parameters()):
                if local_param.grad is not None:
                    shared_param.grad = local_param.grad.clone()

            # Update shared model
            optimizer.step()
            optimizer.zero_grad()

            # Update counters
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
                  f"Reward: {episode_reward:.4f} | IoU: {avg_iou:.4f} | "
                  f"Steps: {steps} | Loss: {total_loss.item():.4f}")

            # Send metrics to queue
            if metrics_queue is not None:
                metrics_queue.put({
                    "episode": episode_count,
                    "train/reward": episode_reward,
                    "train/iou": avg_iou,
                    "train/steps": steps,
                    "train/loss": total_loss.item(),
                    "train/policy_loss": policy_loss.item(),
                    "train/value_loss": value_loss.item(),
                    "train/entropy_loss": entropy_loss.item(),
                    "train/best_iou": global_best_iou.value,
                    f"worker_{worker_id}/reward": episode_reward,
                    f"worker_{worker_id}/iou": avg_iou,
                })

    print(f"[Worker {worker_id}] Finished after {episode_count} episodes")
