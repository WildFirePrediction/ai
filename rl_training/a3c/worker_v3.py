"""
A3C Worker V3 - Correct Formulation with Dense Rewards

Per-cell 8-neighbor prediction with DENSE REWARDS at every timestep.
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
from a3c.model_v2 import A3C_PerCellModel


def compute_iou(pred, target):
    """Compute IoU between predicted and target burn masks."""
    pred_flat = pred.flatten()
    target_flat = target.flatten()

    intersection = (pred_flat * target_flat).sum()
    union = np.clip(pred_flat + target_flat, 0, 1).sum()

    return float(intersection / (union + 1e-8))


def worker_process_correct(worker_id, shared_model, optimizer, filtered_episodes, config,
                           global_episode_counter, global_best_iou, lock, metrics_queue=None):
    """
    Worker process for A3C V3 with correct formulation and dense rewards.

    Args:
        worker_id: Worker ID
        shared_model: Shared global model
        optimizer: Shared optimizer
        filtered_episodes: List of (env_path, start_timestep, max_length) tuples
        config: Training configuration
        global_episode_counter: Shared episode counter
        global_best_iou: Shared best IoU
        lock: Synchronization lock
        metrics_queue: Queue for WandB metrics
    """
    torch.manual_seed(config['seed'] + worker_id)
    np.random.seed(config['seed'] + worker_id)

    # Create local model
    local_model = A3C_PerCellModel(in_channels=15)
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

        # Sample random filtered episode
        env_path, start_t, max_length = filtered_episodes[np.random.randint(len(filtered_episodes))]

        # Load environment
        try:
            env = WildfireEnvSpatial(env_path)
        except Exception as e:
            print(f"[Worker {worker_id}] Error loading env: {e}")
            continue

        # Fast-forward to start_t
        obs, info = env.reset()
        for _ in range(start_t):
            obs, _, _, _ = env.step(np.zeros((env.H, env.W)))

        # Collect trajectory with DENSE REWARDS
        states = []
        actions = []
        rewards = []  # Dense rewards at every timestep!
        log_probs = []
        values = []
        entropies = []

        done = False
        episode_reward = 0
        episode_iou = 0
        steps = 0

        while not done and steps < max_length:
            # Convert obs to tensor
            state_tensor = torch.from_numpy(obs).unsqueeze(0).float()  # (1, 14, H, W)

            # Get current fire mask (channel index depends on your obs structure)
            # Assuming fire mask is at channel 5 (after static features)
            fire_mask = state_tensor[0, 5]  # (H, W) - fire mask channel

            # Get action from local model
            with torch.no_grad():
                action_grid, log_prob, entropy, value, _ = local_model.get_action_and_value(
                    state_tensor, fire_mask
                )

            # Convert action to numpy
            action_np = action_grid.numpy()  # (H, W)

            # Step environment - GET DENSE REWARD!
            next_obs, reward, done, info = env.step(action_np)

            # Store trajectory
            states.append(state_tensor)
            actions.append(action_grid)
            rewards.append(reward)  # Dense reward!
            log_probs.append(log_prob)
            values.append(value)
            entropies.append(entropy)

            episode_reward += reward
            episode_iou += reward  # Reward IS IoU in our case
            steps += 1
            obs = next_obs

        # Skip if episode too short
        if steps < 2:
            continue

        # Compute returns (discounted cumulative rewards)
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + config['gamma'] * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32).unsqueeze(1)  # (T, 1)

        # Recompute values, log_probs, entropies WITH gradients for backprop
        recomputed_log_probs = []
        recomputed_values = []
        recomputed_entropies = []

        for t in range(len(states)):
            state_t = states[t]
            action_t = actions[t]

            # Get fire mask
            fire_mask_t = state_t[0, 5]  # (H, W)

            # Forward pass
            features, value_t = local_model(state_t, fire_mask_t.unsqueeze(0))

            # Get burning cells
            burning_cells = local_model.get_burning_cells(fire_mask_t)

            if len(burning_cells) == 0:
                # No burning cells - skip this timestep
                recomputed_log_probs.append(torch.tensor(0.0))
                recomputed_values.append(value_t)
                recomputed_entropies.append(torch.tensor(0.0))
                continue

            # Recompute log probs and entropy for each burning cell
            step_log_probs = []
            step_entropies = []

            for i, j in burning_cells:
                # Get 8-neighbor logits
                logits_8d = local_model.predict_8_neighbors(features, i, j).squeeze(0)  # (8,)

                # Get probabilities
                probs_8d = torch.sigmoid(logits_8d)
                probs_8d = torch.clamp(probs_8d, 1e-7, 1 - 1e-7)

                # Get action for this cell from action_grid
                neighbors = local_model.get_8_neighbor_coords(i, j, env.H, env.W)
                action_8d = torch.zeros(8)
                for n_idx, neighbor in enumerate(neighbors):
                    if neighbor is not None:
                        ni, nj = neighbor
                        action_8d[n_idx] = action_t[ni, nj]

                # Compute log prob
                log_prob_8d = (action_8d * torch.log(probs_8d) +
                              (1 - action_8d) * torch.log(1 - probs_8d))
                step_log_probs.append(log_prob_8d.sum())

                # Compute entropy
                entropy_8d = -(probs_8d * torch.log(probs_8d) +
                              (1 - probs_8d) * torch.log(1 - probs_8d))
                step_entropies.append(entropy_8d.sum())

            # Aggregate for this timestep
            total_log_prob_t = torch.stack(step_log_probs).sum()
            total_entropy_t = torch.stack(step_entropies).sum()

            recomputed_log_probs.append(total_log_prob_t)
            recomputed_values.append(value_t)
            recomputed_entropies.append(total_entropy_t)

        # Stack into tensors
        log_probs_tensor = torch.stack(recomputed_log_probs)  # (T,)
        values_tensor = torch.cat(recomputed_values)  # (T, 1)
        entropy_tensor = torch.stack(recomputed_entropies)  # (T,)

        # Compute advantages
        advantages = (returns - values_tensor.detach()).squeeze(1)  # (T,)

        # Compute losses
        policy_loss = -(log_probs_tensor * advantages).mean()
        value_loss = F.mse_loss(values_tensor, returns)
        entropy_loss = -entropy_tensor.mean()

        total_loss = (policy_loss +
                     config['value_loss_coef'] * value_loss +
                     config['entropy_coef'] * entropy_loss)

        # Check for NaN
        if torch.isnan(total_loss):
            print(f"[Worker {worker_id}] NaN loss detected, skipping update")
            continue

        # Backprop
        optimizer.zero_grad()
        total_loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(local_model.parameters(), config['max_grad_norm'])

        # Update shared model
        with lock:
            # Copy gradients to shared model
            for shared_param, local_param in zip(shared_model.parameters(), local_model.parameters()):
                if local_param.grad is not None:
                    if shared_param.grad is None:
                        shared_param.grad = local_param.grad.clone()
                    else:
                        shared_param.grad += local_param.grad

            # Update shared model
            optimizer.step()
            optimizer.zero_grad()

            # Update counters
            global_episode_counter.value += 1
            episode_count = global_episode_counter.value

            # Update best IoU and save checkpoint
            avg_iou = episode_iou / max(1, steps)
            if avg_iou > global_best_iou.value:
                global_best_iou.value = avg_iou

                # Save best model checkpoint
                checkpoint_dir = Path(config['checkpoint_dir'])
                best_model_path = checkpoint_dir / 'best_model.pt'
                torch.save({
                    'episode': episode_count,
                    'model_state_dict': shared_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_iou': avg_iou,
                    'worker_id': worker_id
                }, best_model_path)
                print(f"[Worker {worker_id}] NEW BEST IoU: {avg_iou:.4f} at episode {episode_count} - Checkpoint saved!")

        # Log progress
        if episode_count % config.get('log_interval', 10) == 0:
            avg_iou = episode_iou / max(1, steps)
            avg_reward = episode_reward / max(1, steps)
            print(f"Worker {worker_id} | Episode {episode_count} | "
                  f"Reward: {episode_reward:.4f} (avg: {avg_reward:.4f}) | "
                  f"IoU: {avg_iou:.4f} | Steps: {steps} | "
                  f"Loss: {total_loss.item():.4f}")

            # Send metrics to queue
            if metrics_queue is not None:
                metrics_queue.put({
                    "episode": episode_count,
                    "train/episode_reward": episode_reward,
                    "train/avg_reward_per_step": avg_reward,
                    "train/avg_iou": avg_iou,
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
