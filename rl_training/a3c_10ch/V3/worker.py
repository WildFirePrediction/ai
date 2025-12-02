"""
A3C Worker V3 - 10 Channels (Wind-Focused)
Correct Formulation with Dense Rewards

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

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from a3c_10ch.V3.model import A3C_PerCellModel_Deep


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

    local_model = A3C_PerCellModel_Deep(in_channels=10)
    local_model.train()

    episode_count = 0
    max_episodes = config.get('max_episodes', 1000)

    print(f"[Worker {worker_id}] Starting with {len(filtered_episodes)} episodes", flush=True)

    while True:
        with lock:
            if global_episode_counter.value >= max_episodes:
                break

        local_model.load_state_dict(shared_model.state_dict())

        episode_file = filtered_episodes[np.random.randint(len(filtered_episodes))]

        try:
            data = np.load(episode_file)
            states_np = data['states']  # (T, 10, 30, 30) - DEM + Wind + NDVI + FSM
            fire_masks = data['fire_masks']  # (T, 30, 30) - Fire positions
            T = len(states_np)
        except Exception as e:
            print(f"[Worker {worker_id}] Error loading episode: {e}")
            continue

        # Choose random start within episode
        max_length = min(10, T - 1)
        start_t = np.random.randint(0, max(1, T - max_length))

        states = []
        actions = []
        rewards = []
        log_probs = []
        values = []
        entropies = []

        episode_reward = 0
        episode_iou = 0
        steps = 0

        for t in range(start_t, min(start_t + max_length, T - 1)):
            # Current state: 10 channels (DEM + Wind + NDVI + FSM)
            features_t = states_np[t]  # (10, 30, 30)
            current_fire = fire_masks[t]  # (30, 30)
            next_fire = fire_masks[t+1]  # (30, 30)

            # Action is new burns (cells that burned in next timestep)
            action_target = (next_fire > 0.5).astype(np.float32) * (current_fire < 0.5).astype(np.float32)

            # Full state with fire mask
            state_tensor = torch.from_numpy(features_t).unsqueeze(0).float()  # (1, 10, 30, 30)
            current_fire_tensor = torch.from_numpy(current_fire).unsqueeze(0).float()  # (1, 30, 30)

            with torch.no_grad():
                action_grid, log_prob, entropy, value, _ = local_model.get_action_and_value(
                    state_tensor, current_fire_tensor
                )

            action_np = action_grid.numpy()

            # Compute reward based on F1-score (balanced precision and recall)
            actual_mask_t = current_fire > 0
            actual_mask_t1 = next_fire > 0
            new_burns = (actual_mask_t1 & ~actual_mask_t)
            predicted_mask = action_np > 0.5

            # Calculate TP, FP, FN for F1-score
            TP = (predicted_mask & new_burns).sum()
            FP = (predicted_mask & ~new_burns).sum()
            FN = (~predicted_mask & new_burns).sum()

            # F1-score: 2*TP / (2*TP + FP + FN)
            precision = TP / (TP + FP + 1e-8) if (TP + FP) > 0 else 0.0
            recall = TP / (TP + FN + 1e-8) if (TP + FN) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall + 1e-8) if (precision + recall) > 0 else 0.0

            reward = f1_score * 10.0  # Scale reward

            # Also compute IoU for logging purposes
            intersection = TP
            union = (predicted_mask | new_burns).sum()
            step_iou = intersection / (union + 1e-8) if union > 0 else 0.0

            states.append(state_tensor)
            actions.append(action_grid)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            entropies.append(entropy)

            episode_reward += reward
            episode_iou += step_iou
            steps += 1

        if steps < 2:
            continue

        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + config['gamma'] * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32).unsqueeze(1)

        recomputed_log_probs = []
        recomputed_values = []
        recomputed_entropies = []

        # Load the fire masks separately for recomputation
        data = np.load(episode_file)
        fire_masks_np = data['fire_masks']

        for t_idx, t in enumerate(range(start_t, min(start_t + max_length, T - 1))):
            state_t = states[t_idx]
            action_t = actions[t_idx]

            fire_mask_t = torch.from_numpy(fire_masks_np[t]).unsqueeze(0).float()

            features, value_t = local_model(state_t, fire_mask_t)

            burning_cells = local_model.get_burning_cells(fire_mask_t)

            if len(burning_cells) == 0:
                recomputed_log_probs.append(torch.tensor(0.0))
                recomputed_values.append(value_t)
                recomputed_entropies.append(torch.tensor(0.0))
                continue

            step_log_probs = []
            step_entropies = []

            H, W = state_t.shape[2], state_t.shape[3]

            for i, j in burning_cells:
                logits_8d = local_model.predict_8_neighbors(features, i, j).squeeze(0)

                probs_8d = torch.sigmoid(logits_8d)
                probs_8d = torch.clamp(probs_8d, 1e-7, 1 - 1e-7)

                neighbors = local_model.get_8_neighbor_coords(i, j, H, W)
                action_8d = torch.zeros(8)
                for n_idx, neighbor in enumerate(neighbors):
                    if neighbor is not None:
                        ni, nj = neighbor
                        action_8d[n_idx] = action_t[ni, nj]

                log_prob_8d = (action_8d * torch.log(probs_8d) +
                              (1 - action_8d) * torch.log(1 - probs_8d))
                step_log_probs.append(log_prob_8d.sum())

                entropy_8d = -(probs_8d * torch.log(probs_8d) +
                              (1 - probs_8d) * torch.log(1 - probs_8d))
                step_entropies.append(entropy_8d.sum())

            total_log_prob_t = torch.stack(step_log_probs).sum()
            total_entropy_t = torch.stack(step_entropies).sum()

            recomputed_log_probs.append(total_log_prob_t)
            recomputed_values.append(value_t)
            recomputed_entropies.append(total_entropy_t)

        log_probs_tensor = torch.stack(recomputed_log_probs)
        values_tensor = torch.cat(recomputed_values)
        entropy_tensor = torch.stack(recomputed_entropies)

        advantages = (returns - values_tensor.detach()).squeeze(1)

        policy_loss = -(log_probs_tensor * advantages).mean()
        value_loss = F.mse_loss(values_tensor, returns)
        entropy_loss = -entropy_tensor.mean()

        total_loss = (policy_loss +
                     config['value_loss_coef'] * value_loss +
                     config['entropy_coef'] * entropy_loss)

        if torch.isnan(total_loss):
            print(f"[Worker {worker_id}] NaN loss detected, skipping update")
            continue

        optimizer.zero_grad()
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(local_model.parameters(), config['max_grad_norm'])

        with lock:
            for shared_param, local_param in zip(shared_model.parameters(), local_model.parameters()):
                if local_param.grad is not None:
                    if shared_param.grad is None:
                        shared_param.grad = local_param.grad.clone()
                    else:
                        shared_param.grad += local_param.grad

            optimizer.step()
            optimizer.zero_grad()

            global_episode_counter.value += 1
            episode_count = global_episode_counter.value

            avg_iou = episode_iou / max(1, steps)
            if avg_iou > global_best_iou.value:
                global_best_iou.value = avg_iou

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

        if episode_count % config.get('log_interval', 10) == 0:
            avg_iou = episode_iou / max(1, steps)
            avg_reward = episode_reward / max(1, steps)
            print(f"Worker {worker_id} | Episode {episode_count} | "
                  f"Reward: {episode_reward:.4f} (avg: {avg_reward:.4f}) | "
                  f"IoU: {avg_iou:.4f} | Steps: {steps} | "
                  f"Loss: {total_loss.item():.4f}")

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
