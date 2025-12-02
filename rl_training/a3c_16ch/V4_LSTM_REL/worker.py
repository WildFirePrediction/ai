"""
A3C Worker V4 LSTM REL - 16 Channels with IMPROVED REWARD SHAPING
Improvements over V3:
1. Longer LSTM sequence (5 instead of 3)
2. Stronger numerical stability (1e-6 clamping)
3. Reward shaping for aggressive spread (NEW!)

Key Change in V4:
- Adds spread bonus to encourage predicting MORE cells
- Penalizes NO spread when fire is active
- Balances IoU accuracy (70%) with spread aggressiveness (30%)
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
import random
from scipy.ndimage import binary_dilation

torch.set_num_threads(1)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from a3c_16ch.V4_LSTM_REL.model import A3C_PerCellModel_LSTM


def augment_data(states, fire_mask, next_fire):
    """
    Data augmentation: random rotation (0, 90, 180, 270) and horizontal flip

    Args:
        states: (seq_len, 16, 30, 30)
        fire_mask: (30, 30)
        next_fire: (30, 30)

    Returns:
        Augmented states, fire_mask, next_fire (all with positive strides)
    """
    # Random rotation
    k = random.randint(0, 3)  # 0, 90, 180, 270 degrees
    if k > 0:
        states = np.rot90(states, k=k, axes=(2, 3)).copy()  # Rotate spatial dims
        fire_mask = np.rot90(fire_mask, k=k).copy()
        next_fire = np.rot90(next_fire, k=k).copy()

    # Random horizontal flip
    if random.random() > 0.5:
        states = np.flip(states, axis=3).copy()  # Flip width
        fire_mask = np.flip(fire_mask, axis=1).copy()
        next_fire = np.flip(next_fire, axis=1).copy()

    return states, fire_mask, next_fire


def compute_relaxed_iou(pred, target):
    """
    Compute IoU with 8-neighbor tolerance (dilated ground truth).

    Args:
        pred: (H, W) predicted mask
        target: (H, W) ground truth mask

    Returns:
        IoU with dilated target (relaxed matching)
    """
    # 3x3 dilation kernel for 8-neighbor tolerance
    structure = np.ones((3, 3), dtype=bool)

    # Dilate target
    target_dilated = binary_dilation(target > 0.5, structure=structure)
    pred_binary = pred > 0.5

    intersection = (pred_binary & target_dilated).sum()
    union = (pred_binary | target_dilated).sum()

    return float(intersection / (union + 1e-8))


def compute_reward_with_spread_bonus(predicted_mask, new_burns, current_fire):
    """
    V4 IMPROVEMENT: Compute reward with spread bonus to encourage aggressive predictions.

    Args:
        predicted_mask: (H, W) binary predicted mask
        new_burns: (H, W) binary ground truth new burns
        current_fire: (H, W) binary current fire mask

    Returns:
        reward: Combined IoU + spread bonus
    """
    # Base reward: RELAXED IoU (8-neighbor tolerance)
    iou_reward = compute_relaxed_iou(predicted_mask, new_burns)

    # SPREAD BONUS: Reward for predicting MORE cells
    num_current_burning = current_fire.sum()
    num_predicted_new = predicted_mask.sum()

    # Calculate spread bonus
    if num_current_burning > 0:
        # Normalize by current fire size (encourage proportional spread)
        spread_ratio = num_predicted_new / max(1, num_current_burning)

        # Bonus for predicting 10-50% new cells (cap at 0.5)
        # Too aggressive (>50%) gets diminishing returns
        if spread_ratio < 0.1:
            # Too conservative
            spread_bonus = -0.2
        elif spread_ratio < 0.5:
            # Good spread (linear bonus)
            spread_bonus = spread_ratio * 1.0  # Max 0.5
        else:
            # Too aggressive (cap at 0.5)
            spread_bonus = 0.5
    else:
        # No current fire (edge case)
        spread_bonus = 0.0

    # Penalize NO predictions when fire is burning
    fire_moved = new_burns.sum() > 0
    model_predicted = predicted_mask.sum() > 0

    if fire_moved and not model_predicted:
        # Fire moved but model was silent (-10 penalty)
        return -10.0
    elif not fire_moved and model_predicted:
        # False alarm: predicted but fire didn't move (-5 penalty)
        return -5.0

    # Combine IoU (70%) and spread bonus (30%)
    combined_reward = (0.7 * iou_reward + 0.3 * spread_bonus) * 10.0

    return combined_reward


def worker_process_lstm(worker_id, shared_model, optimizer, filtered_episodes, config,
                        global_episode_counter, global_best_iou, lock, metrics_queue=None):
    """
    Worker process for A3C V4 LSTM with IMPROVED REWARD SHAPING.

    Args:
        worker_id: Worker ID
        shared_model: Shared global model
        optimizer: Shared optimizer
        filtered_episodes: List of episode file paths
        config: Training configuration
        global_episode_counter: Shared episode counter
        global_best_iou: Shared best IoU
        lock: Synchronization lock
        metrics_queue: Queue for WandB metrics
    """
    torch.manual_seed(config['seed'] + worker_id)
    np.random.seed(config['seed'] + worker_id)
    random.seed(config['seed'] + worker_id)

    sequence_length = config.get('sequence_length', 5)  # V4: default to 5
    local_model = A3C_PerCellModel_LSTM(in_channels=16, sequence_length=sequence_length)
    local_model.train()

    episode_count = 0
    max_episodes = config.get('max_episodes', 1000)

    print(f"[Worker {worker_id}] Starting V4 with {len(filtered_episodes)} episodes (sequence_length={sequence_length}, SPREAD BONUS)", flush=True)

    while True:
        with lock:
            if global_episode_counter.value >= max_episodes:
                break

        local_model.load_state_dict(shared_model.state_dict())

        episode_file = filtered_episodes[np.random.randint(len(filtered_episodes))]

        try:
            data = np.load(episode_file)
            states_np = data['states']  # (T, 16, 30, 30)
            fire_masks = data['fire_masks']  # (T, 30, 30)
            T = len(states_np)
        except Exception as e:
            print(f"[Worker {worker_id}] Error loading episode: {e}")
            continue

        # Need at least sequence_length+1 timesteps to form sequences
        if T < sequence_length + 1:
            continue

        # Choose random start (must leave room for sequence)
        max_length = min(10, T - sequence_length)
        start_t = np.random.randint(sequence_length, max(sequence_length + 1, T - max_length))

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
            # Get sequence: [t-seq_len+1, ..., t]
            seq_start = t - sequence_length + 1
            if seq_start < 0:
                continue

            sequence_t = states_np[seq_start:t+1]  # (seq_len, 16, 30, 30)
            current_fire = fire_masks[t]  # (30, 30)
            next_fire = fire_masks[t+1]  # (30, 30)

            # DATA AUGMENTATION
            sequence_t_aug, current_fire_aug, next_fire_aug = augment_data(
                sequence_t.copy(), current_fire.copy(), next_fire.copy()
            )

            # Prepare tensors
            sequence_tensor = torch.from_numpy(sequence_t_aug).unsqueeze(0).float()  # (1, seq_len, 16, 30, 30)
            current_fire_tensor = torch.from_numpy(current_fire_aug).unsqueeze(0).float()  # (1, 30, 30)

            with torch.no_grad():
                action_grid, log_prob, entropy, value, _ = local_model.get_action_and_value(
                    sequence_tensor, current_fire_tensor
                )

            action_np = action_grid.numpy()

            # Compute ground truth new burns
            actual_mask_t = current_fire_aug > 0
            actual_mask_t1 = next_fire_aug > 0
            new_burns = (actual_mask_t1 & ~actual_mask_t)
            predicted_mask = action_np > 0.5

            # V4 IMPROVEMENT: Compute reward with spread bonus
            reward = compute_reward_with_spread_bonus(predicted_mask, new_burns, actual_mask_t)

            # Compute IoU for logging
            step_iou = compute_relaxed_iou(predicted_mask, new_burns)

            states.append(sequence_tensor)
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

        # Compute returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + config['gamma'] * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32).unsqueeze(1)

        # Recompute log probs and values with gradients
        recomputed_log_probs = []
        recomputed_values = []
        recomputed_entropies = []

        # Reload data for recomputation (no augmentation for gradient computation)
        data = np.load(episode_file)
        states_reload = data['states']
        fire_masks_np = data['fire_masks']

        for t_idx, t in enumerate(range(start_t, min(start_t + max_length, T - 1))):
            seq_start = t - sequence_length + 1
            if seq_start < 0:
                continue

            sequence_t = states_reload[seq_start:t+1]
            sequence_tensor = torch.from_numpy(sequence_t).unsqueeze(0).float()

            action_t = actions[t_idx]
            fire_mask_t = torch.from_numpy(fire_masks_np[t]).unsqueeze(0).float()

            features, value_t = local_model(sequence_tensor, fire_mask_t)

            burning_cells = local_model.get_burning_cells(fire_mask_t)

            if len(burning_cells) == 0:
                recomputed_log_probs.append(torch.tensor(0.0))
                recomputed_values.append(value_t)
                recomputed_entropies.append(torch.tensor(0.0))
                continue

            step_log_probs = []
            step_entropies = []

            H, W = fire_mask_t.shape[1], fire_mask_t.shape[2]

            for i, j in burning_cells:
                logits_8d = local_model.predict_8_neighbors(features, i, j).squeeze(0)

                probs_8d = torch.sigmoid(logits_8d)
                # V4 IMPROVEMENT: Stronger clamping (1e-6 instead of 1e-7)
                probs_8d = torch.clamp(probs_8d, 1e-6, 1 - 1e-6)

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

        if len(recomputed_log_probs) < 2:
            continue

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
                print(f"[Worker {worker_id}] NEW BEST RELAXED IoU: {avg_iou:.4f} at episode {episode_count} - Checkpoint saved!")

        if episode_count % config.get('log_interval', 10) == 0:
            avg_iou = episode_iou / max(1, steps)
            avg_reward = episode_reward / max(1, steps)
            print(f"Worker {worker_id} | Episode {episode_count} | "
                  f"Reward: {episode_reward:.4f} (avg: {avg_reward:.4f}) | "
                  f"Relaxed IoU: {avg_iou:.4f} | Steps: {steps} | "
                  f"Loss: {total_loss.item():.4f}")

            if metrics_queue is not None:
                metrics_queue.put({
                    "episode": episode_count,
                    "train/episode_reward": episode_reward,
                    "train/avg_reward_per_step": avg_reward,
                    "train/avg_relaxed_iou": avg_iou,
                    "train/steps": steps,
                    "train/loss": total_loss.item(),
                    "train/policy_loss": policy_loss.item(),
                    "train/value_loss": value_loss.item(),
                    "train/entropy_loss": entropy_loss.item(),
                    "train/best_relaxed_iou": global_best_iou.value,
                    f"worker_{worker_id}/reward": episode_reward,
                    f"worker_{worker_id}/relaxed_iou": avg_iou,
                })

    print(f"[Worker {worker_id}] Finished after {episode_count} episodes")
