"""
A3C Worker V6 LSTM RECALL - RECALL-FIRST SUPERAGGRO REWARD SHAPING
16 Channels with RELAXED IoU (8-neighbor tolerance) + RECALL-OPTIMIZED reward system

RECALL-FIRST SUPERAGGRO Reward System:
- Missing fire spread (false negative): -500.0 penalty (CATASTROPHIC - 5x worse!)
- False alarm (false positive): +20.0 REWARD (Encourage over-prediction!)
- Correct prediction with HIGH RECALL (95%+): +200.0 (MASSIVE REWARD)
- Correct prediction with GOOD RECALL (80%+): +100.0
- Correct prediction with OK RECALL (50%+): +50.0
- Correct prediction with LOW RECALL (<50%): -100.0 (Still bad!)
- Small precision penalty: -(1-precision) * 2.0 (tiny discouragement for going too wild)
- Correct silence: 0.0 reward (don't encourage inaction)

Philosophy: Coverage is EVERYTHING. Catch 100% of fire spread, even if it means
predicting 10x more area than necessary. Better to evacuate an entire mountain
than to miss one house.

Risk ratio: 25:1 (-500 vs +20) massively biases toward prediction.

Data Augmentation:
- Random rotation (0, 90, 180, 270 degrees)
- Random horizontal flip
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

from a3c_16ch.V7_LSTM_RECALL.model import A3C_PerCellModel_LSTM


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


def worker_process_lstm(worker_id, shared_model, optimizer, filtered_episodes, config,
                        global_episode_counter, global_best_iou, lock, metrics_queue=None):
    """
    Worker process for A3C V6 LSTM with RECALL-FIRST SUPERAGGRO reward shaping.

    RECALL-FIRST SUPERAGGRO Reward System:
    - Missing fire spread (false negative): -500.0 penalty (CATASTROPHIC!)
    - False alarm (false positive): +20.0 REWARD (Encourage caution!)
    - Correct prediction:
        * Recall >= 95%: +200.0 (MASSIVE - caught almost everything!)
        * Recall >= 80%: +100.0 (Great coverage)
        * Recall >= 50%: +50.0 (OK coverage)
        * Recall < 50%: -100.0 (Missed too much, still bad)
        * Small precision penalty: -(1-precision)*2.0
    - Correct silence: 0.0 reward (don't encourage doing nothing)

    25:1 risk ratio (-500 vs +20) massively biases toward over-prediction.

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

    sequence_length = config.get('sequence_length', 3)
    local_model = A3C_PerCellModel_LSTM(in_channels=16, sequence_length=sequence_length)
    local_model.train()

    episode_count = 0
    max_episodes = config.get('max_episodes', 1000)

    print(f"[Worker {worker_id}] Starting with {len(filtered_episodes)} episodes (sequence_length={sequence_length}, RECALL-FIRST SUPERAGGRO)", flush=True)

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

            # DATA AUGMENTATION - ALWAYS APPLIED
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

            # Compute RELAXED IoU with 8-neighbor tolerance
            actual_mask_t = current_fire_aug > 0
            actual_mask_t1 = next_fire_aug > 0
            new_burns = (actual_mask_t1 & ~actual_mask_t)
            predicted_mask = action_np > 0.5

            # RELAXED IoU: Use dilated ground truth
            step_iou = compute_relaxed_iou(predicted_mask, new_burns)

            # RECALL-FIRST SUPERAGGRO REWARD SHAPING
            fire_moved = new_burns.sum() > 0
            model_predicted = predicted_mask.sum() > 0

            if fire_moved and not model_predicted:
                # CATASTROPHIC FAILURE: Fire spread but model missed it completely
                # People could die - MAXIMUM PENALTY (5x worse than V5!)
                reward = -500.0
                penalty_type = "CATASTROPHIC_MISS"

            elif not fire_moved and model_predicted:
                # False alarm: predicted movement but fire didn't move
                # REWARD THIS HEAVILY! Being overly cautious is EXCELLENT for safety
                # Better to evacuate 100 areas unnecessarily than miss 1 real spread
                reward = +20.0  # 4x higher than V5!
                penalty_type = "FALSE_ALARM_GREAT"

            else:
                # Model behavior aligns with reality
                if fire_moved and model_predicted:
                    # Both predicted and happened - reward based on RECALL (coverage)
                    intersection = (predicted_mask & new_burns).sum()
                    recall = intersection / (new_burns.sum() + 1e-8)
                    precision = intersection / (predicted_mask.sum() + 1e-8)

                    # RECALL-FIRST reward tiers
                    if recall >= 0.95:
                        # Caught 95%+ of spread - MASSIVE REWARD
                        reward = 200.0
                        penalty_type = "EXCELLENT_RECALL"
                    elif recall >= 0.80:
                        # Caught 80%+ - great coverage
                        reward = 100.0
                        penalty_type = "GREAT_RECALL"
                    elif recall >= 0.50:
                        # Caught 50%+ - acceptable
                        reward = 50.0
                        penalty_type = "OK_RECALL"
                    else:
                        # Caught <50% - still bad, missed too much
                        reward = -100.0
                        penalty_type = "LOW_RECALL_BAD"

                    # Small precision penalty (discourage going TOO wild)
                    # But keep it small - we want over-prediction
                    precision_penalty = (1 - precision) * 2.0
                    reward -= precision_penalty

                else:
                    # Both didn't predict and didn't happen
                    # NO REWARD - don't encourage staying quiet
                    reward = 0.0
                    penalty_type = "CORRECT_NEGATIVE_NEUTRAL"

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

        # Compute returns with discounting
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + config['gamma'] * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32).unsqueeze(1)

        # Recompute log probs and values with gradients (no augmentation)
        recomputed_log_probs = []
        recomputed_values = []
        recomputed_entropies = []

        # Reload data for recomputation (NO augmentation for gradient computation)
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
                print(f"[Worker {worker_id}] NEW BEST RELAXED IoU: {avg_iou:.4f} at episode {episode_count} - Checkpoint saved!", flush=True)

        if episode_count % config.get('log_interval', 10) == 0:
            avg_iou = episode_iou / max(1, steps)
            avg_reward = episode_reward / max(1, steps)
            print(f"Worker {worker_id} | Episode {episode_count} | "
                  f"Reward: {episode_reward:.4f} (avg: {avg_reward:.4f}) | "
                  f"Relaxed IoU: {avg_iou:.4f} | Steps: {steps} | "
                  f"Loss: {total_loss.item():.4f}", flush=True)

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

    print(f"[Worker {worker_id}] Finished after {episode_count} episodes", flush=True)
