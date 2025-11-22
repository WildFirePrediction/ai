"""
A3C Worker V8 - Spatial + Channel Attention Training

Worker process for distributed A3C training with V8 attention model.
Handles episode rollouts, gradient computation, and model updates.

Key Features:
- Dense rewards (IoU at every timestep)
- Per-cell 8-neighbor prediction
- Attention-enhanced feature extraction
- Asynchronous gradient updates to shared model

Author: Wildfire Prediction Team
Version: 8.0
Date: 2025-11-22
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
from typing import Dict, List, Tuple, Optional, Any

torch.set_num_threads(1)

sys.path.insert(0, str(Path(__file__).parent.parent))

from wildfire_env_spatial import WildfireEnvSpatial
from a3c.model_v8 import A3C_PerCellModel_V8


def compute_iou(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute Intersection over Union (IoU) between predicted and target masks.
    
    IoU = |A ∩ B| / |A ∪ B|
    
    Args:
        pred: Predicted burn mask (H, W)
        target: Target burn mask (H, W)
    
    Returns:
        IoU score in [0, 1]
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    intersection = (pred_flat * target_flat).sum()
    union = np.clip(pred_flat + target_flat, 0, 1).sum()
    
    return float(intersection / (union + 1e-8))


def worker_process_v8(
    worker_id: int,
    shared_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    filtered_episodes: List[Tuple[Path, int, int]],
    config: Dict[str, Any],
    global_episode_counter,
    global_best_iou,
    lock,
    metrics_queue=None
):
    """
    Worker process for A3C V8 training.
    
    Each worker:
    1. Samples random filtered episode
    2. Collects trajectory with dense rewards
    3. Computes advantages using GAE
    4. Updates shared model asynchronously
    
    Args:
        worker_id: Unique worker identifier
        shared_model: Shared global model (lives in CPU memory)
        optimizer: Shared optimizer
        filtered_episodes: List of (env_path, start_timestep, max_length) tuples
        config: Training configuration dict
        global_episode_counter: Shared counter for total episodes
        global_best_iou: Shared best IoU tracker
        lock: Threading lock for synchronization
        metrics_queue: Queue for WandB metrics (optional)
    """
    # Set seeds for reproducibility
    torch.manual_seed(config['seed'] + worker_id)
    np.random.seed(config['seed'] + worker_id)
    
    # Create local model (each worker has its own copy)
    local_model = A3C_PerCellModel_V8(in_channels=15)
    local_model.train()
    
    # Extract config
    max_episodes = config.get('max_episodes', 1000)
    gamma = config.get('gamma', 0.99)
    gae_lambda = config.get('gae_lambda', 0.95)
    value_loss_coef = config.get('value_loss_coef', 0.5)
    entropy_coef = config.get('entropy_coef', 0.01)
    max_grad_norm = config.get('max_grad_norm', 0.5)
    log_interval = config.get('log_interval', 10)
    
    print(f"[Worker {worker_id}] Starting with {len(filtered_episodes)} filtered episodes", flush=True)
    
    while True:
        # Check if training should stop
        with lock:
            if global_episode_counter.value >= max_episodes:
                print(f"[Worker {worker_id}] Reached max episodes, stopping.", flush=True)
                break
        
        # Sync local model with shared model
        local_model.load_state_dict(shared_model.state_dict())
        
        # Sample random filtered episode
        env_path, start_t, max_length = filtered_episodes[np.random.randint(len(filtered_episodes))]
        
        # Load environment
        try:
            env = WildfireEnvSpatial(env_path)
        except Exception as e:
            print(f"[Worker {worker_id}] Error loading {env_path}: {e}", flush=True)
            continue
        
        # Fast-forward to start_t
        obs, info = env.reset()
        for _ in range(start_t):
            obs, _, _, _ = env.step(np.zeros((env.H, env.W)))
        
        # Collect trajectory with DENSE REWARDS
        states = []
        actions = []
        rewards = []  # Dense IoU reward at every timestep
        log_probs = []
        values = []
        entropies = []
        
        done = False
        episode_reward = 0.0
        episode_iou = 0.0
        steps = 0
        max_steps = min(max_length, env.T - start_t - 1)
        
        # Rollout episode
        while not done and steps < max_steps:
            # Convert observation to tensor
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)  # (1, 15, H, W)
            
            # Extract fire mask (channel 5)
            fire_mask = obs[5:6]  # (1, H, W)
            fire_mask_tensor = torch.FloatTensor(fire_mask)  # (1, H, W)
            
            # Get action from model
            with torch.no_grad():
                action_grid, log_prob, entropy, value, _ = local_model.get_action_and_value(
                    obs_tensor, fire_mask_tensor
                )
            
            # Convert action to numpy
            action_np = action_grid.numpy()
            
            # Step environment (get DENSE reward)
            next_obs, reward, done, info = env.step(action_np)
            
            # Store trajectory
            states.append(obs_tensor)
            actions.append(action_grid)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            entropies.append(entropy)
            
            # Update stats
            episode_reward += reward
            episode_iou += reward  # IoU is the reward
            steps += 1
            
            obs = next_obs
        
        # Handle empty episode (no steps taken)
        if steps == 0:
            continue
        
        # Bootstrap value for last state (if not terminal)
        if not done:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                fire_mask = obs[5:6]
                fire_mask_tensor = torch.FloatTensor(fire_mask)
                _, _, _, last_value, _ = local_model.get_action_and_value(
                    obs_tensor, fire_mask_tensor
                )
                last_value = last_value.item()
        else:
            last_value = 0.0
        
        # Compute advantages using Generalized Advantage Estimation (GAE)
        advantages = []
        returns = []
        
        gae = 0.0
        for t in reversed(range(steps)):
            if t == steps - 1:
                next_value = last_value
            else:
                next_value = values[t + 1].item()
            
            # TD error: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
            delta = rewards[t] + gamma * next_value - values[t].item()
            
            # GAE: A_t = δ_t + γ * λ * A_{t+1}
            gae = delta + gamma * gae_lambda * gae
            advantages.insert(0, gae)
            
            # Return: R_t = A_t + V(s_t)
            returns.insert(0, gae + values[t].item())
        
        # Convert to tensors
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        
        # Normalize advantages (improves training stability)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Recompute log_probs, values, entropies with gradients
        new_log_probs = []
        new_values = []
        new_entropies = []
        
        for t in range(steps):
            # Recompute action log_prob and entropy
            # states[t] is (1, 15, H, W), fire mask is channel 5
            fire_mask_t = states[t][:, 5:6, :, :].squeeze(1)  # (1, H, W)
            _, log_prob, entropy, value, _ = local_model.get_action_and_value(
                states[t], fire_mask_t, action=actions[t]
            )
            
            new_log_probs.append(log_prob)
            new_values.append(value.squeeze())
            new_entropies.append(entropy)
        
        # Stack tensors
        log_probs_tensor = torch.stack(new_log_probs)
        values_tensor = torch.stack(new_values)
        entropies_tensor = torch.stack(new_entropies)
        
        # Compute losses
        # Policy loss: -E[log π(a|s) * A(s,a)]
        policy_loss = -(log_probs_tensor * advantages).mean()
        
        # Value loss: E[(V(s) - R)^2]
        value_loss = F.mse_loss(values_tensor, returns)
        
        # Entropy bonus: -E[H(π)]
        entropy_loss = -entropies_tensor.mean()
        
        # Total loss
        total_loss = (
            policy_loss + 
            value_loss_coef * value_loss + 
            entropy_coef * entropy_loss
        )
        
        # Backpropagation and optimization
        with lock:
            # Zero gradients
            optimizer.zero_grad()
            
            # Compute gradients
            total_loss.backward()
            
            # Clip gradients (prevent exploding gradients)
            torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_grad_norm)
            
            # Update shared model
            for shared_param, local_param in zip(shared_model.parameters(), local_model.parameters()):
                if shared_param.grad is None:
                    shared_param.grad = local_param.grad.clone()
                else:
                    shared_param.grad += local_param.grad
            
            # Apply gradients
            optimizer.step()
            
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
                
                print(f"[Worker {worker_id}] NEW BEST IoU: {avg_iou:.4f} at episode {episode_count} - Checkpoint saved!", flush=True)
        
        # Log progress
        if episode_count % log_interval == 0:
            avg_iou = episode_iou / max(1, steps)
            avg_reward = episode_reward / max(1, steps)
            print(
                f"Worker {worker_id} | Episode {episode_count} | "
                f"Reward: {episode_reward:.4f} (avg: {avg_reward:.4f}) | "
                f"IoU: {avg_iou:.4f} | Steps: {steps} | "
                f"Loss: {total_loss.item():.4f} "
                f"(policy: {policy_loss.item():.4f}, value: {value_loss.item():.4f}, entropy: {entropy_loss.item():.4f})",
                flush=True
            )
        
        # Send metrics to WandB queue
        if metrics_queue is not None:
            avg_iou = episode_iou / max(1, steps)
            metrics = {
                f'worker_{worker_id}/episode_reward': episode_reward,
                f'worker_{worker_id}/episode_iou': avg_iou,
                f'worker_{worker_id}/episode_steps': steps,
                f'worker_{worker_id}/policy_loss': policy_loss.item(),
                f'worker_{worker_id}/value_loss': value_loss.item(),
                f'worker_{worker_id}/entropy': entropies_tensor.mean().item(),
                f'worker_{worker_id}/total_loss': total_loss.item(),
                'global/episode': episode_count,
                'global/best_iou': global_best_iou.value,
            }
            metrics_queue.put(metrics)
    
    print(f"[Worker {worker_id}] Finished training.", flush=True)
