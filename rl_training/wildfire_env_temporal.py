"""
Wildfire Environment with Temporal Context
Agent sees last N timesteps to model fire dynamics
"""
import pickle
from pathlib import Path
from typing import Dict, Tuple
import numpy as np


class WildfireEnvTemporal:
    """
    Environment for cell-level burn prediction with temporal context.

    State: (window_size, 14, H, W) - last N timesteps of environmental features
    Action: (H, W) - probability each cell will burn next timestep
    Reward: IoU between predicted and actual burn masks
    """
    def __init__(self, env_path: Path, window_size: int = 3, max_cache_size: int = 50):
        self.env_path = Path(env_path)
        self.window_size = window_size
        self.max_cache_size = max_cache_size

        with open(self.env_path, 'rb') as f:
            self.data = pickle.load(f)

        self.T = int(self.data['metadata']['num_timesteps'])
        self.H = int(self.data['metadata']['height'])
        self.W = int(self.data['metadata']['width'])
        self.resolution_m = int(self.data['metadata']['resolution_m'])

        # Pre-extract references
        self.static = self.data['static']
        self.fire_masks = self.data['temporal']['fire_masks']
        self.fire_intensities = self.data['temporal']['fire_intensities']
        self.fire_temps = self.data['temporal']['fire_temps']
        self.fire_ages = self.data['temporal']['fire_ages']
        self.weather_states = self.data['temporal']['weather_states']

        # Cache observations (build lazily to avoid slow init) with LRU-style eviction
        self.obs_cache = {}
        self.cache_access_order = []  # Track access order for LRU eviction
        self.t = 0

    def reset(self) -> Tuple[np.ndarray, Dict]:
        """Reset to first timestep, return observation sequence."""
        self.t = 0
        # Clear cache on reset to prevent memory buildup between episodes
        self.obs_cache.clear()
        self.cache_access_order.clear()
        obs_seq = self._get_obs_sequence(0)
        return obs_seq, {'t': 0}

    def step(self, predicted_burn_mask: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in the environment.

        Args:
            predicted_burn_mask: (H, W) binary mask of predicted burns

        Returns:
            obs_seq: Next observation sequence (window_size, 14, H, W)
            reward: IoU score between predicted and actual burns
            done: Whether episode is finished
            info: Additional info
        """
        if self.t >= self.T - 1:
            return self._get_obs_sequence(self.t), 0.0, True, {'t': self.t}

        # Compute reward based on next timestep
        reward = self._compute_reward(self.t, self.t + 1, predicted_burn_mask)

        # Move to next timestep
        self.t += 1
        done = (self.t >= self.T - 1)

        obs_seq = self._get_obs_sequence(self.t)
        return obs_seq, float(reward), done, {'t': self.t}

    def _get_obs_sequence(self, t: int) -> np.ndarray:
        """
        Get observation sequence for timestep t.

        Returns last window_size observations ending at timestep t.
        For early timesteps (t < window_size-1), pad with earliest observation.

        Args:
            t: Current timestep

        Returns:
            obs_seq: (window_size, 14, H, W) observation sequence
        """
        obs_list = []

        for i in range(self.window_size):
            # Calculate which timestep to fetch
            # For window_size=3: fetch [t-2, t-1, t]
            t_i = t - (self.window_size - 1 - i)

            # Pad with first observation if before episode start
            t_i = max(0, t_i)

            obs_i = self._get_obs(t_i)
            obs_list.append(obs_i)

        # Stack: (window_size, 14, H, W)
        obs_seq = np.stack(obs_list, axis=0).astype(np.float32)
        return obs_seq

    def _get_obs(self, t: int) -> np.ndarray:
        """Get observation for timestep t (with LRU caching)."""
        if t not in self.obs_cache:
            # Build new observation
            self.obs_cache[t] = self._build_obs(t)

            # Evict oldest entry if cache is full
            if len(self.obs_cache) > self.max_cache_size:
                # Remove least recently used entry
                oldest_t = self.cache_access_order.pop(0)
                if oldest_t in self.obs_cache:
                    del self.obs_cache[oldest_t]

        # Update access order (move to end if already present, or append)
        if t in self.cache_access_order:
            self.cache_access_order.remove(t)
        self.cache_access_order.append(t)

        return self.obs_cache[t]

    def _build_obs(self, t: int) -> np.ndarray:
        """Build observation for timestep t."""
        # Static continuous (3)
        static_cont = self.static['continuous']

        # Categorical (2)
        lcm = self.static['lcm'][None, ...].astype(np.float32)
        fsm = self.static['fsm'][None, ...].astype(np.float32)
        lcm_max = max(1, int(lcm.max()))
        fsm_max = max(1, int(fsm.max()))
        lcm = lcm / float(lcm_max)
        fsm = fsm / float(fsm_max)

        # Fire (4)
        fire_mask = self.fire_masks[t].astype(np.float32)[None, ...]
        fire_intensity = np.clip(self.fire_intensities[t] / 5.0, 0, 1).astype(np.float32)[None, ...]
        fire_temp = np.clip((self.fire_temps[t] + 20) / 60.0, 0, 1).astype(np.float32)[None, ...]
        fire_age = np.clip(self.fire_ages[t] / 1000.0, 0, 1).astype(np.float32)[None, ...]

        # Weather (5) - broadcast to spatial dimensions
        w = self.weather_states[t].astype(np.float32)
        w_normalized = np.zeros_like(w)
        w_normalized[0] = np.clip((w[0] + 10) / 50.0, 0, 1)
        w_normalized[1] = np.clip(w[1] / 100.0, 0, 1)
        w_normalized[2] = np.clip(w[2] / 20.0, 0, 1)
        w_normalized[3] = np.clip(w[3] / 20.0, 0, 1)
        w_normalized[4] = np.clip(w[4] / 50.0, 0, 1)
        weather = np.tile(w_normalized[:, None, None], (1, self.H, self.W))

        # Concatenate all features: (14, H, W)
        obs = np.concatenate([
            static_cont,      # (3, H, W)
            lcm,              # (1, H, W)
            fsm,              # (1, H, W)
            fire_mask,        # (1, H, W)
            fire_intensity,   # (1, H, W)
            fire_temp,        # (1, H, W)
            fire_age,         # (1, H, W)
            weather           # (5, H, W)
        ], axis=0).astype(np.float32)

        return obs

    def _compute_reward(self, t0: int, t1: int, predicted_burn_mask: np.ndarray) -> float:
        """
        Compute reward as IoU between predicted and actual burn masks.

        Reward encourages:
        1. Predicting burns that actually happen (true positives)
        2. Not predicting burns that don't happen (avoid false positives)
        3. Spatial precision
        """
        # Get actual burns at t1 (new burns, not cumulative)
        actual_mask_t0 = self.fire_masks[t0] > 0
        actual_mask_t1 = self.fire_masks[t1] > 0

        # New burns = cells that are on fire at t1 but weren't at t0
        new_burns = actual_mask_t1 & ~actual_mask_t0

        # Ensure predicted mask is boolean
        predicted_mask = predicted_burn_mask > 0.5

        # Compute IoU for new burns only
        intersection = (predicted_mask & new_burns).sum()
        union = (predicted_mask | new_burns).sum()

        if union == 0:
            # No predicted burns and no actual burns
            # Give small positive reward for correct "no spread" prediction
            if predicted_mask.sum() == 0:
                return 0.5
            else:
                return 0.0

        iou = intersection / float(union)

        # Return IoU directly (0 to 1 range)
        # Dense reward for RL training
        return iou
