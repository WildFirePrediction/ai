"""
Wildfire Environment for Spatial (Cell-Level) Prediction
Agent predicts which cells will burn at next timestep
"""
import pickle
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import torch

class WildfireEnvSpatial:
    """
    Environment for cell-level burn prediction.

    State: (14, H, W) - environmental features for all cells
    Action: (H, W) - probability each cell will burn next timestep
    Reward: IoU between predicted and actual burn masks
    """
    def __init__(self, env_path: Path):
        self.env_path = Path(env_path)
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

        # Cache observations (build lazily to avoid slow init)
        self.obs_cache = {}
        self.t = 0

    def reset(self) -> Tuple[np.ndarray, Dict]:
        """Reset to first timestep."""
        self.t = 0
        return self._get_obs(0), {'t': 0}

    def step(self, predicted_burn_mask: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in the environment.

        Args:
            predicted_burn_mask: (H, W) binary mask of predicted burns

        Returns:
            obs: Next observation
            reward: IoU score between predicted and actual burns
            done: Whether episode is finished
            info: Additional info
        """
        if self.t >= self.T - 1:
            return self._get_obs(self.t), 0.0, True, {'t': self.t}

        # Compute reward based on next timestep
        reward = self._compute_reward(self.t, self.t + 1, predicted_burn_mask)

        # Move to next timestep
        self.t += 1
        done = (self.t >= self.T - 1)

        return self._get_obs(self.t), float(reward), done, {'t': self.t}

    def _get_obs(self, t: int) -> np.ndarray:
        """Get observation for timestep t (with caching)."""
        if t not in self.obs_cache:
            self.obs_cache[t] = self._build_obs(t)
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

        # Weather (6) - broadcast to spatial dimensions
        # [temp, humidity, wind_speed, wind_x, wind_y, rainfall]
        w = self.weather_states[t].astype(np.float32)
        w_normalized = np.zeros_like(w)
        w_normalized[0] = np.clip((w[0] + 10) / 50.0, 0, 1)  # temp
        w_normalized[1] = np.clip(w[1] / 100.0, 0, 1)        # humidity
        w_normalized[2] = np.clip(w[2] / 20.0, 0, 1)         # wind_speed
        w_normalized[3] = np.clip(w[3] / 20.0, 0, 1)         # wind_x
        w_normalized[4] = np.clip(w[4] / 20.0, 0, 1)         # wind_y
        w_normalized[5] = np.clip(w[5] / 50.0, 0, 1)         # rainfall
        weather = np.tile(w_normalized[:, None, None], (1, self.H, self.W))

        # Concatenate all features: (15, H, W)
        obs = np.concatenate([
            static_cont,      # (3, H, W)
            lcm,              # (1, H, W)
            fsm,              # (1, H, W)
            fire_mask,        # (1, H, W)
            fire_intensity,   # (1, H, W)
            fire_temp,        # (1, H, W)
            fire_age,         # (1, H, W)
            weather           # (6, H, W)
        ], axis=0).astype(np.float32)

        return obs

    def _compute_reward(self, t0: int, t1: int, predicted_burn_mask: np.ndarray) -> float:
        """
        Compute reward as IoU between predicted and actual burn masks.

        BALANCED REWARD STRUCTURE:
        - Rewards correct predictions via IoU (0 to 1.0)
        - Small penalty for complete misses to encourage exploration
        - Handles sparse fire spread (most steps have very few new burns)
        
        Reward range: -0.1 to +1.0
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
        
        # Check if there are actual new burns
        has_actual_burns = new_burns.sum() > 0
        has_predictions = predicted_mask.sum() > 0

        if not has_actual_burns:
            # No actual new burns at this timestep
            if not has_predictions:
                # Correctly predicted no spread
                return 0.0
            else:
                # False alarm but small penalty (fire spread is sparse/hard)
                return 0.0
        else:
            # Actual burns exist
            if not has_predictions:
                # Missed all burns: small penalty
                return -0.1
            else:
                # Predicted some burns: reward is IoU (0 to 1.0)
                iou = intersection / float(union)
                return iou
