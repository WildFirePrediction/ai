"""
Temporal Wildfire Environment V3.5 - Architecture Plan Implementation

Following V3.5_ARCHITECTURE_PLAN.md specifications:
1. Temporal window = 5 timesteps (sufficient for fire velocity/acceleration patterns)
2. Memory-optimized observation building (no redundant copies)
3. Simple IoU reward (dense feedback at every timestep)
4. Limited observation cache to prevent memory bloat
5. Proper cleanup of loaded data

Memory profile (grid 347×347, window=5):
- Environment data: ~3GB
- Observation cache (window+2): ~200MB
- Total per environment: ~3.2GB (safe for 2 workers)
"""
import pickle
from pathlib import Path
from typing import Dict, Tuple
import numpy as np


class WildfireEnvTemporal:
    """
    Temporal environment for wildfire prediction with LSTM context.

    Key features:
    - Returns last 5 timesteps as observation sequences
    - Dense IoU rewards (feedback at every step)
    - Memory-optimized with limited caching
    - No fire tracking (keeps things simple and fast)
    """

    def __init__(self, env_path: Path, temporal_window=5):
        self.env_path = Path(env_path)
        self.temporal_window = temporal_window

        with open(self.env_path, 'rb') as f:
            data = pickle.load(f)

        self.T = int(data['metadata']['num_timesteps'])
        self.H = int(data['metadata']['height'])
        self.W = int(data['metadata']['width'])
        self.resolution_m = int(data['metadata']['resolution_m'])

        # CRITICAL FIX: Make explicit copies and delete original data dict
        self.static = {
            'continuous': data['static']['continuous'].copy(),
            'lcm': data['static']['lcm'].copy(),
            'fsm': data['static']['fsm'].copy()
        }
        self.fire_masks = data['temporal']['fire_masks'].copy()
        self.fire_intensities = data['temporal']['fire_intensities'].copy()
        self.fire_temps = data['temporal']['fire_temps'].copy()
        self.fire_ages = data['temporal']['fire_ages'].copy()
        self.weather_states = data['temporal']['weather_states'].copy()

        # CRITICAL FIX: Delete data dict to free memory
        del data

        # Observation cache (limited to temporal_window + 2 for safety)
        self.obs_cache = {}
        self.max_cache_size = temporal_window + 2
        self.t = 0

    def reset(self) -> Tuple[np.ndarray, Dict]:
        """Reset to first timestep."""
        self.t = 0

        # Clear cache completely
        self.obs_cache.clear()

        obs_seq = self._get_obs_sequence(0)
        return obs_seq, {'t': 0}

    def step(self, predicted_burn_mask: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Take a step with simple IoU reward."""
        if self.t >= self.T - 1:
            return self._get_obs_sequence(self.t), 0.0, True, {'t': self.t}

        # Compute simple IoU reward
        reward = self._compute_reward(self.t, self.t + 1, predicted_burn_mask)

        # Move to next timestep
        self.t += 1

        done = (self.t >= self.T - 1)
        info = {'t': self.t}

        return self._get_obs_sequence(self.t), float(reward), done, info

    def _compute_reward(self, t_current, t_next, predicted_burn_mask):
        """
        Compute simple IoU reward.

        No fire tracking, no temporal bonus - just pure IoU.
        """
        # Ensure boolean numpy array
        if not isinstance(predicted_burn_mask, np.ndarray):
            predicted_burn_mask = np.array(predicted_burn_mask, dtype=bool)
        else:
            predicted_burn_mask = predicted_burn_mask.astype(bool)

        actual_mask_t = (self.fire_masks[t_current] > 0).astype(bool)
        actual_mask_t1 = (self.fire_masks[t_next] > 0).astype(bool)
        new_burns = actual_mask_t1 & ~actual_mask_t

        if not new_burns.any():
            return 0.0

        # Simple IoU
        intersection = (predicted_burn_mask & new_burns).sum()
        union = (predicted_burn_mask | new_burns).sum()
        iou = float(intersection) / float(union + 1e-8)

        return iou

    def _get_obs_sequence(self, t: int) -> np.ndarray:
        """
        Get temporal sequence of observations efficiently.

        FIXED: Pre-allocate array and fill without creating intermediate lists.
        Returns last temporal_window timesteps.
        """
        # Pre-allocate output array (15 channels: 3 static + 2 categorical + 4 fire + 6 weather)
        obs_seq = np.zeros((self.temporal_window, 15, self.H, self.W), dtype=np.float32)

        for i in range(self.temporal_window):
            t_i = t - (self.temporal_window - 1 - i)
            if t_i < 0:
                t_i = 0  # Pad with first observation

            obs_seq[i] = self._get_obs(t_i)

        return obs_seq

    def _get_obs(self, t: int) -> np.ndarray:
        """Get observation for timestep t with limited caching."""
        if t not in self.obs_cache:
            # Enforce strict cache limit
            if len(self.obs_cache) >= self.max_cache_size:
                # Remove oldest entry
                oldest_key = min(self.obs_cache.keys())
                del self.obs_cache[oldest_key]

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
        # Fire brightness temp in Kelvin (280-600K range): normalize to [0, 1]
        fire_temp = np.clip((self.fire_temps[t] - 280) / 320.0, 0, 1).astype(np.float32)[None, ...]
        fire_age = np.clip(self.fire_ages[t] / 1000.0, 0, 1).astype(np.float32)[None, ...]

        # Weather (6): [temp, humidity, wind_speed, wind_x, wind_y, rainfall]
        w = self.weather_states[t].astype(np.float32)
        w_normalized = np.zeros_like(w)
        w_normalized[0] = np.clip((w[0] + 10) / 50.0, 0, 1)     # Temp: -10°C to 40°C
        w_normalized[1] = np.clip(w[1] / 100.0, 0, 1)           # Humidity: 0-100%
        w_normalized[2] = np.clip(w[2] / 50.0, 0, 1)            # Wind speed: 0-50 m/s
        w_normalized[3] = np.clip((w[3] + 1) / 2.0, 0, 1)       # Wind X: -1 to 1
        w_normalized[4] = np.clip((w[4] + 1) / 2.0, 0, 1)       # Wind Y: -1 to 1
        w_normalized[5] = np.clip(w[5] / 100.0, 0, 1)           # Rainfall: 0-100 mm

        weather_spatial = np.tile(w_normalized[:, None, None], (1, self.H, self.W))

        # Concatenate: (15, H, W)
        obs = np.concatenate([
            static_cont,      # 0-2: elevation, slope, aspect
            lcm,              # 3: land cover
            fsm,              # 4: fuel spatial model
            fire_mask,        # 5: fire mask
            fire_intensity,   # 6: fire intensity
            fire_temp,        # 7: fire temperature (BRIGHTNESS from VIIRS)
            fire_age,         # 8: fire age
            weather_spatial,  # 9-14: weather (temp, humidity, wind_speed, wind_x, wind_y, rainfall)
        ], axis=0).astype(np.float32)

        return obs
