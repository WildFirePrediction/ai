"""
Temporal Wildfire Environment for V3.5

Key features:
1. Temporal window: returns last N timesteps of observations
2. Fire instance tracking: tracks individual fires through time
3. Cumulative temporal reward: rewards increase for predicting further steps
"""
import pickle
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
from scipy import ndimage


class WildfireEnvTemporal:
    """
    Temporal environment with fire instance tracking.

    Innovations:
    - Returns temporal window (last N obs) instead of single obs
    - Tracks fire instances across timesteps
    - Cumulative reward: more reward for predicting later steps of same fire
    """

    def __init__(self, env_path: Path, temporal_window=5):
        self.env_path = Path(env_path)
        self.temporal_window = temporal_window

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

        # Cache observations
        self.obs_cache = {}
        self.t = 0

        # Fire instance tracking
        self.fire_instances = {}  # {instance_id: {'start_t': int, 'steps': int, 'mask': np.array}}
        self.fire_id_map = None  # (H, W) - current fire instance ID for each cell
        self.next_fire_id = 1

    def reset(self) -> Tuple[np.ndarray, Dict]:
        """Reset to first timestep and initialize fire tracking."""
        self.t = 0
        self.fire_instances = {}
        self.fire_id_map = np.zeros((self.H, self.W), dtype=np.int32)
        self.next_fire_id = 1

        # CRITICAL: Clear observation cache to prevent memory leak
        self.obs_cache.clear()

        # Initialize fire instances at t=0
        self._update_fire_instances()

        obs_seq = self._get_obs_sequence(0)
        return obs_seq, {'t': 0, 'fire_instances': len(self.fire_instances)}

    def step(self, predicted_burn_mask: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step with cumulative temporal reward.

        Reward increases for predicting further steps of the same fire instance.
        """
        if self.t >= self.T - 1:
            return self._get_obs_sequence(self.t), 0.0, True, {'t': self.t}

        # Compute reward with temporal bonus
        reward = self._compute_temporal_reward(self.t, self.t + 1, predicted_burn_mask)

        # Move to next timestep
        self.t += 1
        self._update_fire_instances()

        done = (self.t >= self.T - 1)

        info = {
            't': self.t,
            'fire_instances': len(self.fire_instances),
            'active_fires': np.sum(self.fire_id_map > 0)
        }

        return self._get_obs_sequence(self.t), float(reward), done, info

    def _update_fire_instances(self):
        """
        Update fire instance tracking at current timestep.

        Uses connected component analysis to identify separate fires.
        Tracks each fire through time and counts steps.
        """
        current_fire_mask = self.fire_masks[self.t] > 0

        if not current_fire_mask.any():
            # No fires burning
            self.fire_id_map = np.zeros((self.H, self.W), dtype=np.int32)
            return

        # Label connected components (8-connectivity)
        labeled_fires, num_fires = ndimage.label(current_fire_mask, structure=np.ones((3, 3)))

        # Create new fire ID map
        new_fire_id_map = np.zeros((self.H, self.W), dtype=np.int32)

        for component_id in range(1, num_fires + 1):
            component_mask = (labeled_fires == component_id)

            # Check if this fire instance existed before
            # Look for overlap with previous fire_id_map
            if self.t > 0:
                overlap_ids = self.fire_id_map[component_mask]
                overlap_ids = overlap_ids[overlap_ids > 0]

                if len(overlap_ids) > 0:
                    # This fire existed before - use most common previous ID
                    fire_id = np.bincount(overlap_ids).argmax()

                    # Increment step count
                    if fire_id in self.fire_instances:
                        self.fire_instances[fire_id]['steps'] += 1
                        # Don't store mask - saves memory
                    else:
                        # Shouldn't happen, but create new instance
                        fire_id = self.next_fire_id
                        self.next_fire_id += 1
                        self.fire_instances[fire_id] = {
                            'start_t': self.t,
                            'steps': 1,
                        }
                else:
                    # New fire instance
                    fire_id = self.next_fire_id
                    self.next_fire_id += 1
                    self.fire_instances[fire_id] = {
                        'start_t': self.t,
                        'steps': 1,
                    }
            else:
                # t=0, all fires are new
                fire_id = self.next_fire_id
                self.next_fire_id += 1
                self.fire_instances[fire_id] = {
                    'start_t': self.t,
                    'steps': 1,
                }

            # Assign fire ID to this component
            new_fire_id_map[component_mask] = fire_id

        self.fire_id_map = new_fire_id_map

    def _compute_temporal_reward(self, t_current, t_next, predicted_burn_mask):
        """
        Compute reward with temporal bonus.

        Base reward: IoU between predicted and actual new burns
        Temporal bonus: Multiply by (1 + 0.3 * fire_steps) for each fire instance

        This encourages the model to:
        1. Predict fire spread accurately (IoU)
        2. Maintain temporal consistency (bonus for later steps)
        """
        # Ensure predicted_burn_mask is boolean numpy array
        if not isinstance(predicted_burn_mask, np.ndarray):
            predicted_burn_mask = np.array(predicted_burn_mask, dtype=bool)
        else:
            predicted_burn_mask = predicted_burn_mask.astype(bool)

        actual_mask_t = (self.fire_masks[t_current] > 0).astype(bool)
        actual_mask_t1 = (self.fire_masks[t_next] > 0).astype(bool)
        new_burns = actual_mask_t1 & ~actual_mask_t

        if not new_burns.any():
            return 0.0

        # Base IoU reward
        intersection = (predicted_burn_mask & new_burns).sum()
        union = (predicted_burn_mask | new_burns).sum()
        base_iou = float(intersection) / float(union + 1e-8)

        # Temporal bonus based on fire instance steps
        # For each predicted cell, check which fire instance it belongs to
        # Weight the reward by how many steps that fire has been burning
        total_weight = 0.0
        weighted_reward = 0.0

        # Get fire IDs for cells that actually burned
        fire_ids_in_new_burns = self.fire_id_map[new_burns]
        fire_ids_in_new_burns = fire_ids_in_new_burns[fire_ids_in_new_burns > 0]

        if len(fire_ids_in_new_burns) > 0:
            # Compute average temporal multiplier
            temporal_multipliers = []
            for fire_id in np.unique(fire_ids_in_new_burns):
                if fire_id in self.fire_instances:
                    steps = self.fire_instances[fire_id]['steps']
                    # Multiplier increases with steps: 1.0, 1.3, 1.6, 1.9, ...
                    multiplier = 1.0 + 0.3 * min(steps - 1, 5)  # Cap at 5 steps
                    temporal_multipliers.append(multiplier)

            if temporal_multipliers:
                avg_multiplier = np.mean(temporal_multipliers)
                reward = base_iou * avg_multiplier
            else:
                reward = base_iou
        else:
            reward = base_iou

        return reward

    def _get_obs_sequence(self, t: int) -> np.ndarray:
        """
        Get temporal sequence of observations.

        Returns last temporal_window timesteps.
        For early timesteps (t < window), repeat first observation.

        Returns:
            obs_seq: (temporal_window, 14, H, W)
        """
        obs_list = []

        for i in range(self.temporal_window):
            t_i = t - (self.temporal_window - 1 - i)
            if t_i < 0:
                # Pad with first observation
                t_i = 0
            obs_i = self._get_obs(t_i)
            obs_list.append(obs_i)

        return np.stack(obs_list, axis=0)  # (temporal_window, 14, H, W)

    def _get_obs(self, t: int) -> np.ndarray:
        """Get observation for timestep t (with limited caching)."""
        if t not in self.obs_cache:
            # Limit cache size to prevent memory leak
            if len(self.obs_cache) > 10:
                # Remove oldest cached observation
                oldest_key = min(self.obs_cache.keys())
                del self.obs_cache[oldest_key]
            self.obs_cache[t] = self._build_obs(t)
        return self.obs_cache[t]

    def _build_obs(self, t: int) -> np.ndarray:
        """Build observation for timestep t (same as V3)."""
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

        # Weather (5)
        w = self.weather_states[t].astype(np.float32)
        w_normalized = np.zeros_like(w)
        w_normalized[0] = np.clip((w[0] + 10) / 50.0, 0, 1)
        w_normalized[1] = np.clip(w[1] / 100.0, 0, 1)
        w_normalized[2] = np.clip(w[2] / 50.0, 0, 1)
        w_normalized[3] = np.clip(w[3] / 360.0, 0, 1)
        w_normalized[4] = np.clip(w[4] / 100.0, 0, 1)

        weather_spatial = np.tile(w_normalized[:, None, None], (1, self.H, self.W))

        # Concatenate: (14, H, W)
        obs = np.concatenate([
            static_cont,      # 0-2: elevation, slope, aspect
            lcm,              # 3: land cover
            fsm,              # 4: fuel spatial model
            fire_mask,        # 5: fire mask
            fire_intensity,   # 6: fire intensity
            fire_temp,        # 7: fire temperature
            fire_age,         # 8: fire age
            weather_spatial,  # 9-13: weather (temp, humidity, wind_speed, wind_dir, precip)
        ], axis=0).astype(np.float32)

        return obs

    def _compute_reward(self, t_current, t_next, predicted_burn_mask):
        """Legacy method for compatibility."""
        return self._compute_temporal_reward(t_current, t_next, predicted_burn_mask)
