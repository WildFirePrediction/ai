"""
Temporal Wildfire Environment for V7.5

Wraps WildfireEnvSpatial to provide temporal observation sequences.
Memory-safe implementation with aggressive cache management.
"""
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
from collections import deque

from wildfire_env_spatial import WildfireEnvSpatial


class WildfireEnvTemporal:
    """
    Temporal environment wrapper that provides observation sequences.

    State: (T, 14, H, W) - sequence of last T timesteps
    Action: (H, W) - binary mask of predicted burns
    Reward: IoU between predicted and actual burn masks

    Memory optimizations:
    - Store observations in deque with maxlen (auto-evicts old ones)
    - Clear base environment observation cache periodically
    - Use numpy arrays (not torch tensors) to save memory
    """

    def __init__(self, env_path: Path, window_size: int = 3):
        """
        Args:
            env_path: Path to environment pickle file
            window_size: Number of timesteps in observation sequence
        """
        self.base_env = WildfireEnvSpatial(env_path)
        self.window_size = window_size

        # Use deque for automatic memory management (oldest obs auto-evicted)
        self.obs_history = deque(maxlen=window_size)

        # Store dimensions
        self.T = self.base_env.T
        self.H = self.base_env.H
        self.W = self.base_env.W
        self.resolution_m = self.base_env.resolution_m

        # For memory management
        self.cache_clear_interval = 10  # Clear cache every N steps
        self.steps_since_clear = 0

    def reset(self) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment and return initial observation sequence.

        Returns:
            obs_seq: (window_size, 14, H, W) padded sequence
            info: Environment info
        """
        # Reset base environment
        first_obs, info = self.base_env.reset()

        # Clear observation history
        self.obs_history.clear()

        # Pad with first observation (for t=0, return [obs_0, obs_0, obs_0])
        for _ in range(self.window_size):
            self.obs_history.append(first_obs.copy())

        # Clear cache
        self.steps_since_clear = 0

        return self._get_obs_sequence(), info

    def step(self, predicted_burn_mask: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in the environment.

        Args:
            predicted_burn_mask: (H, W) binary mask of predicted burns

        Returns:
            obs_seq: Next observation sequence
            reward: IoU score
            done: Whether episode is finished
            info: Additional info
        """
        # Step base environment
        next_obs, reward, done, info = self.base_env.step(predicted_burn_mask)

        # Add new observation to history (automatically evicts oldest if full)
        self.obs_history.append(next_obs.copy())

        # Periodic memory management
        self.steps_since_clear += 1
        if self.steps_since_clear >= self.cache_clear_interval:
            self._clear_caches()
            self.steps_since_clear = 0

        return self._get_obs_sequence(), float(reward), done, info

    def _get_obs_sequence(self) -> np.ndarray:
        """
        Get observation sequence from history.

        Returns:
            (window_size, 14, H, W) sequence
        """
        # Stack observations from deque
        obs_seq = np.stack(list(self.obs_history), axis=0)  # (T, 14, H, W)
        return obs_seq.astype(np.float32)

    def _clear_caches(self):
        """
        Clear observation caches to free memory.
        This is safe because we maintain our own obs_history.
        """
        # Clear base environment's observation cache
        if hasattr(self.base_env, 'obs_cache'):
            # Keep only current timestep in base env cache
            current_t = self.base_env.t
            if current_t in self.base_env.obs_cache:
                current_obs = self.base_env.obs_cache[current_t]
                self.base_env.obs_cache.clear()
                self.base_env.obs_cache[current_t] = current_obs
            else:
                self.base_env.obs_cache.clear()


if __name__ == '__main__':
    """Test temporal environment wrapper."""
    import sys
    from pathlib import Path

    print("=" * 80)
    print("Testing Temporal Environment Wrapper")
    print("=" * 80)

    # Find a test environment file
    repo_root = Path('/home/chaseungjoon/code/WildfirePrediction')
    env_dir = repo_root / 'tilling_data' / 'environments'

    # Load train split
    import json
    with open(env_dir / 'train_split.json') as f:
        train_ids = json.load(f)

    # Test with first environment
    test_env_path = env_dir / f'{train_ids[0]}.pkl'

    if not test_env_path.exists():
        print(f"Error: Test environment not found at {test_env_path}")
        sys.exit(1)

    print(f"\nLoading environment: {test_env_path.name}")

    # Create temporal environment
    env = WildfireEnvTemporal(test_env_path, window_size=3)

    print(f"Environment dimensions: {env.H} x {env.W}")
    print(f"Total timesteps: {env.T}")
    print(f"Window size: {env.window_size}")

    # Test reset
    print("\n--- Testing reset() ---")
    obs_seq, info = env.reset()
    print(f"Initial obs_seq shape: {obs_seq.shape}")
    print(f"Expected: (3, 14, {env.H}, {env.W})")
    assert obs_seq.shape == (3, 14, env.H, env.W), "Incorrect obs_seq shape!"
    print("✓ Reset test passed")

    # Test step
    print("\n--- Testing step() ---")
    for i in range(5):
        # Dummy action (all zeros)
        action = np.zeros((env.H, env.W), dtype=np.float32)

        obs_seq, reward, done, info = env.step(action)

        print(f"Step {i+1}: obs_seq shape={obs_seq.shape}, reward={reward:.4f}, done={done}")
        assert obs_seq.shape == (3, 14, env.H, env.W), f"Incorrect obs_seq shape at step {i+1}!"

        if done:
            print(f"Episode finished at step {i+1}")
            break

    print("\n--- Testing memory management ---")
    print(f"Base env cache size: {len(env.base_env.obs_cache)}")
    print(f"Obs history length: {len(env.obs_history)}")
    assert len(env.obs_history) == env.window_size, "Obs history not maintaining window size!"
    print("✓ Memory management test passed")

    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)
