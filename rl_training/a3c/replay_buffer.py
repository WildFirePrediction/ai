"""
Experience Replay Buffer for A3C V6

Store and replay high-quality episodes to prevent catastrophic forgetting.
"""
import numpy as np
from pathlib import Path
from collections import deque
import threading


class EpisodeReplayBuffer:
    """
    Replay buffer for storing and sampling high-quality fire episodes.

    Stores (env_path, start_t, max_length, IoU) tuples for best episodes.
    """

    def __init__(self, capacity=100, min_iou_threshold=0.15):
        """
        Args:
            capacity: Maximum number of episodes to store
            min_iou_threshold: Minimum IoU to consider storing episode
        """
        self.capacity = capacity
        self.min_iou_threshold = min_iou_threshold
        self.buffer = []  # List of (env_path, start_t, max_length, iou)
        self.lock = threading.Lock()

        self.total_added = 0
        self.total_rejected = 0

    def add(self, env_path, start_t, max_length, episode_iou):
        """
        Add episode to buffer if it's good enough.

        Args:
            env_path: Path to environment file
            start_t: Starting timestep
            max_length: Maximum episode length
            episode_iou: Average IoU achieved in episode
        """
        with self.lock:
            # Reject if below threshold
            if episode_iou < self.min_iou_threshold:
                self.total_rejected += 1
                return False

            # Create episode entry
            episode_entry = {
                'env_path': env_path,
                'start_t': start_t,
                'max_length': max_length,
                'iou': episode_iou,
                'count': 0  # Number of times sampled
            }

            # If buffer not full, just add
            if len(self.buffer) < self.capacity:
                self.buffer.append(episode_entry)
                self.total_added += 1
                return True

            # Buffer is full - replace worst episode if this is better
            worst_idx = min(range(len(self.buffer)), key=lambda i: self.buffer[i]['iou'])
            worst_iou = self.buffer[worst_idx]['iou']

            if episode_iou > worst_iou:
                self.buffer[worst_idx] = episode_entry
                self.total_added += 1
                return True
            else:
                self.total_rejected += 1
                return False

    def sample(self, batch_size=1, strategy='uniform'):
        """
        Sample episodes from buffer.

        Args:
            batch_size: Number of episodes to sample
            strategy: Sampling strategy
                - 'uniform': Uniform random sampling
                - 'prioritized': Sample high-IoU episodes more frequently
                - 'least_used': Sample least-used episodes more frequently

        Returns:
            List of (env_path, start_t, max_length) tuples
        """
        with self.lock:
            if len(self.buffer) == 0:
                return []

            batch_size = min(batch_size, len(self.buffer))

            if strategy == 'uniform':
                indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)

            elif strategy == 'prioritized':
                # Probability proportional to IoU
                ious = np.array([ep['iou'] for ep in self.buffer])
                probs = ious / ious.sum()
                indices = np.random.choice(len(self.buffer), size=batch_size, replace=False, p=probs)

            elif strategy == 'least_used':
                # Probability inversely proportional to usage count
                counts = np.array([ep['count'] + 1 for ep in self.buffer])
                probs = (1.0 / counts)
                probs = probs / probs.sum()
                indices = np.random.choice(len(self.buffer), size=batch_size, replace=False, p=probs)

            else:
                raise ValueError(f"Unknown sampling strategy: {strategy}")

            # Increment usage count and return episodes
            sampled_episodes = []
            for idx in indices:
                self.buffer[idx]['count'] += 1
                sampled_episodes.append((
                    self.buffer[idx]['env_path'],
                    self.buffer[idx]['start_t'],
                    self.buffer[idx]['max_length']
                ))

            return sampled_episodes

    def get_stats(self):
        """Get buffer statistics."""
        with self.lock:
            if len(self.buffer) == 0:
                return {
                    'size': 0,
                    'capacity': self.capacity,
                    'total_added': self.total_added,
                    'total_rejected': self.total_rejected,
                    'mean_iou': 0.0,
                    'max_iou': 0.0,
                    'min_iou': 0.0
                }

            ious = [ep['iou'] for ep in self.buffer]
            return {
                'size': len(self.buffer),
                'capacity': self.capacity,
                'total_added': self.total_added,
                'total_rejected': self.total_rejected,
                'mean_iou': np.mean(ious),
                'max_iou': np.max(ious),
                'min_iou': np.min(ious),
                'std_iou': np.std(ious)
            }

    def is_empty(self):
        """Check if buffer is empty."""
        with self.lock:
            return len(self.buffer) == 0

    def get_best_episode(self):
        """Get the best episode in the buffer."""
        with self.lock:
            if len(self.buffer) == 0:
                return None

            best_idx = max(range(len(self.buffer)), key=lambda i: self.buffer[i]['iou'])
            return (
                self.buffer[best_idx]['env_path'],
                self.buffer[best_idx]['start_t'],
                self.buffer[best_idx]['max_length'],
                self.buffer[best_idx]['iou']
            )


# Testing
if __name__ == '__main__':
    print("Testing EpisodeReplayBuffer...")

    buffer = EpisodeReplayBuffer(capacity=5, min_iou_threshold=0.1)

    # Add some episodes
    for i in range(10):
        env_path = Path(f"env_{i}.pkl")
        iou = np.random.rand() * 0.5  # Random IoU between 0-0.5
        added = buffer.add(env_path, start_t=0, max_length=10, episode_iou=iou)
        print(f"Episode {i}: IoU={iou:.3f}, Added={added}")

    # Get stats
    stats = buffer.get_stats()
    print(f"\nBuffer stats: {stats}")

    # Sample episodes
    print(f"\nSampling 3 episodes (uniform):")
    sampled = buffer.sample(batch_size=3, strategy='uniform')
    for env_path, start_t, max_length in sampled:
        print(f"  {env_path}, start_t={start_t}, max_length={max_length}")

    print(f"\nSampling 3 episodes (prioritized):")
    sampled = buffer.sample(batch_size=3, strategy='prioritized')
    for env_path, start_t, max_length in sampled:
        print(f"  {env_path}, start_t={start_t}, max_length={max_length}")

    # Get best episode
    best = buffer.get_best_episode()
    if best:
        print(f"\nBest episode: {best[0]}, IoU={best[3]:.3f}")

    print("\nAll tests passed!")
