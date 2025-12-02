"""
Dataset for AG-UNet wildfire prediction - Multi-Timestep
Loads embedded episodes and creates input-target pairs for t+1, t+2, t+3 prediction
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random


class WildfireMultiTimestepDataset(Dataset):
    """
    Dataset for multi-timestep wildfire spread prediction

    For each timestep t:
    - Input: states[t] (16 channels) + fire_mask[t] (1 channel) = 17 channels
    - Target: new burns from t→t+1, t+1→t+2, t+2→t+3 (3 binary masks)
    """

    def __init__(self, episode_files, augment=False, min_mel=4, n_timesteps=3):
        """
        Args:
            episode_files: List of paths to episode .npz files
            augment: Whether to apply data augmentation
            min_mel: Minimum MEL (timesteps - 1) to include
            n_timesteps: Number of future timesteps to predict (default 3)
        """
        self.episode_files = episode_files
        self.augment = augment
        self.min_mel = min_mel
        self.n_timesteps = n_timesteps

        # Build index of all valid timesteps
        self.samples = []
        for ep_file in episode_files:
            try:
                data = np.load(ep_file)
                states = data['states']  # (T, 16, 30, 30)
                T = len(states)

                # Check MEL
                mel = T - 1
                if mel < min_mel:
                    continue

                # Each timestep t (except last n_timesteps) is a sample
                for t in range(T - n_timesteps):
                    self.samples.append((ep_file, t))
            except Exception as e:
                print(f"Error loading {ep_file}: {e}")
                continue

        print(f"Dataset initialized with {len(self.samples)} samples from {len(episode_files)} episodes")
        print(f"Predicting {n_timesteps} future timesteps per sample")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns:
            input: (17, 30, 30) - 16 env channels + 1 fire mask at time t
            target: (n_timesteps, 30, 30) - new burns at t+1, t+2, ..., t+n_timesteps
        """
        ep_file, t = self.samples[idx]

        # Load episode
        data = np.load(ep_file)
        states = data['states']  # (T, 16, 30, 30)
        fire_masks = data['fire_masks']  # (T, 30, 30)

        # Get current state and fire
        state_t = states[t]  # (16, 30, 30)
        fire_t = fire_masks[t]  # (30, 30)

        # Stack input: 16 channels + 1 fire mask
        input_data = np.concatenate([
            state_t,  # (16, 30, 30)
            fire_t[np.newaxis, :, :]  # (1, 30, 30)
        ], axis=0)  # (17, 30, 30)

        # Compute new burns for each future timestep
        targets = []
        for dt in range(1, self.n_timesteps + 1):
            fire_prev = fire_masks[t + dt - 1]  # (30, 30)
            fire_next = fire_masks[t + dt]  # (30, 30)

            # Compute new burns
            actual_mask_prev = fire_prev > 0.5
            actual_mask_next = fire_next > 0.5
            new_burns = (actual_mask_next & ~actual_mask_prev).astype(np.float32)

            targets.append(new_burns)

        # Stack targets: (n_timesteps, 30, 30)
        target = np.stack(targets, axis=0)

        # Data augmentation
        if self.augment:
            input_data, target = self._augment(input_data, target)

        return torch.from_numpy(input_data).float(), torch.from_numpy(target).float()

    def _augment(self, input_data, target):
        """
        Random rotation and flip augmentation

        Args:
            input_data: (17, 30, 30)
            target: (n_timesteps, 30, 30)

        Returns:
            Augmented input_data, target (with positive strides)
        """
        # Random rotation (0, 90, 180, 270)
        k = random.randint(0, 3)
        if k > 0:
            input_data = np.rot90(input_data, k=k, axes=(1, 2)).copy()
            target = np.rot90(target, k=k, axes=(1, 2)).copy()

        # Random horizontal flip
        if random.random() > 0.5:
            input_data = np.flip(input_data, axis=2).copy()
            target = np.flip(target, axis=2).copy()

        return input_data, target


def get_dataloaders(data_dir, batch_size=16, num_workers=4, min_mel=4, n_timesteps=3):
    """
    Create train and validation dataloaders

    Args:
        data_dir: Directory with embedded episodes
        batch_size: Batch size
        num_workers: Number of dataloader workers
        min_mel: Minimum MEL threshold
        n_timesteps: Number of future timesteps to predict

    Returns:
        train_loader, val_loader
    """
    data_dir = Path(data_dir)
    all_episodes = sorted(data_dir.glob('episode_*.npz'))

    # Split into train/val (80/20)
    n_total = len(all_episodes)
    n_train = int(0.8 * n_total)

    train_episodes = all_episodes[:n_train]
    val_episodes = all_episodes[n_train:]

    print(f"Total episodes: {n_total}")
    print(f"Train episodes: {len(train_episodes)}")
    print(f"Val episodes: {len(val_episodes)}")

    # Create datasets
    train_dataset = WildfireMultiTimestepDataset(
        train_episodes, augment=True, min_mel=min_mel, n_timesteps=n_timesteps
    )
    val_dataset = WildfireMultiTimestepDataset(
        val_episodes, augment=False, min_mel=min_mel, n_timesteps=n_timesteps
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


def compute_iou(pred, target, threshold=0.5):
    """
    Compute IoU metric for multi-timestep prediction

    Args:
        pred: (B, n_timesteps, H, W) - probabilities
        target: (B, n_timesteps, H, W) - ground truth (0 or 1)
        threshold: Threshold for binary prediction

    Returns:
        IoU per timestep (list of scalars)
    """
    n_timesteps = pred.shape[1]
    ious = []

    for t in range(n_timesteps):
        pred_t = pred[:, t:t+1, :, :]  # (B, 1, H, W)
        target_t = target[:, t:t+1, :, :]  # (B, 1, H, W)

        pred_binary = (pred_t > threshold).float()
        target_binary = target_t

        intersection = (pred_binary * target_binary).sum()
        union = ((pred_binary + target_binary) > 0).float().sum()

        iou = (intersection + 1e-8) / (union + 1e-8)
        ious.append(iou.item())

    return ious
