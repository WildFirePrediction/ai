"""
Dataset for U-Net wildfire prediction - V3 with Dilated Ground Truth

Key Change from V2:
- Applies morphological dilation (3x3 kernel) to ground truth targets
- Allows 8-neighbor tolerance for predictions (more realistic for fire spread)
- Tracks both strict and relaxed IoU metrics
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random
from scipy.ndimage import binary_dilation


class WildfireMultiTimestepDataset(Dataset):
    """
    Dataset for multi-timestep wildfire spread prediction with dilated targets

    For each timestep t:
    - Input: states[t] (16 channels) + fire_mask[t] (1 channel) = 17 channels
    - Target (dilated): new burns from t->t+1, t+1->t+2, t+2->t+3 with 3x3 dilation
    - Target (strict): original new burns (kept for metric comparison)
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

        # 3x3 structuring element for dilation (8-neighbors)
        self.dilation_structure = np.ones((3, 3), dtype=bool)

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
        print(f"Using 3x3 dilation for 8-neighbor tolerance")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns:
            input: (17, 30, 30) - 16 env channels + 1 fire mask at time t
            target_dilated: (n_timesteps, 30, 30) - dilated new burns (for training)
            target_strict: (n_timesteps, 30, 30) - original new burns (for metrics)
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
        targets_strict = []
        targets_dilated = []

        for dt in range(1, self.n_timesteps + 1):
            fire_prev = fire_masks[t + dt - 1]  # (30, 30)
            fire_next = fire_masks[t + dt]  # (30, 30)

            # Compute strict new burns
            actual_mask_prev = fire_prev > 0.5
            actual_mask_next = fire_next > 0.5
            new_burns_strict = (actual_mask_next & ~actual_mask_prev).astype(np.float32)

            # Apply dilation (3x3 kernel for 8-neighbor tolerance)
            new_burns_dilated = binary_dilation(
                new_burns_strict > 0.5,
                structure=self.dilation_structure
            ).astype(np.float32)

            targets_strict.append(new_burns_strict)
            targets_dilated.append(new_burns_dilated)

        # Stack targets: (n_timesteps, 30, 30)
        target_strict = np.stack(targets_strict, axis=0)
        target_dilated = np.stack(targets_dilated, axis=0)

        # Data augmentation (apply to both dilated and strict)
        if self.augment:
            input_data, target_dilated, target_strict = self._augment(
                input_data, target_dilated, target_strict
            )

        return (
            torch.from_numpy(input_data).float(),
            torch.from_numpy(target_dilated).float(),
            torch.from_numpy(target_strict).float()
        )

    def _augment(self, input_data, target_dilated, target_strict):
        """
        Random rotation and flip augmentation

        Args:
            input_data: (17, 30, 30)
            target_dilated: (n_timesteps, 30, 30)
            target_strict: (n_timesteps, 30, 30)

        Returns:
            Augmented input_data, target_dilated, target_strict (with positive strides)
        """
        # Random rotation (0, 90, 180, 270)
        k = random.randint(0, 3)
        if k > 0:
            input_data = np.rot90(input_data, k=k, axes=(1, 2)).copy()
            target_dilated = np.rot90(target_dilated, k=k, axes=(1, 2)).copy()
            target_strict = np.rot90(target_strict, k=k, axes=(1, 2)).copy()

        # Random horizontal flip
        if random.random() > 0.5:
            input_data = np.flip(input_data, axis=2).copy()
            target_dilated = np.flip(target_dilated, axis=2).copy()
            target_strict = np.flip(target_strict, axis=2).copy()

        return input_data, target_dilated, target_strict


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
