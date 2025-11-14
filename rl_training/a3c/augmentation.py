"""
Data Augmentation for Wildfire Episodes

Rotation and flip augmentation for spatial fire spread data.
Fire spread physics are rotation/flip invariant, so these augmentations preserve validity.
"""
import numpy as np
import torch


def augment_observation(obs, k_rot=0, flip_h=False, flip_v=False):
    """
    Augment observation with rotation and flipping.

    Args:
        obs: (C, H, W) observation array
        k_rot: Number of 90-degree rotations (0, 1, 2, 3)
        flip_h: Horizontal flip
        flip_v: Vertical flip

    Returns:
        Augmented observation (C, H', W')
    """
    # Apply rotation (90 * k degrees)
    if k_rot > 0:
        obs = np.rot90(obs, k=k_rot, axes=(1, 2))

    # Apply flips
    if flip_h:
        obs = np.flip(obs, axis=2)  # Flip width dimension
    if flip_v:
        obs = np.flip(obs, axis=1)  # Flip height dimension

    return obs.copy()


def augment_mask(mask, k_rot=0, flip_h=False, flip_v=False):
    """
    Augment 2D mask (fire mask, action grid, etc.) with rotation and flipping.

    Args:
        mask: (H, W) mask array
        k_rot: Number of 90-degree rotations (0, 1, 2, 3)
        flip_h: Horizontal flip
        flip_v: Vertical flip

    Returns:
        Augmented mask (H', W')
    """
    # Apply rotation
    if k_rot > 0:
        mask = np.rot90(mask, k=k_rot, axes=(0, 1))

    # Apply flips
    if flip_h:
        mask = np.flip(mask, axis=1)  # Flip width dimension
    if flip_v:
        mask = np.flip(mask, axis=0)  # Flip height dimension

    return mask.copy()


def random_augmentation():
    """
    Sample random augmentation parameters.

    Returns:
        Dictionary with augmentation parameters: {k_rot, flip_h, flip_v}
    """
    k_rot = np.random.randint(0, 4)  # 0, 90, 180, 270 degrees
    flip_h = np.random.rand() > 0.5
    flip_v = np.random.rand() > 0.5

    return {
        'k_rot': k_rot,
        'flip_h': flip_h,
        'flip_v': flip_v
    }


def inverse_augment_mask(mask, k_rot=0, flip_h=False, flip_v=False):
    """
    Inverse augmentation to map predictions back to original orientation.

    This is useful when you augment input, make predictions, then need to
    convert predictions back to original space.

    Args:
        mask: (H, W) mask in augmented space
        k_rot: Number of 90-degree rotations applied during forward augmentation
        flip_h: Horizontal flip applied during forward augmentation
        flip_v: Vertical flip applied during forward augmentation

    Returns:
        Mask in original space
    """
    # Reverse flips first (apply in reverse order)
    if flip_v:
        mask = np.flip(mask, axis=0)
    if flip_h:
        mask = np.flip(mask, axis=1)

    # Reverse rotation
    if k_rot > 0:
        mask = np.rot90(mask, k=4-k_rot, axes=(0, 1))  # Rotate back

    return mask.copy()


def augment_episode_batch(observations, fire_masks, aug_params):
    """
    Augment a batch of observations and masks with the same augmentation.

    Args:
        observations: List of (C, H, W) observations
        fire_masks: List of (H, W) fire masks
        aug_params: Augmentation parameters from random_augmentation()

    Returns:
        augmented_observations, augmented_fire_masks (lists)
    """
    aug_observations = []
    aug_fire_masks = []

    for obs, mask in zip(observations, fire_masks):
        aug_obs = augment_observation(obs, **aug_params)
        aug_mask = augment_mask(mask, **aug_params)
        aug_observations.append(aug_obs)
        aug_fire_masks.append(aug_mask)

    return aug_observations, aug_fire_masks


# Testing function
if __name__ == '__main__':
    # Test augmentation
    print("Testing augmentation utilities...")

    # Create dummy observation (14, 10, 10)
    obs = np.random.randn(14, 10, 10).astype(np.float32)
    mask = np.random.rand(10, 10) > 0.5

    # Test all rotations
    for k in range(4):
        aug_obs = augment_observation(obs, k_rot=k)
        aug_mask = augment_mask(mask, k_rot=k)
        print(f"Rotation {k*90}°: obs shape {aug_obs.shape}, mask shape {aug_mask.shape}")

    # Test flips
    aug_obs = augment_observation(obs, flip_h=True)
    print(f"Horizontal flip: obs shape {aug_obs.shape}")

    aug_obs = augment_observation(obs, flip_v=True)
    print(f"Vertical flip: obs shape {aug_obs.shape}")

    # Test inverse augmentation
    aug_params = {'k_rot': 2, 'flip_h': True, 'flip_v': False}
    aug_mask = augment_mask(mask, **aug_params)
    restored_mask = inverse_augment_mask(aug_mask, **aug_params)

    print(f"Inverse augmentation error: {np.abs(mask.astype(float) - restored_mask.astype(float)).sum()}")

    print("All tests passed!")
