"""
A3C Model V3.5 - Temporal Context with Fire Instance Tracking

Key improvements over V3:
1. Temporal window: sees last 5 timesteps
2. 3D convolutions for efficient temporal processing (velocity, acceleration, wind dynamics)
3. Keeps V3's proven 128-channel encoder (learned from V7 failure)
4. Group normalization for training stability
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class A3C_TemporalModel(nn.Module):
    """
    A3C model with temporal context for fire spread prediction.

    Uses 3D convolutions to efficiently process temporal sequences.
    Learns:
    - Fire velocity and acceleration
    - Wind pattern dynamics
    - Temporal trends in spread
    """

    def __init__(self, in_channels=14, temporal_window=5, hidden_dim=128):
        super().__init__()

        self.in_channels = in_channels
        self.temporal_window = temporal_window
        self.hidden_dim = hidden_dim

        # KEEP V3's FULL ENCODER (128 channels - don't reduce like V7!)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),  # CRITICAL: Keep 128 channels!
            nn.ReLU(),
        )

        # Temporal processing with 3D convolutions (MUCH more efficient than LSTM per pixel!)
        # Processes temporal + spatial dimensions together
        self.temporal_conv = nn.Sequential(
            nn.Conv3d(128, hidden_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
        )

        # Layer norm for stability (applied on channel dimension)
        self.layer_norm = nn.GroupNorm(8, hidden_dim)  # Group norm instead of LayerNorm

        # Policy head (same as V3) - predicts 8-neighbor spread per burning cell
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim * 9, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 8)  # 8 neighbors
        )

        # Value head (same as V3) - global state value
        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def encode_temporal_sequence(self, obs_seq):
        """
        Encode temporal sequence of observations with 3D convolutions.

        Args:
            obs_seq: (B, T, C, H, W) - last T timesteps

        Returns:
            temporal_features: (B, hidden_dim, H, W) - temporally-aware features
        """
        B, T, C, H, W = obs_seq.shape

        # Encode each timestep independently with V3's encoder
        features_seq = []
        for t in range(T):
            feat_t = self.encoder(obs_seq[:, t])  # (B, 128, H, W)
            features_seq.append(feat_t)
        features_seq = torch.stack(features_seq, dim=2)  # (B, 128, T, H, W)

        # Apply 3D convolutions to process temporal + spatial
        # Conv3d expects (B, C, D, H, W) where D is temporal dimension
        temporal_features = self.temporal_conv(features_seq)  # (B, hidden_dim, T, H, W)

        # Take last temporal slice (current timestep with temporal context)
        temporal_features = temporal_features[:, :, -1, :, :]  # (B, hidden_dim, H, W)

        # Apply normalization
        temporal_features = self.layer_norm(temporal_features)

        return temporal_features

    def forward(self, obs_seq, fire_mask):
        """
        Forward pass with temporal sequence.

        Args:
            obs_seq: (B, T, C, H, W) - temporal observation sequence
            fire_mask: (B, H, W) - current fire mask (last timestep)

        Returns:
            features: (B, hidden_dim, H, W) - temporal features
            value: (B, 1) - state value
        """
        # Encode temporal sequence
        features = self.encode_temporal_sequence(obs_seq)  # (B, hidden_dim, H, W)

        # Compute value
        value = self.value_head(features)  # (B, 1)

        return features, value

    def get_burning_cells(self, fire_mask):
        """Extract locations of all burning cells."""
        if fire_mask.dim() == 3:
            fire_mask = fire_mask[0]
        burning_indices = torch.nonzero(fire_mask > 0.5, as_tuple=False)
        return [(int(idx[0]), int(idx[1])) for idx in burning_indices]

    def extract_local_features(self, features, i, j):
        """Extract 3x3 local features around cell (i, j)."""
        B, C, H, W = features.shape
        padded_features = F.pad(features, (1, 1, 1, 1), mode='constant', value=0)
        local = padded_features[:, :, i:i+3, j:j+3]  # (B, C, 3, 3)
        local_flat = local.flatten(1)  # (B, C*9)
        return local_flat

    def predict_8_neighbors(self, features, i, j):
        """Predict 8-neighbor spread for a single burning cell."""
        local_features = self.extract_local_features(features, i, j)
        logits = self.policy_head(local_features)  # (B, 8)
        return logits

    def get_8_neighbor_coords(self, i, j, H, W):
        """Get coordinates of 8 neighbors (N, NE, E, SE, S, SW, W, NW)."""
        neighbors = [
            (i-1, j), (i-1, j+1), (i, j+1), (i+1, j+1),
            (i+1, j), (i+1, j-1), (i, j-1), (i-1, j-1),
        ]
        valid_neighbors = []
        for ni, nj in neighbors:
            if 0 <= ni < H and 0 <= nj < W:
                valid_neighbors.append((ni, nj))
            else:
                valid_neighbors.append(None)
        return valid_neighbors

    def get_action_and_value(self, obs_seq, fire_mask, action=None):
        """
        Get actions for all burning cells and compute value.

        Args:
            obs_seq: (1, T, C, H, W) - temporal sequence (batch=1)
            fire_mask: (1, H, W) or (H, W) - current fire mask
            action: Optional pre-specified action

        Returns:
            action_grid: (H, W) binary prediction grid
            log_prob: Scalar log probability
            entropy: Scalar entropy
            value: (1, 1) state value
            burning_cells_info: Debug info
        """
        if fire_mask.dim() == 2:
            fire_mask = fire_mask.unsqueeze(0)

        B, H, W = fire_mask.shape
        assert B == 1, "Batch size must be 1"

        # Get temporal features and value
        features, value = self.forward(obs_seq, fire_mask)  # (1, hidden_dim, H, W), (1, 1)

        # Find burning cells
        burning_cells = self.get_burning_cells(fire_mask)

        if len(burning_cells) == 0:
            action_grid = torch.zeros(H, W)
            log_prob = torch.tensor(0.0)
            entropy = torch.tensor(0.0)
            return action_grid, log_prob, entropy, value, []

        # Predict for each burning cell
        all_log_probs = []
        all_entropies = []
        burning_cells_info = []
        action_grid = torch.zeros(H, W)

        for i, j in burning_cells:
            # Get 8-neighbor logits
            logits_8d = self.predict_8_neighbors(features, i, j).squeeze(0)  # (8,)

            # Get probabilities
            probs_8d = torch.sigmoid(logits_8d)
            probs_8d = torch.clamp(probs_8d, 1e-7, 1 - 1e-7)

            # Sample action
            if action is None:
                action_8d = torch.bernoulli(probs_8d)
            else:
                action_8d = action[len(burning_cells_info)] if isinstance(action, list) else torch.bernoulli(probs_8d)

            # Compute log probability
            log_prob_8d = (action_8d * torch.log(probs_8d) +
                          (1 - action_8d) * torch.log(1 - probs_8d))
            log_prob_cell = log_prob_8d.sum()

            # Compute entropy
            entropy_8d = -(probs_8d * torch.log(probs_8d) +
                          (1 - probs_8d) * torch.log(1 - probs_8d))
            entropy_cell = entropy_8d.sum()

            all_log_probs.append(log_prob_cell)
            all_entropies.append(entropy_cell)

            # Map action to grid
            neighbors = self.get_8_neighbor_coords(i, j, H, W)
            for n_idx, neighbor in enumerate(neighbors):
                if neighbor is not None and action_8d[n_idx] > 0.5:
                    ni, nj = neighbor
                    action_grid[ni, nj] = 1.0

            burning_cells_info.append((i, j, action_8d, log_prob_cell))

        # Aggregate
        total_log_prob = torch.stack(all_log_probs).sum()
        total_entropy = torch.stack(all_entropies).sum()

        return action_grid, total_log_prob, total_entropy, value, burning_cells_info
