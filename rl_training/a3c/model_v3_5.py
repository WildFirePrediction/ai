"""
A3C Model V3.5 - Temporal Context with Per-Pixel LSTM (Architecture Plan V3.5)

Implements the V3.5_ARCHITECTURE_PLAN.md specification:
1. Full V3 encoder (32 → 64 → 128 channels) - NO COMPROMISE
2. Per-pixel LSTM (2 layers, hidden_dim=128) - captures temporal velocity/acceleration
3. Temporal window = 5 timesteps - sufficient for fire dynamics
4. Layer normalization for training stability
5. Memory-optimized: GPU for model, CPU for workers, grid size filtering

Memory profile (grid 347×347 = 120K cells, 5 timesteps, 2 workers):
- Per worker: ~6GB RAM (safe for 64GB total)
- GPU model: ~500MB VRAM (safe for 12GB total)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class A3C_TemporalModel(nn.Module):
    """
    A3C model with per-pixel LSTM for temporal context.

    Architecture follows V3.5_ARCHITECTURE_PLAN.md exactly:
    - CNN encoder processes each timestep independently
    - Per-pixel LSTM captures temporal patterns (fire velocity, acceleration)
    - Same policy/value heads as V3

    Key difference from V3: Temporal modeling with LSTM (not just current timestep)
    """

    def __init__(self, in_channels=14, temporal_window=5, hidden_dim=128):
        super().__init__()

        self.in_channels = in_channels
        self.temporal_window = temporal_window
        self.hidden_dim = hidden_dim

        # FULL V3 ENCODER - NO REDUCTION (as per architecture plan)
        # 14 → 32 → 64 → 128 channels
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # PER-PIXEL LSTM (as specified in architecture plan)
        # Processes temporal sequences at each spatial location
        # Input: features from CNN encoder (128 dims)
        # Output: temporal features with memory (128 dims)
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )

        # Layer normalization for training stability
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Policy head (same as V3) - per-cell 8-neighbor prediction
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
        Encode temporal observation sequence with per-pixel LSTM.

        This is the CORE of V3.5 - captures fire spread velocity and acceleration.

        Architecture plan process:
        1. CNN encodes each timestep independently: (B, T, C, H, W) → (B, T, 128, H, W)
        2. Reshape to per-pixel sequences: (B*H*W, T, 128)
        3. LSTM processes each pixel's temporal evolution
        4. Reshape back to spatial: (B, 128, H, W)

        Args:
            obs_seq: (B, T, C, H, W) - temporal observation sequence

        Returns:
            temporal_features: (B, hidden_dim, H, W) - features with temporal context
        """
        B, T, C, H, W = obs_seq.shape

        # Step 1: Encode each timestep independently with CNN
        # Process each timestep through encoder
        feature_seq = []
        for t in range(T):
            feat_t = self.encoder(obs_seq[:, t])  # (B, 128, H, W)
            feature_seq.append(feat_t)
        feature_seq = torch.stack(feature_seq, dim=1)  # (B, T, 128, H, W)

        # Step 2: Reshape to per-pixel sequences
        # (B, T, 128, H, W) → (B, H, W, T, 128) → (B*H*W, T, 128)
        x = feature_seq.permute(0, 3, 4, 1, 2)  # (B, H, W, T, 128)
        x = x.reshape(B * H * W, T, self.hidden_dim)  # (B*H*W, T, 128)

        # Step 3: Process temporal sequences with LSTM
        # Each of the B*H*W pixels has a T-length sequence
        # LSTM learns: fire velocity (change between timesteps),
        #              acceleration (change in velocity),
        #              wind dynamics, humidity trends
        lstm_out, _ = self.lstm(x)  # (B*H*W, T, hidden_dim)

        # Take last timestep output (contains full temporal context)
        temporal_features = lstm_out[:, -1, :]  # (B*H*W, hidden_dim)

        # Step 4: Reshape back to spatial
        temporal_features = temporal_features.reshape(B, H, W, self.hidden_dim)
        temporal_features = temporal_features.permute(0, 3, 1, 2)  # (B, hidden_dim, H, W)

        # Apply layer norm for stability
        # Normalize across channel dimension at each spatial location
        B, C, H, W = temporal_features.shape
        temporal_features = temporal_features.permute(0, 2, 3, 1)  # (B, H, W, C)
        temporal_features = self.layer_norm(temporal_features)
        temporal_features = temporal_features.permute(0, 3, 1, 2)  # (B, C, H, W)

        return temporal_features

    def forward(self, obs_seq, fire_mask):
        """
        Forward pass with temporal sequence.

        Args:
            obs_seq: (B, T, C, H, W) - temporal observation sequence
            fire_mask: (B, H, W) - current fire mask (not used in forward, but kept for API consistency)

        Returns:
            features: (B, hidden_dim, H, W) - temporal features with LSTM context
            value: (B, 1) - state value
        """
        # Encode temporal sequence through CNN + LSTM
        features = self.encode_temporal_sequence(obs_seq)

        # Compute global state value
        value = self.value_head(features)

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
        local = padded_features[:, :, i:i+3, j:j+3]
        local_flat = local.flatten(1)
        return local_flat

    def predict_8_neighbors(self, features, i, j):
        """Predict 8-neighbor spread for a single burning cell."""
        local_features = self.extract_local_features(features, i, j)
        logits = self.policy_head(local_features)
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
        features, value = self.forward(obs_seq, fire_mask)

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
            logits_8d = self.predict_8_neighbors(features, i, j).squeeze(0)

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
