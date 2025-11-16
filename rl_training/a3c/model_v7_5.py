"""
A3C Model V7.5 - CORRECT Temporal LSTM Implementation

CRITICAL LESSONS FROM V7 FAILURE:
1. Keep FULL V3 encoder (32→64→128) - DON'T reduce capacity!
2. Use LSTM for temporal state - NOT 3D conv
3. Add LayerNorm for training stability
4. Memory-safe implementation with gradient checkpointing option

Target: 50-55% IoU (from V3's 40.91%)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class A3C_TemporalLSTM(nn.Module):
    """
    A3C model with LSTM temporal modeling for per-cell 8-neighbor prediction.

    Architecture:
    1. CNN Encoder (per timestep): (14, H, W) → (128, H, W) [FULL V3 encoder!]
    2. Temporal LSTM (pixel-wise): Process sequence → (128, H, W)
    3. Policy Head: Extract 3x3 features, predict 8 neighbors
    4. Value Head: Global average pooling → scalar value
    """

    def __init__(self, in_channels=14, window_size=3, lstm_hidden_dim=48,
                 lstm_num_layers=1, lstm_dropout=0.0, use_gradient_checkpointing=False,
                 downsample_factor=2, max_burning_cells=100):
        super().__init__()

        self.window_size = window_size
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.downsample_factor = downsample_factor
        self.max_burning_cells = max_burning_cells

        # CRITICAL: Keep FULL V3 encoder (32→64→128)
        # V7 mistake: Reduced to 64 channels, lost representation power
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),  # DON'T CUT THIS!
            nn.ReLU(inplace=True),
        )

        # Spatial downsampling before LSTM (reduces memory 4x with factor=2)
        if downsample_factor > 1:
            self.downsample = nn.AvgPool2d(kernel_size=downsample_factor, stride=downsample_factor)
            self.upsample = nn.Upsample(scale_factor=downsample_factor, mode='bilinear', align_corners=False)
        else:
            self.downsample = None
            self.upsample = None

        # Temporal LSTM (pixel-wise processing)
        # MEMORY-SAFE: 1 layer, 48 hidden (reduced from 96)
        # With downsampling: 50x50x48 instead of 100x100x96 = 16x less memory!
        self.temporal_lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0.0
        )

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(lstm_hidden_dim)

        # Policy head (per burning cell) - predicts 8-neighbor spread
        # Input: local 3x3 features around burning cell (48 * 9 = 432)
        self.policy_head = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 9, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 8)  # 8 neighbors
        )

        # Value head (global state value)
        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (B, 48, H, W) → (B, 48, 1, 1)
            nn.Flatten(),
            nn.Linear(lstm_hidden_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )

    def encode_sequence(self, x_seq):
        """
        Encode temporal sequence with CNN (per timestep) then LSTM.

        Args:
            x_seq: (B, T, 14, H, W) sequence of observations

        Returns:
            temporal_features: (B, lstm_hidden_dim, H, W) temporal-encoded features
        """
        B, T, C, H, W = x_seq.shape

        # Step 1: CNN encode each timestep independently
        feature_seq = []
        for t in range(T):
            obs_t = x_seq[:, t, :, :, :]  # (B, 14, H, W)

            # Use gradient checkpointing to save memory if enabled
            if self.use_gradient_checkpointing and self.training:
                feat_t = checkpoint(self.encoder, obs_t, use_reentrant=False)
            else:
                feat_t = self.encoder(obs_t)  # (B, 128, H, W)

            # MEMORY OPTIMIZATION: Downsample before LSTM
            if self.downsample is not None:
                feat_t = self.downsample(feat_t)  # (B, 128, H/2, W/2)

            feature_seq.append(feat_t)

        # Stack: (B, T, 128, H', W') where H'=H/downsample_factor
        feature_seq = torch.stack(feature_seq, dim=1)
        _, _, _, H_down, W_down = feature_seq.shape

        # Step 2: Process temporal sequence with LSTM at each spatial location
        # Reshape: (B, T, 128, H', W') → (B, H', W', T, 128)
        x = feature_seq.permute(0, 3, 4, 1, 2)  # (B, H', W', T, 128)
        x = x.reshape(B * H_down * W_down, T, 128)  # (B*H'*W', T, 128)

        # LSTM forward (outputs lstm_hidden_dim features)
        lstm_out, _ = self.temporal_lstm(x)  # (B*H'*W', T, lstm_hidden_dim)

        # Take last timestep output
        temporal_out = lstm_out[:, -1, :]  # (B*H'*W', lstm_hidden_dim)

        # Layer normalization for stability
        temporal_out = self.layer_norm(temporal_out)

        # Reshape back: (B, H', W', lstm_hidden_dim) → (B, lstm_hidden_dim, H', W')
        temporal_features = temporal_out.reshape(B, H_down, W_down, self.lstm_hidden_dim)
        temporal_features = temporal_features.permute(0, 3, 1, 2)  # (B, lstm_hidden_dim, H', W')

        # MEMORY OPTIMIZATION: Upsample back to original size
        if self.upsample is not None:
            temporal_features = self.upsample(temporal_features)  # (B, lstm_hidden_dim, H, W)

        return temporal_features

    def forward(self, x_seq, fire_mask):
        """
        Forward pass through temporal model.

        Args:
            x_seq: (B, T, 14, H, W) sequence of environmental features
            fire_mask: (B, H, W) current fire mask (at timestep T-1)

        Returns:
            temporal_features: (B, 96, H, W) temporal-encoded features
            value: (B, 1) state value
        """
        # Encode temporal sequence
        temporal_features = self.encode_sequence(x_seq)  # (B, 96, H, W)

        # Compute global value
        value = self.value_head(temporal_features)  # (B, 1)

        return temporal_features, value

    # ===== Per-cell prediction methods (same as V3) =====

    def get_burning_cells(self, fire_mask):
        """Extract locations of all burning cells."""
        if fire_mask.dim() == 3:
            fire_mask = fire_mask[0]

        burning_indices = torch.nonzero(fire_mask > 0.5, as_tuple=False)
        return [(int(idx[0]), int(idx[1])) for idx in burning_indices]

    def extract_local_features(self, features, i, j):
        """Extract 3x3 local features around cell (i, j)."""
        B, C, H, W = features.shape

        # Handle boundary cases with padding
        padded_features = F.pad(features, (1, 1, 1, 1), mode='constant', value=0)

        # Extract 3x3 region
        local = padded_features[:, :, i:i+3, j:j+3]  # (B, 128, 3, 3)

        # Flatten spatial dimensions
        local_flat = local.flatten(1)  # (B, 128*9)

        return local_flat

    def predict_8_neighbors(self, features, i, j):
        """Predict 8-neighbor spread for a single burning cell."""
        local_features = self.extract_local_features(features, i, j)
        logits = self.policy_head(local_features)  # (B, 8)
        return logits

    def get_8_neighbor_coords(self, i, j, H, W):
        """Get coordinates of 8 neighbors (N, NE, E, SE, S, SW, W, NW)."""
        neighbors = [
            (i-1, j),      # N
            (i-1, j+1),    # NE
            (i, j+1),      # E
            (i+1, j+1),    # SE
            (i+1, j),      # S
            (i+1, j-1),    # SW
            (i, j-1),      # W
            (i-1, j-1),    # NW
        ]

        valid_neighbors = []
        for ni, nj in neighbors:
            if 0 <= ni < H and 0 <= nj < W:
                valid_neighbors.append((ni, nj))
            else:
                valid_neighbors.append(None)

        return valid_neighbors

    def get_action_and_value(self, x_seq, fire_mask, action=None):
        """
        Get actions for all burning cells and compute value.

        Args:
            x_seq: (1, T, 14, H, W) temporal sequence (batch size must be 1)
            fire_mask: (1, H, W) or (H, W) current fire mask
            action: Optional pre-specified action

        Returns:
            action_grid: (H, W) binary prediction grid
            log_prob: Scalar log probability of action
            entropy: Scalar entropy
            value: (1, 1) state value
            burning_cells_info: List of debug info
        """
        if fire_mask.dim() == 2:
            fire_mask = fire_mask.unsqueeze(0)

        B, H, W = fire_mask.shape
        assert B == 1, "Batch size must be 1"

        # Get temporal features and value
        features, value = self.forward(x_seq, fire_mask)  # (1, 128, H, W), (1, 1)

        # Find all burning cells
        burning_cells = self.get_burning_cells(fire_mask)

        if len(burning_cells) == 0:
            # No burning cells - return zero action
            action_grid = torch.zeros(H, W, device=x_seq.device)
            log_prob = torch.tensor(0.0, device=x_seq.device)
            entropy = torch.tensor(0.0, device=x_seq.device)
            return action_grid, log_prob, entropy, value, []

        # MEMORY OPTIMIZATION: Limit number of burning cells processed
        if len(burning_cells) > self.max_burning_cells:
            # Randomly sample max_burning_cells
            indices = torch.randperm(len(burning_cells))[:self.max_burning_cells]
            burning_cells = [burning_cells[i] for i in indices]

        # For each burning cell, predict 8-neighbor spread
        all_log_probs = []
        all_entropies = []
        burning_cells_info = []

        action_grid = torch.zeros(H, W, device=x_seq.device)

        for cell_idx, (i, j) in enumerate(burning_cells):
            # Get 8-neighbor logits
            logits_8d = self.predict_8_neighbors(features, i, j).squeeze(0)  # (8,)

            # Get probabilities
            probs_8d = torch.sigmoid(logits_8d)
            probs_8d = torch.clamp(probs_8d, 1e-7, 1 - 1e-7)

            # Sample or use provided action
            if action is None:
                action_8d = torch.bernoulli(probs_8d)
            else:
                action_8d = action[cell_idx] if isinstance(action, list) else torch.bernoulli(probs_8d)

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

        # Aggregate log probs and entropies
        total_log_prob = torch.stack(all_log_probs).sum()
        total_entropy = torch.stack(all_entropies).sum()

        return action_grid, total_log_prob, total_entropy, value, burning_cells_info


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test model creation and forward pass
    print("=" * 80)
    print("A3C V7.5 - Temporal LSTM Model (ULTRA Memory-Safe)")
    print("=" * 80)

    model = A3C_TemporalLSTM(
        in_channels=14,
        window_size=3,
        lstm_hidden_dim=48,  # Reduced from 96
        lstm_num_layers=1,
        lstm_dropout=0.0,
        use_gradient_checkpointing=False,
        downsample_factor=2,  # 100x100 -> 50x50 (4x memory reduction)
        max_burning_cells=100  # Limit burning cells
    )

    total_params = count_parameters(model)
    print(f"\nModel Parameters: {total_params:,}")
    print(f"Memory Optimizations:")
    print(f"  - LSTM hidden: 48 (was 96)")
    print(f"  - Spatial downsample: 2x (100x100 -> 50x50)")
    print(f"  - Max burning cells: 100")
    print(f"Expected: ~200K params (V3 had 417K)")

    # Test forward pass
    print("\nTesting forward pass...")
    B, T, C, H, W = 1, 3, 14, 100, 100
    x_seq = torch.randn(B, T, C, H, W)
    fire_mask = torch.zeros(B, H, W)
    fire_mask[0, 50:55, 50:55] = 1.0  # Small fire region

    print(f"Input: {x_seq.shape}")
    print(f"Fire mask: {fire_mask.shape}, burning cells: {int(fire_mask.sum())}")

    with torch.no_grad():
        features, value = model(x_seq, fire_mask)
        print(f"\nOutput features: {features.shape}")
        print(f"Value: {value.shape}, {value.item():.4f}")

        # Test action generation
        action_grid, log_prob, entropy, value, info = model.get_action_and_value(x_seq, fire_mask)
        print(f"\nAction grid: {action_grid.shape}, predictions: {int(action_grid.sum())}")
        print(f"Log prob: {log_prob.item():.4f}")
        print(f"Entropy: {entropy.item():.4f}")
        print(f"Burning cells processed: {len(info)}")

    print("\n" + "=" * 80)
    print("✓ Model test passed!")
    print("=" * 80)
