"""
A3C Model V8 SMALL with LSTM - 16 Channels (REDUCED CAPACITY)
Smaller model to combat overfitting

Improvements over V6:
- LSTM hidden: 128 (was 256) - 2x smaller
- Narrower CNN: 32→64→128 (was 64→128→256) - 2x narrower
- Smaller policy head: 128 hidden (was 256)
- ~500K parameters (was ~1.8M) - 3.5x smaller
- GroupNorm (proven to work in V6)

Goal: Better generalization with less overfitting

Input channels:
- Channel 0-1: DEM (slope, aspect)
- Channel 2-10: Weather (temp, humidity, wind_speed, wind_dir, precip, pressure, cloud, visibility, dew_point)
- Channel 11: NDVI (vegetation)
- Channel 12-15: FSM (forest susceptibility, one-hot)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class A3C_PerCellModel_LSTM(nn.Module):
    """
    SMALL A3C model with CNN encoder + LSTM for temporal context.

    Architecture:
    1. CNN Encoder: 3-layer narrow (16ch -> 128 features) with GroupNorm
    2. LSTM: 128 hidden units (half of V6)
    3. Policy/Value Heads: Smaller capacity
    """

    def __init__(self, in_channels=16, lstm_hidden=128, sequence_length=3, use_groupnorm=True):
        super().__init__()

        self.in_channels = in_channels
        self.lstm_hidden = lstm_hidden
        self.sequence_length = sequence_length
        self.use_groupnorm = use_groupnorm

        # CNN Encoder (per-timestep spatial processing) - NARROWER
        # Input: (batch, 16, 30, 30) -> Output: (batch, 128, 30, 30)

        # Layer 1: 16 -> 32 (was 64 in V6)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(4, 32) if use_groupnorm else nn.Identity()
        self.dropout1 = nn.Dropout2d(0.1)

        # Layer 2: 32 -> 64 (was 128 in V6)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, 64) if use_groupnorm else nn.Identity()
        self.dropout2 = nn.Dropout2d(0.1)

        # Layer 3: 64 -> 128 (was 256 in V6)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.norm3 = nn.GroupNorm(16, 128) if use_groupnorm else nn.Identity()
        self.dropout3 = nn.Dropout2d(0.1)

        # Spatial pooling to get per-timestep feature vector
        # (batch, 128, 30, 30) -> (batch, 128)
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)

        # LSTM for temporal modeling - SMALLER
        # Input: (sequence_length, batch, 128) -> Output: (sequence_length, batch, 128)
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=False,  # (seq, batch, features)
            dropout=0.0
        )

        # Upsampling to restore spatial dimensions
        # (batch, 128) -> (batch, 128, 30, 30)
        self.upsample = nn.Sequential(
            nn.Linear(lstm_hidden, 128 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (128, 2, 2)),
            nn.Upsample(size=(30, 30), mode='bilinear', align_corners=False)
        )

        # Policy head - per-cell 8-neighbor prediction - SMALLER
        self.policy_head = nn.Sequential(
            nn.Linear(128 * 9, 128),  # Was 256*9 -> 256 in V6
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 8)
        )

        # Value head - SMALLER
        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),  # Was 256 -> 64 in V6
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for strong gradients"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

    def encode_timestep(self, x):
        """
        Encode a single timestep spatially with CNN.

        Args:
            x: (batch, 16, 30, 30) - single timestep

        Returns:
            spatial_features: (batch, 128, 30, 30)
            pooled_features: (batch, 128)
        """
        # Layer 1
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.dropout1(x)

        # Layer 2
        x = F.relu(self.norm2(self.conv2(x)))
        x = self.dropout2(x)

        # Layer 3
        spatial_features = F.relu(self.norm3(self.conv3(x)))
        spatial_features = self.dropout3(spatial_features)

        # Pool for LSTM input
        pooled = self.spatial_pool(spatial_features).flatten(1)

        return spatial_features, pooled

    def forward(self, sequence, fire_mask):
        """
        Forward pass with temporal sequence.

        Args:
            sequence: (batch, seq_len, 16, 30, 30) - temporal sequence
            fire_mask: (batch, 30, 30) - current fire mask (only for last timestep)

        Returns:
            spatial_features: (batch, 128, 30, 30) - features for policy head
            value: (batch, 1) - state value
        """
        batch_size, seq_len, C, H, W = sequence.shape

        # Encode each timestep with CNN
        pooled_sequence = []
        for t in range(seq_len):
            _, pooled = self.encode_timestep(sequence[:, t])  # (batch, 128)
            pooled_sequence.append(pooled)

        # Stack into sequence: (seq_len, batch, 128)
        pooled_sequence = torch.stack(pooled_sequence, dim=0)

        # LSTM temporal processing
        lstm_out, (h_n, c_n) = self.lstm(pooled_sequence)

        # Use final hidden state: (batch, lstm_hidden)
        final_hidden = h_n.squeeze(0)  # (1, batch, hidden) -> (batch, hidden)

        # Upsample to spatial features: (batch, 128) -> (batch, 128, 30, 30)
        spatial_features = self.upsample(final_hidden)

        # Value prediction
        value = self.value_head(spatial_features)

        return spatial_features, value

    def get_burning_cells(self, fire_mask):
        """Extract burning cell locations"""
        if fire_mask.dim() == 3:
            fire_mask = fire_mask[0]
        burning_indices = torch.nonzero(fire_mask > 0.5, as_tuple=False)
        return [(int(idx[0]), int(idx[1])) for idx in burning_indices]

    def extract_local_features(self, features, i, j):
        """Extract 3x3 local features around cell (i, j)"""
        B, C, H, W = features.shape
        padded_features = F.pad(features, (1, 1, 1, 1), mode='constant', value=0)
        local = padded_features[:, :, i:i+3, j:j+3]
        return local.flatten(1)

    def predict_8_neighbors(self, features, i, j):
        """Predict 8-neighbor spread for burning cell"""
        local_features = self.extract_local_features(features, i, j)
        logits = self.policy_head(local_features)
        return logits

    def get_8_neighbor_coords(self, i, j, H, W):
        """Get 8-neighbor coordinates"""
        neighbors = [
            (i-1, j), (i-1, j+1), (i, j+1), (i+1, j+1),
            (i+1, j), (i+1, j-1), (i, j-1), (i-1, j-1)
        ]
        valid_neighbors = []
        for ni, nj in neighbors:
            if 0 <= ni < H and 0 <= nj < W:
                valid_neighbors.append((ni, nj))
            else:
                valid_neighbors.append(None)
        return valid_neighbors

    def get_action_and_value(self, sequence, fire_mask, action=None):
        """
        Get actions for all burning cells and compute value.

        Args:
            sequence: (1, seq_len, 16, 30, 30) - temporal sequence
            fire_mask: (1, 30, 30) or (30, 30) - current fire mask
            action: Optional pre-specified action

        Returns:
            action_grid: (H, W) binary prediction grid
            log_prob: Scalar log probability
            entropy: Scalar entropy
            value: (1, 1) state value
            burning_cells_info: List of cell info for debugging
        """
        if fire_mask.dim() == 2:
            fire_mask = fire_mask.unsqueeze(0)

        B, H, W = fire_mask.shape
        assert B == 1, "Batch size must be 1"

        features, value = self.forward(sequence, fire_mask)

        burning_cells = self.get_burning_cells(fire_mask)

        if len(burning_cells) == 0:
            action_grid = torch.zeros(H, W)
            log_prob = torch.tensor(0.0)
            entropy = torch.tensor(0.0)
            return action_grid, log_prob, entropy, value, []

        all_log_probs = []
        all_entropies = []
        burning_cells_info = []
        action_grid = torch.zeros(H, W)

        for cell_idx, (i, j) in enumerate(burning_cells):
            logits_8d = self.predict_8_neighbors(features, i, j).squeeze(0)
            probs_8d = torch.sigmoid(logits_8d)
            probs_8d = torch.clamp(probs_8d, 1e-7, 1 - 1e-7)

            if action is None:
                action_8d = torch.bernoulli(probs_8d)
            else:
                action_8d = action[cell_idx] if isinstance(action, list) else torch.bernoulli(probs_8d)

            log_prob_8d = (action_8d * torch.log(probs_8d) +
                          (1 - action_8d) * torch.log(1 - probs_8d))
            log_prob_cell = log_prob_8d.sum()

            entropy_8d = -(probs_8d * torch.log(probs_8d) +
                          (1 - probs_8d) * torch.log(1 - probs_8d))
            entropy_cell = entropy_8d.sum()

            all_log_probs.append(log_prob_cell)
            all_entropies.append(entropy_cell)

            neighbors = self.get_8_neighbor_coords(i, j, H, W)
            for n_idx, neighbor in enumerate(neighbors):
                if neighbor is not None and action_8d[n_idx] > 0.5:
                    ni, nj = neighbor
                    action_grid[ni, nj] = 1.0

            burning_cells_info.append((i, j, action_8d, log_prob_cell))

        total_log_prob = torch.stack(all_log_probs).sum()
        total_entropy = torch.stack(all_entropies).sum()

        return action_grid, total_log_prob, total_entropy, value, burning_cells_info
