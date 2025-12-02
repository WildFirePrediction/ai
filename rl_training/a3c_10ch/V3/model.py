"""
A3C Model V3 - 10 Channels (Wind-Focused)
Deeper Encoder for Wildfire Spread Prediction

Input channels:
- Channel 0-1: DEM (slope, aspect)
- Channel 2-4: Wind (speed, u-component, v-component)
- Channel 5: NDVI (vegetation fuel)
- Channel 6-9: FSM (forest susceptibility, one-hot encoded)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class A3C_PerCellModel_Deep(nn.Module):
    """
    A3C model with shallow encoder optimized for strong gradients.

    Key changes for better learning:
    - Only 3 conv layers (not 5) to prevent gradient vanishing
    - Larger channels faster (10->64->128->256)
    - Dropout for regularization
    - Simpler policy head
    - Higher learning signal
    """

    def __init__(self, in_channels=10, use_groupnorm=True):
        super().__init__()

        self.use_groupnorm = use_groupnorm

        # SHALLOW encoder - 3 layers only
        # Layer 1: input → 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, 64) if use_groupnorm else nn.Identity()
        self.dropout1 = nn.Dropout2d(0.1)

        # Layer 2: 64 → 128
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(16, 128) if use_groupnorm else nn.Identity()
        self.dropout2 = nn.Dropout2d(0.1)

        # Layer 3: 128 → 256
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.norm3 = nn.GroupNorm(32, 256) if use_groupnorm else nn.Identity()
        self.dropout3 = nn.Dropout2d(0.1)

        # Policy head - SIMPLER for stronger gradients
        self.policy_head = nn.Sequential(
            nn.Linear(256 * 9, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 8)
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

        # Initialize weights with larger values for stronger gradients
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize with larger weights for stronger gradients"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x, fire_mask):
        """
        Args:
            x: (B, 10, H, W) environmental features
            fire_mask: (B, H, W) current fire mask (1 = burning, 0 = not burning)

        Returns:
            feature_map: (B, 256, H, W) encoded features
            value: (B, 1) state value
        """
        # 3-layer encoder
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.dropout1(x)

        x = F.relu(self.norm2(self.conv2(x)))
        x = self.dropout2(x)

        features = F.relu(self.norm3(self.conv3(x)))
        features = self.dropout3(features)

        value = self.value_head(features)
        return features, value

    def get_burning_cells(self, fire_mask):
        """
        Extract locations of all burning cells.

        Args:
            fire_mask: (H, W) or (B, H, W) binary mask

        Returns:
            List of (i, j) tuples for burning cell locations
        """
        if fire_mask.dim() == 3:
            fire_mask = fire_mask[0]

        burning_indices = torch.nonzero(fire_mask > 0.5, as_tuple=False)
        return [(int(idx[0]), int(idx[1])) for idx in burning_indices]

    def extract_local_features(self, features, i, j):
        """
        Extract 3x3 local features around cell (i, j).

        Args:
            features: (B, 256, H, W) feature map
            i, j: Cell coordinates

        Returns:
            (B, 256*9) flattened local features
        """
        B, C, H, W = features.shape
        padded_features = F.pad(features, (1, 1, 1, 1), mode='constant', value=0)
        local = padded_features[:, :, i:i+3, j:j+3]
        local_flat = local.flatten(1)
        return local_flat

    def predict_8_neighbors(self, features, i, j):
        """
        Predict 8-neighbor spread for a single burning cell.

        Args:
            features: (B, 256, H, W) feature map
            i, j: Burning cell coordinates

        Returns:
            logits: (B, 8) logits for 8 neighbors
        """
        local_features = self.extract_local_features(features, i, j)
        logits = self.policy_head(local_features)
        return logits

    def get_8_neighbor_coords(self, i, j, H, W):
        """
        Get coordinates of 8 neighbors.
        Order: N, NE, E, SE, S, SW, W, NW

        Args:
            i, j: Center cell
            H, W: Grid dimensions

        Returns:
            List of (ni, nj) tuples, None for out-of-bounds neighbors
        """
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

    def get_action_and_value(self, x, fire_mask, action=None):
        """
        Get actions for all burning cells and compute value.

        Args:
            x: (1, 10, H, W) environmental features (batch size must be 1)
            fire_mask: (1, H, W) or (H, W) current fire mask
            action: Optional pre-specified action (for evaluation)

        Returns:
            action_grid: (H, W) binary prediction grid
            log_prob: Scalar log probability of action
            entropy: Scalar entropy
            value: (1, 1) state value
            burning_cells_info: List of (i, j, action_8d, log_prob_8d) for debugging
        """
        if fire_mask.dim() == 2:
            fire_mask = fire_mask.unsqueeze(0)

        B, H, W = fire_mask.shape
        assert B == 1, "Batch size must be 1 for now"

        features, value = self.forward(x, fire_mask)

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
            logits_8d = self.predict_8_neighbors(features, i, j)
            logits_8d = logits_8d.squeeze(0)

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
