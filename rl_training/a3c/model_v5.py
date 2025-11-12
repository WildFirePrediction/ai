"""
A3C Model V5 - 4-Neighbor Multi-Task Architecture

Key improvements based on MCTS-A3C insights:
1. 4-neighbor prediction (N, E, S, W) - simpler action space
2. Multi-task learning: burn + intensity + temperature
3. Richer supervision signal to combat class imbalance
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class A3C_PerCellModel_4Neighbor(nn.Module):
    """
    A3C model with 4-neighbor prediction and multi-task learning.

    Predicts for each burning cell:
    - Which 4 neighbors (N, E, S, W) will burn (binary)
    - Intensity of burns (continuous)
    - Temperature of burns (continuous)

    Multi-task learning provides richer gradients and better representations.
    """

    def __init__(self, in_channels=14, hidden_dim=128, use_groupnorm=True):
        super().__init__()

        self.use_groupnorm = use_groupnorm
        self.hidden_dim = hidden_dim

        # Shared encoder (same as V2 but slightly wider)
        layers = []

        # Layer 1: 14 → 48
        layers.append(nn.Conv2d(in_channels, 48, kernel_size=3, padding=1))
        if use_groupnorm:
            layers.append(nn.GroupNorm(6, 48))
        layers.append(nn.ReLU())

        # Layer 2: 48 → 96
        layers.append(nn.Conv2d(48, 96, kernel_size=3, padding=1))
        if use_groupnorm:
            layers.append(nn.GroupNorm(12, 96))
        layers.append(nn.ReLU())

        # Layer 3: 96 → hidden_dim
        layers.append(nn.Conv2d(96, hidden_dim, kernel_size=3, padding=1))
        if use_groupnorm:
            layers.append(nn.GroupNorm(16, hidden_dim))
        layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*layers)

        # Shared feature extraction for per-cell tasks
        # Input: local 3x3 features (hidden_dim * 9)
        self.shared_fc = nn.Sequential(
            nn.Linear(hidden_dim * 9, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # Task 1: Burn prediction head (primary task)
        self.burn_head = nn.Linear(128, 4)  # 4 neighbors: N, E, S, W

        # Task 2: Intensity prediction head (auxiliary)
        self.intensity_head = nn.Linear(128, 4)  # Intensity for each neighbor

        # Task 3: Temperature prediction head (auxiliary)
        self.temp_head = nn.Linear(128, 4)  # Temperature for each neighbor

        # Value head (global state value)
        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (B, hidden_dim, H, W) → (B, hidden_dim, 1, 1)
            nn.Flatten(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, fire_mask):
        """
        Args:
            x: (B, 14, H, W) environmental features
            fire_mask: (B, H, W) current fire mask

        Returns:
            feature_map: (B, hidden_dim, H, W) encoded features
            value: (B, 1) state value
        """
        features = self.encoder(x)  # (B, hidden_dim, H, W)
        value = self.value_head(features)  # (B, 1)
        return features, value

    def get_burning_cells(self, fire_mask):
        """Extract locations of burning cells."""
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

    def predict_4_neighbors(self, features, i, j):
        """
        Predict 4-neighbor spread for a single burning cell.

        Args:
            features: (B, hidden_dim, H, W) feature map
            i, j: Burning cell coordinates

        Returns:
            burn_logits: (B, 4) logits for burn probability (N, E, S, W)
            intensity_pred: (B, 4) predicted intensity
            temp_pred: (B, 4) predicted temperature
        """
        local_features = self.extract_local_features(features, i, j)
        shared_repr = self.shared_fc(local_features)  # (B, 128)

        burn_logits = self.burn_head(shared_repr)  # (B, 4)
        intensity_pred = self.intensity_head(shared_repr)  # (B, 4)
        temp_pred = self.temp_head(shared_repr)  # (B, 4)

        return burn_logits, intensity_pred, temp_pred

    def get_4_neighbor_coords(self, i, j, H, W):
        """
        Get coordinates of 4 cardinal neighbors.
        Order: N, E, S, W

        Returns:
            List of (ni, nj) tuples, None for out-of-bounds neighbors
        """
        neighbors = [
            (i-1, j),  # N
            (i, j+1),  # E
            (i+1, j),  # S
            (i, j-1),  # W
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
            x: (1, 14, H, W) environmental features
            fire_mask: (1, H, W) or (H, W) current fire mask
            action: Optional pre-specified action

        Returns:
            action_grid: (H, W) binary prediction grid
            log_prob: Scalar log probability of action
            entropy: Scalar entropy
            value: (1, 1) state value
            burning_cells_info: List of cell info for debugging
        """
        if fire_mask.dim() == 2:
            fire_mask = fire_mask.unsqueeze(0)

        B, H, W = fire_mask.shape
        assert B == 1, "Batch size must be 1"

        # Get features and value
        features, value = self.forward(x, fire_mask)

        # Find all burning cells
        burning_cells = self.get_burning_cells(fire_mask)

        if len(burning_cells) == 0:
            action_grid = torch.zeros(H, W)
            log_prob = torch.tensor(0.0)
            entropy = torch.tensor(0.0)
            return action_grid, log_prob, entropy, value, []

        # For each burning cell, predict 4-neighbor spread
        all_log_probs = []
        all_entropies = []
        burning_cells_info = []

        action_grid = torch.zeros(H, W)

        for cell_idx, (i, j) in enumerate(burning_cells):
            # Get 4-neighbor predictions (only use burn logits for policy)
            burn_logits, intensity_pred, temp_pred = self.predict_4_neighbors(features, i, j)
            burn_logits = burn_logits.squeeze(0)  # (4,)

            # Get burn probabilities
            probs_4d = torch.sigmoid(burn_logits)
            probs_4d = torch.clamp(probs_4d, 1e-7, 1 - 1e-7)

            # Sample action
            if action is None:
                action_4d = torch.bernoulli(probs_4d)  # (4,)
            else:
                action_4d = action[cell_idx] if isinstance(action, list) else torch.bernoulli(probs_4d)

            # Compute log probability
            log_prob_4d = (action_4d * torch.log(probs_4d) +
                          (1 - action_4d) * torch.log(1 - probs_4d))
            log_prob_cell = log_prob_4d.sum()

            # Compute entropy
            entropy_4d = -(probs_4d * torch.log(probs_4d) +
                          (1 - probs_4d) * torch.log(1 - probs_4d))
            entropy_cell = entropy_4d.sum()

            all_log_probs.append(log_prob_cell)
            all_entropies.append(entropy_cell)

            # Map action to grid
            neighbors = self.get_4_neighbor_coords(i, j, H, W)
            for n_idx, neighbor in enumerate(neighbors):
                if neighbor is not None and action_4d[n_idx] > 0.5:
                    ni, nj = neighbor
                    action_grid[ni, nj] = 1.0

            burning_cells_info.append((i, j, action_4d, log_prob_cell))

        # Aggregate log probs and entropies
        total_log_prob = torch.stack(all_log_probs).sum()
        total_entropy = torch.stack(all_entropies).sum()

        return action_grid, total_log_prob, total_entropy, value, burning_cells_info
