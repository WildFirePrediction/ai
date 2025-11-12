"""
A3C Model V2 - Correct Formulation

Per-cell 8-neighbor prediction model for wildfire spread.
For each burning cell, predicts which of its 8 neighbors will burn next.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class A3C_PerCellModel(nn.Module):
    """
    A3C model for per-cell 8-neighbor fire spread prediction.

    For each burning cell at (i,j), predicts which of its 8 neighbors will burn:
    - Neighbors: N, NE, E, SE, S, SW, W, NW
    - Action per cell: 8-dimensional binary vector
    """

    def __init__(self, in_channels=14):
        super().__init__()

        # Shared CNN encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
        )

        # Policy head (per burning cell) - predicts 8-neighbor spread
        # Input: local 3x3 features around burning cell (128 * 9 = 1152)
        self.policy_head = nn.Sequential(
            nn.Linear(128 * 9, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 8)  # 8 neighbors
        )

        # Value head (global state value)
        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (B, 128, H, W) → (B, 128, 1, 1)
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, fire_mask):
        """
        Args:
            x: (B, 14, H, W) environmental features
            fire_mask: (B, H, W) current fire mask (1 = burning, 0 = not burning)

        Returns:
            feature_map: (B, 128, H, W) encoded features
            value: (B, 1) state value
        """
        # Encode features
        features = self.encoder(x)  # (B, 128, H, W)

        # Compute global value
        value = self.value_head(features)  # (B, 1)

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
            fire_mask = fire_mask[0]  # Take first batch (assume batch=1 for now)

        burning_indices = torch.nonzero(fire_mask > 0.5, as_tuple=False)  # (N, 2)
        return [(int(idx[0]), int(idx[1])) for idx in burning_indices]

    def extract_local_features(self, features, i, j):
        """
        Extract 3x3 local features around cell (i, j).

        Args:
            features: (B, 128, H, W) feature map
            i, j: Cell coordinates

        Returns:
            (B, 128*9) flattened local features
        """
        B, C, H, W = features.shape

        # Handle boundary cases with padding
        padded_features = F.pad(features, (1, 1, 1, 1), mode='constant', value=0)

        # Extract 3x3 region (after padding, indices shift by 1)
        local = padded_features[:, :, i:i+3, j:j+3]  # (B, 128, 3, 3)

        # Flatten spatial dimensions
        local_flat = local.flatten(1)  # (B, 128*9)

        return local_flat

    def predict_8_neighbors(self, features, i, j):
        """
        Predict 8-neighbor spread for a single burning cell.

        Args:
            features: (B, 128, H, W) feature map
            i, j: Burning cell coordinates

        Returns:
            logits: (B, 8) logits for 8 neighbors
        """
        local_features = self.extract_local_features(features, i, j)
        logits = self.policy_head(local_features)  # (B, 8)
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

        # Check bounds
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
            x: (1, 14, H, W) environmental features (batch size must be 1)
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

        # Get features and value
        features, value = self.forward(x, fire_mask)  # (1, 128, H, W), (1, 1)

        # Find all burning cells
        burning_cells = self.get_burning_cells(fire_mask)

        if len(burning_cells) == 0:
            # No burning cells - return zero action
            action_grid = torch.zeros(H, W)
            log_prob = torch.tensor(0.0)
            entropy = torch.tensor(0.0)
            return action_grid, log_prob, entropy, value, []

        # For each burning cell, predict 8-neighbor spread
        all_log_probs = []
        all_entropies = []
        burning_cells_info = []

        action_grid = torch.zeros(H, W)

        for cell_idx, (i, j) in enumerate(burning_cells):
            # Get 8-neighbor logits
            logits_8d = self.predict_8_neighbors(features, i, j)  # (1, 8)
            logits_8d = logits_8d.squeeze(0)  # (8,)

            # Get probabilities
            probs_8d = torch.sigmoid(logits_8d)
            probs_8d = torch.clamp(probs_8d, 1e-7, 1 - 1e-7)

            # Sample or use provided action
            if action is None:
                action_8d = torch.bernoulli(probs_8d)  # (8,)
            else:
                # Extract action for this cell from provided action grid
                # This is for evaluation - need to extract 8-neighbor action from grid
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
