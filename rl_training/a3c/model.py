"""
A3C Model for Spatial Fire Spread Prediction

Actor-Critic architecture with shared CNN backbone for spatial predictions.
Outputs action probabilities for each cell (which cells will burn next).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class A3C_SpatialFireModel(nn.Module):
    """
    A3C model for spatial fire prediction.

    Input: (B, 14, H, W) - environmental features
    Output:
        - policy: (B, H, W) - probability each cell will burn
        - value: (B, 1) - state value estimate
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

        # Policy head (actor) - predicts burn probability per cell
        self.policy_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),  # (B, 1, H, W)
        )

        # Value head (critic) - estimates state value
        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        """
        Args:
            x: (B, 14, H, W) environmental features

        Returns:
            policy_logits: (B, H, W) - logits for burn probability per cell
            value: (B, 1) - state value
        """
        # Shared encoding
        features = self.encoder(x)  # (B, 128, H, W)

        # Policy (actor)
        policy_logits = self.policy_head(features).squeeze(1)  # (B, H, W)

        # Value (critic)
        value = self.value_head(features)  # (B, 1)

        return policy_logits, value

    def get_action_and_value(self, x, action=None):
        """
        Get action probabilities and value for given state.
        Optionally evaluate a specific action.

        Args:
            x: (B, 14, H, W) state
            action: Optional (B, H, W) binary mask of cells predicted to burn

        Returns:
            action: (B, H, W) sampled action if not provided
            log_prob: (B,) log probability of action
            entropy: (B,) entropy of policy distribution
            value: (B, 1) state value
        """
        policy_logits, value = self.forward(x)  # (B, H, W), (B, 1)

        # Flatten spatial dimensions for sampling
        B, H, W = policy_logits.shape
        flat_logits = policy_logits.view(B, -1)  # (B, H*W)

        # Create categorical distribution over cells
        probs = torch.sigmoid(flat_logits)  # Independent Bernoulli per cell

        # Sample action if not provided
        if action is None:
            action_flat = torch.bernoulli(probs)  # (B, H*W)
            action = action_flat.view(B, H, W)  # (B, H, W)
        else:
            action_flat = action.view(B, -1)

        # Compute log probability (sum of independent Bernoulli log probs)
        log_prob = (action_flat * torch.log(probs + 1e-8) +
                    (1 - action_flat) * torch.log(1 - probs + 1e-8)).sum(dim=1)  # (B,)

        # Compute entropy (sum of Bernoulli entropies)
        entropy = -(probs * torch.log(probs + 1e-8) +
                   (1 - probs) * torch.log(1 - probs + 1e-8)).sum(dim=1)  # (B,)

        return action, log_prob, entropy, value
