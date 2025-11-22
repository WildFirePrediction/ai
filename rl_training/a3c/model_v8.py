"""
A3C Model V8 - Spatial + Channel Attention Architecture

Enhances V3 baseline with dual attention mechanisms:
1. Spatial Attention: Focuses on fire boundaries and active spread zones
2. Channel Attention: Dynamically weights important feature channels

Architecture: V3 Encoder → Spatial Attention → Channel Attention → Policy/Value Heads

Author: Wildfire Prediction Team
Version: 8.0
Date: 2025-11-22
Hardware Target: RTX 5070 12GB VRAM, 64GB RAM
Expected Performance: 40-50% IoU (baseline 31.82% → target +8-18%)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module - Focuses on important spatial regions.
    
    Applies attention across spatial dimensions (H, W) to emphasize:
    - Fire boundaries (where spread is most active)
    - Regions with high environmental risk (wind-driven zones)
    - Active fire fronts vs. inactive burned areas
    
    Architecture:
        AvgPool(C) + MaxPool(C) → Conv(2→1) → Sigmoid → Attention Map
    
    Memory: ~500 parameters (7×7 conv kernel)
    """
    
    def __init__(self, kernel_size: int = 7):
        """
        Initialize Spatial Attention module.
        
        Args:
            kernel_size: Size of convolution kernel (default: 7 for large receptive field)
                        Larger kernel captures broader spatial context.
        """
        super().__init__()
        
        assert kernel_size % 2 == 1, "Kernel size must be odd for proper padding"
        
        # Compress channel dimension using both average and max pooling
        # Average: captures overall activation level
        # Max: captures peak activations (fire hotspots)
        self.conv = nn.Conv2d(
            in_channels=2,  # AvgPool + MaxPool outputs
            out_channels=1,  # Single attention map
            kernel_size=kernel_size,
            padding=kernel_size // 2,  # Maintain spatial dimensions
            bias=False  # Bias not needed for attention
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial attention to input features.
        
        Args:
            x: Input feature map (B, C, H, W)
            
        Returns:
            Attended features (B, C, H, W)
        """
        # Channel-wise pooling to compress features
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W) - average activation
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W) - peak activation
        
        # Concatenate pooled features
        pooled = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        
        # Learn spatial attention map
        attention_map = self.sigmoid(self.conv(pooled))  # (B, 1, H, W), range [0, 1]
        
        # Apply attention (element-wise multiplication)
        # High attention weights → emphasize important regions
        # Low attention weights → suppress irrelevant regions
        return x * attention_map


class ChannelAttention(nn.Module):
    """
    Channel Attention Module - Dynamically weights feature channels.
    
    Learns importance of each feature channel based on global context:
    - High weight for wind features in wind-driven fires
    - High weight for humidity in dry conditions
    - Adaptive to different fire scenarios
    
    Architecture:
        GlobalAvgPool → FC(C→C/r) → ReLU → FC(C/r→C) → Sigmoid → Channel Weights
    
    Uses Squeeze-and-Excitation (SE) block design.
    
    Memory: ~2K parameters for 128 channels with reduction=16
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        """
        Initialize Channel Attention module.
        
        Args:
            channels: Number of input channels (typically 128 for V3 encoder)
            reduction: Reduction ratio for bottleneck (default: 16)
                      Higher reduction = fewer parameters, lower expressiveness
        """
        super().__init__()
        
        # Squeeze: Global context via adaptive average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # (B, C, H, W) → (B, C, 1, 1)
        
        # Excitation: Two-layer MLP with bottleneck
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),  # Compress
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),  # Restore
            nn.Sigmoid()  # Attention weights in [0, 1]
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply channel attention to input features.
        
        Args:
            x: Input feature map (B, C, H, W)
            
        Returns:
            Attended features (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Squeeze: Global average pooling captures channel-wise statistics
        # Represents "overall importance" of each channel
        y = self.avg_pool(x).view(B, C)  # (B, C)
        
        # Excitation: Learn channel-wise attention weights
        channel_weights = self.fc(y).view(B, C, 1, 1)  # (B, C, 1, 1)
        
        # Apply attention (channel-wise scaling)
        # Important channels get amplified, less important get suppressed
        return x * channel_weights


class A3C_PerCellModel_V8(nn.Module):
    """
    A3C V8 Model with Spatial and Channel Attention.
    
    Enhanced per-cell 8-neighbor prediction model with dual attention:
    1. Spatial Attention: Focuses on fire boundaries
    2. Channel Attention: Weights important features (wind, terrain, etc.)
    
    Architecture:
        Input (15 channels: terrain, fire state, weather w/ rainfall)
        ↓
        CNN Encoder (32 → 64 → 128 channels, 3 layers)
        ↓
        Spatial Attention (emphasize fire fronts)
        ↓
        Channel Attention (weight feature importance)
        ↓
        Attended Features (128 channels)
        ↓
        Policy Head (per-cell 8-neighbor) + Value Head
    
    Parameters: ~420K (vs. ~417K in V3)
    Memory footprint: ~25 MB per worker during training
    """
    
    def __init__(self, in_channels: int = 15):
        """
        Initialize A3C V8 model.
        
        Args:
            in_channels: Number of input channels (15 for re-embedded data with rainfall)
        """
        super().__init__()
        
        # V3 baseline encoder - proven effective
        # 3-layer CNN: captures hierarchical spatial features
        self.encoder = nn.Sequential(
            # Layer 1: 15 → 32 channels
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),  # Stabilize training
            nn.ReLU(inplace=True),
            
            # Layer 2: 32 → 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Layer 3: 64 → 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # V8 attention modules (NEW)
        self.spatial_attention = SpatialAttention(kernel_size=7)
        self.channel_attention = ChannelAttention(channels=128, reduction=16)
        
        # Policy head - predicts 8-neighbor spread for each burning cell
        # Input: 3×3 local features around burning cell (128 × 9 = 1152)
        self.policy_head = nn.Sequential(
            nn.Linear(128 * 9, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),  # Light regularization
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 8)  # 8 neighbors: N, NE, E, SE, S, SW, W, NW
        )
        
        # Value head - estimates global state value
        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )
        
    def forward(self, x: torch.Tensor, fire_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: Environmental features (B, 15, H, W)
                Channels: [terrain(3), lcm(1), fsm(1), fire_state(4), weather(6)]
            fire_mask: Current fire mask (B, H, W)
                1 = burning, 0 = not burning
        
        Returns:
            features: Attended feature map (B, 128, H, W)
            value: State value estimate (B, 1)
        """
        # Validate input shapes
        assert x.dim() == 4, f"Expected 4D input (B,C,H,W), got {x.dim()}D"
        assert x.size(1) == 15, f"Expected 15 input channels, got {x.size(1)}"
        
        # Encode spatial features
        features = self.encoder(x)  # (B, 128, H, W)
        
        # Apply spatial attention (focus on fire boundaries)
        features = self.spatial_attention(features)  # (B, 128, H, W)
        
        # Apply channel attention (weight important features)
        features = self.channel_attention(features)  # (B, 128, H, W)
        
        # Compute global state value
        value = self.value_head(features)  # (B, 1)
        
        return features, value
    
    def get_burning_cells(self, fire_mask: torch.Tensor) -> List[Tuple[int, int]]:
        """
        Extract locations of all burning cells from fire mask.
        
        Args:
            fire_mask: Binary fire mask (H, W) or (B, H, W)
        
        Returns:
            List of (row, col) tuples for burning cells
        """
        if fire_mask.dim() == 3:
            fire_mask = fire_mask[0]  # Take first batch (assume B=1 during inference)
        
        # Find all cells with fire_mask > 0.5
        burning_indices = torch.nonzero(fire_mask > 0.5, as_tuple=False)  # (N, 2)
        
        return [(int(idx[0]), int(idx[1])) for idx in burning_indices]
    
    def extract_local_features(self, features: torch.Tensor, i: int, j: int) -> torch.Tensor:
        """
        Extract 3×3 local features around a burning cell.
        
        Args:
            features: Feature map (B, 128, H, W)
            i, j: Cell coordinates (row, col)
        
        Returns:
            Flattened local features (B, 128*9)
        """
        B, C, H, W = features.shape
        
        # Pad features to handle boundary cells
        # mode='constant' with value=0 assumes no fire outside boundaries
        padded_features = F.pad(features, (1, 1, 1, 1), mode='constant', value=0)
        
        # Extract 3×3 region (after padding, indices shift by 1)
        local = padded_features[:, :, i:i+3, j:j+3]  # (B, 128, 3, 3)
        
        # Flatten spatial dimensions
        local_flat = local.flatten(1)  # (B, 128*9)
        
        return local_flat
    
    def predict_8_neighbors(self, features: torch.Tensor, i: int, j: int) -> torch.Tensor:
        """
        Predict 8-neighbor spread probabilities for a burning cell.
        
        Args:
            features: Attended feature map (B, 128, H, W)
            i, j: Burning cell coordinates (row, col)
        
        Returns:
            Logits for 8 neighbors (B, 8)
        """
        local_features = self.extract_local_features(features, i, j)
        logits = self.policy_head(local_features)  # (B, 8)
        return logits
    
    def get_8_neighbor_coords(self, i: int, j: int, H: int, W: int) -> List[Optional[Tuple[int, int]]]:
        """
        Get coordinates of 8 neighbors around a cell.
        
        Order: N, NE, E, SE, S, SW, W, NW (clockwise from North)
        
        Args:
            i, j: Center cell coordinates
            H, W: Grid dimensions
        
        Returns:
            List of (row, col) tuples, None for out-of-bounds neighbors
        """
        neighbors = [
            (i-1, j),      # N  (North)
            (i-1, j+1),    # NE (Northeast)
            (i, j+1),      # E  (East)
            (i+1, j+1),    # SE (Southeast)
            (i+1, j),      # S  (South)
            (i+1, j-1),    # SW (Southwest)
            (i, j-1),      # W  (West)
            (i-1, j-1),    # NW (Northwest)
        ]
        
        # Check boundaries - return None for out-of-bounds
        valid_neighbors = []
        for ni, nj in neighbors:
            if 0 <= ni < H and 0 <= nj < W:
                valid_neighbors.append((ni, nj))
            else:
                valid_neighbors.append(None)
        
        return valid_neighbors
    
    def get_action_and_value(
        self, 
        x: torch.Tensor, 
        fire_mask: torch.Tensor, 
        action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List]:
        """
        Get actions for all burning cells and compute value.
        
        This is the main interface for the A3C worker during training/inference.
        
        Args:
            x: Environmental features (1, 15, H, W) - batch size must be 1
            fire_mask: Current fire mask (1, H, W) or (H, W)
            action: Optional pre-specified action for evaluation
        
        Returns:
            action_grid: Binary prediction grid (H, W)
            log_prob: Log probability of action (scalar)
            entropy: Policy entropy (scalar)
            value: State value estimate (1, 1)
            burning_cells_info: Debug info [(i, j, action_8d, log_prob_8d), ...]
        """
        # Ensure fire_mask has batch dimension
        if fire_mask.dim() == 2:
            fire_mask = fire_mask.unsqueeze(0)
        elif fire_mask.dim() == 4:
            # If fire_mask is (B, 1, H, W), squeeze channel dimension
            fire_mask = fire_mask.squeeze(1)
        
        assert fire_mask.dim() == 3, f"fire_mask must be 3D after processing, got {fire_mask.dim()}D with shape {fire_mask.shape}"
        
        B, H, W = fire_mask.shape
        assert B == 1, f"Batch size must be 1 for now, got {B}"
        
        # Get attended features and state value
        features, value = self.forward(x, fire_mask)  # (1, 128, H, W), (1, 1)
        
        # Find all burning cells
        burning_cells = self.get_burning_cells(fire_mask)
        
        # Handle edge case: no burning cells
        if len(burning_cells) == 0:
            action_grid = torch.zeros(H, W)
            log_prob = torch.tensor(0.0)
            entropy = torch.tensor(0.0)
            return action_grid, log_prob, entropy, value, []
        
        # Predict 8-neighbor spread for each burning cell
        all_log_probs = []
        all_entropies = []
        burning_cells_info = []
        action_grid = torch.zeros(H, W)
        
        for cell_idx, (i, j) in enumerate(burning_cells):
            # Get 8-neighbor logits
            logits_8d = self.predict_8_neighbors(features, i, j)  # (1, 8)
            logits_8d = logits_8d.squeeze(0)  # (8,)
            
            # Convert to probabilities via sigmoid (8 independent Bernoulli)
            probs_8d = torch.sigmoid(logits_8d)
            probs_8d = torch.clamp(probs_8d, 1e-7, 1 - 1e-7)  # Numerical stability
            
            # Sample action (or use provided action)
            if action is None:
                action_8d = torch.bernoulli(probs_8d)  # Sample from Bernoulli
            else:
                # Extract action for this cell from provided action grid
                action_8d = action[cell_idx] if isinstance(action, list) else torch.bernoulli(probs_8d)
            
            # Compute log probability (sum of log probs for 8 Bernoulli)
            log_prob_8d = (
                action_8d * torch.log(probs_8d) + 
                (1 - action_8d) * torch.log(1 - probs_8d)
            )
            log_prob_cell = log_prob_8d.sum()
            
            # Compute entropy (sum of entropies for 8 Bernoulli)
            entropy_8d = -(
                probs_8d * torch.log(probs_8d) + 
                (1 - probs_8d) * torch.log(1 - probs_8d)
            )
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
        
        # Aggregate log probs and entropies across all burning cells
        total_log_prob = torch.stack(all_log_probs).sum()
        total_entropy = torch.stack(all_entropies).sum()
        
        return action_grid, total_log_prob, total_entropy, value, burning_cells_info


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Quick test
if __name__ == '__main__':
    print("=" * 80)
    print("A3C V8 Model Test - Spatial + Channel Attention")
    print("=" * 80)
    
    # Create model
    model = A3C_PerCellModel_V8(in_channels=15)
    print(f"\nModel created successfully!")
    print(f"Total parameters: {count_parameters(model):,}")
    
    # Test forward pass
    B, H, W = 1, 282, 616
    x = torch.randn(B, 15, H, W)
    fire_mask = torch.zeros(B, H, W)
    fire_mask[0, 100:110, 200:210] = 1.0  # Simulate burning region
    
    print(f"\nTest input shape: {x.shape}")
    print(f"Fire mask shape: {fire_mask.shape}")
    print(f"Burning cells: {int(fire_mask.sum())}")
    
    # Forward pass
    features, value = model(x, fire_mask)
    print(f"\nForward pass successful!")
    print(f"Features shape: {features.shape}")
    print(f"Value: {value.item():.4f}")
    
    # Test action prediction
    action_grid, log_prob, entropy, value, info = model.get_action_and_value(x, fire_mask)
    print(f"\nAction prediction successful!")
    print(f"Action grid shape: {action_grid.shape}")
    print(f"Predicted burning cells: {int(action_grid.sum())}")
    print(f"Log prob: {log_prob.item():.4f}")
    print(f"Entropy: {entropy.item():.4f}")
    print(f"Burning cells processed: {len(info)}")
    
    print("\n" + "=" * 80)
    print("All tests passed! Model ready for training.")
    print("=" * 80)
