"""
Attention Gate U-Net for Wildfire Spread Prediction - 16 Channels

Improvements over V2:
1. Attention Gates on skip connections (focus on fire boundary regions)
2. Multi-timestep prediction (t+1, t+2, t+3)
3. Focal loss for class imbalance

Architecture inspired by: "Attention U-Net: Learning Where to Look for the Pancreas"
Adapted for wildfire spread prediction with spatial attention on fire boundaries.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
    """
    Attention Gate module for skip connections

    Computes attention coefficients to suppress irrelevant regions
    and highlight salient features (fire boundaries) for decoder.

    Args:
        F_g: Number of feature maps in gating signal (from decoder)
        F_l: Number of feature maps in input (from encoder skip)
        F_int: Number of intermediate feature maps
    """

    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()

        # Transform gating signal (decoder path)
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        # Transform encoder features (skip connection)
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        # Output attention coefficients
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        Args:
            g: Gating signal from decoder (B, F_g, H, W)
            x: Encoder features to be gated (B, F_l, H, W)

        Returns:
            Attention-weighted features (B, F_l, H, W)
        """
        # Transform both inputs
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # Combine and activate
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        # Apply attention (element-wise multiplication)
        return x * psi


class DoubleConv(nn.Module):
    """(Conv -> BN -> ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpWithAttention(nn.Module):
    """Upscaling with Attention Gate then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # Upsampling
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

        # Attention gate
        # in_channels // 2 is the gating signal channels (from decoder after upsampling)
        # in_channels // 2 is also the skip connection channels (from encoder)
        self.attention = AttentionGate(F_g=in_channels // 2, F_l=in_channels // 2, F_int=in_channels // 4)

    def forward(self, x1, x2):
        """
        Args:
            x1: Decoder features to upsample (gating signal)
            x2: Encoder features from skip connection
        """
        # Upsample decoder features
        x1 = self.up(x1)

        # Pad if needed
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # Apply attention gate to encoder features
        x2_att = self.attention(g=x1, x=x2)

        # Concatenate attention-weighted encoder features with decoder features
        x = torch.cat([x2_att, x1], dim=1)

        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class AttentionUNet(nn.Module):
    """
    Attention U-Net for multi-timestep wildfire spread prediction

    Features:
    - Attention gates on all skip connections
    - Multi-timestep output (t+1, t+2, t+3)
    - 16 input channels (DEM + Weather + NDVI + FSM)

    Architecture:
    - Input: (B, 17, 30, 30) - 16 env channels + 1 fire mask
    - Output: (B, 3, 30, 30) - new burns at t+1, t+2, t+3
    """

    def __init__(self, n_channels=17, n_timesteps=3, bilinear=True):
        super(AttentionUNet, self).__init__()
        self.n_channels = n_channels
        self.n_timesteps = n_timesteps
        self.bilinear = bilinear

        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # Decoder with Attention Gates
        self.up1 = UpWithAttention(1024, 512 // factor, bilinear)
        self.up2 = UpWithAttention(512, 256 // factor, bilinear)
        self.up3 = UpWithAttention(256, 128 // factor, bilinear)
        self.up4 = UpWithAttention(128, 64, bilinear)

        # Output
        self.outc = OutConv(64, n_timesteps)

    def forward(self, x):
        # Encoder with skip connections
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder with attention-gated skip connections
        x = self.up1(x5, x4)  # Attention on x4
        x = self.up2(x, x3)   # Attention on x3
        x = self.up3(x, x2)   # Attention on x2
        x = self.up4(x, x1)   # Attention on x1

        logits = self.outc(x)
        return logits


def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """
    Focal loss for binary segmentation

    Args:
        pred: (B, 1, H, W) - logits
        target: (B, 1, H, W) - ground truth (0 or 1)
        alpha: weight for positive class
        gamma: focusing parameter

    Returns:
        Focal loss (scalar)
    """
    pred_prob = torch.sigmoid(pred)

    pred_prob = pred_prob.reshape(-1)
    target = target.reshape(-1)

    bce = F.binary_cross_entropy(pred_prob, target, reduction='none')

    p_t = pred_prob * target + (1 - pred_prob) * (1 - target)
    focal_weight = (1 - p_t) ** gamma

    alpha_t = alpha * target + (1 - alpha) * (1 - target)

    loss = alpha_t * focal_weight * bce

    return loss.mean()


def dice_loss(pred, target, smooth=1.0):
    """
    Dice loss for binary segmentation

    Args:
        pred: (B, 1, H, W) - logits
        target: (B, 1, H, W) - ground truth (0 or 1)
        smooth: smoothing factor

    Returns:
        Dice loss (scalar)
    """
    pred = torch.sigmoid(pred)

    pred = pred.reshape(-1)
    target = target.reshape(-1)

    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    return 1 - dice


def multi_timestep_loss(pred, target, focal_weight=0.7, dice_weight=0.3, alpha=0.25, gamma=2.0):
    """
    Combined loss for multi-timestep prediction

    Args:
        pred: (B, 3, H, W) - logits for t+1, t+2, t+3
        target: (B, 3, H, W) - ground truth for t+1, t+2, t+3
        focal_weight: weight for focal loss
        dice_weight: weight for dice loss
        alpha: focal loss alpha parameter
        gamma: focal loss gamma parameter

    Returns:
        Combined loss (scalar)
    """
    total_loss = 0.0

    for t in range(pred.shape[1]):
        pred_t = pred[:, t:t+1, :, :]
        target_t = target[:, t:t+1, :, :]

        focal = focal_loss(pred_t, target_t, alpha=alpha, gamma=gamma)
        dice = dice_loss(pred_t, target_t)

        timestep_loss = focal_weight * focal + dice_weight * dice

        # Weight earlier timesteps more (t+1 most important)
        timestep_weight = 1.0 / (t + 1)
        total_loss += timestep_weight * timestep_loss

    # Normalize
    total_weight = sum(1.0 / (t + 1) for t in range(pred.shape[1]))
    total_loss = total_loss / total_weight

    return total_loss
