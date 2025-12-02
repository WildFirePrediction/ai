"""
U-Net for Wildfire Spread Prediction - V3 with Dilated Ground Truth

Changes from V2:
- Same architecture
- Trains on dilated ground truth (8-neighbor tolerance)
- More realistic spatial matching for fire spread uncertainty

Input: 17 channels (16 environment + 1 current fire mask)
Output: 3 channels (new burns at t+1, t+2, t+3)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNetMultiTimestep(nn.Module):
    """
    U-Net for multi-timestep wildfire spread prediction

    Architecture:
    - Input: (B, 17, 30, 30) - 16 env channels + 1 fire mask
    - Output: (B, 3, 30, 30) - new burns at t+1, t+2, t+3
    """

    def __init__(self, n_channels=17, n_timesteps=3, bilinear=True):
        super(UNetMultiTimestep, self).__init__()
        self.n_channels = n_channels
        self.n_timesteps = n_timesteps
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_timesteps)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """
    Focal loss for binary segmentation
    Addresses class imbalance and focuses on hard examples

    Args:
        pred: (B, 1, H, W) - logits
        target: (B, 1, H, W) - ground truth (0 or 1)
        alpha: weight for positive class (default 0.25)
        gamma: focusing parameter (default 2.0)

    Returns:
        Focal loss (scalar)
    """
    pred_prob = torch.sigmoid(pred)

    # Flatten (use reshape instead of view for non-contiguous tensors)
    pred_prob = pred_prob.reshape(-1)
    target = target.reshape(-1)

    # Binary cross entropy per pixel
    bce = F.binary_cross_entropy(pred_prob, target, reduction='none')

    # Focal weight: (1 - p_t)^gamma
    p_t = pred_prob * target + (1 - pred_prob) * (1 - target)
    focal_weight = (1 - p_t) ** gamma

    # Alpha weight
    alpha_t = alpha * target + (1 - alpha) * (1 - target)

    # Focal loss
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
    Uses focal loss + dice loss for each timestep

    Args:
        pred: (B, 3, H, W) - logits for t+1, t+2, t+3
        target: (B, 3, H, W) - ground truth for t+1, t+2, t+3 (dilated)
        focal_weight: weight for focal loss
        dice_weight: weight for dice loss
        alpha: focal loss alpha parameter
        gamma: focal loss gamma parameter

    Returns:
        Combined loss (scalar)
    """
    total_loss = 0.0

    # Loss for each timestep
    for t in range(pred.shape[1]):
        pred_t = pred[:, t:t+1, :, :]  # (B, 1, H, W)
        target_t = target[:, t:t+1, :, :]  # (B, 1, H, W)

        # Focal + Dice loss
        focal = focal_loss(pred_t, target_t, alpha=alpha, gamma=gamma)
        dice = dice_loss(pred_t, target_t)

        timestep_loss = focal_weight * focal + dice_weight * dice

        # Weight later timesteps slightly less (t+1 is most important)
        timestep_weight = 1.0 / (t + 1)
        total_loss += timestep_weight * timestep_loss

    # Normalize by sum of weights
    total_weight = sum(1.0 / (t + 1) for t in range(pred.shape[1]))
    total_loss = total_loss / total_weight

    return total_loss
