"""
U-Net Model for Spatial Fire Spread Prediction

U-Net architecture with skip connections for preserving spatial details.
Designed for dense binary prediction (which cells will burn next).
"""
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two consecutive conv layers with BatchNorm and ReLU."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Handle different spatial sizes
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                     diffY // 2, diffY - diffY // 2])

        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net for spatial fire prediction.

    Input: (B, 14, H, W) - environmental features
    Output: (B, 1, H, W) - burn probability per cell
    """
    def __init__(self, in_channels=14, out_channels=1):
        super().__init__()

        # Encoder (downsampling path) - reduced channels
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)

        # Decoder (upsampling path)
        self.up1 = Up(256, 128)
        self.up2 = Up(128, 64)
        self.up3 = Up(64, 32)

        # Output layer
        self.outc = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder with skip connections
        x1 = self.inc(x)      # 32 channels
        x2 = self.down1(x1)   # 64 channels
        x3 = self.down2(x2)   # 128 channels
        x4 = self.down3(x3)   # 256 channels (bottleneck)

        # Decoder with skip connections
        x = self.up1(x4, x3)  # 128 channels
        x = self.up2(x, x2)   # 64 channels
        x = self.up3(x, x1)   # 32 channels

        # Output
        logits = self.outc(x)  # 1 channel
        return logits
