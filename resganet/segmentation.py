from __future__ import annotations

import torch
from torch import nn, Tensor

from .blocks import ResGANetBlock


class ResGANetSegmentation(nn.Module):
    """ResGANet-UNet 分割网络（Encoder-Decoder + Skip Connections）。
    
    编码器：ResGANet-50 的前四个 stage
    解码器：上采样 + 跳跃连接 + 卷积
    """

    def __init__(self, num_classes: int = 1, groups: int = 4, reduction: int = 16):
        super().__init__()
        self.in_channels = 64

        # Encoder (Stem)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Encoder Stages
        self.layer1 = self._make_layer(ResGANetBlock, 64, blocks=3, stride=1, groups=groups, reduction=reduction)
        self.layer2 = self._make_layer(ResGANetBlock, 128, blocks=4, stride=2, groups=groups, reduction=reduction)
        self.layer3 = self._make_layer(ResGANetBlock, 256, blocks=6, stride=2, groups=groups, reduction=reduction)
        self.layer4 = self._make_layer(ResGANetBlock, 512, blocks=3, stride=2, groups=groups, reduction=reduction)

        # Decoder
        self.up4 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.dec4 = self._decoder_block(1024 + 1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec3 = self._decoder_block(512 + 512, 256)

        self.up2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.dec2 = self._decoder_block(256 + 256, 128)

        self.up1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec1 = self._decoder_block(128, 64)

        # Final upsampling to original resolution (from 1/4 to 1/1)
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

        self._init_weights()

    def _make_layer(self, block, mid_channels: int, blocks: int, stride: int, groups: int, reduction: int):
        layers = []
        out_channels = mid_channels * block.expansion
        downsample = None

        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers.append(
            block(
                in_channels=self.in_channels,
                mid_channels=mid_channels,
                stride=stride,
                num_groups=groups,
                reduction=reduction,
                downsample=downsample,
            )
        )
        self.in_channels = out_channels

        for _ in range(1, blocks):
            layers.append(
                block(
                    in_channels=self.in_channels,
                    mid_channels=mid_channels,
                    stride=1,
                    num_groups=groups,
                    reduction=reduction,
                )
            )

        return nn.Sequential(*layers)

    def _decoder_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        # Encoder
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0_pool = self.maxpool(x0)

        x1 = self.layer1(x0_pool)  # 1/4 resolution, 256 channels
        x2 = self.layer2(x1)       # 1/8 resolution, 512 channels
        x3 = self.layer3(x2)       # 1/16 resolution, 1024 channels
        x4 = self.layer4(x3)       # 1/32 resolution, 2048 channels

        # Decoder with skip connections
        d4 = self.up4(x4)          # -> 1/16 resolution, 1024 channels
        d4 = torch.cat([d4, x3], dim=1)  # 1024 + 1024 = 2048
        d4 = self.dec4(d4)         # -> 512 channels

        d3 = self.up3(d4)          # -> 1/8 resolution, 512 channels
        d3 = torch.cat([d3, x2], dim=1)  # 512 + 512 = 1024
        d3 = self.dec3(d3)         # -> 256 channels

        d2 = self.up2(d3)          # -> 1/4 resolution, 256 channels
        d2 = torch.cat([d2, x1], dim=1)  # 256 + 256 = 512
        d2 = self.dec2(d2)         # -> 128 channels

        d1 = self.up1(d2)          # -> 1/2 resolution, 128 channels
        d1 = self.dec1(d1)         # -> 64 channels

        # Final upsampling to original resolution
        out = self.final_up(d1)    # -> 1/1 resolution, 32 channels
        out = self.final_conv(out) # -> num_classes channels
        return out


__all__ = ["ResGANetSegmentation"]

