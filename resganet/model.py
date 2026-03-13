from __future__ import annotations

from typing import List, Type

import torch
from torch import nn, Tensor

from .blocks import ResGANetBlock


class ResGANet(nn.Module):
    """ResGANet-50 分类网络。

    基于 ResNet-50：层配置 [3, 4, 6, 3]，将 Bottleneck 替换为 ResGANetBlock。
    stem: 7x7 conv, stride=2 -> bn -> relu -> 3x3 maxpool, stride=2
    后接四个 stage，每个 stage 第一个 block 可能 stride=2 以降采样。
    """

    def __init__(self, num_classes: int = 1000, groups: int = 4, reduction: int = 16) -> None:
        super().__init__()
        self.in_channels = 64

        # Stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stages
        self.layer1 = self._make_layer(ResGANetBlock, 64, blocks=3, stride=1, groups=groups, reduction=reduction)
        self.layer2 = self._make_layer(ResGANetBlock, 128, blocks=4, stride=2, groups=groups, reduction=reduction)
        self.layer3 = self._make_layer(ResGANetBlock, 256, blocks=6, stride=2, groups=groups, reduction=reduction)
        self.layer4 = self._make_layer(ResGANetBlock, 512, blocks=3, stride=2, groups=groups, reduction=reduction)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * ResGANetBlock.expansion, num_classes)

        # 参数初始化与兼容 common 习惯
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block: Type[ResGANetBlock],
        mid_channels: int,
        blocks: int,
        stride: int,
        groups: int,
        reduction: int,
    ) -> nn.Sequential:
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

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


__all__ = ["ResGANet"]


