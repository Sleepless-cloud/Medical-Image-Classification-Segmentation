from __future__ import annotations

from typing import Optional

import torch
from torch import nn, Tensor

from .attention import GroupAttentionBlock


class ResGANetBlock(nn.Module):
    """ResGANet 的 Bottleneck 变体。

    结构：
    - 1x1 Conv -> BN -> ReLU （降维）
    - GroupAttentionBlock（内部含 3x3 Conv、CAM、聚合、SAM）
    - 1x1 Conv -> BN （升维）
    - 残差连接 + ReLU
    """

    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        stride: int = 1,
        num_groups: int = 4,
        reduction: int = 16,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        out_channels = mid_channels * self.expansion

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.gab = GroupAttentionBlock(
            in_channels=mid_channels,
            out_channels=mid_channels,
            num_groups=num_groups,
            stride=stride,
            reduction=reduction,
        )

        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.gab(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


__all__ = ["ResGANetBlock"]


