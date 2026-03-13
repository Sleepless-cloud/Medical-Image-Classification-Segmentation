from __future__ import annotations

import torch
from torch import nn, Tensor


class ChannelAttentionModule(nn.Module):
    """通道注意力模块（CAM）。

    实现思路：
    - 全局平均池化获得 [B, C, 1, 1]
    - 两层 MLP：C -> C//reduction -> C，使用 ReLU + Sigmoid
    - 将权重与输入逐通道相乘
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        if reduction <= 0:
            raise ValueError("reduction 必须为正整数")
        reduced_channels = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        weights = self.mlp(self.pool(x))
        return x * weights


class SpatialAttentionModule(nn.Module):
    """空间注意力模块（SAM）。

    实现思路：
    - 沿通道维分别做均值池化/最大池化，得到两个 [B, 1, H, W]
    - 拼接为 [B, 2, H, W]，经 7x7 卷积 -> [B, 1, H, W]，Sigmoid 作为空间权重
    - 与输入逐空间位置相乘
    """

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        if kernel_size not in (3, 7):
            # 兼容常用大小，确保 padding 合理
            raise ValueError("kernel_size 仅支持 3 或 7")
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.activation = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        mean_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.activation(self.conv(torch.cat([mean_map, max_map], dim=1)))
        return x * attn


class GroupAttentionBlock(nn.Module):
    """分组注意力块（GAB）。

    步骤：
    1) 将通道均分为 `num_groups` 组；每组经过 3x3 Conv -> BN -> ReLU 的变换
    2) 每组通过 `ChannelAttentionModule` 获取通道权重
    3) 沿通道拼接后，经 1x1 卷积聚合到 `out_channels`
    4) 通过 `SpatialAttentionModule` 获取空间权重
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int = 4,
        stride: int = 1,
        reduction: int = 16,
    ) -> None:
        super().__init__()
        if in_channels % num_groups != 0:
            raise ValueError("in_channels 必须能被 num_groups 整除")

        self.num_groups = num_groups
        self.group_channels = in_channels // num_groups

        # 组内变换：共享结构，不共享权重，因此用 ModuleList
        self.group_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.group_channels, self.group_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(self.group_channels),
                nn.ReLU(inplace=True),
            )
            for _ in range(num_groups)
        ])

        self.group_cams = nn.ModuleList([
            ChannelAttentionModule(self.group_channels, reduction=reduction) for _ in range(num_groups)
        ])

        # 聚合：先拼接为 in_channels 再降/升到 out_channels
        self.aggregate = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.sam = SpatialAttentionModule(kernel_size=7)

    def forward(self, x: Tensor) -> Tensor:
        # 按组切分通道
        group_slices = torch.chunk(x, chunks=self.num_groups, dim=1)
        processed_groups = []
        for idx, g in enumerate(group_slices):
            g = self.group_transforms[idx](g)
            g = self.group_cams[idx](g)
            processed_groups.append(g)

        x_cat = torch.cat(processed_groups, dim=1)
        x_agg = self.aggregate(x_cat)
        x_out = self.sam(x_agg)
        return x_out


__all__ = [
    "ChannelAttentionModule",
    "SpatialAttentionModule",
    "GroupAttentionBlock",
]


