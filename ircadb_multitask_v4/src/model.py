"""端到端级联 2.5D U-Net：恢复相邻切片 -> 双通道分割。

plain 保留 V2 权重键；可选 RCAB 重建分支参考旧包的注意力与亚像素上采样，
但采用 GroupNorm 以适应较小的医学图像 batch。模型不限制肿瘤必须位于肝内。
"""
import math
import torch
from torch import nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        groups = math.gcd(cout, 8)
        self.layers = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                                    nn.GroupNorm(groups, cout), nn.SiLU(inplace=True),
                                    nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                                    nn.GroupNorm(groups, cout), nn.SiLU(inplace=True))

    def forward(self, x):
        return self.layers(x)


class UNet(nn.Module):
    def __init__(self, cin, cout, base):
        super().__init__()
        channels = [base * 2 ** i for i in range(5)]
        self.enc = nn.ModuleList([Block(cin if i == 0 else channels[i - 1], c) for i, c in enumerate(channels)])
        self.dec = nn.ModuleList([Block(channels[i + 1] + channels[i], channels[i]) for i in reversed(range(4))])
        self.head = nn.Conv2d(base, cout, 1)

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.enc):
            x = block(F.max_pool2d(x, 2) if i else x)
            skips.append(x)
        for block, skip in zip(self.dec, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skip], dim=1))
        return self.head(x)


class ResidualChannelAttention(nn.Module):
    """残差通道注意力块；不对空间高频做预设滤波。"""

    def __init__(self, cin, cout):
        super().__init__()
        groups = math.gcd(cout, 8)
        hidden = max(cout // 8, 4)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(groups, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, cout)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(cout, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, cout, 1),
            nn.Sigmoid(),
        )
        self.shortcut = nn.Identity() if cin == cout else nn.Conv2d(cin, cout, 1, bias=False)

    def forward(self, x):
        residual = self.norm2(self.conv2(F.silu(self.norm1(self.conv1(x)))))
        return F.silu(self.shortcut(x) + residual * self.attention(residual))


class AttentionUNet(nn.Module):
    """可选高容量重建网络；输出通道数和 plain UNet 完全相同。"""

    def __init__(self, cin, cout, base, upsample="bilinear"):
        super().__init__()
        if upsample not in ("bilinear", "pixelshuffle"):
            raise ValueError("reconstruction_upsample 必须是 bilinear 或 pixelshuffle")
        self.upsample = upsample
        channels = [base * 2 ** i for i in range(5)]
        self.enc = nn.ModuleList([
            ResidualChannelAttention(cin if i == 0 else channels[i - 1], c)
            for i, c in enumerate(channels)
        ])
        self.dec = nn.ModuleList([
            ResidualChannelAttention(channels[i] * (2 if upsample == "pixelshuffle" else 3), channels[i])
            for i in reversed(range(4))
        ])
        if upsample == "pixelshuffle":
            self.up = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(channels[i + 1], channels[i] * 4, 3, padding=1, bias=False),
                    nn.PixelShuffle(2),
                    nn.GroupNorm(math.gcd(channels[i], 8), channels[i]),
                    nn.SiLU(inplace=True),
                )
                for i in reversed(range(4))
            ])
        self.head = nn.Conv2d(base, cout, 1)

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.enc):
            x = block(F.max_pool2d(x, 2) if i else x)
            skips.append(x)
        for i, (block, skip) in enumerate(zip(self.dec, reversed(skips[:-1]))):
            x = (self.up[i](x) if self.upsample == "pixelshuffle" else
                 F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False))
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skip], dim=1))
        return self.head(x)


class JointUNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        backbone = getattr(cfg, "reconstruction_backbone", "plain")
        if backbone == "plain":
            self.reconstructor = UNet(cfg.context_slices + 1, cfg.context_slices, cfg.base_channels)
        elif backbone == "rcab":
            self.reconstructor = AttentionUNet(
                cfg.context_slices + 1, cfg.context_slices, cfg.base_channels,
                getattr(cfg, "reconstruction_upsample", "bilinear"),
            )
        else:
            raise ValueError("reconstruction_backbone 必须是 plain 或 rcab")
        self.segmenter = UNet(cfg.context_slices + 2, 2, cfg.base_channels)
        nn.init.zeros_(self.reconstructor.head.weight)
        nn.init.zeros_(self.reconstructor.head.bias)

    def forward(self, x, view):
        restored = self.restore(x, view)
        # 前向图像逐像素不变；仅限制分割损失传入重建网络的梯度。
        scale = self.cfg.seg_to_recon_scale
        seg_input = restored.detach() + scale * (restored - restored.detach())
        return restored, self.segment(seg_input, view)

    def restore(self, x, view):
        v = view.reshape(-1, 1, 1, 1).expand(-1, 1, x.shape[-2], x.shape[-1])
        restored = x + self.reconstructor(torch.cat([x, v], dim=1))
        return restored

    def segment(self, restored, view):
        cfg = self.cfg
        c = cfg.context_slices // 2
        hu = restored[:, c:c+1] * (cfg.hu_max - cfg.hu_min) + cfg.hu_min
        window = ((hu - cfg.window_min) / (cfg.window_max - cfg.window_min)).clamp(0, 1)
        v = view.reshape(-1, 1, 1, 1).expand(-1, 1, restored.shape[-2], restored.shape[-1])
        return self.segmenter(torch.cat([restored, window, v], dim=1))
