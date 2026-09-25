"""端到端级联 2.5D 网络：稀疏视角重建 -> 双通道(liver/tumor)分割。

重建分支（reconstruction_mode）：
- image：V2/V3 的纯图像域残差 U-Net（plain 或 RCAB），权重键保持兼容；
- dual_domain（V4 推荐）：正弦图域补全缺失角度 + 已测角度硬数据一致性 -> 与 skimage 一致的
  可微 FBP -> 图像域残差精修。条纹伪影源于缺失角度，先在正弦图域补全，比仅在图像域“擦除”
  条纹更根本，也让网络输出受真实测量约束，减少幻觉结构。
分割分支（seg_backbone）：
- plain：V3 U-Net；
- resattn（V4 推荐）：残差通道注意力编码器 + 注意力门跳连 + 深监督 + 以肝概率为条件的肿瘤头。
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
from .ct_ops import ParallelBeam, angular_interpolate, measured_mask, periodic_angle_pad


def _groups(c):
    return math.gcd(c, 8)


class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        groups = _groups(cout)
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
        groups = _groups(cout)
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


class DilatedContext(nn.Module):
    """瓶颈处多膨胀率残差上下文：稀疏角条纹跨越整幅图，需要更大的感受野。"""

    def __init__(self, channels, rates=(1, 2, 4, 8)):
        super().__init__()
        groups = _groups(channels)
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(channels, channels, 3, padding=r, dilation=r, bias=False),
                          nn.GroupNorm(groups, channels), nn.SiLU(inplace=True)) for r in rates])
        self.fuse = nn.Conv2d(channels * len(rates), channels, 1, bias=False)

    def forward(self, x):
        return x + self.fuse(torch.cat([b(x) for b in self.branches], dim=1))


class AttentionUNet(nn.Module):
    """RCAB 重建网络；输出通道数和 plain UNet 完全相同。"""

    def __init__(self, cin, cout, base, upsample="bilinear", dilated_bottleneck=False):
        super().__init__()
        if upsample not in ("bilinear", "pixelshuffle"):
            raise ValueError("reconstruction_upsample 必须是 bilinear 或 pixelshuffle")
        self.upsample = upsample
        channels = [base * 2 ** i for i in range(5)]
        self.enc = nn.ModuleList([
            ResidualChannelAttention(cin if i == 0 else channels[i - 1], c)
            for i, c in enumerate(channels)
        ])
        # 旧权重没有该模块；仅在显式开启时创建，保持 V3 checkpoint 键兼容。
        if dilated_bottleneck:
            self.context = DilatedContext(channels[-1])
        self.dec = nn.ModuleList([
            ResidualChannelAttention(channels[i] * (2 if upsample == "pixelshuffle" else 3), channels[i])
            for i in reversed(range(4))
        ])
        if upsample == "pixelshuffle":
            self.up = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(channels[i + 1], channels[i] * 4, 3, padding=1, bias=False),
                    nn.PixelShuffle(2),
                    nn.GroupNorm(_groups(channels[i]), channels[i]),
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
        if hasattr(self, "context"):
            x = self.context(x)
        for i, (block, skip) in enumerate(zip(self.dec, reversed(skips[:-1]))):
            x = (self.up[i](x) if self.upsample == "pixelshuffle" else
                 F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False))
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skip], dim=1))
        return self.head(x)


def image_backbone(cfg, cin, cout):
    backbone = getattr(cfg, "reconstruction_backbone", "plain")
    if backbone == "plain":
        return UNet(cin, cout, cfg.base_channels)
    if backbone == "rcab":
        return AttentionUNet(cin, cout, cfg.base_channels, getattr(cfg, "reconstruction_upsample", "bilinear"),
                             getattr(cfg, "reconstruction_dilated_bottleneck", False))
    raise ValueError("reconstruction_backbone 必须是 plain 或 rcab")


class DualDomainReconstructor(nn.Module):
    """正弦图补全(硬数据一致性) -> 可微FBP -> 图像精修；输入/输出均为归一化 HU 的 C 层 2.5D 堆栈。"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        c = cfg.context_slices
        self.op = ParallelBeam(cfg.image_size, cfg.sino_dense_views, cfg.fbp_filter)
        # 输入：C 层线性插值正弦图 + 已测角度掩码 + 视角编码。
        self.sino_net = UNet(c + 2, c, cfg.sino_base_channels)
        # 输入：C 层稀疏FBP + C 层双域FBP + 视角编码。
        self.image_net = image_backbone(cfg, 2 * c + 1, c)
        self.head = self.image_net.head
        nn.init.zeros_(self.sino_net.head.weight)
        nn.init.zeros_(self.sino_net.head.bias)
        self.aux = {}

    def to_normalized(self, mu_ratio):
        cfg = self.cfg
        hu = (mu_ratio - 1.0) * 1000.0
        return (hu - cfg.hu_min) / (cfg.hu_max - cfg.hu_min)

    def complete_sinogram(self, sino, view, n_views):
        cfg = self.cfg
        a = cfg.sino_dense_views
        b, _, _, d = sino.shape
        sino = sino.float()
        filled = torch.empty_like(sino)
        mask = torch.zeros((b, 1, a, 1), device=sino.device, dtype=sino.dtype)
        n_views = n_views.reshape(-1).long()
        for v in torch.unique(n_views).tolist():
            sel = n_views == v
            filled[sel] = angular_interpolate(sino[sel], a // int(v))
            rows = torch.from_numpy(measured_mask(a, int(v))).to(device=sino.device, dtype=sino.dtype)
            mask[sel] = rows.view(1, a, 1)
        scale = 1.0 / cfg.image_size
        v = view.reshape(-1, 1, 1, 1).to(sino.dtype).expand(-1, 1, a, d)
        net_in = torch.cat([filled * scale, mask.expand(-1, 1, a, d), v], dim=1)
        pad = cfg.sino_angle_pad
        delta = self.sino_net(periodic_angle_pad(net_in, pad))[..., pad:pad + a, :].float() / scale
        refined = filled + delta
        if cfg.sino_data_consistency:
            refined = torch.where(mask > 0, sino, refined)
        return refined, filled, mask

    def forward(self, x, view, sino, n_views):
        if sino is None or n_views is None:
            raise ValueError("dual_domain 重建需要稀疏正弦图输入（sino, n_views）")
        refined, filled, mask = self.complete_sinogram(sino, view, n_views)
        dd = self.to_normalized(self.op.fbp(refined)).to(x.dtype)
        v = view.reshape(-1, 1, 1, 1).to(x.dtype).expand(-1, 1, x.shape[-2], x.shape[-1])
        restored = dd + self.image_net(torch.cat([x, dd, v], dim=1))
        self.aux = {"sino": refined, "sino_interp": filled, "sino_mask": mask, "dd_image": dd}
        return restored


class AttentionGate(nn.Module):
    """Attention U-Net 跳连门控：用解码器语义抑制与肝/瘤无关的高频（血管、条纹）响应。"""

    def __init__(self, g_channels, x_channels, inter):
        super().__init__()
        self.wg = nn.Conv2d(g_channels, inter, 1, bias=False)
        self.wx = nn.Conv2d(x_channels, inter, 1, bias=False)
        self.psi = nn.Conv2d(inter, 1, 1)
        nn.init.zeros_(self.psi.weight)
        nn.init.constant_(self.psi.bias, 3.0)  # 初始接近全通，训练中再学习抑制

    def forward(self, g, x):
        return x * torch.sigmoid(self.psi(F.silu(self.wg(g) + self.wx(x))))


class ResAttnSegNet(nn.Module):
    """残差注意力分割网：肝脏头 + 以肝概率为条件的肿瘤头 + 深监督辅助头。"""

    def __init__(self, cin, base, deep_supervision=True):
        super().__init__()
        channels = [base * 2 ** i for i in range(5)]
        self.enc = nn.ModuleList([ResidualChannelAttention(cin if i == 0 else channels[i - 1], c)
                                  for i, c in enumerate(channels)])
        self.context = DilatedContext(channels[-1])
        self.gates = nn.ModuleList([AttentionGate(channels[i + 1], channels[i], max(channels[i] // 2, 4))
                                    for i in reversed(range(4))])
        self.dec = nn.ModuleList([ResidualChannelAttention(channels[i + 1] + channels[i], channels[i])
                                  for i in reversed(range(4))])
        self.liver_head = nn.Conv2d(base, 1, 1)
        self.tumor_head = nn.Sequential(nn.Conv2d(base + 1, base, 3, padding=1, bias=False),
                                        nn.GroupNorm(_groups(base), base), nn.SiLU(inplace=True),
                                        nn.Conv2d(base, 1, 1))
        self.deep_supervision = deep_supervision
        if deep_supervision:
            self.aux_heads = nn.ModuleList([nn.Conv2d(channels[i], 2, 1) for i in (3, 2, 1)])
        self.aux = []

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.enc):
            x = block(F.max_pool2d(x, 2) if i else x)
            skips.append(x)
        x = self.context(x)
        aux = []
        for level, (gate, block, skip) in enumerate(zip(self.gates, self.dec, reversed(skips[:-1]))):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([x, gate(x, skip)], dim=1))
            if self.deep_supervision and self.training and level < 3:
                aux.append(self.aux_heads[level](x))
        liver = self.liver_head(x)
        tumor = self.tumor_head(torch.cat([x, torch.sigmoid(liver)], dim=1))
        self.aux = aux
        return torch.cat([liver, tumor], dim=1)


class JointUNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        c = cfg.context_slices
        mode = getattr(cfg, "reconstruction_mode", "image")
        if mode == "image":
            self.reconstructor = image_backbone(cfg, c + 1, c)
        elif mode == "dual_domain":
            self.reconstructor = DualDomainReconstructor(cfg)
        else:
            raise ValueError("reconstruction_mode 必须是 image 或 dual_domain")
        seg_in = c + 2 + (1 if getattr(cfg, "seg_wide_window", None) else 0)
        seg_backbone = getattr(cfg, "seg_backbone", "plain")
        if seg_backbone == "plain":
            self.segmenter = UNet(seg_in, 2, cfg.base_channels)
        elif seg_backbone == "resattn":
            self.segmenter = ResAttnSegNet(seg_in, cfg.seg_base_channels or cfg.base_channels,
                                           getattr(cfg, "seg_deep_supervision_weight", 0) > 0)
        else:
            raise ValueError("seg_backbone 必须是 plain 或 resattn")
        nn.init.zeros_(self.reconstructor.head.weight)
        nn.init.zeros_(self.reconstructor.head.bias)

    @property
    def dual_domain(self):
        return isinstance(self.reconstructor, DualDomainReconstructor)

    def forward(self, x, view, sino=None, n_views=None):
        restored = self.restore(x, view, sino, n_views)
        # 前向图像逐像素不变；仅限制分割损失传入重建网络的梯度。
        scale = self.cfg.seg_to_recon_scale
        seg_input = restored.detach() + scale * (restored - restored.detach())
        return restored, self.segment(seg_input, view)

    def restore(self, x, view, sino=None, n_views=None):
        if self.dual_domain:
            return self.reconstructor(x, view, sino, n_views)
        v = view.reshape(-1, 1, 1, 1).expand(-1, 1, x.shape[-2], x.shape[-1])
        restored = x + self.reconstructor(torch.cat([x, v], dim=1))
        return restored

    def reconstruction_aux(self):
        return self.reconstructor.aux if self.dual_domain else {}

    def segmentation_aux(self):
        return getattr(self.segmenter, "aux", [])

    def segment(self, restored, view):
        cfg = self.cfg
        c = cfg.context_slices // 2
        hu = restored[:, c:c+1] * (cfg.hu_max - cfg.hu_min) + cfg.hu_min
        channels = [restored, ((hu - cfg.window_min) / (cfg.window_max - cfg.window_min)).clamp(0, 1)]
        wide = getattr(cfg, "seg_wide_window", None)
        if wide:
            channels.append(((hu - wide[0]) / (wide[1] - wide[0])).clamp(0, 1))
        v = view.reshape(-1, 1, 1, 1).expand(-1, 1, restored.shape[-2], restored.shape[-1])
        return self.segmenter(torch.cat(channels + [v.to(restored.dtype)], dim=1))
