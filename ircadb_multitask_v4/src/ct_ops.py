"""可微平行束投影/FBP算子与正弦图工具；几何与 skimage radon/iradon(circle=False) 逐像素一致。

用途：双域重建（正弦图补全 -> 可微FBP -> 图像精修）和投影一致性损失。
坐标约定（与 skimage 相同）：
- 图像旋转中心为像素 (N//2, N//2)；探测器中心为 D//2，D = N + ceil(sqrt(2)N - N)。
- 角度 θ_k = k*180/A；p(θ+180°, u) = p(θ, -u)，即“角度周期 + 探测器翻转”。
正弦图统一存储为 [..., A, D]（角度在前），单位为“水等效像素路径长”：
s = 线积分 / (pixel_mm * mu_water)，因此 FBP(s) = mu/mu_water = HU/1000 + 1，与患者像素尺寸无关。
"""
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def detector_count(image_size):
    """skimage radon(circle=False) 的探测器单元数。"""
    return int(image_size + math.ceil(math.sqrt(2) * image_size - image_size))


def ramp_filter(size, filter_name="ramp"):
    """复制 skimage _get_fourier_filter，保证 FBP 数值一致。"""
    n = np.concatenate((np.arange(1, size / 2 + 1, 2, dtype=int), np.arange(size / 2 - 1, 0, -2, dtype=int)))
    f = np.zeros(size)
    f[0] = 0.25
    f[1::2] = -1 / (np.pi * n) ** 2
    fourier = 2 * np.real(np.fft.fft(f))
    if filter_name == "ramp":
        pass
    elif filter_name == "shepp-logan":
        omega = np.pi * np.fft.fftfreq(size)[1:]
        fourier[1:] *= np.sin(omega) / omega
    elif filter_name == "cosine":
        freq = np.linspace(0, np.pi, size, endpoint=False)
        fourier *= np.fft.fftshift(np.sin(freq))
    elif filter_name == "hamming":
        fourier *= np.fft.fftshift(np.hamming(size))
    elif filter_name == "hann":
        fourier *= np.fft.fftshift(np.hanning(size))
    else:
        raise ValueError(f"未知FBP滤波器: {filter_name}")
    return fourier


class ParallelBeam(nn.Module):
    """A 个等间隔角度（0..180°）的平行束几何；angles 以稠密角度索引选择子集。"""

    def __init__(self, image_size, n_angles, filter_name="ramp", angle_chunk=32):
        super().__init__()
        self.n = int(image_size)
        self.a = int(n_angles)
        self.d = detector_count(self.n)
        self.filter_name = filter_name
        self.angle_chunk = int(angle_chunk)
        self.padded = max(64, int(2 ** np.ceil(np.log2(2 * self.d))))
        self.register_buffer("filter", torch.tensor(ramp_filter(self.padded, filter_name), dtype=torch.float32),
                             persistent=False)
        self.register_buffer("theta", torch.arange(self.a, dtype=torch.float64) * (math.pi / self.a), persistent=False)

    # ------------------------------------------------------------------ FBP
    def filter_sinogram(self, sino):
        """沿探测器方向做斜坡滤波；sino [..., A, D]。"""
        x = F.pad(sino.float(), (0, self.padded - self.d))
        spectrum = torch.fft.fft(x, dim=-1) * self.filter
        return torch.fft.ifft(spectrum, dim=-1).real[..., :self.d]

    def _backproject_chunk(self, sino_chunk, cos, sin):
        # sino_chunk [M, a, D]；cos/sin [a]。返回 [M, N*N]。
        n, d = self.n, self.d
        r = n // 2
        coords = torch.arange(n, device=sino_chunk.device, dtype=torch.float32) - r
        rows, cols = coords[:, None], coords[None, :]
        # skimage: t = col*cos(θ) - row*sin(θ)，探测器索引 = t + D//2
        t = cols[None] * cos[:, None, None] - rows[None] * sin[:, None, None]
        index = t + d // 2
        # np.interp(left=0, right=0)：探测器范围 [0, D-1] 外严格为0，避免图像角落出现部分插值。
        gx = torch.where((index >= 0) & (index <= d - 1), index / (d - 1) * 2 - 1, torch.full_like(index, 3.0))
        count = sino_chunk.shape[1]
        gy = (torch.arange(count, device=sino_chunk.device, dtype=torch.float32) / max(count - 1, 1) * 2 - 1)
        gy = gy[:, None, None].expand_as(gx)
        grid = torch.stack([gx, gy], dim=-1).reshape(1, count, n * n, 2)
        # 通道维承载 M 个正弦图，所有通道共享同一采样网格。
        sampled = F.grid_sample(sino_chunk[None], grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        return sampled[0].sum(1)

    def backproject(self, sino, angle_index=None):
        """未滤波反投影；sino [..., a, D]，angle_index 为稠密角度索引（默认全部）。"""
        lead = sino.shape[:-2]
        count = sino.shape[-2]
        if angle_index is None:
            angle_index = torch.arange(count, device=sino.device)
        theta = self.theta.to(sino.device)[angle_index].float()
        flat = sino.reshape(-1, count, self.d).float()
        out = flat.new_zeros(flat.shape[0], self.n * self.n)
        for start in range(0, count, self.angle_chunk):
            stop = min(start + self.angle_chunk, count)
            args = (flat[:, start:stop], torch.cos(theta[start:stop]), torch.sin(theta[start:stop]))
            if torch.is_grad_enabled() and flat.requires_grad:
                # 重算采样网格以节约显存；FBP 本身是线性无参数算子。
                out = out + checkpoint(self._backproject_chunk, *args, use_reentrant=False)
            else:
                out = out + self._backproject_chunk(*args)
        return out.reshape(*lead, self.n, self.n)

    def fbp(self, sino, angle_index=None):
        """与 skimage.iradon(filter, linear, circle=False, output_size=N) 等价的可微 FBP。"""
        count = sino.shape[-2]
        with torch.autocast(device_type=sino.device.type, enabled=False):
            image = self.backproject(self.filter_sinogram(sino), angle_index)
        return image * (math.pi / (2 * count))

    # ------------------------------------------------------------------ 投影
    def project(self, image, angle_index):
        """与 skimage.radon(circle=False) 等价的可微前向投影；image [..., N, N] -> [..., a, D]。"""
        lead = image.shape[:-2]
        flat = image.reshape(-1, 1, self.n, self.n).float()
        n, d = self.n, self.d
        pad_before = d // 2 - n // 2
        center = d // 2
        theta = self.theta.to(image.device)[angle_index].float()
        coords = torch.arange(d, device=image.device, dtype=torch.float32) - center
        y, x = coords[:, None], coords[None, :]
        rows = []
        with torch.autocast(device_type=image.device.type, enabled=False):
            for c, s in zip(torch.cos(theta), torch.sin(theta)):
                # skimage warp 逆映射：x_in = cos*x + sin*y + c，y_in = -sin*x + cos*y + c（填充图坐标）
                xin = c * x + s * y + center - pad_before
                yin = -s * x + c * y + center - pad_before
                grid = torch.stack([xin / (n - 1) * 2 - 1, yin / (n - 1) * 2 - 1], dim=-1)[None]
                rotated = F.grid_sample(flat, grid.expand(flat.shape[0], -1, -1, -1), mode="bilinear",
                                        padding_mode="zeros", align_corners=True)
                rows.append(rotated.sum(-2)[:, 0])
        return torch.stack(rows, dim=1).reshape(*lead, len(theta), d)


# ---------------------------------------------------------------------- 正弦图工具
def flip_detector(sino):
    """u -> -u（绕探测器中心 D//2）；D 为偶数时越界位置补零。适用于 numpy 与 torch。"""
    d = sino.shape[-1]
    c = d // 2
    idx = 2 * c - np.arange(d)
    valid = (idx >= 0) & (idx < d)
    if isinstance(sino, torch.Tensor):
        out = torch.zeros_like(sino)
        out[..., torch.from_numpy(np.flatnonzero(valid)).to(sino.device)] = sino[..., torch.from_numpy(idx[valid]).to(sino.device)]
        return out
    out = np.zeros_like(sino)
    out[..., valid] = sino[..., idx[valid]]
    return out


def measured_stride(dense_views, views):
    if dense_views % views:
        raise ValueError("sino_dense_views 必须是每个稀疏角度数的整数倍")
    return dense_views // views


def measured_mask(dense_views, views):
    mask = np.zeros(dense_views, dtype=bool)
    mask[::measured_stride(dense_views, views)] = True
    return mask


def mask_unmeasured(sino, dense_views, views):
    """把未测量角度置零；数据集/评估/推理都必须经此函数，避免泄漏稠密投影。"""
    out = np.array(sino, dtype=np.float32, copy=True)
    out[..., ~measured_mask(dense_views, views), :] = 0
    return out


def angular_interpolate(sino, stride):
    """torch：已测角度（每 stride 行）之间做线性插值，180° 处使用探测器翻转周期延拓。"""
    a = sino.shape[-2]
    known = sino[..., ::stride, :]
    wrapped = torch.cat([known, flip_detector(known[..., :1, :])], dim=-2)
    pos = torch.arange(a, device=sino.device)
    k0 = pos // stride
    w = ((pos % stride).float() / stride).to(sino.dtype)[:, None]
    return wrapped[..., k0, :] * (1 - w) + wrapped[..., k0 + 1, :] * w


def periodic_angle_pad(sino, pad):
    """torch：角度方向周期（带探测器翻转）填充 pad 行，供卷积网络感知 0°/180° 连续性。"""
    if pad <= 0:
        return sino
    head = flip_detector(sino[..., -pad:, :])
    tail = flip_detector(sino[..., :pad, :])
    return torch.cat([head, sino, tail], dim=-2)


# ---------------------------------------------------------------------- 精确几何增强
def rotate_image_np(a, k):
    """绕 (N//2, N//2) 逆时针旋转 k*90°（与投影中心一致）。"""
    k %= 4
    n = a.shape[-1]
    shift = 2 * (n // 2) - n + 1
    for _ in range(k):
        a = np.rot90(a, 1, axes=(-2, -1))
        if shift:
            a = np.roll(a, shift, axis=-2)
    return np.ascontiguousarray(a)


def flip_image_np(a, axis):
    """绕 N//2 镜像；axis 为 -2（行）或 -1（列）。"""
    n = a.shape[axis]
    a = np.flip(a, axis=axis)
    shift = 2 * (n // 2) - n + 1
    if shift:
        a = np.roll(a, shift, axis=axis)
    return np.ascontiguousarray(a)


def rotate_sinogram_np(s, k):
    """图像旋转 k*90° 对应的正弦图变换：s'[a] = s[a - kA/2]，跨越 180° 时翻转探测器。"""
    k %= 4
    a = s.shape[-2]
    if k and a % 2:
        raise ValueError("旋转增强要求稠密角度数为偶数")
    out = np.empty_like(s)
    for row in range(a):
        q, m = divmod(row - k * a // 2, a)
        out[..., row, :] = flip_detector(s[..., m, :]) if q % 2 else s[..., m, :]
    return out


def flip_sinogram_np(s, axis):
    """图像绕中心镜像对应的正弦图变换（经 skimage 数值验证）。"""
    a = s.shape[-2]
    order = (-np.arange(a)) % a
    out = s[..., order, :]
    if axis == -2 or axis == 0:
        out[..., 1:, :] = flip_detector(out[..., 1:, :])
    else:
        out[..., :1, :] = flip_detector(out[..., :1, :])
    return np.ascontiguousarray(out)
