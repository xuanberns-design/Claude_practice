"""可复现的二维平行束投影模拟与 FBP；不冒充扫描仪原始正弦图。

参数入口：config.py 的 full_views、views、photons_per_ray、fbp_filter。
投影在非负线性衰减系数域计算，包含像素尺寸，不在肝窗图像上投影。
"""
import numpy as np
from skimage.transform import radon, iradon


def dense_indices(cfg):
    return np.arange(0, cfg.full_views, cfg.full_views // cfg.sino_dense_views)


def to_sinogram_units(sino, pixel_mm, cfg):
    """线积分 -> 水等效像素路径长，并转为 [A, D]；FBP(s) = HU/1000 + 1。"""
    return (np.asarray(sino, np.float64).T / (pixel_mm * cfg.mu_water_per_mm)).astype(np.float32)


def dense_sinograms(hu, pixel_mm, cfg, rng):
    """仅补充双域缓存：返回(测量, 无噪声)稠密正弦图 [A, D]。

    无噪声时只投影稠密角度；有噪声时复现 simulate_slice 的完整投影和同一随机流，
    保证与已缓存的稀疏FBP来自同一次“测量”。
    """
    if cfg.photons_per_ray > 0:
        return simulate_slice(hu, pixel_mm, cfg, rng, return_sinogram=True)[2]
    theta = dense_indices(cfg) * (180.0 / cfg.full_views)
    mu = np.maximum(hu / 1000.0 + 1.0, 0.0).astype(np.float32) * cfg.mu_water_per_mm
    clean = to_sinogram_units(radon(mu, theta=theta, circle=False, preserve_range=True) * pixel_mm, pixel_mm, cfg)
    return clean, clean


def simulate_slice(hu, pixel_mm, cfg, rng, return_sinogram=False):
    theta = np.arange(cfg.full_views, dtype=np.float64) * (180.0 / cfg.full_views)
    mu = np.maximum(hu / 1000.0 + 1.0, 0.0).astype(np.float32) * cfg.mu_water_per_mm
    sino = radon(mu, theta=theta, circle=False, preserve_range=True) * pixel_mm

    def fbp(p, angles):
        mu_hat = iradon(p, theta=angles, output_size=hu.shape[0], filter_name=cfg.fbp_filter,
                        interpolation="linear", circle=False, preserve_range=True) / pixel_mm
        return ((mu_hat / cfg.mu_water_per_mm - 1.0) * 1000.0).astype(np.float32)

    full = fbp(sino, theta)
    measured = sino
    if cfg.photons_per_ray > 0:
        expected = cfg.photons_per_ray * np.exp(-sino.astype(np.float64))
        counts = rng.poisson(expected).astype(np.float64)
        if cfg.readout_noise_std:
            counts += rng.normal(0, cfg.readout_noise_std, counts.shape)
        measured = -np.log(np.maximum(counts, 1.0) / cfg.photons_per_ray)
    sparse = {}
    for v in cfg.views:
        idx = np.arange(0, cfg.full_views, cfg.full_views // v)
        sparse[int(v)] = fbp(measured[:, idx], theta[idx])
    if return_sinogram:
        dense = dense_indices(cfg)
        return full, sparse, (to_sinogram_units(measured[:, dense], pixel_mm, cfg),
                              to_sinogram_units(sino[:, dense], pixel_mm, cfg))
    return full, sparse


def normalize(hu, cfg, clip=False):
    x = (hu - cfg.hu_min) / (cfg.hu_max - cfg.hu_min)
    return np.clip(x, 0, 1).astype(np.float32) if clip else x.astype(np.float32)


def denormalize(x, cfg):
    return x * (cfg.hu_max - cfg.hu_min) + cfg.hu_min
