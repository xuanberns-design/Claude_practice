"""2.5D 同患者物理邻层采样；随机采样只用于训练，评估遍历完整体积。"""
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import binary_erosion
from torch.utils.data import Dataset
from .common import read_json


def context_indices(z, positions, cfg):
    radius = cfg.context_slices // 2
    if cfg.context_step_mm is None:
        return np.clip(np.arange(z - radius, z + radius + 1), 0, len(positions) - 1)
    wanted = positions[z] + np.arange(-radius, radius + 1) * cfg.context_step_mm
    return np.abs(np.asarray(positions)[:, None] - wanted[None, :]).argmin(axis=0)


def view_channel(view):
    return float(np.log2(view) / 10.0)


class PatientCache:
    def __init__(self, cfg, pid):
        self.cfg = cfg
        self.root = Path(cfg.cache_dir) / pid
        self.meta = read_json(self.root / "meta.json")
        self.arrays = {}

    def get(self, name):
        if name not in self.arrays:
            self.arrays[name] = np.load(self.root / (name + ".npy"), mmap_mode="r")
        return self.arrays[name]

    def stack(self, name, z):
        indices = context_indices(z, np.asarray(self.meta["z_mm"]), self.cfg)
        return np.array(self.get(name)[indices], dtype=np.float32, copy=True)


class TrainDataset(Dataset):
    def __init__(self, cfg, ids):
        self.cfg, self.ids, self.epoch = cfg, list(ids), 0
        self.meta = [read_json(Path(cfg.cache_dir) / p / "meta.json") for p in ids]
        self.opened = {}

    def __len__(self):
        return self.cfg.samples_per_epoch

    def __getitem__(self, index):
        cfg = self.cfg
        # 每个样本独立确定种子，兼容 Windows spawn 和训练恢复。
        rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, self.epoch, index]))
        p = int(rng.integers(len(self.ids)))
        pid, meta = self.ids[p], self.meta[p]
        positives = meta["tumor_slices"]
        if positives and rng.random() < cfg.tumor_sample_probability:
            z = int(rng.choice(positives))
        else:
            z = int(rng.integers(meta["n_slices"]))
        v = int(rng.choice(cfg.views, p=np.asarray(cfg.view_weights) / sum(cfg.view_weights)))
        if pid not in self.opened:
            self.opened[pid] = PatientCache(cfg, pid)
        patient = self.opened[pid]
        x, target = patient.stack(f"fbp_{v}", z), patient.stack("target", z)
        label = np.array(patient.get("masks")[z], dtype=np.float32, copy=True)
        indices = context_indices(z, np.asarray(meta["z_mm"]), cfg)
        label_stack = np.array(patient.get("masks")[indices], dtype=np.float32, copy=True)
        if cfg.patch_size and cfg.patch_size < x.shape[-1]:
            ps = cfg.patch_size
            foreground = label[1] > 0
            if not foreground.any():
                foreground = label[0] > 0
            points = np.argwhere(foreground)
            if len(points) and rng.random() < cfg.patch_foreground_probability:
                # 旧版均匀选肿瘤内部，易漏学边界；一部分裁剪改为锚定肿瘤边缘。
                if (label[1] > 0).any() and rng.random() < getattr(cfg, "tumor_boundary_sample_probability", 0):
                    boundary = foreground & ~binary_erosion(foreground, structure=np.ones((3, 3), dtype=bool))
                    boundary_points = np.argwhere(boundary)
                    if len(boundary_points):
                        points = boundary_points
                cy, cx = points[int(rng.integers(len(points)))]
                jitter = int(round(ps * getattr(cfg, "patch_center_jitter_fraction", 0.5)))
                dy = int(rng.integers(-jitter, jitter + 1)) if jitter else 0
                dx = int(rng.integers(-jitter, jitter + 1)) if jitter else 0
                y0 = int(np.clip(cy - ps//2 + dy, 0, x.shape[-2]-ps))
                x0 = int(np.clip(cx - ps//2 + dx, 0, x.shape[-1]-ps))
            else:
                y0, x0 = int(rng.integers(x.shape[-2]-ps+1)), int(rng.integers(x.shape[-1]-ps+1))
            x,target,label,label_stack = [a[...,y0:y0+ps,x0:x0+ps] for a in (x,target,label,label_stack)]
        # 三者同步变换，标签不做线性插值；轻度几何增强不改变HU。
        if rng.random() < 0.5:
            x, target, label, label_stack = [a[..., ::-1] for a in (x, target, label, label_stack)]
        if rng.random() < 0.5:
            x, target, label, label_stack = [a[..., ::-1, :] for a in (x, target, label, label_stack)]
        k = int(rng.integers(4))
        x, target, label, label_stack = [np.rot90(a, k, axes=(-2, -1)).copy() for a in (x, target, label, label_stack)]
        return {"input": torch.from_numpy(x), "target": torch.from_numpy(target),
                "mask": torch.from_numpy(label), "mask_stack": torch.from_numpy(label_stack),
                "view": torch.tensor(view_channel(v), dtype=torch.float32)}
