"""2.5D 同患者物理邻层采样；随机采样只用于训练，评估遍历完整体积。"""
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import binary_erosion, find_objects, label as connected_components
from torch.utils.data import Dataset
from .common import read_json
from .ct_ops import (mask_unmeasured, rotate_image_np, flip_image_np, rotate_sinogram_np, flip_sinogram_np)


def context_indices(z, positions, cfg):
    radius = cfg.context_slices // 2
    if cfg.context_step_mm is None:
        return np.clip(np.arange(z - radius, z + radius + 1), 0, len(positions) - 1)
    wanted = positions[z] + np.arange(-radius, radius + 1) * cfg.context_step_mm
    return np.abs(np.asarray(positions)[:, None] - wanted[None, :]).argmin(axis=0)


def view_channel(view):
    return float(np.log2(view) / 10.0)


def sinogram_target_name(cfg):
    return "sino_clean" if cfg.photons_per_ray > 0 else "sino"


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
        self.lesions = (self._lesion_inventory() if getattr(cfg, "lesion_balanced_sampling", False)
                        else None)

    def _lesion_inventory(self):
        """Index each 3D liver-tumor component once, keeping only its occupied pixels.

        A patient is still drawn uniformly. Within a tumor-positive patient, this
        index lets a small lesion receive the same sampling probability as a large
        one instead of weighting lesions by their slice or voxel count.
        """
        inventory = []
        structure = np.ones((3, 3, 3), dtype=bool)
        for pid in self.ids:
            tumor = np.load(Path(self.cfg.cache_dir) / pid / "masks.npy", mmap_mode="r")[:, 1]
            labels, count = connected_components(tumor, structure=structure)
            patient_lesions = []
            for number, box in enumerate(find_objects(labels), start=1):
                if box is None:
                    continue
                coordinates = np.argwhere(labels[box] == number)
                if not len(coordinates):
                    continue
                coordinates += np.asarray([axis.start for axis in box])
                slices, inverse = np.unique(coordinates[:, 0], return_inverse=True)
                points = [coordinates[inverse == i, 1:].astype(np.int32, copy=False)
                          for i in range(len(slices))]
                patient_lesions.append((slices.astype(np.int32), points))
            if len(patient_lesions) != count:
                raise ValueError(f"{pid}: cached tumor component inventory is inconsistent")
            inventory.append(patient_lesions)
        return inventory

    def __len__(self):
        return self.cfg.samples_per_epoch

    def __getitem__(self, index):
        cfg = self.cfg
        # 每个样本独立确定种子，兼容 Windows spawn 和训练恢复。
        rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, self.epoch, index]))
        p = int(rng.integers(len(self.ids)))
        pid, meta = self.ids[p], self.meta[p]
        positives = meta["tumor_slices"]
        lesion_anchor = None
        if positives and rng.random() < cfg.tumor_sample_probability:
            lesions = self.lesions[p] if self.lesions is not None else []
            if lesions:
                slices, points = lesions[int(rng.integers(len(lesions)))]
                local_z = int(rng.integers(len(slices)))
                z = int(slices[local_z])
                lesion_anchor = points[local_z]
            else:
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
            points = lesion_anchor if lesion_anchor is not None else np.argwhere(foreground)
            if len(points) and (lesion_anchor is not None or rng.random() < cfg.patch_foreground_probability):
                # 旧版均匀选肿瘤内部，易漏学边界；一部分裁剪改为锚定肿瘤边缘。
                if (label[1] > 0).any() and rng.random() < getattr(cfg, "tumor_boundary_sample_probability", 0):
                    boundary = foreground & ~binary_erosion(foreground, structure=np.ones((3, 3), dtype=bool))
                    boundary_points = points[boundary[points[:, 0], points[:, 1]]]
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
        if cfg.reconstruction_mode == "dual_domain":
            return self._dual_domain_sample(rng, patient, z, v, x, target, label, label_stack)
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

    def _dual_domain_sample(self, rng, patient, z, v, x, target, label, label_stack):
        """整图样本 + 同步正弦图；增强为绕投影中心的精确镜像/90°旋转，正弦图按解析关系变换。"""
        cfg = self.cfg
        sino = patient.stack("sino", z)
        target_name = sinogram_target_name(cfg)
        sino_target = sino if target_name == "sino" else patient.stack(target_name, z)
        images = [x, target, label, label_stack]
        if cfg.augment_geometry:
            for axis in (-2, -1):
                if rng.random() < 0.5:
                    images = [flip_image_np(a, axis) for a in images]
                    sino, sino_target = flip_sinogram_np(sino, axis), flip_sinogram_np(sino_target, axis)
            k = int(rng.integers(4))
            if k:
                images = [rotate_image_np(a, k) for a in images]
                sino, sino_target = rotate_sinogram_np(sino, k), rotate_sinogram_np(sino_target, k)
        x, target, label, label_stack = [np.ascontiguousarray(a, dtype=np.float32) for a in images]
        # 输入只保留本视角已测角度，未测角度置零；稠密真值只用于训练损失。
        measured = mask_unmeasured(sino, cfg.sino_dense_views, v)
        return {"input": torch.from_numpy(x), "target": torch.from_numpy(target),
                "mask": torch.from_numpy(label), "mask_stack": torch.from_numpy(label_stack),
                "view": torch.tensor(view_channel(v), dtype=torch.float32),
                "sino": torch.from_numpy(measured),
                "sino_target": torch.from_numpy(np.ascontiguousarray(sino_target, dtype=np.float32)),
                "n_views": torch.tensor(int(v), dtype=torch.int64)}
