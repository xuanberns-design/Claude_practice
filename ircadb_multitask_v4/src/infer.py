"""无专家mask也可推理：输入已做FBP的HU体积，不接收未知几何原始投影。

dual_domain 模型还需要同一协议下的稀疏正弦图（--sinogram，npy [Z, views, D]，单位为
线积分/(pixel_mm*mu_water)，几何同 skimage radon(circle=False) 于 image_size 网格）。
没有正弦图时可用 --reproject-fbp 以 FBP 图重投影近似（会降低质量，仅供演示，结果中注明）。
"""
from pathlib import Path
import numpy as np
import torch
import nibabel as nib
from .common import device_for, write_json
from .evaluate import load_checkpoint, native_resize, load_postprocess_parameters
from .ct_ops import detector_count
from .postprocess import postprocess_volume
from .prepare import resize_slice
from .projection import normalize, denormalize
from .dataset import context_indices, view_channel
from .predict import predict_batch


@torch.inference_mode()
def sparse_sinogram_input(cfg, model, x, view, device, sinogram=None, reproject=False):
    """返回 [Z, A, D] 的稠密角度栅格（仅已测角度非零）。"""
    a, d = cfg.sino_dense_views, detector_count(cfg.image_size)
    stride = a // view
    dense = np.zeros((len(x), a, d), dtype=np.float32)
    if sinogram is not None:
        measured = np.load(sinogram).astype(np.float32)
        if measured.shape != (len(x), view, d):
            raise ValueError(f"--sinogram 形状应为 {(len(x), view, d)}，实际 {measured.shape}")
        dense[:, ::stride] = measured
        return dense, "measured_sinogram"
    if not reproject:
        raise ValueError("dual_domain 模型需要 --sinogram；或显式使用 --reproject-fbp 近似（质量下降）")
    op = model.reconstructor.op
    rows = torch.arange(0, a, stride, device=device)
    for start in range(0, len(x), cfg.batch_size):
        chunk = torch.from_numpy(x[start:start+cfg.batch_size]).to(device)
        mu_ratio = (denormalize(chunk, cfg) / 1000.0 + 1.0).clamp_min(0)
        dense[start:start+len(chunk), ::stride] = op.project(mu_ratio, rows).cpu().numpy()
    return dense, "reprojected_fbp_approximation"


def infer_nifti(cfg, checkpoint, input_path, view, output, sinogram=None, reproject=False):
    if view not in cfg.views:
        raise ValueError("view必须在训练视角列表中")
    img = nib.load(input_path)
    hu_xyz = img.get_fdata(dtype=np.float32)
    if hu_xyz.ndim != 3 or not np.isfinite(hu_xyz).all():
        raise ValueError("需要有限值三维HU NIfTI，轴顺序必须为[col,row,slice]")
    # 只接受与训练轴向方向相符的体积，避免误把冠状/矢状图当作轴向。
    axcodes = nib.aff2axcodes(img.affine)
    if axcodes != ("L", "P", "S"):
        raise ValueError(f"期望LPS方向的[col,row,slice]体积，实际{axcodes}。先按README受控重定向。")
    hu = hu_xyz.transpose(2, 1, 0)
    if hu.shape[1] != hu.shape[2]:
        raise ValueError("需要方形图像，不能直接拉伸非方形体积")
    spacing = img.header.get_zooms()[:3]
    if not np.isclose(spacing[0], spacing[1], atol=1e-4):
        raise ValueError("要求层内等距像素")
    x = np.stack([normalize(resize_slice(s, (cfg.image_size, cfg.image_size)), cfg) for s in hu])
    z_positions = np.arange(len(x)) * spacing[2]
    device = device_for(cfg)
    model, checkpoint_data = load_checkpoint(checkpoint, cfg, device)
    model.eval()
    sino_source = None
    if model.dual_domain:
        dense, sino_source = sparse_sinogram_input(cfg, model, x, view, device, sinogram, reproject)
    rec, probs = [], []
    for start in range(0, len(x), cfg.batch_size):
        indices = range(start, min(start+cfg.batch_size, len(x)))
        batch = np.stack([x[context_indices(z, z_positions, cfg)] for z in indices])
        batch = torch.from_numpy(batch).to(device)
        sino = n_views = None
        if model.dual_domain:
            sino = torch.from_numpy(np.stack([dense[context_indices(z, z_positions, cfg)] for z in indices])).to(device)
            n_views = torch.full((len(batch),), int(view), device=device, dtype=torch.long)
        r, probability = predict_batch(model, batch, torch.full((len(batch),), view_channel(view), device=device),
                                       cfg, sino, n_views)
        rec.append(r.cpu().numpy())
        probs.append(probability.cpu().numpy())
    restored = denormalize(native_resize(np.concatenate(rec), hu.shape[-2:]), cfg)
    probability = native_resize(np.concatenate(probs), hu.shape[-2:])
    dest = Path(output)
    dest.mkdir(parents=True, exist_ok=True)
    params, params_source = load_postprocess_parameters(cfg, checkpoint)
    liver, tumor = postprocess_volume(probability, (float(spacing[2]), float(spacing[1]), float(spacing[0])), cfg, params)
    volumes = {"restored_hu": restored, "liver": liver.astype(np.uint8), "tumor": tumor.astype(np.uint8)}
    for name, volume in volumes.items():
        out = nib.Nifti1Image(volume.transpose(2, 1, 0), img.affine)
        out.header.set_xyzt_units("mm")
        nib.save(out, dest / (name + ".nii.gz"))
    write_json(dest / "inference.json", {"input": str(Path(input_path).resolve()), "views": view,
                                         "checkpoint_epoch": checkpoint_data["epoch"], "output_unit": "HU",
                                         "sinogram_source": sino_source,
                                         "postprocess": {"enabled": cfg.postprocess, "parameters": params,
                                                         "source": params_source},
                                         "warning": "输入必须是匹配训练协议的稀疏FBP；真实扫描域泛化未验证"})
