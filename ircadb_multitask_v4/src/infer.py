"""无专家mask也可推理：输入已做FBP的HU体积，不接收未知几何原始投影。"""
from pathlib import Path
import numpy as np
import torch
import nibabel as nib
from .common import device_for, write_json
from .evaluate import load_checkpoint, native_resize
from .prepare import resize_slice
from .projection import normalize, denormalize
from .dataset import context_indices, view_channel
from .predict import predict_batch


@torch.inference_mode()
def infer_nifti(cfg, checkpoint, input_path, view, output):
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
    rec, probs = [], []
    for start in range(0, len(x), cfg.batch_size):
        batch = np.stack([x[context_indices(z, z_positions, cfg)] for z in range(start, min(start+cfg.batch_size, len(x)))])
        batch = torch.from_numpy(batch).to(device)
        r, probability = predict_batch(model,batch,torch.full((len(batch),),view_channel(view),device=device),cfg)
        rec.append(r.cpu().numpy())
        probs.append(probability.cpu().numpy())
    restored = denormalize(native_resize(np.concatenate(rec), hu.shape[-2:]), cfg)
    probability = native_resize(np.concatenate(probs), hu.shape[-2:])
    dest = Path(output)
    dest.mkdir(parents=True, exist_ok=True)
    volumes = {"restored_hu": restored, "liver": (probability[:, 0] >= cfg.segmentation_threshold).astype(np.uint8),
               "tumor": (probability[:, 1] >= cfg.segmentation_threshold).astype(np.uint8)}
    for name, volume in volumes.items():
        out = nib.Nifti1Image(volume.transpose(2, 1, 0), img.affine)
        out.header.set_xyzt_units("mm")
        nib.save(out, dest / (name + ".nii.gz"))
    write_json(dest / "inference.json", {"input": str(Path(input_path).resolve()), "views": view,
                                         "checkpoint_epoch": checkpoint_data["epoch"], "output_unit": "HU",
                                         "warning": "输入必须是匹配训练协议的稀疏FBP；真实扫描域泛化未验证"})
