"""逐患者缓存，保存原生标签/几何与三档 FBP；断点以 meta.json 为完成标记。

V4 双域模式额外保存稠密角度正弦图 sino.npy [n, A, D]（噪声实验另存 sino_clean.npy）。
已有 V3 缓存无需重做投影：再次运行 prepare 会从缓存的 native_hu 就地补充正弦图，
并用 skimage iradon 复核“正弦图子集的FBP == 已缓存稀疏FBP”，不一致即停止。
"""
from pathlib import Path
import numpy as np
from skimage.transform import resize, iradon
from tqdm import tqdm
from .common import read_json, write_json
from .config import prepare_signature
from .ct_ops import detector_count
from .dicom_io import load_patient
from .projection import simulate_slice, normalize, dense_sinograms, denormalize


def resize_slice(a, shape, mask=False):
    if tuple(a.shape) == tuple(shape):
        return a.copy()
    return resize(a, shape, order=0 if mask else 1, preserve_range=True,
                  anti_aliasing=not mask).astype(a.dtype)


def prepare(cfg, only=None):
    manifest = read_json(Path(cfg.cache_dir) / "audit.json")
    signature = prepare_signature(cfg)
    for row in manifest["patients"]:
        if only and row["id"] not in only:
            continue
        dest = Path(cfg.cache_dir) / row["id"]
        dest.mkdir(parents=True, exist_ok=True)
        meta_path = dest / "meta.json"
        if meta_path.exists():
            old = read_json(meta_path)
            if (old["prepare_signature"] != signature or old["audit_fingerprint"] != manifest["fingerprint"] or old["ct_sha256"] != row["ct_sha256"]
                    or old["mask_sha256"] != row["mask_sha256"]):
                raise ValueError("缓存配置/数据变更，请使用新的 cache_dir 并重新 audit/split/prepare")
            if needs_sinogram(cfg, dest, old):
                add_sinograms(cfg, dest, old)
            else:
                print(f"复用完整缓存: {row['id']}", flush=True)
            continue
        hu, masks, meta = load_patient(row["folder"], row["number"], cfg)
        if meta["ct_sha256"] != row["ct_sha256"] or meta["mask_sha256"] != row["mask_sha256"]:
            raise ValueError("audit 后源 DICOM 已改变，停止处理")
        n, h, w = hu.shape
        size = cfg.image_size
        shape = (n, size, size)
        np.save(dest / "native_hu.npy", hu)
        np.save(dest / "native_masks.npy", masks)
        arrays = {key: np.lib.format.open_memmap(dest / f"{key}.npy", mode="w+", dtype="float32", shape=shape)
                  for key in ["target", "full_fbp"] + [f"fbp_{v}" for v in cfg.views]}
        labels = np.lib.format.open_memmap(dest / "masks.npy", mode="w+", dtype="uint8", shape=(n, 2, size, size))
        pixel_mm = meta["spacing_zyx"][2] * w / size
        dual = cfg.reconstruction_mode == "dual_domain"
        if dual:
            sinos = _open_sinograms(cfg, dest, n)
        for z in tqdm(range(n), desc=f"Project {row['id']}"):
            clean = resize_slice(hu[z], (size, size))
            rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, row["number"], z]))
            if dual:
                full, sparse, (measured, noiseless) = simulate_slice(clean, pixel_mm, cfg, rng, return_sinogram=True)
                sinos["sino"][z] = measured
                if "sino_clean" in sinos:
                    sinos["sino_clean"][z] = noiseless
            else:
                full, sparse = simulate_slice(clean, pixel_mm, cfg, rng)
            arrays["target"][z] = normalize(clean, cfg, clip=True)
            arrays["full_fbp"][z] = normalize(full, cfg)
            for v in cfg.views:
                arrays[f"fbp_{v}"][z] = normalize(sparse[v], cfg)
            for c in range(2):
                labels[z, c] = resize_slice(masks[z, c], (size, size), mask=True)
        for arr in arrays.values():
            arr.flush()
        labels.flush()
        meta.update({"prepare_signature": signature, "processed_shape_zyx": list(shape),
                     "processed_pixel_mm": pixel_mm, "audit_fingerprint": manifest["fingerprint"],
                     "simulation": "parallel_beam_from_ct", "full_reference": "noiseless_simulated_FBP"})
        if dual:
            for arr in sinos.values():
                arr.flush()
            del sinos
            meta["sinogram"] = sinogram_signature(cfg)
            check_sinogram_consistency(cfg, dest, meta)
        write_json(meta_path, meta)


def sinogram_signature(cfg):
    return {"dense_views": int(cfg.sino_dense_views), "detector_count": detector_count(cfg.image_size),
            "full_views": int(cfg.full_views), "layout": "[slice, angle, detector]",
            "units": "line_integral/(pixel_mm*mu_water_per_mm)",
            "noisy": bool(cfg.photons_per_ray > 0), "fbp_filter": cfg.fbp_filter}


def sinogram_files(cfg):
    return ["sino"] + (["sino_clean"] if cfg.photons_per_ray > 0 else [])


def needs_sinogram(cfg, dest, meta):
    if cfg.reconstruction_mode != "dual_domain":
        return False
    if meta.get("sinogram") == sinogram_signature(cfg) and all((dest / f"{k}.npy").is_file() for k in sinogram_files(cfg)):
        return False
    return True


def _open_sinograms(cfg, dest, n, suffix=""):
    shape = (n, cfg.sino_dense_views, detector_count(cfg.image_size))
    return {k: np.lib.format.open_memmap(dest / f"{k}{suffix}.npy", mode="w+", dtype="float32", shape=shape)
            for k in sinogram_files(cfg)}


def add_sinograms(cfg, dest, meta):
    """从缓存 native_hu 就地补充双域正弦图；与原投影使用同一缩放、像素尺寸与随机流。"""
    hu = np.load(dest / "native_hu.npy", mmap_mode="r")
    n, _, w = hu.shape
    size = cfg.image_size
    pixel_mm = meta["spacing_zyx"][2] * w / size
    if not np.isclose(pixel_mm, meta["processed_pixel_mm"]):
        raise ValueError("缓存像素尺寸与 native_hu 推导值不一致，拒绝补充正弦图")
    sinos = _open_sinograms(cfg, dest, n, ".tmp")
    for z in tqdm(range(n), desc=f"Sinogram {meta['id']}"):
        clean = resize_slice(np.asarray(hu[z]), (size, size))
        rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, meta["number"], z]))
        measured, noiseless = dense_sinograms(clean, pixel_mm, cfg, rng)
        sinos["sino"][z] = measured
        if "sino_clean" in sinos:
            sinos["sino_clean"][z] = noiseless
    for arr in sinos.values():
        arr.flush()
    # Windows 下必须先释放全部 memmap 句柄才能重命名。
    del arr, sinos
    for key in sinogram_files(cfg):
        (dest / f"{key}.tmp.npy").replace(dest / f"{key}.npy")
    meta["sinogram"] = sinogram_signature(cfg)
    check_sinogram_consistency(cfg, dest, meta)
    write_json(dest / "meta.json", meta)


def check_sinogram_consistency(cfg, dest, meta, tolerance_hu=1.0):
    """正弦图测量子集的 skimage FBP 必须复现已缓存稀疏 FBP（抽查中间层）。"""
    sino = np.load(dest / "sino.npy", mmap_mode="r")
    z = meta["n_slices"] // 2
    a = cfg.sino_dense_views
    theta = np.arange(a) * (180.0 / a)
    for v in cfg.views:
        idx = np.arange(0, a, a // v)
        mu = iradon(np.asarray(sino[z, idx], np.float64).T, theta=theta[idx], output_size=cfg.image_size,
                    filter_name=cfg.fbp_filter, interpolation="linear", circle=False, preserve_range=True)
        rebuilt = (mu - 1.0) * 1000.0
        cached = denormalize(np.load(dest / f"fbp_{v}.npy", mmap_mode="r")[z].astype(np.float64), cfg)
        error = float(np.abs(rebuilt - cached).max())
        if error > tolerance_hu:
            raise ValueError(f"{meta['id']} 正弦图与缓存FBP不一致（{v}视角最大误差 {error:.3f} HU）；"
                             "请确认 image_size/full_views/mu/seed 与建缓存时相同")


def verify_cache(cfg, ids):
    audit_fingerprint = read_json(Path(cfg.cache_dir) / "audit.json")["fingerprint"]
    for pid in ids:
        folder = Path(cfg.cache_dir) / pid
        meta = read_json(folder / "meta.json")
        if meta["prepare_signature"] != prepare_signature(cfg) or meta["audit_fingerprint"] != audit_fingerprint:
            raise ValueError(f"缓存与配置不一致: {pid}")
        for key in ["target", "masks", "full_fbp", "native_hu", "native_masks"] + [f"fbp_{v}" for v in cfg.views]:
            if not (folder / f"{key}.npy").is_file():
                raise FileNotFoundError(folder / f"{key}.npy")
        if needs_sinogram(cfg, folder, meta):
            raise FileNotFoundError(f"{pid} 缺少与当前配置一致的双域正弦图；请先对同一 cache_dir 运行 prepare 补充")
