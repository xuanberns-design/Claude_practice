"""逐患者缓存，保存原生标签/几何与三档 FBP；断点以 meta.json 为完成标记。"""
from pathlib import Path
import numpy as np
from skimage.transform import resize
from tqdm import tqdm
from .common import read_json, write_json
from .config import prepare_signature
from .dicom_io import load_patient
from .projection import simulate_slice, normalize


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
        for z in tqdm(range(n), desc=f"Project {row['id']}"):
            clean = resize_slice(hu[z], (size, size))
            rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, row["number"], z]))
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
        write_json(meta_path, meta)


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
