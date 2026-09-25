"""只读检查旧缓存；仅在核验原DICOM后新增padding有效区sidecar。

参数入口：Config.hu_min/hu_max、image_size、metric_body_threshold_hu。
不改写原HU、target、投影、meta.json、audit.json或患者划分。
"""
from pathlib import Path
import hashlib
import numpy as np
from pydicom.pixels import apply_modality_lut
from .common import read_json, write_json
from .config import prepare_signature
from .dicom_io import read_dicom, geometry


QUANTILES = (0.0, 0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999, 1.0)


def array_sha256(array):
    """C顺序体素字节hash，与原版ct_sha256和bool sidecar约定一致。"""
    h = hashlib.sha256()
    for a in array:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _load_array(folder, name):
    return np.load(folder / f"{name}.npy", mmap_mode="r", allow_pickle=False)


def _hu_statistics(native, cfg, valid=None):
    count, low, high, body, clip_error = 0, 0, 0, 0, 0.0
    for z, a in enumerate(native):
        if not np.isfinite(a).all():
            raise ValueError("native_hu包含NaN/Inf，不能审计")
        values = np.asarray(a) if valid is None else np.asarray(a)[valid[z]]
        count += values.size
        low += int(np.count_nonzero(values < -1024.0))
        high += int(np.count_nonzero(values > cfg.hu_max))
        body += int(np.count_nonzero(values > cfg.metric_body_threshold_hu))
        # float64累计；只是输出受HU范围约束时的范围误差，不是下采样误差。
        v = values.astype(np.float64)
        clip_error += float(np.abs(v - np.clip(v, cfg.hu_min, cfg.hu_max)).sum())
    if count == 0:
        return {"voxel_count": 0, "min": None, "max": None, "quantiles": {},
                "fraction_below_minus_1024": None, "fraction_above_hu_max": None,
                "body_fraction": None, "range_clip_mae_lower_bound_hu": None}
    # 分位数精确计算，不改变mmap的原始数组；每次只处理一名患者。
    values = np.asarray(native).ravel() if valid is None else np.asarray(native)[valid]
    quantiles = np.quantile(values, QUANTILES)
    return {"voxel_count": count, "min": float(quantiles[0]), "max": float(quantiles[-1]),
            "quantiles": {str(q): float(v) for q, v in zip(QUANTILES, quantiles)},
            "fraction_below_minus_1024": low / count, "fraction_above_hu_max": high / count,
            "body_fraction": body / count, "range_clip_mae_lower_bound_hu": clip_error / count}


def _cache_shapes(folder, meta, native, cfg):
    expected = [len(native), cfg.image_size, cfg.image_size]
    shapes, problems = {"native_hu": list(native.shape)}, []
    for name in ["target", "full_fbp"] + [f"fbp_{v}" for v in cfg.views] + ["masks", "native_masks"]:
        path = folder / f"{name}.npy"
        if not path.is_file():
            shapes[name] = None
            problems.append(f"missing_array:{name}")
            continue
        actual = list(_load_array(folder, name).shape)
        shapes[name] = actual
        desired = ([len(native), 2, *native.shape[-2:]] if name == "native_masks" else
                   [len(native), 2, cfg.image_size, cfg.image_size] if name == "masks" else expected)
        if actual != desired:
            problems.append(f"array_shape_mismatch:{name}")
    if meta.get("processed_shape_zyx") != expected:
        problems.append("meta_processed_shape_mismatch")
    if meta.get("shape_zyx") != list(native.shape) or meta.get("n_slices") != len(native):
        problems.append("meta_native_shape_mismatch")
    if meta.get("prepare_signature") != prepare_signature(cfg):
        problems.append("prepare_signature_mismatch")
    return {"actual_shapes": shapes, "configured_processed_shape_zyx": expected,
            "matches_config": not problems, "issues": problems}


def _read_source(ref, pid, index, header=False):
    try:
        return read_dicom(ref, header=header)
    except (FileNotFoundError, KeyError) as e:
        raise FileNotFoundError(
            f"{pid}: 原DICOM不可读取（meta.ct_refs第{index+1}层）；请恢复该源路径。"
            "无原DICOM时省略--with-dicom仍可诊断缓存。") from e


def _dicom_valid_mask(meta, native, native_hash, cfg):
    pid = meta["id"]
    refs = meta.get("ct_refs", [])
    if len(refs) != len(native):
        raise ValueError(f"{pid}: meta.ct_refs数量与native_hu不一致")
    headers = [(ref, str(i), _read_source(ref, pid, i, header=True)) for i, ref in enumerate(refs)]
    ordered, geom = geometry(headers, cfg)
    if [x[0] for x in ordered] != refs:
        raise ValueError(f"{pid}: ct_refs顺序与原DICOM物理位置不一致")
    for key in ("shape_zyx", "spacing_zyx", "positions_lps", "z_mm", "orientation", "affine_lps_xyz"):
        expected = meta.get(key)
        tolerance = 1e-5 if key in ("spacing_zyx", "orientation") else cfg.alignment_tolerance_mm
        if expected is None or np.shape(expected) != np.shape(geom[key]) or not np.allclose(
                expected, geom[key], atol=tolerance, rtol=0):
            raise ValueError(f"{pid}: 原DICOM几何与缓存meta不一致: {key}")
    valid = np.ones(native.shape, dtype=bool)
    source_hash, tagged, ranged = hashlib.sha256(), 0, 0
    for z, ref in enumerate(refs):
        d = _read_source(ref, pid, z)
        if not hasattr(d, "RescaleSlope") or not hasattr(d, "RescaleIntercept") or float(d.RescaleSlope) == 0:
            raise ValueError(f"{pid}: 原DICOM缺少有效HU标定")
        raw = d.pixel_array
        hu = apply_modality_lut(raw, d).astype(np.float32)
        if hu.shape != native[z].shape or not np.array_equal(hu, native[z]):
            raise ValueError(f"{pid}: 第{z+1}层原DICOM内容与native_hu不一致，拒绝生成有效区")
        source_hash.update(np.ascontiguousarray(hu).tobytes())
        if hasattr(d, "PixelPaddingRangeLimit") and not hasattr(d, "PixelPaddingValue"):
            raise ValueError(f"{pid}: PixelPaddingRangeLimit缺少PixelPaddingValue")
        if hasattr(d, "PixelPaddingValue"):
            tagged += 1
            first = int(d.PixelPaddingValue)
            last = int(getattr(d, "PixelPaddingRangeLimit", first))
            ranged += int(hasattr(d, "PixelPaddingRangeLimit"))
            lo, hi = sorted((first, last))
            # 标签值定义在stored pixel域，必须在rescale之前匹配；端点都包含。
            valid[z] = ~((raw >= lo) & (raw <= hi))
    if source_hash.hexdigest() != native_hash:
        raise ValueError(f"{pid}: 原DICOM体积hash与native_hu不一致")
    padding = {"source": "dicom_stored_pixel_tags" if tagged else "dicom_no_padding_tags_all_valid",
               "dicom_verified": True, "tagged_slices": tagged, "range_tagged_slices": ranged,
               "missing_tag_slices": len(native) - tagged,
               "padding_voxels": int(valid.size - np.count_nonzero(valid)),
               "valid_voxels": int(np.count_nonzero(valid))}
    return valid, padding


def _save_valid(path, valid):
    if path.exists():
        old = np.load(path, mmap_mode="r", allow_pickle=False)
        if old.dtype != np.dtype(bool) or old.shape != valid.shape or not np.array_equal(old, valid):
            raise FileExistsError("已有native_valid.npy与本次DICOM有效区不同，拒绝覆盖；请检查源DICOM及旧审计")
        return
    # exclusive创建，不覆盖其他审计已写入的mask；审计JSON在mask完成后写入。
    with path.open("xb") as f:
        np.save(f, valid, allow_pickle=False)


def _verified_previous_valid(folder, pid, native, native_hash, previous, audit_fingerprint):
    path = folder / "native_valid.npy"
    if not path.exists():
        return None, {"source": "not_checked", "dicom_verified": False}
    rows = [r for r in previous.get("patients", []) if r.get("id") == pid]
    if (previous.get("audit_fingerprint") != audit_fingerprint or len(rows) != 1
            or rows[0].get("native_ct_sha256") != native_hash
            or not rows[0].get("padding", {}).get("dicom_verified")):
        raise ValueError(f"{pid}: native_valid.npy没有匹配的已核验审计，请使用--with-dicom重新核验")
    valid = np.load(path, mmap_mode="r", allow_pickle=False)
    if (valid.dtype != np.dtype(bool) or valid.shape != native.shape
            or array_sha256(valid) != rows[0].get("paddingmask_sha256")):
        raise ValueError(f"{pid}: native_valid.npy内容/hash与已核验审计不一致")
    # 只转写本模块已知的统计字段，不复制路径或其他DICOM个人字段。
    fields = ("source", "dicom_verified", "tagged_slices", "range_tagged_slices",
              "missing_tag_slices", "padding_voxels", "valid_voxels")
    return valid, {k: rows[0]["padding"][k] for k in fields}


def inspect_cache(cfg, output=None, with_dicom=False):
    """返回并保存quality_audit.json；with_dicom=True时额外新增native_valid.npy。

    默认不要求源DICOM仍在本机。output可指定额外JSON文件或报告目录，
    cache/quality_audit.json始终保留，供评估校验sidecar数据来源。
    """
    root = Path(cfg.cache_dir)
    audit = read_json(root / "audit.json")
    destination = root / "quality_audit.json"
    extra = None
    if output is not None:
        extra = Path(output)
        if extra.suffix.lower() != ".json":
            extra = extra / "quality_audit.json"
        protected = {root / "audit.json", root / "splits.json"}
        protected.update(root / r["id"] / "meta.json" for r in audit["patients"])
        if getattr(cfg, "split_path", None):
            protected.add(Path(cfg.split_path))
        if extra.resolve() in {p.resolve() for p in protected}:
            raise ValueError("output不能覆盖原始audit/meta/splits文件")
    previous = read_json(destination) if destination.is_file() else {}
    rows, seen = [], set()
    for item in audit["patients"]:
        pid = item["id"]
        if pid in seen or not isinstance(pid, str) or Path(pid).name != pid or pid in (".", "..") or "/" in pid or "\\" in pid:
            raise ValueError("审计中患者匿名id重复或格式错误")
        seen.add(pid)
        folder = root / pid
        meta = read_json(folder / "meta.json")
        if meta.get("id") != pid or meta.get("audit_fingerprint") != audit["fingerprint"]:
            raise ValueError(f"{pid}: 缓存meta与原始审计不一致")
        native = _load_array(folder, "native_hu")
        if native.ndim != 3 or native.dtype != np.dtype("float32") or not native.size:
            raise ValueError(f"{pid}: native_hu应为非空float32三维数组")
        native_hash = array_sha256(native)
        if native_hash != meta.get("ct_sha256") or native_hash != item.get("ct_sha256"):
            raise ValueError(f"{pid}: native_hu内容/hash与原始审计不一致")
        statistics = _hu_statistics(native, cfg)
        consistency = _cache_shapes(folder, meta, native, cfg)
        if with_dicom:
            valid, padding = _dicom_valid_mask(meta, native, native_hash, cfg)
            _save_valid(folder / "native_valid.npy", valid)
        else:
            valid, padding = _verified_previous_valid(folder, pid, native, native_hash, previous, audit["fingerprint"])
        row = {"id": pid, "native_ct_sha256": native_hash,
               "paddingmask_sha256": array_sha256(valid) if valid is not None else None,
               "native_hu": statistics, "cache_consistency": consistency, "padding": padding}
        if valid is not None:
            row["valid_native_hu"] = _hu_statistics(native, cfg, valid)
        rows.append(row)
    result = {"schema": 1, "audit_fingerprint": audit["fingerprint"], "with_dicom": bool(with_dicom),
              "config": {"image_size": cfg.image_size, "hu_min": cfg.hu_min, "hu_max": cfg.hu_max,
                         "body_threshold_hu": cfg.metric_body_threshold_hu},
              "hash_definition": "SHA256 of C-order array voxel bytes; native=float32, native_valid=bool",
              "range_clip_note": "范围clip MAE下限只适用于输出限制在[hu_min,hu_max]的假设；它不是实际模型"
                                 "（可能未限幅）的必然下限，也不含下采样、条纹或网络误差，不能据此确定全部原因。",
              "padding_note": "只排除DICOM stored pixel域明确padding标签；无标签层全部有效，不自动排除任意低HU。",
              "patients": rows}
    write_json(destination, result)
    if extra is not None:
        if extra.resolve() != destination.resolve():
            write_json(extra, result)
    return result
