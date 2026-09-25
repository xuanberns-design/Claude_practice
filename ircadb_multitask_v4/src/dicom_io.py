"""DICOM/内嵌 ZIP 读取、物理坐标对齐、HU 与标签审计。

参数入口：config.py 的 *_pattern、alignment_tolerance_mm、allow_instance_fallback。
严格拒绝未知几何、多序列混合和无依据的文件名配对。
"""
from pathlib import Path
from io import BytesIO
import hashlib
import json
import re
import zipfile
import warnings
import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError
from pydicom.pixels import apply_modality_lut
from .official import SLICE_COUNTS, EXPECTED_TARGET_MASKS_BY_PATIENT
from .common import read_json, write_json, write_csv, digest


def read_dicom(ref, header=False):
    if "::" in ref:
        archive, member = ref.split("::", 1)
        with zipfile.ZipFile(archive) as z:
            return pydicom.dcmread(BytesIO(z.read(member)), stop_before_pixels=header)
    return pydicom.dcmread(ref, stop_before_pixels=header)


def entries(patient, kind):
    folder = patient / kind
    archive = patient / (kind + ".zip")
    if folder.is_dir():
        return [(str(p.resolve()), p.relative_to(folder).as_posix())
                for p in sorted(folder.rglob("*")) if p.is_file() and not p.name.startswith(".")]
    if archive.is_file():
        with zipfile.ZipFile(archive) as z:
            return [(str(archive.resolve()) + "::" + n, n)
                    for n in sorted(z.namelist()) if not n.endswith("/") and "__MACOSX" not in n]
    raise FileNotFoundError(f"缺少 {folder} 或 {archive}")


def dicom_entries(items):
    out = []
    for ref, name in items:
        try:
            d = read_dicom(ref, header=True)
        except InvalidDicomError:
            if Path(name).suffix.lower() in (".txt", ".pdf", ".xml", ".json") or Path(name).name.startswith("."):
                continue
            raise ValueError(f"无法识别的文件，不能静默忽略: {ref}")
        if hasattr(d, "Rows") and hasattr(d, "Columns"):
            if int(getattr(d, "NumberOfFrames", 1)) != 1:
                raise ValueError("本代码读取单帧切片 DICOM，不支持多帧/SEG 对象")
            out.append((ref, name, d))
    if not out:
        raise ValueError("没有找到 DICOM 切片")
    return out


def discover(root):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"数据目录不存在: {root.resolve()}")
    folders = [root] if (root / "PATIENT_DICOM").exists() or (root / "PATIENT_DICOM.zip").exists() else list(root.iterdir())
    result = []
    for p in folders:
        match = re.fullmatch(r"(?i)3D[-_]?ircadb[-_]?0?1[.\-_](\d+)", p.name)
        if p.is_dir() and match:
            number = int(match.group(1))
            if not 1 <= number <= 20:
                raise ValueError(f"异常病例编号 {p}")
            result.append((number, p))
    if not result or len({n for n, _ in result}) != len(result):
        raise ValueError("未发现患者目录或存在重复病例编号；先解压最外层 ZIP")
    return sorted(result)


def geometry(items, cfg):
    first = items[0][2]
    needed = ["ImageOrientationPatient", "ImagePositionPatient", "PixelSpacing"]
    if not all(hasattr(d, k) for _, _, d in items for k in needed):
        raise ValueError("CT 缺少物理几何标签，不能可靠排序/导出")
    orient = np.asarray(first.ImageOrientationPatient, float)
    row, col = orient[:3], orient[3:]
    normal = np.cross(row, col)
    if not np.isclose(np.linalg.norm(normal), 1, atol=1e-4) or not np.isclose(np.dot(row, col), 0, atol=1e-4):
        raise ValueError("DICOM 方向余弦异常")
    series = {str(getattr(d, "SeriesInstanceUID", "")) for _, _, d in items}
    if len(series) != 1:
        raise ValueError("PATIENT_DICOM 混有多个 SeriesInstanceUID")
    sop = [str(d.SOPInstanceUID) for _, _, d in items]
    if len(set(sop)) != len(sop):
        raise ValueError("重复 SOPInstanceUID")
    for _, _, d in items:
        if (int(d.Rows), int(d.Columns)) != (int(first.Rows), int(first.Columns)):
            raise ValueError("CT 层间矩阵不一致")
        if not np.allclose(d.ImageOrientationPatient, orient, atol=1e-5, rtol=0):
            raise ValueError("CT 层间方向变化")
        if not np.allclose(d.PixelSpacing, first.PixelSpacing, atol=1e-5, rtol=0):
            raise ValueError("CT 层间像素间距变化")
    spacing = np.asarray(first.PixelSpacing, float)
    if np.any(spacing <= 0) or not np.isclose(*spacing, atol=1e-4):
        raise ValueError("当前平行束模拟要求正的正方形像素；请先做经验证的重采样")
    if first.Rows != first.Columns:
        raise ValueError("当前实现要求方形轴向矩阵")
    items.sort(key=lambda x: np.dot(np.asarray(x[2].ImagePositionPatient, float), normal))
    positions = np.asarray([d.ImagePositionPatient for _, _, d in items], float)
    z = positions @ normal
    if len(z) < 2 or np.any(np.diff(z) <= cfg.alignment_tolerance_mm):
        raise ValueError("CT 层数不足、重复位置或几何间隔异常")
    dz = float(np.median(np.diff(z)))
    if not np.allclose(np.diff(z), dz, atol=cfg.alignment_tolerance_mm, rtol=0):
        raise ValueError("不等距/缺失 CT 切片，先检查原始数据")
    delta = positions - positions[0] - (z - z[0])[:, None] * normal
    if np.max(np.abs(delta)) > cfg.alignment_tolerance_mm:
        raise ValueError("检测到 gantry tilt 或层内位移；需先执行受控几何校正")
    # 数组使用 [z,row,col]；NIfTI 输出转为 [col,row,z]。
    affine = np.eye(4)
    affine[:3, 0] = row * spacing[1]
    affine[:3, 1] = col * spacing[0]
    affine[:3, 2] = normal * dz
    affine[:3, 3] = positions[0]
    return items, {"spacing_zyx": [dz, float(spacing[0]), float(spacing[1])],
                   "positions_lps": positions.tolist(), "z_mm": z.tolist(),
                   "orientation": orient.tolist(), "affine_lps_xyz": affine.tolist(),
                   "shape_zyx": [len(items), int(first.Rows), int(first.Columns)]}


def align_mask(ct_items, mask_items, cfg):
    if len(mask_items) != len(ct_items):
        raise ValueError(f"mask切片数 {len(mask_items)} != CT切片数 {len(ct_items)}，不自动补零")
    ct = [x[2] for x in ct_items]
    has_position = all(hasattr(x[2], "ImagePositionPatient") for x in mask_items)
    mode = "physical_position"
    if has_position:
        pos = np.asarray([x[2].ImagePositionPatient for x in mask_items], float)
        indexes = []
        for d in ct:
            dist = np.linalg.norm(pos - np.asarray(d.ImagePositionPatient, float), axis=1)
            candidates = np.flatnonzero(dist <= cfg.alignment_tolerance_mm)
            if len(candidates) != 1:
                raise ValueError("mask 与 CT 物理位置无法唯一配对")
            indexes.append(int(candidates[0]))
    elif cfg.allow_instance_fallback:
        mode = "instance_number_fallback"
        if not all(hasattr(x[2], "InstanceNumber") for x in mask_items) or not all(hasattr(d, "InstanceNumber") for d in ct):
            raise ValueError("InstanceNumber 回退不可用")
        keys = [int(x[2].InstanceNumber) for x in mask_items]
        if len(set(keys)) != len(keys):
            raise ValueError("重复 mask InstanceNumber")
        indexes = [keys.index(int(d.InstanceNumber)) for d in ct]
        warnings.warn("mask 缺少物理位置，显式启用 InstanceNumber 回退；必须人工核对叠加图")
    else:
        raise ValueError("mask 缺少 ImagePositionPatient，默认拒绝按编号猜测；参见 README")
    if len(set(indexes)) != len(ct):
        raise ValueError("mask 对齐不是一一映射")
    output = []
    for d, j in zip(ct, indexes):
        ref, _, m = mask_items[j]
        if (m.Rows, m.Columns) != (d.Rows, d.Columns):
            raise ValueError("mask 与 CT 矩阵不同")
        for key in ("ImageOrientationPatient", "PixelSpacing"):
            if hasattr(m, key) and not np.allclose(getattr(m, key), getattr(d, key), atol=1e-4, rtol=0):
                raise ValueError(f"mask 与 CT {key} 不同")
        # 二值标签绝不应用 CT RescaleSlope/Intercept。
        a = read_dicom(ref).pixel_array
        unique = np.unique(a)
        if unique.min() < 0 or not set(unique.tolist()).issubset({0, 1, 255}):
            raise ValueError(f"预期二值 mask（0/1/255），实际值 {unique[:12]}: {ref}")
        output.append(a > 0)
    return np.stack(output), mode


def load_patient(folder, number, cfg, with_masks=True):
    ct_items, geom = geometry(dicom_entries(entries(Path(folder), "PATIENT_DICOM")), cfg)
    hu = []
    for ref, _, _ in ct_items:
        d = read_dicom(ref)
        if not hasattr(d, "RescaleSlope") or not hasattr(d, "RescaleIntercept"):
            raise ValueError(f"缺少 CT HU 标定字段: {ref}")
        if float(d.RescaleSlope) == 0:
            raise ValueError("CT RescaleSlope 不可为0")
        a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
        if not np.isfinite(a).all():
            raise ValueError("CT 存在非有限像素")
        hu.append(a)
    hu = np.stack(hu)
    metadata = {"id": f"3Dircadb1.{number}", "number": number,
                "folder": str(Path(folder).resolve()), "n_slices": len(hu),
                "sex": str(getattr(ct_items[0][2], "PatientSex", "")), **geom,
                "ct_refs": [x[0] for x in ct_items],
                "ct_sha256": hashlib.sha256(hu.tobytes()).hexdigest()}
    if cfg.strict_official_counts and len(hu) != SLICE_COUNTS[number - 1]:
        raise ValueError(f"{metadata['id']} 本地切片数 {len(hu)} 与官网 {SLICE_COUNTS[number-1]} 不同")
    if not with_masks:
        return hu, None, metadata
    groups = {}
    for ref, name in entries(Path(folder), "MASKS_DICOM"):
        parts = Path(name).parts
        if len(parts) < 2:
            continue
        groups.setdefault(parts[-2], []).append((ref, name))
    liver_names = [n for n in groups if re.fullmatch(cfg.liver_pattern, n)]
    tumor_names = [n for n in groups if re.fullmatch(cfg.tumor_pattern, n)]
    strict_inventory = (cfg.strict_official_counts if cfg.strict_official_mask_inventory is None
                        else cfg.strict_official_mask_inventory)
    expected_names = EXPECTED_TARGET_MASKS_BY_PATIENT[number]
    if strict_inventory and sorted(n.casefold() for n in tumor_names) != sorted(expected_names):
        raise ValueError(f"病例{number}目标tumor mask目录与官方完整清单不符；"
                         f"应有 {list(expected_names)}，实际 {sorted(tumor_names)}；不能把缺失标签视为阴性")
    if len(liver_names) != 1:
        raise ValueError(f"必须唯一识别 liver mask，发现 {liver_names}; 全部目录 {list(groups)}")
    if not tumor_names and number not in cfg.known_tumor_negative:
        raise ValueError(f"病例{number}预期有目标tumor mask但未找到标签，不能静默视作阴性；目录 {list(groups)}")
    liver, mode = align_mask(ct_items, dicom_entries(groups[liver_names[0]]), cfg)
    tumor_raw = np.zeros_like(liver)
    modes = [mode]
    tumor_voxels_by_name = {}
    for name in tumor_names:
        a, mode = align_mask(ct_items, dicom_entries(groups[name]), cfg)
        tumor_raw |= a
        tumor_voxels_by_name[name] = int(a.sum())
        modes.append(mode)
    if not liver.any():
        raise ValueError("liver 标签为空")
    # 肝内占比极低的"肿瘤"视为非肝肿瘤（如7号肾上腺病灶 99.97% 在肝外）
    liver_frac = int((tumor_raw & liver).sum()) / max(int(tumor_raw.sum()), 1)
    tumor = (tumor_raw & liver) if liver_frac >= cfg.min_liver_tumor_fraction \
        else np.zeros_like(liver)
    if not tumor_raw.any() and number not in cfg.known_tumor_negative:
        raise ValueError("预期目标tumor阳性病例的肿瘤标签为空")
    labels = np.stack([liver, tumor], axis=1).astype(np.uint8)
    metadata.update({"has_tumor": bool(tumor.any()), "tumor_voxels": int(tumor.sum()),
                     "liver_voxels": int(liver.sum()), "tumor_slices": np.flatnonzero(tumor.any(axis=(1, 2))).tolist(),
                     "tumor_outside_liver_voxels": int((tumor_raw & ~liver).sum()),
                     "tumor_inside_liver_voxels": int((tumor_raw & liver).sum()),
                     "mask_names": sorted(groups), "tumor_mask_names": sorted(tumor_names),
                     "tumor_mask_voxels_by_name": dict(sorted(tumor_voxels_by_name.items())),
                     "tumor_pattern": cfg.tumor_pattern,
                     "known_tumor_negative": list(cfg.known_tumor_negative),
                     "strict_official_mask_inventory": bool(strict_inventory),
                     "expected_tumor_mask_names": list(expected_names) if strict_inventory else None,
                     "alignment_modes": sorted(set(modes)),
                     "mask_sha256": hashlib.sha256(labels.tobytes()).hexdigest()})
    return hu, labels, metadata


def audit(cfg):
    rows = []
    for number, folder in discover(cfg.data_root):
        print(f"Audit {folder.name}", flush=True)
        _, _, meta = load_patient(folder, number, cfg)
        print(f"  CT层数={meta['n_slices']}；mask目录={meta['mask_names']}；"
              f"目标tumor目录/体素={meta['tumor_mask_voxels_by_name']}；"
              f"tumor与liver外部交集={meta['tumor_outside_liver_voxels']}", flush=True)
        rows.append(meta)
    manifest = {"schema": 1, "patients": rows, "source": "local_dicom", "fingerprint": digest(rows)}
    dest = Path(cfg.cache_dir)
    existing_path = dest / "audit.json"
    if existing_path.is_file() and read_json(existing_path).get("fingerprint") != manifest["fingerprint"]:
        raise ValueError("已有 audit.json 与新标签/数据指纹不同；请使用新的 cache_dir，禁止覆盖旧实验审计")
    write_json(dest / "audit.json", manifest)
    fields = ["id", "number", "n_slices", "sex", "has_tumor", "tumor_voxels", "liver_voxels",
              "tumor_inside_liver_voxels", "tumor_outside_liver_voxels"]
    write_csv(dest / "audit.csv", [{**{k: r[k] for k in fields},
                                     "mask_names": "|".join(r["mask_names"]),
                                     "tumor_mask_names": "|".join(r["tumor_mask_names"]),
                                     "tumor_mask_voxels_by_name": json.dumps(r["tumor_mask_voxels_by_name"],
                                                                               ensure_ascii=False)} for r in rows])
    return manifest
