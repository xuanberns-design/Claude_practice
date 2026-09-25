"""3D 分割后处理与验证集调参；只使用模型概率，不读取专家 mask。

依据：本包标签定义 tumor = 原始肿瘤 ∩ 专家肝脏（dicom_io.load_patient），因此
(1) 肝脏取最大 3D 连通域并逐层填洞，去除脾/肾/血管等离散假阳性；
(2) 肿瘤只保留在“预测肝脏 + margin”内，去除肝外假阳性；
(3) 去除小于 min_tumor_ml 的肿瘤碎片；
(4) 最终 liver |= tumor，与标签的包含关系一致。
肿瘤阈值与最小体积只在验证集上选择（tune），测试集直接套用，不做测试集调参。
"""
import numpy as np
from scipy import ndimage

CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=bool)


def largest_component(mask):
    labels, count = ndimage.label(mask, structure=CONNECTIVITY_26)
    if count <= 1:
        return mask.astype(bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(sizes.argmax())


def fill_holes_per_slice(mask):
    out = np.empty_like(mask, dtype=bool)
    for z in range(len(mask)):
        out[z] = ndimage.binary_fill_holes(mask[z])
    return out


def _bbox(mask, margin):
    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(0) - margin, 0)
    hi = np.minimum(idx.max(0) + margin + 1, mask.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def liver_neighbourhood(liver, spacing_zyx, margin_mm):
    """同层内距预测肝脏 <= margin_mm 的区域（EDT，在肝脏外包框内计算以节约时间）。"""
    region = np.zeros_like(liver, dtype=bool)
    if not liver.any():
        return region
    pad = [0] + [int(np.ceil(margin_mm / s)) + 1 for s in spacing_zyx[1:]]
    box = _bbox(liver, np.asarray(pad))
    # z 方向采样距离设为极大值 -> 只在同一层内找最近肝脏像素。
    distance = ndimage.distance_transform_edt(~liver[box], sampling=(1e6, spacing_zyx[1], spacing_zyx[2]))
    region[box] = distance <= margin_mm
    return region


def remove_small_components(mask, voxel_ml, min_ml, labels=None):
    if min_ml <= 0 or not mask.any():
        return mask
    if labels is None:
        labels, _ = ndimage.label(mask, structure=CONNECTIVITY_26)
    sizes = np.bincount(labels.ravel()) * voxel_ml
    keep = sizes >= min_ml
    keep[0] = False
    return keep[labels]


def process_liver(probability, spacing_zyx, cfg):
    liver = probability[:, 0] >= cfg.segmentation_threshold
    if not cfg.postprocess:
        return liver
    if cfg.postprocess_keep_largest_liver and liver.any():
        liver = largest_component(liver)
    if cfg.postprocess_fill_holes:
        liver = fill_holes_per_slice(liver)
    return liver


def tumor_region(liver, spacing_zyx, cfg):
    if not cfg.postprocess or cfg.tumor_liver_margin_mm is None:
        return None
    return liver_neighbourhood(liver, spacing_zyx, cfg.tumor_liver_margin_mm)


def postprocess_volume(probability, spacing_zyx, cfg, params=None):
    """probability [Z,2,H,W] -> (liver, tumor) 布尔体积。params 为验证集选出的肿瘤参数。"""
    params = params or {}
    threshold = params.get("tumor_threshold", cfg.tumor_threshold)
    threshold = cfg.segmentation_threshold if threshold is None else threshold
    min_ml = params.get("min_tumor_ml", cfg.min_tumor_ml)
    liver = process_liver(probability, spacing_zyx, cfg)
    tumor = probability[:, 1] >= threshold
    if not cfg.postprocess:
        return liver, tumor
    region = tumor_region(liver, spacing_zyx, cfg)
    if region is not None:
        tumor &= region
    tumor = remove_small_components(tumor, float(np.prod(spacing_zyx) / 1000), min_ml)
    return liver | tumor, tumor


def tumor_grid_counts(probability, truth_tumor, spacing_zyx, cfg):
    """对阈值×最小体积网格一次性计算肿瘤 Dice 所需计数，供验证集调参。"""
    liver = process_liver(probability, spacing_zyx, cfg)
    region = tumor_region(liver, spacing_zyx, cfg)
    voxel_ml = float(np.prod(spacing_zyx) / 1000)
    truth = truth_tumor.astype(bool)
    truth_sum = int(truth.sum())
    rows = []
    for threshold in cfg.tune_tumor_thresholds:
        base = probability[:, 1] >= threshold
        if region is not None:
            base &= region
        labels, _ = ndimage.label(base, structure=CONNECTIVITY_26) if base.any() else (np.zeros(base.shape, np.int32), 0)
        for min_ml in cfg.tune_min_tumor_ml:
            pred = remove_small_components(base, voxel_ml, min_ml, labels) if min_ml > 0 else base
            inter = int(np.count_nonzero(pred & truth))
            psum = int(pred.sum())
            rows.append({"tumor_threshold": float(threshold), "min_tumor_ml": float(min_ml),
                         "dice": 2 * inter / (psum + truth_sum) if psum + truth_sum else 1.0,
                         "false_positive_ml": (psum - inter) * voxel_ml, "positive": truth_sum > 0})
    return rows


def choose_parameters(per_case_rows, cfg):
    """per_case_rows: 每个(患者,视角)一份 tumor_grid_counts 结果；返回最佳参数与各候选得分。"""
    keys = sorted({(r["tumor_threshold"], r["min_tumor_ml"]) for rows in per_case_rows for r in rows})
    table = []
    for key in keys:
        chosen = [r for rows in per_case_rows for r in rows if (r["tumor_threshold"], r["min_tumor_ml"]) == key]
        positive = [r["dice"] for r in chosen if r["positive"]]
        negative = [r["false_positive_ml"] for r in chosen if not r["positive"]]
        if not positive:
            raise ValueError("验证集没有肿瘤阳性病例，不能调肿瘤后处理参数")
        score = float(np.mean(positive))
        if negative:
            score = 0.9 * score + 0.1 * float(np.exp(-np.mean(negative) / cfg.selection_fp_scale_ml))
        table.append({"tumor_threshold": key[0], "min_tumor_ml": key[1], "score": score,
                      "tumor_Dice_positive_mean": float(np.mean(positive)),
                      "tumor_FP_ml_negative_mean": float(np.mean(negative)) if negative else None})
    best = max(table, key=lambda r: (r["score"], -abs(r["tumor_threshold"] - 0.5), -r["min_tumor_ml"]))
    return {"tumor_threshold": best["tumor_threshold"], "min_tumor_ml": best["min_tumor_ml"]}, table
