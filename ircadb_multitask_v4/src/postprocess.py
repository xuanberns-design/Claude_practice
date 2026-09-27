"""3D 分割后处理与验证集调参；只使用模型概率，不读取专家 mask。

依据：本包标签定义 tumor = 原始肿瘤 ∩ 专家肝脏（dicom_io.load_patient），因此
(1) 肝脏取最大 3D 连通域并逐层填洞，去除脾/肾/血管等离散假阳性；
(2) 肿瘤只保留在“预测肝脏 + margin”内，去除肝外假阳性；
(3) 去除小于 min_tumor_ml 的肿瘤碎片；
(4) 最终 liver |= tumor，与标签的包含关系一致。
肿瘤阈值、最小体积及 32 views 肝脏门控距离只在验证集上选择，
测试集直接套用冻结参数，不做测试集调参。
"""
import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=bool)
_DEFAULT_MARGIN = object()


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


def tumor_region(liver, spacing_zyx, cfg, margin_mm=_DEFAULT_MARGIN):
    if margin_mm is _DEFAULT_MARGIN:
        margin_mm = cfg.tumor_liver_margin_mm
    if not cfg.postprocess or margin_mm is None:
        return None
    return liver_neighbourhood(liver, spacing_zyx, margin_mm)


def effective_parameters(cfg, params=None, view=None):
    """合并默认值、旧顶层参数及指定视角的验证集参数。"""
    params = params or {}
    selected = {k: params[k] for k in ("tumor_threshold", "min_tumor_ml", "tumor_liver_margin_mm") if k in params}
    if view is not None:
        selected.update(params.get("per_view", {}).get(str(view), {}))
    threshold = selected.get("tumor_threshold", cfg.tumor_threshold)
    threshold = cfg.segmentation_threshold if threshold is None else threshold
    return {"tumor_threshold": float(threshold),
            "min_tumor_ml": float(selected.get("min_tumor_ml", cfg.min_tumor_ml)),
            "tumor_liver_margin_mm": selected.get("tumor_liver_margin_mm", cfg.tumor_liver_margin_mm)}


def postprocess_volume(probability, spacing_zyx, cfg, params=None, view=None, *, return_stages=False):
    """probability [Z,2,H,W] -> (liver, tumor)；可选返回门控前后掩膜。"""
    selected = effective_parameters(cfg, params, view)
    threshold = selected["tumor_threshold"]
    min_ml = selected["min_tumor_ml"]
    liver = process_liver(probability, spacing_zyx, cfg)
    tumor = probability[:, 1] >= threshold
    before_gate = tumor.copy() if return_stages else None
    if not cfg.postprocess:
        if return_stages:
            return liver, tumor, {"before_gate": before_gate, "after_gate": tumor.copy(),
                                  "effective_parameters": selected}
        return liver, tumor
    region = tumor_region(liver, spacing_zyx, cfg, selected["tumor_liver_margin_mm"])
    if region is not None:
        tumor &= region
    after_gate = tumor.copy() if return_stages else None
    tumor = remove_small_components(tumor, float(np.prod(spacing_zyx) / 1000), min_ml)
    if return_stages:
        return liver | tumor, tumor, {"before_gate": before_gate, "after_gate": after_gate,
                                       "effective_parameters": selected}
    return liver | tumor, tumor


def one_to_one_lesion_counts(pred, truth_labels, truth_count):
    """以任意体素重叠为边，最大二部匹配；一个预测域至多检出一个真值病灶。"""
    if not pred.any():
        return 0, 0
    predicted_labels, predicted_count = ndimage.label(pred, structure=CONNECTIVITY_26)
    if not truth_count or not predicted_count:
        return 0, int(predicted_count)
    overlap = (truth_labels > 0) & (predicted_labels > 0)
    gt = truth_labels[overlap].astype(np.int64) - 1
    pr = predicted_labels[overlap].astype(np.int64) - 1
    if not len(gt):
        return 0, int(predicted_count)
    # 编码唯一连通域对，避免体素数规模的稀疏矩阵重复边。
    pair = np.unique(gt * int(predicted_count) + pr)
    edges = csr_matrix((np.ones(len(pair), dtype=np.int8),
                        (pair // predicted_count, pair % predicted_count)),
                       shape=(int(truth_count), int(predicted_count)))
    matching = maximum_bipartite_matching(edges, perm_type="column")
    return int(np.count_nonzero(matching >= 0)), int(predicted_count)


def tumor_grid_counts(probability, truth_tumor, spacing_zyx, cfg, *, thresholds=None,
                      min_ml_options=None, margins=None):
    """验证集网格指标；可扩展搜索肝脏门控距离，不读取测试集标签。"""
    liver = process_liver(probability, spacing_zyx, cfg)
    voxel_ml = float(np.prod(spacing_zyx) / 1000)
    truth = truth_tumor.astype(bool)
    truth_sum = int(truth.sum())
    truth_labels, truth_count = (ndimage.label(truth, structure=CONNECTIVITY_26)
                                 if truth_sum else (None, 0))
    thresholds = cfg.tune_tumor_thresholds if thresholds is None else thresholds
    min_ml_options = cfg.tune_min_tumor_ml if min_ml_options is None else min_ml_options
    margins = (cfg.tumor_liver_margin_mm,) if margins is None else margins
    regions = [(margin, tumor_region(liver, spacing_zyx, cfg, margin)) for margin in margins]
    rows = []
    for threshold in thresholds:
        thresholded = probability[:, 1] >= threshold
        for margin, region in regions:
            base = thresholded & region if region is not None else thresholded
            labels = ndimage.label(base, structure=CONNECTIVITY_26)[0] if base.any() else None
            for min_ml in min_ml_options:
                pred = remove_small_components(base, voxel_ml, min_ml, labels) if min_ml > 0 else base
                inter = int(np.count_nonzero(pred & truth))
                psum = int(pred.sum())
                matched, predicted_count = one_to_one_lesion_counts(pred, truth_labels, truth_count)
                rows.append({"tumor_threshold": float(threshold), "min_tumor_ml": float(min_ml),
                             "tumor_liver_margin_mm": margin,
                             "dice": 2 * inter / (psum + truth_sum) if psum + truth_sum else 1.0,
                             "false_positive_ml": (psum - inter) * voxel_ml, "positive": truth_sum > 0,
                             "lesion_recall": matched / truth_count if truth_count else None,
                             "lesions_truth": int(truth_count), "lesions_predicted": predicted_count,
                             "false_positive_lesions": predicted_count - matched})
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


def choose_32_parameters(per_case_rows, baseline_rows, baseline_params, cfg):
    """32 views：召回优先；阳性 Dice 不低于原方案且阴性假阳性 <= 配置上限。"""
    if not per_case_rows or len(per_case_rows) != len(baseline_rows):
        raise ValueError("32 views 候选与基线必须来自同一批验证集患者")
    baseline_positive = [rows[0]["dice"] for rows in baseline_rows if rows[0]["positive"]]
    if not baseline_positive:
        raise ValueError("验证集没有肝内肿瘤阳性病例，不能校准 32 views")
    baseline_dice = float(np.mean(baseline_positive))
    baseline_recall = float(np.mean([rows[0]["lesion_recall"] for rows in baseline_rows if rows[0]["positive"]]))
    baseline_negative_fp = [rows[0]["false_positive_ml"] for rows in baseline_rows if not rows[0]["positive"]]
    baseline_all_fp = float(np.mean([rows[0]["false_positive_ml"] for rows in baseline_rows]))
    all_patient_guard = not baseline_negative_fp and getattr(cfg, "fp_guard_fallback", "keep_baseline") == "all_patients"
    all_fp_limit = baseline_all_fp + getattr(cfg, "max_fp_increase_ml", 0.0)
    keys = [(r["tumor_threshold"], r["min_tumor_ml"], r["tumor_liver_margin_mm"])
            for r in per_case_rows[0]]
    if len(keys) != len(set(keys)):
        raise ValueError("32 views 后处理网格包含重复候选")
    by_case = [{(r["tumor_threshold"], r["min_tumor_ml"], r["tumor_liver_margin_mm"]): r for r in rows}
               for rows in per_case_rows]
    if any(set(case) != set(keys) for case in by_case):
        raise ValueError("32 views 每名验证患者的候选网格不一致")
    table = []
    for threshold, min_ml, margin in keys:
        chosen = [case[(threshold, min_ml, margin)] for case in by_case]
        positive = [r for r in chosen if r["positive"]]
        negative = [r for r in chosen if not r["positive"]]
        dice = float(np.mean([r["dice"] for r in positive]))
        recall = float(np.mean([r["lesion_recall"] for r in positive]))
        fp = float(np.mean([r["false_positive_ml"] for r in negative])) if negative else None
        fp_lesions = float(np.mean([r["false_positive_lesions"] for r in negative])) if negative else None
        # Without a negative validation patient, the requested FP guard is
        # unmeasurable. Keep the frozen baseline instead of approving a lower
        # threshold on recall alone.
        all_fp = float(np.mean([r["false_positive_ml"] for r in chosen]))
        if negative:
            fp_ok = fp <= cfg.max_negative_fp_ml + 1e-12
        else:
            # 阳性患者真值外的预测体积同样是假阳性，可在无阴性病例时作为替代约束。
            fp_ok = all_patient_guard and all_fp <= all_fp_limit + 1e-12
        eligible = dice + 1e-12 >= baseline_dice and fp_ok
        table.append({"tumor_threshold": threshold, "min_tumor_ml": min_ml,
                      "tumor_liver_margin_mm": margin,
                      "tumor_lesion_recall_positive_mean": recall,
                      "tumor_Dice_positive_mean": dice,
                      "tumor_FP_ml_negative_mean": fp,
                      "tumor_FP_lesions_negative_mean": fp_lesions,
                      "tumor_FP_ml_all_patients_mean": all_fp,
                      "eligible": bool(eligible)})
    eligible = [r for r in table if r["eligible"]]
    # 允许等召回时按 Dice、阴性假阳性和更严格的门控排序。
    best = max(eligible, key=lambda r: (
        r["tumor_lesion_recall_positive_mean"], r["tumor_Dice_positive_mean"],
        -(r["tumor_FP_ml_negative_mean"] or 0),
        -(r["tumor_liver_margin_mm"] if r["tumor_liver_margin_mm"] is not None else float("inf")))) if eligible else None
    baseline = {"tumor_threshold": baseline_params["tumor_threshold"],
                "min_tumor_ml": baseline_params["min_tumor_ml"],
                "tumor_liver_margin_mm": baseline_params["tumor_liver_margin_mm"]}
    selection = {"baseline_tumor_lesion_recall_positive_mean": baseline_recall,
                 "baseline_tumor_Dice_positive_mean": baseline_dice,
                 "baseline_tumor_FP_ml_negative_mean": float(np.mean(baseline_negative_fp)) if baseline_negative_fp else None,
                 "max_negative_fp_ml": cfg.max_negative_fp_ml,
                 "negative_validation_patients": len(baseline_negative_fp),
                 "baseline_tumor_FP_ml_all_patients_mean": baseline_all_fp,
                 "fp_guard": ("negative_patients" if baseline_negative_fp else
                              "all_patients" if all_patient_guard else "unavailable")}
    if all_patient_guard:
        selection["all_patients_fp_limit_ml"] = all_fp_limit
    if not baseline_negative_fp and not all_patient_guard:
        selection["status"] = "kept_baseline_no_negative_validation"
        return baseline, table, selection
    if best is None or (best["tumor_lesion_recall_positive_mean"], best["tumor_Dice_positive_mean"]) <= (baseline_recall, baseline_dice):
        selection["status"] = "kept_baseline"
        return baseline, table, selection
    selection["status"] = "selected_candidate"
    return {k: best[k] for k in baseline}, table, selection
