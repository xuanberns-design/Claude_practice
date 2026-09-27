"""固定窗SSIM/PSNR、原始HU误差、3D患者Dice和配对bootstrap。

参数入口：config.py 的 window_min/max、segmentation_threshold、bootstrap_repeats。
"""
import math
import numpy as np
from scipy import ndimage
from skimage.metrics import structural_similarity


def reconstruction_metrics(pred_hu, target_hu, cfg, roi=None):
    if pred_hu.shape != target_hu.shape:
        raise ValueError("指标输入尺寸不一致")
    low, high = cfg.window_min, cfg.window_max
    absolute, squared, window_squared, count = 0., 0., 0., 0
    roi_absolute, roi_squared, roi_count = 0., 0., 0
    ssim = []
    # 逐层累计，避免512x512x260体积产生多份float64临时数组。
    for z, (a, b) in enumerate(zip(pred_hu, target_hu)):
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("指标输入包含NaN/Inf")
        diff = a-b
        absolute += np.abs(diff).sum()
        squared += np.square(diff).sum()
        count += diff.size
        x = (np.clip(a, low, high)-low)/(high-low)
        y = (np.clip(b, low, high)-low)/(high-low)
        window_squared += np.square(x-y).sum()
        ssim.append(structural_similarity(x, y, data_range=1.0, gaussian_weights=True,
                                          sigma=1.5, use_sample_covariance=False))
        if roi is not None:
            d = diff[np.asarray(roi[z], bool)]
            roi_absolute += np.abs(d).sum()
            roi_squared += np.square(d).sum()
            roi_count += d.size
    mse = float(window_squared/count)
    result = {"SSIM": float(np.mean(ssim)), "PSNR_dB": float(-10*np.log10(mse)) if mse > 0 else math.inf,
              "MAE_HU": float(absolute/count), "RMSE_HU": float(np.sqrt(squared/count))}
    if roi is not None:
        result.update({"liver_MAE_HU": float(roi_absolute/roi_count) if roi_count else None,
                       "liver_RMSE_HU": float(np.sqrt(roi_squared/roi_count)) if roi_count else None})
    return result


def dice_score(pred, truth):
    p, t = np.asarray(pred, bool), np.asarray(truth, bool)
    denom = int(p.sum()) + int(t.sum())
    return float(2*np.count_nonzero(p & t)/denom) if denom else 1.0


def surface_distance_metrics(pred, truth, spacing):
    """HD95 与平均对称表面距离(mm)；在两者外包框内计算 EDT。任一为空返回 None。"""
    pred, truth = np.asarray(pred, bool), np.asarray(truth, bool)
    if not pred.any() or not truth.any():
        return None, None
    idx = np.argwhere(pred | truth)
    lo, hi = np.maximum(idx.min(0) - 2, 0), np.minimum(idx.max(0) + 3, pred.shape)
    box = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    p, t = pred[box], truth[box]
    pb = p & ~ndimage.binary_erosion(p)
    tb = t & ~ndimage.binary_erosion(t)
    to_truth = ndimage.distance_transform_edt(~tb, sampling=spacing)[pb]
    to_pred = ndimage.distance_transform_edt(~pb, sampling=spacing)[tb]
    distances = np.concatenate([to_truth, to_pred])
    return float(np.percentile(distances, 95)), float(distances.mean())


def lesion_detection(pred, truth):
    """26 邻域病灶一对一匹配；一个预测连通域不能同时检出多个真值病灶。"""
    from .postprocess import one_to_one_lesion_counts
    structure = np.ones((3, 3, 3), dtype=bool)
    gt_labels, gt_count = ndimage.label(truth, structure=structure)
    matched, pr_count = one_to_one_lesion_counts(pred, gt_labels, gt_count)
    recall = matched / gt_count if gt_count else None
    precision = matched / pr_count if pr_count else None
    return recall, precision, int(gt_count), int(pr_count)


def segmentation_metrics(probability, truth, spacing, cfg, params=None, surface=True, view=None):
    """主列为后处理结果（postprocess=false 时与 V3 相同的纯阈值结果），*_raw 为纯阈值对照。"""
    from .postprocess import postprocess_volume
    raw = probability >= cfg.segmentation_threshold
    liver, tumor = postprocess_volume(probability, spacing, cfg, params, view)
    t = truth.astype(bool)
    voxel_ml = float(np.prod(spacing) / 1000)
    result = {"liver_Dice": dice_score(liver, t[:, 0]), "tumor_Dice": dice_score(tumor, t[:, 1]),
              "tumor_positive": bool(t[:, 1].any()),
              "tumor_false_positive_ml": float(np.count_nonzero(tumor & ~t[:, 1])*voxel_ml),
              "tumor_ground_truth_ml": float(t[:, 1].sum()*voxel_ml),
              "tumor_predicted_ml": float(tumor.sum()*voxel_ml),
              "liver_Dice_raw": dice_score(raw[:, 0], t[:, 0]), "tumor_Dice_raw": dice_score(raw[:, 1], t[:, 1]),
              "tumor_false_positive_ml_raw": float(np.count_nonzero(raw[:, 1] & ~t[:, 1])*voxel_ml)}
    recall, precision, gt_count, pr_count = lesion_detection(tumor, t[:, 1])
    result.update({"tumor_lesion_recall": recall, "tumor_lesion_precision": precision,
                   "tumor_lesions_truth": gt_count, "tumor_lesions_predicted": pr_count})
    if view == 32 and getattr(cfg, "selection_32_lesion_priority", False):
        # 训练期选模与部署阈值对齐：部署时 32 views 阈值在 tune_32_thresholds 中选择，
        # 若只在默认阈值(0.5)下量召回，选出的权重未必在低阈值下最好。这里取整个候选区间的平均。
        recalls, fps = [], []
        for threshold in cfg.tune_32_thresholds:
            override = {"per_view": {"32": {"tumor_threshold": float(threshold), "min_tumor_ml": 0.0}}}
            _, candidate = postprocess_volume(probability, spacing, cfg, {**(params or {}), **override}, view)
            recalls.append(lesion_detection(candidate, t[:, 1])[0])
            fps.append(float(np.count_nonzero(candidate & ~t[:, 1])*voxel_ml))
        result["tumor_lesion_recall_tuned_range"] = (float(np.mean(recalls)) if recalls[0] is not None else None)
        result["tumor_false_positive_ml_tuned_range"] = float(np.mean(fps))
    if surface:
        for name, pred, gt in (("liver", liver, t[:, 0]), ("tumor", tumor, t[:, 1])):
            hd95, assd = surface_distance_metrics(pred, gt, spacing)
            result.update({f"{name}_HD95_mm": hd95, f"{name}_ASSD_mm": assd})
    return result


def finite_summary(values):
    a = np.asarray([v for v in values if v is not None], float)
    finite = a[np.isfinite(a)]
    return {"mean": float(finite.mean()) if len(finite) else None,
            "std": float(finite.std(ddof=1)) if len(finite) > 1 else None,
            "n": len(a), "nonfinite": int(len(a)-len(finite))}


def paired_bootstrap(deltas, cfg):
    a = np.asarray(deltas, float)
    a = a[np.isfinite(a)]
    if len(a) < 2:
        return {"n": len(a), "mean_improvement": float(a.mean()) if len(a) else None,
                "ci95": None, "note": "至少需要2名患者；不可据此声称显著提高"}
    rng = np.random.default_rng(cfg.seed)
    samples = rng.choice(a, (cfg.bootstrap_repeats, len(a)), replace=True).mean(axis=1)
    return {"n": len(a), "mean_improvement": float(a.mean()),
            "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
            "note": "患者级配对百分位bootstrap；极小样本CI不稳定，不构成临床有效性证据"}


def regional_metrics(pred_hu, target_hu, regions, cfg):
    """同一图像对一次计算SSIM map，按各ROI内像素中心汇总，不改变原全FOV指标。"""
    sums = {name:{"n":0,"ae":0.,"se":0.,"wse":0.,"ssim":0.,"ssim_n":0} for name in regions}
    for z,(pred,truth) in enumerate(zip(pred_hu,target_hu)):
        a,b = np.asarray(pred,np.float64),np.asarray(truth,np.float64)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("区域指标输入非有限值")
        x = (np.clip(a,cfg.window_min,cfg.window_max)-cfg.window_min)/(cfg.window_max-cfg.window_min)
        y = (np.clip(b,cfg.window_min,cfg.window_max)-cfg.window_min)/(cfg.window_max-cfg.window_min)
        _, smap = structural_similarity(x,y,data_range=1,gaussian_weights=True,sigma=1.5,use_sample_covariance=False,full=True)
        centers = np.zeros_like(x,dtype=bool)
        centers[5:-5,5:-5] = True
        for name,roi in regions.items():
            mask = np.asarray(roi[z],bool)
            d = (a-b)[mask]
            s = sums[name]
            s["n"] += d.size
            s["ae"] += np.abs(d).sum()
            s["se"] += np.square(d).sum()
            s["wse"] += np.square(x-y)[mask].sum()
            selected = mask & centers
            s["ssim"] += smap[selected].sum()
            s["ssim_n"] += int(selected.sum())
    results = {}
    for name,s in sums.items():
        n = s["n"]
        mse = s["wse"]/n if n else None
        results[name] = {"voxel_count":n,"SSIM":float(s["ssim"]/s["ssim_n"]) if s["ssim_n"] else None,
                         "PSNR_dB":float(-10*np.log10(mse)) if mse and mse>0 else (math.inf if n else None),
                         "MAE_HU":float(s["ae"]/n) if n else None,"RMSE_HU":float(np.sqrt(s["se"]/n)) if n else None}
    return results
