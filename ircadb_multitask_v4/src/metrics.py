"""固定窗SSIM/PSNR、原始HU误差、3D患者Dice和配对bootstrap。

参数入口：config.py 的 window_min/max、segmentation_threshold、bootstrap_repeats。
"""
import math
import numpy as np
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


def segmentation_metrics(probability, truth, spacing, cfg):
    p, t = probability >= cfg.segmentation_threshold, truth.astype(bool)
    voxel_ml = float(np.prod(spacing) / 1000)
    return {"liver_Dice": dice_score(p[:, 0], t[:, 0]), "tumor_Dice": dice_score(p[:, 1], t[:, 1]),
            "tumor_positive": bool(t[:, 1].any()),
            "tumor_false_positive_ml": float(np.count_nonzero(p[:, 1] & ~t[:, 1])*voxel_ml),
            "tumor_ground_truth_ml": float(t[:, 1].sum()*voxel_ml)}


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
