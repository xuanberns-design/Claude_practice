"""完整患者体积推理与逐视角评估；所有病例等权汇总，禁止切片级伪重复。"""
from pathlib import Path
import hashlib
from shutil import copyfile
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
from .dataset import PatientCache, view_channel
from .projection import denormalize
from .metrics import reconstruction_metrics, segmentation_metrics, finite_summary, paired_bootstrap, regional_metrics
from .common import read_json, write_json, write_csv, device_for, digest
from .config import Config, config_dict, split_file
from .predict import predict_batch
from .model import JointUNet
from .prepare import verify_cache
from .split import validate_split


@torch.inference_mode()
def predict_volume(model, patient, view, cfg, device):
    restored, probs = [], []
    n = patient.meta["n_slices"]
    for start in range(0, n, cfg.batch_size):
        slices = range(start, min(start+cfg.batch_size, n))
        x = torch.from_numpy(np.stack([patient.stack(f"fbp_{view}", z) for z in slices])).to(device)
        v = torch.full((len(x),), view_channel(view), device=device)
        r, probability = predict_batch(model, x, v, cfg)
        restored.append(r.float().cpu().numpy())
        probs.append(probability.float().cpu().numpy())
    return np.concatenate(restored), np.concatenate(probs)


def native_resize(a, shape):
    if tuple(a.shape[-2:]) == tuple(shape):
        return np.asarray(a, dtype=np.float32)
    output = []
    for start in range(0, len(a), 8):
        chunk = np.array(a[start:start+8], copy=True, dtype=np.float32)
        scalar = chunk.ndim == 3
        tensor = torch.from_numpy(chunk[:, None] if scalar else chunk)
        resized = F.interpolate(tensor, size=shape, mode="bilinear", align_corners=False).numpy()
        output.append(resized[:, 0] if scalar else resized)
    return np.concatenate(output)


def save_nifti(path, volume, meta):
    affine = np.diag([-1., -1., 1., 1.]) @ np.asarray(meta["affine_lps_xyz"])
    img = nib.Nifti1Image(np.transpose(volume, (2, 1, 0)), affine)
    img.header.set_xyzt_units("mm")
    nib.save(img, str(path))


def export_slice_figures(truth_hu, fbp, restored, truth_masks, probs, cfg, dest, *, patient, view):
    """导出该患者该视角的全部切片及六张独立面板，并保留代表层入口。"""
    from .visualize import comparison_figure, PANEL_FILES

    n = len(truth_hu)
    if not (n == len(fbp) == len(restored) == len(truth_masks) == len(probs)):
        raise ValueError("逐层可视化需要 GT、FBP、恢复图和标签的切片数一致")
    dest = Path(dest)
    slices_dir = dest / "slices"
    slices_dir.mkdir(parents=True, exist_ok=True)
    representative = (int(np.argmax(truth_masks[:, 1].sum((1, 2))))
                      if truth_masks[:, 1].any() else n // 2)
    for z in range(n):
        slice_dir = slices_dir / f"slice_{z:04d}"
        composite = slice_dir / "comparison.png"
        comparison_figure(truth_hu[z], fbp[z], restored[z], truth_masks[z], probs[z], cfg,
                          composite, patient=patient, view=view, z=z, panel_dir=slice_dir)
        if z == representative:
            copyfile(composite, dest / "comparison.png")
    write_json(dest / "visualization_manifest.json", {
        "patient": patient, "view": view, "n_slices": n,
        "representative_slice": representative,
        "slice_directory_pattern": "slices/slice_{z:04d}",
        "files_per_slice": ["comparison.png", *PANEL_FILES],
    })


def summarize(reconstruction, segmentation, cfg):
    summary = {"aggregation": "patient_macro", "reconstruction": {}, "segmentation": {}, "paired_vs_fbp": {}}
    for view in cfg.views:
        for reference in ("original_CT", "full_FBP"):
            for method in ("FBP", "Joint"):
                rows = [r for r in reconstruction if r["view"] == view and r["reference"] == reference and r["method"] == method]
                key = f"{view}/{reference}/{method}"
                summary["reconstruction"][key] = {m: finite_summary([r[m] for r in rows])
                                                     for m in ("SSIM", "PSNR_dB", "MAE_HU", "RMSE_HU", "liver_MAE_HU", "liver_RMSE_HU")}
            base = {r["patient"]: r for r in reconstruction if r["view"] == view and r["reference"] == reference and r["method"] == "FBP"}
            joint = {r["patient"]: r for r in reconstruction if r["view"] == view and r["reference"] == reference and r["method"] == "Joint"}
            for metric, direction in [("SSIM", 1), ("PSNR_dB", 1), ("MAE_HU", -1), ("RMSE_HU", -1)]:
                summary["paired_vs_fbp"][f"{view}/{reference}/{metric}"] = paired_bootstrap(
                    [direction*(joint[p][metric]-base[p][metric]) for p in sorted(joint)], cfg)
        rows = [r for r in segmentation if r["view"] == view]
        summary["segmentation"][str(view)] = {
            "liver_Dice": finite_summary([r["liver_Dice"] for r in rows]),
            "tumor_Dice_all": finite_summary([r["tumor_Dice"] for r in rows]),
            "tumor_Dice_positive_only": finite_summary([r["tumor_Dice"] for r in rows if r["tumor_positive"]]),
            "tumor_FP_ml_negative_only": finite_summary([r["tumor_false_positive_ml"] for r in rows if not r["tumor_positive"]])}
    return summary


def evaluate_patients(model, ids, cfg, device, output=None, export=False):
    model.eval()
    rec_rows, seg_rows, region_rows = [], [], []
    quality_path = Path(cfg.cache_dir)/"quality_audit.json"
    quality = read_json(quality_path) if quality_path.exists() else None
    for pid in ids:
        print(f"Evaluate {pid}", flush=True)
        p = PatientCache(cfg, pid)
        truth_hu = p.get("native_hu")
        truth_masks = p.get("native_masks")
        shape = truth_hu.shape[-2:]
        full = denormalize(native_resize(p.get("full_fbp"), shape), cfg)
        regions = {"body_reference":truth_hu>cfg.metric_body_threshold_hu,
                   "liver_expert":truth_masks[:,0]>0,"tumor_expert":truth_masks[:,1]>0}
        if (p.root/"native_valid.npy").exists():
            if quality is None:
                raise ValueError("发现未审计的native_valid.npy，请先运行cache-audit --with-dicom")
            if quality.get("schema") != 1 or quality.get("audit_fingerprint") != p.meta["audit_fingerprint"]:
                raise ValueError("padding审计版本/数据集指纹与当前缓存不一致")
            q = next((row for row in quality["patients"] if row["id"]==pid),None)
            valid = p.get("native_valid").astype(bool)
            if (valid.shape != truth_hu.shape or q is None or q.get("paddingmask_sha256") != hashlib.sha256(valid.tobytes()).hexdigest()
                    or q.get("native_ct_sha256") != p.meta["ct_sha256"]):
                raise ValueError("padding sidecar与已审计CT/valid mask不一致")
            regions["dicom_valid"] = valid
        for view in cfg.views:
            restored, probs = predict_volume(model, p, view, cfg, device)
            restored = denormalize(native_resize(restored, shape), cfg)
            probs = native_resize(probs, shape)
            fbp = denormalize(native_resize(p.get(f"fbp_{view}"), shape), cfg)
            for reference, target in (("original_CT", truth_hu), ("full_FBP", full)):
                for method, image in (("FBP", fbp), ("Joint", restored)):
                    metrics = reconstruction_metrics(image, target, cfg, truth_masks[:, 0])
                    rec_rows.append({"patient": pid, "view": view, "reference": reference, "method": method, **metrics})
                    if export:
                        for region,values in regional_metrics(image,target,regions,cfg).items():
                            region_rows.append({"patient":pid,"view":view,"reference":reference,"method":method,"region":region,**values})
            seg_rows.append({"patient": pid, "view": view, **segmentation_metrics(probs, truth_masks, p.meta["spacing_zyx"], cfg)})
            if output and export:
                dest = Path(output) / "volumes" / pid / str(view)
                dest.mkdir(parents=True, exist_ok=True)
                save_nifti(dest / "restored_hu.nii.gz", restored, p.meta)
                save_nifti(dest / "fbp_hu.nii.gz", fbp, p.meta)
                for c, name in enumerate(("liver", "tumor")):
                    save_nifti(dest / f"{name}.nii.gz", (probs[:, c] >= cfg.segmentation_threshold).astype(np.uint8), p.meta)
                export_slice_figures(truth_hu, fbp, restored, truth_masks, probs, cfg, dest,
                                     patient=pid, view=view)
    result = summarize(rec_rows, seg_rows, cfg)
    if output:
        output = Path(output)
        write_csv(output / "reconstruction_per_patient.csv", rec_rows)
        write_csv(output / "segmentation_per_patient.csv", seg_rows)
        write_json(output / "summary.json", result)
        if region_rows:
            write_csv(output/"reconstruction_regions_per_patient.csv",region_rows)
    return result


def reconstruction_score(summary, cfg):
    scores = []
    for v in cfg.views:
        original = summary["reconstruction"][f"{v}/original_CT/Joint"]
        full = summary["reconstruction"][f"{v}/full_FBP/Joint"]
        scores.append(.5*full["SSIM"]["mean"]+.5*np.exp(-original["liver_MAE_HU"]["mean"]/cfg.selection_roi_scale_hu))
    return float(np.mean(scores))


def reconstruction_eligibility(summary, cfg):
    """只根据验证集逐视角均值检查；绝不以此宣称测试集保证不退化。"""
    violations = []
    for view in cfg.views:
        for ref in cfg.selection_guard_references:
            base = summary["reconstruction"][f"{view}/{ref}/FBP"]
            joint = summary["reconstruction"][f"{view}/{ref}/Joint"]
            for metric,tol in [("SSIM",cfg.selection_ssim_tolerance),("PSNR_dB",cfg.selection_psnr_tolerance_db)]:
                b,j = base[metric]["mean"],joint[metric]["mean"]
                if b is None or j is None or j < b-tol:
                    violations.append(f"{view}/{ref}/{metric}: FBP={b}, Joint={j}")
        base = summary["reconstruction"][f"{view}/original_CT/FBP"]["liver_MAE_HU"]["mean"]
        joint = summary["reconstruction"][f"{view}/original_CT/Joint"]["liver_MAE_HU"]["mean"]
        limit = base*(1+cfg.selection_roi_mae_relative_tolerance)+cfg.selection_roi_mae_absolute_tolerance_hu
        if joint > limit:
            violations.append(f"{view}/liver_MAE_HU: FBP={base}, Joint={joint}, limit={limit}")
    return {"eligible": not violations or not cfg.selection_guard, "guard_enabled": cfg.selection_guard,
            "guard_references":list(cfg.selection_guard_references),
            "violations": violations}


def validation_score(summary, cfg):
    rec_score = reconstruction_score(summary,cfg)
    if cfg.segmentation_weight == 0:
        return rec_score
    seg = []
    for v in cfg.views:
        s = summary["segmentation"][str(v)]
        tumor = s["tumor_Dice_positive_only"]["mean"]
        if tumor is None:
            raise ValueError("验证集没有肿瘤阳性患者，不能据此选择联合模型")
        negative_fp = s["tumor_FP_ml_negative_only"]["mean"]
        score = (s["liver_Dice"]["mean"]+tumor)/2
        if negative_fp is not None:
            score = .9*score+.1*np.exp(-negative_fp/cfg.selection_fp_scale_ml)
        seg.append(score)
    weight = cfg.selection_reconstruction_weight
    return float(weight*rec_score+(1-weight)*np.mean(seg))


def load_checkpoint(path, cfg, device):
    # 只加载本实验生成/可信的 checkpoint；weights_only 阻止任意 pickle 类。
    ckpt = torch.load(path, map_location=device, weights_only=True)
    saved_cfg = Config(**ckpt["config"]).validate()
    ignored = {"data_root", "cache_dir", "run_dir", "device", "cpu_threads", "num_workers", "batch_size", "bootstrap_repeats",
               "split_path", "save_slice_every"}
    mismatches = [k for k, v in config_dict(saved_cfg).items()
                  if k not in ignored and v != config_dict(cfg)[k]
                  and not (isinstance(v, (list, tuple)) and list(v) == list(config_dict(cfg)[k]))]
    if mismatches:
        raise ValueError(f"checkpoint与当前配置不一致: {mismatches}；请使用训练时 resolved_config.yaml")
    model = JointUNet(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    return model, ckpt


def evaluate(cfg, checkpoint, split="test", export=True):
    splits = read_json(split_file(cfg))
    audit = read_json(Path(cfg.cache_dir) / "audit.json")
    validate_split(splits, [r["id"] for r in audit["patients"]])
    device = device_for(cfg)
    model, ckpt = load_checkpoint(checkpoint, cfg, device)
    if ckpt["split_fingerprint"] != digest({k: splits[k] for k in ("train", "val", "test")}):
        raise ValueError("评估划分不同于训练时冻结的划分")
    if ckpt["audit_fingerprint"] != audit["fingerprint"]:
        raise ValueError("评估数据审计指纹不同于训练时数据")
    verify_cache(cfg, splits[split])
    output = Path(cfg.run_dir) / f"evaluation_{split}"
    result = evaluate_patients(model, splits[split], cfg, device, output, export)
    result["segmentation_training_enabled"] = cfg.segmentation_weight > 0
    result["checkpoint_after_segmentation_warmup"] = ckpt["epoch"] >= cfg.warmup_epochs
    write_json(output / "summary.json", result)
    write_json(output / "provenance.json", {"checkpoint": str(Path(checkpoint).resolve()), "epoch": ckpt["epoch"],
                                           "split": split, "patients": splits[split], "config": config_dict(cfg),
                                           "audit_fingerprint": audit["fingerprint"], "split_fingerprint": ckpt["split_fingerprint"],
                                           "training_protocol":ckpt.get("training_protocol","v1"),
                                           "checkpoint_sha256":hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                                           "metric_protocol":"rawHU_unclipped_global_v1_plus_explicit_regions_v2"})
    return result
