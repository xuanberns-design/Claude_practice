"""混合视角训练、预热/联合优化、验证选模与epoch边界恢复。

参数入口：config.py 所有训练参数；不读取测试患者缓存。
V4：可选 EMA 权重（验证/保存均用 EMA）、线性学习率预热+余弦退火、双域正弦图输入与损失、
训练结束后在验证集自动选择分割后处理参数。
"""
from pathlib import Path
import copy
import math
import platform
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
from .common import read_json, write_json, seed_all, device_for, digest
from .config import Config, config_dict, prepare_signature, split_file
from .dataset import TrainDataset
from .model import JointUNet
from .losses import joint_loss
from .prepare import verify_cache
from .split import validate_split
from .evaluate import (evaluate_patients, validation_score, load_checkpoint, reconstruction_score,
                       reconstruction_eligibility, segmentation_eligibility, tune_postprocess,
                       tune_32_postprocess)

TRAINING_PROTOCOL = "v4"

# Starting a new segmentation experiment may change sampling, loss and selection
# settings, but must not silently reinterpret the source checkpoint's image
# normalization or network weights. Cache provenance is checked separately.
FINETUNE_MODEL_KEYS = (
    "window_min", "window_max", "base_channels", "context_slices",
    "reconstruction_mode", "reconstruction_backbone", "reconstruction_upsample",
    "reconstruction_dilated_bottleneck", "sino_dense_views", "sino_base_channels",
    "sino_angle_pad", "sino_data_consistency", "seg_backbone", "seg_base_channels",
    "seg_wide_window", "seg_deep_supervision_weight",
)


def load_finetune_source(path, cfg, model, fingerprint, audit_fingerprint):
    """Load weights into a fresh run only after data and architecture checks."""
    source = Path(path).resolve()
    checkpoint = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    if checkpoint.get("training_protocol") != TRAINING_PROTOCOL:
        raise ValueError("分割微调需要 V4 checkpoint；旧版协议不能作为此入口的初始化权重")
    saved_cfg = Config(**checkpoint["config"]).validate()
    if prepare_signature(saved_cfg) != prepare_signature(cfg):
        raise ValueError("微调checkpoint与当前缓存的准备参数/标签定义不一致")
    old, new = config_dict(saved_cfg), config_dict(cfg)
    mismatches = [key for key in FINETUNE_MODEL_KEYS if old[key] != new[key]
                  and not (isinstance(old[key], (list, tuple)) and list(old[key]) == list(new[key]))]
    if mismatches:
        raise ValueError(f"微调checkpoint与当前模型结构或输入窗不一致: {mismatches}")
    if checkpoint.get("split_fingerprint") != fingerprint or checkpoint.get("audit_fingerprint") != audit_fingerprint:
        raise ValueError("微调checkpoint的患者划分或数据审计指纹与当前实验不一致")
    if saved_cfg.segmentation_weight <= 0:
        raise ValueError("微调来源checkpoint未训练分割分支")
    model.load_state_dict(checkpoint["model"], strict=True)
    return {"checkpoint": str(source), "source_epoch": int(checkpoint["epoch"]) + 1,
            "split_fingerprint": fingerprint, "audit_fingerprint": audit_fingerprint}


def reconstructed_input_fraction(cfg, epoch):
    """Fraction of reconstructed CT fed to segmentation during clean pretraining."""
    if not cfg.seg_pretrain_clean or epoch >= cfg.warmup_epochs:
        return 1.0
    mix_epochs = getattr(cfg, "seg_pretrain_mix_epochs", 0)
    if mix_epochs <= 0:
        return 0.0
    return min(1.0, max(0.0, (epoch - (cfg.warmup_epochs - mix_epochs) + 1) / mix_epochs))


def gradient_norm(parameters, reference):
    norms = [p.grad.float().norm() for p in parameters if p.grad is not None]
    return torch.linalg.vector_norm(torch.stack(norms)) if norms else reference.new_zeros(())


def lr_lambda(cfg):
    warm = cfg.lr_warmup_epochs

    def factor(epoch):
        if epoch < warm:
            return (epoch + 1) / warm
        span = max(cfg.epochs - warm, 1)
        return 0.5 * (1 + math.cos(math.pi * (epoch - warm) / span))
    return factor


class EMA:
    """参数指数滑动平均；前期用 (1+t)/(10+t) 缓启动，避免早期验证被随机初始化拖累。"""

    def __init__(self, model, decay):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        self.steps = 0
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.steps += 1
        d = min(self.decay, (1 + self.steps) / (10 + self.steps))
        for e, p in zip(self.model.parameters(), model.parameters()):
            e.lerp_(p.detach(), 1 - d)
        for e, b in zip(self.model.buffers(), model.buffers()):
            e.copy_(b)


def model_inputs(batch):
    return {"sino": batch.get("sino"), "n_views": batch.get("n_views")}


def train(cfg, resume=None):
    if cfg.segmentation_weight > 0 and cfg.epochs <= cfg.warmup_epochs:
        raise ValueError("联合训练必须 epochs > warmup_epochs")
    source_path = getattr(cfg, "seg_finetune_checkpoint", None)
    frozen_reconstruction = getattr(cfg, "freeze_reconstructor", False)
    if frozen_reconstruction and not (source_path or resume):
        raise ValueError("冻结重建器前必须指定 seg_finetune_checkpoint")
    if source_path and (cfg.segmentation_weight <= 0 or cfg.seg_pretrain_clean or cfg.warmup_epochs != 0
                        or cfg.ramp_epochs != 0):
        raise ValueError("从checkpoint微调分割须启用分割、关闭clean预训练并令warmup/ramp为0")
    if frozen_reconstruction and cfg.reconstruction_weight != 0:
        raise ValueError("冻结重建器时 reconstruction_weight 必须为0")
    seed_all(cfg.seed, cfg.cpu_threads)
    device = device_for(cfg)
    root = Path(cfg.run_dir)
    root.mkdir(parents=True, exist_ok=True)
    if not resume and (root / "last.pt").exists():
        raise FileExistsError("run_dir已有训练记录，请传入--resume或更换run_dir")
    splits = read_json(split_file(cfg))
    manifest = read_json(Path(cfg.cache_dir) / "audit.json")
    validate_split(splits, [r["id"] for r in manifest["patients"]])
    if splits["audit_fingerprint"] != manifest["fingerprint"]:
        raise ValueError("audit与冻结划分不一致")
    fingerprint = digest({k: splits[k] for k in ("train", "val", "test")})
    if fingerprint != splits["fingerprint"]:
        raise ValueError("冻结划分文件被修改，请为新实验重新生成划分")
    verify_cache(cfg, splits["train"] + splits["val"])
    dataset = TrainDataset(cfg, splits["train"])
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                        pin_memory=device.type == "cuda", persistent_workers=False)
    model = JointUNet(cfg).to(device)
    source_info = None
    if source_path and not resume:
        source_info = load_finetune_source(source_path, cfg, model, fingerprint, manifest["fingerprint"])
    if frozen_reconstruction:
        for parameter in model.reconstructor.parameters():
            parameter.requires_grad_(False)
        model.reconstructor.eval()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(cfg))
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    start, best, best_rec, stale = 0, -float("inf"), -float("inf"), 0
    history = []
    best_candidate = -float("inf")
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay > 0 else None
    if resume:
        model, checkpoint = load_checkpoint(resume, cfg, device)
        if checkpoint.get("training_protocol") != TRAINING_PROTOCOL:
            raise ValueError("旧版本checkpoint的学习率调度/EMA协议不同，不能续训；请另建run从头训练")
        if "model_raw" in checkpoint:
            model.load_state_dict(checkpoint["model_raw"])
        if frozen_reconstruction:
            for parameter in model.reconstructor.parameters():
                parameter.requires_grad_(False)
            model.reconstructor.eval()
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                      lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(cfg))
        if checkpoint["split_fingerprint"] != fingerprint or checkpoint["audit_fingerprint"] != manifest["fingerprint"]:
            raise ValueError("恢复训练的数据/患者划分指纹发生变化")
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start = checkpoint["epoch"] + 1
        best, best_rec, stale = checkpoint["best"], checkpoint["best_rec"], checkpoint["stale"]
        history = checkpoint["history"]
        best_candidate = checkpoint["best_candidate"]
        source_info = checkpoint.get("finetune_source")
        if cfg.ema_decay > 0:
            ema = EMA(model, cfg.ema_decay)
            ema.model.load_state_dict(checkpoint["model"])
            ema.steps = checkpoint.get("ema_steps", 0)
        torch.set_rng_state(checkpoint["rng_cpu"].cpu())
        if device.type == "cuda" and checkpoint["rng_cuda"]:
            torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint["rng_cuda"]])
    (root / "resolved_config.yaml").write_text(yaml.safe_dump(config_dict(cfg), allow_unicode=True), encoding="utf-8")
    write_json(root / "environment.json", {"python": platform.python_version(), "torch": str(torch.__version__),
                                          "device": str(device), "cuda": torch.version.cuda,
                                          "split_fingerprint": fingerprint, "audit_fingerprint": manifest["fingerprint"]})
    for epoch in range(start, cfg.epochs):
        dataset.epoch = epoch
        model.train()
        if frozen_reconstruction:
            model.reconstructor.eval()
        total, rec_total, seg_total, count = 0., 0., 0., 0
        rec_grad_total, seg_grad_total = 0., 0.
        extra_totals = {}
        for batch in tqdm(loader, desc=f"epoch {epoch+1}/{cfg.epochs}"):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                if cfg.seg_pretrain_clean and epoch < cfg.warmup_epochs:
                    restored = model.restore(batch["input"], batch["view"], **model_inputs(batch))
                    mix = reconstructed_input_fraction(cfg, epoch)
                    seg_restored = restored.detach() + cfg.seg_to_recon_scale * (restored - restored.detach())
                    seg_input = (1 - mix) * batch["target"] + mix * seg_restored
                    logits = model.segment(seg_input, batch["view"])
                else:
                    restored, logits = model(batch["input"], batch["view"], **model_inputs(batch))
                loss, parts = joint_loss(restored, logits, batch, cfg, epoch, model.reconstruction_aux(),
                                         model.segmentation_aux(), model)
            if not torch.isfinite(loss):
                raise FloatingPointError("训练损失非有限值，停止以避免保存损坏模型")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nonfinite = False
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    nonfinite = True
                    break
            if nonfinite:
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                print("Skip non-finite gradient batch", flush=True)
                continue
            rec_norm = gradient_norm(model.reconstructor.parameters(), loss)
            seg_norm = gradient_norm(model.segmenter.parameters(), loss)
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)
            n = len(batch["input"])
            for key in ("sinogram", "dual_domain_image", "projection_consistency"):
                if key in parts:
                    extra_totals[key] = extra_totals.get(key, 0.) + parts[key]*n
            total += float(loss.detach())*n
            rec_total += parts["reconstruction"]*n
            seg_total += parts["segmentation"]*n
            count += n
            rec_grad_total += float(rec_norm)*n
            seg_grad_total += float(seg_norm)*n
        scheduler.step()
        if count == 0:
            raise FloatingPointError(f"epoch {epoch+1} 所有 batch 的梯度都非有限，训练已发散；请降低 lr 或检查输入")
        record = {"epoch": epoch+1, "loss": total/count, "reconstruction_loss": rec_total/count,
                  "segmentation_loss": seg_total/count, "seg_ramp": parts["seg_ramp"], "lr": optimizer.param_groups[0]["lr"],
                  "reconstructor_grad_norm":rec_grad_total/count,"segmenter_grad_norm":seg_grad_total/count,
                  "phase":("segmentation_finetune" if source_info and frozen_reconstruction else
                           "clean_seg_pretrain" if cfg.seg_pretrain_clean and epoch<cfg.warmup_epochs else "joint"),
                  "seg_reconstructed_input_fraction": reconstructed_input_fraction(cfg, epoch),
                  **{f"{k}_loss": v/count for k, v in extra_totals.items()}}
        eval_model = ema.model if ema is not None else model
        improved, improved_rec, improved_candidate = False, False, False
        if (epoch+1) % cfg.validation_every == 0 or epoch+1 == cfg.epochs:
            summary = evaluate_patients(eval_model, splits["val"], cfg, device, root / "validation" / f"epoch_{epoch+1:04d}")
            score = validation_score(summary, cfg)
            rec_score = reconstruction_score(summary,cfg)
            eligibility = reconstruction_eligibility(summary,cfg)
            seg_eligibility = segmentation_eligibility(summary, cfg)
            if rec_score > best_rec:
                best_rec, improved_rec = rec_score, True
            # 联合分割未开始前不参与best.pt选模/早停计数。
            start_epoch = cfg.warmup_epochs if cfg.selection_start_epoch is None else cfg.selection_start_epoch
            if epoch >= start_epoch or cfg.segmentation_weight == 0:
                improved_candidate = score > best_candidate
                if improved_candidate:
                    best_candidate = score
                improved = score > best and eligibility["eligible"] and seg_eligibility["eligible"]
                if improved:
                    best, stale = score, 0
                else:
                    stale += 1
            record.update({"validation_score": score, "validation_reconstruction_score": rec_score,
                           "reconstruction_eligibility": eligibility,
                           "segmentation_eligibility": seg_eligibility})
        history.append(record)
        write_json(root / "history.json", history)
        ckpt = {"schema": 3, "training_protocol": TRAINING_PROTOCOL, "epoch": epoch, "model": eval_model.state_dict(),
                **({"model_raw": model.state_dict(), "ema_steps": ema.steps} if ema is not None else {}),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "config": config_dict(cfg),
                "split_fingerprint": fingerprint, "audit_fingerprint": manifest["fingerprint"],
                "best": best, "best_rec": best_rec, "best_candidate":best_candidate, "stale": stale, "history": history,
                "finetune_source": source_info,
                "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
        saves = [("last.pt", True), ("best.pt", improved), ("best_candidate.pt",improved_candidate), ("best_reconstruction.pt", improved_rec)]
        if cfg.save_every_epochs > 0 and (epoch+1) % cfg.save_every_epochs == 0:
            saves.append((f"epoch_{epoch+1:04d}.pt", True))
        for name, save in saves:
            if save:
                temporary = root / (name + ".tmp")
                torch.save(ckpt, temporary)
                temporary.replace(root / name)
        print(record, flush=True)
        if cfg.patience_validations > 0 and epoch >= cfg.warmup_epochs+cfg.ramp_epochs and stale >= cfg.patience_validations:
            print("Early stopping: 验证综合指标未改善", flush=True)
            break
    selected = root / "best.pt"
    write_json(root/"selection_report.json",{"qualified_checkpoint_exists":selected.exists(),
               "selected_checkpoint":str(selected.resolve()) if selected.exists() else None,
               "candidate_checkpoint":str((root/"best_candidate.pt").resolve()),"guard_enabled":cfg.selection_guard,
               "note":"验证集门槛不构成测试性能保证；如无best.pt，应检查各视角退化而不是自动换用候选模型。"})
    if not selected.exists():
        print("未产生满足验证重建门槛的best.pt。保留best_candidate.pt供诊断；请查看selection_report.json。",flush=True)
    elif cfg.postprocess and cfg.auto_tune_postprocess and cfg.segmentation_weight > 0:
        tune_postprocess(cfg, str(selected), "val")
        if cfg.selection_32_lesion_priority:
            tune_32_postprocess(cfg, str(selected), "val")
    return str(selected) if selected.exists() else None
