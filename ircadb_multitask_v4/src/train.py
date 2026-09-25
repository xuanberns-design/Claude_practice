"""混合视角训练、预热/联合优化、验证选模与epoch边界恢复。

参数入口：config.py 所有训练参数；不读取测试患者缓存。
"""
from pathlib import Path
import platform
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
from .common import read_json, write_json, seed_all, device_for, digest
from .config import config_dict, split_file
from .dataset import TrainDataset
from .model import JointUNet
from .losses import joint_loss
from .prepare import verify_cache
from .split import validate_split
from .evaluate import evaluate_patients, validation_score, load_checkpoint, reconstruction_score, reconstruction_eligibility


def train(cfg, resume=None):
    if cfg.segmentation_weight > 0 and cfg.epochs <= cfg.warmup_epochs:
        raise ValueError("联合训练必须 epochs > warmup_epochs")
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    start, best, best_rec, stale = 0, -float("inf"), -float("inf"), 0
    history = []
    best_candidate = -float("inf")
    if resume:
        model, checkpoint = load_checkpoint(resume, cfg, device)
        if checkpoint.get("training_protocol") != "v2":
            raise ValueError("V1与V2损失和训练阶段不同，不能沿用V1优化器状态续训；请另建run从头训练")
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
        if checkpoint["split_fingerprint"] != fingerprint or checkpoint["audit_fingerprint"] != manifest["fingerprint"]:
            raise ValueError("恢复训练的数据/患者划分指纹发生变化")
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start = checkpoint["epoch"] + 1
        best, best_rec, stale = checkpoint["best"], checkpoint["best_rec"], checkpoint["stale"]
        history = checkpoint["history"]
        best_candidate = checkpoint["best_candidate"]
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
        total, rec_total, seg_total, count = 0., 0., 0., 0
        rec_grad_total, seg_grad_total = 0., 0.
        for batch in tqdm(loader, desc=f"epoch {epoch+1}/{cfg.epochs}"):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                if cfg.seg_pretrain_clean and epoch < cfg.warmup_epochs:
                    restored = model.restore(batch["input"],batch["view"])
                    logits = model.segment(batch["target"],batch["view"])
                else:
                    restored, logits = model(batch["input"], batch["view"])
                loss, parts = joint_loss(restored, logits, batch, cfg, epoch)
            if not torch.isfinite(loss):
                raise FloatingPointError("训练损失非有限值，停止以避免保存损坏模型")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            rec_norm = torch.linalg.vector_norm(torch.stack([p.grad.float().norm() for p in model.reconstructor.parameters() if p.grad is not None]))
            seg_norm = torch.linalg.vector_norm(torch.stack([p.grad.float().norm() for p in model.segmenter.parameters() if p.grad is not None]))
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            n = len(batch["input"])
            total += float(loss.detach())*n
            rec_total += parts["reconstruction"]*n
            seg_total += parts["segmentation"]*n
            count += n
            rec_grad_total += float(rec_norm)*n
            seg_grad_total += float(seg_norm)*n
        scheduler.step()
        record = {"epoch": epoch+1, "loss": total/count, "reconstruction_loss": rec_total/count,
                  "segmentation_loss": seg_total/count, "seg_ramp": parts["seg_ramp"], "lr": optimizer.param_groups[0]["lr"],
                  "reconstructor_grad_norm":rec_grad_total/count,"segmenter_grad_norm":seg_grad_total/count,
                  "phase":"clean_seg_pretrain" if cfg.seg_pretrain_clean and epoch<cfg.warmup_epochs else "joint"}
        improved, improved_rec, improved_candidate = False, False, False
        if (epoch+1) % cfg.validation_every == 0 or epoch+1 == cfg.epochs:
            summary = evaluate_patients(model, splits["val"], cfg, device, root / "validation" / f"epoch_{epoch+1:04d}")
            score = validation_score(summary, cfg)
            rec_score = reconstruction_score(summary,cfg)
            eligibility = reconstruction_eligibility(summary,cfg)
            if rec_score > best_rec:
                best_rec, improved_rec = rec_score, True
            # 联合分割未开始前不参与best.pt选模/早停计数。
            if epoch >= cfg.warmup_epochs or cfg.segmentation_weight == 0:
                improved_candidate = score > best_candidate
                if improved_candidate:
                    best_candidate = score
                improved = score > best and eligibility["eligible"]
                if improved:
                    best, stale = score, 0
                else:
                    stale += 1
            record.update({"validation_score": score, "validation_reconstruction_score": rec_score,
                           "reconstruction_eligibility":eligibility})
        history.append(record)
        write_json(root / "history.json", history)
        ckpt = {"schema": 2,"training_protocol":"v2", "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "config": config_dict(cfg),
                "split_fingerprint": fingerprint, "audit_fingerprint": manifest["fingerprint"],
                "best": best, "best_rec": best_rec, "best_candidate":best_candidate, "stale": stale, "history": history,
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
    return str(selected) if selected.exists() else None
