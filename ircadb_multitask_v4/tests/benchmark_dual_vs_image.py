"""合成数据消融：同等训练步数下，V3 图像域后处理 vs V4 双域重建的稀疏角去条纹能力。

运行：python -m tests.benchmark_dual_vs_image --steps 600 --output work/benchmark.json
仅用随机椭圆“腹部”体模（无患者数据），用于验证方法方向与代码正确性；
真实 3D-IRCADb 上的数值必须用真实数据重新训练后在折外测试集上报告。
"""
import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity

from src.config import Config
from src.ct_ops import ParallelBeam, mask_unmeasured
from src.losses import reconstruction_loss, dual_domain_loss
from src.model import JointUNet


def random_phantom(n, rng):
    yy, xx = np.mgrid[:n, :n] / n - 0.5
    hu = np.full((n, n), -1000.0)

    def ellipse(cx, cy, ax, ay, angle):
        c, s = np.cos(angle), np.sin(angle)
        u, v = (xx - cx) * c + (yy - cy) * s, -(xx - cx) * s + (yy - cy) * c
        return (u / ax) ** 2 + (v / ay) ** 2 < 1

    body = ellipse(rng.uniform(-.03, .03), rng.uniform(-.03, .03), rng.uniform(.36, .44), rng.uniform(.26, .34),
                   rng.uniform(-.2, .2))
    hu[body] = rng.uniform(20, 50)
    hu[body & ellipse(-.22, 0, .05, .05, 0)] = rng.uniform(500, 900)          # 脊柱/骨
    liver = body & ellipse(rng.uniform(.05, .15), rng.uniform(-.1, .05), rng.uniform(.14, .2), rng.uniform(.1, .16),
                           rng.uniform(-.5, .5))
    hu[liver] = rng.uniform(80, 110)
    for _ in range(rng.integers(0, 4)):                                         # 低密度病灶
        hu[liver & ellipse(rng.uniform(0, .2), rng.uniform(-.15, .1), rng.uniform(.015, .05),
                           rng.uniform(.015, .05), 0)] = rng.uniform(30, 60)
    for _ in range(rng.integers(3, 8)):                                         # 血管/肠道高/低密度
        hu[body & ellipse(rng.uniform(-.3, .3), rng.uniform(-.2, .2), rng.uniform(.01, .03),
                          rng.uniform(.01, .03), 0)] = rng.choice([rng.uniform(150, 300), -600])
    return hu + body * rng.normal(0, 8, (n, n))


def make_set(count, n, op, views, seed):
    rng = np.random.default_rng(seed)
    hu = np.stack([random_phantom(n, rng) for _ in range(count)]).astype(np.float32)
    with torch.no_grad():
        dense = op.project(torch.from_numpy(hu / 1000 + 1).clamp_min(0), torch.arange(op.a))
        stride = op.a // views
        sparse = op.fbp(dense[:, ::stride], torch.arange(0, op.a, stride))
    return hu, dense.numpy(), ((sparse.numpy() - 1) * 1000).astype(np.float32)


def window_metrics(pred_hu, true_hu, low=-200, high=250):
    x = (np.clip(pred_hu, low, high) - low) / (high - low)
    y = (np.clip(true_hu, low, high) - low) / (high - low)
    mse = np.mean((x - y) ** 2)
    ssim = np.mean([structural_similarity(a, b, data_range=1, gaussian_weights=True, sigma=1.5,
                                          use_sample_covariance=False) for a, b in zip(x, y)])
    return {"SSIM": float(ssim), "PSNR_dB": float(-10 * np.log10(mse)), "MAE_HU": float(np.abs(pred_hu - true_hu).mean())}


def run(mode, cfg, data, test, steps, seed):
    torch.manual_seed(seed)
    model = JointUNet(cfg)
    opt = torch.optim.AdamW(model.reconstructor.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg.lr, total_steps=steps, pct_start=0.1)
    hu, dense, sparse = data
    norm = lambda a: (a - cfg.hu_min) / (cfg.hu_max - cfg.hu_min)
    rng = np.random.default_rng(seed)
    view = torch.full((cfg.batch_size,), float(np.log2(cfg.views[0]) / 10))
    n_views = torch.full((cfg.batch_size,), cfg.views[0], dtype=torch.long)
    start = time.time()
    for step in range(steps):
        idx = rng.choice(len(hu), cfg.batch_size, replace=False)
        batch = {"input": torch.from_numpy(norm(sparse[idx]))[:, None],
                 "target": torch.from_numpy(np.clip(norm(hu[idx]), 0, 1))[:, None],
                 "sino_target": torch.from_numpy(dense[idx])[:, None]}
        batch["sino"] = torch.from_numpy(mask_unmeasured(dense[idx], cfg.sino_dense_views, cfg.views[0]))[:, None]
        restored = model.restore(batch["input"], view, batch["sino"], n_views)
        loss = reconstruction_loss(restored, batch["target"], cfg, None, batch["input"])
        extra, _ = dual_domain_loss(model.reconstruction_aux(), batch, cfg, restored, model)
        opt.zero_grad()
        (loss + extra).backward()
        opt.step()
        sched.step()
    model.eval()
    t_hu, t_dense, t_sparse = test
    outputs = []
    with torch.no_grad():
        for i in range(0, len(t_hu), cfg.batch_size):
            x = torch.from_numpy(norm(t_sparse[i:i + cfg.batch_size]))[:, None]
            s = torch.from_numpy(mask_unmeasured(t_dense[i:i + cfg.batch_size], cfg.sino_dense_views, cfg.views[0]))[:, None]
            r = model.restore(x, view[:len(x)], s, n_views[:len(x)])
            outputs.append(r[:, 0].numpy() * (cfg.hu_max - cfg.hu_min) + cfg.hu_min)
    return {"mode": mode, "seconds": round(time.time() - start, 1),
            "parameters": sum(p.numel() for p in model.reconstructor.parameters()),
            **window_metrics(np.concatenate(outputs), t_hu)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--views", type=int, default=16)
    parser.add_argument("--dense", type=int, default=128)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--output", default="work/benchmark_dual_vs_image.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    op = ParallelBeam(args.size, args.dense)
    train, test = make_set(400, args.size, op, args.views, 1), make_set(60, args.size, op, args.views, 2)
    base = Config(image_size=args.size, context_slices=1, base_channels=16, views=(args.views,), view_weights=(1,),
                  full_views=args.dense, sino_dense_views=args.dense, batch_size=8, lr=1e-3, correction_weight=0.0)
    image = replace(base, reconstruction_mode="image").validate()
    dual = replace(base, reconstruction_mode="dual_domain", sino_base_channels=8, sino_weight=1.0,
                   dd_image_weight=0.5, fft_weight=0.05).validate()
    results = {"setting": vars(args), "synthetic_only": True,
               "FBP": window_metrics(test[2], test[0]),
               "interpolated_sinogram_FBP_untrained": None}
    with torch.no_grad():
        untrained = JointUNet(dual)
        s = torch.from_numpy(mask_unmeasured(test[1], args.dense, args.views))[:, None]
        v = torch.full((len(s),), float(np.log2(args.views) / 10))
        r = untrained.restore(torch.zeros(len(s), 1, args.size, args.size), v, s, torch.full((len(s),), args.views))
        results["interpolated_sinogram_FBP_untrained"] = window_metrics(r[:, 0].numpy() * 3000 - 1000, test[0])
    results["V3_image_domain"] = run("image", image, train, test, args.steps, 0)
    results["V4_dual_domain"] = run("dual_domain", dual, train, test, args.steps, 0)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
