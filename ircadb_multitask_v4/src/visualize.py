"""在同一 HU 窗下绘制患者单层六联图，并可导出六张独立面板。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import numpy as np
from PIL import Image


FIGURE_SIZE = (15, 10.8)
FIGURE_DPI = 150
LIVER_COLOR = (0.10, 0.85, 0.25)
TUMOR_COLOR = (1.00, 0.12, 0.12)
LIVER_ALPHA = 0.30
TUMOR_ALPHA = 0.65
TITLE_SIZE = 13

# 与六联图的位置一一对应；每层目录同时包含 comparison.png。
PANEL_FILES = (
    "01_ground_truth.png",
    "02_sparse_fbp.png",
    "03_expert_segmentation.png",
    "04_reconstruction_segmentation_overlay.png",
    "05_model_reconstruction.png",
    "06_predicted_segmentation.png",
)


def _labels(channels):
    """仅用于显示：重叠体素中肿瘤颜色优先，不改动原始双通道标签。"""
    labels = np.zeros(channels.shape[-2:], dtype=np.uint8)
    labels[channels[0]] = 1
    labels[channels[1]] = 2
    return labels


def _save_panels(fig, axes, composite_path, panel_dir):
    """复用六联图渲染结果裁取标题与图像，避免全层输出时重复渲染六次。"""
    panel_dir = Path(panel_dir)
    panel_dir.mkdir(parents=True, exist_ok=True)
    renderer = fig.canvas.get_renderer()
    with Image.open(composite_path) as composite:
        width, height = composite.size
        for ax, filename in zip(axes, PANEL_FILES):
            bbox = ax.get_tightbbox(renderer)
            pad = 8
            box = (max(0, int(np.floor(bbox.x0)) - pad),
                   max(0, height - int(np.ceil(bbox.y1)) - pad),
                   min(width, int(np.ceil(bbox.x1)) + pad),
                   min(height, height - int(np.floor(bbox.y0)) + pad))
            if box[2] <= box[0] or box[3] <= box[1]:
                raise RuntimeError(f"面板 {filename} 的裁切区域为空")
            composite.crop(box).save(panel_dir / filename)


def comparison_figure(truth, fbp, restored, masks, probability, cfg, path,
                      *, patient=None, view=None, z=None, panel_dir=None):
    """保存 2×3 六联图；可同时在 panel_dir 保存六张带标题的独立 PNG。

    truth/fbp/restored 为同一切片的二维 HU 数组；masks/probability
    均为 [liver,tumor,H,W]。图 3 是专家标签，图 6 是模型预测标签。
    """
    truth, fbp, restored = (np.asarray(x) for x in (truth, fbp, restored))
    masks, probability = np.asarray(masks), np.asarray(probability)
    if truth.ndim != 2 or fbp.shape != truth.shape or restored.shape != truth.shape:
        raise ValueError("六联图要求 truth/fbp/restored 是相同尺寸的二维 HU 切片")
    if masks.shape != (2, *truth.shape):
        raise ValueError("六联图专家 masks 必须为 [2,H,W]，通道顺序 liver、tumor")
    if probability.shape != (2, *truth.shape):
        raise ValueError("六联图 probability 必须为 [2,H,W]，通道顺序 liver、tumor")
    if not all(np.isfinite(x).all() for x in (truth, fbp, restored, masks, probability)):
        raise ValueError("六联图输入包含 NaN/Inf")
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError("六联图需要 [0,1] 概率，而不是分割 logits")

    expert_labels = _labels(masks > 0)
    # evaluate() passes the *final* postprocessed 0/1 mask here. Standalone
    # callers may still pass probabilities that require the configured cutoff.
    final_binary = bool(np.all((probability == 0) | (probability == 1)))
    predicted_labels = _labels(probability >= cfg.segmentation_threshold)
    overlay = np.zeros((*truth.shape, 4), dtype=np.float32)
    overlay[predicted_labels == 1] = (*LIVER_COLOR, LIVER_ALPHA)
    overlay[predicted_labels == 2] = (*TUMOR_COLOR, TUMOR_ALPHA)
    view_text = f" ({view} views)" if view is not None else ""
    titles = ("1 | Ground Truth (original CT)",
              f"2 | Sparse FBP{view_text}",
              "3 | Expert segmentation labels",
              "4 | Reconstruction + predicted segmentation",
              "5 | Model reconstruction (image only)",
              "6 | Predicted segmentation labels")
    # Figure 与保存使用同一 DPI，轴坐标才能直接映射到 PNG 像素。
    fig, grid = plt.subplots(2, 3, figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    axes = grid.ravel()
    try:
        for ax, title in zip(axes, titles):
            ax.set_title(title, fontsize=TITLE_SIZE, pad=10)
            ax.axis("off")
        for ax, img in zip((axes[0], axes[1], axes[3], axes[4]),
                           (truth, fbp, restored, restored)):
            ax.imshow(img, cmap="gray", vmin=cfg.window_min, vmax=cfg.window_max,
                      interpolation="nearest")
        axes[3].imshow(overlay, interpolation="nearest")
        cmap = ListedColormap([(0, 0, 0), LIVER_COLOR, TUMOR_COLOR])
        norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3)
        for ax, labels, kind in ((axes[2], expert_labels, "Expert"),
                                 (axes[5], predicted_labels, "Predicted")):
            ax.imshow(labels, cmap=cmap, norm=norm, interpolation="nearest")
            ax.legend(handles=[Patch(facecolor=LIVER_COLOR, label="Liver"),
                               Patch(facecolor=TUMOR_COLOR, label="Tumor")],
                      loc="lower center", ncol=2, fontsize=8, framealpha=0.85,
                      title=kind, title_fontsize=8)
        context = []
        if patient is not None:
            context.append(f"Patient {patient}")
        if view is not None:
            context.append(f"{view} sparse views")
        if z is not None:
            context.append(f"Slice {z} (zero-based)")
        fig.suptitle(" | ".join(context) if context else "Sparse-view CT reconstruction and segmentation",
                     fontsize=16, fontweight="bold", y=0.985)
        prediction_note = ("Postprocessed binary prediction" if final_binary else
                           f"Prediction threshold: {cfg.segmentation_threshold:g}")
        fig.text(0.5, 0.018,
                 f"CT window: [{cfg.window_min:g}, {cfg.window_max:g}] HU  |  "
                 f"{prediction_note}  |  "
                 "Panel 3: expert mask; panel 6: model prediction",
                 ha="center", va="center", fontsize=9, color="#444444")
        fig.subplots_adjust(left=0.025, right=0.975, bottom=0.075, top=0.925,
                            wspace=0.045, hspace=0.16)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=FIGURE_DPI, facecolor="white")
        if panel_dir is not None:
            _save_panels(fig, axes, path, panel_dir)
    finally:
        plt.close(fig)
    return fig
