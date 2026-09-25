"""增强分支与困难样本监督的可运行性、梯度和数据采样回归测试。"""
import json

import numpy as np
import pytest
import torch

from src.config import Config
from src.dataset import TrainDataset
from src.losses import reconstruction_loss, segmentation_loss
from src.model import JointUNet


@pytest.mark.parametrize("upsample", ["bilinear", "pixelshuffle"])
def test_attention_reconstructor_initially_preserves_fbp_and_has_gradients(upsample):
    torch.set_num_threads(2)
    cfg = Config(image_size=32, context_slices=3, base_channels=4)
    cfg.reconstruction_backbone = "rcab"
    cfg.reconstruction_upsample = upsample
    model = JointUNet(cfg)
    x = torch.rand(2, 3, 32, 32)
    restored, logits = model(x, torch.tensor([0.5, 0.7]))
    assert torch.equal(restored, x)
    assert logits.shape == (2, 2, 32, 32)
    (restored.square().mean() + logits.square().mean()).backward()
    assert model.reconstructor.head.weight.grad.abs().sum() > 0
    assert model.segmenter.head.weight.grad.abs().sum() > 0


def test_enhanced_losses_are_finite_and_penalize_vessel_like_false_positive():
    cfg = Config(image_size=32, context_slices=1)
    cfg.laplacian_weight = 0.1
    cfg.seg_boundary_weight = 0.2
    cfg.liver_hard_negative_weight = 0.2
    cfg.liver_hard_negative_fraction = 0.02
    restored = torch.full((1, 1, 32, 32), 0.4, requires_grad=True)
    target = torch.full_like(restored, 0.4)
    target[:, :, 8:24, 8:24] += 0.01
    rec_loss = reconstruction_loss(restored, target, cfg)
    mask = torch.zeros((1, 2, 32, 32))
    mask[:, 0, 8:24, 8:24] = 1
    mask[:, 1, 12:18, 12:18] = 1
    clean = torch.full_like(mask, -6.)
    clean[:, 0, 8:24, 8:24] = 6
    clean[:, 1, 12:18, 12:18] = 6
    vessel_fp = clean.clone()
    vessel_fp[:, 0, 2:5, 2:5] = 6
    vessel_fp.requires_grad_()
    assert segmentation_loss(vessel_fp, mask, cfg) > segmentation_loss(clean, mask, cfg)
    loss = rec_loss + segmentation_loss(vessel_fp, mask, cfg)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(restored.grad).all()
    assert torch.isfinite(vessel_fp.grad).all()


def test_tumor_boundary_patch_sampling_keeps_lesion_and_context(tmp_path):
    cfg = Config(image_size=64, context_slices=1, cache_dir=str(tmp_path),
                 patch_size=32, samples_per_epoch=1, views=(32,), view_weights=(1,),
                 tumor_sample_probability=1, patch_foreground_probability=1)
    cfg.tumor_boundary_sample_probability = 1
    cfg.patch_center_jitter_fraction = 0
    folder = tmp_path / "3Dircadb1.1"
    folder.mkdir()
    (folder / "meta.json").write_text(json.dumps({"z_mm": [0], "n_slices": 1,
                                                  "tumor_slices": [0]}), encoding="utf-8")
    image = np.zeros((1, 64, 64), dtype=np.float32)
    masks = np.zeros((1, 2, 64, 64), dtype=np.uint8)
    masks[0, 0, 10:50, 10:50] = 1
    masks[0, 1, 24:40, 24:40] = 1
    np.save(folder / "fbp_32.npy", image)
    np.save(folder / "target.npy", image)
    np.save(folder / "masks.npy", masks)
    sample = TrainDataset(cfg, ["3Dircadb1.1"])[0]
    assert sample["input"].shape == (1, 32, 32)
    assert sample["mask"].shape == (2, 32, 32)
    assert sample["mask"][1].sum() > 0
    assert sample["mask"][1].sum() < 32 * 32
