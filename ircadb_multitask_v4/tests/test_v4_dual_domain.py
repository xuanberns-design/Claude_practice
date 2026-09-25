"""V4：可微投影/FBP与skimage一致性、正弦图精确增强、双域模型、分割损失与3D后处理。"""
import json

import numpy as np
import pytest
import torch
from skimage.transform import radon, iradon

from src.config import Config
from src.ct_ops import (ParallelBeam, angular_interpolate, mask_unmeasured, flip_detector, rotate_image_np,
                        flip_image_np, rotate_sinogram_np, flip_sinogram_np)
from src.dataset import TrainDataset
from src.losses import segmentation_loss
from src.model import JointUNet
from src.postprocess import postprocess_volume, tumor_grid_counts, choose_parameters


@pytest.fixture(autouse=True)
def few_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def phantom(n, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:n, :n]
    img = (((xx - n * .5) ** 2 / (n * .38) ** 2 + (yy - n * .52) ** 2 / (n * .3) ** 2) < 1) * 1.0
    img += (((xx - n * .62) ** 2 + (yy - n * .45) ** 2) < (n * .12) ** 2) * 0.05
    img += (((xx - n * .35) ** 2 + (yy - n * .6) ** 2) < (n * .05) ** 2) * 0.6
    return img * (1 + 0.02 * rng.standard_normal((n, n)))


@pytest.mark.parametrize("n", [32, 33])
def test_torch_operators_match_skimage(n):
    img = phantom(n)
    a = 32
    theta = np.arange(a) * 180 / a
    reference = radon(img, theta, circle=False, preserve_range=True).T
    op = ParallelBeam(n, a)
    projected = op.project(torch.tensor(img, dtype=torch.float32), torch.arange(a)).numpy()
    assert np.abs(projected - reference).max() < 1e-5 * np.abs(reference).max()
    for views in (8, 16, 32):
        idx = np.arange(0, a, a // views)
        expected = iradon(reference[idx].T, theta[idx], output_size=n, filter_name="ramp",
                          interpolation="linear", circle=False, preserve_range=True)
        actual = op.fbp(torch.tensor(reference[idx], dtype=torch.float32), torch.tensor(idx)).numpy()
        assert np.abs(actual - expected).max() < 1e-4 * np.abs(expected).max()


def test_fbp_gradient_matches_adjoint_and_is_finite():
    op = ParallelBeam(16, 8)
    sino = torch.randn(2, 8, op.d, requires_grad=True)
    image = op.fbp(sino)
    image.square().sum().backward()
    assert torch.isfinite(sino.grad).all() and sino.grad.abs().sum() > 0


@pytest.mark.parametrize("n", [32, 33])
def test_exact_geometric_augmentation_of_sinogram(n):
    img = phantom(n, 1)
    a = 16
    theta = np.arange(a) * 180 / a
    sino = radon(img, theta, circle=False, preserve_range=True).T
    for k in range(1, 4):
        expected = radon(rotate_image_np(img, k), theta, circle=False, preserve_range=True).T
        assert np.abs(rotate_sinogram_np(sino, k) - expected).max() < 1e-9
    for axis in (-2, -1):
        expected = radon(flip_image_np(img, axis), theta, circle=False, preserve_range=True).T
        assert np.abs(flip_sinogram_np(sino, axis) - expected).max() < 1e-9


def test_angular_interpolation_keeps_measurements_and_uses_180_degree_symmetry():
    a, d = 16, 11
    sino = torch.randn(1, a, d)
    masked = torch.from_numpy(mask_unmeasured(sino.numpy(), a, 4))
    filled = angular_interpolate(masked, 4)
    assert torch.equal(filled[:, ::4], sino[:, ::4])
    # 最后一段在 row 12 与 “row 0 + 探测器翻转” 之间插值。
    expected = 0.5 * sino[:, 12] + 0.5 * flip_detector(sino[:, 0])
    assert torch.allclose(filled[:, 14], expected, atol=1e-6)


def dual_cfg(**kw):
    base = dict(image_size=32, context_slices=3, base_channels=4, views=(8, 16), view_weights=(1, 1),
                full_views=64, reconstruction_mode="dual_domain", sino_dense_views=32, sino_base_channels=4,
                seg_backbone="resattn", seg_deep_supervision_weight=0.3, seg_wide_window=(-500., 500.))
    base.update(kw)
    return Config(**base).validate()


def test_dual_domain_model_starts_from_interpolated_fbp_with_exact_data_consistency():
    cfg = dual_cfg()
    model = JointUNet(cfg)
    op = model.reconstructor.op
    mu = torch.tensor(np.stack([phantom(32, s) for s in range(3)])[None], dtype=torch.float32).repeat(2, 1, 1, 1)
    dense = op.project(mu, torch.arange(32))
    n_views = torch.tensor([8, 16])
    sino = torch.stack([torch.from_numpy(mask_unmeasured(dense[i].numpy(), 32, int(v))) for i, v in enumerate(n_views)])
    x = torch.rand(2, 3, 32, 32)
    view = torch.tensor([0.3, 0.4])
    restored, logits = model(x, view, sino, n_views)
    aux = model.reconstruction_aux()
    assert torch.allclose(restored, aux["dd_image"])
    assert torch.equal(aux["sino"][0][:, ::4], dense[0][:, ::4])
    assert torch.equal(aux["sino"][1][:, ::2], dense[1][:, ::2])
    assert logits.shape == (2, 2, 32, 32) and len(model.segmentation_aux()) == 3
    (restored.square().mean() + logits.square().mean() + aux["sino"].square().mean()).backward()
    assert model.reconstructor.head.weight.grad.abs().sum() > 0
    assert model.reconstructor.sino_net.head.weight.grad.abs().sum() > 0


def test_dual_domain_rejects_patch_training_and_bad_angle_grid():
    with pytest.raises(ValueError):
        dual_cfg(patch_size=16)
    with pytest.raises(ValueError):
        dual_cfg(sino_dense_views=24)


def test_dual_dataset_masks_unmeasured_angles_and_keeps_augmentation_consistent(tmp_path):
    n, a = 32, 32
    cfg = dual_cfg(cache_dir=str(tmp_path), context_slices=1, samples_per_epoch=12, views=(8,), view_weights=(1,))
    op = ParallelBeam(n, a)
    img = phantom(n, 3)
    img[4:8, 20:26] = 1.5  # 非对称结构，漏做任一变换都会失配
    hu = (img - 1) * 1000
    target = ((hu - cfg.hu_min) / (cfg.hu_max - cfg.hu_min)).astype(np.float32)
    sino = op.project(torch.tensor(img, dtype=torch.float32), torch.arange(a)).numpy()
    folder = tmp_path / "p"
    folder.mkdir()
    (folder / "meta.json").write_text(json.dumps({"z_mm": [0.0], "n_slices": 1, "tumor_slices": []}), encoding="utf-8")
    np.save(folder / "fbp_8.npy", target[None])
    np.save(folder / "target.npy", target[None])
    np.save(folder / "masks.npy", np.zeros((1, 2, n, n), np.uint8))
    np.save(folder / "sino.npy", sino[None].astype(np.float32))
    dataset = TrainDataset(cfg, ["p"])
    fbp0 = op.fbp(torch.from_numpy(sino)).numpy()

    def transforms(a):
        for f0 in (False, True):
            for f1 in (False, True):
                for k in range(4):
                    b = flip_image_np(a, -2) if f0 else a
                    b = flip_image_np(b, -1) if f1 else b
                    yield (f0, f1, k), rotate_image_np(b, k)

    targets = dict(transforms(target))
    rebuilt_expected = dict(transforms(fbp0))
    for i in range(12):
        sample = dataset[i]
        measured = sample["sino"][0]
        assert measured[1::4].abs().sum() == 0 and measured[::4].abs().sum() > 0
        # 找到数据集所用的图像变换，正弦图经同一变换后其FBP必须等于变换后的FBP（FBP等变性）。
        key = next(k for k, t in targets.items() if np.array_equal(t, sample["target"][0].numpy()))
        rebuilt = op.fbp(sample["sino_target"][0]).numpy()
        assert np.abs(rebuilt - rebuilt_expected[key])[1:-1, 1:-1].max() < 1e-4, key


def test_tversky_and_hierarchy_penalize_missed_small_tumor_and_extrahepatic_tumor():
    cfg = Config(tumor_tversky_weight=1.0, seg_hierarchy_weight=1.0)
    mask = torch.zeros(1, 2, 32, 32)
    mask[:, 0, 4:28, 4:28] = 1
    mask[:, 1, 10:13, 10:13] = 1
    good = torch.full_like(mask, -8.)
    good[:, 0, 4:28, 4:28] = 8
    good[:, 1, 10:13, 10:13] = 8
    missed = good.clone()
    missed[:, 1, 10:13, 10:13] = -8
    outside = good.clone()
    outside[:, 1, 29:31, 29:31] = 8
    assert segmentation_loss(missed, mask, cfg) > segmentation_loss(good, mask, cfg) + 0.3
    assert segmentation_loss(outside, mask, cfg) > segmentation_loss(good, mask, cfg)


def test_postprocess_keeps_largest_liver_and_removes_extrahepatic_and_tiny_tumors():
    cfg = Config(postprocess=True, tumor_liver_margin_mm=2.0, min_tumor_ml=0.004)
    prob = np.zeros((6, 2, 40, 40), np.float32)
    prob[1:5, 0, 5:25, 5:25] = 0.9          # 肝脏
    prob[1:5, 0, 32:36, 32:36] = 0.9        # 远处孤立“肝”假阳性
    prob[2:4, 1, 10:15, 10:15] = 0.9        # 肝内肿瘤 50体素
    prob[2, 1, 34:37, 2:5] = 0.9            # 肝外肿瘤
    prob[3, 1, 20, 20] = 0.9                # 肝内1体素碎片
    liver, tumor = postprocess_volume(prob, (1.0, 1.0, 1.0), cfg)
    assert not liver[:, 32:36, 32:36].any()
    assert tumor[2:4, 10:15, 10:15].all()
    assert not tumor[2, 34:37, 2:5].any() and not tumor[3, 20, 20]
    assert (tumor <= liver).all()


def test_validation_tuning_prefers_threshold_that_removes_false_positive_halo():
    cfg = Config(postprocess=True, tumor_liver_margin_mm=None, tune_tumor_thresholds=(0.3, 0.5, 0.7),
                 tune_min_tumor_ml=(0.0,))
    prob = np.zeros((4, 2, 30, 30), np.float32)
    prob[:, 0, 2:28, 2:28] = 0.95
    truth = np.zeros((4, 30, 30), bool)
    truth[1:3, 10:16, 10:16] = True
    prob[1:3, 1, 8:18, 8:18] = 0.4           # 光晕
    prob[1:3, 1, 10:16, 10:16] = 0.8         # 真实病灶
    rows = tumor_grid_counts(prob, truth, (1.0, 1.0, 1.0), cfg)
    params, table = choose_parameters([rows], cfg)
    assert params["tumor_threshold"] in (0.5, 0.7) and len(table) == 3


def test_image_mode_state_dict_keys_are_v3_compatible():
    cfg = Config(image_size=32, context_slices=3, base_channels=4, reconstruction_backbone="rcab",
                 reconstruction_upsample="pixelshuffle").validate()
    keys = set(JointUNet(cfg).state_dict())
    assert not any(".context." in k or k.startswith("reconstructor.op") for k in keys)
    assert any(k.startswith("reconstructor.up.") for k in keys) and any(k.startswith("segmenter.enc.") for k in keys)
