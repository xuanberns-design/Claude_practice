"""合成缓存验证增强同步，以及评估/无mask NIfTI推理的共享patch协议。"""
import importlib
import json

import nibabel as nib
import numpy as np
import pytest
import torch
from torch import nn

from src.config import Config
from src.dataset import PatientCache, TrainDataset, view_channel
from src.evaluate import predict_volume, save_nifti
from src.model import JointUNet
from src.predict import predict_batch
from src.projection import normalize, denormalize


@pytest.fixture
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def _write_meta(folder, n, *, z_origin=0.0):
    folder.mkdir(parents=True, exist_ok=True)
    meta = {"n_slices": n,
            "z_mm": (z_origin + np.arange(n) * 2.5).tolist(),
            "tumor_slices": list(range(n))}
    (folder / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


@pytest.mark.parametrize("foreground_probability", [0.0, 1.0])
def test_crop_flip_rotation_preserve_voxel_correspondence_and_repeatability(
        tmp_path, foreground_probability):
    """从增强后的可解析体素编码推导标签，不复制内部随机变换实现。"""
    folder = tmp_path / "cache" / "synthetic_patient"
    shape = (5, 64, 64)
    code = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)
    # 2的幂除法保留精确体素ID；非对称标签能暴露任意一个数组漏翻转/旋转。
    sparse = code.astype(np.float32) / 32768
    target = sparse * 2 + np.float32(0.125)
    masks = np.stack([(code % 11) < 5, (code % 17) < 3], axis=1).astype(np.uint8)
    _write_meta(folder, shape[0])
    np.save(folder / "fbp_32.npy", sparse)
    np.save(folder / "target.npy", target)
    np.save(folder / "masks.npy", masks)
    cfg = Config(cache_dir=str(folder.parent), image_size=64, patch_size=32,
                 context_slices=3, views=(32,), view_weights=(1.0,),
                 patch_foreground_probability=foreground_probability,
                 samples_per_epoch=64, seed=2026).validate()
    dataset = TrainDataset(cfg, [folder.name])
    orientations, first_voxels = set(), set()
    for index in range(64):
        batch = dataset[index]
        duplicate = dataset[index]
        for name in batch:
            torch.testing.assert_close(batch[name], duplicate[name], rtol=0, atol=0)
        x = batch["input"].numpy()
        assert x.shape == (3, 32, 32)
        assert batch["mask"].shape == (2, 32, 32)
        assert batch["mask_stack"].shape == (3, 2, 32, 32)
        np.testing.assert_array_equal(batch["target"].numpy(), x * 2 + 0.125)
        recovered = np.rint(x * 32768).astype(np.int32)
        expected_stack = np.stack([(recovered % 11) < 5, (recovered % 17) < 3], axis=1)
        np.testing.assert_array_equal(batch["mask_stack"].numpy(), expected_stack)
        np.testing.assert_array_equal(batch["mask"].numpy(), expected_stack[1])
        assert batch["view"].item() == pytest.approx(view_channel(32))
        centre = recovered[1]
        orientations.add((int(centre[1, 0] - centre[0, 0]),
                          int(centre[0, 1] - centre[0, 0])))
        first_voxels.add(int(centre[0, 0]))
    # 覆盖全部8种平面方向及多个裁剪位置；实际执行随机crop/flip/rot90组合。
    assert len(orientations) == 8
    assert len(first_voxels) > 16
    original_epoch_sample = dataset[0]["input"].clone()
    dataset.epoch = 1
    assert not torch.equal(dataset[0]["input"], original_epoch_sample)
    for filename, original in (("fbp_32", sparse), ("target", target), ("masks", masks)):
        np.testing.assert_array_equal(np.load(folder / f"{filename}.npy"), original)


def test_evaluate_and_nifti_infer_share_patch_tta_predictions(
        tmp_path, monkeypatch, cpu_threads):
    """实际小UNet+非零残差head，验证同网格缓存与导出FBP NIfTI结果一致。"""
    infer_module = importlib.import_module("src.infer")
    folder = tmp_path / "cache" / "synthetic_patient"
    z, y, x = np.mgrid[:3, :64, :64]
    hu = (x * 3 + y * 2 + z * 10 - 130).astype(np.float32)
    _write_meta(folder, len(hu), z_origin=30.0)
    cfg = Config(cache_dir=str(folder.parent), image_size=64, patch_size=32,
                 context_slices=3, context_step_mm=2.5, base_channels=2,
                 inference_overlap=0.5, inference_tta=True, batch_size=2,
                 device="cpu", cpu_threads=2, amp=False).validate()
    np.save(folder / "fbp_32.npy", normalize(hu, cfg))
    with torch.random.fork_rng():
        torch.manual_seed(7)
        model = JointUNet(cfg).eval()
        # 初始head为0会退化为恒等映射；非零权重确保真正测试恢复分支。
        nn.init.normal_(model.reconstructor.head.weight, mean=0.0, std=0.02)
        nn.init.constant_(model.reconstructor.head.bias, 0.01)
    tile_shapes = []
    hook = model.register_forward_pre_hook(lambda _model, args: tile_shapes.append(tuple(args[0].shape[-2:])))
    try:
        patient = PatientCache(cfg, folder.name)
        expected_rec, expected_prob = predict_volume(model, patient, 32, cfg, torch.device("cpu"))
        eval_calls = len(tile_shapes)
        assert eval_calls > 2  # 不能退回两批整幅64图的直接前向。
        assert set(tile_shapes) == {(32, 32)}
        assert not np.allclose(expected_rec, normalize(hu, cfg), atol=1e-5)
        affine_lps = np.diag([1.0, 1.0, 2.5, 1.0])
        affine_lps[:3, 3] = [10.0, 20.0, 30.0]
        source = tmp_path / "exported_fbp_hu.nii.gz"
        save_nifti(source, hu, {"affine_lps_xyz": affine_lps.tolist()})
        assert nib.aff2axcodes(nib.load(source).affine) == ("L", "P", "S")
        monkeypatch.setattr(infer_module, "load_checkpoint",
                            lambda _path, _cfg, _device: (model, {"epoch": 3}))
        # 除最终标签外，也观察共享接口输出概率，排除额外sigmoid/阈值掩盖误差。
        seen = []
        real_predict = infer_module.predict_batch

        def capture_prediction(*args, **kwargs):
            r, p = real_predict(*args, **kwargs)
            seen.append((r.cpu().numpy().copy(), p.cpu().numpy().copy()))
            return r, p

        monkeypatch.setattr(infer_module, "predict_batch", capture_prediction)
        destination = tmp_path / "inference"
        infer_module.infer_nifti(cfg, "mock_checkpoint.pt", source, 32, destination)
        assert len(tile_shapes) - eval_calls == eval_calls
        assert set(tile_shapes) == {(32, 32)}
        np.testing.assert_allclose(np.concatenate([r for r, _ in seen]), expected_rec,
                                   rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(np.concatenate([p for _, p in seen]), expected_prob,
                                   rtol=1e-6, atol=1e-7)
        exported = nib.load(destination / "restored_hu.nii.gz")
        np.testing.assert_allclose(exported.get_fdata(dtype=np.float32).transpose(2, 1, 0),
                                   denormalize(expected_rec, cfg), rtol=1e-6, atol=1e-4)
        np.testing.assert_allclose(exported.affine, nib.load(source).affine, rtol=0, atol=0)
        for c, name in enumerate(("liver", "tumor")):
            actual = nib.load(destination / f"{name}.nii.gz").get_fdata().transpose(2, 1, 0)
            np.testing.assert_array_equal(actual, expected_prob[:, c] >= cfg.segmentation_threshold)
    finally:
        hook.remove()


def test_512_grid_256_patch_tta_covers_edges_with_correct_centre(cpu_threads):
    class PointwiseModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.shapes = []

        def forward(self, x, view):
            self.shapes.append(tuple(x.shape[-2:]))
            restored = x + 0.125
            logits = torch.cat([x[:, :1] * 2 - 1, -x[:, -1:] * 3 + 1], dim=1)
            return restored, logits

    cfg = Config(image_size=512, patch_size=256, context_slices=5,
                 inference_overlap=0.5, inference_tta=True).validate()
    x = torch.linspace(0, 1, 5 * 512 * 512).reshape(1, 5, 512, 512)
    model = PointwiseModel().eval()
    restored, probability = predict_batch(model, x, torch.tensor([view_channel(32)]), cfg)
    assert restored.shape == (1, 512, 512)
    assert probability.shape == (1, 2, 512, 512)
    assert len(model.shapes) == 36  # 3x3窗口，每窗口4种翻转。
    assert set(model.shapes) == {(256, 256)}
    assert torch.isfinite(restored).all() and torch.isfinite(probability).all()
    torch.testing.assert_close(restored, x[:, 2] + 0.125, rtol=1e-5, atol=1e-6)
    expected = torch.cat([x[:, :1] * 2 - 1, -x[:, -1:] * 3 + 1], dim=1).sigmoid()
    torch.testing.assert_close(probability, expected, rtol=1e-5, atol=1e-6)
