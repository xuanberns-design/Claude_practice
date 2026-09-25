"""验证专家/预测标签分离以及每层七张 PNG 的完整导出。"""
import numpy as np
import pytest
import torch
from PIL import Image

from src.config import Config
from src.visualize import comparison_figure, PANEL_FILES, LIVER_COLOR, TUMOR_COLOR
from src.evaluate import export_slice_figures
from src.common import read_json
import src.evaluate as evaluate_module


def example_slice():
    y, x = np.mgrid[:32, :32]
    truth = (x * 4 + y * 2 - 80).astype(np.float32)
    fbp = truth + np.sin(x) * 20
    restored = truth + 2
    expert = np.zeros((2, 32, 32), dtype=np.uint8)
    expert[0, 4:20, 4:20] = 1
    expert[1, 8:14, 8:14] = 1
    probability = np.zeros((2, 32, 32), dtype=np.float32)
    probability[0, 8:24, 8:24] = 0.8
    probability[1, 14:18, 14:18] = 0.95
    return truth, fbp, restored, expert, probability


def test_six_panel_uses_correct_ct_expert_and_predicted_labels_without_mutation(tmp_path):
    arrays = example_slice()
    before = [x.copy() for x in arrays]
    cfg = Config()
    out = tmp_path / "patient1" / "32_views_slice16.png"
    fig = comparison_figure(*arrays, cfg, out, patient=1, view=32, z=16)
    assert out.is_file()
    assert len(fig.axes) == 6
    assert [len(a.images) for a in fig.axes] == [1, 1, 1, 2, 1, 1]
    truth, fbp, restored, expert, probability = arrays
    for ax, expected in zip((fig.axes[0], fig.axes[1], fig.axes[3], fig.axes[4]),
                            (truth, fbp, restored, restored)):
        np.testing.assert_array_equal(ax.images[0].get_array(), expected)
        assert ax.images[0].get_clim() == (cfg.window_min, cfg.window_max)
    expert_labels = np.asarray(fig.axes[2].images[0].get_array())
    assert expert_labels[0, 0] == 0
    assert expert_labels[5, 5] == 1
    assert expert_labels[10, 10] == 2
    assert expert_labels[22, 22] == 0
    labels = np.asarray(fig.axes[5].images[0].get_array())
    assert set(np.unique(labels)) == {0, 1, 2}
    assert labels[0, 0] == 0
    assert labels[10, 10] == 1
    assert labels[15, 15] == 2  # 肝/肿瘤重叠处展示肿瘤颜色。
    assert labels[22, 22] == 1
    rgba = np.asarray(fig.axes[3].images[1].get_array())
    assert rgba[0, 0, 3] == 0
    np.testing.assert_allclose(rgba[10, 10, :3], LIVER_COLOR)
    np.testing.assert_allclose(rgba[15, 15, :3], TUMOR_COLOR)
    for actual, original in zip(arrays, before):
        np.testing.assert_array_equal(actual, original)
    assert "32 sparse views" in fig._suptitle.get_text()
    assert "Expert" in fig.axes[2].get_title()


def test_visualization_rejects_misalignment_and_logits(tmp_path):
    truth, fbp, restored, expert, probability = example_slice()
    with pytest.raises(ValueError, match="相同尺寸"):
        comparison_figure(truth, fbp[:31], restored, expert, probability,
                          Config(), tmp_path / "bad.png")
    with pytest.raises(ValueError, match="专家 masks"):
        comparison_figure(truth, fbp, restored, expert[:, :31], probability,
                          Config(), tmp_path / "bad.png")
    with pytest.raises(ValueError, match="logits"):
        comparison_figure(truth, fbp, restored, expert, probability + 2,
                          Config(), tmp_path / "bad.png")


def test_threshold_is_respected(tmp_path):
    arrays = example_slice()
    fig = comparison_figure(*arrays, Config(segmentation_threshold=0.9), tmp_path / "threshold.png")
    labels = np.asarray(fig.axes[5].images[0].get_array())
    assert labels[10, 10] == 0
    assert labels[15, 15] == 2


def test_every_slice_exports_composite_and_six_separate_named_panels(tmp_path):
    truth, fbp, restored, expert, probability = example_slice()
    images = [np.stack([x, x + 1]) for x in (truth, fbp, restored)]
    masks = np.stack([expert, expert])
    masks[1, 1] = 0
    probabilities = np.stack([probability, probability])
    cfg = Config(save_slice_every=0)  # 旧配置值也不能跳过用户要求的任何一层。
    dest = tmp_path / "volumes" / "patient1" / "32"
    export_slice_figures(*images, masks, probabilities, cfg, dest, patient="patient1", view=32)
    for z in range(2):
        slice_dir = dest / "slices" / f"slice_{z:04d}"
        assert {p.name for p in slice_dir.glob("*.png")} == {"comparison.png", *PANEL_FILES}
        with Image.open(slice_dir / "comparison.png") as composite:
            for name in PANEL_FILES:
                with Image.open(slice_dir / name) as panel:
                    assert panel.size[0] >= 400 and panel.size[1] >= 400
                    assert panel.size[0] < composite.size[0]
                    assert panel.size[1] < composite.size[1]
    assert (dest / "comparison.png").read_bytes() == (dest / "slices" / "slice_0000" / "comparison.png").read_bytes()
    manifest = read_json(dest / "visualization_manifest.json")
    assert manifest["n_slices"] == 2
    assert manifest["files_per_slice"] == ["comparison.png", *PANEL_FILES]


def test_evaluate_exports_every_view_and_slice_for_each_patient(tmp_path, monkeypatch):
    truth, _, _, expert, probability = example_slice()

    class FakePatient:
        def __init__(self, cfg, pid):
            self.root = tmp_path / pid
            self.meta = {"n_slices": 2, "spacing_zyx": [1, 1, 1],
                         "affine_lps_xyz": np.eye(4).tolist()}

        def get(self, key):
            if key == "native_hu":
                return np.stack([truth, truth + 1])
            if key == "native_masks":
                return np.stack([expert, expert])
            if key == "full_fbp":
                return np.full((2, 32, 32), 0.50, dtype=np.float32)
            if key.startswith("fbp_"):
                return np.full((2, 32, 32), 0.48, dtype=np.float32)
            raise KeyError(key)

    def fake_predict(*_args):
        return np.full((2, 32, 32), 0.49, dtype=np.float32), np.stack([probability, probability])

    monkeypatch.setattr(evaluate_module, "PatientCache", FakePatient)
    monkeypatch.setattr(evaluate_module, "predict_volume", fake_predict)
    monkeypatch.setattr(evaluate_module, "save_nifti", lambda *_args: None)
    cfg = Config(cache_dir=str(tmp_path / "cache"), bootstrap_repeats=2, save_slice_every=5)
    output = tmp_path / "evaluation"
    evaluate_module.evaluate_patients(torch.nn.Identity(), ["patient6", "patient7"],
                                      cfg, torch.device("cpu"), output, export=True)
    for patient in ("patient6", "patient7"):
        for view in (32, 64, 128):
            root = output / "volumes" / patient / str(view)
            assert (root / "comparison.png").exists()
            assert sorted(p.name for p in (root / "slices").iterdir()) == ["slice_0000", "slice_0001"]
            for z in range(2):
                assert {p.name for p in (root / "slices" / f"slice_{z:04d}").glob("*.png")} == {
                    "comparison.png", *PANEL_FILES}
