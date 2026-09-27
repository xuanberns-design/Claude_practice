"""Regression checks for 32-view calibration and segmentation fine-tuning."""
import numpy as np
import pytest
import torch
from dataclasses import replace

from src.config import Config, config_dict
from src.losses import joint_loss
from src.metrics import lesion_detection, segmentation_metrics
from src.model import JointUNet
from src.postprocess import choose_32_parameters, postprocess_volume
from src.train import load_finetune_source, reconstructed_input_fraction


def test_per_view_calibration_changes_only_32_view_output():
    cfg = Config(postprocess=True, tumor_threshold=0.3, min_tumor_ml=0,
                 tumor_liver_margin_mm=5.0)
    probability = np.zeros((1, 2, 32, 32), dtype=np.float32)
    probability[0, 0, 8:12, 8:12] = 0.9
    probability[0, 1, 10, 19] = 0.2  # 8 mm from the predicted liver
    parameters = {"tumor_threshold": 0.3, "min_tumor_ml": 0.0,
                  "per_view": {"32": {"tumor_threshold": 0.15,
                                      "min_tumor_ml": 0.0,
                                      "tumor_liver_margin_mm": 10.0}}}
    liver32, tumor32 = postprocess_volume(probability, (1., 1., 1.), cfg, parameters, 32)
    liver64, tumor64 = postprocess_volume(probability, (1., 1., 1.), cfg, parameters, 64)
    assert tumor32[0, 10, 19] and liver32[0, 10, 19]
    assert not tumor64.any() and not liver64[0, 10, 19]
    truth = np.zeros_like(probability)
    truth[0, 1, 10, 19] = 1
    result32 = segmentation_metrics(probability, truth, (1., 1., 1.), cfg, parameters,
                                    surface=False, view=32)
    result64 = segmentation_metrics(probability, truth, (1., 1., 1.), cfg, parameters,
                                    surface=False, view=64)
    assert result32["tumor_Dice"] == 1.0
    assert result64["tumor_Dice"] == 0.0


def test_one_predicted_component_cannot_claim_two_lesions():
    truth = np.zeros((1, 12, 12), dtype=bool)
    truth[0, 5, 3] = truth[0, 5, 7] = True
    pred = np.zeros_like(truth)
    pred[0, 5, 3:8] = True
    recall, precision, gt_count, pred_count = lesion_detection(pred, truth)
    assert (recall, precision, gt_count, pred_count) == (0.5, 1.0, 2, 1)


def test_32_calibration_respects_negative_fp_limit():
    cfg = Config(max_negative_fp_ml=5.)
    baseline = {"tumor_threshold": 0.3, "min_tumor_ml": 0.2,
                "tumor_liver_margin_mm": 5.0}

    def row(threshold, recall, dice, fp, positive):
        return {"tumor_threshold": threshold, "min_tumor_ml": 0.,
                "tumor_liver_margin_mm": 10., "lesion_recall": recall if positive else None,
                "dice": dice, "false_positive_ml": fp, "positive": positive,
                "false_positive_lesions": 1 if fp else 0}

    candidates = [[row(.1, 1., .6, 0., True), row(.2, .75, .7, 0., True)],
                  [row(.1, None, 0., 6., False), row(.2, None, 0., 2., False)]]
    baseline_rows = [[{"dice": .5, "lesion_recall": .5, "positive": True,
                       "false_positive_ml": 0.}],
                     [{"dice": 0., "lesion_recall": None, "positive": False,
                       "false_positive_ml": 1.}]]
    selected, table, status = choose_32_parameters(candidates, baseline_rows, baseline, cfg)
    assert selected["tumor_threshold"] == .2
    assert not table[0]["eligible"] and table[1]["eligible"]
    assert status["status"] == "selected_candidate"


def test_32_calibration_keeps_baseline_without_negative_validation_case():
    cfg = Config(max_negative_fp_ml=5.)
    baseline = {"tumor_threshold": 0.3, "min_tumor_ml": 0.2,
                "tumor_liver_margin_mm": 5.0}
    candidate = {"tumor_threshold": .1, "min_tumor_ml": 0.,
                 "tumor_liver_margin_mm": 10., "lesion_recall": 1.,
                 "dice": .7, "false_positive_ml": 0., "positive": True,
                 "false_positive_lesions": 0}
    base_row = {"dice": .5, "lesion_recall": .5, "positive": True,
                "false_positive_ml": 0.}
    selected, _, status = choose_32_parameters([[candidate]], [[base_row]], baseline, cfg)
    assert selected == baseline
    assert status["status"] == "kept_baseline_no_negative_validation"


def test_clean_to_reconstructed_transition_keeps_full_segmentation_weight():
    cfg = Config(image_size=32, context_slices=1, reconstruction_weight=0.,
                 seg_pretrain_clean=True, warmup_epochs=20, seg_pretrain_mix_epochs=10)
    assert reconstructed_input_fraction(cfg, 9) == 0.
    assert 0 < reconstructed_input_fraction(cfg, 10) < 1.
    assert reconstructed_input_fraction(cfg, 19) == 1.
    assert reconstructed_input_fraction(cfg, 20) == 1.
    restored = torch.zeros(1, 1, 32, 32)
    logits = torch.zeros(1, 2, 32, 32, requires_grad=True)
    batch = {"input": restored, "target": restored,
             "mask": torch.zeros(1, 2, 32, 32)}
    for epoch in (19, 20):
        _, parts = joint_loss(restored, logits, batch, cfg, epoch)
        assert parts["seg_ramp"] == 1.0


def test_finetune_source_rejects_other_fold_and_loads_same_fold(tmp_path):
    source_cfg = Config(image_size=32, context_slices=1, base_channels=4).validate()
    source_model = JointUNet(source_cfg)
    path = tmp_path / "source.pt"
    torch.save({"training_protocol": "v4", "config": config_dict(source_cfg),
                "model": source_model.state_dict(), "epoch": 3,
                "split_fingerprint": "fold_0", "audit_fingerprint": "audit"}, path)
    cfg = replace(source_cfg, view_weights=(.7, .15, .15), reconstruction_weight=0.,
                  seg_pretrain_clean=False, warmup_epochs=0, ramp_epochs=0,
                  seg_finetune_checkpoint=str(path), freeze_reconstructor=True).validate()
    target = JointUNet(cfg)
    with pytest.raises(ValueError, match="患者划分"):
        load_finetune_source(path, cfg, target, "fold_1", "audit")
    source = load_finetune_source(path, cfg, target, "fold_0", "audit")
    assert source["source_epoch"] == 4
    assert all(torch.equal(target.state_dict()[k], v) for k, v in source_model.state_dict().items())


def _positive_row(threshold, recall, dice, fp):
    return {"tumor_threshold": threshold, "min_tumor_ml": 0., "tumor_liver_margin_mm": 10.,
            "lesion_recall": recall, "dice": dice, "false_positive_ml": fp, "positive": True,
            "false_positive_lesions": 1 if fp else 0}


def test_32_calibration_all_patient_fp_fallback_without_negative_case():
    cfg = Config(fp_guard_fallback="all_patients", max_fp_increase_ml=5.)
    baseline = {"tumor_threshold": 0.3, "min_tumor_ml": 0.2, "tumor_liver_margin_mm": 5.0}
    base_rows = [[{"dice": .5, "lesion_recall": .5, "positive": True, "false_positive_ml": 2.}]]
    # 0.10：召回最高但真值外体积 +10 mL，超限；0.20：召回提高且只 +3 mL，应被选中。
    candidates = [[_positive_row(.1, 1., .6, 12.), _positive_row(.2, .75, .55, 5.)]]
    selected, table, status = choose_32_parameters(candidates, base_rows, baseline, cfg)
    assert selected["tumor_threshold"] == .2
    assert status["fp_guard"] == "all_patients" and status["status"] == "selected_candidate"
    assert not table[0]["eligible"] and table[1]["eligible"]


def test_selection_recall_is_averaged_over_32_view_tuning_thresholds():
    cfg = Config(postprocess=True, tumor_liver_margin_mm=None, selection_32_lesion_priority=True,
                 tune_32_thresholds=(0.1, 0.2, 0.3, 0.4))
    probability = np.zeros((3, 2, 24, 24), dtype=np.float32)
    probability[:, 0] = 0.9
    probability[1, 1, 4:7, 4:7] = 0.8    # 高置信病灶
    probability[1, 1, 15:18, 15:18] = 0.25  # 只在低阈值下检出的小病灶
    truth = np.zeros_like(probability)
    truth[:, 0] = 1
    truth[1, 1, 4:7, 4:7] = truth[1, 1, 15:18, 15:18] = 1
    result = segmentation_metrics(probability, truth, (1., 1., 1.), cfg, None, surface=False, view=32)
    assert result["tumor_lesion_recall"] == 0.5          # 默认阈值 0.5
    assert result["tumor_lesion_recall_tuned_range"] == pytest.approx((1 + 1 + .5 + .5) / 4)
    assert "tumor_lesion_recall_tuned_range" not in segmentation_metrics(
        probability, truth, (1., 1., 1.), cfg, None, surface=False, view=64)
