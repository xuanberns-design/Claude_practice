"""五折协议、真实原划分保留、患者级汇总与来源校验的回归测试。"""
from dataclasses import replace
import importlib
from pathlib import Path

import pytest
import torch

from src.common import digest, read_json, write_csv, write_json
from src.config import Config, config_dict, load_config
from src.crossval import (make_folds, run_folds, evaluate_folds, aggregate_folds,
                          _load_protocol, EXPECTED_METRIC_PROTOCOL)
from src.split import make_split


@pytest.fixture
def experiment(tmp_path):
    cache = tmp_path / "cache"
    rows = [{"id": f"patient_{i}", "n_slices": 70+i*9, "has_tumor": i % 4 != 0,
             "tumor_voxels": i*100 if i % 4 else 0, "sex": "F" if i % 2 else "M"}
            for i in range(1, 21)]
    audit = {"patients": rows, "fingerprint": digest(rows)}
    split = {"test": ["patient_4", "patient_2", "patient_1"],
             "val": ["patient_8", "patient_5", "patient_3"]}
    split["train"] = [r["id"] for r in rows if r["id"] not in split["test"]+split["val"]]
    split["fingerprint"] = digest({k: split[k] for k in ("train", "val", "test")})
    split["audit_fingerprint"] = audit["fingerprint"]
    write_json(cache / "audit.json", audit)
    write_json(cache / "splits.json", split)
    cfg = Config(cache_dir=str(cache), run_dir=str(tmp_path / "original_run"),
                 split_search_trials=1000, bootstrap_repeats=10)
    return cfg, split, tmp_path / "cv"


def test_exact_original_preservation_full_coverage_and_read_only_cache(experiment):
    cfg, original, destination = experiment
    cache = Path(cfg.cache_dir)
    before = {p.name: p.read_bytes() for p in cache.iterdir()}
    protocol = make_folds(cfg, output=destination)
    assert protocol["outer_test_sizes"] == [3, 4, 4, 4, 5]
    assert protocol["folds"][0]["patients"] == {k: original[k] for k in ("train", "val", "test")}
    assert read_json(protocol["folds"][0]["split_path"]) == original
    ids = [p for f in protocol["folds"] for p in f["patients"]["test"]]
    assert len(ids) == len(set(ids)) == 20
    assert before == {p.name: p.read_bytes() for p in cache.iterdir()}
    for f in protocol["folds"]:
        split = f["patients"]
        assert len(split["val"]) == 3
        assert set(split["train"]).isdisjoint(split["test"]+split["val"])
        assert set(split["val"]).isdisjoint(split["test"])
        assert f["statistics"]["test"]["tumor_negative"] == 1
        fold_cfg = load_config(f["config_path"])
        assert Path(fold_cfg.cache_dir) == cache
        assert fold_cfg.seed == cfg.seed
        assert Path(fold_cfg.split_path).is_file()
    _load_protocol(cfg, destination / "folds.json")


def test_rebind_old_patient_lists_to_new_label_audit_then_cv_fold0(experiment, tmp_path):
    cfg, original, destination = experiment
    old_source = Path(cfg.cache_dir) / "splits.json"
    new_cache = tmp_path / "relabelled_cache"
    updated = read_json(Path(cfg.cache_dir) / "audit.json")
    updated["patients"][7]["has_tumor"] = True
    updated["patients"][7]["tumor_voxels"] = 12345
    updated["fingerprint"] = digest(updated["patients"])
    write_json(new_cache / "audit.json", updated)
    new_cfg = replace(cfg, cache_dir=str(new_cache))
    rebound = make_split(new_cfg, base_split=old_source)
    assert {k: rebound[k] for k in ("train", "val", "test")} == {
        k: original[k] for k in ("train", "val", "test")}
    assert rebound["fingerprint"] == original["fingerprint"]
    assert rebound["audit_fingerprint"] == updated["fingerprint"]
    assert rebound["rebound_from"]["source_audit_fingerprint"] == original["audit_fingerprint"]
    assert rebound["statistics"]["val"]["tumor_positive"] == 3
    protocol = make_folds(new_cfg, output=destination)
    assert protocol["folds"][0]["patients"] == {
        k: original[k] for k in ("train", "val", "test")}


def test_rebind_refuses_modified_old_patient_lists(experiment, tmp_path):
    cfg, old, _ = experiment
    old["test"][0], old["train"][0] = old["train"][0], old["test"][0]
    write_json(Path(cfg.cache_dir) / "splits.json", old)
    write_json(tmp_path / "fresh" / "audit.json", read_json(Path(cfg.cache_dir) / "audit.json"))
    with pytest.raises(ValueError, match="原始划分 fingerprint"):
        make_split(replace(cfg, cache_dir=str(tmp_path / "fresh")),
                   base_split=Path(cfg.cache_dir) / "splits.json")


def test_missing_actual_original_is_not_replaced_with_example(experiment):
    cfg, _, destination = experiment
    with pytest.raises(FileNotFoundError, match="实际原始"):
        make_folds(cfg, base_split=destination / "missing.json", output=destination)


def test_original_modified_without_fingerprint_rejected(experiment):
    cfg, original, destination = experiment
    original["train"][0], original["val"][0] = original["val"][0], original["train"][0]
    write_json(Path(cfg.cache_dir) / "splits.json", original)
    with pytest.raises(ValueError, match="fingerprint"):
        make_folds(cfg, output=destination)


def test_output_cannot_mutate_shared_cache(experiment):
    cfg, _, _ = experiment
    with pytest.raises(ValueError, match="缓存只读"):
        make_folds(cfg, output=Path(cfg.cache_dir) / "cv")


def test_existing_protocol_cannot_be_overwritten(experiment):
    cfg, _, destination = experiment
    make_folds(cfg, output=destination)
    with pytest.raises(FileExistsError, match="已存在"):
        make_folds(cfg, output=destination)


def test_insufficient_minority_is_reported(experiment):
    cfg, original, destination = experiment
    path = Path(cfg.cache_dir) / "audit.json"
    audit = read_json(path)
    for row in audit["patients"]:
        row["has_tumor"] = row["id"] not in ("patient_4", "patient_8")
    audit["fingerprint"] = digest(audit["patients"])
    original["audit_fingerprint"] = audit["fingerprint"]
    write_json(path, audit)
    write_json(Path(cfg.cache_dir) / "splits.json", original)
    with pytest.warns(UserWarning, match="少于5"):
        protocol = make_folds(cfg, output=destination)
    assert any("少于5" in x for x in protocol["warnings"])
    assert sum(f["statistics"]["test"]["tumor_negative"] for f in protocol["folds"]) == 2


def test_frozen_fold_file_tampering_rejected(experiment):
    cfg, _, destination = experiment
    protocol = make_folds(cfg, output=destination)
    path = protocol["folds"][2]["split_path"]
    split = read_json(path)
    split["train"][0], split["val"][0] = split["val"][0], split["train"][0]
    split["fingerprint"] = digest({k: split[k] for k in ("train", "val", "test")})
    write_json(path, split)
    with pytest.raises(ValueError, match="冻结五折协议"):
        _load_protocol(cfg, destination / "folds.json")


def test_frozen_config_tampering_rejected(experiment):
    import yaml
    cfg, _, destination = experiment
    protocol = make_folds(cfg, output=destination)
    path = Path(protocol["folds"][2]["config_path"])
    saved = yaml.safe_load(path.read_text(encoding="utf-8"))
    saved["lr"] *= 2
    path.write_text(yaml.safe_dump(saved), encoding="utf-8")
    with pytest.raises(ValueError, match="config.yaml 已改变"):
        _load_protocol(cfg, destination / "folds.json")


@pytest.mark.parametrize("parameter,value", [("lr", 0.0001), ("window_min", -100.0),
                                              ("segmentation_threshold", 0.6),
                                              ("inference_tta", True),
                                              ("selection_guard_references", ["original_CT", "full_FBP"])])
def test_cross_fold_hyperparameters_must_match_even_after_rehash(experiment, parameter, value):
    """A freshly rehashed collection of differently tuned folds is still not one CV run."""
    import yaml
    cfg, _, destination = experiment
    protocol = make_folds(cfg, output=destination)
    entry = protocol["folds"][2]
    path = Path(entry["config_path"])
    saved = yaml.safe_load(path.read_text(encoding="utf-8"))
    saved[parameter] = value
    path.write_text(yaml.safe_dump(saved), encoding="utf-8")
    entry["config_fingerprint"] = digest(saved)
    protocol["fingerprint"] = digest({k: v for k, v in protocol.items() if k != "fingerprint"})
    write_json(destination / "folds.json", protocol)
    with pytest.raises(ValueError, match="跨折训练/评估超参数必须一致"):
        _load_protocol(cfg, destination / "folds.json")


def test_training_is_from_scratch_and_does_not_test_by_default(experiment, monkeypatch):
    cfg, _, destination = experiment
    protocol = make_folds(cfg, output=destination)
    calls = []
    def fake_train(fold_cfg, resume):
        assert resume is None
        calls.append(fold_cfg)
        return str(Path(fold_cfg.run_dir) / "best.pt")
    monkeypatch.setattr(importlib.import_module("src.train"), "train", fake_train)
    monkeypatch.setattr(importlib.import_module("src.crossval"), "evaluate_folds",
                        lambda *args, **kwargs: pytest.fail("default train must not test"))
    result = run_folds(cfg, destination / "folds.json", folds=[0, 3])
    assert set(result) == {0, 3}
    assert len({c.run_dir for c in calls}) == 2
    assert [c.seed for c in calls] == [cfg.seed, cfg.seed]
    root = Path(protocol["folds"][0]["run_dir"])
    root.mkdir()
    (root / "last.pt").write_bytes(b"existing training")
    with pytest.raises(FileExistsError, match="已有checkpoint"):
        run_folds(cfg, destination / "folds.json", folds=[0])


def _mock_completed_evaluation(experiment, monkeypatch):
    cfg, _, destination = experiment
    protocol = make_folds(cfg, output=destination)
    for entry in protocol["folds"]:
        fold_cfg = load_config(entry["config_path"])
        Path(entry["run_dir"]).mkdir()
        torch.save({"epoch": 30, "split_fingerprint": entry["split_fingerprint"],
                    "audit_fingerprint": protocol["audit_fingerprint"],
                    "config": config_dict(fold_cfg)}, Path(entry["run_dir"]) / "best.pt")
    def fake_evaluate(fold_cfg, checkpoint, split, export):
        assert split == "test"
        partitions = read_json(fold_cfg.split_path)
        folder = Path(fold_cfg.run_dir) / "evaluation_test"
        rec, seg = [], []
        for pid in partitions["test"]:
            number = int(pid.split("_")[-1])
            for view in fold_cfg.views:
                for reference in ("original_CT", "full_FBP"):
                    for method in ("FBP", "Joint"):
                        rec.append({"patient": pid, "view": view, "reference": reference,
                                    "method": method, "SSIM": number/20, "PSNR_dB": number,
                                    "MAE_HU": number, "RMSE_HU": number,
                                    "liver_MAE_HU": number, "liver_RMSE_HU": number})
                seg.append({"patient": pid, "view": view, "liver_Dice": number/20,
                            "tumor_Dice": number/20, "tumor_positive": number % 4 != 0,
                            "tumor_false_positive_ml": 1.0, "tumor_ground_truth_ml": 1.0})
        write_csv(folder / "reconstruction_per_patient.csv", rec)
        write_csv(folder / "segmentation_per_patient.csv", seg)
        write_json(folder / "provenance.json", {"checkpoint": str(checkpoint), "epoch": 30,
                    "split": split, "patients": partitions[split], "config": config_dict(fold_cfg),
                    "audit_fingerprint": protocol["audit_fingerprint"],
                    "split_fingerprint": partitions["fingerprint"],
                    "metric_protocol": EXPECTED_METRIC_PROTOCOL})
        return {}
    monkeypatch.setattr(importlib.import_module("src.evaluate"), "evaluate", fake_evaluate)
    evaluate_folds(cfg, destination / "folds.json", export=False)
    return cfg, protocol, destination


def test_aggregate_patient_macro_with_unequal_fold_sizes(experiment, monkeypatch):
    cfg, protocol, destination = _mock_completed_evaluation(experiment, monkeypatch)
    summary = aggregate_folds(cfg, destination / "folds.json")
    assert summary["n_patients"] == 20
    stats = summary["reconstruction"]["32/original_CT/Joint"]["PSNR_dB"]
    assert stats["n"] == 20
    assert stats["mean"] == pytest.approx(10.5)
    assert summary["segmentation"]["32"]["tumor_Dice_positive_only"]["n"] == 15
    assert len({e["checkpoint_sha256"] for e in summary["fold_provenance"]}) == 5
    assert summary["metric_protocol"] == EXPECTED_METRIC_PROTOCOL
    assert (destination / "aggregate" / "segmentation_per_patient.csv").is_file()


def test_aggregate_rejects_cross_fold_checkpoint(experiment, monkeypatch):
    cfg, protocol, destination = _mock_completed_evaluation(experiment, monkeypatch)
    wrong = read_json(Path(protocol["folds"][0]["run_dir"]) / "evaluation_test" / "provenance.json")
    path = Path(protocol["folds"][1]["run_dir"]) / "evaluation_test" / "provenance.json"
    provenance = read_json(path)
    provenance["checkpoint"] = wrong["checkpoint"]
    write_json(path, provenance)
    with pytest.raises(ValueError, match="所属折独立"):
        aggregate_folds(cfg, destination / "folds.json")


def test_aggregate_rejects_edited_results(experiment, monkeypatch):
    cfg, protocol, destination = _mock_completed_evaluation(experiment, monkeypatch)
    path = Path(protocol["folds"][1]["run_dir"]) / "evaluation_test" / "reconstruction_per_patient.csv"
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="CSV在推理后被修改"):
        aggregate_folds(cfg, destination / "folds.json")


@pytest.mark.parametrize("metric_protocol", [None, "legacy_clipped_metrics"])
def test_aggregate_rejects_missing_or_different_metric_protocol(experiment, monkeypatch, metric_protocol):
    cfg, protocol, destination = _mock_completed_evaluation(experiment, monkeypatch)
    path = Path(protocol["folds"][1]["run_dir"]) / "evaluation_test" / "provenance.json"
    provenance = read_json(path)
    if metric_protocol is None:
        provenance.pop("metric_protocol")
    else:
        provenance["metric_protocol"] = metric_protocol
    write_json(path, provenance)
    with pytest.raises(ValueError, match="metric_protocol"):
        aggregate_folds(cfg, destination / "folds.json")


def test_aggregate_requires_every_fold(experiment, monkeypatch):
    cfg, protocol, destination = _mock_completed_evaluation(experiment, monkeypatch)
    path = Path(protocol["folds"][4]["run_dir"]) / "evaluation_test" / "provenance.json"
    path.unlink()
    with pytest.raises(FileNotFoundError):
        aggregate_folds(cfg, destination / "folds.json")


def test_test_evaluation_never_falls_back_to_candidate(experiment):
    cfg, _, destination = experiment
    protocol = make_folds(cfg, output=destination)
    root = Path(protocol["folds"][0]["run_dir"])
    root.mkdir()
    (root / "best_candidate.pt").write_bytes(b"not a qualified checkpoint")
    with pytest.raises(FileNotFoundError):
        evaluate_folds(cfg, destination / "folds.json", folds=[0])
