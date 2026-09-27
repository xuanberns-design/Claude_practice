"""The delivery archive must refuse incomplete or stale patient results."""
import csv
import json
import zipfile

import pytest

from scripts.package_delivery import RUNS, VIEWS, build, sha256


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _csv(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)


def _mock_run(project, name):
    run = project / "runs" / name
    run.mkdir(parents=True)
    checkpoint = run / "best.pt"
    checkpoint.write_bytes((name + " checkpoint").encode())
    digest = sha256(checkpoint)
    params = {"tumor_threshold": .2, "min_tumor_ml": 0., "tumor_liver_margin_mm": 10.}
    _json(run / "postprocess.json", {"split": "val", "checkpoint_sha256": digest,
                                    "parameters": {"per_view": {"32": params}}})
    for split in (("val", "test") if name == RUNS[0] else ("test",)):
        output = run / f"evaluation_{split}"
        _json(output / "provenance.json", {"split": split, "checkpoint_sha256": digest,
                                            "raw_probabilities_exported": True, "patients": ["P"],
                                            "postprocess": {"parameters": {"per_view": {"32": params}}}})
        _json(output / "summary.json", {"test": True})
        _csv(output / "reconstruction_per_patient.csv", [{"patient": "P", "view": 32}])
        _csv(output / "segmentation_per_patient.csv",
             [{"patient": "P", "view": view} for view in VIEWS])
        _csv(output / "postprocess_diagnostics_32.csv", [{"patient": "P", "view": 32}])
        _csv(output / "postprocess_lesion_diagnostics_32.csv", [{"patient": "P", "view": 32}])
        for view in VIEWS:
            folder = output / "volumes" / "P" / str(view)
            folder.mkdir(parents=True)
            for filename in ("liver_probability.nii.gz", "tumor_probability.nii.gz"):
                (folder / filename).write_bytes(b"NIFTI")
            if split == "test":
                for filename in ("restored_hu.nii.gz", "liver.nii.gz", "tumor.nii.gz",
                                 "comparison.png"):
                    (folder / filename).write_bytes(b"RESULT")
                _json(folder / "visualization_manifest.json", {"n_slices": 1})
                slice_folder = folder / "slices" / "slice_0000"
                slice_folder.mkdir(parents=True)
                (slice_folder / "comparison.png").write_bytes(b"SLICE")
            if view == 32:
                for filename in ("tumor_before_liver_gate.nii.gz", "tumor_after_liver_gate.nii.gz"):
                    (folder / filename).write_bytes(b"MASK")


def test_complete_delivery_archive_and_missing_probability_guard(tmp_path):
    project = tmp_path / "project"
    for name in RUNS:
        _mock_run(project, name)
    output = tmp_path / "deliverable.zip"
    result = build(project, output)
    assert result["files"] > 20
    with zipfile.ZipFile(output) as archive:
        manifest = json.loads(archive.read("ircadb_multitask_v4/DELIVERY_MANIFEST.json"))
        assert set(manifest["evidence"]["checkpoints_sha256"]) == set(RUNS)
        assert archive.testzip() is None
    missing = project / "runs" / RUNS[0] / "evaluation_test" / "volumes" / "P" / "32" / "tumor_probability.nii.gz"
    missing.unlink()
    with pytest.raises(ValueError, match="tumor_probability.nii.gz"):
        build(project, tmp_path / "should_not_exist.zip")
    assert not (tmp_path / "should_not_exist.zip").exists()
