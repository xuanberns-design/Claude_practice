"""Verify both real-patient reruns and package the complete code/results bundle.

Usage (from the repository root):
    python -m scripts.package_delivery --output ../ircadb_32views_complete.zip \
        --last-checkpoint C:/path/to/last.pt

The archive intentionally excludes raw DICOM and the projection cache. It refuses
to package historical CSV/PNG as if they were newly rerun outputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path


VIEWS = (32, 64, 128)
RUNS = ("v4_dualdomain256", "v4_32_seg_finetune")
EXCLUDE_DIRS = {"__pycache__", ".pytest_cache", ".git"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_evaluation(run: Path, split: str, checkpoint_sha: str, tuned_params: dict,
                        problems: list[str]) -> dict | None:
    output = run / f"evaluation_{split}"
    provenance_path = output / "provenance.json"
    if not provenance_path.is_file():
        problems.append(f"缺少 {provenance_path}")
        return None
    provenance = read_json(provenance_path)
    if provenance.get("split") != split:
        problems.append(f"{provenance_path}: split 不是 {split}")
    if provenance.get("checkpoint_sha256") != checkpoint_sha:
        problems.append(f"{provenance_path}: checkpoint SHA256 不匹配")
    if provenance.get("raw_probabilities_exported") is not True:
        problems.append(f"{provenance_path}: 未重新导出原始概率体")
    if provenance.get("postprocess", {}).get("parameters", {}).get("per_view", {}).get("32") != tuned_params:
        problems.append(f"{provenance_path}: 未使用冻结的 32 views 参数")
    patients = provenance.get("patients")
    if not isinstance(patients, list) or not patients:
        problems.append(f"{provenance_path}: 缺少患者名单")
        return provenance
    for filename in ("summary.json", "reconstruction_per_patient.csv", "segmentation_per_patient.csv",
                     "postprocess_diagnostics_32.csv", "postprocess_lesion_diagnostics_32.csv"):
        if not (output / filename).is_file():
            problems.append(f"缺少 {output / filename}")
    segmentation_path = output / "segmentation_per_patient.csv"
    if segmentation_path.is_file():
        actual = [(r["patient"], int(r["view"])) for r in rows(segmentation_path)]
        expected = {(patient, view) for patient in patients for view in VIEWS}
        if len(actual) != len(expected) or set(actual) != expected:
            problems.append(f"{segmentation_path}: 患者/视角不完整或重复")
    for patient in patients:
        for view in VIEWS:
            folder = output / "volumes" / patient / str(view)
            for name in ("liver_probability.nii.gz", "tumor_probability.nii.gz"):
                if not (folder / name).is_file():
                    problems.append(f"缺少 {folder / name}")
            if split == "test":
                for name in ("restored_hu.nii.gz", "liver.nii.gz", "tumor.nii.gz",
                             "comparison.png", "visualization_manifest.json"):
                    if not (folder / name).is_file():
                        problems.append(f"缺少 {folder / name}")
                manifest_path = folder / "visualization_manifest.json"
                if manifest_path.is_file():
                    manifest = read_json(manifest_path)
                    slice_dir = folder / "slices"
                    actual_count = len(list(slice_dir.glob("slice_*/comparison.png")))
                    if actual_count != manifest.get("n_slices"):
                        problems.append(f"{folder}: 逐层 PNG 数量与 manifest 不符")
            if view == 32:
                for name in ("tumor_before_liver_gate.nii.gz", "tumor_after_liver_gate.nii.gz"):
                    if not (folder / name).is_file():
                        problems.append(f"缺少 {folder / name}")
    return provenance


def preflight(project: Path, problems: list[str]) -> dict:
    checkpoints = {}
    patients_by_run = {}
    for name in RUNS:
        run = project / "runs" / name
        checkpoint = run / "best.pt"
        tuning = run / "postprocess.json"
        if not checkpoint.is_file():
            problems.append(f"缺少 {checkpoint}")
            continue
        checkpoint_sha = sha256(checkpoint)
        checkpoints[name] = checkpoint_sha
        if not tuning.is_file():
            problems.append(f"缺少 {tuning}")
            continue
        calibrated = read_json(tuning)
        tuned_params = calibrated.get("parameters", {}).get("per_view", {}).get("32")
        if (calibrated.get("split") != "val" or calibrated.get("checkpoint_sha256") != checkpoint_sha
                or not isinstance(tuned_params, dict)):
            problems.append(f"{tuning}: 不是对应权重的验证集 32 views 校准结果")
            continue
        splits = ("val", "test") if name == RUNS[0] else ("test",)
        for split in splits:
            provenance = validate_evaluation(run, split, checkpoint_sha, tuned_params, problems)
            if provenance:
                patients_by_run[f"{name}/{split}"] = provenance.get("patients", [])
    return {"checkpoints_sha256": checkpoints, "patients_by_run": patients_by_run}


def archive_files(project: Path, last_checkpoint: Path | None):
    for path in sorted(project.rglob("*")):
        if not path.is_file() or any(part in EXCLUDE_DIRS for part in path.relative_to(project).parts):
            continue
        if path.suffix.lower() == ".zip":
            continue
        yield path, "ircadb_multitask_v4/" + path.relative_to(project).as_posix()
    if last_checkpoint is not None:
        yield last_checkpoint, "ircadb_multitask_v4/runs/v4_dualdomain256/last.pt"


def build(project: Path, output: Path, last_checkpoint: Path | None = None):
    project, output = project.resolve(), output.resolve()
    if output.is_relative_to(project):
        raise ValueError("压缩包必须写在项目目录外，避免把自身加入压缩包")
    if last_checkpoint is not None and not last_checkpoint.is_file():
        raise FileNotFoundError(last_checkpoint)
    problems: list[str] = []
    evidence = preflight(project, problems)
    if problems:
        raise ValueError("交付条件尚未满足：\n" + "\n".join(f"- {p}" for p in problems[:50]))
    entries = list(archive_files(project, last_checkpoint))
    names = [name for _, name in entries]
    if len(names) != len(set(names)):
        raise ValueError("压缩包路径重复，请检查 best/last 权重是否已在项目内")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    manifest = {"schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "description": "32 views 两版真实患者重跑结果；原始 DICOM 和投影缓存未打包",
                "evidence": evidence,
                "files": [{"path": name, "size": path.stat().st_size} for path, name in entries]}
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED,
                         allowZip64=True) as archive:
        for path, name in entries:
            archive.write(path, name)
        archive.writestr("ircadb_multitask_v4/DELIVERY_MANIFEST.json",
                         json.dumps(manifest, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(temporary) as archive:
        damaged = archive.testzip()
        if damaged:
            raise IOError(f"压缩包 CRC 校验失败: {damaged}")
    temporary.replace(output)
    return {"output": str(output), "bytes": output.stat().st_size,
            "sha256": sha256(output), "files": len(entries) + 1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--last-checkpoint", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.project, args.output, args.last_checkpoint), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
