"""旧缓存诊断与DICOM padding有效区：原数据不可被审计改写。"""
from dataclasses import replace
import hashlib
import json
import numpy as np
import pydicom
import pytest
from pydicom.pixels import apply_modality_lut
from pydicom.uid import generate_uid
from src.cache_quality import inspect_cache, array_sha256
from src.common import read_json, write_json
from src.config import Config, prepare_signature
from src.dicom_io import geometry
from .phantom import write_slice


def _make_cache(tmp_path, padding=True, ranged=False, reverse_range=False):
    cfg = Config(cache_dir=str(tmp_path / "cache"), image_size=32, strict_official_counts=False)
    root = tmp_path / "source"
    pid = "3Dircadb1.1"
    series = generate_uid()
    raw = np.full((32, 32), 1000, dtype=np.uint16)
    raw[0, :5] = [0, 1, 2, 3, 4095]
    refs, headers, native = [], [], []
    for z in range(2):
        path = root / f"slice_{z}.dcm"
        write_slice(path, raw, z * 2.5, series, z + 1)
        d = pydicom.dcmread(path)
        # 特意让stored 0 -> -2000HU，stored 1 -> -1998HU；二者不能由HU阈值一起删除。
        d.RescaleSlope, d.RescaleIntercept = 2.0, -2000.0
        d.PatientName, d.PatientID = "DO_NOT_EXPORT_PERSONAL_NAME", "PRIVATE_ID"
        if padding:
            d.PixelPaddingValue = 0
            d["PixelPaddingValue"].VR = "US"
            if ranged:
                d.PixelPaddingValue = 2 if reverse_range else 0
                d.PixelPaddingRangeLimit = 0 if reverse_range else 2
                d["PixelPaddingRangeLimit"].VR = "US"
        d.save_as(path, enforce_file_format=True)
        refs.append(str(path))
        headers.append((str(path), str(z), pydicom.dcmread(path, stop_before_pixels=True)))
        native.append(apply_modality_lut(raw, d).astype(np.float32))
    _, geom = geometry(headers, cfg)
    native = np.stack(native)
    ct_hash = array_sha256(native)
    folder = tmp_path / "cache" / pid
    folder.mkdir(parents=True)
    meta = {"id": pid, "ct_sha256": ct_hash, "ct_refs": refs, **geom, "n_slices": 2,
            "processed_shape_zyx": list(native.shape), "audit_fingerprint": "original-audit-fingerprint",
            "prepare_signature": prepare_signature(cfg), "PatientName": "NEVER_COPY_META_FIELDS"}
    write_json(folder / "meta.json", meta)
    write_json(folder.parent / "audit.json", {"fingerprint": meta["audit_fingerprint"],
                                             "patients": [{"id": pid, "ct_sha256": ct_hash}]})
    np.save(folder / "native_hu.npy", native)
    norm = np.clip((native - cfg.hu_min) / (cfg.hu_max - cfg.hu_min), 0, 1)
    for name in ["target", "full_fbp", "fbp_32", "fbp_64", "fbp_128"]:
        np.save(folder / f"{name}.npy", norm)
    for name in ["masks", "native_masks"]:
        np.save(folder / f"{name}.npy", np.zeros((2, 2, 32, 32), dtype=np.uint8))
    write_json(folder.parent / "splits.json", {"fingerprint": "preserve-original-split"})
    return cfg, folder, refs, native


def _file_hashes(folder):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.rglob("*") if p.is_file()}


def test_default_cache_audit_needs_no_source_and_never_changes_original_arrays(tmp_path):
    cfg, folder, refs, native = _make_cache(tmp_path)
    for ref in refs:
        from pathlib import Path
        Path(ref).unlink()
    original = _file_hashes(folder.parent)
    report = inspect_cache(cfg, output=tmp_path / "extra_report")
    row = report["patients"][0]
    assert row["cache_consistency"]["matches_config"]
    assert row["paddingmask_sha256"] is None
    assert row["padding"]["source"] == "not_checked"
    assert not (folder / "native_valid.npy").exists()
    assert row["native_hu"]["min"] == -2000.0
    assert row["native_hu"]["max"] == 6190.0
    expected = np.abs(native.astype(np.float64) - np.clip(native, cfg.hu_min, cfg.hu_max)).mean()
    assert row["native_hu"]["range_clip_mae_lower_bound_hu"] == pytest.approx(expected)
    assert row["native_hu"]["fraction_below_minus_1024"] == 4 / 1024
    assert row["native_hu"]["fraction_above_hu_max"] == 1 / 1024
    assert row["native_hu"]["body_fraction"] == 1020 / 1024
    assert (tmp_path / "extra_report" / "quality_audit.json").is_file()
    for path, digest in original.items():
        from pathlib import Path
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest


def test_padding_matched_before_rescale_and_nonpadding_low_hu_preserved(tmp_path):
    cfg, folder, refs, native = _make_cache(tmp_path)
    original = _file_hashes(folder.parent)
    report = inspect_cache(cfg, with_dicom=True)
    valid = np.load(folder / "native_valid.npy")
    assert valid.dtype == np.dtype(bool)
    assert not valid[:, 0, 0].any()  # raw=0，HU=-2000：明确padding
    assert valid[:, 0, 1].all()      # raw=1，HU=-1998：无padding标签，必须保留
    assert valid[:, 1, 1].all()      # raw=1000，HU=0：不得误按HU==padding_value排除
    row = report["patients"][0]
    assert row["padding"]["padding_voxels"] == 2
    assert row["paddingmask_sha256"] == hashlib.sha256(valid.tobytes()).hexdigest()
    assert row["native_ct_sha256"] == hashlib.sha256(native.tobytes()).hexdigest()
    assert row["valid_native_hu"]["voxel_count"] == native.size - 2
    serialized = json.dumps(report)
    assert "DO_NOT_EXPORT" not in serialized and "PRIVATE_ID" not in serialized
    assert "PatientName" not in serialized and "ct_refs" not in serialized
    for path, digest in original.items():
        from pathlib import Path
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    # 同一sidecar允许复用；没有原始DICOM也可校验已存hash并保留其证明。
    report2 = inspect_cache(cfg, with_dicom=True)
    assert report2["patients"][0]["paddingmask_sha256"] == row["paddingmask_sha256"]
    report3 = inspect_cache(cfg)
    assert report3["patients"][0]["padding"] == row["padding"]


@pytest.mark.parametrize("reverse", [False, True])
def test_padding_range_is_inclusive_in_both_directions(tmp_path, reverse):
    cfg, folder, _, _ = _make_cache(tmp_path, ranged=True, reverse_range=reverse)
    report = inspect_cache(cfg, with_dicom=True)
    valid = np.load(folder / "native_valid.npy")
    assert not valid[:, 0, :3].any()
    assert valid[:, 0, 3:].all()
    assert report["patients"][0]["padding"]["padding_voxels"] == 6


def test_absent_padding_tags_means_all_valid_even_for_extreme_hu(tmp_path):
    cfg, folder, _, native = _make_cache(tmp_path, padding=False)
    row = inspect_cache(cfg, with_dicom=True)["patients"][0]
    assert np.load(folder / "native_valid.npy").all()
    assert row["padding"]["tagged_slices"] == 0
    assert row["padding"]["missing_tag_slices"] == len(native)
    assert row["padding"]["source"] == "dicom_no_padding_tags_all_valid"


def test_source_pixel_mismatch_is_rejected_without_mask(tmp_path):
    cfg, folder, refs, _ = _make_cache(tmp_path)
    d = pydicom.dcmread(refs[0])
    changed = d.pixel_array.copy()
    changed[5, 5] += 1
    d.PixelData = changed.tobytes()
    d.save_as(refs[0], enforce_file_format=True)
    with pytest.raises(ValueError, match="原DICOM内容与native_hu不一致"):
        inspect_cache(cfg, with_dicom=True)
    assert not (folder / "native_valid.npy").exists()


def test_source_geometry_mismatch_is_rejected_even_if_pixels_equal(tmp_path):
    cfg, folder, refs, _ = _make_cache(tmp_path)
    for ref in refs:
        d = pydicom.dcmread(ref)
        d.ImagePositionPatient = [1.0, 0.0, float(d.ImagePositionPatient[2])]
        d.save_as(ref, enforce_file_format=True)
    with pytest.raises(ValueError, match="原DICOM几何与缓存meta不一致"):
        inspect_cache(cfg, with_dicom=True)
    assert not (folder / "native_valid.npy").exists()


def test_native_hash_mismatch_is_rejected(tmp_path):
    cfg, folder, _, native = _make_cache(tmp_path)
    modified = native.copy()
    modified[0, 0, 0] += 1
    np.save(folder / "native_hu.npy", modified)
    with pytest.raises(ValueError, match="native_hu内容/hash"):
        inspect_cache(cfg)


def test_different_valid_mask_cannot_be_overwritten(tmp_path):
    cfg, folder, _, native = _make_cache(tmp_path)
    np.save(folder / "native_valid.npy", np.ones_like(native, dtype=bool))
    with pytest.raises(FileExistsError, match="拒绝覆盖"):
        inspect_cache(cfg, with_dicom=True)
    assert np.load(folder / "native_valid.npy").all()


def test_unverified_or_tampered_sidecar_rejected_without_dicom(tmp_path):
    cfg, folder, _, native = _make_cache(tmp_path)
    np.save(folder / "native_valid.npy", np.ones_like(native, dtype=bool))
    with pytest.raises(ValueError, match="没有匹配的已核验审计"):
        inspect_cache(cfg)
    (folder / "native_valid.npy").unlink()
    inspect_cache(cfg, with_dicom=True)
    mask = np.load(folder / "native_valid.npy")
    mask[0, 3, 3] = False
    np.save(folder / "native_valid.npy", mask)
    with pytest.raises(ValueError, match="内容/hash"):
        inspect_cache(cfg)


def test_missing_original_dicom_is_explicit_error(tmp_path):
    cfg, _, refs, _ = _make_cache(tmp_path)
    from pathlib import Path
    Path(refs[0]).unlink()
    with pytest.raises(FileNotFoundError, match="原DICOM不可读取"):
        inspect_cache(cfg, with_dicom=True)


def test_matrix_mismatch_is_reported_not_hidden(tmp_path):
    cfg, _, _, _ = _make_cache(tmp_path)
    row = inspect_cache(replace(cfg, image_size=64))["patients"][0]
    check = row["cache_consistency"]
    assert not check["matches_config"]
    assert check["actual_shapes"]["target"] == [2, 32, 32]
    assert check["configured_processed_shape_zyx"] == [2, 64, 64]
    assert "array_shape_mismatch:target" in check["issues"]
    assert "prepare_signature_mismatch" in check["issues"]


def test_report_destination_cannot_overwrite_original_audit(tmp_path):
    cfg, folder, _, _ = _make_cache(tmp_path)
    original = (folder.parent / "audit.json").read_bytes()
    with pytest.raises(ValueError, match="不能覆盖"):
        inspect_cache(cfg, output=folder.parent / "audit.json")
    assert (folder.parent / "audit.json").read_bytes() == original
