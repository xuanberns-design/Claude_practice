"""针对错位、泄漏、指标口径和联合梯度的回归测试。"""
from dataclasses import replace
import numpy as np
import pytest
import torch
import pydicom
from src.config import Config
from src.dicom_io import load_patient, dicom_entries, entries, align_mask, geometry, audit
from src.dataset import context_indices
from src.split import balanced_split, validate_split
from src.official import reference_rows
from src.projection import simulate_slice
from src.model import JointUNet
from src.losses import segmentation_loss
from src.metrics import reconstruction_metrics, dice_score
from src.evaluate import save_nifti
from .phantom import make_phantom


def test_dicom_geometry_hu_and_binary_mask(tmp_path):
    root = make_phantom(tmp_path, patients=1)
    cfg = Config(strict_official_counts=False)
    hu, masks, meta = load_patient(root/'3Dircadb1.1', 1, cfg)
    assert hu.min() == -1000 and hu.max() == 101
    assert masks[:, 0].any() and masks[:, 1].any()
    assert meta['z_mm'] == [0, 2.5, 5, 7.5]
    assert set(np.unique(masks)) == {0, 1}
    assert meta['alignment_modes'] == ['physical_position']


def test_wrong_mask_position_rejected(tmp_path):
    make_phantom(tmp_path, patients=1)
    folder = tmp_path/'3Dircadb1.1'
    path = next((folder/'MASKS_DICOM'/'liver').iterdir())
    d = pydicom.dcmread(path)
    d.ImagePositionPatient = [0., 0., 1000.]
    d.save_as(path, enforce_file_format=True)
    with pytest.raises(ValueError, match='物理位置'):
        load_patient(folder, 1, Config(strict_official_counts=False))


def test_missing_tumor_is_not_silently_negative(tmp_path):
    make_phantom(tmp_path, patients=2)
    with pytest.raises(ValueError, match='预期有目标tumor'):
        load_patient(tmp_path/'3Dircadb1.2', 2, Config(strict_official_counts=False))


@pytest.mark.parametrize('folder_name', ['tumor', 'Tumor', 'livertumor',
                                         'livertumors', 'livertumor01', 'livertumor12'])
def test_target_tumor_folder_family_is_read_for_patient7(tmp_path, folder_name):
    root = make_phantom(tmp_path, patients=1)
    patient = root/'3Dircadb1.7'
    (root/'3Dircadb1.1').rename(patient)
    (patient/'MASKS_DICOM'/'livertumor01').rename(patient/'MASKS_DICOM'/folder_name)
    _, masks, meta = load_patient(patient, 7, Config(strict_official_counts=False))
    assert meta['has_tumor'] and masks[:, 1].sum() > 0
    assert meta['tumor_mask_names'] == [folder_name]
    assert meta['tumor_mask_voxels_by_name'][folder_name] == int(masks[:, 1].sum())


def test_patient7_missing_target_mask_fails_closed(tmp_path):
    root = make_phantom(tmp_path, patients=1)
    patient = root/'3Dircadb1.7'
    (root/'3Dircadb1.1').rename(patient)
    masks = patient/'MASKS_DICOM'
    (masks/'livertumor01').rename(masks/'adrenaltumor')
    with pytest.raises(ValueError, match='预期有目标tumor'):
        load_patient(patient, 7, Config(strict_official_counts=False))


def test_official_inventory_rejects_missing_one_of_seven_patient1_masks(tmp_path):
    root = make_phantom(tmp_path, patients=1)
    with pytest.raises(ValueError, match='官方完整清单'):
        load_patient(root/'3Dircadb1.1', 1,
                     Config(strict_official_counts=False,
                            strict_official_mask_inventory=True))


def test_audit_refuses_to_overwrite_old_label_fingerprint(tmp_path):
    from dataclasses import replace
    root = make_phantom(tmp_path/'data', patients=1)
    cache = tmp_path/'audit_cache'
    cfg = Config(data_root=str(root), cache_dir=str(cache), strict_official_counts=False)
    audit(cfg)
    saved = (cache/'audit.json').read_bytes()
    with pytest.raises(ValueError, match='新的 cache_dir'):
        audit(replace(cfg, tumor_pattern=r'(?i)^livertumor01$'))
    assert (cache/'audit.json').read_bytes() == saved


def test_official_zero_liver_tumor_case_may_be_true_target_negative(tmp_path):
    import shutil
    root = make_phantom(tmp_path, patients=1)
    patient = root/'3Dircadb1.5'
    (root/'3Dircadb1.1').rename(patient)
    shutil.rmtree(patient/'MASKS_DICOM'/'livertumor01')
    _, masks, meta = load_patient(patient, 5, Config(strict_official_counts=False))
    assert not meta['has_tumor'] and not masks[:, 1].any()


def test_official_count_mismatch_rejected(tmp_path):
    make_phantom(tmp_path, patients=1)
    with pytest.raises(ValueError, match='官网'):
        load_patient(tmp_path/'3Dircadb1.1', 1, Config())


def test_split_no_patient_leakage():
    rows = reference_rows()
    split = balanced_split(rows, Config(split_search_trials=1000))
    assert [len(split[k]) for k in ['train', 'val', 'test']] == [14, 3, 3]
    validate_split(split, [r['id'] for r in rows])
    split['test'][0] = split['train'][0]
    with pytest.raises(ValueError):
        validate_split(split, [r['id'] for r in rows])


def test_context_stays_in_patient_and_uses_mm():
    cfg = Config(context_slices=5, context_step_mm=2.5)
    assert context_indices(0, np.array([0., 2.5, 5.]), cfg).tolist() == [0, 0, 0, 1, 2]


def test_segmentation_gradient_reaches_reconstructor():
    torch.set_num_threads(2)
    cfg = Config(image_size=32, context_slices=3, base_channels=4)
    model = JointUNet(cfg)
    x = torch.rand(2, 3, 32, 32)
    r, logits = model(x, torch.tensor([.5, .6]))
    assert torch.allclose(r, x)
    segmentation_loss(logits, torch.zeros_like(logits), cfg).backward()
    assert model.reconstructor.head.weight.grad.abs().sum() > 0


def test_metrics_identity_and_empty_convention():
    x = np.zeros((2, 32, 32), dtype=np.float32)
    m = reconstruction_metrics(x, x, Config())
    assert m['SSIM'] == 1 and np.isinf(m['PSNR_dB'])
    assert m['MAE_HU'] == 0 and m['RMSE_HU'] == 0
    assert dice_score(x, x) == 1
    assert dice_score(np.ones_like(x), x) == 0
    m = reconstruction_metrics(x+10, x, Config())
    assert m['MAE_HU'] == 10 and m['RMSE_HU'] == 10


def test_projection_full_better_than_sparse():
    from skimage.data import shepp_logan_phantom
    from skimage.transform import resize
    cfg = Config(full_views=512, image_size=64)
    hu = resize(shepp_logan_phantom(), (64,64), preserve_range=True)*1000-1000
    full, sparse = simulate_slice(hu, .8, cfg, np.random.default_rng(1))
    assert np.mean((full-hu)**2) < np.mean((sparse[32]-hu)**2)
    assert np.isfinite(full).all()


def test_nifti_geometry_roundtrip(tmp_path):
    import nibabel as nib
    a = np.arange(2*3*4, dtype=np.float32).reshape(2,3,4)
    meta = {'affine_lps_xyz': [[.8,0,0,10],[0,.8,0,20],[0,0,2.5,30],[0,0,0,1]]}
    save_nifti(tmp_path/'test.nii.gz', a, meta)
    image = nib.load(tmp_path/'test.nii.gz')
    assert image.shape == (4,3,2)
    assert np.allclose(image.get_fdata().transpose(2,1,0), a)
    assert np.allclose(image.affine[:3,3], [-10,-20,30])


def test_adrenal_tumor_excluded(tmp_path):
    import shutil
    make_phantom(tmp_path, patients=1)
    masks = tmp_path/'3Dircadb1.1'/'MASKS_DICOM'
    shutil.copytree(masks/'liver', masks/'adrenaltumor')
    _, y, meta = load_patient(tmp_path/'3Dircadb1.1', 1, Config(strict_official_counts=False))
    assert meta['tumor_mask_names'] == ['livertumor01']
    assert y[:,1].sum() < y[:,0].sum()


def test_noise_simulation_reproducible():
    cfg = Config(image_size=32, full_views=256, photons_per_ray=10000, readout_noise_std=2)
    hu = np.zeros((32,32),dtype=np.float32)
    full, a = simulate_slice(hu,1,cfg,np.random.default_rng(42))
    _, b = simulate_slice(hu,1,cfg,np.random.default_rng(42))
    clean, c = simulate_slice(hu,1,replace(cfg,photons_per_ray=0,readout_noise_std=0),np.random.default_rng(42))
    assert np.array_equal(a[32],b[32])
    assert not np.array_equal(a[32],c[32])
    assert np.array_equal(full,clean)


def test_config_rejects_invalid_view_grid():
    with pytest.raises(ValueError, match='整数倍'):
        Config(full_views=1000).validate()


def test_stale_patient7_negative_config_rejected():
    with pytest.raises(ValueError, match='7号 tumor'):
        Config(known_tumor_negative=(5, 7, 11, 14, 20)).validate()


def test_metrics_patient_volume_not_mean_slice_dice():
    truth = np.zeros((2,16,16),dtype=np.uint8)
    truth[0,0,0] = 1
    truth[1] = 1
    pred = truth.copy()
    pred[0] = 0
    assert dice_score(pred,truth) == pytest.approx(512/513)
