"""CPU端到端自检：合成DICOM→审计→划分→投影→训练→恢复→评估→推理。

运行：python -m tests.smoke_pipeline --work-dir work/smoke
      python -m tests.smoke_pipeline --work-dir work/smoke_v4 --mode dual   # V4 双域+后处理+EMA
此脚本的所有指标仅验证程序通路，不代表医学性能。
"""
import argparse
from dataclasses import replace
from pathlib import Path
import torch
from src.config import Config
from src.common import write_json, seed_all
from src.dicom_io import audit
from src.split import make_split
from src.prepare import prepare
from src.train import train
from src.evaluate import evaluate, save_nifti
from src.dataset import PatientCache
from src.projection import denormalize
from src.infer import infer_nifti
from .phantom import make_phantom


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work-dir', default='work/smoke')
    parser.add_argument('--mode', choices=['image', 'dual'], default='image')
    args = parser.parse_args()
    root = Path(args.work_dir).resolve()
    if (root/'run'/'last.pt').exists():
        raise FileExistsError('请选择一个新的测试目录，避免覆盖旧结果')
    cfg = Config(data_root=str(root/'data'), cache_dir=str(root/'cache'), run_dir=str(root/'run'),
                 image_size=32, context_slices=3, base_channels=4, full_views=256,
                 strict_official_counts=False, known_tumor_negative=(2,4,6),
                 split_counts=(2,2,2), split_search_trials=500, batch_size=2,
                 samples_per_epoch=4, epochs=2, warmup_epochs=0, ramp_epochs=1,
                 validation_every=1, save_every_epochs=1, device='cpu', amp=False, bootstrap_repeats=100,
                 selection_guard=False)
    if args.mode == 'dual':
        cfg = replace(cfg, reconstruction_mode='dual_domain', sino_dense_views=128, sino_base_channels=4,
                      seg_backbone='resattn', seg_deep_supervision_weight=0.3, tumor_tversky_weight=0.5,
                      seg_hierarchy_weight=0.1, seg_wide_window=(-500., 500.), sino_weight=0.5,
                      dd_image_weight=0.3, fft_weight=0.05, projection_consistency_weight=0.05,
                      projection_consistency_angles=4, postprocess=True, auto_tune_postprocess=True,
                      ema_decay=0.9, lr_warmup_epochs=1, seg_tta=True).validate()
    seed_all(cfg.seed, 2)
    make_phantom(root/'data')
    audit(cfg)
    split = make_split(cfg)
    if args.mode == 'dual':
        # 先按 V3 建缓存，再以双域配置补充正弦图：验证旧缓存就地升级路径与FBP一致性复核。
        prepare(replace(cfg, reconstruction_mode='image'))
    prepare(cfg)
    checkpoint = train(cfg)
    expected = torch.load(root/'run'/'last.pt', map_location='cpu', weights_only=True)['model']
    # 从第一轮checkpoint恢复，实际再完成第二轮，验证optimizer/scheduler恢复路径。
    train(cfg, str(root/'run'/'epoch_0001.pt'))
    resumed = torch.load(root/'run'/'last.pt', map_location='cpu', weights_only=True)['model']
    max_diff = max(float((expected[k]-resumed[k]).abs().max()) for k in expected)
    assert max_diff < 1e-7, f'恢复训练与连续训练不同: {max_diff}'
    results = evaluate(cfg, checkpoint, export=True)
    patient = PatientCache(cfg, split['test'][0])
    save_nifti(root/'input_fbp.nii.gz', denormalize(patient.get('fbp_32'), cfg), patient.meta)
    if args.mode == 'dual':
        import numpy as np
        from src.ct_ops import mask_unmeasured
        sino = np.stack([mask_unmeasured(patient.get('sino')[z], cfg.sino_dense_views, 32)[::cfg.sino_dense_views // 32]
                         for z in range(patient.meta['n_slices'])])
        np.save(root/'input_sino32.npy', sino)
        infer_nifti(cfg, checkpoint, root/'input_fbp.nii.gz', 32, root/'inference', root/'input_sino32.npy')
        infer_nifti(cfg, checkpoint, root/'input_fbp.nii.gz', 32, root/'inference_reproject', reproject=True)
        assert (root/'postprocess.json').exists() or (root/'run'/'postprocess.json').exists()
    else:
        infer_nifti(cfg, checkpoint, root/'input_fbp.nii.gz', 32, root/'inference')
    write_json(root/'smoke_passed.json', {'passed': True, 'type': 'synthetic_pipeline_only',
                                         'resume_max_parameter_difference': max_diff,
                                         'mode': args.mode,
                                         'test_patients': split['test'], 'views': list(cfg.views),
                                         'outputs': ['training', 'checkpoint_resume', 'evaluation', 'nifti_inference']})
    print('SMOKE PIPELINE PASSED')


if __name__ == '__main__':
    main()
