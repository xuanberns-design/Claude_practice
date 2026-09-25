"""CPU端到端自检：合成DICOM→审计→划分→投影→训练→恢复→评估→推理。

运行：python -m tests.smoke_pipeline --work-dir work/smoke
此脚本的所有指标仅验证程序通路，不代表医学性能。
"""
import argparse
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
    seed_all(cfg.seed, 2)
    make_phantom(root/'data')
    audit(cfg)
    split = make_split(cfg)
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
    infer_nifti(cfg, checkpoint, root/'input_fbp.nii.gz', 32, root/'inference')
    write_json(root/'smoke_passed.json', {'passed': True, 'type': 'synthetic_pipeline_only',
                                         'resume_max_parameter_difference': max_diff,
                                         'test_patients': split['test'], 'views': list(cfg.views),
                                         'outputs': ['training', 'checkpoint_resume', 'evaluation', 'nifti_inference']})
    print('SMOKE PIPELINE PASSED')


if __name__ == '__main__':
    main()
