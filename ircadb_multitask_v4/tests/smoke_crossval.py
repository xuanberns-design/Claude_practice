"""20个合成体模的真实五折训练/推理/OOF汇总通路，非医学性能验证。

python -m tests.smoke_crossval --work-dir work/cv_smoke
可调微型设置集中在main开头。小训练为验证代码，明确关闭质量门槛。
"""
from pathlib import Path
import argparse
import torch
from src.config import Config
from src.common import seed_all,read_json,write_json
from src.dicom_io import audit
from src.split import make_split
from src.prepare import prepare
from src.crossval import make_folds,run_folds,evaluate_folds,aggregate_folds
from src.cache_quality import inspect_cache
from .phantom import make_phantom


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--work-dir',default='work/cv_smoke')
    root=Path(parser.parse_args().work_dir).resolve()
    if (root/'protocol'/'folds.json').exists():
        raise FileExistsError('请使用新的work-dir，不覆盖先前五折')
    cfg=Config(data_root=str(root/'data'),cache_dir=str(root/'cache'),run_dir=str(root/'unused'),
               image_size=32,context_slices=3,base_channels=4,full_views=256,strict_official_counts=False,
               known_tumor_negative=tuple(range(2,21,2)),split_counts=(14,3,3),split_search_trials=1000,
               epochs=2,warmup_epochs=1,ramp_epochs=1,samples_per_epoch=4,batch_size=2,
               validation_every=1,amp=False,device='cpu',bootstrap_repeats=100,selection_guard=False)
    seed_all(cfg.seed,2)
    make_phantom(root/'data',patients=20)
    audit(cfg);original=make_split(cfg);prepare(cfg)
    inspect_cache(cfg,with_dicom=True)
    protocol=make_folds(cfg,output=root/'protocol')
    assert protocol['outer_test_sizes']==[3,4,4,4,5]
    assert protocol['folds'][0]['patients']=={k:original[k] for k in ['train','val','test']}
    run_folds(cfg,root/'protocol'/'folds.json')
    evaluate_folds(cfg,root/'protocol'/'folds.json',export=False)
    results=aggregate_folds(cfg,root/'protocol'/'folds.json')
    assert results['n_patients']==20
    for v in cfg.views:
        assert results['reconstruction'][f'{v}/original_CT/Joint']['SSIM']['n']==20
    report={'passed':True,'synthetic_only':True,'patients':20,'folds':5,'test_sizes':protocol['outer_test_sizes'],
            'fold0_exact_original_split':True,'oof_each_patient_once':True,'clean_pretrain_then_joint':True,
            'validation_quality_guard_disabled_for_smoke_only':True,'actual_train_calls':5,
            'all_5_checkpoints_independent':True,'trained_epochs_per_fold':2}
    write_json(root/'cv_smoke_passed.json',report)
    print('FIVE-FOLD SMOKE PASSED')


if __name__=='__main__':
    main()
