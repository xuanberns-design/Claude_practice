"""统一命令入口。默认参数集中于 config.py；各命令 --help 可查。"""
import argparse
from pathlib import Path
import json
from .config import load_config
from .common import seed_all, write_json


def main():
    parser = argparse.ArgumentParser(description="3D-IRCADb 2.5D重建-分割联合实验")
    parser.add_argument("--config", help="YAML配置；置于子命令之前")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit", help="读取DICOM，审计实际切片数、几何和mask")
    split = sub.add_parser("split", help="按患者级冻结划分；可显式保留旧患者名单并更新标签审计指纹")
    split.add_argument("--base-split", help="旧实验真实splits.json；仅迁移患者ID/顺序，写入新cache_dir")
    reference = sub.add_parser("reference-split", help="仅依据官网表生成示例，不能直接用于训练")
    reference.add_argument("--output", default="docs/reference_split.json")
    prepare = sub.add_parser("prepare", help="生成三档稀疏FBP和全视角FBP缓存")
    prepare.add_argument("--patients", nargs="+", help="可只缓存指定病例，如3Dircadb1.18")
    train = sub.add_parser("train", help="训练、只用验证集选模")
    train.add_argument("--resume", help="从可信last.pt恢复")
    evaluate = sub.add_parser("evaluate", help="整患者评估并导出NIfTI")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--split", choices=["val", "test"], default="test")
    evaluate.add_argument("--no-export", action="store_true")
    infer = sub.add_parser("infer", help="对无需mask的稀疏FBP HU NIfTI推理")
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--input", required=True)
    infer.add_argument("--views", type=int, required=True)
    infer.add_argument("--output", default="runs/inference")
    infer.add_argument("--sinogram", help="dual_domain 模型的稀疏正弦图 npy [Z, views, D]")
    infer.add_argument("--reproject-fbp", action="store_true", help="无正弦图时用FBP重投影近似（质量下降）")
    tune = sub.add_parser("tune-postprocess", help="仅在验证集上选择肿瘤阈值/最小体积，写入 run_dir/postprocess.json")
    tune.add_argument("--checkpoint", required=True)
    quality = sub.add_parser("cache-audit",help="核查现有缓存HU分布与可选DICOM padding")
    quality.add_argument("--with-dicom",action="store_true")
    quality.add_argument("--output")
    cv_init = sub.add_parser("cv-init",help="保留原始划分为fold_0，生成五折协议")
    cv_init.add_argument("--base-split")
    cv_init.add_argument("--output")
    cv_train = sub.add_parser("cv-train",help="各折从头训练，不访问外层测试")
    cv_train.add_argument("--folds",required=True)
    cv_train.add_argument("--only",nargs="+",type=int)
    cv_eval = sub.add_parser("cv-evaluate",help="各折仅推理本折外层测试患者")
    cv_eval.add_argument("--folds",required=True)
    cv_eval.add_argument("--only",nargs="+",type=int)
    cv_eval.add_argument("--checkpoint-name",default="best.pt")
    cv_eval.add_argument("--no-export",action="store_true")
    cv_summary = sub.add_parser("cv-aggregate",help="核验五折来源并汇总折外患者结果")
    cv_summary.add_argument("--folds",required=True)
    cv_summary.add_argument("--output")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_all(cfg.seed, cfg.cpu_threads)
    if args.command == "audit":
        from .dicom_io import audit
        result = audit(cfg)
        print(f"已审计 {len(result['patients'])} 名患者")
    elif args.command == "split":
        from .split import make_split
        print(json.dumps(make_split(cfg, args.base_split), ensure_ascii=False, indent=2))
    elif args.command == "reference-split":
        from .official import reference_rows, OFFICIAL_URL
        from .split import balanced_split
        result = balanced_split(reference_rows(), cfg)
        result.update({"reference_only": True, "source": OFFICIAL_URL, "patients": reference_rows()})
        write_json(args.output, result)
        print(json.dumps(result["statistics"], indent=2))
    elif args.command == "prepare":
        from .prepare import prepare
        prepare(cfg, args.patients)
    elif args.command == "train":
        from .train import train
        print(train(cfg, args.resume))
    elif args.command == "evaluate":
        from .evaluate import evaluate
        evaluate(cfg, args.checkpoint, args.split, not args.no_export)
    elif args.command == "infer":
        from .infer import infer_nifti
        infer_nifti(cfg, args.checkpoint, args.input, args.views, args.output, args.sinogram, args.reproject_fbp)
    elif args.command == "tune-postprocess":
        from .evaluate import tune_postprocess
        tune_postprocess(cfg, args.checkpoint, "val")
    elif args.command == "cache-audit":
        from .cache_quality import inspect_cache
        inspect_cache(cfg,args.output,args.with_dicom)
    elif args.command == "cv-init":
        from .crossval import make_folds
        result = make_folds(cfg,args.base_split,args.output)
        print("已冻结五折协议，外层测试大小:",result["outer_test_sizes"])
    elif args.command == "cv-train":
        from .crossval import run_folds
        run_folds(cfg,args.folds,args.only)
    elif args.command == "cv-evaluate":
        from .crossval import evaluate_folds
        evaluate_folds(cfg,args.folds,args.only,args.checkpoint_name,not args.no_export)
    elif args.command == "cv-aggregate":
        from .crossval import aggregate_folds
        aggregate_folds(cfg,args.folds,args.output)


if __name__ == "__main__":
    main()
