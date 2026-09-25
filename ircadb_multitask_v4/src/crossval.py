"""冻结患者级五折协议，保留真实旧划分，并汇总严格的折外预测。

参数沿用 Config 中 seed、split_search_trials 与 split_*_weight；K_FOLDS
位于本文件开头。make_folds 必须读取本地真实 splits.json，不猜测原患者名单。
fold_0 的 train/val/test 列表及顺序完全保留；其余患者均匀分到另四个外层
测试折。例如原测试集3人、总20人时测试大小为3/4/4/4/5，因此是五折但
不是严格等大的五折。每个其余折的验证人数与原划分一致，从该折非测试
患者中选取。切片数、阳性/阴性、log(1+肿瘤体素数)与性别用于基线平衡。

共享 cache_dir 只读，fold split/config/checkpoint/result 保存在独立目录。
保持缓存生成 seed 不变；每折调用一次全新 train，禁止用其他折权重续训。
默认训练不访问外层测试集。先冻结超参数、完成训练，再 evaluate_folds。
aggregate_folds 检查五折真实划分、checkpoint、统一训练/推理参数与指标
协议，以及推理出处后拼接患者级
指标；每名患者仅使用其所属外层测试折的模型，绝不平均五个模型对同一
患者的预测。此协议不是完整嵌套调参；在已看过旧测试集后修改模型，不能
把折0当作从未见过的独立确认集。若要临床结论还需独立外部验证。
"""
K_FOLDS = 5
SPLIT_NAMES = ("train", "val", "test")
# 必须与 evaluate.py 写入 provenance 的指标口径版本一致。
EXPECTED_METRIC_PROTOCOL = "rawHU_unclipped_global_v1_plus_explicit_regions_v2"
FOLD_SPECIFIC_CONFIG_KEYS = {"run_dir", "split_path", "split_counts"}

import copy
import csv
import hashlib
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml

from .common import digest, read_json, write_csv, write_json
from .config import config_dict, load_config
from .split import validate_split


def _parts(split):
    return {name: split[name] for name in SPLIT_NAMES}


def _common_hyperparameters(cfg):
    """Only paths/counts determined by a fold may differ; all experiment knobs freeze."""
    return {k: v for k, v in config_dict(cfg).items() if k not in FOLD_SPECIFIC_CONFIG_KEYS}


def _verify_metric_protocol(provenance):
    protocol = provenance.get("metric_protocol")
    if protocol != EXPECTED_METRIC_PROTOCOL:
        raise ValueError(f"metric_protocol 缺失或与当前指标口径不一致: {protocol!r}；请统一版本重新评估五折")
    return protocol


def _verify_split(split, rows, audit_fingerprint):
    validate_split(split, [r["id"] for r in rows])
    if split.get("fingerprint") != digest(_parts(split)):
        raise ValueError("划分 fingerprint 与实际患者名单不符")
    if split.get("audit_fingerprint") != audit_fingerprint:
        raise ValueError("划分 audit_fingerprint 与实际 audit.json 不符")


def _features(rows):
    return np.asarray([[r["n_slices"], int(bool(r["has_tumor"])),
                        int(not r["has_tumor"]), np.log1p(r["tumor_voxels"]),
                        float(r.get("sex") == "F")] for r in rows], dtype=float)


def _partition(rows, counts, cfg, rng):
    """Search only baseline features; never use any model/test outcome."""
    if sum(counts) != len(rows) or min(counts) < 1:
        raise ValueError("交叉验证组大小不合法")
    features = _features(rows)
    weights = np.asarray([cfg.split_slice_weight, cfg.split_tumor_presence_weight,
                          cfg.split_tumor_presence_weight, cfg.split_tumor_burden_weight,
                          cfg.split_sex_weight])
    target = np.asarray(counts)[:, None] / len(rows) * features.sum(0)
    # If a class has fewer patients than groups, covering every group is impossible.
    required = [c for c in (1, 2) if features[:, c].sum() >= len(counts)]
    if min(counts) < len(required):
        required = []
    best, best_score = None, float("inf")
    for _ in range(max(1, cfg.split_search_trials)):
        groups = np.split(rng.permutation(len(rows)), np.cumsum(counts)[:-1])
        observed = np.asarray([features[g].sum(0) for g in groups])
        if required and any((observed[:, c] == 0).any() for c in required):
            continue
        score = float(np.sum(weights * ((observed-target)/np.maximum(target, 1))**2))
        # Maximise achievable class coverage even when complete stratification is impossible.
        score += 1000 * int(np.count_nonzero(observed[:, 1:3] == 0))
        if score < best_score:
            best, best_score = groups, score
    if best is None:
        raise ValueError("在指定 split_search_trials 内未找到分层方案，请增加搜索次数")
    return [[rows[int(i)]["id"] for i in group] for group in best]


def _statistics(split, rows):
    by_id = {r["id"]: r for r in rows}
    return {name: {"patients": len(split[name]),
                   "slices": sum(int(by_id[p]["n_slices"]) for p in split[name]),
                   "tumor_positive": sum(int(bool(by_id[p]["has_tumor"])) for p in split[name]),
                   "tumor_negative": sum(int(not by_id[p]["has_tumor"]) for p in split[name]),
                   "tumor_voxels": sum(int(by_id[p]["tumor_voxels"]) for p in split[name])}
            for name in SPLIT_NAMES}


def make_folds(cfg, base_split=None, output=None):
    """Create frozen files and return manifest dict; base_split is a real JSON path.

    output accepts a JSON filename or directory; default: run_dir/crossval/folds.json.
    Existing protocols are never overwritten. Run a new protocol in a new directory.
    """
    cache = Path(cfg.cache_dir).resolve()
    audit = read_json(cache / "audit.json")
    source = Path(base_split or getattr(cfg, "split_path", None) or cache / "splits.json").resolve()
    if not source.is_file():
        raise FileNotFoundError(f"必须提供实际原始 splits.json，不能根据示例推测: {source}")
    original = read_json(source)
    rows = audit["patients"]
    _verify_split(original, rows, audit["fingerprint"])
    destination = Path(output or Path(cfg.run_dir) / "crossval").resolve()
    path = destination if destination.suffix.lower() == ".json" else destination / "folds.json"
    if path.is_relative_to(cache):
        raise ValueError("交叉验证输出必须位于共享 cache_dir 之外；缓存只读")
    if path.exists() or any((path.parent / f"fold_{f}").exists() for f in range(K_FOLDS)):
        raise FileExistsError("交叉验证协议/折目录已存在；请复用原 folds.json 或更换输出目录")
    val_size = len(original["val"])
    remaining = [r for r in rows if r["id"] not in set(original["test"])]
    small, extra = divmod(len(remaining), K_FOLDS-1)
    sizes = [small] * (K_FOLDS-1-extra) + [small+1] * extra
    if min(sizes) < 1 or len(rows)-max(sizes+[len(original["test"])])-val_size < 1:
        raise ValueError("患者不足以建立五个非空外层测试集并保留训练/验证患者")
    rng = np.random.default_rng(cfg.seed)
    test_groups = [list(original["test"])] + _partition(remaining, sizes, cfg, rng)
    notices, entries = [], []
    for label, number in (("肿瘤阳性", sum(bool(r["has_tumor"]) for r in rows)),
                          ("肿瘤阴性", sum(not r["has_tumor"] for r in rows))):
        if number < K_FOLDS:
            notices.append(f"{label}仅{number}人，少于5；无法保证每个外层测试折均含此类患者。")
    for f, test in enumerate(test_groups):
        if f == 0:
            split = copy.deepcopy(original)
        else:
            eligible = [r for r in rows if r["id"] not in set(test)]
            train, val = _partition(eligible, [len(eligible)-val_size, val_size], cfg, rng)
            split = {"train": sorted(train), "val": sorted(val), "test": sorted(test)}
            split.update({"seed": cfg.seed, "fingerprint": digest(_parts(split)),
                          "audit_fingerprint": audit["fingerprint"]})
        _verify_split(split, rows, audit["fingerprint"])
        stats = _statistics(split, rows)
        for name, group_stats in stats.items():
            if not group_stats["tumor_positive"] or not group_stats["tumor_negative"]:
                notices.append(f"fold_{f}/{name}: 阳性{group_stats['tumor_positive']}、阴性{group_stats['tumor_negative']}；此组无法同时覆盖两类。")
        if cfg.segmentation_weight > 0 and not stats["val"]["tumor_positive"]:
            raise ValueError(f"fold_{f} 验证集没有肿瘤阳性，现有联合选模无法运行；不能擅改原fold0")
        folder = path.parent / f"fold_{f}"
        split_file, config_file = folder / "splits.json", folder / "config.yaml"
        fold_cfg = replace(cfg, cache_dir=str(cache), data_root=str(Path(cfg.data_root).resolve()),
                           run_dir=str(folder / "run"), split_path=str(split_file),
                           split_counts=tuple(len(split[k]) for k in SPLIT_NAMES))
        entries.append({"fold": f, "split_path": str(split_file), "config_path": str(config_file),
                        "run_dir": fold_cfg.run_dir, "config_fingerprint": digest(config_dict(fold_cfg)),
                        "split_fingerprint": split["fingerprint"], "patients": _parts(split),
                        "statistics": stats, "_split": split, "_config": config_dict(fold_cfg)})
    # Complete validation before creating any deliverables.
    for entry in entries:
        write_json(entry["split_path"], entry.pop("_split"))
        Path(entry["config_path"]).write_text(yaml.safe_dump(entry.pop("_config"), allow_unicode=True), encoding="utf-8")
    protocol = {"schema": 1, "k": K_FOLDS, "seed": cfg.seed, "manifest_path": str(path),
                "cache_dir": str(cache), "audit_fingerprint": audit["fingerprint"],
                "patient_ids": [r["id"] for r in rows], "base_split_path": str(source),
                "base_split_fingerprint": original["fingerprint"], "base_split": _parts(original),
                "outer_test_sizes": [len(g) for g in test_groups],
                "aggregation": "out_of_fold_patient_macro", "folds": entries, "warnings": notices,
                "interpretation": "原测试结果已被查看后再修改模型，fold0不能视为未见过的确认集；本协议不是嵌套调参。"}
    protocol["fingerprint"] = digest(protocol)
    write_json(path, protocol)
    for notice in notices:
        warnings.warn(notice, UserWarning, stacklevel=2)
    print(f"已冻结五折协议: {path}; 外层测试人数={protocol['outer_test_sizes']}")
    return protocol


def _load_protocol(cfg, folds_path):
    protocol = read_json(folds_path)
    if protocol.get("fingerprint") != digest({k: v for k, v in protocol.items() if k != "fingerprint"}):
        raise ValueError("五折协议 fingerprint 不匹配，禁止修改已冻结协议")
    if protocol.get("k") != K_FOLDS or [f["fold"] for f in protocol["folds"]] != list(range(K_FOLDS)):
        raise ValueError("协议必须恰好包含 fold_0 到 fold_4")
    if Path(cfg.cache_dir).resolve() != Path(protocol["cache_dir"]):
        raise ValueError("当前 cache_dir 与五折协议不一致")
    audit = read_json(Path(protocol["cache_dir"]) / "audit.json")
    if audit["fingerprint"] != protocol["audit_fingerprint"] or set(protocol["patient_ids"]) != {r["id"] for r in audit["patients"]}:
        raise ValueError("当前审计数据与五折协议不同")
    test_ids, configs, run_paths = [], {}, []
    common_parameters = None
    for entry in protocol["folds"]:
        split = read_json(entry["split_path"])
        _verify_split(split, audit["patients"], audit["fingerprint"])
        if _parts(split) != entry["patients"] or split["fingerprint"] != entry["split_fingerprint"]:
            raise ValueError("实际折划分与冻结五折协议不符")
        fold_cfg = load_config(entry["config_path"])
        if digest(config_dict(fold_cfg)) != entry["config_fingerprint"]:
            raise ValueError("折 config.yaml 已改变；请为新超参数创建新协议")
        parameters = _common_hyperparameters(fold_cfg)
        if common_parameters is None:
            common_parameters = parameters
        elif digest(parameters) != digest(common_parameters):
            differences = [key for key in parameters
                           if digest(parameters[key]) != digest(common_parameters.get(key))]
            raise ValueError(f"跨折训练/评估超参数必须一致，fold_{entry['fold']} 与 fold_0 不同: {differences}")
        if (Path(fold_cfg.cache_dir).resolve() != Path(protocol["cache_dir"])
                or Path(fold_cfg.split_path).resolve() != Path(entry["split_path"])
                or Path(fold_cfg.run_dir).resolve() != Path(entry["run_dir"])):
            raise ValueError("折配置路径与协议不符")
        run_paths.append(str(Path(fold_cfg.run_dir).resolve()))
        configs[entry["fold"]] = fold_cfg
        test_ids.extend(split["test"])
    if len(set(run_paths)) != K_FOLDS:
        raise ValueError("不同折必须使用独立 run_dir")
    if len(test_ids) != len(set(test_ids)) or set(test_ids) != set(protocol["patient_ids"]):
        raise ValueError("外层测试必须恰好覆盖每名患者一次")
    if protocol["folds"][0]["patients"] != protocol["base_split"]:
        raise ValueError("fold_0 未逐项保留真实原始划分")
    return protocol, configs


def _selected(folds):
    selected = list(range(K_FOLDS)) if folds is None else [int(f) for f in folds]
    if not selected or len(selected) != len(set(selected)) or any(f < 0 or f >= K_FOLDS for f in selected):
        raise ValueError("folds 必须是不重复的0到4整数列表")
    return selected


def run_folds(cfg, folds_path, folds=None, evaluate_after=False):
    """Train requested folds from scratch; test evaluation is opt-in, never implicit."""
    from .train import train
    protocol, configs = _load_protocol(cfg, folds_path)
    selected = _selected(folds)
    for f in selected:
        if any(Path(configs[f].run_dir).glob("*.pt")):
            raise FileExistsError(f"fold_{f} 已有checkpoint；禁止覆盖或用其他折权重初始化")
    results = {}
    for f in selected:
        print(f"开始 fold_{f}，从头训练；外层测试患者不参与训练或选模")
        results[f] = train(configs[f], resume=None)
    if evaluate_after:
        evaluate_folds(cfg, folds_path, selected)
    return results


def _sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8*1024*1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _check_checkpoint(path, entry, protocol):
    path = Path(path).resolve()
    if path.parent != Path(entry["run_dir"]).resolve():
        raise ValueError("checkpoint 不在所属折独立 run_dir 内")
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if (ckpt.get("split_fingerprint") != entry["split_fingerprint"]
            or ckpt.get("audit_fingerprint") != protocol["audit_fingerprint"]):
        raise ValueError("checkpoint 实际训练划分/审计指纹与所属折不一致")
    if digest(ckpt.get("config")) != entry["config_fingerprint"]:
        raise ValueError("checkpoint 实际训练配置与冻结折配置不一致")
    return _sha256(path)


def evaluate_folds(cfg, folds_path, folds=None, checkpoint_name="best.pt", export=True):
    """Evaluate each selected model only on its own outer test patients."""
    from .evaluate import evaluate
    if Path(checkpoint_name).name != checkpoint_name or not checkpoint_name.endswith(".pt"):
        raise ValueError("checkpoint_name 必须是折 run_dir 内的 .pt 文件名")
    protocol, configs = _load_protocol(cfg, folds_path)
    results = {}
    for f in _selected(folds):
        entry = protocol["folds"][f]
        path = Path(entry["run_dir"]) / checkpoint_name
        sha = _check_checkpoint(path, entry, protocol)
        results[f] = evaluate(configs[f], path, split="test", export=export)
        out = Path(entry["run_dir"]) / "evaluation_test"
        provenance = read_json(out / "provenance.json")
        _verify_metric_protocol(provenance)
        provenance.update({"cv_fold": f, "cv_protocol_fingerprint": protocol["fingerprint"],
                           "checkpoint_sha256": sha,
                           "reconstruction_csv_sha256": _sha256(out / "reconstruction_per_patient.csv"),
                           "segmentation_csv_sha256": _sha256(out / "segmentation_per_patient.csv")})
        write_json(out / "provenance.json", provenance)
    return results


def _read_results(path, segmentation=False):
    strings = {"patient", "reference", "method"}
    rows = []
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            row = {}
            for k, value in raw.items():
                if k in strings:
                    row[k] = value
                elif k == "view":
                    row[k] = int(value)
                elif k == "tumor_positive":
                    if value.lower() not in ("true", "false", "1", "0"):
                        raise ValueError("tumor_positive CSV值无效")
                    row[k] = value.lower() in ("true", "1")
                else:
                    row[k] = None if value == "" else float(value)
            rows.append(row)
    return rows


def aggregate_folds(cfg, folds_path, output=None):
    """Require all 5 complete tests; concatenate patients, never average fold means."""
    from .evaluate import summarize
    protocol, configs = _load_protocol(cfg, folds_path)
    rec_all, seg_all, evidence, seen_checkpoints = [], [], [], set()
    for entry in protocol["folds"]:
        f = entry["fold"]
        folder = Path(entry["run_dir"]) / "evaluation_test"
        provenance = read_json(folder / "provenance.json")
        metric_protocol = _verify_metric_protocol(provenance)
        if (provenance.get("split") != "test" or provenance.get("patients") != entry["patients"]["test"]
                or provenance.get("split_fingerprint") != entry["split_fingerprint"]
                or provenance.get("audit_fingerprint") != protocol["audit_fingerprint"]
                or provenance.get("cv_fold") != f
                or provenance.get("cv_protocol_fingerprint") != protocol["fingerprint"]
                or digest(provenance.get("config")) != entry["config_fingerprint"]):
            raise ValueError(f"fold_{f} 推理来源不符；请用 cv-evaluate 生成可审计的测试结果")
        sha = _check_checkpoint(provenance["checkpoint"], entry, protocol)
        if sha != provenance.get("checkpoint_sha256") or sha in seen_checkpoints:
            raise ValueError("checkpoint 被替换或不同折复用了同一checkpoint")
        seen_checkpoints.add(sha)
        rec_path, seg_path = folder / "reconstruction_per_patient.csv", folder / "segmentation_per_patient.csv"
        if (_sha256(rec_path) != provenance.get("reconstruction_csv_sha256")
                or _sha256(seg_path) != provenance.get("segmentation_csv_sha256")):
            raise ValueError("测试CSV在推理后被修改，无法验证结果来源")
        rec, seg = _read_results(rec_path), _read_results(seg_path, True)
        expected_rec = {(p, v, ref, method) for p in entry["patients"]["test"] for v in configs[f].views
                        for ref in ("original_CT", "full_FBP") for method in ("FBP", "Joint")}
        actual_rec = [(r["patient"], r["view"], r["reference"], r["method"]) for r in rec]
        expected_seg = {(p, v) for p in entry["patients"]["test"] for v in configs[f].views}
        actual_seg = [(r["patient"], r["view"]) for r in seg]
        if (set(actual_rec) != expected_rec or len(actual_rec) != len(expected_rec)
                or set(actual_seg) != expected_seg or len(actual_seg) != len(expected_seg)):
            raise ValueError(f"fold_{f} 患者/视角指标缺失、重复或含非测试患者")
        rec_all.extend({"fold": f, **r} for r in rec)
        seg_all.extend({"fold": f, **r} for r in seg)
        evidence.append({"fold": f, "checkpoint": provenance["checkpoint"], "checkpoint_sha256": sha,
                         "split_fingerprint": entry["split_fingerprint"], "patients": entry["patients"]["test"],
                         "metric_protocol": metric_protocol})
    summary = summarize(rec_all, seg_all, configs[0])
    summary.update({"aggregation": "out_of_fold_patient_macro", "n_patients": len(protocol["patient_ids"]),
                    "outer_test_sizes": protocol["outer_test_sizes"], "cv_protocol_fingerprint": protocol["fingerprint"],
                    "metric_protocol": EXPECTED_METRIC_PROTOCOL,
                    "common_hyperparameters_fingerprint": digest(_common_hyperparameters(configs[0])),
                    "fold_provenance": evidence, "warnings": protocol["warnings"],
                    "interpretation": protocol["interpretation"],
                    "ci_note": "按患者配对bootstrap汇总；折模型训练集重叠，因此该区间不量化重新训练的全部不确定性。"})
    dest = Path(output or Path(folds_path).resolve().parent / "aggregate").resolve()
    if dest.is_relative_to(Path(protocol["cache_dir"])):
        raise ValueError("汇总输出必须在只读共享缓存之外")
    write_csv(dest / "reconstruction_per_patient.csv", rec_all)
    write_csv(dest / "segmentation_per_patient.csv", seg_all)
    write_json(dest / "summary.json", summary)
    return summary
