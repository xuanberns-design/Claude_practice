"""患者级分层划分；只用基线数据特征，不用任何模型效果挑选测试集。"""
from pathlib import Path
import hashlib
import numpy as np
from .common import read_json, write_json, digest


def validate_split(splits, ids):
    flat = [p for name in ("train", "val", "test") for p in splits[name]]
    if len(flat) != len(set(flat)) or set(flat) != set(ids):
        raise ValueError("划分包含重复/交叉病例或未覆盖审计数据")
    if any(not splits[k] for k in ("train", "val", "test")):
        raise ValueError("训练/验证/测试集合不能为空")


def balanced_split(rows, cfg):
    counts = np.asarray(cfg.split_counts)
    if sum(counts) != len(rows):
        raise ValueError(f"split_counts={counts.tolist()} 与 {len(rows)} 名患者不符")
    fractions = counts / len(rows)
    features = np.asarray([[r["n_slices"], float(r["has_tumor"]),
                             np.log1p(r["tumor_voxels"]), float(r["sex"] == "F")]
                            for r in rows], float)
    weights = np.asarray([cfg.split_slice_weight, cfg.split_tumor_presence_weight,
                          cfg.split_tumor_burden_weight, cfg.split_sex_weight])
    targets = fractions[:, None] * features.sum(axis=0)
    rng = np.random.default_rng(cfg.seed)
    best, best_score = None, float("inf")
    for _ in range(cfg.split_search_trials):
        groups = np.split(rng.permutation(len(rows)), np.cumsum(counts)[:-1])
        positives = [features[g, 1].sum() for g in groups]
        # 数据充足时，三个集合都同时含肿瘤阳性与阴性。
        if features[:, 1].sum() >= 3 and min(positives) == 0:
            continue
        if (len(rows) - features[:, 1].sum()) >= 3 and any(positives[i] == len(g) for i, g in enumerate(groups)):
            continue
        observed = np.asarray([features[g].sum(axis=0) for g in groups])
        score = float(np.sum(weights * ((observed - targets) / np.maximum(targets, 1)) ** 2))
        if score < best_score:
            best, best_score = groups, score
    if best is None:
        raise ValueError("找不到满足阳性/阴性约束的划分，请检查配置与样本数")
    result = {k: sorted([rows[i]["id"] for i in group], key=lambda s: int(s.split(".")[-1]))
              for k, group in zip(("train", "val", "test"), best)}
    validate_split(result, [r["id"] for r in rows])
    result["statistics"] = {k: {"patients": len(g), "slices": int(features[g, 0].sum()),
                                     "tumor_positive": int(features[g, 1].sum())}
                            for k, g in zip(("train", "val", "test"), best)}
    result["seed"] = cfg.seed
    result["balance_score"] = best_score
    result["fingerprint"] = digest({k: result[k] for k in ("train", "val", "test")})
    return result


def make_split(cfg, base_split=None):
    """Freeze a split, or explicitly rebind an older patient list to a new label audit.

    Rebinding never carries over old tumour statistics: those are recomputed from
    the newly audited DICOM masks. Existing cache splits are never overwritten.
    """
    manifest = read_json(Path(cfg.cache_dir) / "audit.json")
    path = Path(cfg.cache_dir) / "splits.json"
    if path.exists():
        if base_split is not None:
            raise FileExistsError("已有冻结划分；不能用 --base-split 覆盖，请使用新的 cache_dir")
        old = read_json(path)
        validate_split(old, [r["id"] for r in manifest["patients"]])
        if old["fingerprint"] != digest({k: old[k] for k in ("train", "val", "test")}):
            raise ValueError("已有划分文件被修改；请保留原实验，另建cache_dir重做划分")
        if old["audit_fingerprint"] != manifest["fingerprint"]:
            raise ValueError("已有划分的数据指纹不同；请使用新的 cache_dir，禁止静默覆盖")
        print("复用已冻结的 splits.json")
        return old
    if base_split is None:
        result = balanced_split(manifest["patients"], cfg)
    else:
        source = Path(base_split).resolve()
        if not source.is_file() or source == path.resolve():
            raise FileNotFoundError(f"必须提供不同于新缓存划分的真实原始 splits.json: {source}")
        original = read_json(source)
        patients = manifest["patients"]
        validate_split(original, [r["id"] for r in patients])
        groups = {k: list(original[k]) for k in ("train", "val", "test")}
        fingerprint = digest(groups)
        if original.get("fingerprint") != fingerprint:
            raise ValueError("原始划分 fingerprint 与患者名单不符，不能迁移")
        by_id = {r["id"]: r for r in patients}
        result = {**groups,
                  "statistics": {k: {"patients": len(ids),
                                     "slices": sum(int(by_id[p]["n_slices"]) for p in ids),
                                     "tumor_positive": sum(bool(by_id[p]["has_tumor"]) for p in ids),
                                     "tumor_negative": sum(not by_id[p]["has_tumor"] for p in ids),
                                     "tumor_voxels": sum(int(by_id[p]["tumor_voxels"]) for p in ids)}
                                 for k, ids in groups.items()},
                  "seed": original.get("seed", cfg.seed), "fingerprint": fingerprint,
                  "rebound_from": {"source_path": str(source),
                                   "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                                   "source_audit_fingerprint": original.get("audit_fingerprint")}}
    result["audit_fingerprint"] = manifest["fingerprint"]
    write_json(path, result)
    return result
