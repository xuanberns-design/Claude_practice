"""所有默认实验参数集中于本文件开头；YAML > 此处默认值。"""
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import re
import yaml


@dataclass
class Config:
    data_root: str = "data/3Dircadb1"
    cache_dir: str = "cache/default"
    run_dir: str = "runs/joint"
    split_path: str | None = None
    seed: int = 2026
    image_size: int = 512
    context_slices: int = 5
    # None 表示相邻层；固定 mm 可减少不同层厚造成的上下文差异。
    context_step_mm: float | None = 2.5
    hu_min: float = -1000.0
    hu_max: float = 2000.0
    window_min: float = -200.0
    window_max: float = 250.0
    mu_water_per_mm: float = 0.02
    full_views: int = 1024
    views: tuple = (32, 64, 128)
    view_weights: tuple = (0.5, 0.25, 0.25)
    fbp_filter: str = "ramp"
    photons_per_ray: float = 0.0
    readout_noise_std: float = 0.0
    # 读取候选肿瘤目录后，仅保留与专家肝脏标注重叠的肝内部分。
    # 7号的 tumor 目录需审计，但其肾上腺病灶不属于本任务目标。
    liver_pattern: str = r"(?i)^liver$"
    tumor_pattern: str = r"(?i)^(?:tumor|livertumor(?:s|[0-9]+)?)$"
    # 肝内占比低于此阈值的“肿瘤”视为非肝肿瘤（肝内体素/原始肿瘤体素），用于剔除如7号肾上腺病灶。
    min_liver_tumor_fraction: float = 0.5
    allow_instance_fallback: bool = False
    alignment_tolerance_mm: float = 0.1
    strict_official_counts: bool = True
    # None 与 strict_official_counts 同步；可对非标准数据显式关闭目录清单校验。
    strict_official_mask_inventory: bool | None = None
    # 7号的 tumor 目录必须读取以审计肝内占比；不能靠名单静默忽略原标签。
    known_tumor_negative: tuple = (5, 11, 14, 20)
    split_counts: tuple = (14, 3, 3)
    split_search_trials: int = 30000
    split_slice_weight: float = 2.0
    split_tumor_presence_weight: float = 2.0
    split_tumor_burden_weight: float = 0.5
    split_sex_weight: float = 0.25
    base_channels: int = 32
    reconstruction_backbone: str = "plain"
    reconstruction_upsample: str = "bilinear"
    batch_size: int = 2
    num_workers: int = 0
    epochs: int = 180
    warmup_epochs: int = 20
    ramp_epochs: int = 20
    samples_per_epoch: int = 4000
    tumor_sample_probability: float = 0.5
    tumor_boundary_sample_probability: float = 0.0
    patch_center_jitter_fraction: float = 0.5
    lr: float = 0.0002
    weight_decay: float = 0.0001
    grad_clip: float = 5.0
    amp: bool = True
    device: str = "auto"
    cpu_threads: int = 4
    reconstruction_weight: float = 1.0
    segmentation_weight: float = 1.0
    l1_weight: float = 1.0
    ssim_weight: float = 0.2
    gradient_weight: float = 0.1
    bce_weight: float = 0.5
    dice_weight: float = 0.5
    tumor_loss_weight: float = 2.0
    segmentation_threshold: float = 0.5
    validation_every: int = 5
    save_every_epochs: int = 0
    patience_validations: int = 15
    bootstrap_repeats: int = 5000
    # V2训练：不同强度空间的损失用显式HU尺度，不以3000HU压小重建梯度。
    reconstruction_scale_hu: float = 400.0
    recon_background_weight: float = 0.2
    recon_body_weight: float = 1.0
    recon_liver_weight: float = 4.0
    recon_tumor_weight: float = 8.0
    mse_weight: float = 0.5
    correction_weight: float = 0.05
    seg_to_recon_scale: float = 0.1
    seg_pretrain_clean: bool = True
    negative_tumor_weight: float = 0.5
    liver_hard_negative_weight: float = 0.0
    liver_hard_negative_fraction: float = 0.02
    seg_boundary_weight: float = 0.0
    laplacian_weight: float = 0.0
    # 256缓存可直接使用。512缓存+256patch可保留原分辨率而降低训练显存。
    patch_size: int | None = None
    patch_foreground_probability: float = 0.7
    inference_overlap: float = 0.5
    inference_tta: bool = False
    selection_guard: bool = True
    selection_guard_references: tuple = ("original_CT",)
    selection_ssim_tolerance: float = 0.005
    selection_psnr_tolerance_db: float = 0.1
    selection_roi_mae_relative_tolerance: float = 0.05
    selection_roi_mae_absolute_tolerance_hu: float = 1.0
    save_slice_every: int = 0
    # 可用原缓存；升级命令补充padding与HU分布审计，不静默修改旧数据。
    metric_body_threshold_hu: float = -500.0
    selection_reconstruction_weight: float = 0.65
    selection_roi_scale_hu: float = 50.0
    selection_fp_scale_ml: float = 10.0
    # ---------------- V4：双域重建（默认值保持 V3 行为，旧配置/权重可原样使用） ----------------
    # image=V3 图像域后处理；dual_domain=正弦图补全+可微FBP+图像精修（推荐，需整图训练）。
    reconstruction_mode: str = "image"
    # 正弦图补全的稠密角度数：须整除 full_views，且为每个 views 的整数倍（偶数）。
    sino_dense_views: int = 256
    sino_base_channels: int = 16
    # 角度方向周期填充行数（带探测器翻转），让卷积在 0°/180° 处连续。
    sino_angle_pad: int = 8
    # 已测角度直接写回测量值（硬数据一致性），网络只负责缺失角度。
    sino_data_consistency: bool = True
    # RCAB 瓶颈加入多膨胀率上下文，扩大条纹去除所需的感受野（仅新实验开启，旧权重无此模块）。
    reconstruction_dilated_bottleneck: bool = False
    # 损失：缺失角度正弦图 L1、双域中间 FBP 图 L1、频域 L1、测量角度投影一致性。
    sino_weight: float = 0.0
    dd_image_weight: float = 0.0
    fft_weight: float = 0.0
    projection_consistency_weight: float = 0.0
    projection_consistency_angles: int = 16
    # 双域模式下的精确几何增强（绕投影中心的 90° 旋转/镜像，正弦图同步变换）。
    augment_geometry: bool = True
    # ---------------- V4：分割 ----------------
    seg_backbone: str = "plain"
    seg_base_channels: int | None = None
    # 额外宽窗输入（HU），如 [-500, 500]；null 表示与 V3 相同只用肝窗。
    seg_wide_window: tuple | None = None
    seg_deep_supervision_weight: float = 0.0
    # 肿瘤 Focal-Tversky（alpha 罚假阳，beta 罚假阴；beta>alpha 偏向召回小病灶）。
    tumor_tversky_weight: float = 0.0
    tumor_tversky_alpha: float = 0.3
    tumor_tversky_beta: float = 0.7
    tumor_tversky_gamma: float = 0.75
    # 标签定义 tumor ⊆ liver：惩罚 p(tumor) > p(liver) 的不一致预测。
    seg_hierarchy_weight: float = 0.0
    # 推理期分割翻转 TTA（双域模式只对分割做，重建只算一次）。
    seg_tta: bool = False
    # ---------------- V4：3D 后处理（验证集自动调参） ----------------
    postprocess: bool = False
    tumor_threshold: float | None = None
    postprocess_keep_largest_liver: bool = True
    postprocess_fill_holes: bool = True
    # 肿瘤只保留在“预测肝脏向外扩张 margin_mm”内；null 关闭该约束。
    tumor_liver_margin_mm: float | None = 5.0
    min_tumor_ml: float = 0.0
    auto_tune_postprocess: bool = False
    tune_tumor_thresholds: tuple = (0.3, 0.4, 0.5, 0.6, 0.7)
    tune_min_tumor_ml: tuple = (0.0, 0.05, 0.2, 0.5)
    # 32视角召回优先校准；只用验证集选择，64/128保持原参数。
    tune_32_thresholds: tuple = (0.10, 0.15, 0.20, 0.25, 0.30)
    tune_32_min_ml: tuple = (0.0, 0.05, 0.2)
    tune_32_liver_margins: tuple = (5.0, 10.0, None)
    max_negative_fp_ml: float = 5.0
    # ---------------- V4：优化 ----------------
    ema_decay: float = 0.0
    lr_warmup_epochs: int = 0
    # best.pt 从该 epoch 起才参与选模/早停计数；null=warmup_epochs（V3行为）。
    # V3 的 best.pt 出现在分割刚满权重约10轮时（epoch 69），分割欠训练；V4 设为 warmup+ramp。
    selection_start_epoch: int | None = None
    # 新实验的32视角病灶召回优先选模；默认保持V4历史行为。
    selection_32_lesion_priority: bool = False
    # 独立的新run从已有权重初始化；resume仍只恢复同一run。
    seg_finetune_checkpoint: str | None = None
    freeze_reconstructor: bool = False
    lesion_balanced_sampling: bool = False
    # clean预训练末期逐步混入重建图，0为不混入。
    seg_pretrain_mix_epochs: int = 0

    def validate(self):
        if self.image_size < 32 or self.image_size % 16:
            raise ValueError("image_size 必须 >=32 且为16的倍数")
        if self.context_slices < 1 or self.context_slices % 2 != 1:
            raise ValueError("context_slices 必须是正奇数")
        if self.strict_official_mask_inventory not in (None, True, False):
            raise ValueError("strict_official_mask_inventory 必须为 true/false/null")
        try:
            re.compile(self.liver_pattern)
            re.compile(self.tumor_pattern)
        except re.error as exc:
            raise ValueError(f"mask目录正则表达式无效: {exc}") from exc
        if 7 in self.known_tumor_negative and re.fullmatch(self.tumor_pattern, "tumor"):
            raise ValueError("7号 tumor 目录必须存在（用于计算肝内占比），其阴性由 min_liver_tumor_fraction 决定，不能靠 known_tumor_negative 静默置负")
        if not 0 < self.min_liver_tumor_fraction <= 1:
            raise ValueError("min_liver_tumor_fraction 必须在(0,1]")
        if self.hu_min >= self.hu_max or self.window_min >= self.window_max:
            raise ValueError("HU/window 范围错误")
        if not (self.hu_min <= self.window_min < self.window_max <= self.hu_max):
            raise ValueError("window 必须在 HU 范围内")
        if not self.views or any(v <= 0 or self.full_views % v for v in self.views):
            raise ValueError("full_views 必须是每个稀疏角度数的整数倍")
        if len(self.views) != len(self.view_weights) or any(w <= 0 for w in self.view_weights):
            raise ValueError("view_weights 必须对应 views 且为正")
        if len(self.split_counts) != 3 or min(self.split_counts) < 1:
            raise ValueError("split_counts 必须有三个正整数")
        if self.epochs < 1 or self.validation_every < 1 or self.batch_size < 1:
            raise ValueError("训练整数参数必须为正")
        if self.context_step_mm is not None and self.context_step_mm <= 0:
            raise ValueError("context_step_mm 必须为正或 null")
        if not 0 <= self.tumor_sample_probability <= 1:
            raise ValueError("tumor_sample_probability 必须在[0,1]")
        if not 0 <= self.tumor_boundary_sample_probability <= 1:
            raise ValueError("tumor_boundary_sample_probability 必须在[0,1]")
        if not 0 <= self.patch_center_jitter_fraction <= 0.5:
            raise ValueError("patch_center_jitter_fraction 必须在[0,0.5]")
        if self.reconstruction_backbone not in ("plain", "rcab"):
            raise ValueError("reconstruction_backbone 必须为 plain 或 rcab")
        if self.reconstruction_upsample not in ("bilinear", "pixelshuffle"):
            raise ValueError("reconstruction_upsample 必须为 bilinear 或 pixelshuffle")
        if self.reconstruction_upsample == "pixelshuffle" and self.reconstruction_backbone != "rcab":
            raise ValueError("pixelshuffle 上采样仅支持 rcab 重建主干")
        if self.mu_water_per_mm <= 0 or self.photons_per_ray < 0 or self.readout_noise_std < 0:
            raise ValueError("投影物理参数错误")
        if self.readout_noise_std and not self.photons_per_ray:
            raise ValueError("读出噪声需要 photons_per_ray > 0")
        if self.samples_per_epoch < 1 or self.base_channels < 1 or self.num_workers < 0 or self.cpu_threads < 1:
            raise ValueError("样本数/通道数/线程数应为正，worker数应非负")
        if self.warmup_epochs < 0 or self.ramp_epochs < 0 or self.save_every_epochs < 0:
            raise ValueError("预热/渐增/保存间隔不能为负")
        if not 0 < self.segmentation_threshold < 1 or self.bootstrap_repeats < 1:
            raise ValueError("阈值应在(0,1)，bootstrap重复数应为正")
        if self.lr <= 0 or self.grad_clip <= 0 or self.weight_decay < 0 or self.tumor_loss_weight <= 0:
            raise ValueError("优化器/类别权重参数无效")
        if min(self.reconstruction_weight, self.segmentation_weight) < 0 or self.reconstruction_weight+self.segmentation_weight <= 0:
            raise ValueError("任务权重应非负且不能全部为零")
        if self.patch_size is not None and (self.patch_size < 32 or self.patch_size % 16 or self.patch_size > self.image_size):
            raise ValueError("patch_size需为32以上的16倍数，且不能大于image_size")
        if not 0 <= self.seg_to_recon_scale <= 1 or not 0 <= self.patch_foreground_probability <= 1:
            raise ValueError("梯度缩放/前景裁剪概率需在[0,1]")
        if not 0 <= self.inference_overlap < 1 or self.reconstruction_scale_hu <= 0:
            raise ValueError("滑窗重叠需在[0,1)，HU损失尺度必须为正")
        if min(self.recon_background_weight,self.recon_body_weight,self.recon_liver_weight,self.recon_tumor_weight) <= 0:
            raise ValueError("重建区域权重必须为正")
        if self.save_slice_every < 0 or self.negative_tumor_weight < 0:
            raise ValueError("输出间隔/阴性惩罚不可为负")
        if self.liver_hard_negative_weight < 0 or not 0 < self.liver_hard_negative_fraction <= 1:
            raise ValueError("肝脏难负样本权重/占比无效")
        if self.seg_boundary_weight < 0 or self.laplacian_weight < 0:
            raise ValueError("边界/Laplacian损失权重不可为负")
        if min(self.l1_weight,self.mse_weight,self.ssim_weight,self.gradient_weight,self.correction_weight,
               self.bce_weight,self.dice_weight) < 0:
            raise ValueError("损失权重不可为负")
        if not 0 <= self.selection_reconstruction_weight <= 1 or min(self.selection_roi_scale_hu,self.selection_fp_scale_ml) <= 0:
            raise ValueError("验证评分权重/尺度无效")
        if not self.selection_guard_references or any(r not in ("original_CT","full_FBP") for r in self.selection_guard_references):
            raise ValueError("选模参照必须为original_CT或full_FBP的非空列表")
        if min(self.selection_ssim_tolerance,self.selection_psnr_tolerance_db,
               self.selection_roi_mae_relative_tolerance,self.selection_roi_mae_absolute_tolerance_hu) < 0:
            raise ValueError("验证退化容差不可为负")
        self._validate_v4()
        return self

    def _validate_v4(self):
        if self.reconstruction_mode not in ("image", "dual_domain"):
            raise ValueError("reconstruction_mode 必须为 image 或 dual_domain")
        if self.reconstruction_mode == "dual_domain":
            if self.patch_size is not None and self.patch_size != self.image_size:
                raise ValueError("dual_domain 的投影/FBP 是整幅图算子，patch_size 必须为 null 或等于 image_size")
            a = self.sino_dense_views
            if a < 2 or a % 2 or self.full_views % a:
                raise ValueError("sino_dense_views 必须为偶数并整除 full_views")
            if any(a % v or v % 2 for v in self.views):
                raise ValueError("sino_dense_views 必须是每个稀疏视角数的整数倍，且稀疏视角数为偶数")
            if self.sino_base_channels < 4 or self.sino_angle_pad < 0:
                raise ValueError("sino_base_channels>=4，sino_angle_pad>=0")
        if min(self.sino_weight, self.dd_image_weight, self.fft_weight, self.projection_consistency_weight) < 0:
            raise ValueError("V4 重建损失权重不可为负")
        if self.projection_consistency_weight > 0 and self.reconstruction_mode != "dual_domain":
            raise ValueError("projection_consistency_weight 需要 dual_domain 模式（需要测量正弦图）")
        if self.projection_consistency_angles < 1:
            raise ValueError("projection_consistency_angles 必须为正")
        if self.seg_backbone not in ("plain", "resattn"):
            raise ValueError("seg_backbone 必须为 plain 或 resattn")
        if self.seg_base_channels is not None and self.seg_base_channels < 4:
            raise ValueError("seg_base_channels 必须 >=4 或 null")
        if self.seg_wide_window is not None and (len(self.seg_wide_window) != 2
                                                 or self.seg_wide_window[0] >= self.seg_wide_window[1]):
            raise ValueError("seg_wide_window 必须为 [low, high]")
        if min(self.seg_deep_supervision_weight, self.tumor_tversky_weight, self.seg_hierarchy_weight) < 0:
            raise ValueError("V4 分割损失权重不可为负")
        if self.seg_deep_supervision_weight > 0 and self.seg_backbone != "resattn":
            raise ValueError("深监督仅支持 seg_backbone=resattn")
        if not (0 <= self.tumor_tversky_alpha <= 1 and 0 <= self.tumor_tversky_beta <= 1
                and self.tumor_tversky_alpha + self.tumor_tversky_beta > 0 and self.tumor_tversky_gamma > 0):
            raise ValueError("Tversky 参数无效")
        thresholds = list(self.tune_tumor_thresholds) + ([self.tumor_threshold] if self.tumor_threshold is not None else [])
        if not thresholds or any(not 0 < t < 1 for t in thresholds):
            raise ValueError("肿瘤阈值必须在(0,1)")
        if not self.tune_min_tumor_ml or min(list(self.tune_min_tumor_ml) + [self.min_tumor_ml]) < 0:
            raise ValueError("最小肿瘤体积不可为负")
        if self.tumor_liver_margin_mm is not None and self.tumor_liver_margin_mm < 0:
            raise ValueError("tumor_liver_margin_mm 必须非负或 null")
        if not self.tune_32_thresholds or any(not 0 < t < 1 for t in self.tune_32_thresholds):
            raise ValueError("tune_32_thresholds 必须是(0,1)内的非空列表")
        if not self.tune_32_min_ml or any(v < 0 for v in self.tune_32_min_ml):
            raise ValueError("tune_32_min_ml 必须是非负的非空列表")
        if not self.tune_32_liver_margins or any(v is not None and v < 0 for v in self.tune_32_liver_margins):
            raise ValueError("tune_32_liver_margins 必须是非负毫米数或 null 的非空列表")
        if self.max_negative_fp_ml < 0:
            raise ValueError("max_negative_fp_ml 不可为负")
        if self.seg_pretrain_mix_epochs < 0 or self.seg_pretrain_mix_epochs > self.warmup_epochs:
            raise ValueError("seg_pretrain_mix_epochs 必须在[0,warmup_epochs]内")
        if self.freeze_reconstructor and not self.seg_finetune_checkpoint:
            raise ValueError("固定重建器需要 seg_finetune_checkpoint")
        if self.seg_finetune_checkpoint and self.segmentation_weight <= 0:
            raise ValueError("分割微调需要 segmentation_weight > 0")
        if self.selection_start_epoch is not None and not 0 <= self.selection_start_epoch < self.epochs:
            raise ValueError("selection_start_epoch 必须在[0, epochs)")
        if not 0 <= self.ema_decay < 1 or self.lr_warmup_epochs < 0:
            raise ValueError("ema_decay 需在[0,1)，lr_warmup_epochs 非负")


def load_config(path=None):
    cfg = Config(**(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})) if path else Config()
    return cfg.validate()


def config_dict(cfg):
    return asdict(cfg)


def split_file(cfg):
    return Path(cfg.split_path) if cfg.split_path else Path(cfg.cache_dir) / "splits.json"


def prepare_signature(cfg):
    keys = ["image_size", "hu_min", "hu_max", "mu_water_per_mm", "full_views", "views",
            "fbp_filter", "photons_per_ray", "readout_noise_std", "seed", "liver_pattern",
            "tumor_pattern", "known_tumor_negative", "strict_official_mask_inventory", "min_liver_tumor_fraction",
            "allow_instance_fallback", "alignment_tolerance_mm"]
    obj = {k: asdict(cfg)[k] for k in keys}
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()
