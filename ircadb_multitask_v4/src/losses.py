"""损失函数；权重均在 config.py 顶部，图像动态范围固定为1。

V4 新增（默认权重为0，保持 V3 行为）：
- 双域：缺失角度正弦图 L1、双域中间 FBP 图加权 L1、测量角度投影一致性；
- 频域 L1：稀疏角条纹是沿特定方向的高频能量，频域误差直接惩罚这类结构；
- 分割：肿瘤逐样本 Focal-Tversky、tumor⊆liver 层级一致性、深监督。
"""
import torch
import torch.nn.functional as F


def ssim_loss(x, y):
    x, y = x.float(), y.float()
    # valid 11x11 窗，不给边缘引入零填充偏差；训练近似SSIM，报告用skimage标准实现。
    mx, my = F.avg_pool2d(x, 11, 1), F.avg_pool2d(y, 11, 1)
    vx = F.avg_pool2d(x*x, 11, 1) - mx*mx
    vy = F.avg_pool2d(y*y, 11, 1) - my*my
    cov = F.avg_pool2d(x*y, 11, 1) - mx*my
    score = ((2*mx*my + 0.01**2) * (2*cov + 0.03**2)) / ((mx*mx + my*my + 0.01**2) * (vx + vy + 0.03**2))
    return 1 - score.mean()


def reconstruction_loss(pred, target, cfg, mask_stack=None, sparse=None):
    pred, target = pred.float(), target.float()
    target_hu = target*(cfg.hu_max-cfg.hu_min)+cfg.hu_min
    weights = torch.full_like(target, cfg.recon_background_weight)
    weights = torch.where(target_hu > cfg.metric_body_threshold_hu, cfg.recon_body_weight, weights)
    if mask_stack is not None:
        weights = torch.where(mask_stack[:,:,0] > 0, cfg.recon_liver_weight, weights)
        weights = torch.where(mask_stack[:,:,1] > 0, cfg.recon_tumor_weight, weights)
    scale = (cfg.hu_max-cfg.hu_min)/cfg.reconstruction_scale_hu
    error = (pred-target)*scale
    denom = weights.sum().clamp_min(1)
    l1 = (weights*error.abs()).sum()/denom
    mse = (weights*error.square()).sum()/denom
    def window(a):
        hu = a*(cfg.hu_max-cfg.hu_min)+cfg.hu_min
        return ((hu-cfg.window_min)/(cfg.window_max-cfg.window_min)).clamp(0,1)
    ssim = ssim_loss(window(pred), window(target))
    dx = error[..., :, 1:]-error[..., :, :-1]
    dy = error[..., 1:, :]-error[..., :-1, :]
    wx = (weights[..., :, 1:]+weights[..., :, :-1])/2
    wy = (weights[..., 1:, :]+weights[..., :-1, :])/2
    gradient = ((dx.abs()*wx).sum()/wx.sum().clamp_min(1)+(dy.abs()*wy).sum()/wy.sum().clamp_min(1))/2
    # 对真值相对误差的二阶边缘做轻量监督；只在显式开启时计算。
    # 与旧包直接锐化输出不同，这一项不能奖励无真值依据的高频条纹。
    laplacian = pred.new_zeros(())
    if getattr(cfg, "laplacian_weight", 0) > 0:
        window_error = window(pred) - window(target)
        b, c, h, w = window_error.shape
        kernel = window_error.new_tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])[None, None]
        edge_error = F.conv2d(F.pad(window_error.reshape(b*c, 1, h, w),
                                    (1, 1, 1, 1), mode="replicate"), kernel).reshape_as(window_error)
        laplacian = (edge_error.abs() * weights).sum() / denom
    # 训练时已知GT，对每个样本超过自身FBP误差的部分额外惩罚，不在测试时选像素。
    regression = pred.new_zeros(())
    if sparse is not None:
        axes = tuple(range(1,pred.ndim))
        per_denom = weights.sum(axes).clamp_min(1)
        model_mse = (error.square()*weights).sum(axes)/per_denom
        fbp_mse = (((sparse.float()-target)*scale).square()*weights).sum(axes)/per_denom
        regression = (model_mse-fbp_mse).clamp_min(0).mean()
    frequency = pred.new_zeros(())
    if getattr(cfg, "fft_weight", 0) > 0:
        # 体部加权后的窗内误差做2D FFT；幅度按像素数归一，条纹对应的方向性高频被直接惩罚。
        body = (weights >= cfg.recon_body_weight).to(pred.dtype)
        spectrum = torch.fft.rfft2((window(pred) - window(target)) * body, norm="ortho")
        frequency = spectrum.abs().mean()
    return (cfg.l1_weight*l1+cfg.mse_weight*mse+cfg.ssim_weight*ssim
            +cfg.gradient_weight*gradient+cfg.correction_weight*regression
            +getattr(cfg, "laplacian_weight", 0)*laplacian+getattr(cfg, "fft_weight", 0)*frequency)


def region_weights(target, cfg, mask_stack=None):
    target_hu = target*(cfg.hu_max-cfg.hu_min)+cfg.hu_min
    weights = torch.full_like(target, cfg.recon_background_weight)
    weights = torch.where(target_hu > cfg.metric_body_threshold_hu, cfg.recon_body_weight, weights)
    if mask_stack is not None:
        weights = torch.where(mask_stack[:,:,0] > 0, cfg.recon_liver_weight, weights)
        weights = torch.where(mask_stack[:,:,1] > 0, cfg.recon_tumor_weight, weights)
    return weights


def dual_domain_loss(aux, batch, cfg, restored, model=None):
    """正弦图补全与中间FBP图的监督；aux 来自 DualDomainReconstructor。"""
    parts = {}
    total = restored.new_zeros((), dtype=torch.float32)
    if not aux:
        return total, parts
    scale = 1.0 / cfg.image_size
    if cfg.sino_weight > 0:
        missing = (aux["sino_mask"] == 0).float()
        error = (aux["sino"].float() - batch["sino_target"].float()).abs() * scale
        sino = (error * missing).sum() / (missing.sum() * error.shape[1] * error.shape[-1]).clamp_min(1)
        total = total + cfg.sino_weight * sino
        parts["sinogram"] = float(sino.detach())
    if cfg.dd_image_weight > 0:
        target = batch["target"].float()
        weights = region_weights(target, cfg, batch.get("mask_stack"))
        err = (aux["dd_image"].float() - target).abs() * (cfg.hu_max - cfg.hu_min) / cfg.reconstruction_scale_hu
        dd = (weights * err).sum() / weights.sum().clamp_min(1)
        total = total + cfg.dd_image_weight * dd
        parts["dual_domain_image"] = float(dd.detach())
    if cfg.projection_consistency_weight > 0 and model is not None:
        # 最终输出的中心层重投影到随机抽取的“已测角度”，与测量值比较（训练期已知测量，推理不需要）。
        op = model.reconstructor.op
        c = cfg.context_slices // 2
        mask_rows = aux["sino_mask"][:, 0, :, 0] > 0
        common = mask_rows.all(0).nonzero()[:, 0]
        if len(common):
            pick = common[torch.randperm(len(common), device=common.device)[:cfg.projection_consistency_angles]]
            hu = restored[:, c].float()*(cfg.hu_max-cfg.hu_min)+cfg.hu_min
            mu_ratio = (hu/1000.0+1.0).clamp_min(0)
            projected = op.project(mu_ratio, pick)
            measured = batch["sino"][:, c][:, pick].float()
            pc = (projected-measured).abs().mean()*scale
            total = total + cfg.projection_consistency_weight * pc
            parts["projection_consistency"] = float(pc.detach())
    return total, parts


def segmentation_loss(logits, target, cfg):
    logits, target = logits.float(), target.float()
    bce_map = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = bce_map.mean((0, 2, 3))
    p = logits.sigmoid()
    intersection = (p*target).sum((0, 2, 3))
    dice = 1 - (2*intersection + 1e-5) / (p.sum((0, 2, 3)) + target.sum((0, 2, 3)) + 1e-5)
    weights = logits.new_tensor([1, cfg.tumor_loss_weight])
    loss = ((cfg.bce_weight*bce + cfg.dice_weight*dice)*weights).sum() / weights.sum()
    empty = target[:,1].sum((1,2)) == 0
    if empty.any():
        # 空肿瘤层保留明确BCE负监督，避免平滑Dice的梯度过小。
        negative = F.binary_cross_entropy_with_logits(logits[empty,1], torch.zeros_like(logits[empty,1]))
        loss = loss+cfg.negative_tumor_weight*negative
    boundary_weight = getattr(cfg, "seg_boundary_weight", 0)
    if boundary_weight > 0:
        # 对真值边缘的细小缺口/外溢提高权重，分别处理 liver 与 tumor。
        dilated = F.max_pool2d(target, 3, stride=1, padding=1)
        eroded = -F.max_pool2d(-target, 3, stride=1, padding=1)
        boundary = (dilated - eroded).clamp(0, 1)
        edge_bce = (bce_map*boundary).sum((0, 2, 3)) / boundary.sum((0, 2, 3)).clamp_min(1)
        loss = loss + boundary_weight * (edge_bce*weights).sum()/weights.sum()
    tversky_weight = getattr(cfg, "tumor_tversky_weight", 0)
    if tversky_weight > 0:
        # 逐样本计算：批量级Dice会被大病灶主导，小病灶样本几乎没有梯度。
        p_t, t_t = p[:, 1], target[:, 1]
        tp = (p_t*t_t).sum((1, 2))
        fp = (p_t*(1-t_t)).sum((1, 2))
        fn = ((1-p_t)*t_t).sum((1, 2))
        tversky = (tp+1.0)/(tp+cfg.tumor_tversky_alpha*fp+cfg.tumor_tversky_beta*fn+1.0)
        positive = t_t.sum((1, 2)) > 0
        if positive.any():
            focal = (1-tversky[positive]).clamp_min(0).pow(cfg.tumor_tversky_gamma).mean()
            loss = loss + tversky_weight*focal
    hierarchy_weight = getattr(cfg, "seg_hierarchy_weight", 0)
    if hierarchy_weight > 0:
        loss = loss + hierarchy_weight*F.relu(p[:, 1]-p[:, 0]).mean()
    hard_negative_weight = getattr(cfg, "liver_hard_negative_weight", 0)
    if hard_negative_weight > 0:
        # 从非肝像素中只选择预测最像肝的部分，重点惩罚外侧血管类假阳性。
        fraction = getattr(cfg, "liver_hard_negative_fraction", 0.02)
        hard_terms = []
        for i in range(logits.shape[0]):
            negative_errors = bce_map[i, 0][target[i, 0] < 0.5]
            if negative_errors.numel():
                k = max(1, int(round(fraction * negative_errors.numel())))
                hard_terms.append(negative_errors.topk(min(k, negative_errors.numel())).values.mean())
        if hard_terms:
            loss = loss + hard_negative_weight * torch.stack(hard_terms).mean()
    return loss


def deep_supervision_loss(aux_logits, target, cfg):
    """解码器低分辨率辅助输出：标签以最大池化下采样，保证小病灶不被平均掉。"""
    if not aux_logits:
        return target.new_zeros(())
    terms, weights = [], []
    for level, logits in enumerate(aux_logits):
        factor = target.shape[-1] // logits.shape[-1]
        small = F.adaptive_max_pool2d(target, logits.shape[-2:]) if factor > 1 else target
        terms.append(segmentation_loss(logits, small, cfg))
        weights.append(0.5 ** level)
    return sum(w*t for w, t in zip(weights, terms)) / sum(weights)


def joint_loss(restored, logits, batch, cfg, epoch, rec_aux=None, seg_aux=None, model=None):
    rec = reconstruction_loss(restored, batch["target"], cfg, batch.get("mask_stack"), batch["input"])
    dd, dd_parts = dual_domain_loss(rec_aux, batch, cfg, restored, model)
    seg = segmentation_loss(logits, batch["mask"], cfg)
    if seg_aux and cfg.seg_deep_supervision_weight > 0:
        seg = seg + cfg.seg_deep_supervision_weight*deep_supervision_loss(seg_aux, batch["mask"].float(), cfg)
    ramp = (1.0 if cfg.seg_pretrain_clean and epoch < cfg.warmup_epochs else
            min(1.0, max(0.0, (epoch - cfg.warmup_epochs + 1) / max(cfg.ramp_epochs, 1))))
    loss = cfg.reconstruction_weight*(rec + dd) + cfg.segmentation_weight*ramp*seg
    return loss, {"reconstruction": float(rec.detach()), "segmentation": float(seg.detach()), "seg_ramp": ramp,
                  **dd_parts}
