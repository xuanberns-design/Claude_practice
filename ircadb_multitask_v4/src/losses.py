"""损失函数；权重均在 config.py 顶部，图像动态范围固定为1。"""
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
    return (cfg.l1_weight*l1+cfg.mse_weight*mse+cfg.ssim_weight*ssim
            +cfg.gradient_weight*gradient+cfg.correction_weight*regression
            +getattr(cfg, "laplacian_weight", 0)*laplacian)


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


def joint_loss(restored, logits, batch, cfg, epoch):
    rec = reconstruction_loss(restored, batch["target"], cfg, batch.get("mask_stack"), batch["input"])
    seg = segmentation_loss(logits, batch["mask"], cfg)
    ramp = (1.0 if cfg.seg_pretrain_clean and epoch < cfg.warmup_epochs else
            min(1.0, max(0.0, (epoch - cfg.warmup_epochs + 1) / max(cfg.ramp_epochs, 1))))
    loss = cfg.reconstruction_weight*rec + cfg.segmentation_weight*ramp*seg
    return loss, {"reconstruction": float(rec.detach()), "segmentation": float(seg.detach()), "seg_ramp": ramp}
