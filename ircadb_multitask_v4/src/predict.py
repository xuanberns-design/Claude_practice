"""训练patch匹配的重叠滑窗推理；evaluate与infer必须共用本文件。

参数入口：Config.patch_size/inference_overlap/inference_tta/seg_tta。TTA默认关闭。
"""
import torch


def sliding_starts(length, patch, overlap):
    if patch >= length:
        return [0]
    stride = max(1, int(round(patch*(1-overlap))))
    starts = list(range(0,length-patch+1,stride))
    if starts[-1] != length-patch:
        starts.append(length-patch)
    return starts


def _segment_tta(model, restored, view, cfg):
    flips = [()] if not getattr(cfg, "seg_tta", False) else [(), (-1,), (-2,), (-2, -1)]
    prob = 0
    for dims in flips:
        r = torch.flip(restored, dims) if dims else restored
        p = model.segment(r, view).float().sigmoid()
        prob = prob + (torch.flip(p, dims) if dims else p)
    return prob / len(flips)


@torch.inference_mode()
def predict_batch(model, x, view, cfg, sino=None, n_views=None):
    """返回中心恢复层[B,H,W]和概率[B,2,H,W]，不读取GT或mask。

    dual_domain：整图一次性重建（投影/FBP为全局算子，不能分块），分割可选翻转TTA。
    """
    if getattr(model, "dual_domain", False):
        restored = model.restore(x, view, sino, n_views)
        c = cfg.context_slices // 2
        return restored[:, c].float(), _segment_tta(model, restored, view, cfg)
    ps = cfg.patch_size or x.shape[-1]
    ps = min(ps,x.shape[-2],x.shape[-1])
    # 下限确保边界权重非零，避免图像四周0/0或不连续拼接。
    taper = torch.hann_window(ps,periodic=False,device=x.device,dtype=torch.float32).clamp_min(.05)
    weight = (taper[:,None]*taper[None,:])[None,None]
    rec = torch.zeros((len(x),1,*x.shape[-2:]),device=x.device,dtype=torch.float32)
    prob = torch.zeros((len(x),2,*x.shape[-2:]),device=x.device,dtype=torch.float32)
    norm = torch.zeros_like(rec)
    flips = [()] if not cfg.inference_tta else [(),(-1,),(-2,),(-2,-1)]
    for y in sliding_starts(x.shape[-2],ps,cfg.inference_overlap):
        for z in sliding_starts(x.shape[-1],ps,cfg.inference_overlap):
            tile = x[...,y:y+ps,z:z+ps]
            tr = torch.zeros_like(tile[:,0:1],dtype=torch.float32)
            tp = torch.zeros((len(x),2,ps,ps),device=x.device,dtype=torch.float32)
            for dims in flips:
                r, logits = model(torch.flip(tile,dims) if dims else tile,view)
                r = r[:,cfg.context_slices//2:cfg.context_slices//2+1].float()
                p = logits.float().sigmoid()
                tr += torch.flip(r,dims) if dims else r
                tp += torch.flip(p,dims) if dims else p
            rec[...,y:y+ps,z:z+ps] += tr/len(flips)*weight
            prob[...,y:y+ps,z:z+ps] += tp/len(flips)*weight
            norm[...,y:y+ps,z:z+ps] += weight
    if not (norm>0).all():
        raise RuntimeError("滑窗未覆盖全部像素")
    return (rec/norm)[:,0], prob/norm
