"""V2损失、梯度、滑窗及模型选择的验证。"""
from dataclasses import replace
import pytest
import torch
from src.config import Config
from src.model import JointUNet
from src.predict import predict_batch
from src.losses import reconstruction_loss,segmentation_loss,joint_loss
from src.evaluate import reconstruction_eligibility


def test_gradient_scaling_preserves_forward_and_seg_gradient():
    torch.set_num_threads(2)
    cfg=Config(image_size=32,context_slices=3,base_channels=4,seg_to_recon_scale=1)
    a=JointUNet(cfg)
    b=JointUNet(replace(cfg,seg_to_recon_scale=.1));b.load_state_dict(a.state_dict())
    x=torch.rand(2,3,32,32);v=torch.tensor([.5,.7])
    ra,sa=a(x,v);rb,sb=b(x,v)
    assert torch.equal(ra,rb) and torch.equal(sa,sb)
    sa.square().mean().backward();sb.square().mean().backward()
    assert torch.allclose(a.reconstructor.head.weight.grad*.1,b.reconstructor.head.weight.grad,atol=1e-7,rtol=1e-5)
    assert torch.allclose(a.segmenter.head.weight.grad,b.segmenter.head.weight.grad)


def test_foreground_loss_penalizes_same_liver_error_more():
    cfg=Config(image_size=32,context_slices=1,ssim_weight=0,gradient_weight=0)
    truth=torch.full((1,1,32,32),.35)
    masks=torch.zeros(1,1,2,32,32);masks[:,:,0,0:8,0:8]=1
    liver=truth.clone();liver[:,:,1:5,1:5]+=.01
    outside=truth.clone();outside[:,:,20:24,20:24]+=.01
    assert reconstruction_loss(liver,truth,cfg,masks)>reconstruction_loss(outside,truth,cfg,masks)


class PointModel(torch.nn.Module):
    def forward(self,x,view):
        return x+.03,torch.stack([x[:,1]*2,x[:,1]-1],dim=1)


@pytest.mark.parametrize('tta',[False,True])
@pytest.mark.parametrize('overlap',[0,.5,.9])
def test_sliding_has_no_holes_or_center_channel_error(tta,overlap):
    cfg=Config(image_size=96,patch_size=32,context_slices=3,inference_tta=tta,inference_overlap=overlap)
    x=torch.rand(2,3,65,81)
    r,p=predict_batch(PointModel(),x,torch.tensor([.5,.7]),cfg)
    assert torch.allclose(r,x[:,1]+.03,atol=1e-6)
    assert torch.allclose(p,torch.stack([x[:,1]*2,x[:,1]-1],dim=1).sigmoid(),atol=1e-6)


def test_guard_catches_high_view_liver_regression():
    cfg=Config()
    summary={'reconstruction':{}}
    for v in cfg.views:
        for reference in ['original_CT','full_FBP']:
            for method in ['FBP','Joint']:
                summary['reconstruction'][f'{v}/{reference}/{method}']={k:{'mean':value} for k,value in [('SSIM',.9),('PSNR_dB',30),('liver_MAE_HU',10)]}
    assert reconstruction_eligibility(summary,cfg)['eligible']
    summary['reconstruction']['128/original_CT/Joint']['liver_MAE_HU']['mean']=30
    result=reconstruction_eligibility(summary,cfg)
    assert not result['eligible'] and any('128/liver_MAE' in s for s in result['violations'])


def test_negative_tumor_penalty_is_explicit():
    cfg=Config()
    truth=torch.zeros(2,2,32,32)
    logits=torch.zeros_like(truth)
    assert segmentation_loss(logits,truth,cfg)>segmentation_loss(logits,truth,replace(cfg,negative_tumor_weight=0))


def test_clean_seg_pretrain_no_seg_gradient_to_reconstruction():
    cfg=Config(image_size=32,context_slices=3,base_channels=4,warmup_epochs=2,seg_pretrain_clean=True)
    model=JointUNet(cfg)
    target=torch.rand(2,3,32,32)
    logits=model.segment(target,torch.tensor([.5,.5]))
    segmentation_loss(logits,torch.zeros_like(logits),cfg).backward()
    assert all(p.grad is None for p in model.reconstructor.parameters())


def test_loss_all_terms_have_finite_gradients():
    cfg=Config(image_size=32,context_slices=3)
    pred=torch.rand(2,3,32,32,requires_grad=True)
    logits=torch.randn(2,2,32,32,requires_grad=True)
    batch={'target':torch.rand_like(pred),'input':torch.rand_like(pred),'mask':torch.zeros_like(logits),
           'mask_stack':torch.zeros(2,3,2,32,32)}
    loss,_=joint_loss(pred,logits,batch,cfg,0)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(pred.grad).all() and torch.isfinite(logits.grad).all()


def test_negative_weights_rejected():
    with pytest.raises(ValueError,match='不可为负'):
        Config(correction_weight=-1).validate()


def test_guard_does_not_conflate_original_with_simulated_full_fbp():
    cfg=Config()
    summary={'reconstruction':{}}
    for v in cfg.views:
        for ref in ('original_CT','full_FBP'):
            for method in ('FBP','Joint'):
                summary['reconstruction'][f'{v}/{ref}/{method}']={k:{'mean':value} for k,value in [('SSIM',.9),('PSNR_dB',30),('liver_MAE_HU',10)]}
    summary['reconstruction']['128/full_FBP/Joint']['PSNR_dB']['mean']=28
    assert reconstruction_eligibility(summary,cfg)['eligible']
    both=replace(cfg,selection_guard_references=('original_CT','full_FBP'))
    assert not reconstruction_eligibility(summary,both)['eligible']


def test_reconstruction_only_selection_needs_no_segmentation_score():
    from src.evaluate import validation_score
    cfg=Config(segmentation_weight=0)
    summary={'reconstruction':{}}
    for v in cfg.views:
        summary['reconstruction'][f'{v}/original_CT/Joint']={'liver_MAE_HU':{'mean':10}}
        summary['reconstruction'][f'{v}/full_FBP/Joint']={'SSIM':{'mean':.9}}
    assert 0 < validation_score(summary,cfg) < 1
