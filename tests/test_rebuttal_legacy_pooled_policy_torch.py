"""Legacy control execution contracts on small fixtures, never task evidence."""
from dataclasses import replace
import dataclasses
import os
import pytest

torch=pytest.importorskip('torch')
from fastwam.models.warm.retrospection_model import WARM_RETROSPECTION_CHECKPOINT_VERSION

if WARM_RETROSPECTION_CHECKPOINT_VERSION != 3:
    if os.environ.get('WARM_REQUIRE_LEGACY_TESTS')=='1':
        raise RuntimeError('Bind the original frozen v3 source before the legacy tests')
    pytest.skip('Legacy full-policy tests require the separately frozen original source',allow_module_level=True)

from fastwam.research.legacy_pooled_memory_prior import (
    LegacyPooledMemoryEvent,LegacyConstantGate,legacy_pooled_model_class,legacy_api,SCHEMA,
)
from tests import test_warm_retrospection_model_torch as fixture
from tests.test_warm_source_model_torch import _TinyVideoExpert,_TinyActionExpert,_TinyMoT,_TinyVAE,_run_contract,TEXT_DIM


def control():
    cls=legacy_pooled_model_class();v=_TinyVideoExpert();a=_TinyActionExpert()
    model=cls(video_expert=v,action_expert=a,mot=_TinyMoT(v,a),vae=_TinyVAE(),
        text_dim=TEXT_DIM,proprio_dim=fixture.PROPRIO_DIM,device='cpu',torch_dtype=torch.float32,
        video_train_shift=1.,video_infer_shift=1.,action_train_shift=1.,action_infer_shift=1.)
    model.configure_warm_source(policy='fixed_context_top1',memory_sigma=.2,run_contract=_run_contract())
    model.configure_warm_retrospection(replace(fixture._config(),loss_retrieval=0,loss_effect=0,loss_gate=0,loss_adaptation=0))
    return model


def context(*,with_teachers):
    # The old generic unit fixture has two spatial cells. This control's real
    # bank contract is four; expand the synthetic fixture only, never a cache.
    ctx=fixture._source_context(with_teachers=with_teachers)
    return replace(ctx,candidate_effect_pre=ctx.candidate_effect_pre.repeat(1,1,2,1),
                   candidate_effect_delta=ctx.candidate_effect_delta.repeat(1,1,2,1))


def test_common_initialization_and_rng_are_identical_to_original_full():
    torch.manual_seed(3407);full=fixture._model();rng=torch.get_rng_state()
    torch.manual_seed(3407);model=control()
    assert torch.equal(rng,torch.get_rng_state())
    for name in ['semantic_bridge','retrospective_gist','episode_action_projection','gist_to_text','action_context_to_text','action_expert']:
        assert all(torch.equal(a,b) for a,b in zip(getattr(full,name).parameters(),getattr(model,name).parameters(),strict=True))
    assert not tuple(model.utility_reranker.parameters()) and not tuple(model.source_confidence_gate.parameters())


def test_complete_source_keeps_history_tokens_and_exact_null_rows():
    model=control();torch.manual_seed(8)
    gaussian,out=fixture._resolve(model,context(with_teachers=False),phase='infer')
    assert out.source_gate.tolist()==[.25,0.]
    assert out.conditioning_tokens.shape==(2,12,TEXT_DIM)
    assert torch.equal(gaussian[1],out.source[1])
    assert torch.count_nonzero(out.conditioning_tokens[1,8:])==0
    assert out.component_indices.tolist()==[1,0]
    assert model._last_retrospection_diagnostics['factual_vae_latent'].shape==(2,2,4,8)


def test_complete_training_route_backpropagates_into_pool_and_gist():
    model=control();model.configure_trainable_modules();ctx=context(with_teachers=True)
    _,out=fixture._resolve(model,ctx,phase='train')
    assert torch.isfinite(out.auxiliary_loss)
    loss=out.conditioning_tokens.square().mean()+out.auxiliary_loss
    loss.backward()
    assert model.retrospective_event_adapter.record_projection[0].weight.grad.abs().sum()>0
    assert model.retrospective_gist.future_projection.weight.grad.abs().sum()>0
    assert all(out.auxiliary_metrics['loss_warm_'+k]==0 for k in ['retrieval','effect','gate','adaptation'])


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_autocast_float32_video_taps_with_bfloat16_context_keep_training_finite(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA AMP boundary requires a GPU')
    model=control().to(device=device,dtype=torch.bfloat16);model.torch_dtype=torch.bfloat16;model.device=torch.device(device)
    model.configure_trainable_modules();ctx=context(with_teachers=True)
    ctx=replace(ctx,**{f.name:getattr(ctx,f.name).to(device=device,dtype=torch.bfloat16 if getattr(ctx,f.name).is_floating_point() else getattr(ctx,f.name).dtype)
        for f in dataclasses.fields(ctx) if torch.is_tensor(getattr(ctx,f.name))})
    rand=lambda *shape:torch.randn(*shape,dtype=torch.bfloat16,device=device)
    # A video residual can remain fp32 under autocast even though the model,
    # factual history, text and Gaussian action path are explicitly bf16.
    with torch.autocast(device_type=device,dtype=torch.bfloat16):
        out=model._resolve_action_source(base_gaussian=rand(2,fixture.ACTION_HORIZON,fixture.ACTION_DIM),
            action_source_context=ctx,memory_sigma=.2,phase='train',final_video_tokens=None,
            world_token_streams=(torch.randn(2,5,TEXT_DIM,device=device),torch.randn(2,5,TEXT_DIM,device=device)),
            text_context=rand(2,3,TEXT_DIM),text_context_mask=torch.ones(2,3,dtype=torch.bool,device=device),
            current_proprio=rand(2,fixture.PROPRIO_DIM),current_video_latent=rand(2,2,1,2,2))
        loss=out.conditioning_tokens.float().square().mean()+out.auxiliary_loss
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(out.source).all()
    assert model.retrospective_event_adapter.record_projection[0].weight.grad.abs().sum()>0


def test_teacher_perturbations_do_not_change_decision_inputs():
    model=control();ctx=context(with_teachers=True)
    modified=replace(ctx,target_action=ctx.target_action+100,target_effect=ctx.target_effect+100,
                     current_semantic_teacher=ctx.current_semantic_teacher+100)
    torch.manual_seed(9);_,a=fixture._resolve(model,ctx,phase='train')
    torch.manual_seed(9);_,b=fixture._resolve(model,modified,phase='train')
    assert torch.equal(a.source,b.source) and torch.equal(a.conditioning_tokens,b.conditioning_tokens)


@pytest.mark.parametrize('field',['candidate_actions','candidate_effect_pre','candidate_effect_delta','candidate_context','candidate_timing'])
def test_nonselected_record_is_visible_to_the_complete_conditioning_path(field):
    model=control();ctx=context(with_teachers=False)
    # Make two valid events so changing slot1 cannot be hidden behind coarse top1.
    mask=ctx.candidate_valid_mask.clone();mask[0,1]=True
    ctx=replace(ctx,candidate_valid_mask=mask)
    changed=getattr(ctx,field).clone();changed[0,1]+=1.0
    other=replace(ctx,**{field:changed})
    torch.manual_seed(10);_,a=fixture._resolve(model,ctx,phase='infer')
    torch.manual_seed(10);_,b=fixture._resolve(model,other,phase='infer')
    assert torch.equal(a.source,b.source)
    assert not torch.equal(a.conditioning_tokens[0,8:],b.conditioning_tokens[0,8:])


def test_fact_history_changes_conditioning_without_teacher_or_extra_phase_fields():
    model=control();ctx=context(with_teachers=False)
    history=ctx.episode_tokens.clone();history[...,0]+=1
    changed=replace(ctx,episode_tokens=history)
    torch.manual_seed(11);_,a=fixture._resolve(model,ctx,phase='infer')
    torch.manual_seed(11);_,b=fixture._resolve(model,changed,phase='infer')
    assert not torch.equal(a.conditioning_tokens[0],b.conditioning_tokens[0])


def test_checkpoint_schema_isolation_and_legacy_metadata_roundtrip():
    model=control();full=fixture._model();state=model._retrospection_state()
    assert state['schema']==SCHEMA
    model._validate_retrospection_state(state)
    model.validate_trainer_state_metadata(model.trainer_state_metadata())
    with pytest.raises(ValueError):model._validate_retrospection_state(full._retrospection_state())
    with pytest.raises(ValueError):full._validate_retrospection_state(state)
    malformed=dict(state,source_weight=.5)
    with pytest.raises(ValueError):model._validate_retrospection_state(malformed)


def test_declared_recipe_cannot_retain_losses_for_removed_heads():
    m=control()
    with pytest.raises(ValueError,match='zero losses'):
        m.configure_warm_retrospection(fixture._config())


def test_inference_does_not_fall_through_to_the_original_full_policy():
    m=control();m.configure_online_experiment(ablation_mode='context_only',memory_corruption='clean',experiment_id='invalid-for-control')
    with pytest.raises(ValueError,match='Full/clean'):
        fixture._resolve(m,context(with_teachers=False),phase='infer')


def test_factory_binds_the_real_legacy_class_and_preserves_runtime_contract(monkeypatch):
    from fastwam.research.legacy_pooled_memory_prior import create_legacy_pooled_memory_prior
    from fastwam import runtime
    captured={}
    def build(**kwargs):captured.update(kwargs);return 'constructed'
    monkeypatch.setattr(runtime,'create_warm_source',build)
    cfg=replace(fixture._config(),loss_retrieval=0,loss_effect=0,loss_gate=0,loss_adaptation=0)
    assert create_legacy_pooled_memory_prior(retrospection=cfg,model_id='local-only',run_contract_path='frozen-contract')=='constructed'
    assert captured['_warm_model_class'] is legacy_pooled_model_class()
    assert captured['_warm_pretrained_extra']=={'warm_retrospection_config':cfg}
    assert captured['run_contract_path']=='frozen-contract'
    assert captured['source_policy']=='fixed_context_top1'
    with pytest.raises(ValueError,match='retrieval family'):
        create_legacy_pooled_memory_prior(retrospection=cfg,source_policy='gaussian_null')
