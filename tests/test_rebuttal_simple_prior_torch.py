"""Trainable-control behavior on small fixtures, not robotics efficacy."""
from dataclasses import replace
import pytest
torch=pytest.importorskip('torch')

from fastwam.research.simple_prior import SimplePriorFastWAM,SCHEMA
from fastwam.models.warm.retrospection_model import WarmRetrospectionFastWAM
from tests import test_warm_retrospection_model_torch as fixture
from tests.test_warm_source_model_torch import (
    _TinyVideoExpert,_TinyActionExpert,_TinyMoT,_TinyVAE,_run_contract,TEXT_DIM)


def control():
    video=_TinyVideoExpert();action=_TinyActionExpert()
    model=SimplePriorFastWAM(video_expert=video,action_expert=action,mot=_TinyMoT(video,action),
        vae=_TinyVAE(),text_dim=TEXT_DIM,proprio_dim=fixture.PROPRIO_DIM,device='cpu',torch_dtype=torch.float32,
        video_train_shift=1.,video_infer_shift=1.,action_train_shift=1.,action_infer_shift=1.)
    model.configure_warm_source(policy='fixed_context_top1',memory_sigma=.2,run_contract=_run_contract())
    model.configure_warm_retrospection(replace(fixture._config(),loss_retrieval=0,loss_effect=0,loss_gate=0,loss_adaptation=0))
    return model


def test_control_removes_learned_decision_modules_and_keeps_common_initialization():
    torch.manual_seed(3407);full=fixture._model()
    torch.manual_seed(3407);simple=control()
    for name in ('retrospective_event_adapter','utility_reranker','source_confidence_gate'):
        assert sum(p.numel() for p in getattr(simple,name).parameters())==0
    for name in ('semantic_bridge','retrospective_gist','action_expert','gist_to_text'):
        assert all(torch.equal(a,b) for a,b in zip(getattr(full,name).parameters(),getattr(simple,name).parameters(),strict=True))


def test_control_is_a_training_recipe_with_factual_history_conditioning():
    simple=control();simple.configure_trainable_modules()
    simple.configure_online_experiment(ablation_mode='context_only',memory_corruption='clean',experiment_id='fixture')
    _,output=fixture._resolve(simple,fixture._source_context(with_teachers=True),phase='train')
    assert output.source_gate.tolist()==[.25,0.]
    assert output.conditioning_tokens is not None and output.conditioning_tokens.shape==(2,12,TEXT_DIM)
    output.conditioning_tokens.sum().backward()
    assert simple.gist_to_text.weight.grad is not None and bool(simple.gist_to_text.weight.grad.abs().sum()>0)
    assert not any(p.requires_grad for p in simple.action_context_to_text.parameters())


def test_control_cannot_be_loaded_as_full_or_accept_a_full_checkpoint():
    full=fixture._model();simple=control()
    assert simple._retrospection_state()['schema']==SCHEMA
    with pytest.raises(ValueError,match='schema'):full._validate_retrospection_state(simple._retrospection_state())
    with pytest.raises(ValueError,match='coarse-prior'):simple._validate_retrospection_state(full._retrospection_state())
    simple._validate_retrospection_state(simple._retrospection_state())
