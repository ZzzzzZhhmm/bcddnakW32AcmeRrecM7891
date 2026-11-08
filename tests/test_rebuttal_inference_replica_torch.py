"""Execution-equivalence checks; native large-model qualification is separate."""
import pytest
from types import SimpleNamespace,MappingProxyType

torch=pytest.importorskip('torch')
from fastwam.research.inference_replica import shared_weight_inference_replica,assert_shared_weights_unchanged
from tests.test_warm_retrospection_model_torch import _model,_source_context,_resolve


def test_replica_shares_only_weight_storage_and_preserves_cold_state():
    model=_model().eval();model.diagnostic={'values':[1]}
    replica,receipt=shared_weight_inference_replica(model)
    assert replica is not model and replica.action_expert is not model.action_expert
    assert next(model.parameters()) is next(replica.parameters())
    replica.diagnostic['values'].append(2)
    assert model.diagnostic=={'values':[1]}
    assert_shared_weights_unchanged(receipt)


def test_replica_does_not_copy_or_share_a_deployed_retriever_capability():
    model=_model().eval()
    retriever=SimpleNamespace(bank=MappingProxyType({'payload':'immutable'}),consumed=set())
    model._warm_online_retriever=retriever
    replica,receipt=shared_weight_inference_replica(model)
    assert model._warm_online_retriever is retriever
    assert replica._warm_online_retriever is None
    assert_shared_weights_unchanged(receipt)


def test_replica_matches_an_independent_null_source_and_conditioning():
    torch.manual_seed(3407);model=_model().eval()
    torch.manual_seed(3407);independent=_model().eval()
    replica,receipt=shared_weight_inference_replica(model)
    replica.configure_research_probe(capture=True,force_null=True)
    independent.configure_research_probe(capture=True,force_null=True)
    with torch.inference_mode():
        torch.manual_seed(3410)
        noise1,first=_resolve(replica,_source_context(with_teachers=False),phase='infer')
        torch.manual_seed(3410)
        noise2,second=_resolve(independent,_source_context(with_teachers=False),phase='infer')
    assert torch.equal(noise1,noise2)
    for name in ('source','conditioning_tokens','source_gate'):
        assert torch.equal(getattr(first,name),getattr(second,name))
    assert getattr(model,'_warm_online_experiment_locked',False) is False
    assert_shared_weights_unchanged(receipt)


def test_rejects_training_used_or_gradient_bearing_models_and_detects_writes():
    model=_model()
    with pytest.raises(ValueError,match='eval'):shared_weight_inference_replica(model)
    model.eval();model._warm_online_experiment_locked=True
    with pytest.raises(ValueError,match='before first'):shared_weight_inference_replica(model)
    model._warm_online_experiment_locked=False
    parameter=next(model.parameters());parameter.grad=torch.ones_like(parameter)
    with pytest.raises(ValueError,match='gradients'):shared_weight_inference_replica(model)
    parameter.grad=None
    _,receipt=shared_weight_inference_replica(model)
    with torch.no_grad():parameter.add_(1)
    with pytest.raises(RuntimeError,match='wrote shared'):assert_shared_weights_unchanged(receipt)
