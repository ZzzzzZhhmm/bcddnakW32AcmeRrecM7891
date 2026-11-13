"""Information coverage and numerical isolation; no robotics efficacy claims."""
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from fastwam.research.pooled_memory_prior import (
    POOLED_SCHEMA, PooledMemoryEvent, PooledMemoryPriorFastWAM,
)
from fastwam.research.simple_prior import ConditionedPriorFastWAM, SimplePriorFastWAM
from tests import test_warm_retrospection_model_torch as fixture
from tests.test_rebuttal_simple_prior_torch import control


def inputs(model):
    ctx = fixture._source_context(with_teachers=False)
    b, k = ctx.candidate_valid_mask.shape
    return dict(warped_actions=ctx.candidate_actions, candidate_valid_mask=ctx.candidate_valid_mask,
                event_pre_tokens=ctx.candidate_effect_pre, event_delta_tokens=ctx.candidate_effect_delta,
                event_delta_mask=ctx.candidate_valid_mask[...,None].expand(b,k,4).clone(),
                candidate_context=ctx.candidate_context,
                candidate_phase_features=torch.zeros(b,k,4), gripper_timing=ctx.candidate_timing,
                gist_tokens=torch.randn(b,8,model._require_retrospection().gist_dim),
                query_context=ctx.query_context, proprio=torch.randn(b,fixture.PROPRIO_DIM))


@pytest.mark.parametrize("field", ["warped_actions", "event_pre_tokens", "event_delta_tokens",
                                  "candidate_context", "gripper_timing", "candidate_phase_features"])
def test_every_nonselected_valid_record_field_affects_pooled_conditioning(field):
    torch.manual_seed(41)
    model = control(PooledMemoryPriorFastWAM)
    adapter = model.retrospective_event_adapter
    original = inputs(model)
    before = adapter(**original).action_context_tokens[0,0]
    changed = dict(original)
    changed[field] = original[field].clone()
    changed[field][0,1].flatten()[0] += 5.0  # Candidate 1 is not the fixed coarse prior.
    after = adapter(**changed).action_context_tokens[0,0]
    assert not torch.equal(before, after)
    differentiable = dict(original)
    differentiable[field] = original[field].clone().requires_grad_(True)
    tokens = adapter(**differentiable).action_context_tokens[0,0]
    (tokens * torch.randn_like(tokens)).sum().backward()
    assert bool(differentiable[field].grad[0,1].abs().sum() > 0)
    assert bool(adapter.record_projection[0].weight.grad.abs().sum() > 0)


def test_padding_and_missing_spatial_tokens_are_masked_and_empty_rows_stay_finite_zero():
    model = control(PooledMemoryPriorFastWAM)
    adapter = model.retrospective_event_adapter
    original = inputs(model)
    original["event_delta_mask"][0,1,3] = False
    before = adapter(**original)
    changed = {k: v.clone() for k,v in original.items()}
    for field in ("warped_actions", "event_pre_tokens", "event_delta_tokens",
                  "candidate_context", "gripper_timing", "candidate_phase_features"):
        changed[field][1] = float("nan")
    for field in ("event_pre_tokens", "event_delta_tokens"):
        changed[field][0,1,3] = float("nan")
    after = adapter(**changed)
    assert torch.equal(before.action_context_tokens, after.action_context_tokens)
    assert torch.isfinite(after.action_context_tokens).all()
    assert torch.count_nonzero(after.action_context_tokens[1]) == 0
    assert torch.count_nonzero(after.adapted_action_mean[1]) == 0
    assert torch.count_nonzero(after.action_residual) == 0
    assert torch.count_nonzero(after.predicted_effect) == 0


def test_invalid_active_values_and_incompatible_masks_fail_before_pooling():
    model = control(PooledMemoryPriorFastWAM)
    adapter = model.retrospective_event_adapter
    original = inputs(model)
    broken = dict(original, event_delta_mask=original["event_delta_mask"].float())
    with pytest.raises(ValueError, match="boolean"): adapter(**broken)
    broken = dict(original, event_pre_tokens=original["event_pre_tokens"].clone())
    broken["event_pre_tokens"][0,1,0,0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"): adapter(**broken)


def test_full_source_only_control_receives_records_but_never_future_teachers():
    torch.manual_seed(78)
    model = control(PooledMemoryPriorFastWAM)
    model.configure_trainable_modules()
    ctx = fixture._source_context(with_teachers=True)
    nn = torch.nn
    nn.init.normal_(model.action_context_to_text.weight, std=.02)
    changed = replace(ctx, target_effect=ctx.target_effect + 999,
                      target_action=ctx.target_action + 999,
                      current_semantic_teacher=ctx.current_semantic_teacher + 999)
    torch.manual_seed(90)
    _, first = fixture._resolve(model, ctx, phase="train")
    torch.manual_seed(90)
    _, second = fixture._resolve(model, changed, phase="train")
    assert torch.equal(first.conditioning_tokens, second.conditioning_tokens)
    assert first.conditioning_tokens.shape == (2,12,fixture.TEXT_DIM)
    assert first.source_gate.tolist() == [.25,0.]
    assert torch.count_nonzero(first.conditioning_tokens[0,8:]) > 0
    first.conditioning_tokens.square().sum().backward()
    assert bool(model.retrospective_event_adapter.record_projection[0].weight.grad.abs().sum() > 0)
    assert not tuple(model.utility_reranker.parameters())
    assert not tuple(model.source_confidence_gate.parameters())


def test_pool_initialization_preserves_common_weights_rng_and_checkpoint_isolation():
    torch.manual_seed(3407)
    conditioned = control(ConditionedPriorFastWAM)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(3407)
    pooled = control(PooledMemoryPriorFastWAM)
    assert torch.equal(expected_rng, torch.get_rng_state())
    for name in ("semantic_bridge", "retrospective_gist", "action_expert", "gist_to_text"):
        assert all(torch.equal(a,b) for a,b in zip(getattr(conditioned,name).parameters(),
                                                 getattr(pooled,name).parameters(),strict=True))
    state = pooled._retrospection_state()
    assert state["schema"] == POOLED_SCHEMA
    pooled._validate_retrospection_state(state)
    pooled._preflight_checkpoint_extra_state(pooled._checkpoint_extra_state())
    for other in (conditioned, control(SimplePriorFastWAM), fixture._model()):
        with pytest.raises(ValueError): other._validate_retrospection_state(state)
        with pytest.raises(ValueError): pooled._validate_retrospection_state(other._retrospection_state())


def test_phase_free_module_uses_only_legacy_fields_and_isolates_checkpoints():
    model = control(PooledMemoryPriorFastWAM)
    phase_aware = model.retrospective_event_adapter
    phase_free = PooledMemoryEvent(phase_aware.config,
        context_dim=model._require_retrospection().context_dim, phase_feature_dim=0)
    original = inputs(model)
    legacy = dict(original)
    del legacy["candidate_phase_features"]
    # The original phase-aware contract stays strict, rather than accepting
    # invented zeros in order to pass a legacy-data check.
    with pytest.raises(ValueError, match="phase"):
        phase_aware(**legacy)
    result = phase_free(**legacy)
    assert torch.isfinite(result.action_context_tokens).all()
    assert phase_free.record_projection[0].in_features == phase_aware.record_projection[0].in_features - 4
    with pytest.raises(ValueError, match="phase-free contract"):
        phase_free(**original)
    # A failed torch load may already copy same-shape tensors. Use a fresh
    # module for the rejection test and never resume a graph after that load.
    rejected = PooledMemoryEvent(phase_aware.config,
        context_dim=model._require_retrospection().context_dim, phase_feature_dim=0)
    with pytest.raises(RuntimeError, match="size mismatch"):
        rejected.load_state_dict(phase_aware.state_dict())
    (result.action_context_tokens * torch.randn_like(result.action_context_tokens)).sum().backward()
    assert bool(phase_free.record_projection[0].weight.grad.abs().sum() > 0)


def test_explicit_default_phase_contract_preserves_weights_and_rng():
    model = control(PooledMemoryPriorFastWAM)
    cfg = model.retrospective_event_adapter.config
    dim = model._require_retrospection().context_dim
    torch.manual_seed(3407)
    default = PooledMemoryEvent(cfg, context_dim=dim)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(3407)
    explicit = PooledMemoryEvent(cfg, context_dim=dim, phase_feature_dim=4)
    assert torch.equal(expected_rng, torch.get_rng_state())
    assert default.state_dict().keys() == explicit.state_dict().keys()
    assert all(torch.equal(a, explicit.state_dict()[k]) for k,a in default.state_dict().items())
