"""Intervention semantics against the real WARM source graph (tiny experts)."""
from dataclasses import replace, fields
import os

import pytest

torch = pytest.importorskip("torch")
from tests.test_warm_retrospection_model_torch import _model, _resolve, _source_context


def resolve(model, context, phase="infer"):
    torch.manual_seed(831)
    return _resolve(model, context, phase=phase)


def twin():
    torch.manual_seed(41)
    return _model().eval()


def test_default_probe_is_exactly_legacy_full():
    a, b = twin(), twin()
    b.configure_research_probe(capture=True)
    ctx = _source_context(with_teachers=False)
    _, x = resolve(a, ctx)
    _, y = resolve(b, ctx)
    assert torch.equal(x.source, y.source)
    assert torch.equal(x.conditioning_tokens, y.conditioning_tokens)
    assert a._last_research_probe is None
    assert b._last_research_probe["evidence_type"] == "I"


def test_forced_null_removes_both_candidate_paths_with_equal_token_shape():
    a, b = twin(), twin()
    for m in (a, b):
        m.configure_research_probe(force_null=True, capture=True)
    ctx = _source_context(with_teachers=False)
    changed = replace(ctx, candidate_actions=ctx.candidate_actions * -3,
                      candidate_effect_delta=ctx.candidate_effect_delta * -9,
                      candidate_effect_pre=ctx.candidate_effect_pre + 6,
                      candidate_context=ctx.candidate_context * -4)
    eps, x = resolve(a, ctx)
    _, y = resolve(b, changed)
    assert torch.equal(x.source, eps)
    assert torch.equal(x.source, y.source)
    assert torch.equal(x.conditioning_tokens, y.conditioning_tokens)
    assert not x.memory_mask.any()
    assert not a._last_research_probe["arrays"]["g"].any()
    assert x.conditioning_tokens.shape == y.conditioning_tokens.shape


def test_gaussian_and_scale_only_preserve_full_context_and_gate():
    models = [twin() for _ in range(3)]
    for m, mode in zip(models, ("full", "gaussian", "scale_only")):
        m.configure_research_probe(source_mode=mode, capture=True)
    ctx = _source_context(with_teachers=False)
    outputs = [resolve(m, ctx) for m in models]
    eps, full = outputs[0]
    _, gauss = outputs[1]
    _, scale = outputs[2]
    assert torch.equal(gauss.source, eps)
    for out in (gauss, scale):
        assert torch.equal(full.conditioning_tokens, out.conditioning_tokens)
    g = models[0]._last_research_probe["arrays"]["g"]
    for m in models[1:]:
        assert torch.equal(g, m._last_research_probe["arrays"]["g"])
    c = 1 - g * (1 - models[0]._require_retrospection().source_sigma_min)
    torch.testing.assert_close(scale.source, c[:, None, None] * eps)
    torch.testing.assert_close(full.source - scale.source, g[:, None, None] * full.selected_means)


def test_no_comparison_zeroes_selection_and_gate_input_but_keeps_predictor():
    m = twin()
    m.configure_research_probe(comparison=False, capture=True)
    received = []
    hook = m.source_confidence_gate.register_forward_pre_hook(lambda _m, args: received.append(args[0]))
    _, out = resolve(m, _source_context(with_teachers=False))
    hook.remove()
    assert not received[0].selected_consistency.any()
    arrays = m._last_research_probe["arrays"]
    assert not arrays["selection_kappa"].any()
    assert arrays["predicted_effect"].abs().sum() > 0
    assert arrays["adapted_actions"].abs().sum() > 0
    assert out.conditioning_tokens is not None


def test_probe_controls_never_change_training_or_checkpoint():
    a, b = twin(), twin()
    b.configure_research_probe(comparison=False, force_null=True, source_mode="scale_only", capture=True)
    ctx = _source_context(with_teachers=True)
    _, x = resolve(a, ctx, "train")
    _, y = resolve(b, ctx, "train")
    assert torch.equal(x.source, y.source)
    assert torch.equal(x.auxiliary_loss, y.auxiliary_loss)
    assert b._last_research_probe is None
    assert a.trainer_state_metadata() == b.trainer_state_metadata()


def test_controls_are_closed_and_immutable_after_first_inference():
    m = twin()
    with pytest.raises(ValueError):
        m.configure_research_probe(force_null="false")
    with pytest.raises(ValueError):
        m.configure_research_probe(source_mode="invented")
    m.configure_research_probe(capture=True)
    resolve(m, _source_context(with_teachers=False))
    m.configure_research_probe(capture=True)
    with pytest.raises(ValueError, match="locked"):
        m.configure_research_probe(force_null=True)


def test_offline_kappa_matches_model_zero_and_small_norm_convention():
    from fastwam.models.warm.consequence import consequence_consistency
    from fastwam.research.statistics import kappa
    effects = torch.tensor([[[0., 0.], [1e-8, -1e-8], [1., -2.]]])
    required = torch.tensor([[1e-8, 1.]])
    expected = consequence_consistency(effects, required, magnitude_weight=.25)
    actual = torch.tensor([[kappa(e, required[0], .25) for e in effects[0]]])
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cuda_forced_null_source_graph(dtype):
    if not torch.cuda.is_available():
        if os.environ.get("WARM_REQUIRE_CUDA_PROBE") == "1":
            pytest.fail("CUDA qualification requested but no CUDA device is available")
        pytest.skip("CUDA-only numerical qualification")
    from tests.test_warm_retrospection_model_torch import ACTION_HORIZON, ACTION_DIM, TEXT_DIM, PROPRIO_DIM
    m = twin().to(device="cuda", dtype=dtype)
    m.device, m.torch_dtype = torch.device("cuda"), dtype
    m.configure_research_probe(force_null=True, capture=True)
    ctx = _source_context(with_teachers=False)
    ctx = replace(ctx, **{f.name: getattr(ctx, f.name).to("cuda", dtype=dtype if getattr(ctx, f.name).is_floating_point() else getattr(ctx, f.name).dtype)
                          for f in fields(ctx) if isinstance(getattr(ctx, f.name), torch.Tensor)})
    gaussian = torch.randn(2, ACTION_HORIZON, ACTION_DIM, device="cuda", dtype=dtype)
    kwargs = dict(base_gaussian=gaussian, memory_sigma=.2, phase="infer", final_video_tokens=None,
                  world_token_streams=tuple(torch.randn(2, 5, TEXT_DIM, device="cuda", dtype=dtype) for _ in range(2)),
                  text_context=torch.randn(2, 3, TEXT_DIM, device="cuda", dtype=dtype),
                  text_context_mask=torch.ones(2, 3, device="cuda", dtype=torch.bool),
                  current_proprio=torch.randn(2, PROPRIO_DIM, device="cuda", dtype=dtype),
                  current_video_latent=torch.randn(2, 2, 1, 2, 2, device="cuda", dtype=dtype))
    a = m._resolve_action_source(action_source_context=ctx, **kwargs)
    identical = m._resolve_action_source(action_source_context=ctx, **kwargs)
    tolerance = (a.conditioning_tokens - identical.conditioning_tokens).abs().max().item()
    changed = replace(ctx, candidate_actions=ctx.candidate_actions * -2,
                      candidate_effect_delta=ctx.candidate_effect_delta * -3)
    b = m._resolve_action_source(action_source_context=changed, **kwargs)
    assert torch.equal(a.source, gaussian) and torch.equal(b.source, gaussian)
    assert torch.isfinite(b.conditioning_tokens).all()
    assert (a.conditioning_tokens - b.conditioning_tokens).abs().max().item() <= tolerance


def test_online_bimanual_timing_is_derived_after_validating_legacy_bank_timing():
    from types import SimpleNamespace
    import numpy as np
    from fastwam.memory.schema import EventId
    from tests.test_warm_retrospection_model_torch import SEMANTIC_DIM, CONTEXT_DIM
    m = twin()
    # Exercise the actual context boundary without allocating full-sized experts.
    cfg = replace(m._require_retrospection(), action_dim=14, proprio_dim=14,
                  timing_dim=8, canonical_action_mode="start_proprio_delta",
                  canonical_gripper_dims=(6, 13))
    m.warm_retrospection_config = cfg
    valid = np.array([True, False])
    actions = np.zeros((2, cfg.action_horizon, 14), dtype=np.float32)
    actions[0, 1:, 6] = -1
    actions[0, 2:, 13] = 1
    step = SimpleNamespace(query_id=SimpleNamespace(episode_index=0, frame_index=0),
                           candidate_valid_mask=valid, event_ids=(EventId("train", 0, 1, 0), None),
                           context_key=np.zeros(CONTEXT_DIM, dtype=np.float32), candidate_means=actions)
    facts = SimpleNamespace(candidate_valid_mask=valid,
                            context_keys=np.zeros((2, CONTEXT_DIM), dtype=np.float32),
                            effect_pre=np.zeros((2, 4, SEMANTIC_DIM), dtype=np.float32),
                            effect_delta=np.zeros((2, 4, SEMANTIC_DIM), dtype=np.float32),
                            start_proprio=np.zeros((2, 14), dtype=np.float32),
                            gripper_timing=np.zeros((2, 4), dtype=np.float32),
                            support=valid.astype(np.float32), normalized_phase=np.zeros(2),
                            event_ordinal=np.zeros(2, dtype=np.int64))
    ctx = m._context_from_online_facts(online_step=step, facts=facts, episode_tokens=None,
            episode_mask=None, episode_action_summaries=None, episode_action_mask=None,
            episode_role_ids=None, episode_relative_age=None, episode_action_relative_age=None)
    assert ctx.candidate_timing.shape == (1, 2, 8)
    assert ctx.candidate_timing[0, 0, 1].item() == 1  # left close present
    assert ctx.candidate_timing[0, 0, 7].item() == 1  # right open present
    assert not ctx.candidate_timing[0, 1].any()
