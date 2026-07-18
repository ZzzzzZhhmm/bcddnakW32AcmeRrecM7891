from __future__ import annotations

from dataclasses import replace
import os
from types import SimpleNamespace

import numpy as np
import pytest


if os.environ.get("WARM_REQUIRE_TORCH_TESTS") == "1":
    try:
        import torch
    except ModuleNotFoundError as error:  # pragma: no cover - server contract
        raise RuntimeError(
            "WARM_REQUIRE_TORCH_TESTS=1 but PyTorch is unavailable"
        ) from error
else:
    torch = pytest.importorskip("torch")


import fastwam.runtime as runtime  # noqa: E402
from fastwam.datasets.warm_candidates import (  # noqa: E402
    WARM_CANDIDATE_MASK,
    WARM_CANDIDATE_MU,
)
from fastwam.datasets.warm_retrospective import (  # noqa: E402
    WARM_CANDIDATE_CONTEXT,
    WARM_CANDIDATE_EFFECT_DELTA,
    WARM_CANDIDATE_EFFECT_PRE,
    WARM_CANDIDATE_START_PROPRIO,
    WARM_CANDIDATE_SUPPORT,
    WARM_CANDIDATE_TIMING,
    WARM_CURRENT_CONTEXT,
    WARM_CURRENT_SEMANTIC,
    WARM_EPISODE_ACTION_MASK,
    WARM_EPISODE_ACTION_SUMMARIES,
    WARM_EPISODE_MASK,
    WARM_EPISODE_TOKENS,
    WARM_FUTURE_VALID,
    WARM_TARGET_EFFECT,
)
from fastwam.models.warm.retrospection_config import (  # noqa: E402
    WarmRetrospectionConfig,
)
from fastwam.models.warm.retrospection_model import (  # noqa: E402
    RetrospectiveSourceContext,
    WarmRetrospectionError,
    WarmRetrospectionFastWAM,
    _apply_inference_memory_corruption,
    _smooth_rms,
    _warp_candidate_actions,
    _wrong_event_indices,
)
from tests.test_warm_source_model_torch import (  # noqa: E402
    ACTION_DIM,
    ACTION_HORIZON,
    TEXT_DIM,
    _TinyActionExpert,
    _TinyMoT,
    _TinyVAE,
    _TinyVideoExpert,
    _run_contract,
)


SEMANTIC_DIM = 4
PROPRIO_DIM = 3
CONTEXT_DIM = 2


def test_smooth_rms_is_zero_and_has_finite_zero_gradient_at_origin() -> None:
    residual = torch.zeros(3, 4, 2, requires_grad=True)

    deformation = _smooth_rms(residual, dims=(-1, -2))
    deformation.sum().backward()

    assert torch.equal(deformation, torch.zeros(3))
    assert residual.grad is not None
    assert torch.isfinite(residual.grad).all()
    assert torch.count_nonzero(residual.grad).item() == 0


def test_smooth_rms_matches_ordinary_rms_away_from_origin() -> None:
    residual = torch.tensor([[[3.0, 4.0]]])
    expected = residual.square().mean(dim=(-1, -2)).sqrt()

    actual = _smooth_rms(residual, dims=(-1, -2))

    torch.testing.assert_close(actual, expected, atol=2.0e-6, rtol=1.0e-6)


def _config() -> WarmRetrospectionConfig:
    return WarmRetrospectionConfig(
        context_dim=CONTEXT_DIM,
        semantic_dim=SEMANTIC_DIM,
        video_dim=TEXT_DIM,
        text_dim=TEXT_DIM,
        proprio_dim=PROPRIO_DIM,
        action_dim=ACTION_DIM,
        action_horizon=ACTION_HORIZON,
        timing_dim=4,
        gist_dim=8,
        event_model_dim=8,
        bridge_heads=2,
        gist_heads=2,
        event_heads=2,
        reranker_hidden_dim=8,
        gate_hidden_dim=4,
    )


def _model() -> WarmRetrospectionFastWAM:
    video = _TinyVideoExpert()
    action = _TinyActionExpert()
    model = WarmRetrospectionFastWAM(
        video_expert=video,
        action_expert=action,
        mot=_TinyMoT(video, action),
        vae=_TinyVAE(),
        text_dim=TEXT_DIM,
        proprio_dim=PROPRIO_DIM,
        device="cpu",
        torch_dtype=torch.float32,
        video_train_shift=1.0,
        video_infer_shift=1.0,
        action_train_shift=1.0,
        action_infer_shift=1.0,
    )
    model.configure_warm_source(
        policy="fixed_context_top1",
        memory_sigma=0.2,
        run_contract=_run_contract(),
    )
    model.configure_warm_retrospection(_config())
    return model


def _source_context(*, with_teachers: bool) -> RetrospectiveSourceContext:
    batch, candidates, event_tokens = 2, 2, 2
    valid = torch.tensor([[True, True], [False, False]])
    actions = torch.arange(
        batch * candidates * ACTION_HORIZON * ACTION_DIM,
        dtype=torch.float32,
    ).reshape(batch, candidates, ACTION_HORIZON, ACTION_DIM)
    effect_pre = torch.arange(
        batch * candidates * event_tokens * SEMANTIC_DIM,
        dtype=torch.float32,
    ).reshape(batch, candidates, event_tokens, SEMANTIC_DIM)
    effect_delta = effect_pre * 0.01 + 0.1
    timing = torch.tensor(
        [
            [[0.25, 1.0, 0.75, 1.0], [0.5, 1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    candidate_context = torch.tensor(
        [[[1.0, 0.0], [0.8, 0.2]], [[0.0, 0.0], [0.0, 0.0]]]
    )
    for tensor in (actions, effect_pre, effect_delta, timing, candidate_context):
        tensor[~valid] = 0.0
    common = {
        "query_context": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "candidate_context": candidate_context,
        "candidate_actions": actions,
        "candidate_start_proprio": torch.zeros(
            batch, candidates, PROPRIO_DIM
        ),
        "candidate_effect_pre": effect_pre,
        "candidate_effect_delta": effect_delta,
        "candidate_timing": timing,
        "candidate_support": valid.to(torch.float32),
        "candidate_valid_mask": valid,
        "episode_tokens": torch.randn(batch, 3, SEMANTIC_DIM),
        "episode_mask": torch.tensor(
            [[True, True, False], [True, False, False]]
        ),
        "episode_action_summaries": torch.randn(
            batch, 2, 3 * ACTION_DIM + 4
        ),
        "episode_action_mask": torch.tensor(
            [[True, True], [True, False]]
        ),
        # Explicitly choose memory for row zero and the null component for row
        # one, independently of random initial reranker weights.
        "forced_candidate_indices": torch.tensor([0, -1]),
    }
    if with_teachers:
        common.update(
            {
                "current_semantic_teacher": torch.randn(
                    batch, 4, SEMANTIC_DIM
                ),
                "target_effect": torch.randn(
                    batch, event_tokens, SEMANTIC_DIM
                ),
                "target_action": torch.randn(
                    batch, ACTION_HORIZON, ACTION_DIM
                ),
                "target_action_valid_mask": torch.ones(
                    batch, ACTION_HORIZON, dtype=torch.bool
                ),
                "future_valid_mask": torch.tensor([True, False]),
            }
        )
    return RetrospectiveSourceContext(**common)


def test_start_proprio_warp_shifts_arms_preserves_gripper_and_masks_padding() -> None:
    config = replace(
        _config(),
        proprio_dim=ACTION_DIM,
        canonical_action_mode="start_proprio_delta",
        canonical_gripper_dims=(1,),
    )
    actions = torch.tensor(
        [[[[1.0, 9.0], [2.0, 8.0]], [[50.0, 60.0], [70.0, 80.0]]]]
    )
    starts = torch.tensor([[[2.0, 30.0], [100.0, 200.0]]])
    current = torch.tensor([[5.0, -10.0]])
    valid = torch.tensor([[True, False]])

    warped = _warp_candidate_actions(actions, starts, current, valid, config)

    assert torch.equal(
        warped[0, 0], torch.tensor([[4.0, 9.0], [5.0, 8.0]])
    )
    assert torch.count_nonzero(warped[0, 1]).item() == 0


def test_training_context_carries_factual_candidate_start_proprio() -> None:
    model = _model()
    source = _source_context(with_teachers=True)
    starts = torch.arange(
        2 * 2 * PROPRIO_DIM, dtype=torch.float32
    ).reshape(2, 2, PROPRIO_DIM)
    starts[~source.candidate_valid_mask] = 0.0
    sample = {
        "action": source.target_action,
        "action_is_pad": ~source.target_action_valid_mask,
        "proprio": torch.zeros(2, 1, PROPRIO_DIM),
        "warm_query_split": ["dev", "dev"],
        WARM_CANDIDATE_MU: source.candidate_actions,
        WARM_CANDIDATE_MASK: source.candidate_valid_mask,
        WARM_CANDIDATE_START_PROPRIO: starts,
        WARM_CANDIDATE_CONTEXT: source.candidate_context,
        WARM_CANDIDATE_EFFECT_PRE: source.candidate_effect_pre,
        WARM_CANDIDATE_EFFECT_DELTA: source.candidate_effect_delta,
        WARM_CANDIDATE_TIMING: source.candidate_timing,
        WARM_CANDIDATE_SUPPORT: source.candidate_support,
        WARM_CURRENT_CONTEXT: source.query_context,
        WARM_CURRENT_SEMANTIC: source.current_semantic_teacher,
        WARM_TARGET_EFFECT: source.target_effect,
        WARM_FUTURE_VALID: source.future_valid_mask,
        WARM_EPISODE_TOKENS: source.episode_tokens,
        WARM_EPISODE_MASK: source.episode_mask,
        WARM_EPISODE_ACTION_SUMMARIES: source.episode_action_summaries,
        WARM_EPISODE_ACTION_MASK: source.episode_action_mask,
    }

    context = model._training_source_context(sample)

    assert torch.equal(context.candidate_start_proprio, starts)


def test_online_context_carries_factual_candidate_start_proprio() -> None:
    model = _model()
    valid = np.asarray([True, False], dtype=np.bool_)
    starts = np.asarray(
        [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], dtype=np.float32
    )
    online_step = SimpleNamespace(
        context_key=np.zeros((CONTEXT_DIM,), dtype=np.float32),
        candidate_means=np.zeros(
            (2, ACTION_HORIZON, ACTION_DIM), dtype=np.float32
        ),
    )
    facts = SimpleNamespace(
        candidate_valid_mask=valid,
        context_keys=np.zeros((2, CONTEXT_DIM), dtype=np.float32),
        effect_pre=np.zeros((2, 4, SEMANTIC_DIM), dtype=np.float32),
        effect_delta=np.zeros((2, 4, SEMANTIC_DIM), dtype=np.float32),
        start_proprio=starts,
        gripper_timing=np.zeros((2, 4), dtype=np.float32),
        support=valid.astype(np.float32),
    )

    context = model._context_from_online_facts(
        online_step=online_step,
        facts=facts,
        episode_tokens=None,
        episode_mask=None,
        episode_action_summaries=None,
        episode_action_mask=None,
    )

    assert torch.equal(
        context.candidate_start_proprio,
        torch.from_numpy(starts).unsqueeze(0),
    )


def _resolve(model, context, *, phase: str):
    batch = 2
    gaussian = torch.randn(batch, ACTION_HORIZON, ACTION_DIM)
    output = model._resolve_action_source(
        base_gaussian=gaussian,
        action_source_context=context,
        memory_sigma=0.2,
        phase=phase,
        final_video_tokens=torch.randn(batch, 5, TEXT_DIM),
        world_token_streams=(
            torch.randn(batch, 5, TEXT_DIM),
            torch.randn(batch, 5, TEXT_DIM),
        ),
        text_context=torch.randn(batch, 3, TEXT_DIM),
        text_context_mask=torch.tensor(
            [[True, True, False], [True, True, True]]
        ),
        current_proprio=torch.randn(batch, PROPRIO_DIM),
        current_video_latent=torch.randn(batch, 2, 1, 2, 2),
    )
    return gaussian, output


def test_full_source_path_preserves_shapes_and_exact_null_fallback() -> None:
    model = _model()
    gaussian, output = _resolve(
        model, _source_context(with_teachers=False), phase="infer"
    )

    assert output.source.shape == (2, ACTION_HORIZON, ACTION_DIM)
    assert output.selected_means.shape == output.source.shape
    assert output.conditioning_tokens.shape == (2, 12, TEXT_DIM)
    assert output.source_gate.shape == (2,)
    assert output.component_indices.tolist() == [1, 0]
    assert output.memory_mask.tolist() == [True, False]
    assert 0.0 < float(output.source_gate[0]) < 1.0
    assert float(output.source_gate[1]) == 0.0
    assert torch.equal(output.source[1], gaussian[1])
    assert torch.count_nonzero(output.selected_means[1]).item() == 0
    assert output.memory_sigma == pytest.approx(0.2)
    assert model._last_retrospection_diagnostics[
        "factual_vae_latent"
    ].shape == (2, 2, 4, 8)


def test_full_training_source_exposes_finite_auxiliary_losses() -> None:
    model = _model()
    _, output = _resolve(
        model, _source_context(with_teachers=True), phase="train"
    )

    assert output.auxiliary_loss.ndim == 0
    assert torch.isfinite(output.auxiliary_loss)
    assert set(output.auxiliary_metrics) == {
        "loss_warm_retrieval",
        "loss_warm_bridge",
        "loss_warm_gist",
        "loss_warm_effect",
        "loss_warm_gate",
        "loss_warm_adaptation",
        "warm_gate_mean",
        "warm_selected_memory_rate",
        "warm_consequence_mean",
    }
    assert all(
        bool(torch.isfinite(value).item())
        for value in output.auxiliary_metrics.values()
    )


def test_full_auxiliary_backward_has_only_finite_gradients() -> None:
    model = _model()
    model.configure_trainable_modules()
    final_gate = model.source_confidence_gate.network[-1]
    assert isinstance(final_gate, torch.nn.Linear)
    # The production gate starts with a zero final weight.  Exercise the first
    # state after that weight has learned, when gradients reach the exactly
    # zero-initialized event residual through the deformation-norm feature.
    with torch.no_grad():
        final_gate.weight.fill_(0.1)

    _, output = _resolve(
        model, _source_context(with_teachers=True), phase="train"
    )
    output.auxiliary_loss.backward()

    gradients = [
        gradient
        for parameter in model.parameters()
        if parameter.requires_grad
        for gradient in [parameter.grad]
        if gradient is not None
    ]
    assert gradients
    assert all(bool(torch.isfinite(gradient).all().item()) for gradient in gradients)


def test_online_experiment_controls_are_closed_and_lock_after_inference() -> None:
    model = _model()
    with pytest.raises(WarmRetrospectionError, match="ablation_mode"):
        model.configure_online_experiment(
            ablation_mode="unknown",
            memory_corruption="clean",
            experiment_id="bad",
        )
    with pytest.raises(WarmRetrospectionError, match="memory_corruption"):
        model.configure_online_experiment(
            ablation_mode="full",
            memory_corruption="random",
            experiment_id="bad",
        )
    with pytest.raises(WarmRetrospectionError, match="experiment_id"):
        model.configure_online_experiment(
            ablation_mode="full",
            memory_corruption="clean",
            experiment_id="../escape",
        )

    model.configure_online_experiment(
        ablation_mode="context_only",
        memory_corruption="clean",
        experiment_id="context_only",
    )
    _resolve(model, _source_context(with_teachers=False), phase="infer")
    # Exact repetition is safe for policy setup retries.
    model.configure_online_experiment(
        ablation_mode="context_only",
        memory_corruption="clean",
        experiment_id="context_only",
    )
    with pytest.raises(WarmRetrospectionError, match="locked"):
        model.configure_online_experiment(
            ablation_mode="full",
            memory_corruption="clean",
            experiment_id="full_warm",
        )


def test_context_only_keeps_memory_conditioning_but_source_is_exact_gaussian() -> None:
    model = _model().eval()
    model.configure_online_experiment(
        ablation_mode="context_only",
        memory_corruption="clean",
        experiment_id="context_only",
    )

    gaussian, output = _resolve(
        model, _source_context(with_teachers=False), phase="infer"
    )

    assert torch.equal(output.source, gaussian)
    assert output.conditioning_tokens is not None
    assert output.conditioning_tokens.shape == (2, 12, TEXT_DIM)
    assert output.component_indices.tolist() == [0, 0]
    assert output.memory_mask.tolist() == [False, False]
    assert torch.count_nonzero(output.source_gate).item() == 0
    diagnostics = model._last_retrospection_diagnostics
    assert diagnostics["candidate_indices"].tolist() == [0, -1]
    assert diagnostics["ablation_mode"] == "context_only"
    assert diagnostics["experiment_id"] == "context_only"


def test_source_only_uses_source_without_consequence_or_memory_context() -> None:
    model = _model().eval()
    model.configure_online_experiment(
        ablation_mode="source_only_no_consequence",
        memory_corruption="clean",
        experiment_id="source_only_no_consequence",
    )

    gaussian, output = _resolve(
        model, _source_context(with_teachers=False), phase="infer"
    )

    assert output.conditioning_tokens is None
    assert output.component_indices.tolist() == [1, 0]
    assert output.memory_mask.tolist() == [True, False]
    assert not torch.equal(output.source[0], gaussian[0])
    diagnostics = model._last_retrospection_diagnostics
    assert torch.count_nonzero(diagnostics["consistency"]).item() == 0
    assert diagnostics["ablation_mode"] == "source_only_no_consequence"


def test_online_payload_corruptions_are_deterministic_and_keep_padding_zero() -> None:
    actions = torch.arange(1, 17, dtype=torch.float32).reshape(1, 2, 4, 2)
    pre = torch.arange(1, 9, dtype=torch.float32).reshape(1, 2, 2, 2)
    delta = pre * 0.1
    timing = torch.tensor([[[0.25, 1.0, 0.75, 1.0], [9.0, 9.0, 9.0, 9.0]]])
    valid = torch.tensor([[True, False]])

    reversed_payload = _apply_inference_memory_corruption(
        actions=actions,
        effect_pre=pre,
        effect_delta=delta,
        timing=timing,
        valid=valid,
        mode="reversed_action",
    )
    assert torch.equal(reversed_payload.actions[0, 0], actions[0, 0].flip(0))

    shifted = _apply_inference_memory_corruption(
        actions=actions,
        effect_pre=pre,
        effect_delta=delta,
        timing=timing,
        valid=valid,
        mode="phase_shift",
    )
    assert torch.equal(shifted.actions[0, 0], torch.roll(actions[0, 0], 2, 0))
    assert torch.equal(shifted.timing[0, 0], torch.tensor([0.75, 1.0, 0.25, 1.0]))

    mismatched = _apply_inference_memory_corruption(
        actions=actions,
        effect_pre=pre,
        effect_delta=delta,
        timing=timing,
        valid=valid,
        mode="effect_mismatch",
    )
    assert torch.equal(mismatched.effect_delta[0, 0], -delta[0, 0])
    for payload in (reversed_payload, shifted, mismatched):
        assert torch.count_nonzero(payload.actions[0, 1]).item() == 0
        assert torch.count_nonzero(payload.effect_pre[0, 1]).item() == 0
        assert torch.count_nonzero(payload.effect_delta[0, 1]).item() == 0
        assert torch.count_nonzero(payload.timing[0, 1]).item() == 0


def test_wrong_event_selects_distinct_slot_and_null_fallback() -> None:
    selected = torch.tensor([1, 0, -1], dtype=torch.long)
    valid = torch.tensor(
        [[True, True, True], [True, False, False], [False, False, False]]
    )

    forced, fallback = _wrong_event_indices(selected, valid)

    assert forced.tolist() == [0, -1, -1]
    assert fallback.tolist() == [False, True, True]


def test_online_controls_do_not_change_training_source_semantics() -> None:
    model = _model().eval()
    context = _source_context(with_teachers=True)
    torch.manual_seed(777)
    gaussian_full, output_full = _resolve(model, context, phase="train")

    model.configure_online_experiment(
        ablation_mode="context_only",
        memory_corruption="reversed_action",
        experiment_id="training_is_unchanged",
    )
    torch.manual_seed(777)
    gaussian_controlled, output_controlled = _resolve(model, context, phase="train")

    assert torch.equal(gaussian_full, gaussian_controlled)
    assert torch.equal(output_full.source, output_controlled.source)
    assert torch.equal(
        output_full.conditioning_tokens, output_controlled.conditioning_tokens
    )
    diagnostics = model._last_retrospection_diagnostics
    assert diagnostics["ablation_mode"] == "full"
    assert diagnostics["memory_corruption"] == "clean"
    assert diagnostics["configured_ablation_mode"] == "context_only"


def test_null_conditioning_and_required_transition_ignore_candidates() -> None:
    model = _model().eval()
    original = _source_context(with_teachers=False)
    forced_null = torch.full((2,), -1, dtype=torch.long)
    original = replace(original, forced_candidate_indices=forced_null)
    valid4 = original.candidate_valid_mask[:, :, None, None]
    altered = replace(
        original,
        candidate_actions=original.candidate_actions * -7.0,
        candidate_effect_pre=(original.candidate_effect_pre + 13.0) * valid4,
        candidate_effect_delta=original.candidate_effect_delta * -5.0,
    )

    torch.manual_seed(91)
    gaussian_a, output_a = _resolve(model, original, phase="infer")
    required_a = model._last_retrospection_diagnostics[
        "required_transition"
    ].clone()
    torch.manual_seed(91)
    gaussian_b, output_b = _resolve(model, altered, phase="infer")
    required_b = model._last_retrospection_diagnostics[
        "required_transition"
    ].clone()

    assert torch.equal(gaussian_a, gaussian_b)
    assert torch.equal(output_a.source, output_b.source)
    assert torch.equal(output_a.conditioning_tokens, output_b.conditioning_tokens)
    assert torch.equal(required_a, required_b)


def test_trainable_scope_adapts_action_dit_and_action_context_has_gradient() -> None:
    model = _model()
    trainable = model.configure_trainable_modules()

    assert trainable
    assert model.mot.training
    assert model.action_expert.input_proj.weight.requires_grad
    assert all(
        parameter.requires_grad
        for parameter in model.action_expert.parameters()
    )
    assert not next(model.video_expert.parameters()).requires_grad
    inner = model.retrospective_event_adapter.action_context_output_projection
    assert torch.count_nonzero(inner.weight).item() == 0
    assert torch.count_nonzero(model.action_context_to_text.weight).item() > 0

    _, output = _resolve(
        model, _source_context(with_teachers=True), phase="train"
    )
    output.conditioning_tokens.sum().backward()
    assert inner.weight.grad is not None
    assert torch.count_nonzero(inner.weight.grad).item() > 0


def test_forced_hard_negative_after_slot_zero_preserves_prefix_contract() -> None:
    model = _model().eval()
    context = replace(
        _source_context(with_teachers=True),
        forced_candidate_indices=torch.tensor([1, -1], dtype=torch.long),
    )

    _, output = _resolve(model, context, phase="train")

    assert output.component_indices.tolist() == [2, 0]
    assert output.memory_mask.tolist() == [True, False]
    assert torch.isfinite(output.auxiliary_loss)


def test_complete_runtime_factory_forces_full_model_and_closed_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_create_warm_source(**kwargs):
        captured.update(kwargs)
        return "sentinel-model"

    monkeypatch.setattr(runtime, "create_warm_source", fake_create_warm_source)
    result = runtime.create_warm_retrospection(
        retrospection=_config().to_dict(),
        model_id="unused",
        tokenizer_model_id="unused",
        video_dit_config={},
    )

    assert result == "sentinel-model"
    assert captured["source_policy"] == "fixed_context_top1"
    assert captured["_warm_model_class"] is WarmRetrospectionFastWAM
    assert captured["_warm_pretrained_extra"] == {
        "warm_retrospection_config": _config()
    }
    assert captured["model_id"] == "unused"

    with pytest.raises(ValueError, match="fixed_context_top1 only"):
        runtime.create_warm_retrospection(
            retrospection=_config(),
            source_policy="gaussian_null",
        )
