from __future__ import annotations

from dataclasses import replace
import os

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
from fastwam.models.warm.retrospection_config import (  # noqa: E402
    WarmRetrospectionConfig,
)
from fastwam.models.warm.retrospection_model import (  # noqa: E402
    RetrospectiveSourceContext,
    WarmRetrospectionFastWAM,
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
    assert model.action_expert.action_encoder.weight.requires_grad
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
