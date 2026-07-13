from __future__ import annotations

import os
import inspect

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


from fastwam.models.wan22.fastwam import FastWAM  # noqa: E402
from fastwam.models.wan22.mot import MoT  # noqa: E402
from fastwam.models.warm.video_adapter import (  # noqa: E402
    ResidualVideoLayerAdapter,
    resolve_video_adapter_layers,
)
from tests.test_warm_retrospection_model_torch import (  # noqa: E402
    _model,
    _source_context,
)


def test_video_only_loss_never_constructs_action_expert_tokens() -> None:
    source = inspect.getsource(FastWAM.training_loss_video_only)
    assert "action_expert" not in source
    assert "training_loss_action_only" not in source


def test_video_adapter_is_exact_identity_at_initialization_and_trainable() -> None:
    adapter = ResidualVideoLayerAdapter(hidden_dim=8, rank=2)
    tokens = torch.randn(2, 5, 8)

    output = adapter(tokens)

    assert torch.equal(output, tokens)
    output.square().mean().backward()
    assert adapter.up.weight.grad is not None
    assert torch.count_nonzero(adapter.up.weight.grad).item() > 0


def test_auto_video_adapter_layers_match_world_feature_taps() -> None:
    assert resolve_video_adapter_layers(30, ()) == (9, 19)
    assert resolve_video_adapter_layers(30, (4, 12)) == (4, 12)
    with pytest.raises(ValueError, match="outside"):
        resolve_video_adapter_layers(30, (30,))


def test_mot_adapter_hook_is_selected_layer_only() -> None:
    adapters = torch.nn.ModuleDict(
        {"1": ResidualVideoLayerAdapter(hidden_dim=4, rank=1)}
    )
    with torch.no_grad():
        adapters["1"].up.weight.fill_(0.5)
        adapters["1"].down.weight.zero_()
        adapters["1"].down.weight[0, 0] = 1.0
    tokens = torch.tensor([[[0.0, 1.0, 2.0, 4.0], [4.0, 2.0, 1.0, 0.0]]])

    untouched = MoT._apply_optional_layer_adapter(
        layer_adapters=adapters, layer_idx=0, tokens=tokens
    )
    adapted = MoT._apply_optional_layer_adapter(
        layer_adapters=adapters, layer_idx=1, tokens=tokens
    )

    assert torch.equal(untouched, tokens)
    assert adapted.shape == tokens.shape
    assert not torch.equal(adapted, tokens)


def test_complete_warm_sums_disjoint_action_and_video_losses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    model.loss_lambda_video = 1.0
    context = _source_context(with_teachers=True)
    monkeypatch.setattr(model, "_training_source_context", lambda _sample: context)

    action = torch.tensor(2.0, requires_grad=True)
    video = torch.tensor(3.0, requires_grad=True)

    def action_only(_self, _sample, **_kwargs):
        return action, {"loss_action": 2.0}

    def video_only(_self, _sample, **_kwargs):
        return video, {"loss_video": 3.0}

    monkeypatch.setattr(FastWAM, "training_loss_action_only", action_only)
    monkeypatch.setattr(FastWAM, "training_loss_video_only", video_only)

    loss, metrics = model.training_loss({})

    assert float(loss.detach()) == pytest.approx(5.0)
    assert metrics == {"loss_action": 2.0, "loss_video": 3.0}
