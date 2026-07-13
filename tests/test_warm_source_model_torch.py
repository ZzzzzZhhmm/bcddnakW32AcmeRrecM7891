from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import MethodType

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


from fastwam.models.wan22.fastwam import FastWAM
from fastwam.models.wan22.mot import MoT, VideoPrefillOutput
from fastwam.models.warm.source_contract import WarmSourceRunContract
from fastwam.models.warm.source_model import WarmSourceFastWAM
from fastwam.models.warm.source_transport import (
    ActionSourceContext,
    SourceTransportError,
)
from fastwam.runtime import create_warm_source
from fastwam.trainer import Wan22Trainer


ACTION_HORIZON = 3
ACTION_DIM = 2
TEXT_DIM = 4


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _run_contract(
    *,
    label: str = "default",
    query_split: str = "train",
    base_checkpoint_sha256: str | None = None,
) -> WarmSourceRunContract:
    return WarmSourceRunContract(
        bank_manifest_sha256=_digest(f"{label}:bank-manifest"),
        bank_content_sha256=_digest(f"{label}:bank-content"),
        candidate_manifest_sha256=_digest(f"{label}:candidate-manifest"),
        query_corpus_sha256=_digest(f"{label}:query-corpus"),
        catalog_sha256=_digest(f"{label}:catalog"),
        audit_sha256=_digest(f"{label}:audit"),
        normalization_stats_sha256=_digest(f"{label}:normalization"),
        action_space_contract_sha256=_digest(f"{label}:action-space"),
        base_checkpoint_sha256=(
            _digest(f"{label}:base-checkpoint")
            if base_checkpoint_sha256 is None
            else base_checkpoint_sha256
        ),
        query_split=query_split,
        global_sample_stride=1,
        action_horizon=ACTION_HORIZON,
        action_dim=ACTION_DIM,
    )


class _TinyVAE(torch.nn.Module):
    temporal_downsample_factor = 1

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def encode(self, video, **_kwargs):
        if not isinstance(video, torch.Tensor):
            raise TypeError("tiny VAE expects a tensor")
        return video.mean(dim=(1, 3, 4), keepdim=True) * self.scale


class _TinyVideoExpert(torch.nn.Module):
    fuse_vae_embedding_in_latents = False

    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(1, TEXT_DIM, bias=False)

    def pre_dit(
        self,
        *,
        x,
        timestep,
        context,
        context_mask,
        action,
        fuse_vae_embedding_in_latents,
    ):
        del timestep, action, fuse_vae_embedding_in_latents
        scalar_tokens = x.mean(dim=(1, 3, 4)).unsqueeze(-1)
        tokens = self.proj(scalar_tokens)
        return {
            "tokens": tokens,
            "freqs": torch.zeros(
                (tokens.shape[1], 1, 1), dtype=tokens.dtype, device=tokens.device
            ),
            "t_mod": torch.zeros(
                (tokens.shape[0], 1), dtype=tokens.dtype, device=tokens.device
            ),
            "context": context,
            "context_mask": context_mask,
            "meta": {"tokens_per_frame": 1},
        }

    @staticmethod
    def build_video_to_video_mask(
        *, video_seq_len, video_tokens_per_frame, device
    ):
        del video_tokens_per_frame
        return torch.ones(
            (video_seq_len, video_seq_len), dtype=torch.bool, device=device
        )


class _TinyActionExpert(torch.nn.Module):
    action_dim = ACTION_DIM

    def __init__(self) -> None:
        super().__init__()
        self.input_proj = torch.nn.Linear(ACTION_DIM, TEXT_DIM)
        self.mix = torch.nn.Linear(TEXT_DIM, TEXT_DIM)
        self.output_proj = torch.nn.Linear(TEXT_DIM, ACTION_DIM)

    def pre_dit(
        self,
        *,
        action_tokens,
        timestep,
        context,
        context_mask,
    ):
        del timestep
        tokens = self.input_proj(action_tokens)
        return {
            "tokens": tokens,
            "freqs": torch.zeros(
                (tokens.shape[1], 1, 1), dtype=tokens.dtype, device=tokens.device
            ),
            "t_mod": torch.zeros(
                (tokens.shape[0], 1), dtype=tokens.dtype, device=tokens.device
            ),
            "context": context,
            "context_mask": context_mask,
        }

    def post_dit(self, tokens, _pre):
        return self.output_proj(tokens)


class _TinyMoT(torch.nn.Module):
    def __init__(self, video: _TinyVideoExpert, action: _TinyActionExpert) -> None:
        super().__init__()
        self.video = video
        self.action = action
        self.frozen_gain = torch.nn.Parameter(torch.tensor(0.5))

    @staticmethod
    def prefill_video_cache_with_tokens(
        *,
        video_tokens,
        video_freqs,
        video_t_mod,
        video_context_payload,
        video_attention_mask,
    ):
        del video_freqs, video_t_mod, video_context_payload, video_attention_mask
        cache = [{"k": video_tokens, "v": video_tokens}]
        return VideoPrefillOutput(kv_cache=cache, final_tokens=video_tokens)

    def forward_action_with_video_cache(
        self,
        *,
        action_tokens,
        action_freqs,
        action_t_mod,
        action_context_payload,
        video_kv_cache,
        attention_mask,
        video_seq_len,
    ):
        del (
            action_freqs,
            action_t_mod,
            action_context_payload,
            attention_mask,
            video_seq_len,
        )
        video_signal = video_kv_cache[0]["v"].mean(dim=1, keepdim=True)
        return self.action.mix(
            action_tokens + self.frozen_gain * video_signal
        )


def _new_model(
    *,
    policy: str = "gaussian_null",
    memory_sigma: float = 0.2,
    contract: WarmSourceRunContract | None = None,
    proprio_dim: int | None = None,
) -> WarmSourceFastWAM:
    video = _TinyVideoExpert()
    action = _TinyActionExpert()
    mot = _TinyMoT(video, action)
    model = WarmSourceFastWAM(
        video_expert=video,
        action_expert=action,
        mot=mot,
        vae=_TinyVAE(),
        text_dim=TEXT_DIM,
        proprio_dim=proprio_dim,
        device="cpu",
        torch_dtype=torch.float32,
        video_train_shift=1.0,
        video_infer_shift=1.0,
        action_train_shift=1.0,
        action_infer_shift=1.0,
    )
    model.configure_warm_source(
        policy=policy,
        memory_sigma=memory_sigma,
        run_contract=contract,
    )
    return model


def _new_base_model(*, proprio_dim: int | None = None) -> FastWAM:
    video = _TinyVideoExpert()
    action = _TinyActionExpert()
    mot = _TinyMoT(video, action)
    return FastWAM(
        video_expert=video,
        action_expert=action,
        mot=mot,
        vae=_TinyVAE(),
        text_dim=TEXT_DIM,
        proprio_dim=proprio_dim,
        device="cpu",
        torch_dtype=torch.float32,
        video_train_shift=1.0,
        video_infer_shift=1.0,
        action_train_shift=1.0,
        action_infer_shift=1.0,
    )


def _sample(*, future_fill: float = 0.0) -> dict[str, torch.Tensor]:
    video = torch.zeros((2, 3, 4, 16, 16), dtype=torch.float32)
    video[:, :, 0] = 0.25
    video[:, :, 1:] = future_fill
    return {
        "video": video,
        "context": torch.randn((2, 2, TEXT_DIM)),
        "context_mask": torch.ones((2, 2), dtype=torch.bool),
        "action": torch.randn((2, ACTION_HORIZON, ACTION_DIM)),
        "action_is_pad": torch.tensor(
            [[False, False, True], [False, False, False]], dtype=torch.bool
        ),
    }


def test_non_null_policy_requires_complete_compatible_contract() -> None:
    with pytest.raises(SourceTransportError, match="requires a complete"):
        _new_model(policy="fixed_context_top1")

    incompatible = WarmSourceRunContract(
        **{
            **_run_contract().to_dict(),
            "action_dim": ACTION_DIM + 1,
        }
    )
    with pytest.raises(SourceTransportError, match="action_dim does not match"):
        _new_model(policy="fixed_context_top1", contract=incompatible)


def test_warm_checkpoint_roundtrip_is_bound_to_policy_sigma_and_contract(
    tmp_path: Path,
) -> None:
    contract = _run_contract()
    source = _new_model(policy="fixed_context_top1", contract=contract)
    with torch.no_grad():
        source.action_expert.output_proj.weight.fill_(0.75)
    checkpoint = tmp_path / "warm.pt"
    source.save_checkpoint(checkpoint, step=17)

    restored = _new_model(policy="fixed_context_top1", contract=contract)
    payload = restored.load_checkpoint(checkpoint)
    assert payload["step"] == 17
    torch.testing.assert_close(
        restored.action_expert.output_proj.weight,
        source.action_expert.output_proj.weight,
    )

    wrong_sigma = _new_model(
        policy="fixed_context_top1", memory_sigma=0.3, contract=contract
    )
    before_failed_preflight = {
        name: value.detach().clone()
        for name, value in wrong_sigma.mot.state_dict().items()
    }
    with pytest.raises(ValueError, match="memory_sigma"):
        wrong_sigma.load_checkpoint(checkpoint)
    for name, value in wrong_sigma.mot.state_dict().items():
        torch.testing.assert_close(value, before_failed_preflight[name])

    wrong_contract = _new_model(
        policy="fixed_context_top1", contract=_run_contract(label="other")
    )
    with pytest.raises(ValueError, match="run contract"):
        wrong_contract.load_checkpoint(checkpoint)

    wrong_policy = _new_model(policy="gaussian_null")
    with pytest.raises(ValueError, match="source policy"):
        wrong_policy.load_checkpoint(checkpoint)


def test_warm_checkpoint_requires_exact_mot_and_cannot_downgrade_strictness(
    tmp_path: Path,
) -> None:
    contract = _run_contract()
    source = _new_model(policy="fixed_context_top1", contract=contract)
    checkpoint = tmp_path / "warm.pt"
    source.save_checkpoint(checkpoint)
    original = torch.load(checkpoint, map_location="cpu")

    missing = dict(original)
    missing["mot"] = dict(original["mot"])
    missing["mot"].pop(next(iter(missing["mot"])))
    missing_checkpoint = tmp_path / "warm-missing.pt"
    torch.save(missing, missing_checkpoint)
    with pytest.raises(RuntimeError, match="Missing key"):
        _new_model(
            policy="fixed_context_top1", contract=contract
        ).load_checkpoint(missing_checkpoint)

    unexpected = dict(original)
    unexpected["mot"] = dict(original["mot"])
    unexpected["mot"]["not_a_real_parameter"] = torch.zeros(1)
    unexpected_checkpoint = tmp_path / "warm-unexpected.pt"
    torch.save(unexpected, unexpected_checkpoint)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        _new_model(
            policy="fixed_context_top1", contract=contract
        ).load_checkpoint(unexpected_checkpoint)

    destination = _new_model(policy="fixed_context_top1", contract=contract)
    with pytest.raises(TypeError, match="strict_model_state"):
        destination.load_checkpoint(checkpoint, strict_model_state=False)


def test_base_checkpoint_loader_accepts_fastwam_and_rejects_warm(
    tmp_path: Path,
) -> None:
    base = _new_base_model()
    with torch.no_grad():
        base.action_expert.output_proj.bias.fill_(1.25)
    base_checkpoint = tmp_path / "base.pt"
    base.save_checkpoint(base_checkpoint, step=5)
    base_sha256 = hashlib.sha256(base_checkpoint.read_bytes()).hexdigest()
    contract = _run_contract(base_checkpoint_sha256=base_sha256)

    warm = _new_model(
        policy="fixed_context_top1", contract=contract
    )
    payload = warm.load_base_checkpoint(base_checkpoint)
    assert payload["step"] == 5
    torch.testing.assert_close(
        warm.action_expert.output_proj.bias,
        base.action_expert.output_proj.bias,
    )

    warm_checkpoint = tmp_path / "warm.pt"
    warm.save_checkpoint(warm_checkpoint, step=6)
    warm_sha256 = hashlib.sha256(warm_checkpoint.read_bytes()).hexdigest()
    destination = _new_model(
        policy="fixed_context_top1",
        contract=_run_contract(base_checkpoint_sha256=warm_sha256),
    )
    with pytest.raises(ValueError, match="forbidden state keys"):
        destination.load_base_checkpoint(warm_checkpoint)

    with pytest.raises(ValueError, match="missing the warm_source state"):
        destination.load_checkpoint(base_checkpoint)


def test_base_checkpoint_loader_requires_complete_mot_and_exact_proprio(
    tmp_path: Path,
) -> None:
    base = _new_base_model()
    incomplete_payload = {
        "mot": dict(base.mot.state_dict()),
        "step": 1,
        "torch_dtype": str(base.torch_dtype),
    }
    incomplete_payload["mot"].pop(next(iter(incomplete_payload["mot"])))
    incomplete_checkpoint = tmp_path / "incomplete.pt"
    torch.save(incomplete_payload, incomplete_checkpoint)
    incomplete_hash = hashlib.sha256(incomplete_checkpoint.read_bytes()).hexdigest()
    incomplete_destination = _new_model(
        policy="fixed_context_top1",
        contract=_run_contract(base_checkpoint_sha256=incomplete_hash),
    )
    with pytest.raises(RuntimeError, match="Missing key"):
        incomplete_destination.load_base_checkpoint(incomplete_checkpoint)

    legacy_checkpoint = tmp_path / "legacy.pt"
    torch.save({"dit": base.video_expert.state_dict()}, legacy_checkpoint)
    legacy_hash = hashlib.sha256(legacy_checkpoint.read_bytes()).hexdigest()
    legacy_destination = _new_model(
        policy="fixed_context_top1",
        contract=_run_contract(base_checkpoint_sha256=legacy_hash),
    )
    with pytest.raises(ValueError, match="complete `mot` state"):
        legacy_destination.load_base_checkpoint(legacy_checkpoint)

    proprio_base = _new_base_model(proprio_dim=3)
    proprio_checkpoint = tmp_path / "with-proprio.pt"
    proprio_base.save_checkpoint(proprio_checkpoint)
    proprio_hash = hashlib.sha256(proprio_checkpoint.read_bytes()).hexdigest()
    no_proprio_destination = _new_model(
        policy="fixed_context_top1",
        contract=_run_contract(base_checkpoint_sha256=proprio_hash),
    )
    with pytest.raises(ValueError, match="proprio_encoder presence"):
        no_proprio_destination.load_base_checkpoint(proprio_checkpoint)

    no_proprio_checkpoint = tmp_path / "without-proprio.pt"
    base.save_checkpoint(no_proprio_checkpoint)
    no_proprio_hash = hashlib.sha256(no_proprio_checkpoint.read_bytes()).hexdigest()
    proprio_destination = _new_model(
        policy="fixed_context_top1",
        contract=_run_contract(base_checkpoint_sha256=no_proprio_hash),
        proprio_dim=3,
    )
    with pytest.raises(ValueError, match="proprio_encoder presence"):
        proprio_destination.load_base_checkpoint(no_proprio_checkpoint)


def test_runtime_rejects_base_checkpoint_hash_before_model_construction(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "not-the-bound-checkpoint.pt"
    checkpoint.write_bytes(b"immutable checkpoint bytes")
    contract = _run_contract(base_checkpoint_sha256=_digest("different bytes"))

    with pytest.raises(ValueError, match="base checkpoint SHA256"):
        create_warm_source(
            model_id="unused",
            tokenizer_model_id="unused",
            video_dit_config={},
            source_policy="fixed_context_top1",
            run_contract=contract,
            base_checkpoint_path=str(checkpoint),
            device="cpu",
            model_dtype=torch.float32,
        )


def test_formal_runtime_requires_contract_and_base_for_gaussian_null() -> None:
    with pytest.raises(ValueError, match="including gaussian_null"):
        create_warm_source(
            model_id="unused",
            tokenizer_model_id="unused",
            video_dit_config={},
            source_policy="gaussian_null",
            run_contract=None,
            base_checkpoint_path=None,
            device="cpu",
            model_dtype=torch.float32,
        )


def test_inference_policy_safety_and_fixed_rank_zero_semantics() -> None:
    contract = _run_contract()
    fixed = _new_model(policy="fixed_context_top1", contract=contract)
    with pytest.raises(SourceTransportError, match="requires candidate means"):
        fixed.build_inference_source_context()
    with pytest.raises(SourceTransportError, match="requires raw candidate"):
        fixed.infer_action()

    means = torch.randn((2, 2, ACTION_HORIZON, ACTION_DIM))
    valid = torch.tensor([[False, True], [True, True]], dtype=torch.bool)
    context = fixed.build_inference_source_context(
        candidate_means=means,
        candidate_valid_mask=valid,
        memory_enabled_mask=torch.tensor([True, False]),
        batch_size=2,
    )
    # Row 0 must not skip invalid rank zero and silently use rank one. Row 1
    # is explicitly disabled even though rank zero is valid.
    assert context.component_indices.tolist() == [0, 0]

    enabled = fixed.build_inference_source_context(
        candidate_means=means,
        candidate_valid_mask=valid,
        batch_size=2,
    )
    assert enabled.component_indices.tolist() == [0, 1]

    bypass = ActionSourceContext(
        component_indices=torch.tensor([2], dtype=torch.long),
        candidate_means=means[:1],
        candidate_valid_mask=valid[:1],
    )
    with pytest.raises(SourceTransportError, match="prebuilt ActionSourceContext"):
        fixed.infer_action(action_source_context=bypass)
    with pytest.raises(SourceTransportError, match="online retrieval bridge"):
        fixed.infer_joint()
    with pytest.raises(SourceTransportError, match="online retrieval bridge"):
        fixed.infer()

    oracle = _new_model(policy="oracle_action_top1", contract=contract)
    with pytest.raises(SourceTransportError, match="forbidden"):
        oracle.build_inference_source_context()
    with pytest.raises(SourceTransportError, match="cannot run infer_action"):
        oracle.infer_action()
    with pytest.raises(SourceTransportError, match="cannot run infer_joint"):
        oracle.infer_joint()
    with pytest.raises(SourceTransportError, match="cannot run infer"):
        oracle.infer()


def test_infer_action_rebuilds_policy_context_from_raw_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[ActionSourceContext] = []

    def fake_base_infer(self, *args, **kwargs):
        del self, args
        captured.append(kwargs["action_source_context"])
        return {"action": torch.zeros((ACTION_HORIZON, ACTION_DIM))}

    monkeypatch.setattr(FastWAM, "infer_action", fake_base_infer)
    means = torch.randn((2, ACTION_HORIZON, ACTION_DIM))
    valid = torch.tensor([True, True], dtype=torch.bool)

    null = _new_model(policy="gaussian_null")
    null.infer_action(
        candidate_means=means,
        candidate_valid_mask=valid,
    )
    assert captured[-1].component_indices.tolist() == [0]
    assert captured[-1].candidate_means is None

    fixed = _new_model(
        policy="fixed_context_top1",
        contract=_run_contract(),
    )
    fixed.infer_action(
        candidate_means=means,
        candidate_valid_mask=valid,
    )
    assert captured[-1].component_indices.tolist() == [1]
    assert captured[-1].candidate_means is not means


def test_prefill_legacy_and_extended_apis_share_one_internal_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mot = object.__new__(MoT)
    torch.nn.Module.__init__(mot)
    cache = [{"k": torch.randn(1, 2, 3), "v": torch.randn(1, 2, 3)}]
    final_tokens = torch.randn(1, 2, 3)
    calls: list[dict[str, object]] = []

    def fake_prefill(self, **kwargs):
        del self
        calls.append(kwargs)
        return VideoPrefillOutput(kv_cache=cache, final_tokens=final_tokens)

    monkeypatch.setattr(
        mot,
        "_prefill_video_cache_with_tokens",
        MethodType(fake_prefill, mot),
    )
    kwargs = {
        "video_tokens": torch.randn(1, 2, 3),
        "video_freqs": torch.randn(2, 1, 1),
        "video_t_mod": torch.randn(1, 1),
        "video_context_payload": None,
        "video_attention_mask": torch.ones((2, 2), dtype=torch.bool),
    }

    legacy = mot.prefill_video_cache(**kwargs)
    extended = mot.prefill_video_cache_with_tokens(**kwargs)

    assert legacy is cache
    assert extended.kv_cache is cache
    assert extended.final_tokens is final_tokens
    assert len(calls) == 2
    assert calls[0].keys() == calls[1].keys() == kwargs.keys()
    for key, value in kwargs.items():
        assert calls[0][key] is value
        assert calls[1][key] is value


def test_action_only_path_ignores_future_frames_and_freezes_world_modules() -> None:
    model = _new_model(policy="gaussian_null")
    trainable = Wan22Trainer._apply_dit_only_train_mode(model)
    trainable_ids = {id(parameter) for parameter in trainable}
    assert trainable_ids == {id(parameter) for parameter in model.action_expert.parameters()}
    assert all(parameter.requires_grad for parameter in model.action_expert.parameters())
    assert not any(parameter.requires_grad for parameter in model.video_expert.parameters())
    assert not any(parameter.requires_grad for parameter in model.vae.parameters())
    assert not model.mot.frozen_gain.requires_grad
    assert model.action_expert.training
    assert not model.video_expert.training

    sample_a = _sample(future_fill=-10.0)
    sample_b = {key: value.clone() for key, value in sample_a.items()}
    sample_b["video"][:, :, 1:] = 10.0

    torch.manual_seed(1234)
    loss_a, metrics_a = model.training_loss(sample_a)
    torch.manual_seed(1234)
    loss_b, metrics_b = model.training_loss(sample_b)
    torch.testing.assert_close(loss_a, loss_b, rtol=0.0, atol=0.0)
    assert metrics_a == metrics_b
    assert set(metrics_a) == {
        "loss_action",
        "source_memory_rate",
        "source_rms",
        "source_l2",
        "gaussian_source_rms",
        "source_rms_reduction",
    }

    loss_a.backward()
    assert any(
        parameter.grad is not None
        and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in model.action_expert.parameters()
    )
    assert all(parameter.grad is None for parameter in model.video_expert.parameters())
    assert all(parameter.grad is None for parameter in model.vae.parameters())
    assert model.mot.frozen_gain.grad is None


def test_trainer_trainable_hook_includes_optional_proprio_only() -> None:
    model = _new_model(policy="gaussian_null", proprio_dim=3)
    trainable = Wan22Trainer._apply_dit_only_train_mode(model)

    expected = {
        id(parameter) for parameter in model.action_expert.parameters()
    } | {id(parameter) for parameter in model.proprio_encoder.parameters()}
    assert {id(parameter) for parameter in trainable} == expected
    assert all(parameter.requires_grad for parameter in trainable)
    assert model.action_expert.training
    assert model.proprio_encoder.training
    assert not model.video_expert.training
    assert not any(parameter.requires_grad for parameter in model.video_expert.parameters())
    assert not any(parameter.requires_grad for parameter in model.vae.parameters())


class _AcceleratorStub:
    def __init__(self, model) -> None:
        self.model = model
        self.loaded_state: str | None = None
        self.num_processes = 1

    def unwrap_model(self, _model):
        return self.model

    def load_state(self, *, input_dir):
        self.loaded_state = str(input_dir)

    @staticmethod
    def wait_for_everyone() -> None:
        return None


class _SamplerStub:
    def __init__(self) -> None:
        self.epoch: int | None = None
        self.batch: int | None = None

    def set_epoch_offset(self, value: int) -> None:
        self.epoch = value

    def set_resume_batch_offset(self, value: int) -> None:
        self.batch = value

    def clear_resume_batch_offset(self) -> None:
        self.batch = 0


def _trainer_stub(model: WarmSourceFastWAM) -> Wan22Trainer:
    trainer = object.__new__(Wan22Trainer)
    trainer.model = model
    trainer.accelerator = _AcceleratorStub(model)
    trainer.train_sampler = _SamplerStub()
    trainer.global_step = 11
    trainer.epoch = 2
    trainer.batch_in_epoch = 7
    trainer.batch_size = 2
    return trainer


def test_trainer_state_persists_contract_and_validates_before_accelerate_load(
    tmp_path: Path,
) -> None:
    model = _new_model(
        policy="fixed_context_top1", contract=_run_contract()
    )
    trainer = _trainer_stub(model)
    trainer._save_trainer_state(str(tmp_path))

    state_file = tmp_path / "trainer_state.json"
    payload = json.loads(state_file.read_text(encoding="utf-8"))
    assert payload["model_contract"] == model.trainer_state_metadata()

    restored = _trainer_stub(
        _new_model(policy="fixed_context_top1", contract=_run_contract())
    )
    restored.load_training_state(str(tmp_path))
    assert restored.accelerator.loaded_state == str(tmp_path)
    assert restored.global_step == 11
    assert restored.epoch == 2
    assert restored.batch_in_epoch == 7
    assert restored.train_sampler.epoch == 2
    assert restored.train_sampler.batch == 7

    payload["model_contract"]["policy"] = "gaussian_null"
    state_file.write_text(json.dumps(payload), encoding="utf-8")
    rejected = _trainer_stub(
        _new_model(policy="fixed_context_top1", contract=_run_contract())
    )
    with pytest.raises(ValueError, match="source policy"):
        rejected.load_training_state(str(tmp_path))
    assert rejected.accelerator.loaded_state is None


def test_trainer_rejects_missing_contract_metadata_before_accelerate_load(
    tmp_path: Path,
) -> None:
    model = _new_model(
        policy="fixed_context_top1", contract=_run_contract()
    )
    trainer = _trainer_stub(model)
    (tmp_path / "trainer_state.json").write_text(
        json.dumps({"global_step": 1, "epoch": 0, "batch_in_epoch": 0}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing its model_contract"):
        trainer.load_training_state(str(tmp_path))
    assert trainer.accelerator.loaded_state is None
