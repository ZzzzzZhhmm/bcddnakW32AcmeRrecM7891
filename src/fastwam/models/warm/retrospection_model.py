"""Complete WARM: consequence-aligned retrospective source transport.

This module composes the deliberately isolated M1/M2 infrastructure and the
lightweight learned adapters.  The base FastWAM video path runs once; factual
cross-episode events are reranked and validated by their predicted effect;
only then does a gated event action reshape the Action DiT source.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fastwam.datasets.warm_candidates import (
    WARM_CANDIDATE_MASK,
    WARM_CANDIDATE_MU,
)
from fastwam.datasets.warm_retrospective import (
    WARM_CANDIDATE_CONTEXT,
    WARM_CANDIDATE_EFFECT_DELTA,
    WARM_CANDIDATE_EFFECT_PRE,
    WARM_CANDIDATE_START_PROPRIO,
    WARM_CANDIDATE_SUPPORT,
    WARM_CANDIDATE_TIMING,
    WARM_CURRENT_CONTEXT,
    WARM_CURRENT_SEMANTIC,
    WARM_EPISODE_MASK,
    WARM_EPISODE_ACTION_MASK,
    WARM_EPISODE_ACTION_SUMMARIES,
    WARM_EPISODE_TOKENS,
    WARM_FUTURE_VALID,
    WARM_TARGET_EFFECT,
)
from fastwam.memory.manifest import sha256_canonical_json
from fastwam.models.wan22.fastwam import FastWAM

from .adapter_config import (
    RetrospectiveEventAdapterConfig,
    RetrospectiveGistConfig,
    SemanticBridgeConfig,
)
from .consequence import (
    ActionUtilityReranker,
    SourceConfidenceGate,
    build_action_effect_utility_targets,
    build_corruption_controls,
    consequence_consistency,
    select_consequence_candidate,
    utility_kl_divergence,
    utility_supervised_gate_bce,
    utility_supervised_gate_target,
)
from .consequence_curriculum import (
    CorruptionDecision,
    derive_corruption_batch,
    select_hard_negative_index,
)
from .retrospection_config import WarmRetrospectionConfig
from .retrospective_event_adapter import RetrospectiveEventAdapter
from .retrospective_gist import RetrospectiveGistAdapter
from .semantic_bridge import WorldFeatureSemanticBridge
from .source_model import (
    WarmSourceFastWAM,
    _event_id_json,
    _fixed_retrieval_telemetry,
    _json_safe,
)
from .source_transport import (
    ActionSourceContext,
    ActionSourceOutput,
    SourceTransportError,
)
from .video_adapter import build_video_layer_adapters


WARM_RETROSPECTION_CHECKPOINT_SCHEMA = "warm.retrospection-checkpoint"
WARM_RETROSPECTION_CHECKPOINT_VERSION = 3
WARM_ONLINE_ABLATION_MODES = frozenset(
    {"full", "context_only", "source_only_no_consequence"}
)
WARM_ONLINE_MEMORY_CORRUPTIONS = frozenset(
    {"clean", "wrong_event", "reversed_action", "phase_shift", "effect_mismatch"}
)
_EXPERIMENT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


class WarmRetrospectionError(ValueError):
    """Raised when the complete WARM path loses a factual contract."""


@dataclass(frozen=True, slots=True)
class OnlineExperimentControls:
    """Closed, runtime-only controls for one reproducible online experiment."""

    ablation_mode: str
    memory_corruption: str
    experiment_id: str


@dataclass(frozen=True, slots=True)
class _CorruptedCandidatePayload:
    """Inference-only candidate payload with exact padding preserved."""

    actions: torch.Tensor
    effect_pre: torch.Tensor
    effect_delta: torch.Tensor
    timing: torch.Tensor
    applied: torch.Tensor


@dataclass(frozen=True, slots=True)
class RetrospectiveSourceContext:
    """All factual/coarse-retrieval evidence consumed by the learned source."""

    query_context: torch.Tensor
    candidate_context: torch.Tensor
    candidate_actions: torch.Tensor
    candidate_start_proprio: torch.Tensor
    candidate_effect_pre: torch.Tensor
    candidate_effect_delta: torch.Tensor
    candidate_timing: torch.Tensor
    candidate_support: torch.Tensor
    candidate_valid_mask: torch.Tensor
    episode_tokens: torch.Tensor
    episode_mask: torch.Tensor
    episode_action_summaries: torch.Tensor
    episode_action_mask: torch.Tensor
    current_semantic_teacher: torch.Tensor | None = None
    target_effect: torch.Tensor | None = None
    target_action: torch.Tensor | None = None
    target_action_valid_mask: torch.Tensor | None = None
    future_valid_mask: torch.Tensor | None = None
    forced_candidate_indices: torch.Tensor | None = None


def _as_config(
    value: WarmRetrospectionConfig | Mapping[str, Any],
) -> WarmRetrospectionConfig:
    if isinstance(value, WarmRetrospectionConfig):
        return value
    if isinstance(value, Mapping):
        return WarmRetrospectionConfig.from_dict(value)
    raise TypeError("warm_retrospection_config must be a config or mapping")


def _candidate_action_summary(actions: torch.Tensor) -> torch.Tensor:
    """Mean, final displacement, and mean curvature: [B,K,3*Da]."""

    mean = actions.mean(dim=2)
    final = actions[:, :, -1]
    if actions.shape[2] > 1:
        variation = (actions[:, :, 1:] - actions[:, :, :-1]).abs().mean(dim=2)
    else:
        variation = torch.zeros_like(mean)
    return torch.cat((mean, final, variation), dim=-1)


def _warp_candidate_actions(
    candidate_actions: torch.Tensor,
    candidate_start_proprio: torch.Tensor,
    current_proprio: torch.Tensor,
    candidate_valid_mask: torch.Tensor,
    config: WarmRetrospectionConfig,
) -> torch.Tensor:
    """Map stored event actions to the current factual robot start state.

    The operation is intentionally closed rather than learned.  In
    ``start_proprio_delta`` mode both actions and proprioception are the same
    normalized native absolute-qpos representation.  Arm command dimensions
    receive the current-minus-event start offset; discrete/continuous gripper
    command dimensions remain byte-for-byte unchanged.  Invalid padded slots
    are restored to exact zeros after warping.
    """

    if candidate_actions.ndim != 4:
        raise WarmRetrospectionError("candidate_actions must be [B,K,H,Da]")
    batch, candidates, _, action_dim = candidate_actions.shape
    expected_start = (batch, candidates, config.proprio_dim)
    if tuple(candidate_start_proprio.shape) != expected_start:
        raise WarmRetrospectionError(
            "candidate_start_proprio must have shape "
            f"{expected_start}, got {tuple(candidate_start_proprio.shape)}"
        )
    if tuple(current_proprio.shape) != (batch, config.proprio_dim):
        raise WarmRetrospectionError(
            "current_proprio must have shape "
            f"{(batch, config.proprio_dim)}, got {tuple(current_proprio.shape)}"
        )
    if tuple(candidate_valid_mask.shape) != (batch, candidates):
        raise WarmRetrospectionError(
            "candidate_valid_mask must match candidate [B,K] dimensions"
        )
    if candidate_valid_mask.dtype != torch.bool:
        raise WarmRetrospectionError("candidate_valid_mask must be bool")
    if action_dim != config.action_dim:
        raise WarmRetrospectionError(
            "candidate action width does not match retrospection config"
        )

    warped = candidate_actions
    if config.canonical_action_mode == "start_proprio_delta":
        # Config validation proves proprio_dim == action_dim in this mode.
        offset = current_proprio[:, None, :] - candidate_start_proprio
        arm_mask = torch.ones(
            (config.action_dim,),
            dtype=candidate_actions.dtype,
            device=candidate_actions.device,
        )
        if config.canonical_gripper_dims:
            arm_mask[list(config.canonical_gripper_dims)] = 0.0
        warped = candidate_actions + offset[:, :, None, :].to(
            dtype=candidate_actions.dtype
        ) * arm_mask.view(1, 1, 1, -1)
    elif config.canonical_action_mode != "none":  # defensive after config load
        raise WarmRetrospectionError("unsupported canonical_action_mode")

    return warped * candidate_valid_mask[:, :, None, None].to(
        dtype=warped.dtype
    )


def _mean_effect(tokens: torch.Tensor) -> torch.Tensor:
    return tokens.mean(dim=-2)


def _smooth_rms(
    value: torch.Tensor,
    *,
    dims: tuple[int, ...],
    epsilon: float = 1.0e-6,
) -> torch.Tensor:
    """Differentiable RMS that is finite at an exactly-zero residual.

    The event adapter intentionally zero-initializes its residual head.  A
    bare ``sqrt(mean(x**2))`` has an infinite derivative at that exact
    initialization and produces ``0 * inf -> NaN`` during the first
    backward.  The shifted smooth norm keeps both the value and gradient zero
    at the origin while converging to the ordinary RMS away from it.
    """

    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError("value must be a floating-point tensor")
    if not dims:
        raise ValueError("dims must not be empty")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be positive and finite")
    value_f = value.float()
    mean_square = value_f.square().mean(dim=dims)
    # Algebraically equal to sqrt(mean_square + eps**2) - eps, but this
    # rationalized form is exactly zero at the origin and avoids subtracting
    # two nearly equal floating-point values.
    root = torch.sqrt(mean_square + float(epsilon) ** 2)
    return mean_square / (root + float(epsilon))


def _gather_candidate(
    values: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Gather [B,K,...] with -1 rows mapped to exact zeros."""

    batch = values.shape[0]
    output = values.new_zeros((batch, *values.shape[2:]))
    valid = indices >= 0
    if bool(valid.any().item()):
        rows = torch.nonzero(valid, as_tuple=False).squeeze(1)
        output[rows] = values[rows, indices[rows]]
    return output


def _apply_inference_memory_corruption(
    *,
    actions: torch.Tensor,
    effect_pre: torch.Tensor,
    effect_delta: torch.Tensor,
    timing: torch.Tensor,
    valid: torch.Tensor,
    mode: str,
) -> _CorruptedCandidatePayload:
    """Apply a deterministic inference corruption without touching facts.

    ``wrong_event`` is a selection corruption and is therefore handled after
    the clean candidate ranking has been computed.  All payload corruptions
    are functional: the bound online facts remain immutable, and invalid
    candidate slots are restored to exact zeros before any learned module can
    consume them.
    """

    if mode not in WARM_ONLINE_MEMORY_CORRUPTIONS:
        raise WarmRetrospectionError(f"unsupported memory_corruption {mode!r}")
    if actions.ndim != 4 or valid.ndim != 2 or actions.shape[:2] != valid.shape:
        raise WarmRetrospectionError(
            "online candidate actions/mask must have shapes [B,K,H,D] and [B,K]"
        )
    if valid.dtype != torch.bool or valid.device != actions.device:
        raise WarmRetrospectionError(
            "online candidate mask must be bool on the action device"
        )
    batch, candidates = valid.shape
    for field, value in (
        ("effect_pre", effect_pre),
        ("effect_delta", effect_delta),
        ("timing", timing),
    ):
        if value.ndim < 3 or value.shape[:2] != (batch, candidates):
            raise WarmRetrospectionError(
                f"online {field} must begin with candidate dimensions [B,K]"
            )
        if value.device != actions.device or value.dtype != actions.dtype:
            raise WarmRetrospectionError(
                f"online {field} must share action device and dtype"
            )

    corrupted_actions = actions
    corrupted_pre = effect_pre
    corrupted_delta = effect_delta
    corrupted_timing = timing
    applied = torch.zeros((batch,), dtype=torch.bool, device=actions.device)
    has_candidate = valid.any(dim=1)

    if mode == "reversed_action":
        corrupted_actions = torch.flip(actions, dims=(2,))
        applied = has_candidate
    elif mode == "phase_shift":
        shift = max(1, actions.shape[2] // 2)
        corrupted_actions = torch.roll(actions, shifts=shift, dims=2)
        corrupted_timing = timing.clone()
        # Timing is encoded as repeated [phase, validity] pairs.  Shift only
        # factual phases and leave absent-event sentinels at exact zero.
        for phase_index in range(0, timing.shape[-1] - 1, 2):
            validity = timing[..., phase_index + 1] > 0.5
            shifted = torch.remainder(timing[..., phase_index] + 0.5, 1.0)
            corrupted_timing[..., phase_index] = torch.where(
                validity, shifted, torch.zeros_like(shifted)
            )
        applied = has_candidate
    elif mode == "effect_mismatch":
        # A sign reversal is deterministic and remains a mismatch even when a
        # row has only one candidate (unlike candidate permutation).
        corrupted_delta = -effect_delta
        applied = has_candidate

    action_mask = valid[:, :, None, None].to(dtype=actions.dtype)
    effect_mask = valid.reshape(
        batch, candidates, *((1,) * (effect_delta.ndim - 2))
    ).to(dtype=effect_delta.dtype)
    pre_mask = valid.reshape(
        batch, candidates, *((1,) * (effect_pre.ndim - 2))
    ).to(dtype=effect_pre.dtype)
    timing_mask = valid.reshape(
        batch, candidates, *((1,) * (timing.ndim - 2))
    ).to(dtype=timing.dtype)
    return _CorruptedCandidatePayload(
        actions=corrupted_actions * action_mask,
        effect_pre=corrupted_pre * pre_mask,
        effect_delta=corrupted_delta * effect_mask,
        timing=corrupted_timing * timing_mask,
        applied=applied,
    )


def _wrong_event_indices(
    selected_indices: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose a deterministic distinct valid slot, or explicit null fallback."""

    if (
        selected_indices.ndim != 1
        or valid.ndim != 2
        or selected_indices.shape[0] != valid.shape[0]
        or selected_indices.dtype != torch.long
        or valid.dtype != torch.bool
        or selected_indices.device != valid.device
    ):
        raise WarmRetrospectionError("wrong-event selection inputs are incompatible")
    forced = torch.full_like(selected_indices, -1)
    fallback = torch.ones_like(selected_indices, dtype=torch.bool)
    for row in range(valid.shape[0]):
        slots = torch.nonzero(valid[row], as_tuple=False).flatten()
        selected = int(selected_indices[row].item())
        alternatives = slots[slots != selected]
        if alternatives.numel() > 0:
            # Lowest distinct physical slot is stable across devices and does
            # not depend on approximate ANN score ties.
            forced[row] = alternatives[0]
            fallback[row] = False
    return forced, fallback


class WarmRetrospectionFastWAM(WarmSourceFastWAM):
    """FastWAM with state-action-effect retrospection and Gaussian fallback."""

    @classmethod
    def from_wan22_pretrained(
        cls,
        *args: Any,
        warm_retrospection_config: WarmRetrospectionConfig | Mapping[str, Any],
        **kwargs: Any,
    ) -> "WarmRetrospectionFastWAM":
        config = _as_config(warm_retrospection_config)
        model = super().from_wan22_pretrained(*args, **kwargs)
        if not isinstance(model, cls):
            raise TypeError("FastWAM factory did not preserve full WARM subclass")
        model.configure_warm_retrospection(config)
        return model

    def configure_warm_retrospection(
        self,
        config: WarmRetrospectionConfig | Mapping[str, Any],
    ) -> None:
        cfg = _as_config(config)
        if self.warm_source_policy != "fixed_context_top1":
            raise WarmRetrospectionError(
                "complete WARM requires fixed_context_top1 coarse retrieval"
            )
        if abs(float(self.warm_memory_sigma) - cfg.source_sigma_min) > 1.0e-12:
            raise WarmRetrospectionError(
                "memory_sigma must equal retrospection.source_sigma_min"
            )
        if cfg.action_dim != int(self.action_expert.action_dim):
            raise WarmRetrospectionError(
                "retrospection action_dim does not match Action DiT"
            )
        if cfg.text_dim != int(self.text_dim):
            raise WarmRetrospectionError(
                "retrospection text_dim does not match FastWAM"
            )
        if self.proprio_dim is None or cfg.proprio_dim != int(self.proprio_dim):
            raise WarmRetrospectionError(
                "complete WARM requires the configured proprio dimension"
            )
        video_dim = int(
            getattr(self.video_expert, "dim", getattr(self.video_expert, "hidden_dim", -1))
        )
        if video_dim > 0 and cfg.video_dim != video_dim:
            raise WarmRetrospectionError(
                f"retrospection video_dim {cfg.video_dim} != Video DiT {video_dim}"
            )

        bridge_cfg = SemanticBridgeConfig(
            early_dim=cfg.video_dim,
            late_dim=cfg.video_dim,
            semantic_dim=cfg.semantic_dim,
            num_heads=cfg.bridge_heads,
        )
        gist_cfg = RetrospectiveGistConfig(
            world_dim=cfg.semantic_dim,
            semantic_dim=cfg.semantic_dim,
            episode_dim=cfg.semantic_dim,
            event_dim=cfg.semantic_dim,
            text_dim=cfg.text_dim,
            model_dim=cfg.gist_dim,
            future_target_dim=cfg.semantic_dim,
            num_heads=cfg.gist_heads,
        )
        event_cfg = RetrospectiveEventAdapterConfig(
            action_dim=cfg.action_dim,
            action_horizon=cfg.action_horizon,
            timing_dim=cfg.timing_dim,
            world_dim=cfg.semantic_dim,
            gist_dim=cfg.gist_dim,
            proprio_dim=cfg.proprio_dim,
            text_dim=cfg.text_dim,
            event_dim=cfg.semantic_dim,
            model_dim=cfg.event_model_dim,
            effect_dim=cfg.semantic_dim,
            num_heads=cfg.event_heads,
            rho=cfg.residual_rho,
        )
        self.semantic_bridge = WorldFeatureSemanticBridge(bridge_cfg)
        self.retrospective_gist = RetrospectiveGistAdapter(gist_cfg)
        self.retrospective_event_adapter = RetrospectiveEventAdapter(event_cfg)
        self.utility_reranker = ActionUtilityReranker(
            query_dim=cfg.context_dim,
            candidate_context_dim=cfg.context_dim,
            action_summary_dim=3 * cfg.action_dim,
            effect_dim=cfg.semantic_dim,
            timing_dim=cfg.timing_dim,
            hidden_dim=cfg.reranker_hidden_dim,
        )
        self.source_confidence_gate = SourceConfidenceGate(cfg.gate_hidden_dim)
        self.episode_action_projection = nn.Sequential(
            nn.LayerNorm(cfg.episode_action_summary_dim),
            nn.Linear(cfg.episode_action_summary_dim, cfg.semantic_dim),
            nn.GELU(),
            nn.Linear(cfg.semantic_dim, cfg.semantic_dim),
        )
        self.gist_to_text = nn.Linear(cfg.gist_dim, cfg.text_dim, bias=False)
        self.action_context_to_text = nn.Linear(
            cfg.event_model_dim, cfg.text_dim, bias=False
        )
        nn.init.zeros_(self.gist_to_text.weight)
        # The event adapter's action-context output projection is already
        # zero-initialized.  Keeping this second projection non-zero lets the
        # inner projection receive gradients from the first training step.
        nn.init.xavier_uniform_(self.action_context_to_text.weight)
        blocks = getattr(self.video_expert, "blocks", None)
        if blocks is None:
            # Synthetic unit fixtures need not emulate the 5B transformer.
            # Production Wan Video DiT always exposes ``blocks``.
            self.video_layer_adapters = nn.ModuleDict()
        else:
            self.video_layer_adapters = build_video_layer_adapters(
                hidden_dim=cfg.video_dim,
                rank=cfg.video_adapter_rank,
                scale=cfg.video_adapter_scale,
                num_layers=len(blocks),
                requested_layers=cfg.video_adapter_layers,
            )
        self.warm_retrospection_config = cfg
        self._last_retrospection_diagnostics: dict[str, Any] = {}
        self._warm_online_experiment_controls = OnlineExperimentControls(
            ablation_mode="full",
            memory_corruption="clean",
            experiment_id="default",
        )
        self._warm_online_experiment_locked = False
        self.to(device=self.device, dtype=self.torch_dtype)

    def _require_retrospection(self) -> WarmRetrospectionConfig:
        config = getattr(self, "warm_retrospection_config", None)
        if not isinstance(config, WarmRetrospectionConfig):
            raise RuntimeError("WarmRetrospectionFastWAM has not been configured")
        return config

    def configure_online_experiment(
        self,
        *,
        ablation_mode: str,
        memory_corruption: str,
        experiment_id: str,
    ) -> None:
        """Bind one closed RMBench/runtime experiment before first inference.

        These controls are intentionally absent from checkpoint state: they
        change evaluation semantics, never learned weights.  Rebinding after
        inference would make a result stream ambiguous, so only an idempotent
        repeat of the exact same controls is accepted once inference starts.
        """

        self._require_retrospection()
        if not isinstance(ablation_mode, str) or (
            ablation_mode not in WARM_ONLINE_ABLATION_MODES
        ):
            raise WarmRetrospectionError(
                "ablation_mode must be one of "
                f"{sorted(WARM_ONLINE_ABLATION_MODES)}"
            )
        if not isinstance(memory_corruption, str) or (
            memory_corruption not in WARM_ONLINE_MEMORY_CORRUPTIONS
        ):
            raise WarmRetrospectionError(
                "memory_corruption must be one of "
                f"{sorted(WARM_ONLINE_MEMORY_CORRUPTIONS)}"
            )
        if (
            not isinstance(experiment_id, str)
            or _EXPERIMENT_ID.fullmatch(experiment_id) is None
        ):
            raise WarmRetrospectionError(
                "experiment_id must match [A-Za-z0-9][A-Za-z0-9_.-]{0,127}"
            )
        controls = OnlineExperimentControls(
            ablation_mode=ablation_mode,
            memory_corruption=memory_corruption,
            experiment_id=experiment_id,
        )
        current = self._online_experiment_controls()
        if bool(getattr(self, "_warm_online_experiment_locked", False)):
            if controls != current:
                raise WarmRetrospectionError(
                    "online experiment controls are locked after first inference"
                )
            return
        self._warm_online_experiment_controls = controls

    def _online_experiment_controls(self) -> OnlineExperimentControls:
        value = getattr(self, "_warm_online_experiment_controls", None)
        if not isinstance(value, OnlineExperimentControls):
            # Compatibility for a fully configured checkpoint created before
            # runtime experiment controls were introduced.
            return OnlineExperimentControls("full", "clean", "default")
        return value

    def configure_trainable_modules(self):
        """Adapt Action DiT, WARM modules, and selected Video DiT adapters.

        The action velocity field must see the same memory-conditioned source
        distribution used at inference.  VAE, text encoder and the 5B Video
        DiT backbone stay frozen; its tiny bottleneck adapters, the existing
        Action DiT (not a second policy), proprio bridge, and compact WARM
        modules are optimized together.
        """

        self._require_warm_configuration()
        self._require_retrospection()
        self.eval()
        self.requires_grad_(False)
        # MoT's activation-checkpoint guards depend on ``self.mot.training``.
        # Re-enable training mode for the execution graph while keeping the
        # 5B backbone parameters frozen below; otherwise the full video-only
        # branch retains all post-adapter activations and is likely to OOM.
        self.mot.train()
        trainable_modules = (
            self.action_expert,
            self.semantic_bridge,
            self.retrospective_gist,
            self.retrospective_event_adapter,
            self.utility_reranker,
            self.source_confidence_gate,
            self.episode_action_projection,
            self.gist_to_text,
            self.action_context_to_text,
            self.video_layer_adapters,
        )
        if self.proprio_encoder is not None:
            trainable_modules = (*trainable_modules, self.proprio_encoder)
        for module in trainable_modules:
            module.train()
            module.requires_grad_(True)
        return tuple(
            parameter for parameter in self.parameters() if parameter.requires_grad
        )

    def _validate_retrospective_dataset(self, dataset: object, *, purpose: str) -> None:
        cfg = self._require_retrospection()
        feature_store = getattr(dataset, "feature_store", None)
        resolver = getattr(dataset, "resolver", None)
        if feature_store is None or resolver is None:
            raise SourceTransportError(
                f"complete WARM {purpose} requires RuntimeRetrospectiveDatasetAdapter"
            )
        expected_semantic = (4, cfg.semantic_dim)
        if tuple(feature_store.semantic_shape) != expected_semantic:
            raise SourceTransportError(
                f"{purpose} semantic feature shape {feature_store.semantic_shape} "
                f"!= {expected_semantic}"
            )
        if int(feature_store.context_dim) != cfg.context_dim:
            raise SourceTransportError(
                f"{purpose} context width does not match retrospection config"
            )
        if int(feature_store.action_dim) != cfg.action_dim:
            raise SourceTransportError(
                f"{purpose} feature action width does not match retrospection config"
            )
        if int(feature_store.action_summary_dim) != cfg.episode_action_summary_dim:
            raise SourceTransportError(
                f"{purpose} episode action-summary width is incompatible"
            )
        if int(feature_store.action_summary_chunk_size) != cfg.episode_action_chunk_size:
            raise SourceTransportError(
                f"{purpose} executed-action chunk size does not match retrospection config"
            )
        if int(resolver.action_horizon) != cfg.action_horizon:
            raise SourceTransportError(
                f"{purpose} action horizon does not match retrospection config"
            )

    def validate_training_dataset(self, dataset: object) -> None:
        super().validate_training_dataset(dataset)
        self._validate_retrospective_dataset(dataset, purpose="training")

    def validate_validation_dataset(self, dataset: object) -> None:
        super().validate_validation_dataset(dataset)
        self._validate_retrospective_dataset(dataset, purpose="validation")

    def _tensor(
        self,
        sample: Mapping[str, Any],
        key: str,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        value = sample.get(key)
        if not isinstance(value, torch.Tensor):
            raise WarmRetrospectionError(f"sample is missing tensor {key!r}")
        return value.to(device=self.device, dtype=dtype, non_blocking=True)

    @staticmethod
    def _batch_ints(value: object, batch: int, field: str) -> tuple[int, ...]:
        if isinstance(value, torch.Tensor):
            flat = value.detach().cpu().reshape(-1).tolist()
        elif isinstance(value, (list, tuple)):
            flat = list(value)
        else:
            flat = [value]
        if len(flat) != batch:
            raise WarmRetrospectionError(f"{field} must contain {batch} values")
        result: list[int] = []
        for item in flat:
            if isinstance(item, bool) or not isinstance(item, (int, np.integer)):
                raise WarmRetrospectionError(f"{field} values must be integers")
            result.append(int(item))
        return tuple(result)

    def _training_source_context(
        self,
        sample: Mapping[str, Any],
    ) -> RetrospectiveSourceContext:
        cfg = self._require_retrospection()
        action = self._tensor(sample, "action", dtype=self.torch_dtype)
        batch = int(action.shape[0])
        candidate_actions = self._tensor(
            sample, WARM_CANDIDATE_MU, dtype=self.torch_dtype
        )
        valid = self._tensor(sample, WARM_CANDIDATE_MASK, dtype=torch.bool)
        candidate_start_proprio = self._tensor(
            sample, WARM_CANDIDATE_START_PROPRIO, dtype=self.torch_dtype
        )
        candidate_effect = self._tensor(
            sample, WARM_CANDIDATE_EFFECT_DELTA, dtype=self.torch_dtype
        )
        target_effect = self._tensor(
            sample, WARM_TARGET_EFFECT, dtype=self.torch_dtype
        )
        split = self._query_split_from_sample(sample, batch_size=batch)

        decisions: Sequence[CorruptionDecision]
        if split == "dev":
            decisions = tuple(
                CorruptionDecision("normal", 0.0, 0) for _ in range(batch)
            )
        else:
            dataset_indices = self._batch_ints(
                sample.get("dataset_index"), batch, "dataset_index"
            )
            episode_indices = self._batch_ints(
                sample.get("episode_index"), batch, "episode_index"
            )
            frame_indices = self._batch_ints(
                sample.get("frame_index"), batch, "frame_index"
            )
            identities = tuple(
                f"{split}:{d}:{e}:{f}"
                for d, e, f in zip(
                    dataset_indices, episode_indices, frame_indices, strict=True
                )
            )
            decisions = derive_corruption_batch(cfg.corruption_seed, identities)

        # Pick an actually incompatible raw event for hard-negative rows: the
        # farthest valid action/effect candidate, excluding the best when
        # alternatives exist.  The curriculum token breaks deterministic ties.
        sample_proprio = self._tensor(sample, "proprio", dtype=self.torch_dtype)
        if (
            sample_proprio.ndim != 3
            or sample_proprio.shape[0] != batch
            or sample_proprio.shape[2] != cfg.proprio_dim
        ):
            raise WarmRetrospectionError(
                "sample proprio must be [B,T,proprio_dim]"
            )
        current_proprio = sample_proprio[:, 0]
        warped_candidate_actions = _warp_candidate_actions(
            candidate_actions,
            candidate_start_proprio,
            current_proprio,
            valid,
            cfg,
        )
        raw_action_distance = (
            warped_candidate_actions.float() - action.unsqueeze(1).float()
        ).square().mean(dim=(-1, -2))
        raw_effect_distance = (
            _mean_effect(candidate_effect).float()
            - _mean_effect(target_effect).unsqueeze(1).float()
        ).square().mean(dim=-1)
        incompatibility = raw_action_distance + raw_effect_distance
        hard_indices = torch.zeros((batch,), dtype=torch.long, device=self.device)
        normalized_decisions: list[CorruptionDecision] = []
        for row, decision in enumerate(decisions):
            if decision.mode != "hard_negative":
                normalized_decisions.append(decision)
                continue
            slots = torch.nonzero(valid[row], as_tuple=False).flatten().tolist()
            if not slots:
                normalized_decisions.append(
                    CorruptionDecision(
                        "null", decision.unit_draw, decision.hard_negative_token
                    )
                )
                continue
            ordered = sorted(
                (int(slot) for slot in slots),
                key=lambda slot: (-float(incompatibility[row, slot].item()), slot),
            )
            upper = ordered[: max(1, (len(ordered) + 1) // 2)]
            hard_indices[row] = select_hard_negative_index(decision, upper)
            normalized_decisions.append(decision)
        controls = build_corruption_controls(
            valid,
            tuple(normalized_decisions),
            hard_negative_indices=hard_indices,
        )
        effective = controls.candidate_valid_mask

        def masked(key: str, *, dtype: torch.dtype) -> torch.Tensor:
            value = self._tensor(sample, key, dtype=dtype)
            shape = (batch, effective.shape[1]) + (1,) * (value.ndim - 2)
            return value * effective.reshape(shape).to(dtype=value.dtype)

        future_valid = self._tensor(
            sample, WARM_FUTURE_VALID, dtype=torch.bool
        ).reshape(batch)
        action_is_pad = sample.get("action_is_pad")
        if action_is_pad is None:
            action_valid = torch.ones(
                action.shape[:2], dtype=torch.bool, device=self.device
            )
        else:
            if not isinstance(action_is_pad, torch.Tensor):
                raise WarmRetrospectionError("action_is_pad must be a tensor")
            action_is_pad = action_is_pad.to(
                device=self.device, dtype=torch.bool, non_blocking=True
            )
            if action_is_pad.shape != action.shape[:2]:
                raise WarmRetrospectionError(
                    "action_is_pad must match the target action [B,H]"
                )
            action_valid = ~action_is_pad
        return RetrospectiveSourceContext(
            query_context=self._tensor(
                sample, WARM_CURRENT_CONTEXT, dtype=self.torch_dtype
            ),
            candidate_context=masked(
                WARM_CANDIDATE_CONTEXT, dtype=self.torch_dtype
            ),
            candidate_actions=candidate_actions
            * effective[:, :, None, None].to(dtype=candidate_actions.dtype),
            candidate_start_proprio=candidate_start_proprio
            * effective[:, :, None].to(dtype=candidate_start_proprio.dtype),
            candidate_effect_pre=masked(
                WARM_CANDIDATE_EFFECT_PRE, dtype=self.torch_dtype
            ),
            candidate_effect_delta=candidate_effect
            * effective[:, :, None, None].to(dtype=candidate_effect.dtype),
            candidate_timing=masked(
                WARM_CANDIDATE_TIMING, dtype=self.torch_dtype
            ),
            candidate_support=masked(
                WARM_CANDIDATE_SUPPORT, dtype=self.torch_dtype
            ),
            candidate_valid_mask=effective,
            episode_tokens=self._tensor(
                sample, WARM_EPISODE_TOKENS, dtype=self.torch_dtype
            ),
            episode_mask=self._tensor(
                sample, WARM_EPISODE_MASK, dtype=torch.bool
            ),
            episode_action_summaries=self._tensor(
                sample, WARM_EPISODE_ACTION_SUMMARIES, dtype=self.torch_dtype
            ),
            episode_action_mask=self._tensor(
                sample, WARM_EPISODE_ACTION_MASK, dtype=torch.bool
            ),
            current_semantic_teacher=self._tensor(
                sample, WARM_CURRENT_SEMANTIC, dtype=self.torch_dtype
            ),
            target_effect=target_effect,
            target_action=action,
            target_action_valid_mask=action_valid,
            future_valid_mask=future_valid,
            forced_candidate_indices=controls.forced_candidate_indices,
        )

    def training_loss(self, sample, tiled: bool = False):
        if not isinstance(sample, Mapping):
            raise TypeError("sample must be a mapping")
        context = self._training_source_context(sample)
        action_loss, metrics = FastWAM.training_loss_action_only(
            self,
            sample,
            action_source_context=context,  # type: ignore[arg-type]
            memory_sigma=self._require_retrospection().source_sigma_min,
            tiled=tiled,
        )
        if self.loss_lambda_video <= 0.0:
            return action_loss, metrics
        video_loss, video_metrics = FastWAM.training_loss_video_only(
            self, sample, tiled=tiled
        )
        overlap = set(metrics) & set(video_metrics)
        if overlap:
            raise WarmRetrospectionError(
                f"duplicate action/video loss metrics: {sorted(overlap)}"
            )
        return action_loss + video_loss, {**metrics, **video_metrics}

    def _resolve_action_source(
        self,
        *,
        base_gaussian: torch.Tensor,
        action_source_context: ActionSourceContext | RetrospectiveSourceContext | None,
        memory_sigma: float,
        phase: str,
        final_video_tokens: torch.Tensor | None,
        world_token_streams: tuple[torch.Tensor, ...] | None = None,
        text_context: torch.Tensor | None = None,
        text_context_mask: torch.Tensor | None = None,
        current_proprio: torch.Tensor | None = None,
        current_video_latent: torch.Tensor | None = None,
    ) -> ActionSourceOutput:
        if not isinstance(action_source_context, RetrospectiveSourceContext):
            return super()._resolve_action_source(
                base_gaussian=base_gaussian,
                action_source_context=action_source_context,
                memory_sigma=memory_sigma,
                phase=phase,
                final_video_tokens=final_video_tokens,
                world_token_streams=world_token_streams,
                text_context=text_context,
                text_context_mask=text_context_mask,
                current_proprio=current_proprio,
                current_video_latent=current_video_latent,
            )
        cfg = self._require_retrospection()
        ctx = action_source_context
        if phase not in {"train", "infer"}:
            raise WarmRetrospectionError("phase must be train or infer")
        configured_controls = self._online_experiment_controls()
        if phase == "infer":
            self._warm_online_experiment_locked = True
            ablation_mode = configured_controls.ablation_mode
            memory_corruption = configured_controls.memory_corruption
        else:
            # Evaluation controls must never alter optimization semantics.
            ablation_mode = "full"
            memory_corruption = "clean"
        if world_token_streams is None or len(world_token_streams) != 2:
            raise WarmRetrospectionError(
                "complete WARM requires the two Video DiT tap streams"
            )
        early, late = world_token_streams
        if early.shape != late.shape or early.shape[0] != base_gaussian.shape[0]:
            raise WarmRetrospectionError("Video DiT tap streams are not aligned")
        if text_context is None or text_context_mask is None:
            raise WarmRetrospectionError("complete WARM requires text context")
        if current_proprio is None:
            raise WarmRetrospectionError("complete WARM requires proprioception")
        candidate_payload = _apply_inference_memory_corruption(
            actions=ctx.candidate_actions,
            effect_pre=ctx.candidate_effect_pre,
            effect_delta=ctx.candidate_effect_delta,
            timing=ctx.candidate_timing,
            valid=ctx.candidate_valid_mask,
            mode=memory_corruption,
        )
        warped_candidate_actions = _warp_candidate_actions(
            candidate_payload.actions,
            ctx.candidate_start_proprio,
            current_proprio,
            ctx.candidate_valid_mask,
            cfg,
        )
        world_mask = torch.ones(
            early.shape[:2], dtype=torch.bool, device=early.device
        )
        bridge = self.semantic_bridge(early, late, world_mask)
        valid = ctx.candidate_valid_mask
        event_mask = valid[:, :, None].expand(
            -1, -1, candidate_payload.effect_delta.shape[2]
        )
        semantic_mask = torch.ones(
            bridge.semantic_tokens.shape[:2],
            dtype=torch.bool,
            device=bridge.semantic_tokens.device,
        )
        if (
            ctx.episode_action_summaries.ndim != 3
            or ctx.episode_action_summaries.shape[:2]
            != ctx.episode_action_mask.shape
            or ctx.episode_action_summaries.shape[-1]
            != cfg.episode_action_summary_dim
        ):
            raise WarmRetrospectionError(
                "episode action summaries/mask violate the configured shape"
            )
        action_history_tokens = self.episode_action_projection(
            ctx.episode_action_summaries
        ).masked_fill(~ctx.episode_action_mask.unsqueeze(-1), 0.0)
        episode_tokens = torch.cat(
            (ctx.episode_tokens, action_history_tokens), dim=1
        )
        episode_mask = torch.cat(
            (ctx.episode_mask, ctx.episode_action_mask), dim=1
        )
        # The required world transition is a query-only prediction.  Long-term
        # candidates are explicitly hidden here so an event can never help
        # manufacture the criterion that later accepts that same event.
        no_event_mask = torch.zeros_like(event_mask)
        required_gist = self.retrospective_gist(
            world_tokens=bridge.world_tokens,
            world_mask=bridge.token_mask,
            semantic_tokens=bridge.semantic_tokens,
            semantic_mask=semantic_mask,
            episode_tokens=episode_tokens,
            episode_mask=episode_mask,
            event_pre_tokens=torch.zeros_like(candidate_payload.effect_pre),
            event_delta_tokens=torch.zeros_like(candidate_payload.effect_delta),
            event_token_mask=no_event_mask,
            text_tokens=text_context,
            text_mask=text_context_mask,
        )
        source_only = ablation_mode == "source_only_no_consequence"
        event_delta_input = (
            torch.zeros_like(candidate_payload.effect_delta)
            if source_only
            else candidate_payload.effect_delta
        )
        event = self.retrospective_event_adapter(
            warped_actions=warped_candidate_actions,
            gripper_timing=candidate_payload.timing,
            candidate_valid_mask=valid,
            world_tokens=bridge.world_tokens,
            world_mask=bridge.token_mask,
            gist_tokens=required_gist.gist_tokens,
            proprio=current_proprio,
            text_tokens=text_context,
            text_mask=text_context_mask,
            event_delta_tokens=event_delta_input,
            event_delta_mask=event_mask,
        )
        observed_effect = _mean_effect(candidate_payload.effect_delta)
        action_summary = _candidate_action_summary(event.adapted_action_mean)
        reranker_scores = self.utility_reranker(
            ctx.query_context,
            ctx.candidate_context,
            action_summary,
            torch.zeros_like(observed_effect) if source_only else observed_effect,
            candidate_payload.timing,
            valid,
        )
        transition_gist = self.retrospective_gist.future_projection(
            required_gist.pooled_gist
        )
        consistency = consequence_consistency(
            event.predicted_effect,
            transition_gist,
            magnitude_weight=cfg.magnitude_weight,
        )
        selection_consistency = (
            torch.zeros_like(consistency) if source_only else consistency
        )
        selection = select_consequence_candidate(
            reranker_scores,
            selection_consistency,
            ctx.candidate_support,
            valid,
            consequence_weight=0.0 if source_only else cfg.consequence_weight,
            support_weight=cfg.support_weight,
            # Candidate-only ranking logits have no absolute null calibration;
            # the separately utility-supervised gate owns continuous fallback.
            automatic_null=False,
            forced_candidate_indices=ctx.forced_candidate_indices,
        )
        corruption_fallback = torch.zeros_like(
            selection.memory_mask, dtype=torch.bool
        )
        if phase == "infer" and memory_corruption == "wrong_event":
            forced_wrong, corruption_fallback = _wrong_event_indices(
                selection.candidate_indices, valid
            )
            selection = select_consequence_candidate(
                reranker_scores,
                selection_consistency,
                ctx.candidate_support,
                valid,
                consequence_weight=(
                    0.0 if source_only else cfg.consequence_weight
                ),
                support_weight=cfg.support_weight,
                automatic_null=False,
                forced_candidate_indices=forced_wrong,
            )
        selected_mean = _gather_candidate(
            event.adapted_action_mean, selection.candidate_indices
        )
        selected_effect = _gather_candidate(
            event.predicted_effect, selection.candidate_indices
        )
        selected_residual = _gather_candidate(
            event.action_residual, selection.candidate_indices
        )
        deformation = _smooth_rms(
            selected_residual,
            dims=(-1, -2),
        ).to(dtype=base_gaussian.dtype)
        gate = self.source_confidence_gate(selection, deformation)
        relevance_gate = gate.probability.to(dtype=base_gaussian.dtype)
        gate_value = (
            torch.zeros_like(relevance_gate)
            if ablation_mode == "context_only"
            else relevance_gate
        )
        source = (
            gate_value[:, None, None]
            * (selected_mean + cfg.source_sigma_min * base_gaussian)
            + (1.0 - gate_value[:, None, None]) * base_gaussian
        )

        selected_action_context = _gather_candidate(
            event.action_context_tokens, selection.candidate_indices
        )
        selected_event_pre = _gather_candidate(
            candidate_payload.effect_pre, selection.candidate_indices
        ).unsqueeze(1)
        selected_event_delta = _gather_candidate(
            candidate_payload.effect_delta, selection.candidate_indices
        ).unsqueeze(1)
        selected_event_mask = selection.memory_mask[:, None, None].expand(
            -1, 1, candidate_payload.effect_delta.shape[2]
        )
        if source_only:
            # This ablation modifies only the action-flow source.  It neither
            # selects by predicted consequence nor appends memory tokens to
            # the Action DiT condition sequence.
            conditioning = None
        else:
            predictive_gist = self.retrospective_gist(
                world_tokens=bridge.world_tokens,
                world_mask=bridge.token_mask,
                semantic_tokens=bridge.semantic_tokens,
                semantic_mask=semantic_mask,
                episode_tokens=episode_tokens,
                episode_mask=episode_mask,
                event_pre_tokens=selected_event_pre,
                event_delta_tokens=selected_event_delta,
                event_token_mask=selected_event_mask,
                text_tokens=text_context,
                text_mask=text_context_mask,
            )
            # Context-only retains the learned memory relevance even though
            # its action-flow source is exactly Gaussian.
            conditioning_gate = relevance_gate.detach()[:, None, None]
            blended_gist_tokens = required_gist.gist_tokens + conditioning_gate * (
                predictive_gist.gist_tokens - required_gist.gist_tokens
            )
            conditioning = torch.cat(
                (
                    self.gist_to_text(blended_gist_tokens),
                    self.action_context_to_text(selected_action_context)
                    * conditioning_gate,
                ),
                dim=1,
            )

        zero = base_gaussian.sum() * 0.0
        losses = {
            "retrieval": zero,
            "bridge": zero,
            "gist": zero,
            "effect": zero,
            "gate": zero,
            "adaptation": zero,
        }
        if phase == "train":
            if (
                ctx.target_action is None
                or ctx.target_effect is None
                or ctx.current_semantic_teacher is None
                or ctx.target_action_valid_mask is None
                or ctx.future_valid_mask is None
            ):
                raise WarmRetrospectionError(
                    "training context is missing stop-gradient teachers"
                )
            target_effect = _mean_effect(ctx.target_effect)
            action_valid = ctx.target_action_valid_mask
            if (
                action_valid.dtype != torch.bool
                or action_valid.shape != ctx.target_action.shape[:2]
                or action_valid.device != ctx.target_action.device
            ):
                raise WarmRetrospectionError(
                    "target_action_valid_mask must be bool [B,H]"
                )
            future_valid = ctx.future_valid_mask
            supervised_rows = future_valid & action_valid.any(dim=1)
            utility_valid = valid & supervised_rows.unsqueeze(1)
            non_hard_rows = (
                torch.ones_like(supervised_rows)
                if ctx.forced_candidate_indices is None
                else ctx.forced_candidate_indices < 0
            )
            utility = build_action_effect_utility_targets(
                # The deterministic factual start-state warp defines the
                # contextualized ranking label; the learnable bounded adapter
                # still cannot move its own target.
                warped_candidate_actions,
                ctx.target_action,
                observed_effect,
                target_effect,
                utility_valid,
                action_valid_mask=action_valid,
                effect_weight=cfg.utility_effect_weight,
                temperature=cfg.utility_temperature,
            )
            losses["retrieval"] = utility_kl_divergence(
                reranker_scores, utility.probabilities, utility_valid
            )
            losses["bridge"] = self.semantic_bridge.alignment_loss(
                bridge, ctx.current_semantic_teacher
            )
            if bool(supervised_rows.any().item()):
                losses["gist"] = self.retrospective_gist.future_alignment_loss(
                    required_gist,
                    target_effect,
                    sample_mask=supervised_rows,
                    magnitude_weight=cfg.magnitude_weight,
                )
            if bool(valid.any().item()):
                # A retrieved event was not executed in the current query, so
                # the query's future must never be used as a counterfactual
                # label for that candidate.  Each effect branch is supervised
                # only by the event's own stored factual pre/post transition.
                losses["effect"] = (
                    self.retrospective_event_adapter.effect_alignment_loss(
                        event,
                        observed_effect,
                        candidate_mask=valid,
                        magnitude_weight=cfg.magnitude_weight,
                    )
                )
            gate_target = utility_supervised_gate_target(
                selected_mean,
                ctx.target_action,
                selected_effect,
                target_effect,
                selection.memory_mask & supervised_rows,
                action_valid_mask=action_valid,
                effect_weight=cfg.gate_effect_weight,
                temperature=cfg.gate_temperature,
            )
            losses["gate"] = utility_supervised_gate_bce(
                gate, gate_target, sample_mask=supervised_rows
            )
            action_squared = (
                event.adapted_action_mean.float()
                - ctx.target_action.unsqueeze(1).float()
            ).square().mean(dim=-1)
            action_weights = action_valid[:, None, :].to(
                dtype=action_squared.dtype
            )
            squared = (action_squared * action_weights).sum(dim=-1) / (
                action_weights.sum(dim=-1).clamp(min=1.0)
            )
            per_row = (utility.probabilities * squared).sum(dim=1)
            adaptation_rows = utility.valid_rows & non_hard_rows
            if bool(adaptation_rows.any().item()):
                losses["adaptation"] = per_row[adaptation_rows].mean()

        auxiliary = (
            cfg.loss_retrieval * losses["retrieval"]
            + cfg.loss_bridge * losses["bridge"]
            + cfg.loss_gist * losses["gist"]
            + cfg.loss_effect * losses["effect"]
            + cfg.loss_gate * losses["gate"]
            + cfg.loss_adaptation * losses["adaptation"]
        )
        source_memory_mask = (
            torch.zeros_like(selection.memory_mask)
            if ablation_mode == "context_only"
            else selection.memory_mask
        )
        source_component_indices = torch.where(
            source_memory_mask,
            selection.component_indices,
            torch.zeros_like(selection.component_indices),
        )
        corruption_applied = candidate_payload.applied
        if phase == "infer" and memory_corruption == "wrong_event":
            corruption_applied = valid.any(dim=1)
        metrics = {
            "loss_warm_retrieval": losses["retrieval"],
            "loss_warm_bridge": losses["bridge"],
            "loss_warm_gist": losses["gist"],
            "loss_warm_effect": losses["effect"],
            "loss_warm_gate": losses["gate"],
            "loss_warm_adaptation": losses["adaptation"],
            "warm_gate_mean": gate_value.mean(),
            "warm_selected_memory_rate": selection.memory_mask.float().mean(),
            "warm_consequence_mean": selection_consistency[valid].mean()
            if bool(valid.any().item())
            else zero,
        }
        self._last_retrospection_diagnostics = {
            "candidate_indices": selection.candidate_indices.detach(),
            "gate": gate_value.detach(),
            "memory_relevance_gate": relevance_gate.detach(),
            "source_memory_mask": source_memory_mask.detach(),
            "consistency": selection_consistency.detach(),
            "predicted_consistency": consistency.detach(),
            "reranker_scores": reranker_scores.detach(),
            "required_transition": transition_gist.detach(),
            "configured_ablation_mode": configured_controls.ablation_mode,
            "configured_memory_corruption": configured_controls.memory_corruption,
            "ablation_mode": ablation_mode,
            "memory_corruption": memory_corruption,
            "experiment_id": configured_controls.experiment_id,
            "corruption_applied": corruption_applied.detach(),
            "corruption_fallback": corruption_fallback.detach(),
        }
        if phase == "infer":
            if current_video_latent is None:
                raise WarmRetrospectionError(
                    "online full WARM requires the factual current VAE latent"
                )
            if current_video_latent.ndim != 5 or current_video_latent.shape[2] != 1:
                raise WarmRetrospectionError(
                    "online factual VAE latent must have shape [B,C,1,H,W]"
                )
            # Keep the online event-change signal identical to the factual
            # feature cache: singleton time is removed and spatial features
            # are pooled to [C,4,8] before entering EpisodeMemory.
            factual_vae_latent = F.adaptive_avg_pool2d(
                current_video_latent[:, :, 0],
                output_size=(4, 8),
            )
            self._last_retrospection_diagnostics.update(
                {
                    "factual_world_tokens": bridge.semantic_tokens.detach().to(
                        device="cpu", dtype=torch.float32
                    ),
                    "factual_vae_latent": factual_vae_latent.detach().to(
                        device="cpu", dtype=torch.float32
                    ),
                    "factual_proprio": current_proprio.detach().to(
                        device="cpu", dtype=torch.float32
                    ),
                }
            )
        return ActionSourceOutput(
            source=source,
            base_gaussian=base_gaussian,
            component_indices=source_component_indices.detach(),
            memory_mask=source_memory_mask.detach(),
            selected_means=selected_mean.detach(),
            memory_sigma=float(cfg.source_sigma_min),
            conditioning_tokens=conditioning,
            auxiliary_loss=auxiliary,
            auxiliary_metrics=metrics,
            source_gate=gate_value,
        )

    def _context_from_online_facts(
        self,
        *,
        online_step: Any,
        facts: Any,
        episode_tokens: torch.Tensor | np.ndarray | None,
        episode_mask: torch.Tensor | np.ndarray | None,
        episode_action_summaries: torch.Tensor | np.ndarray | None,
        episode_action_mask: torch.Tensor | np.ndarray | None,
    ) -> RetrospectiveSourceContext:
        cfg = self._require_retrospection()

        def tensor(value: Any, dtype: torch.dtype) -> torch.Tensor:
            return torch.as_tensor(
                np.array(value, copy=True), device=self.device, dtype=dtype
            ).unsqueeze(0)

        if tuple(np.asarray(online_step.context_key).shape) != (cfg.context_dim,):
            raise WarmRetrospectionError("online context key width is incompatible")
        expected_k = int(np.asarray(facts.candidate_valid_mask).shape[0])
        expected_shapes = {
            "context_keys": (expected_k, cfg.context_dim),
            "effect_pre": (expected_k, 4, cfg.semantic_dim),
            "effect_delta": (expected_k, 4, cfg.semantic_dim),
            "start_proprio": (expected_k, cfg.proprio_dim),
            "gripper_timing": (expected_k, cfg.timing_dim),
            "support": (expected_k,),
        }
        for field, expected in expected_shapes.items():
            if tuple(np.asarray(getattr(facts, field)).shape) != expected:
                raise WarmRetrospectionError(
                    f"online {field} shape does not match complete WARM: "
                    f"{np.asarray(getattr(facts, field)).shape} != {expected}"
                )
        if tuple(np.asarray(online_step.candidate_means).shape) != (
            expected_k,
            cfg.action_horizon,
            cfg.action_dim,
        ):
            raise WarmRetrospectionError(
                "online candidate action shape does not match complete WARM"
            )

        if episode_tokens is None:
            episode = torch.zeros(
                (1, 1, cfg.semantic_dim),
                device=self.device,
                dtype=self.torch_dtype,
            )
            episode_valid = torch.zeros(
                (1, 1), device=self.device, dtype=torch.bool
            )
        else:
            episode = torch.as_tensor(
                episode_tokens, device=self.device, dtype=self.torch_dtype
            )
            if episode.ndim == 2:
                episode = episode.unsqueeze(0)
            if (
                episode.ndim != 3
                or episode.shape[0] != 1
                or episode.shape[2] != cfg.semantic_dim
            ):
                raise WarmRetrospectionError(
                    "online episode_tokens must be [N,semantic_dim]"
                )
            if episode_mask is None:
                episode_valid = torch.ones(
                    episode.shape[:2], device=self.device, dtype=torch.bool
                )
            else:
                episode_valid = torch.as_tensor(
                    episode_mask, device=self.device, dtype=torch.bool
                )
                if episode_valid.ndim == 1:
                    episode_valid = episode_valid.unsqueeze(0)
                if episode_valid.shape != episode.shape[:2]:
                    raise WarmRetrospectionError(
                        "online episode_mask must match episode_tokens"
                    )
        if episode_action_summaries is None:
            action_history = torch.zeros(
                (1, 1, cfg.episode_action_summary_dim),
                device=self.device,
                dtype=self.torch_dtype,
            )
            action_history_valid = torch.zeros(
                (1, 1), device=self.device, dtype=torch.bool
            )
        else:
            action_history = torch.as_tensor(
                episode_action_summaries,
                device=self.device,
                dtype=self.torch_dtype,
            )
            if action_history.ndim == 2:
                action_history = action_history.unsqueeze(0)
            if (
                action_history.ndim != 3
                or action_history.shape[0] != 1
                or action_history.shape[2] != cfg.episode_action_summary_dim
            ):
                raise WarmRetrospectionError(
                    "online episode_action_summaries have an invalid shape"
                )
            if episode_action_mask is None:
                action_history_valid = torch.ones(
                    action_history.shape[:2], device=self.device, dtype=torch.bool
                )
            else:
                action_history_valid = torch.as_tensor(
                    episode_action_mask, device=self.device, dtype=torch.bool
                )
                if action_history_valid.ndim == 1:
                    action_history_valid = action_history_valid.unsqueeze(0)
                if action_history_valid.shape != action_history.shape[:2]:
                    raise WarmRetrospectionError(
                        "online episode_action_mask must match action summaries"
                    )
        return RetrospectiveSourceContext(
            query_context=tensor(online_step.context_key, self.torch_dtype),
            candidate_context=tensor(facts.context_keys, self.torch_dtype),
            candidate_actions=tensor(
                online_step.candidate_means, self.torch_dtype
            ),
            candidate_start_proprio=tensor(
                facts.start_proprio, self.torch_dtype
            ),
            candidate_effect_pre=tensor(facts.effect_pre, self.torch_dtype),
            candidate_effect_delta=tensor(
                facts.effect_delta, self.torch_dtype
            ),
            candidate_timing=tensor(facts.gripper_timing, self.torch_dtype),
            candidate_support=tensor(facts.support, self.torch_dtype),
            candidate_valid_mask=tensor(
                facts.candidate_valid_mask, torch.bool
            ),
            episode_tokens=episode,
            episode_mask=episode_valid,
            episode_action_summaries=action_history,
            episode_action_mask=action_history_valid,
        )

    @torch.no_grad()
    def infer_action(
        self,
        *args: Any,
        online_step: object | None = None,
        episode_tokens: torch.Tensor | np.ndarray | None = None,
        episode_mask: torch.Tensor | np.ndarray | None = None,
        episode_action_summaries: torch.Tensor | np.ndarray | None = None,
        episode_action_mask: torch.Tensor | np.ndarray | None = None,
        negative_prompt: str | None = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: float | None = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ):
        """Run the full learned online path from one bound retrieval step."""

        if args:
            raise SourceTransportError(
                "full WARM online inputs are owned by BoundOnlineStep; only "
                "episode history and operational sampler options may be supplied"
            )
        retriever = self._warm_online_retriever
        if retriever is None:
            raise SourceTransportError(
                "full WARM requires bind_online_retriever before inference"
            )
        from fastwam.memory.online_retrieval import BoundOnlineStep

        if type(online_step) is not BoundOnlineStep:
            raise SourceTransportError(
                "full WARM infer_action requires a retriever BoundOnlineStep"
            )
        self._validate_online_step_model_binding(online_step)
        facts = retriever.validate_and_gather_candidate_facts(
            online_step,
            prompt=online_step.prompt,
            proprio=online_step.proprio,
            input_image=online_step.model_input,
        )
        source_context = self._context_from_online_facts(
            online_step=online_step,
            facts=facts,
            episode_tokens=episode_tokens,
            episode_mask=episode_mask,
            episode_action_summaries=episode_action_summaries,
            episode_action_mask=episode_action_mask,
        )
        output = FastWAM.infer_action(
            self,
            prompt=online_step.prompt,
            input_image=torch.tensor(
                online_step.model_input,
                dtype=self.torch_dtype,
                device=self.device,
            ),
            action_horizon=self._require_retrospection().action_horizon,
            proprio=(
                None
                if online_step.proprio is None
                else torch.tensor(
                    online_step.proprio,
                    dtype=self.torch_dtype,
                    device=self.device,
                )
            ),
            seed=int(online_step.derived_seed),
            negative_prompt=negative_prompt,
            text_cfg_scale=float(text_cfg_scale),
            num_inference_steps=int(num_inference_steps),
            sigma_shift=sigma_shift,
            rand_device=str(rand_device),
            tiled=bool(tiled),
            action_source_context=source_context,  # type: ignore[arg-type]
            memory_sigma=self._require_retrospection().source_sigma_min,
        )
        diagnostics = self._last_retrospection_diagnostics
        controls = self._online_experiment_controls()
        output["warm_retrospection"] = {
            "selected_candidate_index": int(
                diagnostics["candidate_indices"].reshape(-1)[0].cpu().item()
            ),
            "gate": float(diagnostics["gate"].reshape(-1)[0].cpu().item()),
            "online_step_sha256": online_step.step_sha256,
            "ablation_mode": controls.ablation_mode,
            "memory_corruption": controls.memory_corruption,
            "experiment_id": controls.experiment_id,
            "corruption_applied": bool(
                diagnostics["corruption_applied"].reshape(-1)[0].cpu().item()
            ),
            "corruption_fallback": bool(
                diagnostics["corruption_fallback"].reshape(-1)[0].cpu().item()
            ),
        }
        selected_index = int(
            diagnostics["candidate_indices"].reshape(-1)[0].cpu().item()
        )
        memory_selected = selected_index >= 0
        selected_event = (
            online_step.event_ids[selected_index] if memory_selected else None
        )
        source_memory_selected = bool(
            diagnostics["source_memory_mask"].reshape(-1)[0].cpu().item()
        )
        output["warm_online_telemetry"] = _json_safe(
            {
                "experiment": {
                    "experiment_id": controls.experiment_id,
                    "ablation_mode": controls.ablation_mode,
                    "memory_corruption": controls.memory_corruption,
                    "corruption_applied": bool(
                        diagnostics["corruption_applied"]
                        .reshape(-1)[0]
                        .cpu()
                        .item()
                    ),
                    "corruption_fallback": bool(
                        diagnostics["corruption_fallback"]
                        .reshape(-1)[0]
                        .cpu()
                        .item()
                    ),
                },
                "retrieval": _fixed_retrieval_telemetry(online_step),
                "source": {
                    "policy": "consequence_aligned_retrospection",
                    "component": (
                        selected_index + 1 if source_memory_selected else 0
                    ),
                    "selected_rank": selected_index if memory_selected else None,
                    "selected_event_id": _event_id_json(selected_event),
                    "candidate_selected": memory_selected,
                    "memory_selected": source_memory_selected,
                    "memory_sigma": float(
                        self._require_retrospection().source_sigma_min
                    ),
                    "gate": float(
                        diagnostics["gate"].reshape(-1)[0].cpu().item()
                    ),
                    "memory_relevance_gate": float(
                        diagnostics["memory_relevance_gate"]
                        .reshape(-1)[0]
                        .cpu()
                        .item()
                    ),
                    "derived_seed": int(online_step.derived_seed),
                },
            },
            field="complete WARM online telemetry",
        )
        output["warm_factual_observation"] = {
            "world_tokens": diagnostics["factual_world_tokens"][0],
            "vae_latent": diagnostics["factual_vae_latent"][0],
            "proprio": diagnostics["factual_proprio"].reshape(-1),
        }
        return output

    def _retrospection_state(self) -> dict[str, Any]:
        cfg = self._require_retrospection()
        config = cfg.to_json_dict()
        return {
            "schema": WARM_RETROSPECTION_CHECKPOINT_SCHEMA,
            "version": WARM_RETROSPECTION_CHECKPOINT_VERSION,
            "config": config,
            "config_sha256": sha256_canonical_json(config),
        }

    def _checkpoint_extra_state(self) -> dict[str, Any]:
        payload = super()._checkpoint_extra_state()
        payload["warm_retrospection"] = self._retrospection_state()
        module_names = (
            "semantic_bridge",
            "retrospective_gist",
            "retrospective_event_adapter",
            "utility_reranker",
            "source_confidence_gate",
            "episode_action_projection",
            "gist_to_text",
            "action_context_to_text",
            "video_layer_adapters",
        )
        payload["warm_retrospection_modules"] = {
            name: getattr(self, name).state_dict() for name in module_names
        }
        return payload

    def _validate_retrospection_state(self, value: object) -> None:
        fields = {"schema", "version", "config", "config_sha256"}
        if not isinstance(value, Mapping) or set(value) != fields:
            actual = set(value) if isinstance(value, Mapping) else set()
            raise ValueError(
                "invalid retrospection checkpoint fields; "
                f"missing={sorted(fields - actual)}, "
                f"extra={sorted(actual - fields)}"
            )
        if (
            value["schema"] != WARM_RETROSPECTION_CHECKPOINT_SCHEMA
            or value["version"] != WARM_RETROSPECTION_CHECKPOINT_VERSION
        ):
            raise ValueError("unsupported retrospection checkpoint schema/version")
        config_value = value["config"]
        if not isinstance(config_value, Mapping):
            raise ValueError("retrospection checkpoint config must be a mapping")
        try:
            loaded = WarmRetrospectionConfig.from_dict(config_value)
        except (TypeError, ValueError) as error:
            raise ValueError("invalid retrospection checkpoint config") from error
        loaded_json = loaded.to_json_dict()
        loaded_sha256 = sha256_canonical_json(loaded_json)
        if value["config_sha256"] != loaded_sha256:
            raise ValueError("retrospection checkpoint config_sha256 is invalid")

        configured = self._require_retrospection()
        if loaded != configured:
            loaded_dict = loaded.to_dict()
            configured_dict = configured.to_dict()
            differing = sorted(
                field
                for field in configured_dict
                if loaded_dict[field] != configured_dict[field]
            )
            raise ValueError(
                "checkpoint retrospection architecture does not match "
                "configured WARM: "
                + ", ".join(differing)
            )

    def _preflight_checkpoint_extra_state(self, payload: dict[str, Any]) -> None:
        super()._preflight_checkpoint_extra_state(payload)
        self._validate_retrospection_state(payload.get("warm_retrospection"))
        modules = payload.get("warm_retrospection_modules")
        if not isinstance(modules, Mapping):
            raise ValueError("checkpoint is missing WARM retrospective modules")
        expected_modules = {
            "semantic_bridge",
            "retrospective_gist",
            "retrospective_event_adapter",
            "utility_reranker",
            "source_confidence_gate",
            "episode_action_projection",
            "gist_to_text",
            "action_context_to_text",
            "video_layer_adapters",
        }
        if set(modules) != expected_modules:
            raise ValueError("checkpoint WARM module set is incomplete")
        # Validate every key and tensor shape before the base checkpoint loader
        # mutates this model.  A malformed retrospective payload therefore
        # fails atomically instead of leaving a partially updated MoT.
        for name in sorted(expected_modules):
            supplied = modules[name]
            if not isinstance(supplied, Mapping):
                raise ValueError(
                    f"checkpoint WARM module {name!r} state must be a mapping"
                )
            expected_state = getattr(self, name).state_dict()
            if set(supplied) != set(expected_state):
                raise ValueError(
                    f"checkpoint WARM module {name!r} state keys are incompatible"
                )
            for key, expected_tensor in expected_state.items():
                value = supplied[key]
                if not isinstance(value, torch.Tensor):
                    raise ValueError(
                        f"checkpoint WARM module {name!r}.{key} must be a tensor"
                    )
                if tuple(value.shape) != tuple(expected_tensor.shape):
                    raise ValueError(
                        f"checkpoint WARM module {name!r}.{key} shape "
                        f"{tuple(value.shape)} != {tuple(expected_tensor.shape)}"
                    )

    def _load_checkpoint_extra_state(self, payload: dict[str, Any]) -> None:
        modules = payload["warm_retrospection_modules"]
        expected = {
            "semantic_bridge",
            "retrospective_gist",
            "retrospective_event_adapter",
            "utility_reranker",
            "source_confidence_gate",
            "episode_action_projection",
            "gist_to_text",
            "action_context_to_text",
            "video_layer_adapters",
        }
        if set(modules) != expected:
            raise ValueError("checkpoint WARM module set is incomplete")
        for name in sorted(expected):
            getattr(self, name).load_state_dict(modules[name], strict=True)

    def trainer_state_metadata(self) -> dict[str, Any]:
        return {
            "warm_source": super().trainer_state_metadata(),
            "warm_retrospection": self._retrospection_state(),
        }

    def validate_trainer_state_metadata(self, value: object) -> None:
        if not isinstance(value, Mapping) or set(value) != {
            "warm_source",
            "warm_retrospection",
        }:
            raise ValueError("trainer state lacks complete WARM metadata")
        self._validate_warm_source_state(value["warm_source"])
        self._validate_retrospection_state(value["warm_retrospection"])

    def training_attestation_metadata(self) -> dict[str, Any]:
        # Keep the closed source-attestation schema owned by WarmSourceFastWAM.
        # The resolved training-config digest already binds the complete
        # retrospection config, while checkpoint extra state independently
        # binds and validates its canonical SHA256.
        return super().training_attestation_metadata()


__all__ = [
    "OnlineExperimentControls",
    "RetrospectiveSourceContext",
    "WARM_ONLINE_ABLATION_MODES",
    "WARM_ONLINE_MEMORY_CORRUPTIONS",
    "WARM_RETROSPECTION_CHECKPOINT_SCHEMA",
    "WARM_RETROSPECTION_CHECKPOINT_VERSION",
    "WarmRetrospectionError",
    "WarmRetrospectionFastWAM",
]
