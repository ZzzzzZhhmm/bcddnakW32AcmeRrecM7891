"""Bounded action/effect/context adapter for retrieved retrospective events."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .adapter_config import RetrospectiveEventAdapterConfig


class RetrospectiveEventAdapterError(ValueError):
    """Raised when candidate payloads or current context are inconsistent."""


def _finite_tensor(
    value: object,
    *,
    field: str,
    rank: int,
    width: int,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{field} must be a torch.Tensor")
    if value.ndim != rank or value.shape[-1] != width:
        raise RetrospectiveEventAdapterError(
            f"{field} must have rank {rank} and trailing width {width}, "
            f"got {tuple(value.shape)}"
        )
    if any(int(size) <= 0 for size in value.shape[:-1]):
        raise RetrospectiveEventAdapterError(
            f"{field} must have non-empty leading dimensions"
        )
    if not value.is_floating_point():
        raise TypeError(f"{field} must have a floating dtype")
    if not bool(torch.isfinite(value).all().item()):
        raise RetrospectiveEventAdapterError(
            f"{field} must contain only finite values"
        )
    return value


def _bool_mask(
    value: object,
    *,
    field: str,
    shape: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{field} must be a torch.Tensor")
    if value.dtype != torch.bool:
        raise TypeError(f"{field} must have dtype bool")
    if tuple(value.shape) != shape:
        raise RetrospectiveEventAdapterError(
            f"{field} must have shape {shape}, got {tuple(value.shape)}"
        )
    if value.device != device:
        raise RetrospectiveEventAdapterError(
            f"{field} must share the candidate device"
        )
    return value


def _masked_mean(tokens: torch.Tensor, mask: torch.Tensor, *, dim: int) -> torch.Tensor:
    weights = mask.to(dtype=tokens.dtype).unsqueeze(-1)
    denominator = weights.sum(dim=dim).clamp(min=1.0)
    return (tokens * weights).sum(dim=dim) / denominator


@dataclass(frozen=True, slots=True)
class RetrospectiveEventAdapterOutput:
    """Per-candidate source mean, semantic effect, and action context."""

    adapted_action_mean: torch.Tensor
    action_residual: torch.Tensor
    predicted_effect: torch.Tensor
    action_context_tokens: torch.Tensor
    candidate_valid_mask: torch.Tensor


class RetrospectiveEventAdapter(nn.Module):
    """Adapt factual normalized action exemplars without becoming a second policy.

    ``warped_actions`` is a legacy interface name.  In the complete LIBERO
    path it contains the bank event in the same normalized model action space
    as the current target; this module does not claim geometric
    canonicalization or cross-embodiment warping. ``gripper_timing`` is the
    compact candidate-level timing vector; its
    configured width must include any phase-validity bits used by the dataset,
    so an absent phase is never represented as an unmarked fabricated value.
    The learnable deformation is bounded elementwise by ``rho`` times a
    configured per-dimension scale.  The default scale is one normalized
    action unit, not an estimated dataset standard deviation.
    Invalid padded candidates must carry zero action and timing payloads and
    are kept exactly zero in every output branch.
    """

    def __init__(
        self,
        config: RetrospectiveEventAdapterConfig,
        *,
        action_std: torch.Tensor | Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(config, RetrospectiveEventAdapterConfig):
            raise TypeError("config must be RetrospectiveEventAdapterConfig")
        self.config = config
        if action_std is None:
            std = torch.ones(config.action_dim, dtype=torch.float32)
        else:
            std = torch.as_tensor(action_std, dtype=torch.float32).detach().clone()
        if tuple(std.shape) != (config.action_dim,):
            raise RetrospectiveEventAdapterError(
                "action_std must have shape " f"{(config.action_dim,)}"
            )
        if not bool(torch.isfinite(std).all().item()) or not bool(
            (std > 0).all().item()
        ):
            raise RetrospectiveEventAdapterError(
                "action_std must contain finite positive scales"
            )
        self.register_buffer("action_std", std, persistent=True)

        dim = config.model_dim
        self.action_projection = nn.Linear(config.action_dim, dim)
        self.timing_projection = nn.Linear(config.timing_dim, dim)
        self.temporal_embedding = nn.Parameter(
            torch.empty(config.action_horizon, dim)
        )
        nn.init.normal_(self.temporal_embedding, mean=0.0, std=0.02)

        self.world_projection = nn.Linear(config.world_dim, dim)
        self.gist_projection = nn.Linear(config.gist_dim, dim)
        self.proprio_projection = nn.Linear(config.proprio_dim, dim)
        self.text_projection = nn.Linear(config.text_dim, dim)
        self.context_type_embeddings = nn.Parameter(torch.empty(4, dim))
        nn.init.normal_(self.context_type_embeddings, mean=0.0, std=0.02)

        self.action_query_norm = nn.LayerNorm(dim)
        self.global_context_norm = nn.LayerNorm(dim)
        self.action_cross_attention = nn.MultiheadAttention(
            dim,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.action_cross_norm = nn.LayerNorm(dim)
        hidden = max(int(dim * config.mlp_ratio), 1)
        self.action_feed_forward = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(config.dropout),
        )
        self.action_output_norm = nn.LayerNorm(dim)

        self.residual_head = nn.Linear(dim, config.action_dim)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

        self.event_delta_projection = nn.Linear(config.event_dim, dim)
        # The consequence branch is deliberately candidate-factual: it sees
        # the stored action, stored effect and current world, but never the
        # required-transition gist or language-conditioned action hidden.
        # This prevents the effect predictor from copying the very target it
        # is later compared against.
        self.effect_action_projection = nn.Linear(config.action_dim, dim)
        self.effect_mixer = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Linear(dim, config.effect_dim),
        )
        effect_residual = self.effect_mixer[-1]
        assert isinstance(effect_residual, nn.Linear)
        nn.init.zeros_(effect_residual.weight)
        nn.init.zeros_(effect_residual.bias)
        self.effect_to_context = nn.Linear(config.effect_dim, dim)

        self.action_context_queries = nn.Parameter(
            torch.empty(config.context_query_count, dim)
        )
        nn.init.normal_(self.action_context_queries, mean=0.0, std=0.02)
        self.action_context_source_norm = nn.LayerNorm(dim)
        self.action_context_attention = nn.MultiheadAttention(
            dim,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.action_context_output_projection = nn.Linear(dim, dim)
        # The context branch is initially inert and becomes visible only after
        # training supplies evidence that it helps Action DiT.
        nn.init.zeros_(self.action_context_output_projection.weight)
        nn.init.zeros_(self.action_context_output_projection.bias)

    def _global_context(
        self,
        *,
        world: torch.Tensor,
        world_mask: torch.Tensor,
        gist: torch.Tensor,
        proprio: torch.Tensor,
        text: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        world_projected = self.world_projection(world)
        gist_projected = self.gist_projection(gist)
        proprio_projected = self.proprio_projection(proprio).unsqueeze(1)
        text_projected = self.text_projection(text)
        groups = (
            world_projected + self.context_type_embeddings[0],
            gist_projected + self.context_type_embeddings[1],
            proprio_projected + self.context_type_embeddings[2],
            text_projected + self.context_type_embeddings[3],
        )
        masks = (
            world_mask,
            torch.ones(gist.shape[:2], dtype=torch.bool, device=gist.device),
            torch.ones(
                (gist.shape[0], 1), dtype=torch.bool, device=gist.device
            ),
            text_mask,
        )
        context = torch.cat(groups, dim=1)
        context_mask = torch.cat(masks, dim=1)
        world_pooled = _masked_mean(
            world_projected, world_mask, dim=1
        )
        return context, context_mask, world_pooled

    def forward(
        self,
        *,
        warped_actions: torch.Tensor,
        gripper_timing: torch.Tensor,
        candidate_valid_mask: torch.Tensor,
        world_tokens: torch.Tensor,
        world_mask: torch.Tensor,
        gist_tokens: torch.Tensor,
        proprio: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        event_delta_tokens: torch.Tensor,
        event_delta_mask: torch.Tensor,
    ) -> RetrospectiveEventAdapterOutput:
        cfg = self.config
        actions = _finite_tensor(
            warped_actions,
            field="warped_actions",
            rank=4,
            width=cfg.action_dim,
        )
        batch, candidates, horizon, _ = actions.shape
        if int(horizon) != cfg.action_horizon:
            raise RetrospectiveEventAdapterError(
                "warped_actions horizon does not match the configured action horizon"
            )
        timing = _finite_tensor(
            gripper_timing,
            field="gripper_timing",
            rank=3,
            width=cfg.timing_dim,
        )
        if tuple(timing.shape[:2]) != (batch, candidates):
            raise RetrospectiveEventAdapterError(
                "gripper_timing must share warped_actions [B,K]"
            )
        world = _finite_tensor(
            world_tokens, field="world_tokens", rank=3, width=cfg.world_dim
        )
        gist = _finite_tensor(
            gist_tokens, field="gist_tokens", rank=3, width=cfg.gist_dim
        )
        state = _finite_tensor(
            proprio, field="proprio", rank=2, width=cfg.proprio_dim
        )
        text = _finite_tensor(
            text_tokens, field="text_tokens", rank=3, width=cfg.text_dim
        )
        event_delta = _finite_tensor(
            event_delta_tokens,
            field="event_delta_tokens",
            rank=4,
            width=cfg.event_dim,
        )
        if gist.shape[1] != 8:
            raise RetrospectiveEventAdapterError(
                "gist_tokens must contain the eight RetrospectiveGistAdapter queries"
            )
        if tuple(event_delta.shape[:2]) != (batch, candidates):
            raise RetrospectiveEventAdapterError(
                "event_delta_tokens must share warped_actions [B,K]"
            )
        other = (timing, world, gist, state, text, event_delta)
        if any(int(value.shape[0]) != batch for value in other):
            raise RetrospectiveEventAdapterError(
                "all event-adapter inputs must share batch size"
            )
        if any(
            value.device != actions.device or value.dtype != actions.dtype
            for value in other
        ):
            raise RetrospectiveEventAdapterError(
                "all floating inputs must share device and dtype"
            )

        valid = _bool_mask(
            candidate_valid_mask,
            field="candidate_valid_mask",
            shape=(batch, candidates),
            device=actions.device,
        )
        counts = valid.sum(dim=1)
        positions = torch.arange(candidates, device=actions.device).unsqueeze(0)
        if not bool((valid == (positions < counts.unsqueeze(1))).all().item()):
            raise RetrospectiveEventAdapterError(
                "candidate_valid_mask must be a true prefix in every sample"
            )
        world_valid = _bool_mask(
            world_mask,
            field="world_mask",
            shape=(batch, int(world.shape[1])),
            device=actions.device,
        )
        text_valid = _bool_mask(
            text_mask,
            field="text_mask",
            shape=(batch, int(text.shape[1])),
            device=actions.device,
        )
        if not bool(world_valid.any(dim=1).all().item()) or not bool(
            text_valid.any(dim=1).all().item()
        ):
            raise RetrospectiveEventAdapterError(
                "world_mask and text_mask must retain a token per sample"
            )
        delta_valid = _bool_mask(
            event_delta_mask,
            field="event_delta_mask",
            shape=(batch, candidates, int(event_delta.shape[2])),
            device=actions.device,
        )
        delta_any = delta_valid.any(dim=2)
        if bool((delta_any & ~valid).any().item()) or not bool(
            (delta_any | ~valid).all().item()
        ):
            raise RetrospectiveEventAdapterError(
                "each valid candidate needs event delta evidence and padded candidates need none"
            )
        invalid = ~valid
        if bool((actions[invalid] != 0).any().item()) or bool(
            (timing[invalid] != 0).any().item()
        ):
            raise RetrospectiveEventAdapterError(
                "invalid candidate action and timing payloads must be zero"
            )

        context, context_mask, world_pooled = self._global_context(
            world=world,
            world_mask=world_valid,
            gist=gist,
            proprio=state,
            text=text,
            text_mask=text_valid,
        )
        action_tokens = (
            self.action_projection(actions)
            + self.timing_projection(timing).unsqueeze(2)
            + self.temporal_embedding.view(1, 1, horizon, cfg.model_dim)
        )
        action_flat = action_tokens.reshape(
            batch * candidates, horizon, cfg.model_dim
        )
        context_flat = context.unsqueeze(1).expand(
            -1, candidates, -1, -1
        ).reshape(batch * candidates, context.shape[1], cfg.model_dim)
        context_mask_flat = context_mask.unsqueeze(1).expand(
            -1, candidates, -1
        ).reshape(batch * candidates, context_mask.shape[1])
        attended, _ = self.action_cross_attention(
            self.action_query_norm(action_flat),
            self.global_context_norm(context_flat),
            self.global_context_norm(context_flat),
            key_padding_mask=~context_mask_flat,
            need_weights=False,
        )
        hidden_flat = self.action_cross_norm(action_flat + attended)
        hidden_flat = self.action_output_norm(
            hidden_flat + self.action_feed_forward(hidden_flat)
        )
        hidden = hidden_flat.reshape(
            batch, candidates, horizon, cfg.model_dim
        )

        scale = (cfg.rho * self.action_std).to(dtype=actions.dtype).view(
            1, 1, 1, cfg.action_dim
        )
        residual = scale * torch.tanh(self.residual_head(hidden))
        residual = residual.masked_fill(~valid[:, :, None, None], 0.0)
        adapted = (actions + residual).masked_fill(
            ~valid[:, :, None, None], 0.0
        )

        delta_projected = self.event_delta_projection(event_delta)
        delta_projected = delta_projected.masked_fill(
            ~delta_valid.unsqueeze(-1), 0.0
        )
        pooled_delta = _masked_mean(delta_projected, delta_valid, dim=2)
        pooled_observed_effect = _masked_mean(
            event_delta, delta_valid, dim=2
        )
        pooled_action = self.effect_action_projection(actions).mean(dim=2)
        expanded_world = world_pooled.unsqueeze(1).expand(-1, candidates, -1)
        # Predict a bounded-in-spirit residual around the immutable factual
        # event effect.  Zero initialization makes the initial consequence
        # exactly the stored observation and preserves distinct candidates.
        predicted_effect = pooled_observed_effect + self.effect_mixer(
            torch.cat((pooled_action, pooled_delta, expanded_world), dim=-1)
        )
        predicted_effect = predicted_effect.masked_fill(
            ~valid.unsqueeze(-1), 0.0
        )

        effect_token = self.effect_to_context(predicted_effect).unsqueeze(2)
        context_source = torch.cat(
            (hidden, delta_projected, effect_token), dim=2
        )
        context_source_flat = context_source.reshape(
            batch * candidates, context_source.shape[2], cfg.model_dim
        )
        context_source_mask = torch.cat(
            (
                torch.ones(
                    (batch, candidates, horizon),
                    dtype=torch.bool,
                    device=actions.device,
                ),
                delta_valid,
                valid.unsqueeze(-1),
            ),
            dim=2,
        ).reshape(batch * candidates, -1)
        context_queries = self.action_context_queries.unsqueeze(0).expand(
            batch * candidates, -1, -1
        )
        context_hidden, _ = self.action_context_attention(
            context_queries,
            self.action_context_source_norm(context_source_flat),
            self.action_context_source_norm(context_source_flat),
            key_padding_mask=~context_source_mask,
            need_weights=False,
        )
        action_context = self.action_context_output_projection(context_hidden)
        action_context = action_context.reshape(
            batch,
            candidates,
            cfg.context_query_count,
            cfg.model_dim,
        ).masked_fill(~valid[:, :, None, None], 0.0)

        return RetrospectiveEventAdapterOutput(
            adapted_action_mean=adapted,
            action_residual=residual,
            predicted_effect=predicted_effect,
            action_context_tokens=action_context,
            candidate_valid_mask=valid,
        )

    def effect_alignment_loss(
        self,
        output: RetrospectiveEventAdapterOutput | torch.Tensor,
        semantic_delta_target: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
        candidate_weights: torch.Tensor | None = None,
        *,
        magnitude_weight: float = 0.25,
    ) -> torch.Tensor:
        """Align effects to factual or utility-weighted detached targets.

        ``semantic_delta_target`` may be one target per query ``[B,E]`` or a
        distinct factual target per candidate ``[B,K,E]``.  The latter is the
        default training use and prevents all retrieved events from collapsing
        to the demonstrated query future.  Direction and log-magnitude are
        both supervised because consequence scoring consumes both quantities.
        """

        predicted = output.predicted_effect if isinstance(
            output, RetrospectiveEventAdapterOutput
        ) else output
        predicted = _finite_tensor(
            predicted,
            field="predicted_effect",
            rank=3,
            width=self.config.effect_dim,
        )
        batch, candidates, _ = predicted.shape
        if not isinstance(semantic_delta_target, torch.Tensor):
            raise TypeError("semantic_delta_target must be a torch.Tensor")
        if semantic_delta_target.ndim == 2:
            target = _finite_tensor(
                semantic_delta_target,
                field="semantic_delta_target",
                rank=2,
                width=self.config.effect_dim,
            )
            target = target.unsqueeze(1).expand_as(predicted)
        elif semantic_delta_target.ndim == 3:
            target = _finite_tensor(
                semantic_delta_target,
                field="semantic_delta_target",
                rank=3,
                width=self.config.effect_dim,
            )
            if target.shape != predicted.shape:
                raise RetrospectiveEventAdapterError(
                    "candidate semantic targets must match predicted_effect"
                )
        else:
            raise RetrospectiveEventAdapterError(
                "semantic_delta_target must be [B,E] or [B,K,E]"
            )
        if target.shape[0] != batch or target.device != predicted.device:
            raise RetrospectiveEventAdapterError(
                "semantic_delta_target must share predicted effect batch/device"
            )
        if candidate_mask is None:
            if isinstance(output, RetrospectiveEventAdapterOutput):
                valid = output.candidate_valid_mask
            else:
                raise RetrospectiveEventAdapterError(
                    "candidate_mask is required for a raw predicted-effect tensor"
                )
        else:
            valid = _bool_mask(
                candidate_mask,
                field="candidate_mask",
                shape=(batch, candidates),
                device=predicted.device,
            )
        if not bool(valid.any().item()):
            raise RetrospectiveEventAdapterError(
                "effect alignment requires at least one valid candidate"
            )
        if (
            isinstance(magnitude_weight, bool)
            or not isinstance(magnitude_weight, (int, float))
            or not math.isfinite(float(magnitude_weight))
            or not 0.0 <= float(magnitude_weight)
        ):
            raise RetrospectiveEventAdapterError(
                "magnitude_weight must be a finite non-negative number"
            )
        weights = valid.to(dtype=torch.float32)
        if candidate_weights is not None:
            if (
                not isinstance(candidate_weights, torch.Tensor)
                or candidate_weights.shape != valid.shape
                or not candidate_weights.is_floating_point()
                or candidate_weights.device != predicted.device
                or not bool(torch.isfinite(candidate_weights).all().item())
                or bool((candidate_weights < 0).any().item())
            ):
                raise RetrospectiveEventAdapterError(
                    "candidate_weights must be finite non-negative [B,K]"
                )
            if bool((candidate_weights.masked_select(~valid) != 0).any().item()):
                raise RetrospectiveEventAdapterError(
                    "candidate_weights must be zero outside candidate_mask"
                )
            weights = candidate_weights.float()
        denominator = weights.sum()
        if not bool((denominator > 0).item()):
            raise RetrospectiveEventAdapterError(
                "effect alignment requires positive candidate weight"
            )
        detached_target = target.detach()
        cosine = F.cosine_similarity(
            predicted.float(),
            detached_target.float(),
            dim=-1,
            eps=self.config.cosine_eps,
        ).clamp(min=-1.0, max=1.0)
        predicted_norm = torch.linalg.vector_norm(predicted.float(), dim=-1)
        target_norm = torch.linalg.vector_norm(detached_target.float(), dim=-1)
        magnitude = F.smooth_l1_loss(
            torch.log1p(predicted_norm),
            torch.log1p(target_norm),
            reduction="none",
        )
        per_candidate = 1.0 - cosine + float(magnitude_weight) * magnitude
        return (per_candidate * weights).sum() / denominator


__all__ = [
    "RetrospectiveEventAdapter",
    "RetrospectiveEventAdapterError",
    "RetrospectiveEventAdapterOutput",
]
