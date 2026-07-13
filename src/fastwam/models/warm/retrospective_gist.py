"""Two-block Perceiver-style retrospective transition gist adapter."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .adapter_config import RetrospectiveGistConfig


class RetrospectiveGistError(ValueError):
    """Raised when heterogeneous retrospective inputs are inconsistent."""


def _tokens(
    value: object,
    *,
    field: str,
    width: int,
    rank: int = 3,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{field} must be a torch.Tensor")
    if value.ndim != rank or value.shape[-1] != width:
        shape = "[B,N,D]" if rank == 3 else "[B,K,N,D]"
        raise RetrospectiveGistError(
            f"{field} must have shape {shape} with D={width}, got {tuple(value.shape)}"
        )
    if any(int(size) <= 0 for size in value.shape[:-1]):
        raise RetrospectiveGistError(f"{field} must have non-empty token axes")
    if not value.is_floating_point():
        raise TypeError(f"{field} must have a floating dtype")
    if not bool(torch.isfinite(value).all().item()):
        raise RetrospectiveGistError(f"{field} must contain finite values")
    return value


def _mask(
    value: object,
    *,
    field: str,
    shape: tuple[int, ...],
    device: torch.device,
    require_each_sample: bool,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{field} must be a torch.Tensor")
    if value.dtype != torch.bool:
        raise TypeError(f"{field} must have dtype bool")
    if tuple(value.shape) != shape:
        raise RetrospectiveGistError(
            f"{field} must have shape {shape}, got {tuple(value.shape)}"
        )
    if value.device != device:
        raise RetrospectiveGistError(f"{field} must share its token device")
    if require_each_sample:
        flattened = value.reshape(value.shape[0], -1)
        if not bool(flattened.any(dim=1).all().item()):
            raise RetrospectiveGistError(
                f"{field} must retain at least one token for every sample"
            )
    return value


class _PerceiverCrossBlock(nn.Module):
    def __init__(self, config: RetrospectiveGistConfig) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(config.model_dim)
        self.source_norm = nn.LayerNorm(config.model_dim)
        self.cross_attention = nn.MultiheadAttention(
            config.model_dim,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.cross_output_norm = nn.LayerNorm(config.model_dim)
        hidden = max(int(config.model_dim * config.mlp_ratio), 1)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.model_dim, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, config.model_dim),
            nn.Dropout(config.dropout),
        )
        self.output_norm = nn.LayerNorm(config.model_dim)

    def forward(
        self,
        queries: torch.Tensor,
        source: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> torch.Tensor:
        attended, _ = self.cross_attention(
            self.query_norm(queries),
            self.source_norm(source),
            self.source_norm(source),
            key_padding_mask=~source_mask,
            need_weights=False,
        )
        hidden = self.cross_output_norm(queries + attended)
        return self.output_norm(hidden + self.feed_forward(hidden))


@dataclass(frozen=True, slots=True)
class RetrospectiveGistOutput:
    """Eight predictive gist tokens and their pooled transition embedding."""

    gist_tokens: torch.Tensor
    pooled_gist: torch.Tensor
    source_mask: torch.Tensor


class RetrospectiveGistAdapter(nn.Module):
    """Fuse current world state, episode history, and retrieved event evidence.

    Event inputs are explicit ``[B,K,N,D]`` tensors.  They are flattened only
    after their candidate/token mask has been validated, so invalid padded
    events cannot become visible to either Perceiver block.
    """

    _SOURCE_NAMES = (
        "world",
        "semantic",
        "episode",
        "event_pre",
        "event_delta",
        "text",
    )

    def __init__(self, config: RetrospectiveGistConfig) -> None:
        super().__init__()
        if not isinstance(config, RetrospectiveGistConfig):
            raise TypeError("config must be RetrospectiveGistConfig")
        self.config = config
        self.projections = nn.ModuleDict(
            {
                "world": nn.Linear(config.world_dim, config.model_dim),
                "semantic": nn.Linear(config.semantic_dim, config.model_dim),
                "episode": nn.Linear(config.episode_dim, config.model_dim),
                "event_pre": nn.Linear(config.event_dim, config.model_dim),
                "event_delta": nn.Linear(config.event_dim, config.model_dim),
                "text": nn.Linear(config.text_dim, config.model_dim),
            }
        )
        self.type_embeddings = nn.Parameter(
            torch.empty(len(self._SOURCE_NAMES), config.model_dim)
        )
        nn.init.normal_(self.type_embeddings, mean=0.0, std=0.02)
        self.gist_queries = nn.Parameter(
            torch.empty(config.query_count, config.model_dim)
        )
        nn.init.normal_(self.gist_queries, mean=0.0, std=0.02)
        # The method specifies exactly two lightweight cross-attention blocks.
        self.blocks = nn.ModuleList(
            [_PerceiverCrossBlock(config), _PerceiverCrossBlock(config)]
        )
        self.output_norm = nn.LayerNorm(config.model_dim)
        self.future_projection = nn.Linear(
            config.model_dim, config.future_target_dim
        )

    def _project_group(
        self,
        name: str,
        tokens: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        index = self._SOURCE_NAMES.index(name)
        projected = self.projections[name](tokens)
        projected = projected + self.type_embeddings[index].to(
            dtype=projected.dtype
        )
        return projected.masked_fill(~mask.unsqueeze(-1), 0.0)

    def forward(
        self,
        *,
        world_tokens: torch.Tensor,
        world_mask: torch.Tensor,
        semantic_tokens: torch.Tensor,
        semantic_mask: torch.Tensor,
        episode_tokens: torch.Tensor,
        episode_mask: torch.Tensor,
        event_pre_tokens: torch.Tensor,
        event_delta_tokens: torch.Tensor,
        event_token_mask: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> RetrospectiveGistOutput:
        cfg = self.config
        world = _tokens(
            world_tokens, field="world_tokens", width=cfg.world_dim
        )
        semantic = _tokens(
            semantic_tokens, field="semantic_tokens", width=cfg.semantic_dim
        )
        episode = _tokens(
            episode_tokens, field="episode_tokens", width=cfg.episode_dim
        )
        event_pre = _tokens(
            event_pre_tokens,
            field="event_pre_tokens",
            width=cfg.event_dim,
            rank=4,
        )
        event_delta = _tokens(
            event_delta_tokens,
            field="event_delta_tokens",
            width=cfg.event_dim,
            rank=4,
        )
        text = _tokens(text_tokens, field="text_tokens", width=cfg.text_dim)
        batch = int(world.shape[0])
        float_inputs = (semantic, episode, event_pre, event_delta, text)
        if any(int(value.shape[0]) != batch for value in float_inputs):
            raise RetrospectiveGistError("all token groups must share batch size")
        if event_pre.shape != event_delta.shape:
            raise RetrospectiveGistError(
                "event_pre_tokens and event_delta_tokens must have identical shapes"
            )
        if any(
            value.device != world.device or value.dtype != world.dtype
            for value in float_inputs
        ):
            raise RetrospectiveGistError(
                "all token groups must share device and floating dtype"
            )

        world_valid = _mask(
            world_mask,
            field="world_mask",
            shape=(batch, int(world.shape[1])),
            device=world.device,
            require_each_sample=True,
        )
        semantic_valid = _mask(
            semantic_mask,
            field="semantic_mask",
            shape=(batch, int(semantic.shape[1])),
            device=world.device,
            require_each_sample=True,
        )
        episode_valid = _mask(
            episode_mask,
            field="episode_mask",
            shape=(batch, int(episode.shape[1])),
            device=world.device,
            require_each_sample=False,
        )
        event_valid = _mask(
            event_token_mask,
            field="event_token_mask",
            shape=tuple(int(size) for size in event_pre.shape[:3]),
            device=world.device,
            require_each_sample=False,
        )
        text_valid = _mask(
            text_mask,
            field="text_mask",
            shape=(batch, int(text.shape[1])),
            device=world.device,
            require_each_sample=True,
        )

        event_pre_flat = event_pre.reshape(batch, -1, cfg.event_dim)
        event_delta_flat = event_delta.reshape(batch, -1, cfg.event_dim)
        event_valid_flat = event_valid.reshape(batch, -1)
        groups = (
            self._project_group("world", world, world_valid),
            self._project_group("semantic", semantic, semantic_valid),
            self._project_group("episode", episode, episode_valid),
            self._project_group(
                "event_pre", event_pre_flat, event_valid_flat
            ),
            self._project_group(
                "event_delta", event_delta_flat, event_valid_flat
            ),
            self._project_group("text", text, text_valid),
        )
        masks = (
            world_valid,
            semantic_valid,
            episode_valid,
            event_valid_flat,
            event_valid_flat,
            text_valid,
        )
        source = torch.cat(groups, dim=1)
        source_mask = torch.cat(masks, dim=1)
        if not bool(source_mask.any(dim=1).all().item()):
            raise RetrospectiveGistError(
                "combined retrospective source is empty for at least one sample"
            )

        queries = self.gist_queries.unsqueeze(0).expand(batch, -1, -1)
        for block in self.blocks:
            queries = block(queries, source, source_mask)
        gist = self.output_norm(queries)
        return RetrospectiveGistOutput(
            gist_tokens=gist,
            pooled_gist=gist.mean(dim=1),
            source_mask=source_mask,
        )

    def future_alignment_loss(
        self,
        output: RetrospectiveGistOutput | torch.Tensor,
        future_semantic_target: torch.Tensor,
        sample_mask: torch.Tensor | None = None,
        *,
        magnitude_weight: float = 0.25,
    ) -> torch.Tensor:
        """Align pooled gist to a fixed future target with stop-gradient.

        ``future_semantic_target`` is the already computed, fixed
        PCA/whitening projection of the future DINO state and its current-to-
        future delta.  Keeping target construction outside this trainable
        module makes it impossible to accidentally learn through the teacher.
        """

        gist_tokens = output.gist_tokens if isinstance(
            output, RetrospectiveGistOutput
        ) else output
        gist = _tokens(
            gist_tokens,
            field="gist_tokens",
            width=self.config.model_dim,
        )
        if gist.shape[1] != self.config.query_count:
            raise RetrospectiveGistError(
                "gist token count does not match the eight-query adapter"
            )
        target = future_semantic_target
        if not isinstance(target, torch.Tensor):
            raise TypeError("future_semantic_target must be a torch.Tensor")
        expected = (int(gist.shape[0]), self.config.future_target_dim)
        if tuple(target.shape) != expected:
            raise RetrospectiveGistError(
                f"future_semantic_target must have shape {expected}, got {tuple(target.shape)}"
            )
        if not target.is_floating_point():
            raise TypeError("future_semantic_target must have a floating dtype")
        if target.device != gist.device:
            raise RetrospectiveGistError(
                "future_semantic_target must share the gist device"
            )
        if not bool(torch.isfinite(target).all().item()):
            raise RetrospectiveGistError(
                "future_semantic_target must contain finite values"
            )
        predicted = self.future_projection(gist.mean(dim=1))
        if sample_mask is None:
            valid = torch.ones(
                (gist.shape[0],), dtype=torch.bool, device=gist.device
            )
        else:
            valid = _mask(
                sample_mask,
                field="sample_mask",
                shape=(int(gist.shape[0]),),
                device=gist.device,
                require_each_sample=False,
            )
            if not bool(valid.any().item()):
                raise RetrospectiveGistError(
                    "sample_mask must select at least one alignment target"
                )
        if (
            isinstance(magnitude_weight, bool)
            or not isinstance(magnitude_weight, (int, float))
            or not math.isfinite(float(magnitude_weight))
            or float(magnitude_weight) < 0.0
        ):
            raise RetrospectiveGistError(
                "magnitude_weight must be a finite non-negative number"
            )
        predicted_f = predicted.float()
        target_f = target.detach().float()
        cosine = F.cosine_similarity(
            predicted_f,
            target_f,
            dim=-1,
            eps=self.config.cosine_eps,
        )
        magnitude = F.smooth_l1_loss(
            torch.log1p(torch.linalg.vector_norm(predicted_f, dim=-1)),
            torch.log1p(torch.linalg.vector_norm(target_f, dim=-1)),
            reduction="none",
        )
        return (
            1.0 - cosine[valid]
            + float(magnitude_weight) * magnitude[valid]
        ).mean()


__all__ = [
    "RetrospectiveGistAdapter",
    "RetrospectiveGistError",
    "RetrospectiveGistOutput",
]
