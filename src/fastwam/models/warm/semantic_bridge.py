"""World-feature tap and DINO-aligned semantic query bridge.

This module is intentionally independent from the FastWAM backbone.  Callers
provide the two intermediate VideoDiT token streams selected by the method
(approximately layers ``L/3`` and ``2L/3``); no hook registration or second
VideoDiT pass is hidden here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .adapter_config import SemanticBridgeConfig


class SemanticBridgeError(ValueError):
    """Raised when world/teacher tokens violate the bridge contract."""


def _token_stream(
    value: object,
    *,
    field: str,
    width: int,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{field} must be a torch.Tensor")
    if value.ndim != 3 or value.shape[-1] != width:
        raise SemanticBridgeError(
            f"{field} must have shape [B,N,{width}], got {tuple(value.shape)}"
        )
    if value.shape[0] <= 0 or value.shape[1] <= 0:
        raise SemanticBridgeError(f"{field} must have non-empty B and N")
    if not value.is_floating_point():
        raise TypeError(f"{field} must have a floating dtype")
    if not bool(torch.isfinite(value).all().item()):
        raise SemanticBridgeError(f"{field} must contain only finite values")
    return value


def _token_mask(
    value: object,
    *,
    field: str,
    shape: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{field} must be a torch.Tensor")
    if value.dtype != torch.bool:
        raise TypeError(f"{field} must have dtype bool")
    if tuple(value.shape) != shape:
        raise SemanticBridgeError(
            f"{field} must have shape {shape}, got {tuple(value.shape)}"
        )
    if value.device != device:
        raise SemanticBridgeError(f"{field} must share the token device")
    if not bool(value.any(dim=1).all().item()):
        raise SemanticBridgeError(
            f"{field} must retain at least one token for every sample"
        )
    return value


@dataclass(frozen=True, slots=True)
class WorldFeatureTapOutput:
    """Projected world tokens and the normalized two-layer weights."""

    tokens: torch.Tensor
    token_mask: torch.Tensor
    layer_weights: torch.Tensor


@dataclass(frozen=True, slots=True)
class SemanticBridgeOutput:
    """All compact values needed by retrieval and later WARM adapters."""

    world_tokens: torch.Tensor
    semantic_tokens: torch.Tensor
    token_mask: torch.Tensor
    layer_weights: torch.Tensor


class WorldFeatureTap(nn.Module):
    """Softmax-combine two aligned VideoDiT intermediate token streams."""

    def __init__(self, config: SemanticBridgeConfig) -> None:
        super().__init__()
        if not isinstance(config, SemanticBridgeConfig):
            raise TypeError("config must be SemanticBridgeConfig")
        self.config = config
        self.early_projection = nn.Linear(config.early_dim, config.semantic_dim)
        self.late_projection = nn.Linear(config.late_dim, config.semantic_dim)
        # Equal contribution is the neutral initialization.
        self.layer_logits = nn.Parameter(torch.zeros(2, dtype=torch.float32))

    def forward(
        self,
        early_tokens: torch.Tensor,
        late_tokens: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> WorldFeatureTapOutput:
        early = _token_stream(
            early_tokens, field="early_tokens", width=self.config.early_dim
        )
        late = _token_stream(
            late_tokens, field="late_tokens", width=self.config.late_dim
        )
        if early.shape[:2] != late.shape[:2]:
            raise SemanticBridgeError(
                "early_tokens and late_tokens must have aligned [B,N] positions"
            )
        if early.device != late.device or early.dtype != late.dtype:
            raise SemanticBridgeError(
                "early_tokens and late_tokens must share device and dtype"
            )
        mask = _token_mask(
            token_mask,
            field="token_mask",
            shape=(int(early.shape[0]), int(early.shape[1])),
            device=early.device,
        )
        weights = torch.softmax(self.layer_logits, dim=0).to(dtype=early.dtype)
        mixed = (
            weights[0] * self.early_projection(early)
            + weights[1] * self.late_projection(late)
        )
        mixed = mixed.masked_fill(~mask.unsqueeze(-1), 0.0)
        return WorldFeatureTapOutput(
            tokens=mixed,
            token_mask=mask,
            layer_weights=weights,
        )


class SemanticQueryBridge(nn.Module):
    """Compress aligned world tokens into exactly four semantic tokens."""

    def __init__(self, config: SemanticBridgeConfig) -> None:
        super().__init__()
        if not isinstance(config, SemanticBridgeConfig):
            raise TypeError("config must be SemanticBridgeConfig")
        self.config = config
        self.queries = nn.Parameter(
            torch.empty(config.query_count, config.semantic_dim)
        )
        nn.init.normal_(self.queries, mean=0.0, std=0.02)
        self.source_norm = nn.LayerNorm(config.semantic_dim)
        self.cross_attention = nn.MultiheadAttention(
            config.semantic_dim,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(config.semantic_dim)
        hidden = max(int(config.semantic_dim * config.mlp_ratio), 1)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.semantic_dim, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, config.semantic_dim),
            nn.Dropout(config.dropout),
        )
        self.output_norm = nn.LayerNorm(config.semantic_dim)

    def forward(
        self,
        world_tokens: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        tokens = _token_stream(
            world_tokens,
            field="world_tokens",
            width=self.config.semantic_dim,
        )
        mask = _token_mask(
            token_mask,
            field="token_mask",
            shape=(int(tokens.shape[0]), int(tokens.shape[1])),
            device=tokens.device,
        )
        queries = self.queries.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        attended, _ = self.cross_attention(
            queries,
            self.source_norm(tokens),
            self.source_norm(tokens),
            key_padding_mask=~mask,
            need_weights=False,
        )
        hidden = self.attention_norm(queries + attended)
        return self.output_norm(hidden + self.feed_forward(hidden))

    def alignment_loss(
        self,
        semantic_tokens: torch.Tensor,
        dino_teacher_tokens: torch.Tensor,
        teacher_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return ``1-cos`` while stopping all gradients through DINO targets."""

        predicted = _token_stream(
            semantic_tokens,
            field="semantic_tokens",
            width=self.config.semantic_dim,
        )
        teacher = _token_stream(
            dino_teacher_tokens,
            field="dino_teacher_tokens",
            width=self.config.semantic_dim,
        )
        if predicted.shape != teacher.shape:
            raise SemanticBridgeError(
                "semantic_tokens and DINO teacher tokens must have identical shapes"
            )
        if predicted.shape[1] != self.config.query_count:
            raise SemanticBridgeError(
                "semantic token count does not match the four-query bridge"
            )
        if predicted.device != teacher.device:
            raise SemanticBridgeError(
                "semantic_tokens and DINO teacher tokens must share a device"
            )
        if teacher_mask is None:
            mask = torch.ones(
                predicted.shape[:2], dtype=torch.bool, device=predicted.device
            )
        else:
            mask = _token_mask(
                teacher_mask,
                field="teacher_mask",
                shape=(int(predicted.shape[0]), int(predicted.shape[1])),
                device=predicted.device,
            )
        cosine = F.cosine_similarity(
            predicted.float(),
            teacher.detach().float(),
            dim=-1,
            eps=self.config.cosine_eps,
        )
        return (1.0 - cosine[mask]).mean()


class WorldFeatureSemanticBridge(nn.Module):
    """Convenience composition of :class:`WorldFeatureTap` and the bridge."""

    def __init__(self, config: SemanticBridgeConfig) -> None:
        super().__init__()
        self.config = config
        self.world_tap = WorldFeatureTap(config)
        self.semantic_bridge = SemanticQueryBridge(config)

    def forward(
        self,
        early_tokens: torch.Tensor,
        late_tokens: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> SemanticBridgeOutput:
        tapped = self.world_tap(early_tokens, late_tokens, token_mask)
        semantic = self.semantic_bridge(tapped.tokens, tapped.token_mask)
        return SemanticBridgeOutput(
            world_tokens=tapped.tokens,
            semantic_tokens=semantic,
            token_mask=tapped.token_mask,
            layer_weights=tapped.layer_weights,
        )

    def alignment_loss(
        self,
        output: SemanticBridgeOutput,
        dino_teacher_tokens: torch.Tensor,
        teacher_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not isinstance(output, SemanticBridgeOutput):
            raise TypeError("output must be SemanticBridgeOutput")
        return self.semantic_bridge.alignment_loss(
            output.semantic_tokens, dino_teacher_tokens, teacher_mask
        )


__all__ = [
    "SemanticBridgeError",
    "SemanticBridgeOutput",
    "SemanticQueryBridge",
    "WorldFeatureSemanticBridge",
    "WorldFeatureTap",
    "WorldFeatureTapOutput",
]
