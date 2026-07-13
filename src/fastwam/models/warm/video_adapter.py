"""Parameter-efficient selected-layer adaptation for the WARM Video DiT.

Adapters live outside the immutable baseline MoT state.  A contract-bound
FastWAM checkpoint can therefore still load strictly, while complete-WARM
checkpoints serialize these small modules in their retrospective payload.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class ResidualVideoLayerAdapter(nn.Module):
    """Zero-initialized bottleneck residual adapter for one Video DiT layer."""

    def __init__(self, hidden_dim: int, rank: int, *, scale: float = 1.0) -> None:
        super().__init__()
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer")
        if isinstance(rank, bool) or not isinstance(rank, int) or not 0 < rank < hidden_dim:
            raise ValueError("rank must be a positive integer smaller than hidden_dim")
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or float(scale) <= 0:
            raise ValueError("scale must be a positive number")

        self.hidden_dim = int(hidden_dim)
        self.rank = int(rank)
        self.scale = float(scale)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.down = nn.Linear(self.hidden_dim, self.rank, bias=False)
        self.activation = nn.GELU()
        self.up = nn.Linear(self.rank, self.hidden_dim, bias=False)
        # Start as the exact immutable FastWAM function.
        nn.init.zeros_(self.up.weight)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
            raise ValueError("video adapter tokens must have shape [B,S,D]")
        if int(tokens.shape[-1]) != self.hidden_dim:
            raise ValueError(
                "video adapter token width mismatch: "
                f"{tokens.shape[-1]} != {self.hidden_dim}"
            )
        residual = self.up(self.activation(self.down(self.norm(tokens))))
        return tokens + self.scale * residual


def resolve_video_adapter_layers(
    num_layers: int,
    requested_layers: Sequence[int],
) -> tuple[int, ...]:
    """Resolve explicit indices or the two semantic-bridge tap layers."""

    if isinstance(num_layers, bool) or not isinstance(num_layers, int) or num_layers <= 0:
        raise ValueError("num_layers must be a positive integer")
    layers = tuple(requested_layers)
    if not layers:
        layers = tuple(
            dict.fromkeys(
                (
                    max(0, min(num_layers - 1, num_layers // 3 - 1)),
                    max(0, min(num_layers - 1, (2 * num_layers) // 3 - 1)),
                )
            )
        )
    for layer in layers:
        if isinstance(layer, bool) or not isinstance(layer, int):
            raise TypeError("video adapter layer indices must be integers")
        if layer < 0 or layer >= num_layers:
            raise ValueError(
                f"video adapter layer {layer} is outside [0,{num_layers})"
            )
    if len(set(layers)) != len(layers):
        raise ValueError("video adapter layer indices must be unique")
    return tuple(layers)


def build_video_layer_adapters(
    *,
    hidden_dim: int,
    rank: int,
    scale: float,
    num_layers: int,
    requested_layers: Sequence[int],
) -> nn.ModuleDict:
    layers = resolve_video_adapter_layers(num_layers, requested_layers)
    return nn.ModuleDict(
        {
            str(layer): ResidualVideoLayerAdapter(
                hidden_dim=hidden_dim,
                rank=rank,
                scale=scale,
            )
            for layer in layers
        }
    )


__all__ = [
    "ResidualVideoLayerAdapter",
    "build_video_layer_adapters",
    "resolve_video_adapter_layers",
]
