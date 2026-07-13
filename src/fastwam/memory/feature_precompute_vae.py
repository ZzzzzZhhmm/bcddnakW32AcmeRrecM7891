"""Wan2.2 VAE features from independent factual observation frames.

This module intentionally has no top-level ``torch`` or Wan model import so
the offline cache-contract test suite remains runnable on CPU-only developer
machines.  The server path imports torch lazily when encoding starts.

Most importantly, observations are never encoded as an episode video.  A
micro-batch is represented as ``[B, C, 1, H, W]``: adjacent observations share
only the batch dimension, while the temporal dimension is always one.  This
prevents the causal video VAE from leaking information across factual states.
"""

from __future__ import annotations

from typing import Any

import numpy as np


DEFAULT_VAE_POOL_SIZE = (4, 8)


class FactualVAEEncodingError(ValueError):
    """Raised when a factual-frame VAE input or output violates its contract."""


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised on the GPU server
        raise RuntimeError(
            "Wan2.2 VAE feature extraction requires the server torch environment"
        ) from exc
    return torch


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{field} must be a positive integer")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return normalized


def _model_dtype(vae: Any) -> Any | None:
    parameters = getattr(vae, "parameters", None)
    if not callable(parameters):
        return None
    try:
        return next(iter(parameters())).dtype
    except StopIteration:
        return None


def encode_wan22_factual_frames(
    vae: Any,
    frames: Any,
    *,
    device: str,
    batch_size: int = 8,
    pool_size: tuple[int, int] = DEFAULT_VAE_POOL_SIZE,
    torch_dtype: Any | None = None,
) -> np.ndarray:
    """Encode model-ready RGB frames independently with the Wan2.2 VAE.

    Args:
        vae: A frozen ``WanVideoVAE38`` (normally from
            ``load_wan22_vae_only(...).vae``).
        frames: A torch tensor or array-like ``[N, 3, H, W]``.  Values must
            already follow the exact FastWAM image preprocessing contract,
            including resize and normalization; this function does not alter
            camera semantics.
        device: VAE computation device such as ``"cuda"``.
        batch_size: Number of independent singleton-time frames per VAE call.
        pool_size: Spatial adaptive-average-pooling target.  WARM v1 uses 4x8.
        torch_dtype: Optional input compute dtype.  If omitted, the dtype of
            the first VAE parameter is used when available.

    Returns:
        A finite C-contiguous float32 NumPy array ``[N, C_latent, 4, 8]`` by
        default.  The result contains one factual feature per input frame.
    """

    if not callable(getattr(vae, "single_encode", None)):
        raise TypeError("vae must expose callable single_encode(video, device)")
    if not isinstance(device, str) or not device.strip():
        raise ValueError("device must be a non-empty string")
    micro_batch = _positive_int(batch_size, "batch_size")
    if not isinstance(pool_size, tuple) or len(pool_size) != 2:
        raise TypeError("pool_size must be a (height, width) tuple")
    pooled_height = _positive_int(pool_size[0], "pool_size[0]")
    pooled_width = _positive_int(pool_size[1], "pool_size[1]")

    torch = _require_torch()
    frame_tensor = torch.as_tensor(frames)
    if frame_tensor.ndim != 4:
        raise FactualVAEEncodingError(
            "frames must be [N,3,H,W]; episode/video tensors are forbidden, "
            f"got shape {tuple(frame_tensor.shape)}"
        )
    if frame_tensor.shape[0] <= 0 or frame_tensor.shape[1] != 3:
        raise FactualVAEEncodingError(
            f"frames must be a non-empty [N,3,H,W] tensor, got {tuple(frame_tensor.shape)}"
        )
    if frame_tensor.shape[2] <= 0 or frame_tensor.shape[3] <= 0:
        raise FactualVAEEncodingError("frame height and width must be positive")
    if not torch.is_floating_point(frame_tensor):
        raise FactualVAEEncodingError(
            "frames must already be floating-point FastWAM model inputs; "
            "uint8 observations require the canonical image preprocessor first"
        )
    spatial_factor = getattr(vae, "upsampling_factor", 16)
    spatial_factor = _positive_int(spatial_factor, "vae.upsampling_factor")
    if (
        int(frame_tensor.shape[2]) % spatial_factor != 0
        or int(frame_tensor.shape[3]) % spatial_factor != 0
    ):
        raise FactualVAEEncodingError(
            "frame height and width must be divisible by the Wan VAE spatial "
            f"factor {spatial_factor}, got {tuple(frame_tensor.shape[-2:])}"
        )

    compute_dtype = torch_dtype if torch_dtype is not None else _model_dtype(vae)
    pooled_batches: list[Any] = []
    with torch.inference_mode():
        for start in range(0, int(frame_tensor.shape[0]), micro_batch):
            stop = min(start + micro_batch, int(frame_tensor.shape[0]))
            frame_batch = frame_tensor[start:stop]
            to_kwargs: dict[str, Any] = {"device": device}
            if compute_dtype is not None:
                to_kwargs["dtype"] = compute_dtype
            frame_batch = frame_batch.to(**to_kwargs)

            # The only temporal axis presented to Wan's causal VAE is a
            # singleton.  Different timestamps remain independent batch rows.
            singleton_time_video = frame_batch.unsqueeze(2)
            if singleton_time_video.shape[2] != 1:
                raise AssertionError("factual VAE temporal dimension must remain one")
            latent = vae.single_encode(singleton_time_video, device=device)
            if latent.ndim != 5 or latent.shape[0] != stop - start:
                raise FactualVAEEncodingError(
                    "Wan2.2 VAE single_encode must return [B,C,1,h,w], "
                    f"got {tuple(latent.shape)}"
                )
            if latent.shape[2] != 1:
                raise FactualVAEEncodingError(
                    "factual singleton-frame encoding produced more than one "
                    f"latent timestep: {tuple(latent.shape)}"
                )
            if latent.shape[3] < pooled_height or latent.shape[4] < pooled_width:
                raise FactualVAEEncodingError(
                    "VAE latent spatial size must be at least the requested pool "
                    f"size {(pooled_height, pooled_width)}, got {tuple(latent.shape[-2:])}"
                )

            spatial_latent = latent[:, :, 0, :, :]
            pooled = torch.nn.functional.adaptive_avg_pool2d(
                spatial_latent,
                output_size=(pooled_height, pooled_width),
            )
            pooled_batches.append(
                pooled.detach().to(device="cpu", dtype=torch.float32)
            )

    result = torch.cat(pooled_batches, dim=0).numpy()
    result = np.ascontiguousarray(result, dtype=np.float32)
    expected_shape = (
        int(frame_tensor.shape[0]),
        int(result.shape[1]),
        pooled_height,
        pooled_width,
    )
    if result.shape != expected_shape:
        raise FactualVAEEncodingError(
            f"pooled VAE features have shape {result.shape}, expected {expected_shape}"
        )
    if not np.isfinite(result).all():
        raise FactualVAEEncodingError("pooled VAE features contain non-finite values")
    return result


__all__ = [
    "DEFAULT_VAE_POOL_SIZE",
    "FactualVAEEncodingError",
    "encode_wan22_factual_frames",
]
