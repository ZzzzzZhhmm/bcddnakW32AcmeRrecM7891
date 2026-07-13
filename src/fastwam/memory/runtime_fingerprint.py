"""Numerical runtime identity shared by M1 encoding, parity, and rollout."""

from __future__ import annotations

import os
import platform
from typing import Any


class RuntimeFingerprintError(RuntimeError):
    """Raised when the contracted numerical device cannot be inspected."""


def current_encoder_runtime(device: str) -> dict[str, Any]:
    """Reproduce the exact runtime mapping emitted by M1 feature precompute."""

    import numpy as np
    import torch
    import torchvision
    import transformers

    resolved_device = str(device)
    cuda_matmul = getattr(getattr(torch.backends, "cuda", None), "matmul", None)
    cudnn = getattr(torch.backends, "cudnn", None)

    def optional_bool(owner: object, name: str) -> bool | None:
        value = getattr(owner, name, None) if owner is not None else None
        return None if value is None else bool(value)

    result: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": None if cudnn is None else cudnn.version(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": optional_bool(cuda_matmul, "allow_tf32"),
        "cuda_matmul_allow_fp16_reduced_precision_reduction": optional_bool(
            cuda_matmul, "allow_fp16_reduced_precision_reduction"
        ),
        "cuda_matmul_allow_bf16_reduced_precision_reduction": optional_bool(
            cuda_matmul, "allow_bf16_reduced_precision_reduction"
        ),
        "cudnn_enabled": optional_bool(cudnn, "enabled"),
        "cudnn_allow_tf32": optional_bool(cudnn, "allow_tf32"),
        "cudnn_benchmark": optional_bool(cudnn, "benchmark"),
        "cudnn_deterministic": optional_bool(cudnn, "deterministic"),
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_warn_only": (
            bool(torch.is_deterministic_algorithms_warn_only_enabled())
            if hasattr(torch, "is_deterministic_algorithms_warn_only_enabled")
            else None
        ),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "nvidia_tf32_override": os.environ.get("NVIDIA_TF32_OVERRIDE"),
    }
    if resolved_device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeFingerprintError(
                f"contracted CUDA runtime is unavailable: {resolved_device}"
            )
        try:
            index = torch.device(resolved_device).index
            index = torch.cuda.current_device() if index is None else index
            result["cuda_device_name"] = torch.cuda.get_device_name(index)
            result["cuda_capability"] = list(
                torch.cuda.get_device_capability(index)
            )
        except (AssertionError, RuntimeError, ValueError) as exc:
            raise RuntimeFingerprintError(
                f"cannot inspect contracted device {resolved_device!r}"
            ) from exc
    return result


__all__ = ["RuntimeFingerprintError", "current_encoder_runtime"]
