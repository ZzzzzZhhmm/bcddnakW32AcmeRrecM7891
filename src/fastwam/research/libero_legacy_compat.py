"""Explicit interoperability for the trained step019100 LIBERO checkpoint.

The signature expansion is the same operation as the historical successful
warm-step019100-eval-v2 launcher. It changes no learned tensors or model layers.
Numerical flags are restored from the saved encoder contract before CUDA setup;
actual runtime provenance is retained, never replaced by a historical mapping.
"""
import json
import os
from pathlib import Path

import numpy as np


def expanded_signature(summary):
    if (summary.mean_displacement.shape != (7,) or summary.final_displacement.shape != (7,)
            or summary.terminal_gripper_values.shape != (1,)):
        raise ValueError("step019100 LIBERO action-summary shape mismatch")
    terminal = np.zeros(7, dtype=np.float64)
    terminal[6] = summary.terminal_gripper_values[0]
    signature = np.ascontiguousarray(np.concatenate((summary.mean_displacement.astype(np.float64),
                                                    summary.final_displacement.astype(np.float64), terminal)))
    if not np.isfinite(signature).all():
        raise ValueError("nonfinite action summary")
    return signature


def install(encoder_contract_path):
    expected = json.loads(Path(encoder_contract_path).read_text())["runtime"]
    for name, key in (("CUBLAS_WORKSPACE_CONFIG", "cublas_workspace_config"),
                      ("NVIDIA_TF32_OVERRIDE", "nvidia_tf32_override")):
        if expected[key] is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = expected[key]
    import torch
    torch.set_float32_matmul_precision(expected["float32_matmul_precision"])
    for attr in ("allow_tf32", "allow_fp16_reduced_precision_reduction", "allow_bf16_reduced_precision_reduction"):
        setattr(torch.backends.cuda.matmul, attr, expected["cuda_matmul_"+attr])
    for attr in ("enabled", "allow_tf32", "benchmark", "deterministic"):
        setattr(torch.backends.cudnn, attr, expected["cudnn_"+attr])
    torch.use_deterministic_algorithms(expected["deterministic_algorithms"],
                                      warn_only=bool(expected.get("deterministic_warn_only")))
    from fastwam.memory.runtime_fingerprint import current_encoder_runtime
    actual = current_encoder_runtime("cuda")
    differences = {k: dict(expected=expected.get(k), actual=actual.get(k))
                   for k in set(expected)|set(actual) if expected.get(k) != actual.get(k)}
    strict = {k: v for k, v in differences.items() if k not in {"platform", "cuda_device_name"}}
    if strict:
        raise ValueError(f"LIBERO encoder numerical runtime differs: {strict}")
    from fastwam.memory.episode_memory import ActionSummary
    ActionSummary.signature = expanded_signature
    return dict(patch="step019100-terminal-gripper-expansion", source="historical eval-v2 semantics",
                actual_encoder_runtime=actual, runtime_provenance_differences=differences)
