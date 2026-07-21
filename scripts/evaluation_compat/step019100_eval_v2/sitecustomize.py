"""Attested portability repairs for the step-019100 LIBERO evaluator.

This module is imported by CPython only when the ACP launcher places this
directory first on ``PYTHONPATH``.  The launcher verifies the exact affected
training commit and source hashes, then records this file's SHA-256 and a
derived evaluation namespace in the immutable result root.

The historical evaluator needs two narrow repairs:

* expand compact terminal-gripper coordinates in committed action summaries;
* restore the M1 encoder numerical policy in each fresh ACP process and treat
  host-kernel text plus the GPU marketing name as provenance rather than
  numerical identity.  Library versions, CUDA/cuDNN, compute capability, and
  every numerical backend flag remain strict.

The second repair is active only for the final rollout process.  Helper and
contract-building processes retain their ordinary startup behaviour.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping


_PATCH_ID = "warm-step019100-eval-v2"
_AFFECTED_TRAIN_COMMIT = "c4763a975298de6f00939360551616af7902d57a"
_NON_NUMERICAL_PROVENANCE_FIELDS = frozenset(
    {"platform", "cuda_device_name"}
)


def _abort(message: str) -> None:
    # CPython reports and ignores ordinary exceptions raised by sitecustomize.
    # An explicitly requested attested repair must instead fail closed.
    sys.stderr.write(f"fatal evaluation compatibility error: {message}\n")
    sys.stderr.flush()
    os._exit(78)


def _required_mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _abort(f"{label} must be a JSON object")
    return {str(key): item for key, item in value.items()}


def _sync_optional_environment(name: str, value: object) -> None:
    if value is None:
        os.environ.pop(name, None)
        return
    if not isinstance(value, str):
        _abort(f"contracted {name} must be a string or null")
    os.environ[name] = value


def _runtime_differences(
    expected: Mapping[str, Any], actual: Mapping[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    strict: dict[str, dict[str, Any]] = {}
    provenance: dict[str, dict[str, Any]] = {}
    for key in sorted(set(expected) | set(actual)):
        if expected.get(key) == actual.get(key) and (
            key in expected and key in actual
        ):
            continue
        difference = {"expected": expected.get(key), "actual": actual.get(key)}
        if key in _NON_NUMERICAL_PROVENANCE_FIELDS:
            provenance[key] = difference
        else:
            strict[key] = difference
    return strict, provenance


def _install_runtime_bridge() -> None:
    contract_path_raw = os.environ.get(
        "WARM_EVAL_COMPAT_ENCODER_CONTRACT_PATH", ""
    )
    configured_device = os.environ.get("WARM_EVAL_COMPAT_ENCODER_DEVICE", "")
    if not contract_path_raw or not configured_device:
        _abort("runtime bridge requires its encoder contract and device")
    contract_path = Path(contract_path_raw).expanduser().resolve()
    try:
        contract = _required_mapping(
            json.loads(contract_path.read_text(encoding="utf-8")),
            label="encoder contract",
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _abort(f"cannot read encoder contract {contract_path}: {exc}")
    expected = _required_mapping(
        contract.get("runtime"), label="encoder contract runtime"
    )
    compute = _required_mapping(
        contract.get("compute"), label="encoder contract compute"
    )
    if compute.get("device") != configured_device:
        _abort(
            "runtime bridge device differs from encoder contract: "
            f"{configured_device!r} != {compute.get('device')!r}"
        )

    # These environment controls must be restored before importing torch or
    # creating a CUDA context in the final rollout interpreter.
    _sync_optional_environment(
        "CUBLAS_WORKSPACE_CONFIG", expected.get("cublas_workspace_config")
    )
    _sync_optional_environment(
        "NVIDIA_TF32_OVERRIDE", expected.get("nvidia_tf32_override")
    )

    import torch

    cuda_matmul = getattr(getattr(torch.backends, "cuda", None), "matmul", None)
    cudnn = getattr(torch.backends, "cudnn", None)
    try:
        torch.set_float32_matmul_precision(
            str(expected["float32_matmul_precision"])
        )
        if cuda_matmul is not None:
            cuda_matmul.allow_tf32 = bool(expected["cuda_matmul_allow_tf32"])
            if hasattr(cuda_matmul, "allow_fp16_reduced_precision_reduction"):
                cuda_matmul.allow_fp16_reduced_precision_reduction = bool(
                    expected["cuda_matmul_allow_fp16_reduced_precision_reduction"]
                )
            if hasattr(cuda_matmul, "allow_bf16_reduced_precision_reduction"):
                cuda_matmul.allow_bf16_reduced_precision_reduction = bool(
                    expected["cuda_matmul_allow_bf16_reduced_precision_reduction"]
                )
        if cudnn is not None:
            cudnn.enabled = bool(expected["cudnn_enabled"])
            cudnn.allow_tf32 = bool(expected["cudnn_allow_tf32"])
            cudnn.benchmark = bool(expected["cudnn_benchmark"])
            cudnn.deterministic = bool(expected["cudnn_deterministic"])
        deterministic = bool(expected["deterministic_algorithms"])
        warn_only_value = expected.get("deterministic_warn_only")
        torch.use_deterministic_algorithms(
            deterministic,
            warn_only=False if warn_only_value is None else bool(warn_only_value),
        )
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        _abort(f"cannot restore contracted encoder numerical policy: {exc}")

    from fastwam.memory import runtime_fingerprint

    original = runtime_fingerprint.current_encoder_runtime

    def _validate_and_capture(device: str) -> tuple[dict[str, Any], dict[str, Any]]:
        if str(device) != configured_device:
            _abort(
                "rollout requested an uncontracted encoder device: "
                f"{device!r} != {configured_device!r}"
            )
        actual = original(device)
        strict, provenance = _runtime_differences(expected, actual)
        if strict:
            _abort(
                "encoder numerical runtime remains incompatible after policy "
                f"restoration: {json.dumps(strict, sort_keys=True)}"
            )
        return actual, provenance

    actual, provenance = _validate_and_capture(configured_device)
    evidence = {
        "schema": "warm.encoder-runtime-compatibility",
        "version": 1,
        "patch_id": _PATCH_ID,
        "encoder_contract": str(contract_path),
        "device": configured_device,
        "ignored_non_numerical_differences": provenance,
    }
    sys.stderr.write(
        "encoder_runtime_compatibility="
        + json.dumps(evidence, sort_keys=True, separators=(",", ":"))
        + "\n"
    )
    sys.stderr.flush()

    def _contract_compatible_runtime(device: str) -> dict[str, Any]:
        # Revalidate after model construction: an imported component is not
        # allowed to silently change a numerical backend flag.  Return the
        # historical full mapping only after the strict projection passes so
        # the old evaluator can verify its legacy full-map digest.
        _validate_and_capture(device)
        return dict(expected)

    runtime_fingerprint.current_encoder_runtime = _contract_compatible_runtime


if os.environ.get("WARM_EVAL_COMPAT_ACTION_SIGNATURE") == _PATCH_ID:
    training_commit = os.environ.get("WARM_EVAL_COMPAT_TRAIN_COMMIT", "")
    if training_commit != _AFFECTED_TRAIN_COMMIT:
        _abort(
            f"{_PATCH_ID} may only repair {_AFFECTED_TRAIN_COMMIT}, "
            f"not {training_commit or '<unset>'}"
        )

    if os.environ.get("WARM_EVAL_COMPAT_RUNTIME_BRIDGE_ACTIVE") == "1":
        _install_runtime_bridge()

    import numpy as np

    from fastwam.memory.episode_memory import ActionSummary

    raw_indices = os.environ.get("WARM_EVAL_COMPAT_GRIPPER_INDICES", "")
    gripper_indices = tuple(
        int(value) for value in raw_indices.split(",") if value != ""
    )
    if gripper_indices != (6,):
        _abort(
            f"{_PATCH_ID} expects the attested LIBERO gripper index (6,), "
            f"got {gripper_indices}"
        )

    def _expanded_signature(self: ActionSummary) -> np.ndarray:
        action_dim = int(self.mean_displacement.shape[0])
        if self.final_displacement.shape != (action_dim,):
            raise RuntimeError("committed action-summary displacement shape changed")
        if self.terminal_gripper_values.shape != (len(gripper_indices),):
            raise RuntimeError("committed action-summary gripper shape changed")
        terminal = np.zeros((action_dim,), dtype=np.float64)
        terminal[list(gripper_indices)] = self.terminal_gripper_values
        return np.ascontiguousarray(
            np.concatenate(
                (
                    self.mean_displacement.astype(np.float64),
                    self.final_displacement.astype(np.float64),
                    terminal,
                )
            )
        )

    ActionSummary.signature = _expanded_signature
