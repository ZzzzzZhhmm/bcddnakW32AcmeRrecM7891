"""Fail-closed provenance for shared-WARM to specialist initialization.

A fork is deliberately different from a resume: only complete model weights
are loaded, while optimizer, scheduler, sampler, epoch, and global-step state
start from a fresh specialist recipe.  The manifest proves that the parent and
child use the same tensor architecture and sensor/action contract.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Mapping

from fastwam.memory.manifest import sha256_file

from .training_attestation import (
    TrainingAttestationError,
    WarmTrainingRunContext,
    training_attestation_path,
    training_config_hashes,
    verify_training_attestation,
)


FORK_MANIFEST_SCHEMA = "warm.training-fork"
FORK_MANIFEST_VERSION = 1
_FIELDS = frozenset(
    {
        "schema",
        "version",
        "parent_checkpoint_sha256",
        "parent_training_attestation_sha256",
        "parent_checkpoint_step",
        "parent_git_commit",
        "parent_resolved_train_config_sha256",
        "child_resolved_train_config_sha256",
        "compatibility_contract",
        "compatibility_contract_sha256",
        "fork_reason",
    }
)


def _canonical_bytes(value: object) -> bytes:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        json.loads(payload)
    except (TypeError, ValueError) as error:
        raise TrainingAttestationError(
            "training-fork manifest must contain canonical finite JSON"
        ) from error
    return payload


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TrainingAttestationError(f"{field} must be a mapping")
    return value


def _select(root: Mapping[str, Any], path: str) -> Any:
    value: Any = root
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise TrainingAttestationError(
                f"resolved training config is missing compatibility field {path}"
            )
        value = value[part]
    return value


_COMPATIBILITY_PATHS = (
    "model._target_",
    "model.action_dit_config",
    "model.retrospection",
    "data.train.shape_meta",
    "data.train.num_frames",
    "data.train.action_video_freq_ratio",
    "data.train.video_size",
    "data.train.concat_multi_camera",
    "data.train.processor.num_output_cameras",
    "data.train.processor.action_output_dim",
    "data.train.processor.proprio_output_dim",
)


def compatibility_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact model/sensor tensor contract that a fork may reuse."""

    contract = {path: _select(config, path) for path in _COMPATIBILITY_PATHS}
    retrospection = _mapping(contract["model.retrospection"], "model.retrospection")
    required = {
        "action_dim": 14,
        "action_horizon": 32,
        "episode_action_chunk_size": 4,
    }
    for field, expected in required.items():
        actual = retrospection.get(field)
        if actual != expected:
            raise TrainingAttestationError(
                f"RMBench fork requires model.retrospection.{field}={expected}, "
                f"got {actual!r}"
            )
    shape_meta = _mapping(contract["data.train.shape_meta"], "data.train.shape_meta")
    images = shape_meta.get("images")
    if not isinstance(images, list) or [item.get("key") for item in images] != [
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    ]:
        raise TrainingAttestationError(
            "RMBench fork requires the ordered high/left-wrist/right-wrist cameras"
        )
    for item in images:
        if not isinstance(item, Mapping) or item.get("raw_shape") != [3, 240, 320]:
            raise TrainingAttestationError(
                "RMBench fork requires raw camera shape [3,240,320]"
            )
    if contract["data.train.video_size"] != [384, 320]:
        raise TrainingAttestationError(
            "RMBench fork requires model input video_size [384,320]"
        )
    if contract["data.train.processor.action_output_dim"] != 14:
        raise TrainingAttestationError("RMBench fork requires 14D actions")
    if contract["data.train.processor.proprio_output_dim"] != 14:
        raise TrainingAttestationError("RMBench fork requires 14D proprioception")
    return contract


def build_training_fork_manifest(
    *,
    parent_checkpoint: str | Path,
    parent_config: Mapping[str, Any],
    child_config: Mapping[str, Any],
    fork_reason: str,
) -> dict[str, Any]:
    reason = str(fork_reason).strip()
    if not reason or len(reason) > 512:
        raise TrainingAttestationError(
            "fork_reason must be a non-empty string of at most 512 characters"
        )
    checkpoint = Path(parent_checkpoint).expanduser().resolve()
    parent_sidecar = training_attestation_path(checkpoint)
    parent = verify_training_attestation(checkpoint, parent_sidecar)
    parent_config_hash = training_config_hashes(parent_config)[0]
    if parent_config_hash != parent.resolved_train_config_sha256:
        raise TrainingAttestationError(
            "parent config does not match the parent checkpoint attestation"
        )
    parent_allowlist = _select(parent_config, "data.train.episode_task_allowlist")
    child_allowlist = _select(child_config, "data.train.episode_task_allowlist")
    if parent_allowlist is not None:
        raise TrainingAttestationError(
            "specialist parent must be the task-shared WARM run (null task allowlist)"
        )
    if not isinstance(child_allowlist, list) or len(child_allowlist) != 1:
        raise TrainingAttestationError(
            "specialist child must select exactly one RMBench task"
        )
    parent_contract = compatibility_contract(parent_config)
    child_contract = compatibility_contract(child_config)
    if _canonical_bytes(parent_contract) != _canonical_bytes(child_contract):
        differing = sorted(
            key
            for key in parent_contract
            if parent_contract[key] != child_contract.get(key)
        )
        raise TrainingAttestationError(
            "shared parent and specialist child tensor contracts differ: "
            + ", ".join(differing)
        )
    manifest = {
        "schema": FORK_MANIFEST_SCHEMA,
        "version": FORK_MANIFEST_VERSION,
        "parent_checkpoint_sha256": parent.checkpoint_sha256,
        "parent_training_attestation_sha256": sha256_file(parent_sidecar),
        "parent_checkpoint_step": parent.checkpoint_step,
        "parent_git_commit": parent.git_commit,
        "parent_resolved_train_config_sha256": parent.resolved_train_config_sha256,
        "child_resolved_train_config_sha256": training_config_hashes(child_config)[0],
        "compatibility_contract": child_contract,
        "compatibility_contract_sha256": _sha256_json(child_contract),
        "fork_reason": reason,
    }
    validate_training_fork_manifest(manifest)
    return manifest


def validate_training_fork_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _FIELDS:
        actual = set(value) if isinstance(value, Mapping) else set()
        raise TrainingAttestationError(
            "invalid training-fork fields; "
            f"missing={sorted(_FIELDS - actual)}, extra={sorted(actual - _FIELDS)}"
        )
    result = dict(value)
    if result["schema"] != FORK_MANIFEST_SCHEMA or result["version"] != 1:
        raise TrainingAttestationError("unsupported training-fork schema/version")
    for field in (
        "parent_checkpoint_sha256",
        "parent_training_attestation_sha256",
        "parent_resolved_train_config_sha256",
        "child_resolved_train_config_sha256",
        "compatibility_contract_sha256",
    ):
        digest = result[field]
        if not isinstance(digest, str) or len(digest) != 64 or any(
            char not in "0123456789abcdef" for char in digest
        ):
            raise TrainingAttestationError(f"{field} must be a SHA-256 digest")
    commit = result["parent_git_commit"]
    if not isinstance(commit, str) or len(commit) != 40 or any(
        char not in "0123456789abcdef" for char in commit
    ):
        raise TrainingAttestationError("parent_git_commit must be a full Git SHA")
    step = result["parent_checkpoint_step"]
    if isinstance(step, bool) or not isinstance(step, int) or step < 1:
        raise TrainingAttestationError("parent_checkpoint_step must be positive")
    reason = result["fork_reason"]
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 512:
        raise TrainingAttestationError("invalid fork_reason")
    contract = _mapping(result["compatibility_contract"], "compatibility_contract")
    if _sha256_json(contract) != result["compatibility_contract_sha256"]:
        raise TrainingAttestationError("compatibility contract digest mismatch")
    return result


def load_training_fork_manifest(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    raw = source.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TrainingAttestationError("training-fork manifest is invalid JSON") from error
    manifest = validate_training_fork_manifest(value)
    if raw != _canonical_bytes(manifest) + b"\n":
        raise TrainingAttestationError("training-fork manifest is not canonical JSON")
    return manifest


def publish_training_fork_manifest(path: str | Path, manifest: Mapping[str, Any]) -> Path:
    output = Path(path).expanduser().resolve()
    payload = _canonical_bytes(validate_training_fork_manifest(manifest)) + b"\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"training-fork manifest already exists: {output}")
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb", prefix=f".{output.name}.", suffix=".tmp", dir=output.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output


def prepare_formal_fork_lineage(
    parent_checkpoint: str | Path,
    fork_manifest_path: str | Path,
    *,
    current_context: WarmTrainingRunContext,
) -> dict[str, Any]:
    checkpoint = Path(parent_checkpoint).expanduser().resolve()
    sidecar = training_attestation_path(checkpoint)
    parent = verify_training_attestation(checkpoint, sidecar)
    manifest_path = Path(fork_manifest_path).expanduser().resolve()
    manifest = load_training_fork_manifest(manifest_path)
    expected = {
        "parent_checkpoint_sha256": parent.checkpoint_sha256,
        "parent_training_attestation_sha256": sha256_file(sidecar),
        "parent_checkpoint_step": parent.checkpoint_step,
        "parent_git_commit": parent.git_commit,
        "parent_resolved_train_config_sha256": parent.resolved_train_config_sha256,
        "child_resolved_train_config_sha256": current_context.resolved_train_config_sha256,
    }
    differing = sorted(key for key, value in expected.items() if manifest[key] != value)
    if differing:
        raise TrainingAttestationError(
            "training-fork manifest contradicts checkpoint/live child: "
            + ", ".join(differing)
        )
    source_facts = (
        "source_policy",
        "train_source_contract_sha256",
        "dev_source_contract_sha256",
        "base_checkpoint_sha256",
    )
    differing_source = sorted(
        field for field in source_facts if getattr(parent, field) != getattr(current_context, field)
    )
    if differing_source:
        raise TrainingAttestationError(
            "shared parent source/bank schema differs from specialist child: "
            + ", ".join(differing_source)
        )
    return {
        "parent_checkpoint_sha256": parent.checkpoint_sha256,
        "parent_training_attestation_sha256": sha256_file(sidecar),
        "fork_manifest_sha256": sha256_file(manifest_path),
        "fork_reason": manifest["fork_reason"],
        "fork_parent_git_commit": parent.git_commit,
        "fork_parent_resolved_train_config_sha256": parent.resolved_train_config_sha256,
    }


__all__ = [
    "FORK_MANIFEST_SCHEMA",
    "FORK_MANIFEST_VERSION",
    "build_training_fork_manifest",
    "compatibility_contract",
    "load_training_fork_manifest",
    "prepare_formal_fork_lineage",
    "publish_training_fork_manifest",
    "validate_training_fork_manifest",
]
