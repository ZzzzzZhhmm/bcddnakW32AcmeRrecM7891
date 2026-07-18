#!/usr/bin/env python3
"""Build a closed-world fixed/null policy-checkpoint comparison contract.

The builder reopens both policy checkpoints and their canonical trainer-emitted
sidecars so optimizer, scheduler, distributed runtime, data, base checkpoint,
seed, and step fairness are proved rather than inferred.  Other model,
encoder, memory-bank, and simulator artifacts remain bound by the two online
contracts; the passing retrieval-parity report is also revalidated and bound.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence
from uuid import uuid4

from fastwam.memory.manifest import sha256_canonical_json, sha256_file
from fastwam.models.warm.online_contract import WarmOnlineRunContract
from fastwam.models.warm.parity_report import (
    validate_passing_online_parity_report,
)
from fastwam.models.warm.online_pair_contract import (
    ALLOWED_CONFIG_DIFFERENCE_PATHS,
    WarmOnlinePairContract,
)
from fastwam.models.warm.training_attestation import (
    TrainingAttestationError,
    WarmTrainingAttestation,
    verify_training_attestation,
)
from fastwam.utils.artifact_claim import artifact_claim


SUMMARY_SCHEMA = "warm.online-policy-checkpoint-pair-build-summary"
SUMMARY_SCHEMA_VERSION = 1

# The two policy-specific identities are intentionally different.  Every
# other online-contract field is a science-comparable identity and must match.
_POLICY_SPECIFIC_ONLINE_FIELDS = frozenset(
    {
        "source_policy",
        "warm_checkpoint_sha256",
        "training_attestation_sha256",
        "resolved_eval_config_sha256",
    }
)


class OnlinePairBuildError(RuntimeError):
    """Raised when two online jobs do not form a valid formal comparison."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Bind fixed-context and Gaussian-null policy-specific WARM "
            "checkpoint jobs into one closed-world comparison contract."
        )
    )
    parser.add_argument("--fixed-online-contract", required=True, type=Path)
    parser.add_argument("--fixed-resolved-eval-config", required=True, type=Path)
    parser.add_argument("--fixed-training-attestation", required=True, type=Path)
    parser.add_argument("--gaussian-null-online-contract", required=True, type=Path)
    parser.add_argument(
        "--gaussian-null-resolved-eval-config", required=True, type=Path
    )
    parser.add_argument(
        "--gaussian-null-training-attestation", required=True, type=Path
    )
    parser.add_argument("--parity-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _read_json_mapping(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OnlinePairBuildError(f"cannot read {label} JSON at {path}") from exc
    if not isinstance(value, Mapping):
        raise OnlinePairBuildError(f"{label} must be a JSON object")
    return value


def _read_resolved_config(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise OnlinePairBuildError(f"cannot read {label} at {path}") from exc
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError):
        try:
            from omegaconf import OmegaConf

            value = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        except Exception as exc:  # pragma: no cover - depends on server extras
            raise OnlinePairBuildError(
                f"{label} must be valid JSON or fully resolved OmegaConf YAML"
            ) from exc
    if not isinstance(value, Mapping):
        raise OnlinePairBuildError(f"{label} must decode to a mapping")
    try:
        normalized = json.loads(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise OnlinePairBuildError(
            f"{label} must contain only finite JSON-compatible values"
        ) from exc
    if not isinstance(normalized, Mapping):  # pragma: no cover - round-trip guard
        raise OnlinePairBuildError(f"{label} must normalize to a mapping")
    return normalized


def _config_value(config: Mapping[str, Any], *path: str) -> Any:
    current: Any = config
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            raise OnlinePairBuildError(
                "resolved config is missing required field " + ".".join(path)
            )
        current = current[key]
    return current


def _normalized_nonempty(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or "\x00" in value
    ):
        raise OnlinePairBuildError(f"{label} must be a normalized non-empty string")
    return value


def _resolved_config_path(value: Any, *, label: str) -> Path:
    raw = _normalized_nonempty(value, label=label)
    return Path(os.path.expanduser(os.path.expandvars(raw))).resolve()


def _json_pointer(path: tuple[str, ...]) -> str:
    return "/" + "/".join(
        component.replace("~", "~0").replace("/", "~1") for component in path
    )


def _different_leaf_paths(
    left: Any,
    right: Any,
    *,
    path: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Return deterministic JSON-pointer paths for every differing leaf."""

    if isinstance(left, Mapping) and isinstance(right, Mapping):
        result: list[str] = []
        for key in sorted(set(left) | set(right)):
            if not isinstance(key, str):
                raise OnlinePairBuildError(
                    "resolved config mappings must use string keys"
                )
            child = path + (key,)
            if key not in left or key not in right:
                result.append(_json_pointer(child))
            else:
                result.extend(
                    _different_leaf_paths(left[key], right[key], path=child)
                )
        return tuple(result)
    if isinstance(left, list) and isinstance(right, list):
        result = []
        for index in range(max(len(left), len(right))):
            child = path + (str(index),)
            if index >= len(left) or index >= len(right):
                result.append(_json_pointer(child))
            else:
                result.extend(
                    _different_leaf_paths(left[index], right[index], path=child)
                )
        return tuple(result)
    if type(left) is not type(right) or left != right:
        # A root-level scalar is impossible for a resolved config, but keep a
        # well-defined representation for defensive completeness.
        return (_json_pointer(path) if path else "/",)
    return ()


def _git_identity(repository: Path) -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise OnlinePairBuildError("cannot inspect WARM Git identity") from exc
    return commit, bool(status.strip())


def _validate_policy_config(
    *,
    config: Mapping[str, Any],
    contract: WarmOnlineRunContract,
    contract_path: Path,
    pair_contract_path: Path,
    expected_policy: str,
    label: str,
) -> tuple[Path, Path]:
    if contract.source_policy != expected_policy:
        raise OnlinePairBuildError(
            f"{label} online contract must use source_policy={expected_policy!r}"
        )
    config_policy = _config_value(config, "model", "source_policy")
    if config_policy != expected_policy:
        raise OnlinePairBuildError(
            f"{label} resolved config must use source_policy={expected_policy!r}"
        )
    configured_contract = _resolved_config_path(
        _config_value(config, "EVALUATION", "warm_online", "contract_path"),
        label=f"{label} online contract path",
    )
    if configured_contract != contract_path:
        raise OnlinePairBuildError(
            f"{label} config points to a different online contract: "
            f"{configured_contract} != {contract_path}"
        )
    configured_pair = _resolved_config_path(
        _config_value(
            config, "EVALUATION", "warm_online", "pair_contract_path"
        ),
        label=f"{label} online pair contract path",
    )
    if configured_pair != pair_contract_path:
        raise OnlinePairBuildError(
            f"{label} config points to a different pair contract: "
            f"{configured_pair} != {pair_contract_path}"
        )
    checkpoint_path = _resolved_config_path(
        _config_value(config, "ckpt"), label=f"{label} checkpoint"
    )
    output_path = _resolved_config_path(
        _config_value(config, "EVALUATION", "output_dir"),
        label=f"{label} output directory",
    )
    if _config_value(config, "EVALUATION", "warm_online", "enabled") is not True:
        raise OnlinePairBuildError(
            f"{label} resolved config must enable formal online evaluation"
        )
    return checkpoint_path, output_path


def _shared_online_identity(contract: WarmOnlineRunContract) -> dict[str, Any]:
    value = contract.to_dict()
    for field in _POLICY_SPECIFIC_ONLINE_FIELDS:
        value.pop(field)
    return value


def _load_inputs(
    args: argparse.Namespace,
) -> tuple[
    WarmOnlineRunContract,
    Mapping[str, Any],
    WarmOnlineRunContract,
    Mapping[str, Any],
]:
    fixed_contract_path = args.fixed_online_contract.expanduser().resolve()
    null_contract_path = args.gaussian_null_online_contract.expanduser().resolve()
    fixed_config_path = args.fixed_resolved_eval_config.expanduser().resolve()
    null_config_path = args.gaussian_null_resolved_eval_config.expanduser().resolve()
    for label, path in (
        ("fixed online contract", fixed_contract_path),
        ("fixed resolved eval config", fixed_config_path),
        ("Gaussian-null online contract", null_contract_path),
        ("Gaussian-null resolved eval config", null_config_path),
    ):
        if not path.is_file():
            raise OnlinePairBuildError(f"{label} is not a regular file: {path}")
    if fixed_contract_path == null_contract_path:
        raise OnlinePairBuildError("fixed and Gaussian-null contract files must differ")
    if fixed_config_path == null_config_path:
        raise OnlinePairBuildError("fixed and Gaussian-null config files must differ")

    fixed_contract = WarmOnlineRunContract.from_dict(
        _read_json_mapping(fixed_contract_path, label="fixed online contract")
    )
    null_contract = WarmOnlineRunContract.from_dict(
        _read_json_mapping(null_contract_path, label="Gaussian-null online contract")
    )
    fixed_config = _read_resolved_config(
        fixed_config_path, label="fixed resolved eval config"
    )
    null_config = _read_resolved_config(
        null_config_path, label="Gaussian-null resolved eval config"
    )

    fixed_pair_value = _config_value(
        fixed_config, "EVALUATION", "warm_online", "pair_contract_path"
    )
    null_pair_value = _config_value(
        null_config, "EVALUATION", "warm_online", "pair_contract_path"
    )
    if fixed_pair_value != null_pair_value:
        raise OnlinePairBuildError(
            "fixed and Gaussian-null configs must contain the same "
            "EVALUATION.warm_online.pair_contract_path"
        )

    parity_path = args.parity_report.expanduser().resolve()
    fixed_parity_value = _config_value(
        fixed_config, "EVALUATION", "warm_online", "parity_report_path"
    )
    null_parity_value = _config_value(
        null_config, "EVALUATION", "warm_online", "parity_report_path"
    )
    if fixed_parity_value != null_parity_value:
        raise OnlinePairBuildError(
            "fixed and Gaussian-null configs must contain the same "
            "EVALUATION.warm_online.parity_report_path"
        )
    configured_parity = _resolved_config_path(
        fixed_parity_value, label="online parity report path"
    )
    if configured_parity != parity_path:
        raise OnlinePairBuildError(
            "resolved configs point to a different parity report: "
            f"{configured_parity} != {parity_path}"
        )

    if sha256_canonical_json(fixed_config) != fixed_contract.resolved_eval_config_sha256:
        raise OnlinePairBuildError(
            "fixed resolved config does not match its online contract"
        )
    if sha256_canonical_json(null_config) != null_contract.resolved_eval_config_sha256:
        raise OnlinePairBuildError(
            "Gaussian-null resolved config does not match its online contract"
        )
    pair_contract_path = args.output.expanduser().resolve()
    fixed_checkpoint_path, fixed_output_path = _validate_policy_config(
        config=fixed_config,
        contract=fixed_contract,
        contract_path=fixed_contract_path,
        pair_contract_path=pair_contract_path,
        expected_policy="fixed_context_top1",
        label="fixed",
    )
    null_checkpoint_path, null_output_path = _validate_policy_config(
        config=null_config,
        contract=null_contract,
        contract_path=null_contract_path,
        pair_contract_path=pair_contract_path,
        expected_policy="gaussian_null",
        label="Gaussian-null",
    )
    if fixed_checkpoint_path == null_checkpoint_path:
        raise OnlinePairBuildError(
            "fixed and Gaussian-null checkpoint paths must resolve to different files"
        )
    if fixed_output_path == null_output_path:
        raise OnlinePairBuildError(
            "fixed and Gaussian-null output directories must resolve differently"
        )
    return fixed_contract, fixed_config, null_contract, null_config


def _validated_parity_report_sha256(
    args: argparse.Namespace,
    *,
    fixed_contract: WarmOnlineRunContract,
) -> str:
    path = args.parity_report.expanduser().resolve()
    if not path.is_file():
        raise OnlinePairBuildError(
            f"passing online parity report is not a regular file: {path}"
        )
    before = sha256_file(path)
    value = _read_json_mapping(path, label="online parity report")
    try:
        validate_passing_online_parity_report(
            value, fixed_online_contract=fixed_contract
        )
    except (TypeError, ValueError) as exc:
        raise OnlinePairBuildError(
            "online parity report is not a valid passing gate"
        ) from exc
    if sha256_file(path) != before:
        raise OnlinePairBuildError("online parity report changed while it was read")
    return before


def _validated_training_attestation(
    *,
    path: Path,
    config: Mapping[str, Any],
    contract: WarmOnlineRunContract,
    expected_policy: str,
    label: str,
) -> WarmTrainingAttestation:
    configured_path = _resolved_config_path(
        _config_value(
            config, "EVALUATION", "warm_online", "training_attestation_path"
        ),
        label=f"{label} training attestation path",
    )
    if configured_path != path:
        raise OnlinePairBuildError(
            f"{label} config points to a different training attestation: "
            f"{configured_path} != {path}"
        )
    checkpoint_path = _resolved_config_path(
        _config_value(config, "ckpt"), label=f"{label} checkpoint"
    )
    if not path.is_file() or not checkpoint_path.is_file():
        raise OnlinePairBuildError(
            f"{label} checkpoint and training attestation must be regular files"
        )
    before = sha256_file(path)
    try:
        attestation = verify_training_attestation(checkpoint_path, path)
    except (OSError, TrainingAttestationError) as exc:
        raise OnlinePairBuildError(
            f"{label} training attestation is invalid"
        ) from exc
    if sha256_file(path) != before:
        raise OnlinePairBuildError(
            f"{label} training attestation changed while it was verified"
        )
    if before != contract.training_attestation_sha256:
        raise OnlinePairBuildError(
            f"{label} training attestation does not match its online contract"
        )
    expected = {
        "source_policy": expected_policy,
        "checkpoint_sha256": contract.warm_checkpoint_sha256,
        "train_source_contract_sha256": contract.training_run_contract_sha256,
        "dev_source_contract_sha256": contract.validation_run_contract_sha256,
        "shared_recipe_sha256": contract.shared_training_recipe_sha256,
        "training_runtime_sha256": contract.training_runtime_sha256,
        "git_commit": contract.git_commit,
    }
    actual = {
        "source_policy": attestation.source_policy,
        "checkpoint_sha256": attestation.checkpoint_sha256,
        "train_source_contract_sha256": attestation.train_source_contract_sha256,
        "dev_source_contract_sha256": attestation.dev_source_contract_sha256,
        "shared_recipe_sha256": attestation.shared_recipe_sha256,
        "training_runtime_sha256": attestation.training_runtime_sha256,
        "git_commit": attestation.git_commit,
    }
    if actual != expected:
        differing = sorted(key for key in expected if actual[key] != expected[key])
        raise OnlinePairBuildError(
            f"{label} training attestation contradicts its online contract: "
            + ", ".join(differing)
        )
    return attestation


def _shared_training_identity(
    attestation: WarmTrainingAttestation,
) -> dict[str, Any]:
    value = attestation.to_dict()
    for field in (
        "checkpoint_sha256",
        "source_policy",
        "resolved_train_config_sha256",
        # Each ablation branch necessarily resumes from policy-specific
        # checkpoint/state bytes.  The common resume_step remains in the
        # identity, while each branch is independently verified against its
        # own cryptographic parent chain before this comparison.
        "parent_checkpoint_sha256",
        "parent_training_attestation_sha256",
        "resume_state_sha256",
    ):
        value.pop(field, None)
    return value


def _validated_training_pair(
    args: argparse.Namespace,
    *,
    fixed_contract: WarmOnlineRunContract,
    fixed_config: Mapping[str, Any],
    null_contract: WarmOnlineRunContract,
    null_config: Mapping[str, Any],
) -> tuple[WarmTrainingAttestation, WarmTrainingAttestation]:
    fixed_path = args.fixed_training_attestation.expanduser().resolve()
    null_path = args.gaussian_null_training_attestation.expanduser().resolve()
    if fixed_path == null_path:
        raise OnlinePairBuildError(
            "fixed and Gaussian-null training-attestation files must differ"
        )
    fixed = _validated_training_attestation(
        path=fixed_path,
        config=fixed_config,
        contract=fixed_contract,
        expected_policy="fixed_context_top1",
        label="fixed",
    )
    null = _validated_training_attestation(
        path=null_path,
        config=null_config,
        contract=null_contract,
        expected_policy="gaussian_null",
        label="Gaussian-null",
    )
    fixed_shared = _shared_training_identity(fixed)
    null_shared = _shared_training_identity(null)
    if fixed_shared != null_shared:
        differing = _different_leaf_paths(fixed_shared, null_shared)
        raise OnlinePairBuildError(
            "training attestations disagree on fairness-critical facts: "
            + ", ".join(differing)
        )
    if fixed.resolved_train_config_sha256 == null.resolved_train_config_sha256:
        raise OnlinePairBuildError(
            "policy-specific training configs must have distinct full identities"
        )
    if fixed.checkpoint_sha256 == null.checkpoint_sha256:
        raise OnlinePairBuildError(
            "policy-specific training attestations must bind distinct checkpoints"
        )
    return fixed, null


def _build_pair(args: argparse.Namespace) -> WarmOnlinePairContract:
    fixed_contract, fixed_config, null_contract, null_config = _load_inputs(args)
    if fixed_contract.sha256 == null_contract.sha256:
        raise OnlinePairBuildError(
            "fixed and Gaussian-null online contract identities must differ"
        )
    if (
        fixed_contract.resolved_eval_config_sha256
        == null_contract.resolved_eval_config_sha256
    ):
        raise OnlinePairBuildError(
            "fixed and Gaussian-null resolved config identities must differ"
        )

    fixed_shared = _shared_online_identity(fixed_contract)
    null_shared = _shared_online_identity(null_contract)
    if fixed_shared != null_shared:
        differing = _different_leaf_paths(fixed_shared, null_shared)
        raise OnlinePairBuildError(
            "online contracts disagree on science-comparable identities: "
            + ", ".join(differing)
        )
    if fixed_contract.warm_checkpoint_sha256 == null_contract.warm_checkpoint_sha256:
        raise OnlinePairBuildError(
            "M2 policy-specific comparison requires distinct checkpoint identities; "
            "a same-weights null intervention must use a separate experiment label"
        )
    fixed_training, null_training = _validated_training_pair(
        args,
        fixed_contract=fixed_contract,
        fixed_config=fixed_config,
        null_contract=null_contract,
        null_config=null_config,
    )
    parity_report_sha256 = _validated_parity_report_sha256(
        args, fixed_contract=fixed_contract
    )

    observed = _different_leaf_paths(fixed_config, null_config)
    unexpected = sorted(set(observed) - set(ALLOWED_CONFIG_DIFFERENCE_PATHS))
    if unexpected:
        raise OnlinePairBuildError(
            "resolved configs differ outside the closed-world allowlist: "
            + ", ".join(unexpected)
        )
    missing = sorted(set(ALLOWED_CONFIG_DIFFERENCE_PATHS) - set(observed))
    if missing:
        raise OnlinePairBuildError(
            "policy-specific configs must differ at every allowed path; "
            "missing differences: " + ", ".join(missing)
        )

    repository = Path(__file__).resolve().parent.parent
    git_commit, git_dirty = _git_identity(repository)
    if git_dirty:
        raise OnlinePairBuildError(
            "formal online pair contract requires a clean Git tree"
        )
    if fixed_contract.git_commit != git_commit or null_contract.git_commit != git_commit:
        raise OnlinePairBuildError(
            "both online contracts must bind the current clean Git commit"
        )

    return WarmOnlinePairContract(
        fixed_online_run_contract_sha256=fixed_contract.sha256,
        gaussian_null_online_run_contract_sha256=null_contract.sha256,
        fixed_resolved_eval_config_sha256=fixed_contract.resolved_eval_config_sha256,
        gaussian_null_resolved_eval_config_sha256=(
            null_contract.resolved_eval_config_sha256
        ),
        fixed_warm_checkpoint_sha256=fixed_contract.warm_checkpoint_sha256,
        gaussian_null_warm_checkpoint_sha256=null_contract.warm_checkpoint_sha256,
        fixed_training_attestation_sha256=(
            fixed_contract.training_attestation_sha256
        ),
        gaussian_null_training_attestation_sha256=(
            null_contract.training_attestation_sha256
        ),
        shared_training_recipe_sha256=fixed_training.shared_recipe_sha256,
        shared_training_runtime_sha256=fixed_training.training_runtime_sha256,
        parity_report_sha256=parity_report_sha256,
        shared_science_identity_sha256=sha256_canonical_json(fixed_shared),
        allowed_config_difference_paths=ALLOWED_CONFIG_DIFFERENCE_PATHS,
        observed_config_difference_paths=observed,
        git_commit=git_commit,
        git_dirty=False,
    )


def _assert_inputs_unchanged(
    args: argparse.Namespace,
    expected: WarmOnlinePairContract,
) -> None:
    fixed_contract, fixed_config, null_contract, null_config = _load_inputs(args)
    fixed_training, null_training = _validated_training_pair(
        args,
        fixed_contract=fixed_contract,
        fixed_config=fixed_config,
        null_contract=null_contract,
        null_config=null_config,
    )
    if fixed_contract.sha256 != expected.fixed_online_run_contract_sha256:
        raise OnlinePairBuildError("fixed online contract changed during pair build")
    if null_contract.sha256 != expected.gaussian_null_online_run_contract_sha256:
        raise OnlinePairBuildError(
            "Gaussian-null online contract changed during pair build"
        )
    if sha256_canonical_json(fixed_config) != expected.fixed_resolved_eval_config_sha256:
        raise OnlinePairBuildError("fixed resolved config changed during pair build")
    if sha256_canonical_json(null_config) != expected.gaussian_null_resolved_eval_config_sha256:
        raise OnlinePairBuildError(
            "Gaussian-null resolved config changed during pair build"
        )
    if _shared_online_identity(fixed_contract) != _shared_online_identity(null_contract):
        raise OnlinePairBuildError("shared online identity changed during pair build")
    if _different_leaf_paths(fixed_config, null_config) != (
        expected.observed_config_difference_paths
    ):
        raise OnlinePairBuildError("resolved config differences changed during pair build")
    if _validated_parity_report_sha256(
        args, fixed_contract=fixed_contract
    ) != expected.parity_report_sha256:
        raise OnlinePairBuildError("online parity report changed during pair build")
    if (
        sha256_file(args.fixed_training_attestation.expanduser().resolve())
        != expected.fixed_training_attestation_sha256
        or sha256_file(
            args.gaussian_null_training_attestation.expanduser().resolve()
        )
        != expected.gaussian_null_training_attestation_sha256
        or fixed_training.shared_recipe_sha256
        != expected.shared_training_recipe_sha256
        or fixed_training.training_runtime_sha256
        != expected.shared_training_runtime_sha256
        or _shared_training_identity(fixed_training)
        != _shared_training_identity(null_training)
    ):
        raise OnlinePairBuildError(
            "training-attestation fairness identity changed during pair build"
        )
    git_commit, git_dirty = _git_identity(Path(__file__).resolve().parent.parent)
    if git_dirty or git_commit != expected.git_commit:
        raise OnlinePairBuildError("Git identity changed during pair build")


def _write_atomic(path: Path, value: Mapping[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"online pair contract already exists at {path}")
    encoded = (
        json.dumps(
            dict(value),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
            temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    inputs = {
        args.fixed_online_contract.expanduser().resolve(),
        args.fixed_resolved_eval_config.expanduser().resolve(),
        args.gaussian_null_online_contract.expanduser().resolve(),
        args.gaussian_null_resolved_eval_config.expanduser().resolve(),
        args.fixed_training_attestation.expanduser().resolve(),
        args.gaussian_null_training_attestation.expanduser().resolve(),
        args.parity_report.expanduser().resolve(),
    }
    if output in inputs:
        raise OnlinePairBuildError("pair output must not overwrite any pair input")

    pair = _build_pair(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.parent / f".{output.name}.warm-artifact.lock"
    claim = artifact_claim(
        lock_path, purpose=f"publish WARM online policy-checkpoint pair: {output}"
    )
    with claim:
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"online pair contract already exists at {output}")
        _assert_inputs_unchanged(args, pair)
        _write_atomic(output, pair.to_dict(), overwrite=args.overwrite)

    print(
        json.dumps(
            {
                "schema": SUMMARY_SCHEMA,
                "version": SUMMARY_SCHEMA_VERSION,
                "output": str(output),
                "online_pair_contract_sha256": pair.sha256,
                "comparison_kind": pair.comparison_kind,
                "contract": pair.to_dict(),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
