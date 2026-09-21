"""Closed-world identity for one policy-specific checkpoint comparison.

The pair contract does not make a same-weights intervention claim.  M2 writes
the source policy into each checkpoint, so ``fixed_context_top1`` and
``gaussian_null`` require distinct policy-specific checkpoint files.  This
contract proves that their scientific inputs and evaluation settings agree,
binds both trainer-emitted checkpoint attestations, and binds the passing
online/offline retrieval parity gate.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from numbers import Integral
import re
from typing import Any, Mapping


ONLINE_PAIR_SCHEMA = "warm.online-policy-checkpoint-pair-contract"
ONLINE_PAIR_SCHEMA_VERSION = 1
ONLINE_PAIR_KIND = (
    "policy_specific_trained_checkpoint_fixed_context_top1_vs_gaussian_null"
)

# These are the *only* resolved-config leaves that may differ.  Keeping this
# list in the serialized contract makes the comparison rule auditable.
ALLOWED_CONFIG_DIFFERENCE_PATHS = (
    "/EVALUATION/output_dir",
    "/EVALUATION/warm_online/contract_path",
    "/EVALUATION/warm_online/training_attestation_path",
    "/ckpt",
    "/model/source_policy",
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_FIELDS = frozenset(
    {
        "schema",
        "version",
        "comparison_kind",
        "fixed_online_run_contract_sha256",
        "gaussian_null_online_run_contract_sha256",
        "fixed_resolved_eval_config_sha256",
        "gaussian_null_resolved_eval_config_sha256",
        "fixed_warm_checkpoint_sha256",
        "gaussian_null_warm_checkpoint_sha256",
        "fixed_training_attestation_sha256",
        "gaussian_null_training_attestation_sha256",
        "shared_training_recipe_sha256",
        "shared_training_runtime_sha256",
        "parity_report_sha256",
        "shared_science_identity_sha256",
        "allowed_config_difference_paths",
        "observed_config_difference_paths",
        "git_commit",
        "git_dirty",
    }
)


class OnlinePairContractError(ValueError):
    """Raised when a fixed/null comparison identity is incomplete."""


def _digest(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise OnlinePairContractError(
            f"{field} must be a lowercase SHA-256 digest"
        )
    return value


def _path_tuple(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{field} must be a tuple/list of JSON-pointer paths")
    result = tuple(value)
    if any(
        not isinstance(item, str)
        or not item.startswith("/")
        or item.strip() != item
        or "\x00" in item
        for item in result
    ):
        raise OnlinePairContractError(
            f"{field} must contain normalized JSON-pointer paths"
        )
    if tuple(sorted(set(result))) != result:
        raise OnlinePairContractError(f"{field} must be sorted and unique")
    return result


@dataclass(frozen=True, slots=True)
class WarmOnlinePairContract:
    """Immutable identity of one policy-specific checkpoint comparison."""

    fixed_online_run_contract_sha256: str
    gaussian_null_online_run_contract_sha256: str
    fixed_resolved_eval_config_sha256: str
    gaussian_null_resolved_eval_config_sha256: str
    fixed_warm_checkpoint_sha256: str
    gaussian_null_warm_checkpoint_sha256: str
    fixed_training_attestation_sha256: str
    gaussian_null_training_attestation_sha256: str
    shared_training_recipe_sha256: str
    shared_training_runtime_sha256: str
    parity_report_sha256: str
    shared_science_identity_sha256: str
    allowed_config_difference_paths: tuple[str, ...]
    observed_config_difference_paths: tuple[str, ...]
    git_commit: str
    git_dirty: bool
    comparison_kind: str = ONLINE_PAIR_KIND
    schema: str = ONLINE_PAIR_SCHEMA
    version: int = ONLINE_PAIR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema != ONLINE_PAIR_SCHEMA:
            raise OnlinePairContractError(
                f"unsupported online-pair schema {self.schema!r}"
            )
        if (
            isinstance(self.version, bool)
            or not isinstance(self.version, Integral)
            or int(self.version) != ONLINE_PAIR_SCHEMA_VERSION
        ):
            raise OnlinePairContractError(
                f"unsupported online-pair version {self.version!r}"
            )
        if self.comparison_kind != ONLINE_PAIR_KIND:
            raise OnlinePairContractError(
                f"comparison_kind must be {ONLINE_PAIR_KIND!r}"
            )
        for field in (
            "fixed_online_run_contract_sha256",
            "gaussian_null_online_run_contract_sha256",
            "fixed_resolved_eval_config_sha256",
            "gaussian_null_resolved_eval_config_sha256",
            "fixed_warm_checkpoint_sha256",
            "gaussian_null_warm_checkpoint_sha256",
            "fixed_training_attestation_sha256",
            "gaussian_null_training_attestation_sha256",
            "shared_training_recipe_sha256",
            "shared_training_runtime_sha256",
            "parity_report_sha256",
            "shared_science_identity_sha256",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field))

        allowed = _path_tuple(
            self.allowed_config_difference_paths,
            "allowed_config_difference_paths",
        )
        if allowed != ALLOWED_CONFIG_DIFFERENCE_PATHS:
            raise OnlinePairContractError(
                "allowed_config_difference_paths must equal the schema-v1 "
                "closed-world allowlist"
            )
        observed = _path_tuple(
            self.observed_config_difference_paths,
            "observed_config_difference_paths",
        )
        if not set(observed).issubset(allowed):
            raise OnlinePairContractError(
                "observed config differences exceed the closed-world allowlist"
            )
        required = {
            "/EVALUATION/output_dir",
            "/EVALUATION/warm_online/contract_path",
            "/EVALUATION/warm_online/training_attestation_path",
            "/ckpt",
            "/model/source_policy",
        }
        if not required.issubset(observed):
            raise OnlinePairContractError(
                "a policy-checkpoint pair must use distinct checkpoints, policies, "
                "online contracts, and output directories"
            )
        object.__setattr__(self, "allowed_config_difference_paths", allowed)
        object.__setattr__(self, "observed_config_difference_paths", observed)

        if (
            self.fixed_online_run_contract_sha256
            == self.gaussian_null_online_run_contract_sha256
        ):
            raise OnlinePairContractError(
                "fixed and Gaussian-null online contract identities must differ"
            )
        if (
            self.fixed_resolved_eval_config_sha256
            == self.gaussian_null_resolved_eval_config_sha256
        ):
            raise OnlinePairContractError(
                "fixed and Gaussian-null resolved config identities must differ"
            )
        if self.fixed_warm_checkpoint_sha256 == self.gaussian_null_warm_checkpoint_sha256:
            raise OnlinePairContractError(
                "policy-specific comparison requires distinct checkpoint identities; "
                "same-weights null intervention is a different experiment"
            )
        if (
            self.fixed_training_attestation_sha256
            == self.gaussian_null_training_attestation_sha256
        ):
            raise OnlinePairContractError(
                "policy-specific comparison requires distinct training attestations"
            )
        if not isinstance(self.git_commit, str) or _GIT_COMMIT.fullmatch(
            self.git_commit
        ) is None:
            raise OnlinePairContractError(
                "git_commit must be a lowercase 40-character commit SHA"
            )
        if not isinstance(self.git_dirty, bool):
            raise TypeError("git_dirty must be a boolean")
        object.__setattr__(self, "comparison_kind", ONLINE_PAIR_KIND)
        object.__setattr__(self, "schema", ONLINE_PAIR_SCHEMA)
        object.__setattr__(self, "version", ONLINE_PAIR_SCHEMA_VERSION)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "version": self.version,
            "comparison_kind": self.comparison_kind,
            "fixed_online_run_contract_sha256": self.fixed_online_run_contract_sha256,
            "gaussian_null_online_run_contract_sha256": (
                self.gaussian_null_online_run_contract_sha256
            ),
            "fixed_resolved_eval_config_sha256": (
                self.fixed_resolved_eval_config_sha256
            ),
            "gaussian_null_resolved_eval_config_sha256": (
                self.gaussian_null_resolved_eval_config_sha256
            ),
            "fixed_warm_checkpoint_sha256": self.fixed_warm_checkpoint_sha256,
            "gaussian_null_warm_checkpoint_sha256": (
                self.gaussian_null_warm_checkpoint_sha256
            ),
            "fixed_training_attestation_sha256": (
                self.fixed_training_attestation_sha256
            ),
            "gaussian_null_training_attestation_sha256": (
                self.gaussian_null_training_attestation_sha256
            ),
            "shared_training_recipe_sha256": self.shared_training_recipe_sha256,
            "shared_training_runtime_sha256": self.shared_training_runtime_sha256,
            "parity_report_sha256": self.parity_report_sha256,
            "shared_science_identity_sha256": self.shared_science_identity_sha256,
            "allowed_config_difference_paths": list(
                self.allowed_config_difference_paths
            ),
            "observed_config_difference_paths": list(
                self.observed_config_difference_paths
            ),
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WarmOnlinePairContract":
        if not isinstance(value, Mapping):
            raise TypeError("online-pair contract must be a mapping")
        actual = set(value)
        if actual != _FIELDS:
            raise OnlinePairContractError(
                "invalid online-pair contract fields; "
                f"missing={sorted(_FIELDS - actual)}, extra={sorted(actual - _FIELDS)}"
            )
        return cls(**dict(value))

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "ALLOWED_CONFIG_DIFFERENCE_PATHS",
    "ONLINE_PAIR_KIND",
    "ONLINE_PAIR_SCHEMA",
    "ONLINE_PAIR_SCHEMA_VERSION",
    "OnlinePairContractError",
    "WarmOnlinePairContract",
]
