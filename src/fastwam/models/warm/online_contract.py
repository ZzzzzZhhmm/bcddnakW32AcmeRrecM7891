"""Closed-world contract for one online WARM rollout job.

Training and online retrieval have different identities.  The M2 training
contract binds a stride-one candidate cache, while a rollout dynamically
queries the immutable train event bank.  This contract binds that online job
without pretending that simulator queries came from the offline cache.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral, Real
import re
from typing import Any, Mapping


ONLINE_RUN_SCHEMA = "warm.online-run-contract"
ONLINE_RUN_SCHEMA_VERSION = 1
ONLINE_RETRIEVAL_IMPLEMENTATION = "exact_cosine_frozen_dino_v1"

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_FIELDS = frozenset(
    {
        "schema",
        "version",
        "training_run_contract_sha256",
        "validation_run_contract_sha256",
        "warm_checkpoint_sha256",
        "training_attestation_sha256",
        "shared_training_recipe_sha256",
        "training_runtime_sha256",
        "bank_manifest_sha256",
        "bank_content_sha256",
        "encoder_contract_sha256",
        "encoder_runtime_sha256",
        "camera_contract_sha256",
        "m1_data_config_sha256",
        "dino_checkpoint_tree_sha256",
        "dino_checkpoint_file_count",
        "normalization_stats_sha256",
        "action_space_contract_sha256",
        "catalog_sha256",
        "audit_sha256",
        "resolved_eval_config_sha256",
        "vae_checkpoint_sha256",
        "text_encoder_tree_sha256",
        "tokenizer_tree_sha256",
        "evaluation_namespace_sha256",
        "task_suite",
        "task_id",
        "task_description",
        "root_seed",
        "initial_states_sha256",
        "bddl_sha256",
        "retrieval_implementation",
        "top_k",
        "source_policy",
        "memory_sigma",
        "action_horizon",
        "action_dim",
        "git_commit",
        "git_dirty",
    }
)


class OnlineRunContractError(ValueError):
    """Raised when an online rollout contract is incomplete or ambiguous."""


def _digest(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise OnlineRunContractError(
            f"{field} must be a lowercase SHA-256 digest"
        )
    return value


def _normalized_string(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or "\x00" in value
    ):
        raise OnlineRunContractError(f"{field} must be a normalized non-empty string")
    return value


def _nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise OnlineRunContractError(f"{field} must be a non-negative integer")
    return result


def _positive_int(value: object, field: str) -> int:
    result = _nonnegative_int(value, field)
    if result == 0:
        raise OnlineRunContractError(f"{field} must be positive")
    return result


def _positive_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise OnlineRunContractError(f"{field} must be a finite positive number")
    return result


@dataclass(frozen=True, slots=True)
class WarmOnlineRunContract:
    """All immutable identities needed for one LIBERO online rollout job."""

    training_run_contract_sha256: str
    validation_run_contract_sha256: str
    warm_checkpoint_sha256: str
    training_attestation_sha256: str
    shared_training_recipe_sha256: str
    training_runtime_sha256: str
    bank_manifest_sha256: str
    bank_content_sha256: str
    encoder_contract_sha256: str
    encoder_runtime_sha256: str
    camera_contract_sha256: str
    m1_data_config_sha256: str
    dino_checkpoint_tree_sha256: str
    dino_checkpoint_file_count: int
    normalization_stats_sha256: str
    action_space_contract_sha256: str
    catalog_sha256: str
    audit_sha256: str
    resolved_eval_config_sha256: str
    vae_checkpoint_sha256: str
    text_encoder_tree_sha256: str
    tokenizer_tree_sha256: str
    evaluation_namespace_sha256: str
    task_suite: str
    task_id: int
    task_description: str
    root_seed: int
    initial_states_sha256: str
    bddl_sha256: str
    retrieval_implementation: str
    top_k: int
    source_policy: str
    memory_sigma: float
    action_horizon: int
    action_dim: int
    git_commit: str
    git_dirty: bool
    schema: str = ONLINE_RUN_SCHEMA
    version: int = ONLINE_RUN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema != ONLINE_RUN_SCHEMA:
            raise OnlineRunContractError(
                f"unsupported online-run schema {self.schema!r}"
            )
        if (
            isinstance(self.version, bool)
            or not isinstance(self.version, Integral)
            or int(self.version) != ONLINE_RUN_SCHEMA_VERSION
        ):
            raise OnlineRunContractError(
                f"unsupported online-run version {self.version!r}"
            )
        for field in (
            "training_run_contract_sha256",
            "validation_run_contract_sha256",
            "warm_checkpoint_sha256",
            "training_attestation_sha256",
            "shared_training_recipe_sha256",
            "training_runtime_sha256",
            "bank_manifest_sha256",
            "bank_content_sha256",
            "encoder_contract_sha256",
            "encoder_runtime_sha256",
            "camera_contract_sha256",
            "m1_data_config_sha256",
            "dino_checkpoint_tree_sha256",
            "normalization_stats_sha256",
            "action_space_contract_sha256",
            "catalog_sha256",
            "audit_sha256",
            "resolved_eval_config_sha256",
            "vae_checkpoint_sha256",
            "text_encoder_tree_sha256",
            "tokenizer_tree_sha256",
            "evaluation_namespace_sha256",
            "initial_states_sha256",
            "bddl_sha256",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        for field in ("task_suite", "task_description"):
            object.__setattr__(
                self, field, _normalized_string(getattr(self, field), field)
            )
        object.__setattr__(self, "task_id", _nonnegative_int(self.task_id, "task_id"))
        object.__setattr__(
            self, "root_seed", _nonnegative_int(self.root_seed, "root_seed")
        )
        object.__setattr__(
            self,
            "dino_checkpoint_file_count",
            _positive_int(
                self.dino_checkpoint_file_count, "dino_checkpoint_file_count"
            ),
        )
        object.__setattr__(self, "top_k", _positive_int(self.top_k, "top_k"))
        object.__setattr__(
            self,
            "action_horizon",
            _positive_int(self.action_horizon, "action_horizon"),
        )
        object.__setattr__(
            self, "action_dim", _positive_int(self.action_dim, "action_dim")
        )
        if self.retrieval_implementation != ONLINE_RETRIEVAL_IMPLEMENTATION:
            raise OnlineRunContractError(
                "online retrieval implementation must be "
                f"{ONLINE_RETRIEVAL_IMPLEMENTATION!r}"
            )
        if self.source_policy not in {"gaussian_null", "fixed_context_top1"}:
            raise OnlineRunContractError(
                "online source_policy must be gaussian_null or fixed_context_top1"
            )
        object.__setattr__(
            self, "memory_sigma", _positive_float(self.memory_sigma, "memory_sigma")
        )
        if not isinstance(self.git_commit, str) or _GIT_COMMIT.fullmatch(
            self.git_commit
        ) is None:
            raise OnlineRunContractError(
                "git_commit must be a lowercase 40-character commit SHA"
            )
        if not isinstance(self.git_dirty, bool):
            raise TypeError("git_dirty must be a boolean")
        if self.git_dirty:
            raise OnlineRunContractError(
                "formal online rollout contracts require a clean Git tree"
            )
        object.__setattr__(self, "schema", ONLINE_RUN_SCHEMA)
        object.__setattr__(self, "version", ONLINE_RUN_SCHEMA_VERSION)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "version": self.version,
            "training_run_contract_sha256": self.training_run_contract_sha256,
            "validation_run_contract_sha256": self.validation_run_contract_sha256,
            "warm_checkpoint_sha256": self.warm_checkpoint_sha256,
            "training_attestation_sha256": self.training_attestation_sha256,
            "shared_training_recipe_sha256": self.shared_training_recipe_sha256,
            "training_runtime_sha256": self.training_runtime_sha256,
            "bank_manifest_sha256": self.bank_manifest_sha256,
            "bank_content_sha256": self.bank_content_sha256,
            "encoder_contract_sha256": self.encoder_contract_sha256,
            "encoder_runtime_sha256": self.encoder_runtime_sha256,
            "camera_contract_sha256": self.camera_contract_sha256,
            "m1_data_config_sha256": self.m1_data_config_sha256,
            "dino_checkpoint_tree_sha256": self.dino_checkpoint_tree_sha256,
            "dino_checkpoint_file_count": self.dino_checkpoint_file_count,
            "normalization_stats_sha256": self.normalization_stats_sha256,
            "action_space_contract_sha256": self.action_space_contract_sha256,
            "catalog_sha256": self.catalog_sha256,
            "audit_sha256": self.audit_sha256,
            "resolved_eval_config_sha256": self.resolved_eval_config_sha256,
            "vae_checkpoint_sha256": self.vae_checkpoint_sha256,
            "text_encoder_tree_sha256": self.text_encoder_tree_sha256,
            "tokenizer_tree_sha256": self.tokenizer_tree_sha256,
            "evaluation_namespace_sha256": self.evaluation_namespace_sha256,
            "task_suite": self.task_suite,
            "task_id": self.task_id,
            "task_description": self.task_description,
            "root_seed": self.root_seed,
            "initial_states_sha256": self.initial_states_sha256,
            "bddl_sha256": self.bddl_sha256,
            "retrieval_implementation": self.retrieval_implementation,
            "top_k": self.top_k,
            "source_policy": self.source_policy,
            "memory_sigma": self.memory_sigma,
            "action_horizon": self.action_horizon,
            "action_dim": self.action_dim,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WarmOnlineRunContract":
        if not isinstance(value, Mapping):
            raise TypeError("online-run contract must be a mapping")
        actual = set(value)
        if actual != _FIELDS:
            raise OnlineRunContractError(
                "invalid online-run contract fields; "
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
    "ONLINE_RETRIEVAL_IMPLEMENTATION",
    "ONLINE_RUN_SCHEMA",
    "ONLINE_RUN_SCHEMA_VERSION",
    "OnlineRunContractError",
    "WarmOnlineRunContract",
]
