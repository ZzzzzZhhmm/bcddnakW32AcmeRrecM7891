"""Closed-world artifact binding for one M2 source-only WARM run."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from numbers import Integral
from typing import Any, Mapping


SOURCE_RUN_SCHEMA = "warm.source-run-contract"
SOURCE_RUN_SCHEMA_VERSION = 1

_FIELDS = frozenset(
    {
        "schema",
        "version",
        "bank_manifest_sha256",
        "bank_content_sha256",
        "candidate_manifest_sha256",
        "query_corpus_sha256",
        "catalog_sha256",
        "audit_sha256",
        "normalization_stats_sha256",
        "action_space_contract_sha256",
        "base_checkpoint_sha256",
        "query_split",
        "global_sample_stride",
        "action_horizon",
        "action_dim",
    }
)


class SourceRunContractError(ValueError):
    """Raised when M2 artifacts cannot be bound to one exact run."""


def _digest(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SourceRunContractError(
            f"{field} must be a lowercase SHA-256 digest"
        )
    return value


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise SourceRunContractError(f"{field} must be a positive integer")
    return result


@dataclass(frozen=True, slots=True)
class WarmSourceRunContract:
    """Hashes and shapes that must agree before an M2 run may start.

    The contract intentionally stores content digests rather than paths.  Local
    and server files may live at different absolute locations while still
    proving that training used the same bank, candidate table, normalizer,
    action semantics, and base checkpoint.
    """

    bank_manifest_sha256: str
    bank_content_sha256: str
    candidate_manifest_sha256: str
    query_corpus_sha256: str
    catalog_sha256: str
    audit_sha256: str
    normalization_stats_sha256: str
    action_space_contract_sha256: str
    base_checkpoint_sha256: str
    query_split: str
    global_sample_stride: int
    action_horizon: int
    action_dim: int
    schema: str = SOURCE_RUN_SCHEMA
    version: int = SOURCE_RUN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema != SOURCE_RUN_SCHEMA:
            raise SourceRunContractError(
                f"unsupported source-run schema {self.schema!r}"
            )
        if (
            isinstance(self.version, bool)
            or not isinstance(self.version, Integral)
            or int(self.version) != SOURCE_RUN_SCHEMA_VERSION
        ):
            raise SourceRunContractError(
                f"unsupported source-run version {self.version!r}"
            )
        for field in (
            "bank_manifest_sha256",
            "bank_content_sha256",
            "candidate_manifest_sha256",
            "query_corpus_sha256",
            "catalog_sha256",
            "audit_sha256",
            "normalization_stats_sha256",
            "action_space_contract_sha256",
            "base_checkpoint_sha256",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        if self.query_split not in ("train", "dev"):
            raise SourceRunContractError("query_split must be 'train' or 'dev'")
        stride = _positive_int(self.global_sample_stride, "global_sample_stride")
        if stride != 1:
            raise SourceRunContractError(
                "M2 requires global_sample_stride=1 so QueryId frame indices match "
                "raw event-bank time"
            )
        object.__setattr__(self, "schema", SOURCE_RUN_SCHEMA)
        object.__setattr__(self, "version", SOURCE_RUN_SCHEMA_VERSION)
        object.__setattr__(self, "global_sample_stride", stride)
        object.__setattr__(
            self, "action_horizon", _positive_int(self.action_horizon, "action_horizon")
        )
        object.__setattr__(
            self, "action_dim", _positive_int(self.action_dim, "action_dim")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "version": self.version,
            "bank_manifest_sha256": self.bank_manifest_sha256,
            "bank_content_sha256": self.bank_content_sha256,
            "candidate_manifest_sha256": self.candidate_manifest_sha256,
            "query_corpus_sha256": self.query_corpus_sha256,
            "catalog_sha256": self.catalog_sha256,
            "audit_sha256": self.audit_sha256,
            "normalization_stats_sha256": self.normalization_stats_sha256,
            "action_space_contract_sha256": self.action_space_contract_sha256,
            "base_checkpoint_sha256": self.base_checkpoint_sha256,
            "query_split": self.query_split,
            "global_sample_stride": self.global_sample_stride,
            "action_horizon": self.action_horizon,
            "action_dim": self.action_dim,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WarmSourceRunContract":
        if not isinstance(value, Mapping):
            raise TypeError("source-run contract must be a mapping")
        actual = set(value)
        if actual != _FIELDS:
            raise SourceRunContractError(
                "invalid source-run contract fields; "
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
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "SOURCE_RUN_SCHEMA",
    "SOURCE_RUN_SCHEMA_VERSION",
    "SourceRunContractError",
    "WarmSourceRunContract",
]
