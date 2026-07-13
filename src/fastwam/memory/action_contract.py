"""Strict, versioned action-space contract for WARM artifacts.

An action tensor is not interpretable from its final dimension alone.  The
same numeric array can encode joint positions, end-effector deltas, or a
normalized control vector with different gripper semantics.  This module
defines the small, portable contract that must accompany such arrays.

The parser is intentionally closed-world: unknown fields and unsupported
schema versions are rejected instead of being silently ignored.  This keeps
artifact hashes meaningful and prevents an older reader from accepting newer
semantics it does not understand.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any, Mapping


ACTION_SPACE_SCHEMA = "warm.action-space"
ACTION_SPACE_SCHEMA_VERSION = 1

_CONTRACT_FIELDS = frozenset(
    {
        "schema",
        "version",
        "action_dim",
        "arm_dims",
        "gripper_dims",
        "gripper_threshold",
        "normalization_mode",
        "normalization_stats_sha256",
        "control_mode",
        "embodiment",
    }
)


class ActionSpaceContractError(ValueError):
    """Raised when action-space metadata is incomplete or ambiguous."""


def _canonical_string(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ActionSpaceContractError(f"{field} must be a string")
    if not value or not value.strip():
        raise ActionSpaceContractError(f"{field} must be non-empty")
    if value != value.strip():
        raise ActionSpaceContractError(
            f"{field} must not contain leading or trailing whitespace"
        )
    if "\x00" in value:
        raise ActionSpaceContractError(f"{field} must not contain NUL characters")
    return value


def _positive_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ActionSpaceContractError(f"{field} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ActionSpaceContractError(f"{field} must be a positive integer")
    return result


def _dimension_tuple(value: object, field: str, action_dim: int) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        raise ActionSpaceContractError(f"{field} must be a tuple of action dimensions")
    if not value:
        raise ActionSpaceContractError(f"{field} must contain at least one dimension")

    dimensions: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, Integral):
            raise ActionSpaceContractError(f"{field} entries must be integers")
        dimension = int(item)
        if dimension < 0 or dimension >= action_dim:
            raise ActionSpaceContractError(
                f"{field} dimension {dimension} is outside [0, {action_dim})"
            )
        dimensions.append(dimension)

    if len(set(dimensions)) != len(dimensions):
        raise ActionSpaceContractError(f"{field} must not contain duplicate dimensions")
    if dimensions != sorted(dimensions):
        raise ActionSpaceContractError(f"{field} must be sorted in ascending order")
    return tuple(dimensions)


def _finite_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ActionSpaceContractError(f"{field} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ActionSpaceContractError(f"{field} must be a finite real number")
    return result


def _sha256_digest(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ActionSpaceContractError(
            f"{field} must be a lowercase SHA-256 digest"
        )
    return value


@dataclass(frozen=True, slots=True)
class ActionSpaceContract:
    """Complete semantics for one WARM action vector.

    ``arm_dims`` and ``gripper_dims`` form a disjoint, exhaustive partition of
    ``range(action_dim)``.  The explicit partition prevents downstream code
    from guessing which channel controls the gripper when computing action
    distance or event timing.
    """

    action_dim: int
    arm_dims: tuple[int, ...]
    gripper_dims: tuple[int, ...]
    gripper_threshold: float
    normalization_mode: str
    normalization_stats_sha256: str
    control_mode: str
    embodiment: str
    schema: str = ACTION_SPACE_SCHEMA
    version: int = ACTION_SPACE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema != ACTION_SPACE_SCHEMA:
            raise ActionSpaceContractError(
                f"unsupported action-space schema {self.schema!r}; "
                f"expected {ACTION_SPACE_SCHEMA!r}"
            )
        if (
            isinstance(self.version, bool)
            or not isinstance(self.version, Integral)
            or int(self.version) != ACTION_SPACE_SCHEMA_VERSION
        ):
            raise ActionSpaceContractError(
                f"unsupported action-space version {self.version!r}; "
                f"expected {ACTION_SPACE_SCHEMA_VERSION}"
            )

        action_dim = _positive_integer(self.action_dim, "action_dim")
        arm_dims = _dimension_tuple(self.arm_dims, "arm_dims", action_dim)
        gripper_dims = _dimension_tuple(self.gripper_dims, "gripper_dims", action_dim)

        overlap = sorted(set(arm_dims) & set(gripper_dims))
        if overlap:
            raise ActionSpaceContractError(
                f"arm_dims and gripper_dims overlap at dimensions {overlap}"
            )
        covered = set(arm_dims) | set(gripper_dims)
        expected = set(range(action_dim))
        if covered != expected:
            missing = sorted(expected - covered)
            extra = sorted(covered - expected)
            raise ActionSpaceContractError(
                "arm_dims and gripper_dims must cover every action dimension exactly once; "
                f"missing={missing}, extra={extra}"
            )

        object.__setattr__(self, "schema", ACTION_SPACE_SCHEMA)
        object.__setattr__(self, "version", ACTION_SPACE_SCHEMA_VERSION)
        object.__setattr__(self, "action_dim", action_dim)
        object.__setattr__(self, "arm_dims", arm_dims)
        object.__setattr__(self, "gripper_dims", gripper_dims)
        object.__setattr__(
            self,
            "gripper_threshold",
            _finite_float(self.gripper_threshold, "gripper_threshold"),
        )
        object.__setattr__(
            self,
            "normalization_mode",
            _canonical_string(self.normalization_mode, "normalization_mode"),
        )
        object.__setattr__(
            self,
            "normalization_stats_sha256",
            _sha256_digest(
                self.normalization_stats_sha256,
                "normalization_stats_sha256",
            ),
        )
        object.__setattr__(
            self, "control_mode", _canonical_string(self.control_mode, "control_mode")
        )
        object.__setattr__(
            self, "embodiment", _canonical_string(self.embodiment, "embodiment")
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical, JSON-serializable representation."""

        return {
            "schema": self.schema,
            "version": self.version,
            "action_dim": self.action_dim,
            "arm_dims": list(self.arm_dims),
            "gripper_dims": list(self.gripper_dims),
            "gripper_threshold": self.gripper_threshold,
            "normalization_mode": self.normalization_mode,
            "normalization_stats_sha256": self.normalization_stats_sha256,
            "control_mode": self.control_mode,
            "embodiment": self.embodiment,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ActionSpaceContract":
        """Parse one exact-version JSON object into an immutable contract."""

        if not isinstance(value, Mapping):
            raise ActionSpaceContractError("action-space contract must be a mapping")
        actual = set(value)
        if actual != _CONTRACT_FIELDS:
            missing = sorted(_CONTRACT_FIELDS - actual)
            extra = sorted(actual - _CONTRACT_FIELDS)
            raise ActionSpaceContractError(
                f"invalid action-space contract fields; missing={missing}, extra={extra}"
            )

        arm_dims = value["arm_dims"]
        gripper_dims = value["gripper_dims"]
        if not isinstance(arm_dims, list):
            raise ActionSpaceContractError("arm_dims must be a JSON list")
        if not isinstance(gripper_dims, list):
            raise ActionSpaceContractError("gripper_dims must be a JSON list")

        return cls(
            schema=value["schema"],
            version=value["version"],
            action_dim=value["action_dim"],
            arm_dims=tuple(arm_dims),
            gripper_dims=tuple(gripper_dims),
            gripper_threshold=value["gripper_threshold"],
            normalization_mode=value["normalization_mode"],
            normalization_stats_sha256=value["normalization_stats_sha256"],
            control_mode=value["control_mode"],
            embodiment=value["embodiment"],
        )


def validate_action_space_contract(value: Mapping[str, Any]) -> ActionSpaceContract:
    """Validate a JSON-like mapping and return its canonical typed form."""

    return ActionSpaceContract.from_dict(value)
