"""Factual, bounded working memory for one WARM environment episode.

The long-term WARM event bank stores cross-episode experience.  This module is
deliberately narrower: it records only observations that have actually been
returned by the environment in the *current* rollout.  It does not infer named
subtasks and it never accepts predicted future features.

The public write API is capability based.  ``begin_episode`` returns a token
bound to both the memory instance and its current episode generation.  Reset,
end, restore, or beginning another episode invalidates every older token.  This
is useful when environment observations are produced by asynchronous workers:
a delayed write from the previous episode fails closed instead of silently
polluting the next one.

The implementation is NumPy-first and intentionally independent of Torch.
Every array exposed by a public value object is a C-contiguous, finite,
read-only copy with an explicitly validated shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
import threading
from typing import Any, Mapping, Sequence

import numpy as np


EPISODE_MEMORY_SCHEMA = "warm.episode-working-memory"
EPISODE_MEMORY_VERSION = 1
FACTUAL_OBSERVATION_PROVENANCE = "environment_observation"

_FACTUAL_FACTORY_TOKEN = object()
_EPS = 1.0e-8


class EpisodeMemoryError(RuntimeError):
    """Base error for the episode working-memory contract."""


class EpisodeMemoryValidationError(EpisodeMemoryError, ValueError):
    """Raised when a value violates a shape, finiteness, or schema contract."""


class EpisodeMemoryLifecycleError(EpisodeMemoryError):
    """Raised when an operation is invalid for the current lifecycle state."""


class EpisodeMemoryCapabilityError(EpisodeMemoryLifecycleError):
    """Raised when a write capability is foreign, expired, or cross-episode."""


class EpisodeMemoryPhase(str, Enum):
    EMPTY = "empty"
    ACTIVE = "active"
    ENDED = "ended"


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise EpisodeMemoryValidationError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise EpisodeMemoryValidationError(f"{name} must be >= {minimum}")
    return result


def _strict_float(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise EpisodeMemoryValidationError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise EpisodeMemoryValidationError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise EpisodeMemoryValidationError(f"{name} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise EpisodeMemoryValidationError(f"{name} must be <= {maximum}")
    return result


def _normalise_episode_id(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        raise EpisodeMemoryValidationError("episode_id must be a non-empty string or integer")
    if isinstance(value, (int, np.integer)):
        value = str(int(value))
    if not isinstance(value, str) or not value.strip():
        raise EpisodeMemoryValidationError("episode_id must be a non-empty string or integer")
    return value.strip()


def _normalise_identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EpisodeMemoryValidationError(f"{name} must be a non-empty string")
    return value.strip()


def _readonly_float32(
    value: Any,
    name: str,
    *,
    ndim: int | None = None,
    shape: tuple[int, ...] | None = None,
    nonempty: bool = True,
) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except Exception as exc:  # pragma: no cover - NumPy supplies the detail.
        raise EpisodeMemoryValidationError(f"{name} must be numeric") from exc
    if raw.dtype.kind not in "fiu":
        raise EpisodeMemoryValidationError(f"{name} must contain real numeric values")
    try:
        array = np.array(raw, dtype=np.float32, order="C", copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise EpisodeMemoryValidationError(f"{name} cannot be represented as float32") from exc
    if ndim is not None and array.ndim != ndim:
        raise EpisodeMemoryValidationError(
            f"{name} must have ndim={ndim}, got shape {array.shape}"
        )
    if shape is not None and array.shape != shape:
        raise EpisodeMemoryValidationError(
            f"{name} must have shape {shape}, got {array.shape}"
        )
    if nonempty and (array.size == 0 or any(dimension <= 0 for dimension in array.shape)):
        raise EpisodeMemoryValidationError(f"{name} must be non-empty")
    if not np.isfinite(array).all():
        raise EpisodeMemoryValidationError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


def _readonly_int64(
    value: Any,
    name: str,
    *,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iub":
        raise EpisodeMemoryValidationError(f"{name} must contain integers")
    array = np.array(raw, dtype=np.int64, order="C", copy=True)
    if shape is not None and array.shape != shape:
        raise EpisodeMemoryValidationError(
            f"{name} must have shape {shape}, got {array.shape}"
        )
    if (array < 0).any():
        raise EpisodeMemoryValidationError(f"{name} cannot contain negative values")
    array.setflags(write=False)
    return array


def _copy_readonly(array: np.ndarray) -> np.ndarray:
    result = np.array(array, dtype=array.dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


def _relative_change(before: np.ndarray, after: np.ndarray) -> float:
    numerator = float(np.linalg.norm(after.astype(np.float64) - before.astype(np.float64)))
    denominator = float(np.linalg.norm(before.astype(np.float64))) + float(
        np.linalg.norm(after.astype(np.float64))
    )
    if denominator <= _EPS:
        return 0.0 if numerator <= _EPS else 1.0
    return float(min(1.0, numerator / (denominator + _EPS)))


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left64 = np.asarray(left, dtype=np.float64).reshape(-1)
    right64 = np.asarray(right, dtype=np.float64).reshape(-1)
    left_norm = float(np.linalg.norm(left64))
    right_norm = float(np.linalg.norm(right64))
    if left_norm <= _EPS and right_norm <= _EPS:
        return 1.0
    if left_norm <= _EPS or right_norm <= _EPS:
        return 0.0
    return float(np.clip(np.dot(left64, right64) / (left_norm * right_norm), -1.0, 1.0))


def _array_to_dict(array: np.ndarray) -> dict[str, Any]:
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "values": array.reshape(-1).tolist(),
    }


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise EpisodeMemoryValidationError(
            f"{name} fields mismatch; missing={missing}, extra={extra}"
        )


def _array_from_dict(
    value: Any,
    name: str,
    *,
    dtype: str,
    ndim: int | None = None,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    if not isinstance(value, Mapping):
        raise EpisodeMemoryValidationError(f"{name} must be an array object")
    _require_exact_keys(value, {"dtype", "shape", "values"}, name)
    if value["dtype"] != dtype:
        raise EpisodeMemoryValidationError(f"{name}.dtype must be {dtype!r}")
    raw_shape = value["shape"]
    if not isinstance(raw_shape, list):
        raise EpisodeMemoryValidationError(f"{name}.shape must be a list")
    parsed_shape = tuple(_strict_int(item, f"{name}.shape[{index}]") for index, item in enumerate(raw_shape))
    if ndim is not None and len(parsed_shape) != ndim:
        raise EpisodeMemoryValidationError(f"{name} must have ndim={ndim}")
    if shape is not None and parsed_shape != shape:
        raise EpisodeMemoryValidationError(
            f"{name} must have shape {shape}, got {parsed_shape}"
        )
    values = value["values"]
    if not isinstance(values, list):
        raise EpisodeMemoryValidationError(f"{name}.values must be a list")
    expected_size = math.prod(parsed_shape)
    if len(values) != expected_size:
        raise EpisodeMemoryValidationError(
            f"{name}.values has {len(values)} elements, expected {expected_size}"
        )
    if dtype == "float32":
        return _readonly_float32(
            np.asarray(values, dtype=np.float32).reshape(parsed_shape),
            name,
            ndim=ndim,
            shape=shape,
            nonempty=expected_size > 0,
        )
    if dtype == "int64":
        return _readonly_int64(
            np.asarray(values, dtype=np.int64).reshape(parsed_shape),
            name,
            shape=shape,
        )
    raise EpisodeMemoryValidationError(f"unsupported serialized dtype {dtype!r}")


@dataclass(frozen=True, slots=True)
class EpisodeMemoryConfig:
    """Static dimensions and deterministic bounded-memory policy."""

    action_dim: int
    gripper_indices: tuple[int, ...] = ()
    max_recent_events: int = 6
    max_action_summaries: int = 2
    change_history_size: int = 32
    change_warmup: int = 3
    change_threshold_floor: float = 0.08
    change_threshold_ceiling: float = 0.85
    change_mad_scale: float = 1.5
    # gripper, action-direction, world-feature, VAE-latent
    change_weights: tuple[float, float, float, float] = (0.25, 0.15, 0.35, 0.25)
    gripper_threshold: float = 0.0
    gripper_closed_when_below: bool = True
    motion_world_change_threshold: float = 0.08
    stationary_world_change_threshold: float = 0.03
    repetition_cosine_threshold: float = 0.97
    repetition_distance_threshold: float = 0.20

    def __post_init__(self) -> None:
        action_dim = _strict_int(self.action_dim, "action_dim", minimum=1)
        gripper_indices = tuple(
            _strict_int(index, f"gripper_indices[{position}]")
            for position, index in enumerate(self.gripper_indices)
        )
        if len(set(gripper_indices)) != len(gripper_indices):
            raise EpisodeMemoryValidationError("gripper_indices must be unique")
        if any(index >= action_dim for index in gripper_indices):
            raise EpisodeMemoryValidationError("gripper_indices must be within action_dim")
        object.__setattr__(self, "action_dim", action_dim)
        object.__setattr__(self, "gripper_indices", tuple(sorted(gripper_indices)))
        object.__setattr__(
            self,
            "max_recent_events",
            _strict_int(self.max_recent_events, "max_recent_events", minimum=2),
        )
        object.__setattr__(
            self,
            "max_action_summaries",
            _strict_int(self.max_action_summaries, "max_action_summaries", minimum=1),
        )
        object.__setattr__(
            self,
            "change_history_size",
            _strict_int(self.change_history_size, "change_history_size", minimum=1),
        )
        object.__setattr__(
            self,
            "change_warmup",
            _strict_int(self.change_warmup, "change_warmup", minimum=0),
        )
        floor = _strict_float(
            self.change_threshold_floor,
            "change_threshold_floor",
            minimum=0.0,
            maximum=1.0,
        )
        ceiling = _strict_float(
            self.change_threshold_ceiling,
            "change_threshold_ceiling",
            minimum=0.0,
            maximum=1.0,
        )
        if ceiling < floor:
            raise EpisodeMemoryValidationError(
                "change_threshold_ceiling must be >= change_threshold_floor"
            )
        object.__setattr__(self, "change_threshold_floor", floor)
        object.__setattr__(self, "change_threshold_ceiling", ceiling)
        object.__setattr__(
            self,
            "change_mad_scale",
            _strict_float(self.change_mad_scale, "change_mad_scale", minimum=0.0),
        )
        if not isinstance(self.change_weights, tuple) or len(self.change_weights) != 4:
            raise EpisodeMemoryValidationError("change_weights must be a tuple of four values")
        weights = tuple(
            _strict_float(weight, f"change_weights[{index}]", minimum=0.0)
            for index, weight in enumerate(self.change_weights)
        )
        if sum(weights) <= 0.0:
            raise EpisodeMemoryValidationError("change_weights must have positive total mass")
        object.__setattr__(self, "change_weights", weights)
        object.__setattr__(
            self,
            "gripper_threshold",
            _strict_float(self.gripper_threshold, "gripper_threshold"),
        )
        if not isinstance(self.gripper_closed_when_below, bool):
            raise EpisodeMemoryValidationError("gripper_closed_when_below must be bool")
        for field_name in (
            "motion_world_change_threshold",
            "stationary_world_change_threshold",
            "repetition_cosine_threshold",
            "repetition_distance_threshold",
        ):
            object.__setattr__(
                self,
                field_name,
                _strict_float(
                    getattr(self, field_name), field_name, minimum=0.0, maximum=1.0
                ),
            )

    @property
    def movement_indices(self) -> tuple[int, ...]:
        gripper = set(self.gripper_indices)
        return tuple(index for index in range(self.action_dim) if index not in gripper)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_dim": self.action_dim,
            "gripper_indices": list(self.gripper_indices),
            "max_recent_events": self.max_recent_events,
            "max_action_summaries": self.max_action_summaries,
            "change_history_size": self.change_history_size,
            "change_warmup": self.change_warmup,
            "change_threshold_floor": self.change_threshold_floor,
            "change_threshold_ceiling": self.change_threshold_ceiling,
            "change_mad_scale": self.change_mad_scale,
            "change_weights": list(self.change_weights),
            "gripper_threshold": self.gripper_threshold,
            "gripper_closed_when_below": self.gripper_closed_when_below,
            "motion_world_change_threshold": self.motion_world_change_threshold,
            "stationary_world_change_threshold": self.stationary_world_change_threshold,
            "repetition_cosine_threshold": self.repetition_cosine_threshold,
            "repetition_distance_threshold": self.repetition_distance_threshold,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "EpisodeMemoryConfig":
        if not isinstance(value, Mapping):
            raise EpisodeMemoryValidationError("config must be an object")
        expected = set(cls(action_dim=1).to_dict())
        _require_exact_keys(value, expected, "config")
        if not isinstance(value["gripper_indices"], list):
            raise EpisodeMemoryValidationError("config.gripper_indices must be a list")
        if not isinstance(value["change_weights"], list) or len(value["change_weights"]) != 4:
            raise EpisodeMemoryValidationError("config.change_weights must be a list of four values")
        return cls(
            action_dim=value["action_dim"],
            gripper_indices=tuple(value["gripper_indices"]),
            max_recent_events=value["max_recent_events"],
            max_action_summaries=value["max_action_summaries"],
            change_history_size=value["change_history_size"],
            change_warmup=value["change_warmup"],
            change_threshold_floor=value["change_threshold_floor"],
            change_threshold_ceiling=value["change_threshold_ceiling"],
            change_mad_scale=value["change_mad_scale"],
            change_weights=tuple(value["change_weights"]),
            gripper_threshold=value["gripper_threshold"],
            gripper_closed_when_below=value["gripper_closed_when_below"],
            motion_world_change_threshold=value["motion_world_change_threshold"],
            stationary_world_change_threshold=value["stationary_world_change_threshold"],
            repetition_cosine_threshold=value["repetition_cosine_threshold"],
            repetition_distance_threshold=value["repetition_distance_threshold"],
        )


@dataclass(frozen=True, slots=True, init=False)
class FactualObservation:
    """An immutable feature observation certified as environment feedback.

    The constructor requires a module-private token.  Callers must use
    :meth:`from_environment`; there is intentionally no predicted-feature
    factory.
    """

    episode_id: str
    frame_index: int
    observation_id: str
    world_tokens: np.ndarray
    vae_latent: np.ndarray
    proprio: np.ndarray
    provenance: str

    def __init__(
        self,
        *,
        episode_id: Any,
        frame_index: Any,
        world_tokens: Any,
        vae_latent: Any,
        proprio: Any,
        observation_id: str | None = None,
        _factory_token: object | None = None,
    ) -> None:
        if _factory_token is not _FACTUAL_FACTORY_TOKEN:
            raise EpisodeMemoryValidationError(
                "FactualObservation must be created with from_environment()"
            )
        parsed_episode_id = _normalise_episode_id(episode_id)
        parsed_frame = _strict_int(frame_index, "frame_index")
        if observation_id is None:
            observation_id = f"{parsed_episode_id}:{parsed_frame}"
        parsed_observation_id = _normalise_identifier(observation_id, "observation_id")
        world = _readonly_float32(world_tokens, "world_tokens", ndim=2)
        latent = _readonly_float32(vae_latent, "vae_latent")
        if latent.ndim < 1:
            raise EpisodeMemoryValidationError("vae_latent must have ndim >= 1")
        state = _readonly_float32(proprio, "proprio", ndim=1)
        object.__setattr__(self, "episode_id", parsed_episode_id)
        object.__setattr__(self, "frame_index", parsed_frame)
        object.__setattr__(self, "observation_id", parsed_observation_id)
        object.__setattr__(self, "world_tokens", world)
        object.__setattr__(self, "vae_latent", latent)
        object.__setattr__(self, "proprio", state)
        object.__setattr__(self, "provenance", FACTUAL_OBSERVATION_PROVENANCE)

    @classmethod
    def from_environment(
        cls,
        *,
        episode_id: Any,
        frame_index: Any,
        world_tokens: Any,
        vae_latent: Any,
        proprio: Any,
        observation_id: str | None = None,
    ) -> "FactualObservation":
        return cls(
            episode_id=episode_id,
            frame_index=frame_index,
            world_tokens=world_tokens,
            vae_latent=vae_latent,
            proprio=proprio,
            observation_id=observation_id,
            _factory_token=_FACTUAL_FACTORY_TOKEN,
        )

    @property
    def sha256(self) -> str:
        digest = hashlib.sha256()
        for text in (
            self.episode_id,
            str(self.frame_index),
            self.observation_id,
            self.provenance,
        ):
            encoded = text.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        for array in (self.world_tokens, self.vae_latent, self.proprio):
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
            digest.update(array.tobytes(order="C"))
        return digest.hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "frame_index": self.frame_index,
            "observation_id": self.observation_id,
            "provenance": self.provenance,
            "world_tokens": _array_to_dict(self.world_tokens),
            "vae_latent": _array_to_dict(self.vae_latent),
            "proprio": _array_to_dict(self.proprio),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "FactualObservation":
        if not isinstance(value, Mapping):
            raise EpisodeMemoryValidationError("observation must be an object")
        _require_exact_keys(
            value,
            {
                "episode_id",
                "frame_index",
                "observation_id",
                "provenance",
                "world_tokens",
                "vae_latent",
                "proprio",
            },
            "observation",
        )
        if value["provenance"] != FACTUAL_OBSERVATION_PROVENANCE:
            raise EpisodeMemoryValidationError(
                "serialized observation is not factual environment feedback"
            )
        world = _array_from_dict(value["world_tokens"], "observation.world_tokens", dtype="float32", ndim=2)
        latent = _array_from_dict(value["vae_latent"], "observation.vae_latent", dtype="float32")
        if latent.ndim < 1:
            raise EpisodeMemoryValidationError("observation.vae_latent must have ndim >= 1")
        proprio = _array_from_dict(value["proprio"], "observation.proprio", dtype="float32", ndim=1)
        return cls(
            episode_id=value["episode_id"],
            frame_index=value["frame_index"],
            observation_id=value["observation_id"],
            world_tokens=world,
            vae_latent=latent,
            proprio=proprio,
            _factory_token=_FACTUAL_FACTORY_TOKEN,
        )


@dataclass(frozen=True, slots=True)
class EpisodeWriteCapability:
    """Opaque permission to mutate one active episode generation."""

    episode_id: str
    _generation: int = field(repr=False, compare=False)
    _owner: object = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class ActionSummary:
    start_frame: int
    end_frame: int
    step_count: int
    mean_displacement: np.ndarray
    final_displacement: np.ndarray
    terminal_gripper_values: np.ndarray
    gripper_transition_counts: np.ndarray
    curvature: float
    repetition_similarity: float
    repeated: bool

    def __post_init__(self) -> None:
        start = _strict_int(self.start_frame, "action_summary.start_frame")
        end = _strict_int(self.end_frame, "action_summary.end_frame")
        if end <= start:
            raise EpisodeMemoryValidationError(
                "action_summary.end_frame must be greater than start_frame"
            )
        steps = _strict_int(self.step_count, "action_summary.step_count", minimum=1)
        mean = _readonly_float32(self.mean_displacement, "action_summary.mean_displacement", ndim=1)
        final = _readonly_float32(
            self.final_displacement,
            "action_summary.final_displacement",
            ndim=1,
            shape=mean.shape,
        )
        terminal = _readonly_float32(
            self.terminal_gripper_values,
            "action_summary.terminal_gripper_values",
            ndim=1,
            nonempty=False,
        )
        transitions = _readonly_int64(
            self.gripper_transition_counts,
            "action_summary.gripper_transition_counts",
            shape=(terminal.shape[0], 2),
        )
        curvature = _strict_float(self.curvature, "action_summary.curvature", minimum=0.0, maximum=1.0)
        similarity = _strict_float(
            self.repetition_similarity,
            "action_summary.repetition_similarity",
            minimum=-1.0,
            maximum=1.0,
        )
        if not isinstance(self.repeated, bool):
            raise EpisodeMemoryValidationError("action_summary.repeated must be bool")
        object.__setattr__(self, "start_frame", start)
        object.__setattr__(self, "end_frame", end)
        object.__setattr__(self, "step_count", steps)
        object.__setattr__(self, "mean_displacement", mean)
        object.__setattr__(self, "final_displacement", final)
        object.__setattr__(self, "terminal_gripper_values", terminal)
        object.__setattr__(self, "gripper_transition_counts", transitions)
        object.__setattr__(self, "curvature", curvature)
        object.__setattr__(self, "repetition_similarity", similarity)

    @property
    def close_count(self) -> int:
        return int(self.gripper_transition_counts[:, 0].sum())

    @property
    def release_count(self) -> int:
        return int(self.gripper_transition_counts[:, 1].sum())

    def signature(self) -> np.ndarray:
        return np.concatenate(
            [
                self.mean_displacement.astype(np.float64),
                self.final_displacement.astype(np.float64),
                # Boundary transition counts depend on the command immediately
                # preceding this chunk.  Terminal gripper commands do not, so
                # they let repetition describe the chunk itself rather than
                # its incidental predecessor.
                self.terminal_gripper_values.astype(np.float64),
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "step_count": self.step_count,
            "mean_displacement": _array_to_dict(self.mean_displacement),
            "final_displacement": _array_to_dict(self.final_displacement),
            "terminal_gripper_values": _array_to_dict(self.terminal_gripper_values),
            "gripper_transition_counts": _array_to_dict(self.gripper_transition_counts),
            "curvature": self.curvature,
            "repetition_similarity": self.repetition_similarity,
            "repeated": self.repeated,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "ActionSummary":
        if not isinstance(value, Mapping):
            raise EpisodeMemoryValidationError("action_summary must be an object")
        _require_exact_keys(
            value,
            {
                "start_frame",
                "end_frame",
                "step_count",
                "mean_displacement",
                "final_displacement",
                "terminal_gripper_values",
                "gripper_transition_counts",
                "curvature",
                "repetition_similarity",
                "repeated",
            },
            "action_summary",
        )
        mean = _array_from_dict(
            value["mean_displacement"], "action_summary.mean_displacement", dtype="float32", ndim=1
        )
        final = _array_from_dict(
            value["final_displacement"],
            "action_summary.final_displacement",
            dtype="float32",
            ndim=1,
            shape=mean.shape,
        )
        terminal = _array_from_dict(
            value["terminal_gripper_values"],
            "action_summary.terminal_gripper_values",
            dtype="float32",
            ndim=1,
        )
        transitions = _array_from_dict(
            value["gripper_transition_counts"],
            "action_summary.gripper_transition_counts",
            dtype="int64",
            ndim=2,
            shape=(terminal.shape[0], 2),
        )
        return cls(
            start_frame=value["start_frame"],
            end_frame=value["end_frame"],
            step_count=value["step_count"],
            mean_displacement=mean,
            final_displacement=final,
            terminal_gripper_values=terminal,
            gripper_transition_counts=transitions,
            curvature=value["curvature"],
            repetition_similarity=value["repetition_similarity"],
            repeated=value["repeated"],
        )


@dataclass(frozen=True, slots=True)
class EpisodeEventStatus:
    close_count: int = 0
    release_count: int = 0
    last_close_frame: int | None = None
    last_release_frame: int | None = None
    gripper_closed_recently: bool = False
    visual_motion_after_close: bool = False
    release_happened: bool = False
    stationary_after_release: bool = False
    repeated_similar_action: bool = False
    repeated_attempt_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "close_count", _strict_int(self.close_count, "event_status.close_count"))
        object.__setattr__(self, "release_count", _strict_int(self.release_count, "event_status.release_count"))
        for name in ("last_close_frame", "last_release_frame"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _strict_int(value, f"event_status.{name}"))
        for name in (
            "gripper_closed_recently",
            "visual_motion_after_close",
            "release_happened",
            "stationary_after_release",
            "repeated_similar_action",
        ):
            if not isinstance(getattr(self, name), bool):
                raise EpisodeMemoryValidationError(f"event_status.{name} must be bool")
        object.__setattr__(
            self,
            "repeated_attempt_count",
            _strict_int(self.repeated_attempt_count, "event_status.repeated_attempt_count"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "close_count": self.close_count,
            "release_count": self.release_count,
            "last_close_frame": self.last_close_frame,
            "last_release_frame": self.last_release_frame,
            "gripper_closed_recently": self.gripper_closed_recently,
            "visual_motion_after_close": self.visual_motion_after_close,
            "release_happened": self.release_happened,
            "stationary_after_release": self.stationary_after_release,
            "repeated_similar_action": self.repeated_similar_action,
            "repeated_attempt_count": self.repeated_attempt_count,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "EpisodeEventStatus":
        if not isinstance(value, Mapping):
            raise EpisodeMemoryValidationError("event_status must be an object")
        expected = set(cls().to_dict())
        _require_exact_keys(value, expected, "event_status")
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class EpisodeEvent:
    start_frame: int
    end_frame: int
    start_observation_id: str
    end_observation_id: str
    pre_world_tokens: np.ndarray
    post_world_tokens: np.ndarray
    representative_world_tokens: np.ndarray
    world_delta: np.ndarray
    vae_delta: np.ndarray
    change_components: np.ndarray
    change_score: float
    write_threshold: float
    action_summary: ActionSummary
    mass: int = 1
    compressed: bool = False

    def __post_init__(self) -> None:
        start = _strict_int(self.start_frame, "event.start_frame")
        end = _strict_int(self.end_frame, "event.end_frame")
        if end <= start:
            raise EpisodeMemoryValidationError("event.end_frame must be greater than start_frame")
        pre = _readonly_float32(self.pre_world_tokens, "event.pre_world_tokens", ndim=2)
        post = _readonly_float32(
            self.post_world_tokens, "event.post_world_tokens", ndim=2, shape=pre.shape
        )
        representative = _readonly_float32(
            self.representative_world_tokens,
            "event.representative_world_tokens",
            ndim=2,
            shape=pre.shape,
        )
        delta = _readonly_float32(self.world_delta, "event.world_delta", ndim=2, shape=pre.shape)
        if not np.array_equal(delta, np.asarray(post - pre, dtype=np.float32)):
            raise EpisodeMemoryValidationError("event.world_delta must equal post_world_tokens - pre_world_tokens")
        vae_delta = _readonly_float32(self.vae_delta, "event.vae_delta")
        if vae_delta.ndim < 1:
            raise EpisodeMemoryValidationError("event.vae_delta must have ndim >= 1")
        components = _readonly_float32(
            self.change_components, "event.change_components", ndim=1, shape=(4,)
        )
        if (components < 0.0).any() or (components > 1.0).any():
            raise EpisodeMemoryValidationError("event.change_components must be within [0, 1]")
        score = _strict_float(self.change_score, "event.change_score", minimum=0.0, maximum=1.0)
        threshold = _strict_float(self.write_threshold, "event.write_threshold", minimum=0.0, maximum=1.0)
        mass = _strict_int(self.mass, "event.mass", minimum=1)
        if not isinstance(self.action_summary, ActionSummary):
            raise EpisodeMemoryValidationError("event.action_summary must be ActionSummary")
        if not isinstance(self.compressed, bool):
            raise EpisodeMemoryValidationError("event.compressed must be bool")
        object.__setattr__(self, "start_frame", start)
        object.__setattr__(self, "end_frame", end)
        object.__setattr__(
            self, "start_observation_id", _normalise_identifier(self.start_observation_id, "event.start_observation_id")
        )
        object.__setattr__(
            self, "end_observation_id", _normalise_identifier(self.end_observation_id, "event.end_observation_id")
        )
        object.__setattr__(self, "pre_world_tokens", pre)
        object.__setattr__(self, "post_world_tokens", post)
        object.__setattr__(self, "representative_world_tokens", representative)
        object.__setattr__(self, "world_delta", delta)
        object.__setattr__(self, "vae_delta", vae_delta)
        object.__setattr__(self, "change_components", components)
        object.__setattr__(self, "change_score", score)
        object.__setattr__(self, "write_threshold", threshold)
        object.__setattr__(self, "mass", mass)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_observation_id": self.start_observation_id,
            "end_observation_id": self.end_observation_id,
            "pre_world_tokens": _array_to_dict(self.pre_world_tokens),
            "post_world_tokens": _array_to_dict(self.post_world_tokens),
            "representative_world_tokens": _array_to_dict(self.representative_world_tokens),
            "world_delta": _array_to_dict(self.world_delta),
            "vae_delta": _array_to_dict(self.vae_delta),
            "change_components": _array_to_dict(self.change_components),
            "change_score": self.change_score,
            "write_threshold": self.write_threshold,
            "action_summary": self.action_summary.to_dict(),
            "mass": self.mass,
            "compressed": self.compressed,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "EpisodeEvent":
        if not isinstance(value, Mapping):
            raise EpisodeMemoryValidationError("event must be an object")
        expected = {
            "start_frame",
            "end_frame",
            "start_observation_id",
            "end_observation_id",
            "pre_world_tokens",
            "post_world_tokens",
            "representative_world_tokens",
            "world_delta",
            "vae_delta",
            "change_components",
            "change_score",
            "write_threshold",
            "action_summary",
            "mass",
            "compressed",
        }
        _require_exact_keys(value, expected, "event")
        pre = _array_from_dict(value["pre_world_tokens"], "event.pre_world_tokens", dtype="float32", ndim=2)
        return cls(
            start_frame=value["start_frame"],
            end_frame=value["end_frame"],
            start_observation_id=value["start_observation_id"],
            end_observation_id=value["end_observation_id"],
            pre_world_tokens=pre,
            post_world_tokens=_array_from_dict(
                value["post_world_tokens"], "event.post_world_tokens", dtype="float32", ndim=2, shape=pre.shape
            ),
            representative_world_tokens=_array_from_dict(
                value["representative_world_tokens"],
                "event.representative_world_tokens",
                dtype="float32",
                ndim=2,
                shape=pre.shape,
            ),
            world_delta=_array_from_dict(
                value["world_delta"], "event.world_delta", dtype="float32", ndim=2, shape=pre.shape
            ),
            vae_delta=_array_from_dict(value["vae_delta"], "event.vae_delta", dtype="float32"),
            change_components=_array_from_dict(
                value["change_components"], "event.change_components", dtype="float32", ndim=1, shape=(4,)
            ),
            change_score=value["change_score"],
            write_threshold=value["write_threshold"],
            action_summary=ActionSummary.from_dict(value["action_summary"]),
            mass=value["mass"],
            compressed=value["compressed"],
        )


@dataclass(frozen=True, slots=True)
class EpisodeMemoryCounters:
    observation_updates: int = 0
    event_writes: int = 0
    event_merges: int = 0

    def __post_init__(self) -> None:
        for name in ("observation_updates", "event_writes", "event_merges"):
            object.__setattr__(self, name, _strict_int(getattr(self, name), f"counters.{name}"))

    def to_dict(self) -> dict[str, int]:
        return {
            "observation_updates": self.observation_updates,
            "event_writes": self.event_writes,
            "event_merges": self.event_merges,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "EpisodeMemoryCounters":
        if not isinstance(value, Mapping):
            raise EpisodeMemoryValidationError("counters must be an object")
        expected = set(cls().to_dict())
        _require_exact_keys(value, expected, "counters")
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class EpisodeMemorySnapshot:
    config: EpisodeMemoryConfig
    phase: EpisodeMemoryPhase
    episode_id: str | None
    initial_anchor: FactualObservation | None
    latest_observation: FactualObservation | None
    recent_events: tuple[EpisodeEvent, ...]
    executed_action_summaries: tuple[ActionSummary, ...]
    event_status: EpisodeEventStatus
    change_score_history: tuple[float, ...]
    counters: EpisodeMemoryCounters

    def __post_init__(self) -> None:
        if not isinstance(self.config, EpisodeMemoryConfig):
            raise EpisodeMemoryValidationError("snapshot.config must be EpisodeMemoryConfig")
        try:
            phase = EpisodeMemoryPhase(self.phase)
        except ValueError as exc:
            raise EpisodeMemoryValidationError("snapshot.phase is invalid") from exc
        object.__setattr__(self, "phase", phase)
        if phase is EpisodeMemoryPhase.EMPTY:
            if self.episode_id is not None or self.initial_anchor is not None or self.latest_observation is not None:
                raise EpisodeMemoryValidationError("empty snapshot cannot contain episode observations")
            if self.recent_events or self.executed_action_summaries or self.change_score_history:
                raise EpisodeMemoryValidationError("empty snapshot cannot contain episode history")
            if self.event_status != EpisodeEventStatus() or self.counters != EpisodeMemoryCounters():
                raise EpisodeMemoryValidationError("empty snapshot counters and status must be zero")
        else:
            episode_id = _normalise_episode_id(self.episode_id)
            object.__setattr__(self, "episode_id", episode_id)
            if self.initial_anchor is None or self.latest_observation is None:
                raise EpisodeMemoryValidationError("non-empty snapshot requires initial and latest observations")
            if self.initial_anchor.episode_id != episode_id or self.latest_observation.episode_id != episode_id:
                raise EpisodeMemoryValidationError("snapshot observation episode_id mismatch")
            if self.latest_observation.frame_index < self.initial_anchor.frame_index:
                raise EpisodeMemoryValidationError(
                    "snapshot latest observation precedes the initial anchor"
                )
            if self.latest_observation.world_tokens.shape != self.initial_anchor.world_tokens.shape:
                raise EpisodeMemoryValidationError("snapshot world token shapes do not match")
            if self.latest_observation.vae_latent.shape != self.initial_anchor.vae_latent.shape:
                raise EpisodeMemoryValidationError("snapshot VAE latent shapes do not match")
            if self.latest_observation.proprio.shape != self.initial_anchor.proprio.shape:
                raise EpisodeMemoryValidationError("snapshot proprio shapes do not match")
        events = tuple(self.recent_events)
        summaries = tuple(self.executed_action_summaries)
        if len(events) > self.config.max_recent_events:
            raise EpisodeMemoryValidationError("snapshot has too many recent events")
        if len(summaries) > self.config.max_action_summaries:
            raise EpisodeMemoryValidationError("snapshot has too many action summaries")
        if any(not isinstance(event, EpisodeEvent) for event in events):
            raise EpisodeMemoryValidationError("snapshot recent_events must contain EpisodeEvent")
        if any(not isinstance(summary, ActionSummary) for summary in summaries):
            raise EpisodeMemoryValidationError(
                "snapshot executed_action_summaries must contain ActionSummary"
            )
        object.__setattr__(self, "recent_events", events)
        object.__setattr__(self, "executed_action_summaries", summaries)
        for index, summary in enumerate(summaries):
            if summary.mean_displacement.shape != (self.config.action_dim,):
                raise EpisodeMemoryValidationError(
                    f"snapshot action summary {index} does not match action_dim"
                )
            if summary.terminal_gripper_values.shape != (
                len(self.config.gripper_indices),
            ):
                raise EpisodeMemoryValidationError(
                    f"snapshot action summary {index} does not match gripper_indices"
                )
        for index, event in enumerate(events):
            if self.initial_anchor is None:
                raise EpisodeMemoryValidationError("events require an initial anchor")
            if event.pre_world_tokens.shape != self.initial_anchor.world_tokens.shape:
                raise EpisodeMemoryValidationError(
                    f"snapshot event {index} world token shape mismatch"
                )
            if event.vae_delta.shape != self.initial_anchor.vae_latent.shape:
                raise EpisodeMemoryValidationError(
                    f"snapshot event {index} VAE latent shape mismatch"
                )
            if event.action_summary.mean_displacement.shape != (
                self.config.action_dim,
            ):
                raise EpisodeMemoryValidationError(
                    f"snapshot event {index} action_dim mismatch"
                )
            if index and events[index - 1].end_frame > event.start_frame:
                raise EpisodeMemoryValidationError(
                    "snapshot recent events must be temporally ordered"
                )
            if self.latest_observation is not None and event.end_frame > self.latest_observation.frame_index:
                raise EpisodeMemoryValidationError(
                    f"snapshot event {index} extends past latest observation"
                )
        if not isinstance(self.event_status, EpisodeEventStatus):
            raise EpisodeMemoryValidationError("snapshot.event_status must be EpisodeEventStatus")
        history = tuple(
            _strict_float(score, f"change_score_history[{index}]", minimum=0.0, maximum=1.0)
            for index, score in enumerate(self.change_score_history)
        )
        if len(history) > self.config.change_history_size:
            raise EpisodeMemoryValidationError("snapshot change_score_history exceeds configured bound")
        object.__setattr__(self, "change_score_history", history)
        if not isinstance(self.counters, EpisodeMemoryCounters):
            raise EpisodeMemoryValidationError("snapshot.counters must be EpisodeMemoryCounters")
        counters = self.counters
        if counters.event_writes > counters.observation_updates:
            raise EpisodeMemoryValidationError("snapshot event writes exceed observation updates")
        if counters.event_merges > counters.event_writes:
            raise EpisodeMemoryValidationError("snapshot event merges exceed event writes")
        if counters.event_writes - counters.event_merges != len(events):
            raise EpisodeMemoryValidationError(
                "snapshot event counters do not match retained event slots"
            )
        if sum(event.mass for event in events) != counters.event_writes:
            raise EpisodeMemoryValidationError(
                "snapshot event masses do not match total event writes"
            )
        expected_history = min(counters.observation_updates, self.config.change_history_size)
        if len(history) != expected_history:
            raise EpisodeMemoryValidationError(
                "snapshot change-score history length does not match update count"
            )
        expected_summaries = min(
            counters.observation_updates, self.config.max_action_summaries
        )
        if len(summaries) != expected_summaries:
            raise EpisodeMemoryValidationError(
                "snapshot action-summary length does not match update count"
            )
        if counters.observation_updates == 0 and phase is not EpisodeMemoryPhase.EMPTY:
            if (
                self.initial_anchor is None
                or self.latest_observation is None
                or self.initial_anchor.sha256 != self.latest_observation.sha256
            ):
                raise EpisodeMemoryValidationError(
                    "snapshot with zero updates must keep latest equal to initial anchor"
                )
        elif self.latest_observation is not None:
            if summaries[-1].end_frame != self.latest_observation.frame_index:
                raise EpisodeMemoryValidationError(
                    "snapshot latest action summary does not end at latest observation"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": EPISODE_MEMORY_SCHEMA,
            "version": EPISODE_MEMORY_VERSION,
            "config": self.config.to_dict(),
            "state": {
                "phase": self.phase.value,
                "episode_id": self.episode_id,
                "initial_anchor": None if self.initial_anchor is None else self.initial_anchor.to_dict(),
                "latest_observation": None
                if self.latest_observation is None
                else self.latest_observation.to_dict(),
                "recent_events": [event.to_dict() for event in self.recent_events],
                "executed_action_summaries": [
                    summary.to_dict() for summary in self.executed_action_summaries
                ],
                "event_status": self.event_status.to_dict(),
                "change_score_history": list(self.change_score_history),
                "counters": self.counters.to_dict(),
            },
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class EpisodeMemoryUpdate:
    observation_id: str
    action_summary: ActionSummary
    change_components: np.ndarray
    change_score: float
    write_threshold: float
    event_written: bool
    latest_event: EpisodeEvent | None
    event_status: EpisodeEventStatus
    counters: EpisodeMemoryCounters

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observation_id",
            _normalise_identifier(self.observation_id, "update.observation_id"),
        )
        if not isinstance(self.action_summary, ActionSummary):
            raise EpisodeMemoryValidationError("update.action_summary must be ActionSummary")
        components = _readonly_float32(
            self.change_components, "update.change_components", ndim=1, shape=(4,)
        )
        object.__setattr__(self, "change_components", components)
        object.__setattr__(
            self,
            "change_score",
            _strict_float(self.change_score, "update.change_score", minimum=0.0, maximum=1.0),
        )
        object.__setattr__(
            self,
            "write_threshold",
            _strict_float(self.write_threshold, "update.write_threshold", minimum=0.0, maximum=1.0),
        )
        if not isinstance(self.event_written, bool):
            raise EpisodeMemoryValidationError("update.event_written must be bool")
        if self.event_written != (self.latest_event is not None):
            raise EpisodeMemoryValidationError(
                "update.latest_event presence must equal event_written"
            )
        if not isinstance(self.event_status, EpisodeEventStatus):
            raise EpisodeMemoryValidationError("update.event_status must be EpisodeEventStatus")
        if not isinstance(self.counters, EpisodeMemoryCounters):
            raise EpisodeMemoryValidationError("update.counters must be EpisodeMemoryCounters")


class EpisodeWorkingMemory:
    """Bounded factual working memory for a single active rollout."""

    def __init__(self, config: EpisodeMemoryConfig):
        if not isinstance(config, EpisodeMemoryConfig):
            raise EpisodeMemoryValidationError("config must be EpisodeMemoryConfig")
        self._config = config
        self._owner = object()
        self._generation = 0
        self._lock = threading.RLock()
        self._clear(EpisodeMemoryPhase.EMPTY)

    @property
    def config(self) -> EpisodeMemoryConfig:
        return self._config

    @property
    def phase(self) -> EpisodeMemoryPhase:
        with self._lock:
            return self._phase

    @property
    def active_episode_id(self) -> str | None:
        with self._lock:
            return self._episode_id if self._phase is EpisodeMemoryPhase.ACTIVE else None

    def _clear(self, phase: EpisodeMemoryPhase) -> None:
        self._phase = phase
        self._episode_id: str | None = None
        self._initial_anchor: FactualObservation | None = None
        self._latest_observation: FactualObservation | None = None
        self._recent_events: list[EpisodeEvent] = []
        self._action_summaries: list[ActionSummary] = []
        self._event_status = EpisodeEventStatus()
        self._change_history: list[float] = []
        self._counters = EpisodeMemoryCounters()

    def reset(self) -> None:
        """Clear all state and invalidate every outstanding write token."""

        with self._lock:
            self._generation += 1
            self._clear(EpisodeMemoryPhase.EMPTY)

    def begin_episode(self, initial_observation: FactualObservation) -> EpisodeWriteCapability:
        """Start a fresh episode and return its exclusive write capability."""

        if not isinstance(initial_observation, FactualObservation):
            raise EpisodeMemoryValidationError(
                "initial_observation must be a FactualObservation from the environment"
            )
        with self._lock:
            self._generation += 1
            self._clear(EpisodeMemoryPhase.ACTIVE)
            self._episode_id = initial_observation.episode_id
            self._initial_anchor = initial_observation
            self._latest_observation = initial_observation
            return EpisodeWriteCapability(
                episode_id=initial_observation.episode_id,
                _generation=self._generation,
                _owner=self._owner,
            )

    def _validate_capability(self, capability: EpisodeWriteCapability) -> None:
        if not isinstance(capability, EpisodeWriteCapability):
            raise EpisodeMemoryCapabilityError("a valid EpisodeWriteCapability is required")
        if capability._owner is not self._owner:
            raise EpisodeMemoryCapabilityError("write capability belongs to another memory instance")
        if capability._generation != self._generation:
            raise EpisodeMemoryCapabilityError("write capability is stale")
        if self._phase is not EpisodeMemoryPhase.ACTIVE:
            raise EpisodeMemoryLifecycleError("episode working memory is not active")
        if capability.episode_id != self._episode_id:
            raise EpisodeMemoryCapabilityError("write capability belongs to another episode")

    def _validate_observation(self, observation: FactualObservation) -> None:
        if not isinstance(observation, FactualObservation):
            raise EpisodeMemoryValidationError(
                "observation must be a FactualObservation from the environment"
            )
        assert self._latest_observation is not None
        if observation.episode_id != self._episode_id:
            raise EpisodeMemoryCapabilityError("observation belongs to another episode")
        if observation.frame_index <= self._latest_observation.frame_index:
            raise EpisodeMemoryLifecycleError(
                "observation frame_index must increase strictly within an episode"
            )
        if observation.observation_id == self._latest_observation.observation_id:
            raise EpisodeMemoryLifecycleError("observation_id must not repeat")
        if observation.world_tokens.shape != self._latest_observation.world_tokens.shape:
            raise EpisodeMemoryValidationError(
                "world_tokens shape changed within the active episode"
            )
        if observation.vae_latent.shape != self._latest_observation.vae_latent.shape:
            raise EpisodeMemoryValidationError(
                "vae_latent shape changed within the active episode"
            )
        if observation.proprio.shape != self._latest_observation.proprio.shape:
            raise EpisodeMemoryValidationError(
                "proprio shape changed within the active episode"
            )

    def _is_closed(self, values: np.ndarray) -> np.ndarray:
        if self._config.gripper_closed_when_below:
            return values <= self._config.gripper_threshold
        return values >= self._config.gripper_threshold

    def _gripper_transitions(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        count = len(self._config.gripper_indices)
        if count == 0:
            return _readonly_int64(np.zeros((0, 2), dtype=np.int64), "gripper_transitions", shape=(0, 2)), _readonly_float32(
                np.zeros((0,), dtype=np.float32), "terminal_gripper_values", ndim=1, nonempty=False
            )
        channels = actions[:, self._config.gripper_indices]
        terminal = _readonly_float32(
            channels[-1], "terminal_gripper_values", ndim=1, shape=(count,)
        )
        states = self._is_closed(channels)
        if self._action_summaries:
            prior_values = self._action_summaries[-1].terminal_gripper_values.reshape(1, count)
            prior_states = self._is_closed(prior_values)
            states = np.concatenate([prior_states, states], axis=0)
        transitions = np.diff(states.astype(np.int8), axis=0)
        close = (transitions == 1).sum(axis=0, dtype=np.int64)
        release = (transitions == -1).sum(axis=0, dtype=np.int64)
        counts = np.stack([close, release], axis=1)
        return _readonly_int64(counts, "gripper_transitions", shape=(count, 2)), terminal

    def _curvature(self, actions: np.ndarray) -> float:
        movement = self._config.movement_indices
        if len(movement) == 0 or actions.shape[0] < 2:
            return 0.0
        vectors = actions[:, movement].astype(np.float64)
        bends: list[float] = []
        for previous, current in zip(vectors[:-1], vectors[1:]):
            previous_norm = float(np.linalg.norm(previous))
            current_norm = float(np.linalg.norm(current))
            if previous_norm <= _EPS or current_norm <= _EPS:
                continue
            cosine = float(
                np.clip(
                    np.dot(previous, current) / (previous_norm * current_norm),
                    -1.0,
                    1.0,
                )
            )
            bends.append(0.5 * (1.0 - cosine))
        return 0.0 if not bends else float(np.mean(bends))

    def _make_action_summary(
        self,
        actions: np.ndarray,
        *,
        start_frame: int,
        end_frame: int,
    ) -> ActionSummary:
        mean = np.zeros((self._config.action_dim,), dtype=np.float32)
        final = np.zeros((self._config.action_dim,), dtype=np.float32)
        movement = self._config.movement_indices
        if movement:
            mean[list(movement)] = np.mean(actions[:, movement], axis=0, dtype=np.float32)
            final[list(movement)] = np.sum(actions[:, movement], axis=0, dtype=np.float32)
        transitions, terminal = self._gripper_transitions(actions)
        provisional = ActionSummary(
            start_frame=start_frame,
            end_frame=end_frame,
            step_count=actions.shape[0],
            mean_displacement=mean,
            final_displacement=final,
            terminal_gripper_values=terminal,
            gripper_transition_counts=transitions,
            curvature=self._curvature(actions),
            repetition_similarity=0.0,
            repeated=False,
        )
        best_similarity = -1.0
        best_distance = math.inf
        signature = provisional.signature()
        for previous in self._action_summaries:
            previous_signature = previous.signature()
            similarity = _cosine_similarity(signature, previous_signature)
            denominator = float(np.linalg.norm(signature)) + float(
                np.linalg.norm(previous_signature)
            )
            distance = (
                0.0
                if denominator <= _EPS
                else float(np.linalg.norm(signature - previous_signature) / (denominator + _EPS))
            )
            if similarity > best_similarity or (
                math.isclose(similarity, best_similarity) and distance < best_distance
            ):
                best_similarity = similarity
                best_distance = distance
        if not self._action_summaries:
            best_similarity = 0.0
            best_distance = math.inf
        repeated = bool(
            best_similarity >= self._config.repetition_cosine_threshold
            and best_distance <= self._config.repetition_distance_threshold
        )
        return ActionSummary(
            start_frame=provisional.start_frame,
            end_frame=provisional.end_frame,
            step_count=provisional.step_count,
            mean_displacement=provisional.mean_displacement,
            final_displacement=provisional.final_displacement,
            terminal_gripper_values=provisional.terminal_gripper_values,
            gripper_transition_counts=provisional.gripper_transition_counts,
            curvature=provisional.curvature,
            repetition_similarity=best_similarity,
            repeated=repeated,
        )

    def _action_direction_change(self, summary: ActionSummary) -> float:
        if not self._action_summaries:
            return summary.curvature
        previous = self._action_summaries[-1]
        movement = self._config.movement_indices
        if not movement:
            return 0.0
        similarity = _cosine_similarity(
            summary.final_displacement[list(movement)],
            previous.final_displacement[list(movement)],
        )
        return float(np.clip(0.5 * (1.0 - similarity), 0.0, 1.0))

    def _adaptive_threshold(self) -> float:
        config = self._config
        if not self._change_history or len(self._change_history) < config.change_warmup:
            return config.change_threshold_floor
        history = np.asarray(self._change_history, dtype=np.float64)
        median = float(np.median(history))
        mad = float(np.median(np.abs(history - median)))
        robust_sigma = 1.4826 * mad
        value = median + config.change_mad_scale * robust_sigma
        return float(
            np.clip(value, config.change_threshold_floor, config.change_threshold_ceiling)
        )

    def _merge_action_summaries(
        self, older: ActionSummary, newer: ActionSummary, older_mass: int, newer_mass: int
    ) -> ActionSummary:
        total = older_mass + newer_mass
        mean = (
            older.mean_displacement.astype(np.float64) * older_mass
            + newer.mean_displacement.astype(np.float64) * newer_mass
        ) / total
        # Net displacement and transition counts are additive across the
        # compressed temporal interval; other statistics are mass weighted.
        final = older.final_displacement.astype(np.float64) + newer.final_displacement.astype(np.float64)
        transitions = older.gripper_transition_counts + newer.gripper_transition_counts
        curvature = (older.curvature * older_mass + newer.curvature * newer_mass) / total
        return ActionSummary(
            start_frame=older.start_frame,
            end_frame=newer.end_frame,
            step_count=older.step_count + newer.step_count,
            mean_displacement=mean,
            final_displacement=final,
            terminal_gripper_values=newer.terminal_gripper_values,
            gripper_transition_counts=transitions,
            curvature=curvature,
            repetition_similarity=max(
                older.repetition_similarity, newer.repetition_similarity
            ),
            repeated=older.repeated or newer.repeated,
        )

    def _merge_events(self, older: EpisodeEvent, newer: EpisodeEvent) -> EpisodeEvent:
        total = older.mass + newer.mass
        representative = (
            older.representative_world_tokens.astype(np.float64) * older.mass
            + newer.representative_world_tokens.astype(np.float64) * newer.mass
        ) / total
        components = (
            older.change_components.astype(np.float64) * older.mass
            + newer.change_components.astype(np.float64) * newer.mass
        ) / total
        pre = older.pre_world_tokens
        post = newer.post_world_tokens
        # The boundary states preserve the real temporal transition while the
        # representative token is the DIM-style mass-weighted compressed slot.
        return EpisodeEvent(
            start_frame=older.start_frame,
            end_frame=newer.end_frame,
            start_observation_id=older.start_observation_id,
            end_observation_id=newer.end_observation_id,
            pre_world_tokens=pre,
            post_world_tokens=post,
            representative_world_tokens=representative,
            world_delta=np.asarray(post - pre, dtype=np.float32),
            vae_delta=(
                older.vae_delta.astype(np.float64) * older.mass
                + newer.vae_delta.astype(np.float64) * newer.mass
            )
            / total,
            change_components=components,
            change_score=(
                older.change_score * older.mass + newer.change_score * newer.mass
            )
            / total,
            write_threshold=(
                older.write_threshold * older.mass + newer.write_threshold * newer.mass
            )
            / total,
            action_summary=self._merge_action_summaries(
                older.action_summary, newer.action_summary, older.mass, newer.mass
            ),
            mass=total,
            compressed=True,
        )

    def _compress_if_needed(self) -> None:
        while len(self._recent_events) > self._config.max_recent_events:
            # The newly appended latest event is deliberately excluded.  The
            # initial anchor is stored separately and therefore cannot be a
            # merge candidate either.
            candidate_count = len(self._recent_events) - 2
            if candidate_count < 1:  # Config validation makes this unreachable.
                raise EpisodeMemoryLifecycleError("no legal middle event pair to merge")
            similarities = [
                _cosine_similarity(
                    self._recent_events[index].representative_world_tokens,
                    self._recent_events[index + 1].representative_world_tokens,
                )
                for index in range(candidate_count)
            ]
            # np.argmax/first max gives deterministic earliest-pair tie break.
            merge_index = int(np.argmax(np.asarray(similarities, dtype=np.float64)))
            merged = self._merge_events(
                self._recent_events[merge_index], self._recent_events[merge_index + 1]
            )
            self._recent_events[merge_index : merge_index + 2] = [merged]
            self._counters = EpisodeMemoryCounters(
                observation_updates=self._counters.observation_updates,
                event_writes=self._counters.event_writes,
                event_merges=self._counters.event_merges + 1,
            )

    def _update_status(
        self,
        summary: ActionSummary,
        *,
        frame_index: int,
        world_change: float,
    ) -> EpisodeEventStatus:
        previous = self._event_status
        close_count = previous.close_count + summary.close_count
        release_count = previous.release_count + summary.release_count
        last_close = frame_index if summary.close_count > 0 else previous.last_close_frame
        last_release = frame_index if summary.release_count > 0 else previous.last_release_frame
        terminal_closed = bool(
            summary.terminal_gripper_values.size
            and self._is_closed(summary.terminal_gripper_values).any()
        )
        active_close = last_close is not None and (
            last_release is None or last_close > last_release
        )
        motion_after_close = previous.visual_motion_after_close
        if summary.close_count > 0:
            motion_after_close = False
        if active_close and world_change >= self._config.motion_world_change_threshold:
            motion_after_close = True
        release_happened = release_count > 0
        release_is_latest = last_release is not None and (
            last_close is None or last_release >= last_close
        )
        stationary_after_release = bool(
            release_is_latest
            and world_change <= self._config.stationary_world_change_threshold
        )
        repeated_attempt_count = (
            previous.repeated_attempt_count + 1 if summary.repeated else 0
        )
        return EpisodeEventStatus(
            close_count=close_count,
            release_count=release_count,
            last_close_frame=last_close,
            last_release_frame=last_release,
            gripper_closed_recently=terminal_closed,
            visual_motion_after_close=motion_after_close,
            release_happened=release_happened,
            stationary_after_release=stationary_after_release,
            repeated_similar_action=summary.repeated,
            repeated_attempt_count=repeated_attempt_count,
        )

    def record_observation(
        self,
        capability: EpisodeWriteCapability,
        *,
        observation: FactualObservation,
        executed_actions: Any,
    ) -> EpisodeMemoryUpdate:
        """Record a real post-action observation and its actually executed chunk.

        Action summaries are retained for every chunk.  A recent event is
        written only when the robust adaptive change score is strictly greater
        than the threshold computed from *earlier* chunks.
        """

        with self._lock:
            self._validate_capability(capability)
            self._validate_observation(observation)
            actions = _readonly_float32(
                executed_actions,
                "executed_actions",
                ndim=2,
            )
            if actions.shape[1] != self._config.action_dim:
                raise EpisodeMemoryValidationError(
                    f"executed_actions must have shape [T, {self._config.action_dim}], "
                    f"got {actions.shape}"
                )
            previous = self._latest_observation
            assert previous is not None
            summary = self._make_action_summary(
                actions,
                start_frame=previous.frame_index,
                end_frame=observation.frame_index,
            )
            gripper_component = float(
                min(1, summary.close_count + summary.release_count)
            )
            action_component = self._action_direction_change(summary)
            world_component = _relative_change(
                previous.world_tokens, observation.world_tokens
            )
            vae_component = _relative_change(
                previous.vae_latent, observation.vae_latent
            )
            components = _readonly_float32(
                [gripper_component, action_component, world_component, vae_component],
                "change_components",
                ndim=1,
                shape=(4,),
            )
            weights = np.asarray(self._config.change_weights, dtype=np.float64)
            weights /= weights.sum()
            score = float(np.dot(components.astype(np.float64), weights))
            threshold = self._adaptive_threshold()
            event_written = score > threshold

            # Commit only after every input and derived value has validated.
            self._latest_observation = observation
            self._action_summaries.append(summary)
            if len(self._action_summaries) > self._config.max_action_summaries:
                del self._action_summaries[0]
            self._change_history.append(score)
            if len(self._change_history) > self._config.change_history_size:
                del self._change_history[0]
            self._event_status = self._update_status(
                summary,
                frame_index=observation.frame_index,
                world_change=world_component,
            )
            self._counters = EpisodeMemoryCounters(
                observation_updates=self._counters.observation_updates + 1,
                event_writes=self._counters.event_writes + int(event_written),
                event_merges=self._counters.event_merges,
            )
            latest_event: EpisodeEvent | None = None
            if event_written:
                latest_event = EpisodeEvent(
                    start_frame=previous.frame_index,
                    end_frame=observation.frame_index,
                    start_observation_id=previous.observation_id,
                    end_observation_id=observation.observation_id,
                    pre_world_tokens=previous.world_tokens,
                    post_world_tokens=observation.world_tokens,
                    representative_world_tokens=observation.world_tokens,
                    world_delta=np.asarray(
                        observation.world_tokens - previous.world_tokens,
                        dtype=np.float32,
                    ),
                    vae_delta=np.asarray(
                        observation.vae_latent - previous.vae_latent,
                        dtype=np.float32,
                    ),
                    change_components=components,
                    change_score=score,
                    write_threshold=threshold,
                    action_summary=summary,
                )
                self._recent_events.append(latest_event)
                self._compress_if_needed()
                # Compression never touches the last slot.
                latest_event = self._recent_events[-1]
            return EpisodeMemoryUpdate(
                observation_id=observation.observation_id,
                action_summary=summary,
                change_components=components,
                change_score=score,
                write_threshold=threshold,
                event_written=event_written,
                latest_event=latest_event,
                event_status=self._event_status,
                counters=self._counters,
            )

    def update(
        self,
        capability: EpisodeWriteCapability,
        *,
        observation: FactualObservation,
        executed_actions: Any,
    ) -> EpisodeMemoryUpdate:
        """Evaluator-friendly alias for :meth:`record_observation`."""

        return self.record_observation(
            capability,
            observation=observation,
            executed_actions=executed_actions,
        )

    def snapshot(self) -> EpisodeMemorySnapshot:
        """Return an immutable, deterministic view of the current state."""

        with self._lock:
            return EpisodeMemorySnapshot(
                config=self._config,
                phase=self._phase,
                episode_id=self._episode_id,
                initial_anchor=self._initial_anchor,
                latest_observation=self._latest_observation,
                recent_events=tuple(self._recent_events),
                executed_action_summaries=tuple(self._action_summaries),
                event_status=self._event_status,
                change_score_history=tuple(self._change_history),
                counters=self._counters,
            )

    def end_episode(
        self, capability: EpisodeWriteCapability
    ) -> EpisodeMemorySnapshot:
        """Seal the active episode, invalidate its token, and return its snapshot."""

        with self._lock:
            self._validate_capability(capability)
            self._phase = EpisodeMemoryPhase.ENDED
            self._generation += 1
            return self.snapshot()

    def to_json(self) -> str:
        return self.snapshot().to_json()

    @classmethod
    def from_json(
        cls, payload: str | bytes | bytearray
    ) -> tuple["EpisodeWorkingMemory", EpisodeWriteCapability | None]:
        """Restore a snapshot and mint a fresh token only if it was active."""

        if isinstance(payload, (bytes, bytearray)):
            try:
                payload = bytes(payload).decode("utf-8")
            except UnicodeDecodeError as exc:
                raise EpisodeMemoryValidationError("snapshot bytes must be UTF-8") from exc
        if not isinstance(payload, str):
            raise EpisodeMemoryValidationError("snapshot payload must be str or UTF-8 bytes")
        try:
            value = json.loads(payload)
        except (json.JSONDecodeError, ValueError) as exc:
            raise EpisodeMemoryValidationError("snapshot is not valid JSON") from exc
        if not isinstance(value, Mapping):
            raise EpisodeMemoryValidationError("snapshot root must be an object")
        _require_exact_keys(value, {"schema", "version", "config", "state"}, "snapshot")
        if value["schema"] != EPISODE_MEMORY_SCHEMA:
            raise EpisodeMemoryValidationError("snapshot schema is not supported")
        if value["version"] != EPISODE_MEMORY_VERSION:
            raise EpisodeMemoryValidationError("snapshot version is not supported")
        config = EpisodeMemoryConfig.from_dict(value["config"])
        state = value["state"]
        if not isinstance(state, Mapping):
            raise EpisodeMemoryValidationError("snapshot.state must be an object")
        state_fields = {
            "phase",
            "episode_id",
            "initial_anchor",
            "latest_observation",
            "recent_events",
            "executed_action_summaries",
            "event_status",
            "change_score_history",
            "counters",
        }
        _require_exact_keys(state, state_fields, "snapshot.state")
        try:
            phase = EpisodeMemoryPhase(state["phase"])
        except (ValueError, TypeError) as exc:
            raise EpisodeMemoryValidationError("snapshot.state.phase is invalid") from exc
        initial = (
            None
            if state["initial_anchor"] is None
            else FactualObservation.from_dict(state["initial_anchor"])
        )
        latest = (
            None
            if state["latest_observation"] is None
            else FactualObservation.from_dict(state["latest_observation"])
        )
        if not isinstance(state["recent_events"], list):
            raise EpisodeMemoryValidationError("snapshot.state.recent_events must be a list")
        if not isinstance(state["executed_action_summaries"], list):
            raise EpisodeMemoryValidationError(
                "snapshot.state.executed_action_summaries must be a list"
            )
        if not isinstance(state["change_score_history"], list):
            raise EpisodeMemoryValidationError(
                "snapshot.state.change_score_history must be a list"
            )
        snapshot = EpisodeMemorySnapshot(
            config=config,
            phase=phase,
            episode_id=state["episode_id"],
            initial_anchor=initial,
            latest_observation=latest,
            recent_events=tuple(
                EpisodeEvent.from_dict(event) for event in state["recent_events"]
            ),
            executed_action_summaries=tuple(
                ActionSummary.from_dict(summary)
                for summary in state["executed_action_summaries"]
            ),
            event_status=EpisodeEventStatus.from_dict(state["event_status"]),
            change_score_history=tuple(state["change_score_history"]),
            counters=EpisodeMemoryCounters.from_dict(state["counters"]),
        )
        memory = cls(config)
        with memory._lock:
            memory._phase = snapshot.phase
            memory._episode_id = snapshot.episode_id
            memory._initial_anchor = snapshot.initial_anchor
            memory._latest_observation = snapshot.latest_observation
            memory._recent_events = list(snapshot.recent_events)
            memory._action_summaries = list(snapshot.executed_action_summaries)
            memory._event_status = snapshot.event_status
            memory._change_history = list(snapshot.change_score_history)
            memory._counters = snapshot.counters
            memory._generation += 1
            capability = None
            if snapshot.phase is EpisodeMemoryPhase.ACTIVE:
                assert snapshot.episode_id is not None
                capability = EpisodeWriteCapability(
                    episode_id=snapshot.episode_id,
                    _generation=memory._generation,
                    _owner=memory._owner,
                )
        # Re-serialization is an inexpensive full invariant and canonical-form
        # check, including all nested arrays and factual provenance markers.
        if memory.to_json() != snapshot.to_json():
            raise EpisodeMemoryValidationError("snapshot did not restore canonically")
        return memory, capability


__all__ = [
    "EPISODE_MEMORY_SCHEMA",
    "EPISODE_MEMORY_VERSION",
    "FACTUAL_OBSERVATION_PROVENANCE",
    "ActionSummary",
    "EpisodeEvent",
    "EpisodeEventStatus",
    "EpisodeMemoryCapabilityError",
    "EpisodeMemoryConfig",
    "EpisodeMemoryCounters",
    "EpisodeMemoryError",
    "EpisodeMemoryLifecycleError",
    "EpisodeMemoryPhase",
    "EpisodeMemorySnapshot",
    "EpisodeMemoryUpdate",
    "EpisodeMemoryValidationError",
    "EpisodeWorkingMemory",
    "EpisodeWriteCapability",
    "FactualObservation",
]
