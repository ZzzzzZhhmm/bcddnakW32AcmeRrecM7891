"""Semantic validation for the canonical WARM v1 event-bank payload."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .event_bank import EventBank
from .payload_names import (
    CONTAINS_FORCED_GRIPPER,
    EFFECT_POST,
    EFFECT_PRE,
    EVENT_SCORE,
    FEATURE_EPISODE_SHA256,
    MODEL_SPACE_ACTION,
    OBSERVED_GRIPPER_STATE,
    SOURCE_EPISODE_SHA256,
    START_PROPRIO,
    TASK_INDEX,
)


class WarmBankContractError(ValueError):
    """Raised when an EventBank is structurally valid but not WARM-compatible."""


@dataclass(frozen=True, slots=True)
class WarmBankSummary:
    num_events: int
    num_source_episodes: int
    action_horizon: int
    action_dim: int
    context_dim: int
    effect_shape: tuple[int, ...]
    proprio_dim: int


def _require_payload(bank: EventBank, name: str) -> np.ndarray:
    try:
        return bank.payload(name)
    except KeyError as exc:
        raise WarmBankContractError(
            f"WARM v1 event bank is missing required payload {name!r}"
        ) from exc


def _expect(
    array: np.ndarray,
    *,
    name: str,
    dtype: np.dtype,
    shape: tuple[int | None, ...],
) -> None:
    if array.dtype != dtype:
        raise WarmBankContractError(f"{name} must have dtype {dtype}, got {array.dtype}")
    if array.ndim != len(shape):
        raise WarmBankContractError(
            f"{name} must have rank {len(shape)}, got shape {array.shape}"
        )
    for index, (actual, expected) in enumerate(zip(array.shape, shape, strict=True)):
        if expected is not None and actual != expected:
            raise WarmBankContractError(
                f"{name} dimension {index} must be {expected}, got {actual}"
            )


def validate_warm_v1_bank(
    bank: EventBank,
    *,
    expected_action_horizon: int | None = None,
    expected_action_dim: int | None = None,
    reject_duplicate_source_content: bool = True,
    reject_duplicate_feature_content: bool = True,
) -> WarmBankSummary:
    """Validate names, layouts, and episode-content leakage invariants.

    ``EventBank`` intentionally accepts arbitrary numeric payloads. This
    validator is therefore mandatory at every WARM production boundary.
    """

    if not isinstance(bank, EventBank):
        raise TypeError("bank must be EventBank")
    count = len(bank)
    action = _require_payload(bank, MODEL_SPACE_ACTION)
    if action.dtype != np.dtype(np.float32) or action.ndim != 3:
        raise WarmBankContractError(
            f"{MODEL_SPACE_ACTION} must be float32 [N,H,D], got {action.shape}/{action.dtype}"
        )
    if action.shape[0] != count or action.shape[1] <= 0 or action.shape[2] <= 0:
        raise WarmBankContractError(
            f"{MODEL_SPACE_ACTION} must have non-empty shape [N,H,D] with N={count}"
        )
    horizon, action_dim = int(action.shape[1]), int(action.shape[2])
    if expected_action_horizon is not None and horizon != expected_action_horizon:
        raise WarmBankContractError(
            f"action horizon mismatch: expected {expected_action_horizon}, got {horizon}"
        )
    if expected_action_dim is not None and action_dim != expected_action_dim:
        raise WarmBankContractError(
            f"action dimension mismatch: expected {expected_action_dim}, got {action_dim}"
        )

    pre = _require_payload(bank, EFFECT_PRE)
    post = _require_payload(bank, EFFECT_POST)
    if pre.dtype != np.dtype(np.float32) or post.dtype != np.dtype(np.float32):
        raise WarmBankContractError("effect_pre and effect_post must use float32")
    if pre.ndim < 2 or pre.shape != post.shape or pre.shape[0] != count:
        raise WarmBankContractError(
            "effect_pre and effect_post must have identical non-scalar [N,...] shapes"
        )

    proprio = _require_payload(bank, START_PROPRIO)
    _expect(
        proprio,
        name=START_PROPRIO,
        dtype=np.dtype(np.float32),
        shape=(count, None),
    )
    if proprio.shape[1] <= 0:
        raise WarmBankContractError("start_proprio must have a non-empty feature dimension")
    gripper = _require_payload(bank, OBSERVED_GRIPPER_STATE)
    _expect(
        gripper,
        name=OBSERVED_GRIPPER_STATE,
        dtype=np.dtype(np.float32),
        shape=(count, horizon + 1),
    )
    _expect(
        _require_payload(bank, TASK_INDEX),
        name=TASK_INDEX,
        dtype=np.dtype(np.int64),
        shape=(count,),
    )
    _expect(
        _require_payload(bank, EVENT_SCORE),
        name=EVENT_SCORE,
        dtype=np.dtype(np.float32),
        shape=(count,),
    )
    _expect(
        _require_payload(bank, CONTAINS_FORCED_GRIPPER),
        name=CONTAINS_FORCED_GRIPPER,
        dtype=np.dtype(np.bool_),
        shape=(count,),
    )
    source_hashes = _require_payload(bank, SOURCE_EPISODE_SHA256)
    _expect(
        source_hashes,
        name=SOURCE_EPISODE_SHA256,
        dtype=np.dtype(np.uint8),
        shape=(count, 32),
    )
    feature_hashes = _require_payload(bank, FEATURE_EPISODE_SHA256)
    _expect(
        feature_hashes,
        name=FEATURE_EPISODE_SHA256,
        dtype=np.dtype(np.uint8),
        shape=(count, 32),
    )

    episode_to_hash: dict[tuple[str, int, int], bytes] = {}
    hash_to_episode: dict[bytes, tuple[str, int, int]] = {}
    for event_id, digest_row in zip(bank.event_ids, source_hashes, strict=True):
        digest = bytes(memoryview(digest_row).cast("B"))
        existing = episode_to_hash.setdefault(event_id.episode_key, digest)
        if existing != digest:
            raise WarmBankContractError(
                f"source hash changes inside episode {event_id.episode_key!r}"
            )
        other_episode = hash_to_episode.setdefault(digest, event_id.episode_key)
        if reject_duplicate_source_content and other_episode != event_id.episode_key:
            raise WarmBankContractError(
                "duplicate source episode content appears under different identities: "
                f"{other_episode!r} and {event_id.episode_key!r}"
            )

    episode_to_feature_hash: dict[tuple[str, int, int], bytes] = {}
    feature_hash_to_episode: dict[bytes, tuple[str, int, int]] = {}
    for event_id, digest_row in zip(bank.event_ids, feature_hashes, strict=True):
        digest = bytes(memoryview(digest_row).cast("B"))
        existing = episode_to_feature_hash.setdefault(event_id.episode_key, digest)
        if existing != digest:
            raise WarmBankContractError(
                f"feature hash changes inside episode {event_id.episode_key!r}"
            )
        other_episode = feature_hash_to_episode.setdefault(
            digest, event_id.episode_key
        )
        if reject_duplicate_feature_content and other_episode != event_id.episode_key:
            raise WarmBankContractError(
                "duplicate feature episode content appears under different identities: "
                f"{other_episode!r} and {event_id.episode_key!r}"
            )

    return WarmBankSummary(
        num_events=count,
        num_source_episodes=len(episode_to_hash),
        action_horizon=horizon,
        action_dim=action_dim,
        context_dim=int(bank.context_keys.shape[1]),
        effect_shape=tuple(int(value) for value in pre.shape[1:]),
        proprio_dim=int(proprio.shape[1]),
    )


__all__ = [
    "WarmBankContractError",
    "WarmBankSummary",
    "validate_warm_v1_bank",
]
