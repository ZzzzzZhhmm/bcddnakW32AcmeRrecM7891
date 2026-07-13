"""Compose complete-episode features into an immutable WARM event bank."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

import numpy as np

from .event_bank import EventBank
from .event_mining import EventMiningConfig, StartMode, mine_episode_events
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
from .schema import EventId


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _float32_time_array(
    name: str,
    value: np.ndarray,
    *,
    exact_rank: int | None = None,
    min_rank: int = 2,
) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.float32):
        raise TypeError(f"{name} must use float32, got {array.dtype}")
    if exact_rank is not None and array.ndim != exact_rank:
        raise ValueError(
            f"{name} must be a non-empty rank-{exact_rank} array, got {array.shape}"
        )
    if array.ndim < min_rank or any(dimension <= 0 for dimension in array.shape):
        raise ValueError(
            f"{name} must be non-empty with rank >= {min_rank}, got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    result = np.array(array, copy=True, order="C")
    result.flags.writeable = False
    return result


@dataclass(frozen=True, slots=True)
class EpisodeFeatures:
    """Fixed-encoder features and model-space controls for one full episode.

    ``model_actions`` must already use the exact FastWAM normalizer and action
    layout recorded in the eventual bank manifest.  ``semantic_features``
    contains ``T+1`` factual states so every action window has both pre and
    post observations; it is not a generated future.
    """

    dataset_id: str
    dataset_index: int
    episode_index: int
    task_index: int
    source_episode_sha256: str
    model_actions: np.ndarray
    proprio: np.ndarray
    gripper: np.ndarray
    context_keys: np.ndarray
    semantic_features: np.ndarray
    vae_features: np.ndarray | None = None
    feature_episode_sha256: str | None = None

    def __post_init__(self) -> None:
        # EventId owns the canonical scalar validation contract.
        identity = EventId(self.dataset_id, self.dataset_index, self.episode_index, 0)
        if isinstance(self.task_index, bool) or not isinstance(
            self.task_index, (int, np.integer)
        ):
            raise TypeError("task_index must be an integer")
        if int(self.task_index) < 0:
            raise ValueError("task_index must be non-negative")
        if (
            not isinstance(self.source_episode_sha256, str)
            or _SHA256.fullmatch(self.source_episode_sha256) is None
        ):
            raise ValueError(
                "source_episode_sha256 must be a lowercase 64-character SHA-256 digest"
            )
        if self.feature_episode_sha256 is not None and (
            not isinstance(self.feature_episode_sha256, str)
            or _SHA256.fullmatch(self.feature_episode_sha256) is None
        ):
            raise ValueError(
                "feature_episode_sha256 must be a lowercase 64-character "
                "SHA-256 digest or None"
            )

        actions = _float32_time_array("model_actions", self.model_actions, exact_rank=2)
        proprio = _float32_time_array("proprio", self.proprio, exact_rank=2)
        context = _float32_time_array("context_keys", self.context_keys, exact_rank=2)
        semantics = _float32_time_array("semantic_features", self.semantic_features)
        gripper = np.asarray(self.gripper)
        if gripper.dtype != np.dtype(np.float32):
            raise TypeError(f"gripper must use float32, got {gripper.dtype}")
        if gripper.ndim == 2 and gripper.shape[1] == 1:
            gripper = gripper[:, 0]
        if gripper.ndim != 1 or gripper.size == 0 or not np.isfinite(gripper).all():
            raise ValueError("gripper must be a finite [T+1] float32 array")

        steps = actions.shape[0]
        for name, array in (
            ("proprio", proprio),
            ("context_keys", context),
            ("gripper", gripper),
        ):
            if array.shape[0] != steps + 1:
                raise ValueError(f"{name} must have T+1 factual states for T actions")
        if semantics.shape[0] != steps + 1:
            raise ValueError(
                "semantic_features must have T+1 factual states for T actions"
            )

        vae = None
        if self.vae_features is not None:
            vae = _float32_time_array("vae_features", self.vae_features)
            if vae.shape[0] != steps + 1:
                raise ValueError("vae_features must have T+1 factual states for T actions")

        context_norms = np.linalg.norm(context.astype(np.float64), axis=1)
        if np.any(context_norms <= 0.0):
            bad_index = int(np.flatnonzero(context_norms <= 0.0)[0])
            raise ValueError(
                "context_keys must have non-zero cosine norm at every factual state; "
                f"first invalid frame={bad_index}"
            )

        readonly_gripper = np.array(gripper, copy=True, order="C")
        readonly_gripper.flags.writeable = False

        object.__setattr__(self, "dataset_id", identity.dataset_id)
        object.__setattr__(self, "dataset_index", identity.dataset_index)
        object.__setattr__(self, "episode_index", identity.episode_index)
        object.__setattr__(self, "task_index", int(self.task_index))
        object.__setattr__(self, "source_episode_sha256", self.source_episode_sha256)
        object.__setattr__(self, "model_actions", actions)
        object.__setattr__(self, "proprio", proprio)
        object.__setattr__(self, "gripper", readonly_gripper)
        object.__setattr__(self, "context_keys", context)
        object.__setattr__(self, "semantic_features", semantics)
        object.__setattr__(self, "vae_features", vae)
        object.__setattr__(self, "feature_episode_sha256", self.feature_episode_sha256)


def _assert_shared_shape(
    episodes: Sequence[EpisodeFeatures],
    attribute: str,
    trailing_shape: tuple[int, ...],
) -> None:
    for episode in episodes:
        actual = tuple(getattr(episode, attribute).shape[1:])
        if actual != trailing_shape:
            raise ValueError(
                f"All episodes must share {attribute} trailing shape; "
                f"expected {trailing_shape}, got {actual}"
            )


def build_event_bank(
    episodes: Sequence[EpisodeFeatures],
    *,
    mining_config: EventMiningConfig,
    start_mode: StartMode = "hybrid",
) -> EventBank:
    """Build one fixed-horizon bank without padding or action resampling."""

    if not episodes:
        raise ValueError("At least one episode is required")
    episode_list = tuple(episodes)
    action_shape = tuple(episode_list[0].model_actions.shape[1:])
    proprio_shape = tuple(episode_list[0].proprio.shape[1:])
    context_shape = tuple(episode_list[0].context_keys.shape[1:])
    semantic_shape = tuple(episode_list[0].semantic_features.shape[1:])
    for attribute, shape in (
        ("model_actions", action_shape),
        ("proprio", proprio_shape),
        ("context_keys", context_shape),
        ("semantic_features", semantic_shape),
    ):
        _assert_shared_shape(episode_list, attribute, shape)

    event_ids: list[EventId] = []
    keys: list[np.ndarray] = []
    action_chunks: list[np.ndarray] = []
    effect_pre: list[np.ndarray] = []
    effect_post: list[np.ndarray] = []
    start_proprio: list[np.ndarray] = []
    gripper_sequence: list[np.ndarray] = []
    task_indices: list[int] = []
    event_scores: list[float] = []
    contains_forced_gripper: list[bool] = []
    source_episode_hashes: list[np.ndarray] = []
    feature_episode_hashes: list[np.ndarray] = []

    for episode in episode_list:
        if episode.feature_episode_sha256 is None:
            raise ValueError(
                "feature_episode_sha256 is required before building an event bank; "
                "load episodes through the verified feature-cache collection"
            )
        result = mine_episode_events(
            episode.model_actions,
            episode.proprio,
            episode.gripper,
            config=mining_config,
            start_mode=start_mode,
            semantic_features=episode.semantic_features,
            vae_features=episode.vae_features,
        )
        forced_set = set(int(value) for value in result.forced_gripper_indices)
        for start, stop in result.windows.tolist():
            event_ids.append(
                EventId(
                    episode.dataset_id,
                    episode.dataset_index,
                    episode.episode_index,
                    int(start),
                )
            )
            keys.append(episode.context_keys[start])
            action_chunks.append(episode.model_actions[start:stop])
            effect_pre.append(episode.semantic_features[start])
            effect_post.append(episode.semantic_features[stop])
            start_proprio.append(episode.proprio[start])
            # This is observed gripper state metadata.  The executable gripper
            # command remains part of model_actions and is never interpolated.
            # Actions span ``[start, stop)`` while factual states span the
            # closed interval ``[start, stop]``.  Keep H+1 gripper states so
            # a close/release caused by the final action is not truncated.
            gripper_sequence.append(episode.gripper[start : stop + 1])
            task_indices.append(episode.task_index)
            event_scores.append(float(np.max(result.scores[start:stop])))
            contains_forced_gripper.append(
                any(index in forced_set for index in range(start, stop))
            )
            source_episode_hashes.append(
                np.frombuffer(bytes.fromhex(episode.source_episode_sha256), dtype=np.uint8)
            )
            feature_episode_hashes.append(
                np.frombuffer(bytes.fromhex(episode.feature_episode_sha256), dtype=np.uint8)
            )

    if not event_ids:
        raise ValueError(
            "Event selection produced an empty bank; use uniform/hybrid mode or audit signals"
        )

    return EventBank.from_arrays(
        event_ids,
        np.ascontiguousarray(np.stack(keys).astype(np.float32, copy=False)),
        **{
            MODEL_SPACE_ACTION: np.ascontiguousarray(
                np.stack(action_chunks).astype(np.float32, copy=False)
            ),
            EFFECT_PRE: np.ascontiguousarray(
                np.stack(effect_pre).astype(np.float32, copy=False)
            ),
            EFFECT_POST: np.ascontiguousarray(
                np.stack(effect_post).astype(np.float32, copy=False)
            ),
            START_PROPRIO: np.ascontiguousarray(
                np.stack(start_proprio).astype(np.float32, copy=False)
            ),
            OBSERVED_GRIPPER_STATE: np.ascontiguousarray(
                np.stack(gripper_sequence).astype(np.float32, copy=False)
            ),
            TASK_INDEX: np.ascontiguousarray(np.asarray(task_indices, dtype=np.int64)),
            EVENT_SCORE: np.ascontiguousarray(
                np.asarray(event_scores, dtype=np.float32)
            ),
            CONTAINS_FORCED_GRIPPER: np.ascontiguousarray(
                np.asarray(contains_forced_gripper, dtype=np.bool_)
            ),
            SOURCE_EPISODE_SHA256: np.ascontiguousarray(
                np.stack(source_episode_hashes)
            ),
            FEATURE_EPISODE_SHA256: np.ascontiguousarray(
                np.stack(feature_episode_hashes)
            ),
        },
    )


__all__ = ["EpisodeFeatures", "build_event_bank"]
