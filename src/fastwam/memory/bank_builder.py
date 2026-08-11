"""Compose complete-episode features into an immutable WARM event bank."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

import numpy as np

from .event_bank import EventBank
from .event_mining import EventMiningConfig, StartMode, mine_episode_events
from .payload_names import (
    ACTION_VALID_MASK,
    CONTAINS_FORCED_GRIPPER,
    EFFECT_POST,
    EFFECT_PRE,
    EVENT_ORDINAL,
    EVENT_SCORE,
    FEATURE_EPISODE_SHA256,
    MODEL_SPACE_ACTION,
    NORMALIZED_PHASE,
    OBSERVED_GRIPPER_STATE,
    SOURCE_EPISODE_SHA256,
    START_PROPRIO,
    SUCCESSOR_EVENT_START_FRAME,
    SUCCESSOR_ROW,
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


def _validate_temporal_bank_arrays(
    event_ids: Sequence[EventId],
    *,
    action: np.ndarray,
    effect_pre: np.ndarray,
    effect_post: np.ndarray,
    observed_gripper: np.ndarray,
    normalized_phase: np.ndarray,
    event_ordinal: np.ndarray,
    successor_bank_row: np.ndarray,
    successor_start_frame: np.ndarray,
    action_valid_mask: np.ndarray,
) -> None:
    """Fail closed if one state-action-effect row loses temporal identity."""

    count = len(event_ids)
    if action.ndim != 3 or action.shape[0] != count:
        raise AssertionError("action payload must be [events,horizon,action_dim]")
    horizon = int(action.shape[1])
    if effect_pre.shape != effect_post.shape or effect_pre.shape[0] != count:
        raise AssertionError("pre/post effect payloads must be row aligned")
    if observed_gripper.shape != (count, horizon + 1):
        raise AssertionError("gripper timing must cover both ends of every action")
    if action_valid_mask.shape != (count, horizon) or not action_valid_mask.all():
        raise AssertionError("factual fixed-horizon actions must be fully valid")
    if normalized_phase.shape != (count,) or (
        np.any(normalized_phase < 0.0) or np.any(normalized_phase > 1.0)
    ):
        raise AssertionError("normalized event phases must lie in [0,1]")
    for name, value in (
        (EVENT_ORDINAL, event_ordinal),
        (SUCCESSOR_ROW, successor_bank_row),
        (SUCCESSOR_EVENT_START_FRAME, successor_start_frame),
    ):
        if value.dtype != np.dtype(np.int64) or value.shape != (count,):
            raise AssertionError(f"{name} must be int64 [events]")

    for row, event_id in enumerate(event_ids):
        ordinal = int(event_ordinal[row])
        if ordinal < 0:
            raise AssertionError("event ordinals must be non-negative")
        successor = int(successor_bank_row[row])
        successor_start = int(successor_start_frame[row])
        if successor < 0:
            if successor != -1 or successor_start != -1:
                raise AssertionError("terminal successors must use the -1 sentinel")
            continue
        if successor >= count:
            raise AssertionError("successor bank row is out of bounds")
        successor_id = event_ids[successor]
        if successor_id.episode_key != event_id.episode_key:
            raise AssertionError("successor must remain inside the factual episode")
        if int(event_ordinal[successor]) != ordinal + 1:
            raise AssertionError("successor must be the next event ordinal")
        if successor_id.start_frame <= event_id.start_frame:
            raise AssertionError("successor must advance factual episode time")
        if successor_id.start_frame != successor_start:
            raise AssertionError("successor row and successor event id disagree")


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
    normalized_phases: list[float] = []
    event_ordinals: list[int] = []
    successor_bank_rows: list[int] = []
    successor_start_frames: list[int] = []
    action_valid_masks: list[np.ndarray] = []
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
        episode_bank_row = len(event_ids)
        for local_index, (start, stop) in enumerate(result.windows.tolist()):
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
            normalized_phases.append(float(result.normalized_phases[local_index]))
            event_ordinals.append(int(result.event_ordinals[local_index]))
            successor_local = int(result.successor_window_indices[local_index])
            successor_bank_rows.append(
                -1 if successor_local < 0 else episode_bank_row + successor_local
            )
            successor_start_frames.append(
                int(result.successor_start_frames[local_index])
            )
            # Event windows are never padded or resampled.  Retaining the
            # explicit validity mask makes this invariant machine-checkable
            # and prevents a future online adapter from silently treating a
            # shifted/padded suffix as a factual bank event.
            action_valid_masks.append(
                np.ones((mining_config.action_horizon,), dtype=np.bool_)
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

    arrays = {
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
            NORMALIZED_PHASE: np.ascontiguousarray(
                np.asarray(normalized_phases, dtype=np.float32)
            ),
            EVENT_ORDINAL: np.ascontiguousarray(
                np.asarray(event_ordinals, dtype=np.int64)
            ),
            SUCCESSOR_ROW: np.ascontiguousarray(
                np.asarray(successor_bank_rows, dtype=np.int64)
            ),
            SUCCESSOR_EVENT_START_FRAME: np.ascontiguousarray(
                np.asarray(successor_start_frames, dtype=np.int64)
            ),
            ACTION_VALID_MASK: np.ascontiguousarray(
                np.stack(action_valid_masks).astype(np.bool_, copy=False)
            ),
            SOURCE_EPISODE_SHA256: np.ascontiguousarray(
                np.stack(source_episode_hashes)
            ),
            FEATURE_EPISODE_SHA256: np.ascontiguousarray(
                np.stack(feature_episode_hashes)
            ),
    }
    _validate_temporal_bank_arrays(
        event_ids,
        action=arrays[MODEL_SPACE_ACTION],
        effect_pre=arrays[EFFECT_PRE],
        effect_post=arrays[EFFECT_POST],
        observed_gripper=arrays[OBSERVED_GRIPPER_STATE],
        normalized_phase=arrays[NORMALIZED_PHASE],
        event_ordinal=arrays[EVENT_ORDINAL],
        successor_bank_row=arrays[SUCCESSOR_ROW],
        successor_start_frame=arrays[SUCCESSOR_EVENT_START_FRAME],
        action_valid_mask=arrays[ACTION_VALID_MASK],
    )

    return EventBank.from_arrays(
        event_ids,
        np.ascontiguousarray(np.stack(keys).astype(np.float32, copy=False)),
        **arrays,
    )


__all__ = [
    "ACTION_VALID_MASK",
    "EVENT_ORDINAL",
    "NORMALIZED_PHASE",
    "SUCCESSOR_EVENT_START_FRAME",
    "SUCCESSOR_ROW",
    "EpisodeFeatures",
    "build_event_bank",
]
