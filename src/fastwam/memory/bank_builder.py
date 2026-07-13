"""Compose complete-episode features into an immutable WARM event bank."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .event_bank import EventBank
from .event_mining import EventMiningConfig, StartMode, mine_episode_events
from .schema import EventId


def _float32_time_array(name: str, value: np.ndarray, *, rank: int = 2) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.float32):
        raise TypeError(f"{name} must use float32, got {array.dtype}")
    if array.ndim != rank or any(dimension <= 0 for dimension in array.shape):
        raise ValueError(f"{name} must be a non-empty rank-{rank} array, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array)


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
    model_actions: np.ndarray
    proprio: np.ndarray
    gripper: np.ndarray
    context_keys: np.ndarray
    semantic_features: np.ndarray
    vae_features: np.ndarray | None = None

    def __post_init__(self) -> None:
        # EventId owns the canonical scalar validation contract.
        EventId(self.dataset_id, self.dataset_index, self.episode_index, 0)
        if isinstance(self.task_index, bool) or not isinstance(
            self.task_index, (int, np.integer)
        ):
            raise TypeError("task_index must be an integer")
        if int(self.task_index) < 0:
            raise ValueError("task_index must be non-negative")

        actions = _float32_time_array("model_actions", self.model_actions)
        proprio = _float32_time_array("proprio", self.proprio)
        context = _float32_time_array("context_keys", self.context_keys)
        semantics = _float32_time_array("semantic_features", self.semantic_features)
        gripper = np.asarray(self.gripper)
        if gripper.dtype != np.dtype(np.float32):
            raise TypeError(f"gripper must use float32, got {gripper.dtype}")
        if gripper.ndim == 2 and gripper.shape[1] == 1:
            gripper = gripper[:, 0]
        if gripper.ndim != 1 or gripper.size == 0 or not np.isfinite(gripper).all():
            raise ValueError("gripper must be a finite [T] or [T+1] float32 array")

        steps = actions.shape[0]
        for name, array in (
            ("proprio", proprio),
            ("context_keys", context),
            ("gripper", gripper),
        ):
            if array.shape[0] not in (steps, steps + 1):
                raise ValueError(f"{name} must have T or T+1 entries for T actions")
        if semantics.shape[0] != steps + 1:
            raise ValueError(
                "semantic_features must have T+1 factual states for T actions"
            )

        vae = None
        if self.vae_features is not None:
            vae = _float32_time_array("vae_features", self.vae_features)
            if vae.shape[0] not in (steps, steps + 1):
                raise ValueError("vae_features must have T or T+1 entries for T actions")

        object.__setattr__(self, "task_index", int(self.task_index))
        object.__setattr__(self, "model_actions", actions)
        object.__setattr__(self, "proprio", proprio)
        object.__setattr__(self, "gripper", np.ascontiguousarray(gripper))
        object.__setattr__(self, "context_keys", context)
        object.__setattr__(self, "semantic_features", semantics)
        object.__setattr__(self, "vae_features", vae)


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

    for episode in episode_list:
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
            gripper_sequence.append(episode.gripper[start:stop])
            task_indices.append(episode.task_index)
            event_scores.append(float(np.max(result.scores[start:stop])))
            contains_forced_gripper.append(
                any(index in forced_set for index in range(start, stop))
            )

    if not event_ids:
        raise ValueError(
            "Event selection produced an empty bank; use uniform/hybrid mode or audit signals"
        )

    return EventBank.from_arrays(
        event_ids,
        np.ascontiguousarray(np.stack(keys).astype(np.float32, copy=False)),
        model_action=np.ascontiguousarray(np.stack(action_chunks).astype(np.float32, copy=False)),
        effect_pre=np.ascontiguousarray(np.stack(effect_pre).astype(np.float32, copy=False)),
        effect_post=np.ascontiguousarray(np.stack(effect_post).astype(np.float32, copy=False)),
        start_proprio=np.ascontiguousarray(
            np.stack(start_proprio).astype(np.float32, copy=False)
        ),
        gripper_state=np.ascontiguousarray(
            np.stack(gripper_sequence).astype(np.float32, copy=False)
        ),
        task_index=np.ascontiguousarray(np.asarray(task_indices, dtype=np.int64)),
        event_score=np.ascontiguousarray(np.asarray(event_scores, dtype=np.float32)),
        contains_forced_gripper=np.ascontiguousarray(
            np.asarray(contains_forced_gripper, dtype=np.bool_)
        ),
    )


__all__ = ["EpisodeFeatures", "build_event_bank"]
