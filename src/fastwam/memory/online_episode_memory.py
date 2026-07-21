"""Evaluator bridge for factual WARM episode working memory.

The learned policy owns no mutable rollout history.  Instead, the evaluator
records only features returned by ``WarmRetrospectionFastWAM`` for the current
*real* environment observation and the exact action prefix that was actually
sent to the simulator.  At the next replan the resulting bounded history is
provided as semantic tokens.

The one-step ordering is intentional: the current observation is already an
input to the Video DiT and must not also be presented as historical evidence.
An observation returned by replan ``r`` becomes eligible history only for
replan ``r + 1``.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Mapping

import numpy as np

from .episode_memory import (
    EpisodeMemoryConfig,
    EpisodeMemoryLifecycleError,
    EpisodeMemoryPhase,
    EpisodeMemorySnapshot,
    EpisodeWorkingMemory,
    EpisodeWriteCapability,
    FactualObservation,
    action_summary_signature,
)


class OnlineEpisodeMemoryError(ValueError):
    """Raised when evaluator/model factual-memory boundaries are violated."""


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Integral, np.integer)
    ):
        raise TypeError(f"{name} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _readonly(value: Any, *, dtype: np.dtype[Any], name: str, rank: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != rank or raw.dtype.kind not in "fiub":
        raise OnlineEpisodeMemoryError(
            f"{name} must be a numeric rank-{rank} array, got {raw.shape}/{raw.dtype}"
        )
    array = np.ascontiguousarray(raw, dtype=dtype)
    if not array.size or any(int(size) <= 0 for size in array.shape):
        raise OnlineEpisodeMemoryError(f"{name} must have non-empty dimensions")
    if array.dtype.kind == "f" and not np.isfinite(array).all():
        raise OnlineEpisodeMemoryError(f"{name} must contain finite values")
    frozen = np.frombuffer(array.tobytes(order="C"), dtype=dtype)
    return frozen.reshape(array.shape)


@dataclass(frozen=True, slots=True)
class EpisodeHistoryInputs:
    """Immutable semantic history admitted to one model replan."""

    episode_tokens: np.ndarray
    episode_mask: np.ndarray
    episode_action_summaries: np.ndarray
    episode_action_mask: np.ndarray
    snapshot_sha256: str
    observation_count: int
    event_count: int
    action_summary_count: int

    def __post_init__(self) -> None:
        tokens = _readonly(
            self.episode_tokens,
            dtype=np.dtype(np.float32),
            name="episode_tokens",
            rank=2,
        )
        mask = _readonly(
            self.episode_mask,
            dtype=np.dtype(np.bool_),
            name="episode_mask",
            rank=1,
        )
        if tokens.shape[0] != mask.shape[0]:
            raise OnlineEpisodeMemoryError(
                "episode_tokens and episode_mask must share their token dimension"
            )
        if not bool(mask.all()):
            raise OnlineEpisodeMemoryError(
                "online history is compact and cannot contain padded invalid tokens"
            )
        action_summaries = np.ascontiguousarray(
            np.asarray(self.episode_action_summaries), dtype=np.float32
        )
        action_mask = np.ascontiguousarray(
            np.asarray(self.episode_action_mask), dtype=np.bool_
        )
        if action_summaries.ndim != 2 or action_mask.ndim != 1:
            raise OnlineEpisodeMemoryError(
                "episode action summaries/mask must have ranks 2/1"
            )
        if action_summaries.shape[0] != action_mask.shape[0]:
            raise OnlineEpisodeMemoryError(
                "episode action summaries and mask must share their row dimension"
            )
        if action_summaries.shape[1] <= 0 or not np.isfinite(action_summaries).all():
            raise OnlineEpisodeMemoryError(
                "episode action summaries need a finite non-empty feature dimension"
            )
        if not bool(action_mask.all()):
            raise OnlineEpisodeMemoryError(
                "online action history is compact and cannot contain padded rows"
            )
        action_summaries = np.frombuffer(
            action_summaries.tobytes(order="C"), dtype=np.float32
        ).reshape(action_summaries.shape)
        action_mask = np.frombuffer(
            action_mask.tobytes(order="C"), dtype=np.bool_
        ).reshape(action_mask.shape)
        if (
            not isinstance(self.snapshot_sha256, str)
            or len(self.snapshot_sha256) != 64
        ):
            raise OnlineEpisodeMemoryError("snapshot_sha256 must be a SHA-256 digest")
        object.__setattr__(self, "episode_tokens", tokens)
        object.__setattr__(self, "episode_mask", mask)
        object.__setattr__(self, "episode_action_summaries", action_summaries)
        object.__setattr__(self, "episode_action_mask", action_mask)
        for field in (
            "observation_count",
            "event_count",
            "action_summary_count",
        ):
            object.__setattr__(self, field, _nonnegative_int(getattr(self, field), field))

    def model_kwargs(self) -> dict[str, np.ndarray]:
        # PyTorch refuses to promise safe semantics for tensors backed by
        # read-only NumPy storage.  Hand the model private writable copies while
        # keeping this evidence object itself immutable.
        return {
            "episode_tokens": np.array(self.episode_tokens, copy=True, order="C"),
            "episode_mask": np.array(self.episode_mask, copy=True, order="C"),
            "episode_action_summaries": np.array(
                self.episode_action_summaries, copy=True, order="C"
            ),
            "episode_action_mask": np.array(
                self.episode_action_mask, copy=True, order="C"
            ),
        }

    def evidence(self) -> dict[str, Any]:
        return {
            "snapshot_sha256": self.snapshot_sha256,
            "token_count": int(self.episode_tokens.shape[0]),
            "semantic_dim": int(self.episode_tokens.shape[1]),
            "observation_count": self.observation_count,
            "event_count": self.event_count,
            "action_summary_count": self.action_summary_count,
            "action_summary_width": int(self.episode_action_summaries.shape[1]),
        }


class OnlineRetrospectiveEpisodeMemory:
    """Capability-safe adapter between a rollout evaluator and WARM memory."""

    def __init__(
        self,
        *,
        action_dim: int,
        action_horizon: int,
        semantic_dim: int,
        gripper_indices: tuple[int, ...] = (),
        recent_event_capacity: int = 6,
        episode_namespace: str = "libero-eval",
    ) -> None:
        self._semantic_dim = _nonnegative_int(semantic_dim, "semantic_dim")
        if self._semantic_dim == 0:
            raise ValueError("semantic_dim must be positive")
        self._action_horizon = _nonnegative_int(action_horizon, "action_horizon")
        if self._action_horizon == 0:
            raise ValueError("action_horizon must be positive")
        if (
            not isinstance(episode_namespace, str)
            or not episode_namespace
            or episode_namespace.strip() != episode_namespace
            or "\x00" in episode_namespace
            or ":" in episode_namespace
        ):
            raise ValueError(
                "episode_namespace must be normalized, non-empty, and contain no ':'"
            )
        self._episode_namespace = episode_namespace
        self._memory = EpisodeWorkingMemory(
            EpisodeMemoryConfig(
                action_dim=action_dim,
                gripper_indices=gripper_indices,
                max_recent_events=recent_event_capacity,
            )
        )
        self._episode_index: int | None = None
        self._episode_id: str | None = None
        self._capability: EpisodeWriteCapability | None = None
        self._last_recorded_frame: int | None = None

    @property
    def active(self) -> bool:
        return self._episode_index is not None

    def begin_episode(self, episode_index: int) -> None:
        """Reset state before the first factual observation of an episode."""

        index = _nonnegative_int(episode_index, "episode_index")
        self._memory.reset()
        self._episode_index = index
        self._episode_id = f"{self._episode_namespace}:{index}"
        self._capability = None
        self._last_recorded_frame = None

    def _require_active(self) -> None:
        if self._episode_index is None or self._episode_id is None:
            raise EpisodeMemoryLifecycleError(
                "begin_episode() is required before using online episode memory"
            )

    def history_inputs(
        self,
        *,
        executed_actions_since_previous: Any | None = None,
    ) -> EpisodeHistoryInputs | None:
        """Return history strictly preceding the current model replan."""

        self._require_active()
        if self._capability is None:
            if executed_actions_since_previous is not None and np.asarray(
                executed_actions_since_previous
            ).size:
                raise OnlineEpisodeMemoryError(
                    "initial episode history cannot contain policy actions"
                )
            return None
        snapshot = self._memory.snapshot()
        if snapshot.phase is not EpisodeMemoryPhase.ACTIVE:
            raise EpisodeMemoryLifecycleError("episode memory is not active")
        preview = None
        if executed_actions_since_previous is not None:
            preview = np.asarray(executed_actions_since_previous, dtype=np.float32)
            if (
                preview.ndim != 2
                or preview.shape[0] <= 0
                or preview.shape[1] != snapshot.config.action_dim
                or not np.isfinite(preview).all()
            ):
                raise OnlineEpisodeMemoryError(
                    "pending executed action prefix must be finite [T,action_dim]"
                )
            preview = np.ascontiguousarray(preview)
        return self._snapshot_inputs(snapshot, preview_actions=preview)

    def _summary_vector(self, summary: Any, config: EpisodeMemoryConfig) -> np.ndarray:
        action_dim = config.action_dim
        vector = np.zeros((3 * action_dim + 4,), dtype=np.float32)
        vector[:action_dim] = summary.mean_displacement
        vector[action_dim : 2 * action_dim] = summary.final_displacement
        terminal = vector[2 * action_dim : 3 * action_dim]
        for terminal_index, action_index in enumerate(config.gripper_indices):
            terminal[action_index] = summary.terminal_gripper_values[
                terminal_index
            ]
        vector[3 * action_dim :] = np.asarray(
            (
                min(1.0, float(summary.step_count) / self._action_horizon),
                summary.curvature,
                summary.repetition_similarity,
                float(summary.repeated),
            ),
            dtype=np.float32,
        )
        return vector

    def _preview_summary_vector(
        self,
        actions: np.ndarray,
        snapshot: EpisodeMemorySnapshot,
    ) -> np.ndarray:
        config = snapshot.config
        action_dim = config.action_dim
        movement = config.movement_indices
        mean = np.zeros((action_dim,), dtype=np.float32)
        displacement = np.zeros((action_dim,), dtype=np.float32)
        if movement:
            mean[list(movement)] = actions[:, movement].mean(
                axis=0, dtype=np.float32
            )
            displacement[list(movement)] = actions[:, movement].sum(
                axis=0, dtype=np.float32
            )
        terminal = np.zeros((action_dim,), dtype=np.float32)
        if config.gripper_indices:
            terminal[list(config.gripper_indices)] = actions[
                -1, list(config.gripper_indices)
            ]
        bends: list[float] = []
        if movement and actions.shape[0] > 1:
            vectors = actions[:, movement].astype(np.float64, copy=False)
            for previous, current in zip(vectors[:-1], vectors[1:], strict=True):
                denominator = float(
                    np.linalg.norm(previous) * np.linalg.norm(current)
                )
                if denominator > 1.0e-8:
                    cosine = float(
                        np.clip(np.dot(previous, current) / denominator, -1.0, 1.0)
                    )
                    bends.append(0.5 * (1.0 - cosine))
        curvature = 0.0 if not bends else float(np.mean(bends))
        # ``terminal`` above is expanded to ``action_dim`` for the learned
        # feature vector. Repetition uses the canonical compact signature,
        # whose terminal block contains only gripper channels. Committed
        # ActionSummary instances use the same representation.
        terminal_gripper_values = (
            actions[-1, list(config.gripper_indices)]
            if config.gripper_indices
            else np.empty((0,), dtype=np.float32)
        )
        signature = action_summary_signature(
            mean,
            displacement,
            terminal_gripper_values,
        )
        best_similarity = 0.0
        best_distance = np.inf
        if snapshot.executed_action_summaries:
            best_similarity = -1.0
            for previous in snapshot.executed_action_summaries:
                prior = previous.signature().astype(np.float64, copy=False)
                denominator = float(np.linalg.norm(signature) * np.linalg.norm(prior))
                if denominator <= 1.0e-8:
                    similarity = 1.0 if not np.any(signature) and not np.any(prior) else 0.0
                else:
                    similarity = float(
                        np.clip(np.dot(signature, prior) / denominator, -1.0, 1.0)
                    )
                distance_denominator = float(np.linalg.norm(signature)) + float(
                    np.linalg.norm(prior)
                )
                distance = (
                    0.0
                    if distance_denominator <= 1.0e-8
                    else float(
                        np.linalg.norm(signature - prior)
                        / (distance_denominator + 1.0e-8)
                    )
                )
                if similarity > best_similarity or (
                    np.isclose(similarity, best_similarity) and distance < best_distance
                ):
                    best_similarity = similarity
                    best_distance = distance
        repeated = bool(
            best_similarity >= config.repetition_cosine_threshold
            and best_distance <= config.repetition_distance_threshold
        )
        return np.ascontiguousarray(
            np.concatenate(
                (
                    mean,
                    displacement,
                    terminal,
                    np.asarray(
                        (
                            min(1.0, float(actions.shape[0]) / self._action_horizon),
                            curvature,
                            best_similarity,
                            float(repeated),
                        ),
                        dtype=np.float32,
                    ),
                )
            ),
            dtype=np.float32,
        )

    def _snapshot_inputs(
        self,
        snapshot: EpisodeMemorySnapshot,
        *,
        preview_actions: np.ndarray | None,
    ) -> EpisodeHistoryInputs:
        assert snapshot.initial_anchor is not None
        assert snapshot.latest_observation is not None

        initial = snapshot.initial_anchor
        # Every slot is a factual observation.  Keep the immutable initial
        # anchor, then bounded event endpoints, and protect the latest factual
        # observation even when its change score did not trigger an event.
        recent: list[tuple[int, np.ndarray]] = [
            (event.end_frame, event.post_world_tokens)
            for event in snapshot.recent_events
        ]
        latest_frame = snapshot.latest_observation.frame_index
        if latest_frame != initial.frame_index and not any(
            frame == latest_frame for frame, _ in recent
        ):
            recent.append((latest_frame, snapshot.latest_observation.world_tokens))
        recent.sort(key=lambda item: item[0])
        recent = recent[-snapshot.config.max_recent_events :]

        blocks = [initial.world_tokens, *(tokens for _, tokens in recent)]
        for index, block in enumerate(blocks):
            if block.ndim != 2 or int(block.shape[1]) != self._semantic_dim:
                raise OnlineEpisodeMemoryError(
                    f"history block {index} must be [N,{self._semantic_dim}], "
                    f"got {block.shape}"
                )
        tokens = np.ascontiguousarray(np.concatenate(blocks, axis=0), dtype=np.float32)
        mask = np.ones((tokens.shape[0],), dtype=np.bool_)
        vectors = [
            self._summary_vector(summary, snapshot.config)
            for summary in snapshot.executed_action_summaries
        ]
        if preview_actions is not None:
            vectors.append(self._preview_summary_vector(preview_actions, snapshot))
        vectors = vectors[-snapshot.config.max_action_summaries :]
        action_width = 3 * snapshot.config.action_dim + 4
        action_summaries = (
            np.stack(vectors, axis=0).astype(np.float32, copy=False)
            if vectors
            else np.zeros((0, action_width), dtype=np.float32)
        )
        action_mask = np.ones(
            (action_summaries.shape[0],), dtype=np.bool_
        )
        return EpisodeHistoryInputs(
            episode_tokens=tokens,
            episode_mask=mask,
            episode_action_summaries=action_summaries,
            episode_action_mask=action_mask,
            snapshot_sha256=snapshot.sha256,
            observation_count=1 + snapshot.counters.observation_updates,
            event_count=len(snapshot.recent_events),
            action_summary_count=len(vectors),
        )

    def record_factual_observation(
        self,
        *,
        frame_index: int,
        factual_payload: Mapping[str, Any],
        executed_actions_since_previous: Any | None,
        include_snapshot_sha256: bool = True,
    ) -> dict[str, Any]:
        """Commit one model-certified *current real observation* after inference.

        ``executed_actions_since_previous`` must be exactly the prefix sent to
        the environment since the preceding replan.  On the initial
        observation it must be empty because no policy action precedes it.

        ``include_snapshot_sha256=False`` is reserved for deterministic
        offline replay that consumes only update/event counters. It avoids
        serializing large factual VAE arrays after every frame while leaving
        the online/audited default unchanged.
        """

        self._require_active()
        if not isinstance(include_snapshot_sha256, bool):
            raise TypeError("include_snapshot_sha256 must be a boolean")
        frame = _nonnegative_int(frame_index, "frame_index")
        if self._last_recorded_frame is not None and frame <= self._last_recorded_frame:
            raise EpisodeMemoryLifecycleError(
                "factual observation frame_index must increase strictly"
            )
        if not isinstance(factual_payload, Mapping):
            raise OnlineEpisodeMemoryError("warm_factual_observation must be a mapping")
        expected = {"world_tokens", "vae_latent", "proprio"}
        if set(factual_payload) != expected:
            raise OnlineEpisodeMemoryError(
                "warm_factual_observation fields must be exactly "
                f"{sorted(expected)}"
            )
        world = _readonly(
            factual_payload["world_tokens"],
            dtype=np.dtype(np.float32),
            name="warm_factual_observation.world_tokens",
            rank=2,
        )
        if int(world.shape[1]) != self._semantic_dim:
            raise OnlineEpisodeMemoryError(
                "warm_factual_observation semantic dimension does not match the model"
            )
        latent_raw = np.asarray(factual_payload["vae_latent"])
        if latent_raw.ndim < 1:
            raise OnlineEpisodeMemoryError(
                "warm_factual_observation.vae_latent must have rank >= 1"
            )
        latent = np.ascontiguousarray(latent_raw, dtype=np.float32)
        if not latent.size or not np.isfinite(latent).all():
            raise OnlineEpisodeMemoryError(
                "warm_factual_observation.vae_latent must be finite and non-empty"
            )
        proprio = _readonly(
            factual_payload["proprio"],
            dtype=np.dtype(np.float32),
            name="warm_factual_observation.proprio",
            rank=1,
        )
        observation = FactualObservation.from_environment(
            episode_id=self._episode_id,
            frame_index=frame,
            observation_id=f"{self._episode_id}:{frame}",
            world_tokens=world,
            vae_latent=latent,
            proprio=proprio,
        )

        actions = np.asarray(
            [] if executed_actions_since_previous is None else executed_actions_since_previous,
            dtype=np.float32,
        )
        if self._capability is None:
            if actions.size:
                raise OnlineEpisodeMemoryError(
                    "the initial factual observation cannot have preceding policy actions"
                )
            self._capability = self._memory.begin_episode(observation)
            update_evidence: dict[str, Any] = {
                "event_written": False,
                "change_score": None,
                "write_threshold": None,
            }
            observation_updates = 0
            event_writes = 0
            event_merges = 0
        else:
            if actions.ndim != 2 or actions.shape[1] != self._memory.config.action_dim:
                raise OnlineEpisodeMemoryError(
                    "executed action prefix must have shape [T, action_dim]"
                )
            if actions.shape[0] <= 0 or not np.isfinite(actions).all():
                raise OnlineEpisodeMemoryError(
                    "non-initial factual observations require a finite non-empty "
                    "executed action prefix"
                )
            update = self._memory.update(
                self._capability,
                observation=observation,
                executed_actions=np.ascontiguousarray(actions),
            )
            update_evidence = {
                "event_written": bool(update.event_written),
                "change_score": float(update.change_score),
                "write_threshold": float(update.write_threshold),
                "repeated_attempt_count": int(
                    update.event_status.repeated_attempt_count
                ),
            }
            observation_updates = int(update.counters.observation_updates)
            event_writes = int(update.counters.event_writes)
            event_merges = int(update.counters.event_merges)
        self._last_recorded_frame = frame
        snapshot_sha256 = (
            self._memory.snapshot().sha256
            if include_snapshot_sha256
            else None
        )
        return {
            **update_evidence,
            "snapshot_sha256_after_record": snapshot_sha256,
            "observation_updates": observation_updates,
            "event_writes": event_writes,
            "event_merges": event_merges,
        }

    def end_episode(
        self, *, include_snapshot_sha256: bool = True
    ) -> dict[str, Any]:
        """Seal current history and invalidate its capability."""

        self._require_active()
        if not isinstance(include_snapshot_sha256, bool):
            raise TypeError("include_snapshot_sha256 must be a boolean")
        if self._capability is None:
            evidence = {
                "snapshot_sha256": None,
                "observation_updates": 0,
                "event_writes": 0,
                "event_merges": 0,
            }
        else:
            snapshot = self._memory.end_episode(self._capability)
            evidence = {
                "snapshot_sha256": (
                    snapshot.sha256 if include_snapshot_sha256 else None
                ),
                "observation_updates": snapshot.counters.observation_updates,
                "event_writes": snapshot.counters.event_writes,
                "event_merges": snapshot.counters.event_merges,
            }
        self._capability = None
        self._episode_index = None
        self._episode_id = None
        self._last_recorded_frame = None
        return evidence


__all__ = [
    "EpisodeHistoryInputs",
    "OnlineEpisodeMemoryError",
    "OnlineRetrospectiveEpisodeMemory",
]
