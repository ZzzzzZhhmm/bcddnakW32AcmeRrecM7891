"""Benchmark-neutral causal episode lifecycle for online WARM policies.

The simulator adapters for LIBERO, RoboTwin, and RMBench all need the same
state machine: issue monotonic query identities, expose only factual history
that precedes a replan, remember the exact commands that were actually sent
to the environment, and commit a real observation only after model inference.
Keeping that state machine here prevents benchmark wrappers from quietly
drifting apart.

This module deliberately has no Torch or simulator imports.  The retriever and
episode-memory objects are capability-style dependencies supplied by a
benchmark-specific, contract-validating loader.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


@dataclass(slots=True)
class OnlineEpisodeController:
    """Own one fail-closed online WARM episode lifecycle.

    ``frame_index`` is the number of policy commands already executed, not a
    simulator-internal clock.  This remains stable when a benchmark performs
    reset waits, expert seed checks, rendering, or other non-policy steps.
    """

    contract: Any
    retriever: Any | None
    retrospective_episode_memory: Any | None
    _active_episode_index: int | None = field(default=None, init=False, repr=False)
    _last_frame_index: int | None = field(default=None, init=False, repr=False)
    _seen_episode_indices: set[int] = field(
        default_factory=set, init=False, repr=False
    )
    _executed_actions_since_replan: list[np.ndarray] = field(
        default_factory=list, init=False, repr=False
    )
    _executed_environment_actions_since_replan: list[np.ndarray] = field(
        default_factory=list, init=False, repr=False
    )

    @property
    def active_episode_index(self) -> int | None:
        return self._active_episode_index

    @property
    def last_frame_index(self) -> int | None:
        return self._last_frame_index

    def begin_episode(self, episode_index: int) -> None:
        if isinstance(episode_index, bool) or not isinstance(episode_index, int):
            raise TypeError("episode_index must be a non-negative integer")
        if episode_index < 0:
            raise ValueError("episode_index must be a non-negative integer")
        if self._active_episode_index is not None:
            raise RuntimeError("end_episode() is required before the next episode")
        if episode_index in self._seen_episode_indices:
            raise ValueError(f"episode_index {episode_index} was already evaluated")
        if self.retriever is not None:
            self.retriever.begin_episode(episode_index)
        if self.retrospective_episode_memory is not None:
            self.retrospective_episode_memory.begin_episode(episode_index)
        self._executed_actions_since_replan.clear()
        self._executed_environment_actions_since_replan.clear()
        self._seen_episode_indices.add(episode_index)
        self._active_episode_index = episode_index
        self._last_frame_index = None

    def issue_query_id(self, frame_index: int) -> Any:
        if isinstance(frame_index, bool) or not isinstance(frame_index, int):
            raise TypeError("frame_index must be a non-negative integer")
        if frame_index < 0:
            raise ValueError("frame_index must be a non-negative integer")
        if self._active_episode_index is None:
            raise RuntimeError("begin_episode() is required before online replanning")
        if self._last_frame_index is not None and frame_index <= self._last_frame_index:
            raise ValueError("online frame_index must increase strictly within an episode")
        if self.retriever is not None:
            query_id = self.retriever.make_query_id(frame_index)
        else:
            from fastwam.memory.online_retrieval import make_online_query_id

            query_id = make_online_query_id(
                self.contract, self._active_episode_index, frame_index
            )
        self._last_frame_index = frame_index
        return query_id

    def retrospective_history_kwargs(
        self,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Return factual history strictly preceding the current replan."""

        if self.retrospective_episode_memory is None:
            return {}, None
        pending = (
            None
            if not self._executed_actions_since_replan
            else np.stack(self._executed_actions_since_replan, axis=0)
        )
        history = self.retrospective_episode_memory.history_inputs(
            executed_actions_since_previous=pending
        )
        if history is None:
            return {}, None
        return history.model_kwargs(), history.evidence()

    def note_executed_action(
        self,
        environment_action: Any,
        *,
        model_space_action: Any,
    ) -> None:
        """Record exactly one command after the simulator accepted it."""

        if self._active_episode_index is None:
            raise RuntimeError("cannot record an action outside an active episode")
        if self.retrospective_episode_memory is None:
            return
        env = np.asarray(environment_action, dtype=np.float32)
        model = np.asarray(model_space_action, dtype=np.float32)
        if env.ndim != 1 or not env.size or not np.isfinite(env).all():
            raise ValueError("executed environment action must be one finite vector")
        if model.shape != env.shape or not np.isfinite(model).all():
            raise ValueError(
                "model-space executed action must match the environment command shape"
            )
        self._executed_environment_actions_since_replan.append(
            np.ascontiguousarray(env)
        )
        self._executed_actions_since_replan.append(np.ascontiguousarray(model))

    def commit_factual_replan_observation(
        self,
        *,
        frame_index: int,
        model_output: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Commit model-certified features of the current real observation."""

        if self._active_episode_index is None:
            raise RuntimeError("cannot commit an observation outside an active episode")
        if self.retrospective_episode_memory is None:
            return None
        if frame_index != self._last_frame_index:
            raise ValueError("committed frame_index must equal the issued query frame")
        payload = model_output.get("warm_factual_observation")
        if not isinstance(payload, Mapping):
            raise RuntimeError(
                "full WARM inference returned no warm_factual_observation"
            )
        actions = (
            None
            if not self._executed_actions_since_replan
            else np.stack(self._executed_actions_since_replan, axis=0)
        )
        evidence = self.retrospective_episode_memory.record_factual_observation(
            frame_index=frame_index,
            factual_payload=payload,
            executed_actions_since_previous=actions,
        )
        if self._executed_environment_actions_since_replan:
            from fastwam.memory.manifest import sha256_array

            exact = np.stack(
                self._executed_environment_actions_since_replan, axis=0
            )
            evidence["executed_environment_prefix_sha256"] = sha256_array(exact)
            evidence["executed_environment_prefix_count"] = int(exact.shape[0])
        else:
            evidence["executed_environment_prefix_sha256"] = None
            evidence["executed_environment_prefix_count"] = 0
        self._executed_actions_since_replan.clear()
        self._executed_environment_actions_since_replan.clear()
        return evidence

    def end_episode(self) -> dict[str, Any] | None:
        """Seal an episode, preserving any terminal unpaired action evidence."""

        if self._active_episode_index is None:
            raise RuntimeError("no active episode to end")
        evidence: dict[str, Any] | None = None
        if self.retrospective_episode_memory is not None:
            evidence = self.retrospective_episode_memory.end_episode()
            evidence["unpaired_terminal_action_count"] = len(
                self._executed_actions_since_replan
            )
            if self._executed_environment_actions_since_replan:
                from fastwam.memory.manifest import sha256_array

                terminal = np.stack(
                    self._executed_environment_actions_since_replan, axis=0
                )
                evidence["unpaired_terminal_environment_actions_sha256"] = (
                    sha256_array(terminal)
                )
            else:
                evidence["unpaired_terminal_environment_actions_sha256"] = None
        self._executed_actions_since_replan.clear()
        self._executed_environment_actions_since_replan.clear()
        self._active_episode_index = None
        self._last_frame_index = None
        return evidence


__all__ = ["OnlineEpisodeController"]
