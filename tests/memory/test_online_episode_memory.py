from __future__ import annotations

import numpy as np
import pytest

from fastwam.memory.episode_memory import (
    EpisodeMemoryLifecycleError,
    EpisodeMemorySnapshot,
)
from fastwam.memory.online_episode_memory import (
    OnlineEpisodeMemoryError,
    OnlineRetrospectiveEpisodeMemory,
)


def _payload(value: float, *, semantic_dim: int = 8) -> dict[str, np.ndarray]:
    return {
        "world_tokens": np.full((4, semantic_dim), value, dtype=np.float32),
        "vae_latent": np.full((2, 3, 2, 2), value, dtype=np.float32),
        "proprio": np.full((9,), value, dtype=np.float32),
    }


def _runtime() -> OnlineRetrospectiveEpisodeMemory:
    return OnlineRetrospectiveEpisodeMemory(
        action_dim=7,
        action_horizon=16,
        semantic_dim=8,
        gripper_indices=(6,),
        recent_event_capacity=3,
    )


def test_history_is_empty_until_first_model_certified_factual_observation() -> None:
    runtime = _runtime()
    runtime.begin_episode(3)
    assert runtime.history_inputs() is None

    evidence = runtime.record_factual_observation(
        frame_index=5,
        factual_payload=_payload(0.0),
        executed_actions_since_previous=[],
    )
    assert evidence["observation_updates"] == 0

    history = runtime.history_inputs()
    assert history is not None
    assert history.episode_tokens.shape == (4, 8)
    assert history.episode_mask.tolist() == [True] * 4
    assert history.observation_count == 1
    assert history.action_summary_count == 0
    assert not history.episode_tokens.flags.writeable
    assert not history.episode_mask.flags.writeable
    model_kwargs = history.model_kwargs()
    assert model_kwargs["episode_tokens"].flags.writeable
    assert model_kwargs["episode_mask"].flags.writeable


def test_offline_replay_can_skip_expensive_snapshot_serialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    runtime.begin_episode(0)

    def fail_digest(_snapshot: EpisodeMemorySnapshot) -> str:
        raise AssertionError("offline replay must not serialize a snapshot")

    monkeypatch.setattr(EpisodeMemorySnapshot, "sha256", property(fail_digest))
    initial = runtime.record_factual_observation(
        frame_index=0,
        factual_payload=_payload(0.0),
        executed_actions_since_previous=None,
        include_snapshot_sha256=False,
    )
    update = runtime.record_factual_observation(
        frame_index=4,
        factual_payload=_payload(1.0),
        executed_actions_since_previous=np.ones((4, 7), dtype=np.float32),
        include_snapshot_sha256=False,
    )
    sealed = runtime.end_episode(include_snapshot_sha256=False)

    assert initial["snapshot_sha256_after_record"] is None
    assert initial["observation_updates"] == 0
    assert update["snapshot_sha256_after_record"] is None
    assert update["observation_updates"] == 1
    assert sealed["snapshot_sha256"] is None


def test_exact_executed_prefix_is_committed_only_after_next_real_observation() -> None:
    runtime = _runtime()
    runtime.begin_episode(0)
    runtime.record_factual_observation(
        frame_index=5,
        factual_payload=_payload(0.0),
        executed_actions_since_previous=None,
    )
    before = runtime.history_inputs()
    assert before is not None and before.action_summary_count == 0

    executed = np.zeros((5, 7), dtype=np.float32)
    executed[:, 0] = np.linspace(0.0, 1.0, 5)
    executed[-2:, 6] = 1.0
    preview = runtime.history_inputs(
        executed_actions_since_previous=executed
    )
    assert preview is not None
    assert preview.action_summary_count == 1
    preview_vector = preview.episode_action_summaries[-1].copy()
    update = runtime.record_factual_observation(
        frame_index=10,
        factual_payload=_payload(1.0),
        executed_actions_since_previous=executed,
    )
    assert update["observation_updates"] == 1
    assert update["event_writes"] == 1

    history = runtime.history_inputs()
    assert history is not None
    assert history.observation_count == 2
    assert history.action_summary_count == 1
    assert history.episode_action_summaries.shape == (1, 25)
    assert history.episode_action_mask.tolist() == [True]
    # Exact summary layout: mean/final/terminal command + four scalars.
    assert history.episode_action_summaries[0, 20] == 1.0
    assert history.episode_action_summaries[0, 21] == pytest.approx(5 / 16)
    np.testing.assert_allclose(
        history.episode_action_summaries[-1], preview_vector, rtol=0.0, atol=1e-7
    )
    assert history.event_count == 1
    # initial anchor + factual post-event observation
    assert history.episode_tokens.shape == (8, 8)


def test_initial_observation_rejects_actions_and_later_observation_requires_them() -> None:
    runtime = _runtime()
    runtime.begin_episode(0)
    with pytest.raises(OnlineEpisodeMemoryError, match="initial factual"):
        runtime.record_factual_observation(
            frame_index=5,
            factual_payload=_payload(0.0),
            executed_actions_since_previous=np.zeros((1, 7), dtype=np.float32),
        )

    runtime.record_factual_observation(
        frame_index=5,
        factual_payload=_payload(0.0),
        executed_actions_since_previous=None,
    )
    with pytest.raises(OnlineEpisodeMemoryError, match="non-empty"):
        runtime.record_factual_observation(
            frame_index=10,
            factual_payload=_payload(0.2),
            executed_actions_since_previous=np.zeros((0, 7), dtype=np.float32),
        )


def test_model_payload_is_strict_and_cannot_smuggle_predicted_future_fields() -> None:
    runtime = _runtime()
    runtime.begin_episode(1)
    payload = _payload(0.0)
    payload["predicted_future"] = np.zeros((1,), dtype=np.float32)
    with pytest.raises(OnlineEpisodeMemoryError, match="fields must be exactly"):
        runtime.record_factual_observation(
            frame_index=5,
            factual_payload=payload,
            executed_actions_since_previous=None,
        )


def test_episode_reset_invalidates_old_history_and_requires_new_initial_anchor() -> None:
    runtime = _runtime()
    runtime.begin_episode(1)
    runtime.record_factual_observation(
        frame_index=5,
        factual_payload=_payload(0.0),
        executed_actions_since_previous=None,
    )
    sealed = runtime.end_episode()
    assert sealed["snapshot_sha256"] is not None
    with pytest.raises(EpisodeMemoryLifecycleError, match="begin_episode"):
        runtime.history_inputs()

    runtime.begin_episode(2)
    assert runtime.history_inputs() is None
    runtime.record_factual_observation(
        frame_index=5,
        factual_payload=_payload(4.0),
        executed_actions_since_previous=None,
    )
    history = runtime.history_inputs()
    assert history is not None
    np.testing.assert_array_equal(history.episode_tokens, 4.0)


def test_frames_must_increase_strictly_within_episode() -> None:
    runtime = _runtime()
    runtime.begin_episode(0)
    runtime.record_factual_observation(
        frame_index=5,
        factual_payload=_payload(0.0),
        executed_actions_since_previous=None,
    )
    with pytest.raises(EpisodeMemoryLifecycleError, match="increase strictly"):
        runtime.record_factual_observation(
            frame_index=5,
            factual_payload=_payload(1.0),
            executed_actions_since_previous=np.zeros((1, 7), dtype=np.float32),
        )
