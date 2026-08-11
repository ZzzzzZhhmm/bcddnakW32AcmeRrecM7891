from __future__ import annotations

import json

import numpy as np
import pytest

from fastwam.memory.episode_memory import (
    FACTUAL_OBSERVATION_PROVENANCE,
    EpisodeMemoryCapabilityError,
    EpisodeMemoryConfig,
    EpisodeMemoryLifecycleError,
    EpisodeMemoryPhase,
    EpisodeMemoryValidationError,
    EpisodeWorkingMemory,
    FactualObservation,
)


def _observation(
    episode: str | int,
    frame: int,
    world: np.ndarray | float,
    *,
    vae: np.ndarray | float | None = None,
    proprio: np.ndarray | None = None,
) -> FactualObservation:
    world_value = np.asarray(world, dtype=np.float32)
    if world_value.ndim == 0:
        world_value = np.full((2, 3), float(world_value), dtype=np.float32)
    if vae is None:
        vae_value = np.full((2, 2, 2), float(np.mean(world_value)), dtype=np.float32)
    else:
        vae_value = np.asarray(vae, dtype=np.float32)
        if vae_value.ndim == 0:
            vae_value = np.full((2, 2, 2), float(vae_value), dtype=np.float32)
    return FactualObservation.from_environment(
        episode_id=episode,
        frame_index=frame,
        observation_id=f"obs-{episode}-{frame}",
        world_tokens=world_value,
        vae_latent=vae_value,
        proprio=np.zeros((4,), dtype=np.float32) if proprio is None else proprio,
    )


def _config(**overrides: object) -> EpisodeMemoryConfig:
    values: dict[str, object] = {
        "action_dim": 3,
        "gripper_indices": (2,),
        "change_warmup": 100,
        "change_threshold_floor": 0.01,
    }
    values.update(overrides)
    return EpisodeMemoryConfig(**values)


def _actions(
    scale: float = 0.0,
    *,
    gripper: tuple[float, ...] = (-1.0, -1.0),
) -> np.ndarray:
    result = np.zeros((len(gripper), 3), dtype=np.float32)
    result[:, 0] = scale
    result[:, 2] = np.asarray(gripper, dtype=np.float32)
    return result


def test_factual_observation_has_no_prediction_write_path_and_is_read_only() -> None:
    raw_world = np.ones((2, 3), dtype=np.float64)
    observation = _observation("episode-a", 0, raw_world)
    raw_world[:] = 9.0

    assert observation.provenance == FACTUAL_OBSERVATION_PROVENANCE
    assert observation.episode_id == "episode-a"
    assert observation.world_tokens.dtype == np.float32
    assert observation.world_tokens.flags.c_contiguous
    assert not observation.world_tokens.flags.writeable
    assert not observation.vae_latent.flags.writeable
    assert not observation.proprio.flags.writeable
    np.testing.assert_array_equal(observation.world_tokens, np.ones((2, 3)))

    with pytest.raises(EpisodeMemoryValidationError, match="from_environment"):
        FactualObservation(
            episode_id="episode-a",
            frame_index=1,
            world_tokens=np.zeros((2, 3)),
            vae_latent=np.zeros((2, 2, 2)),
            proprio=np.zeros((4,)),
        )
    with pytest.raises(ValueError):
        observation.world_tokens[0, 0] = 2.0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("world_tokens", np.zeros((3,), dtype=np.float32), "ndim=2"),
        ("world_tokens", np.asarray([[np.nan]], dtype=np.float32), "finite"),
        ("vae_latent", np.asarray([np.inf], dtype=np.float32), "finite"),
        ("proprio", np.zeros((1, 2), dtype=np.float32), "ndim=1"),
    ],
)
def test_factual_observation_rejects_bad_shape_or_nonfinite(
    field: str, value: np.ndarray, message: str
) -> None:
    kwargs = {
        "episode_id": "e",
        "frame_index": 0,
        "world_tokens": np.zeros((2, 3), dtype=np.float32),
        "vae_latent": np.zeros((2, 2), dtype=np.float32),
        "proprio": np.zeros((4,), dtype=np.float32),
    }
    kwargs[field] = value
    with pytest.raises(EpisodeMemoryValidationError, match=message):
        FactualObservation.from_environment(**kwargs)


def test_capabilities_are_instance_episode_and_generation_bound() -> None:
    first = EpisodeWorkingMemory(_config())
    second = EpisodeWorkingMemory(_config())
    cap_a = first.begin_episode(_observation("a", 0, 0.0))
    cap_b = second.begin_episode(_observation("a", 0, 0.0))

    with pytest.raises(EpisodeMemoryCapabilityError, match="another memory"):
        first.record_observation(
            cap_b,
            observation=_observation("a", 1, 1.0),
            executed_actions=_actions(),
        )
    with pytest.raises(EpisodeMemoryCapabilityError, match="another episode"):
        first.record_observation(
            cap_a,
            observation=_observation("b", 1, 1.0),
            executed_actions=_actions(),
        )

    cap_new = first.begin_episode(_observation("new", 0, 0.0))
    with pytest.raises(EpisodeMemoryCapabilityError, match="stale"):
        first.record_observation(
            cap_a,
            observation=_observation("a", 1, 1.0),
            executed_actions=_actions(),
        )
    first.end_episode(cap_new)
    with pytest.raises(EpisodeMemoryCapabilityError, match="stale"):
        first.record_observation(
            cap_new,
            observation=_observation("new", 1, 1.0),
            executed_actions=_actions(),
        )


def test_reset_is_empty_and_invalidates_token_deterministically() -> None:
    memory = EpisodeWorkingMemory(_config())
    cap = memory.begin_episode(_observation("a", 0, 0.0))
    memory.reset()
    first = memory.to_json()
    memory.reset()
    second = memory.to_json()
    assert first == second
    snapshot = memory.snapshot()
    assert snapshot.phase is EpisodeMemoryPhase.EMPTY
    assert snapshot.episode_id is None
    assert snapshot.initial_anchor is None
    assert snapshot.recent_events == ()
    with pytest.raises(EpisodeMemoryCapabilityError, match="stale"):
        memory.record_observation(
            cap,
            observation=_observation("a", 1, 1.0),
            executed_actions=_actions(),
        )


def test_frame_and_feature_shapes_are_fixed_without_partial_mutation() -> None:
    memory = EpisodeWorkingMemory(_config())
    cap = memory.begin_episode(_observation("a", 0, 0.0))
    before = memory.to_json()
    with pytest.raises(EpisodeMemoryLifecycleError, match="increase strictly"):
        memory.record_observation(
            cap,
            observation=_observation("a", 0, 1.0),
            executed_actions=_actions(),
        )
    assert memory.to_json() == before

    wrong = FactualObservation.from_environment(
        episode_id="a",
        frame_index=1,
        world_tokens=np.zeros((4, 3), dtype=np.float32),
        vae_latent=np.zeros((2, 2, 2), dtype=np.float32),
        proprio=np.zeros((4,), dtype=np.float32),
    )
    with pytest.raises(EpisodeMemoryValidationError, match="world_tokens shape"):
        memory.record_observation(cap, observation=wrong, executed_actions=_actions())
    assert memory.to_json() == before

    with pytest.raises(EpisodeMemoryValidationError, match=r"shape \[T, 3\]"):
        memory.record_observation(
            cap,
            observation=_observation("a", 1, 1.0),
            executed_actions=np.zeros((2, 4), dtype=np.float32),
        )
    assert memory.to_json() == before


def test_action_summary_contains_displacement_gripper_curvature_and_repetition() -> None:
    memory = EpisodeWorkingMemory(
        _config(max_action_summaries=2, change_weights=(0.0, 0.0, 1.0, 0.0))
    )
    cap = memory.begin_episode(_observation("a", 0, 0.0))
    actions = np.asarray(
        [
            [1.0, 0.0, -1.0],
            [0.0, 1.0, -1.0],
            [-1.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    first = memory.record_observation(
        cap, observation=_observation("a", 1, 1.0), executed_actions=actions
    )
    np.testing.assert_allclose(first.action_summary.mean_displacement, [0.0, 1.0 / 3.0, 0.0])
    np.testing.assert_allclose(first.action_summary.final_displacement, [0.0, 1.0, 0.0])
    np.testing.assert_array_equal(first.action_summary.gripper_transition_counts, [[0, 1]])
    assert first.action_summary.close_count == 0
    assert first.action_summary.release_count == 1
    assert first.action_summary.curvature == pytest.approx(0.5)
    assert first.action_summary.repeated is False

    # A repeated command is stagnation only when the factual world did not
    # progress.  This avoids labelling normal multi-chunk approaches as loops.
    second = memory.record_observation(
        cap, observation=_observation("a", 2, 1.0), executed_actions=actions
    )
    assert second.action_summary.repeated is True
    assert second.action_summary.repetition_similarity == pytest.approx(1.0)
    # The previous terminal command was open and the repeated chunk begins
    # closed, so the factual cross-chunk transition is counted as well.
    np.testing.assert_array_equal(second.action_summary.gripper_transition_counts, [[1, 1]])
    assert second.event_status.repeated_similar_action is True
    assert second.event_status.repeated_attempt_count == 1
    assert len(memory.snapshot().executed_action_summaries) == 2
    assert not second.action_summary.mean_displacement.flags.writeable


def test_absolute_target_summary_is_relative_to_factual_start_and_progress_coupled() -> None:
    memory = EpisodeWorkingMemory(
        _config(
            action_mode="absolute_target",
            max_action_summaries=3,
            change_weights=(0.0, 0.0, 1.0, 0.0),
        )
    )
    start = np.asarray([10.0, -3.0, 0.0, 99.0], dtype=np.float32)
    cap = memory.begin_episode(_observation("absolute", 0, 0.0, proprio=start))
    targets = np.asarray(
        [[10.2, -2.9, -1.0], [10.4, -2.8, -1.0]], dtype=np.float32
    )
    first = memory.record_observation(
        cap,
        observation=_observation("absolute", 1, 0.0, proprio=start),
        executed_actions=targets,
    )
    np.testing.assert_allclose(
        first.action_summary.mean_displacement, [0.3, 0.15, 0.0], atol=1e-6
    )
    np.testing.assert_allclose(
        first.action_summary.final_displacement, [0.4, 0.2, 0.0], atol=1e-6
    )
    second = memory.record_observation(
        cap,
        observation=_observation("absolute", 2, 0.0, proprio=start),
        executed_actions=targets,
    )
    assert second.action_summary.repeated is True
    progressed = memory.record_observation(
        cap,
        observation=_observation("absolute", 3, 1.0, proprio=start),
        executed_actions=targets,
    )
    assert progressed.action_summary.repetition_similarity == pytest.approx(1.0)
    assert progressed.action_summary.repeated is False
    assert progressed.event_status.repeated_similar_action is False
    assert progressed.event_status.repeated_attempt_count == 0


def test_vae_only_change_does_not_clear_factual_stagnation() -> None:
    memory = EpisodeWorkingMemory(
        _config(
            max_action_summaries=3,
            motion_world_change_threshold=0.08,
            stationary_world_change_threshold=0.03,
        )
    )
    actions = _actions(scale=0.25)
    cap = memory.begin_episode(_observation("vae-noise", 0, 1.0, vae=1.0))
    memory.record_observation(
        cap,
        observation=_observation("vae-noise", 1, 1.0, vae=1.0),
        executed_actions=actions,
    )

    repeated = memory.record_observation(
        cap,
        observation=_observation("vae-noise", 2, 1.0, vae=2.0),
        executed_actions=actions,
    )

    assert repeated.change_components[2] == pytest.approx(0.0)
    assert repeated.change_components[3] > 0.03
    assert repeated.action_summary.repeated is True
    assert repeated.event_status.repeated_similar_action is True
    assert repeated.event_status.repeated_attempt_count == 1


def test_world_progress_or_cross_modal_progress_clears_stagnation() -> None:
    config = _config(
        max_action_summaries=3,
        motion_world_change_threshold=0.08,
        stationary_world_change_threshold=0.03,
    )
    actions = _actions(scale=0.25)

    decisive = EpisodeWorkingMemory(config)
    decisive_cap = decisive.begin_episode(
        _observation("world-progress", 0, 1.0, vae=1.0)
    )
    decisive.record_observation(
        decisive_cap,
        observation=_observation("world-progress", 1, 1.0, vae=1.0),
        executed_actions=actions,
    )
    world_progress = decisive.record_observation(
        decisive_cap,
        observation=_observation("world-progress", 2, 2.0, vae=1.0),
        executed_actions=actions,
    )
    assert world_progress.change_components[2] >= 0.08
    assert world_progress.action_summary.repeated is False
    assert world_progress.event_status.repeated_attempt_count == 0

    corroborated = EpisodeWorkingMemory(config)
    corroborated_cap = corroborated.begin_episode(
        _observation("cross-modal-progress", 0, 1.0, vae=1.0)
    )
    corroborated.record_observation(
        corroborated_cap,
        observation=_observation("cross-modal-progress", 1, 1.0, vae=1.0),
        executed_actions=actions,
    )
    cross_modal_progress = corroborated.record_observation(
        corroborated_cap,
        observation=_observation("cross-modal-progress", 2, 1.1, vae=1.1),
        executed_actions=actions,
    )
    assert 0.03 < cross_modal_progress.change_components[2] < 0.08
    assert cross_modal_progress.change_components[3] > 0.03
    assert cross_modal_progress.action_summary.repeated is False
    assert cross_modal_progress.event_status.repeated_attempt_count == 0


def test_adaptive_threshold_uses_only_prior_scores_and_bounds_history() -> None:
    config = _config(
        change_weights=(0.0, 0.0, 1.0, 0.0),
        change_warmup=2,
        change_history_size=3,
        change_threshold_floor=0.10,
        change_threshold_ceiling=0.40,
        change_mad_scale=2.0,
    )
    memory = EpisodeWorkingMemory(config)
    cap = memory.begin_episode(_observation("a", 0, 1.0))

    first = memory.record_observation(
        cap, observation=_observation("a", 1, 1.0), executed_actions=_actions()
    )
    second = memory.record_observation(
        cap, observation=_observation("a", 2, 1.0), executed_actions=_actions()
    )
    third = memory.record_observation(
        cap, observation=_observation("a", 3, 3.0), executed_actions=_actions()
    )
    assert first.write_threshold == pytest.approx(0.10)
    assert second.write_threshold == pytest.approx(0.10)
    # Earlier scores are both zero; the current large change cannot raise its
    # own threshold and therefore writes an event.
    assert third.write_threshold == pytest.approx(0.10)
    assert third.event_written is True

    memory.record_observation(
        cap, observation=_observation("a", 4, 4.0), executed_actions=_actions()
    )
    history = memory.snapshot().change_score_history
    assert len(history) == 3
    assert history == tuple(memory.snapshot().change_score_history)


def test_event_status_is_low_level_factual_state_not_semantic_subtask_label() -> None:
    config = _config(
        change_weights=(1.0, 0.0, 0.0, 0.0),
        motion_world_change_threshold=0.10,
        stationary_world_change_threshold=0.02,
    )
    memory = EpisodeWorkingMemory(config)
    cap = memory.begin_episode(_observation("a", 0, 0.0))

    close = memory.record_observation(
        cap,
        observation=_observation("a", 1, 1.0),
        executed_actions=_actions(gripper=(1.0, -1.0)),
    )
    assert close.event_status.close_count == 1
    assert close.event_status.gripper_closed_recently is True
    assert close.event_status.visual_motion_after_close is True
    assert not hasattr(close.event_status, "subtask")

    release = memory.record_observation(
        cap,
        observation=_observation("a", 2, 1.0),
        executed_actions=_actions(gripper=(-1.0, 1.0)),
    )
    assert release.event_status.release_count == 1
    assert release.event_status.release_happened is True
    assert release.event_status.gripper_closed_recently is False
    assert release.event_status.stationary_after_release is True


def test_bounded_events_merge_most_similar_adjacent_middle_pair_and_protect_latest() -> None:
    config = _config(
        max_recent_events=3,
        change_threshold_floor=0.0,
        change_weights=(0.0, 0.0, 1.0, 0.0),
    )
    memory = EpisodeWorkingMemory(config)
    cap = memory.begin_episode(
        _observation("a", 0, np.asarray([[0.0, 0.0]], dtype=np.float32))
    )
    values = (
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        np.asarray([[0.9, 0.1]], dtype=np.float32),
        np.asarray([[0.0, 1.0]], dtype=np.float32),
        np.asarray([[-1.0, 0.0]], dtype=np.float32),
    )
    for frame, world in enumerate(values, start=1):
        update = memory.record_observation(
            cap,
            observation=_observation("a", frame, world, vae=float(frame)),
            executed_actions=_actions(scale=float(frame)),
        )
        assert update.event_written is True

    snapshot = memory.snapshot()
    assert len(snapshot.recent_events) == 3
    assert snapshot.counters.event_writes == 4
    assert snapshot.counters.event_merges == 1
    merged, middle, latest = snapshot.recent_events
    assert merged.mass == 2
    assert merged.compressed is True
    np.testing.assert_allclose(
        merged.representative_world_tokens,
        (values[0] + values[1]) / 2.0,
    )
    assert merged.start_frame == 0
    assert merged.end_frame == 2
    assert middle.end_frame == 3
    assert latest.end_frame == 4
    np.testing.assert_array_equal(latest.representative_world_tokens, values[-1])
    # Initial anchor is separate from and untouched by slot compression.
    np.testing.assert_array_equal(snapshot.initial_anchor.world_tokens, [[0.0, 0.0]])


def test_active_snapshot_round_trip_is_canonical_and_mints_new_capability() -> None:
    memory = EpisodeWorkingMemory(_config())
    old_cap = memory.begin_episode(_observation("a", 0, 0.0))
    memory.record_observation(
        old_cap,
        observation=_observation("a", 1, 1.0),
        executed_actions=_actions(scale=0.25, gripper=(1.0, -1.0)),
    )
    payload = memory.to_json()
    restored, restored_cap = EpisodeWorkingMemory.from_json(payload)
    assert restored_cap is not None
    assert restored.to_json() == payload
    assert restored.snapshot().sha256 == memory.snapshot().sha256

    with pytest.raises(EpisodeMemoryCapabilityError, match="another memory"):
        restored.record_observation(
            old_cap,
            observation=_observation("a", 2, 2.0),
            executed_actions=_actions(),
        )
    restored.record_observation(
        restored_cap,
        observation=_observation("a", 2, 2.0),
        executed_actions=_actions(),
    )


def test_ended_and_empty_round_trip_do_not_mint_write_capability() -> None:
    empty = EpisodeWorkingMemory(_config())
    empty_restored, empty_cap = EpisodeWorkingMemory.from_json(empty.to_json())
    assert empty_cap is None
    assert empty_restored.phase is EpisodeMemoryPhase.EMPTY

    memory = EpisodeWorkingMemory(_config())
    cap = memory.begin_episode(_observation("a", 0, 0.0))
    ended = memory.end_episode(cap)
    restored, restored_cap = EpisodeWorkingMemory.from_json(ended.to_json())
    assert restored_cap is None
    assert restored.phase is EpisodeMemoryPhase.ENDED
    assert restored.to_json() == ended.to_json()


def test_serialization_is_closed_world_and_rejects_predicted_provenance() -> None:
    memory = EpisodeWorkingMemory(_config())
    memory.begin_episode(_observation("a", 0, 0.0))
    value = json.loads(memory.to_json())
    value["unexpected"] = 1
    with pytest.raises(EpisodeMemoryValidationError, match="fields mismatch"):
        EpisodeWorkingMemory.from_json(json.dumps(value))

    value = json.loads(memory.to_json())
    value["state"]["initial_anchor"]["provenance"] = "predicted_future"
    with pytest.raises(EpisodeMemoryValidationError, match="not factual"):
        EpisodeWorkingMemory.from_json(json.dumps(value))


def test_config_validation_is_strict() -> None:
    with pytest.raises(EpisodeMemoryValidationError, match="unique"):
        EpisodeMemoryConfig(action_dim=3, gripper_indices=(2, 2))
    with pytest.raises(EpisodeMemoryValidationError, match="within action_dim"):
        EpisodeMemoryConfig(action_dim=3, gripper_indices=(3,))
    with pytest.raises(EpisodeMemoryValidationError, match="max_recent_events"):
        EpisodeMemoryConfig(action_dim=3, max_recent_events=1)
    with pytest.raises(EpisodeMemoryValidationError, match="positive total"):
        EpisodeMemoryConfig(action_dim=3, change_weights=(0.0, 0.0, 0.0, 0.0))
    with pytest.raises(EpisodeMemoryValidationError, match="ceiling"):
        EpisodeMemoryConfig(
            action_dim=3,
            change_threshold_floor=0.5,
            change_threshold_ceiling=0.4,
        )
