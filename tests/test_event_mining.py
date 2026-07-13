import numpy as np
import pytest

from fastwam.memory.event_mining import (
    EventMiningConfig,
    mine_episode_events,
    robust_mad_normalize,
    select_window_starts,
)


def _constant_episode(num_actions: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actions = np.zeros((num_actions, 2), dtype=np.float32)
    proprio = np.zeros((num_actions + 1, 3), dtype=np.float32)
    gripper = np.zeros(num_actions + 1, dtype=np.float32)
    return actions, proprio, gripper


def test_robust_mad_normalization_is_one_sided_and_finite() -> None:
    values = np.asarray([0.0, 1.0, 2.0, 3.0, 100.0])
    normalized = robust_mad_normalize(values, epsilon=1e-6, clip=7.0)

    expected_scale = 1.4826 * 1.0 + 1e-6
    np.testing.assert_allclose(normalized[:4], [0.0, 0.0, 0.0, 1.0 / expected_scale])
    assert normalized[-1] == 7.0
    np.testing.assert_array_equal(
        robust_mad_normalize(np.ones(6)),
        np.zeros(6),
    )


def test_gripper_transitions_are_forced_even_inside_nms_radius() -> None:
    actions, proprio, gripper = _constant_episode(12)
    # T+1 convention: changes g[2] -> g[3] and g[3] -> g[4] map to
    # action indices 2 and 3.  Both must survive, despite being adjacent.
    gripper[3] = 1.0
    config = EventMiningConfig(action_horizon=4, nms_radius=5)

    result = mine_episode_events(
        actions,
        proprio,
        gripper,
        config=config,
        start_mode="event",
    )

    np.testing.assert_array_equal(result.forced_gripper_indices, [2, 3])
    np.testing.assert_array_equal(result.candidate_indices, [2, 3])
    np.testing.assert_array_equal(result.window_starts, [0, 1])
    np.testing.assert_array_equal(result.windows, [[0, 4], [1, 5]])


def test_local_maxima_and_nms_keep_strong_separated_visual_changes() -> None:
    num_actions = 20
    actions, proprio, gripper = _constant_episode(num_actions)

    # Construct T+1 state features from known per-action increments.  The
    # varying background gives the trajectory a non-zero MAD; peaks at 5 and
    # 7 compete under NMS, while the peak at 15 is far enough to survive.
    increments = np.asarray(([0.0, 1.0, 2.0] * 7)[:num_actions])
    increments[[5, 7, 15]] = [10.0, 8.0, 9.0]
    semantic = np.concatenate([[0.0], np.cumsum(increments)])[:, None]
    config = EventMiningConfig(
        action_horizon=4,
        score_quantile=0.85,
        local_max_radius=1,
        nms_radius=3,
    )

    result = mine_episode_events(
        actions,
        proprio,
        gripper,
        semantic_features=semantic,
        config=config,
        start_mode="event",
    )

    np.testing.assert_array_equal(result.candidate_indices, [5, 15])
    np.testing.assert_array_equal(result.window_starts, [3, 13])
    assert result.scores[5] > result.scores[7]
    assert result.scores[15] > 0.0


def test_uniform_event_and_hybrid_starts_have_explicit_boundary_behavior() -> None:
    config = EventMiningConfig(action_horizon=4, uniform_stride=4)
    candidates = np.asarray([0, 5, 9], dtype=np.int64)

    uniform = select_window_starts(
        num_actions=10,
        candidate_indices=candidates,
        config=config,
        mode="uniform",
    )
    event = select_window_starts(
        num_actions=10,
        candidate_indices=candidates,
        config=config,
        mode="event",
    )
    hybrid = select_window_starts(
        num_actions=10,
        candidate_indices=candidates,
        config=config,
        mode="hybrid",
    )

    # Uniform appends the tail-aligned start.  Event centering clamps the
    # first and last candidate.  Hybrid is their deterministic sorted union.
    np.testing.assert_array_equal(uniform, [0, 4, 6])
    np.testing.assert_array_equal(event, [0, 3, 6])
    np.testing.assert_array_equal(hybrid, [0, 3, 4, 6])


def test_windows_are_exact_slices_of_original_actions_without_resampling() -> None:
    num_actions = 13
    horizon = 4
    actions = np.arange(num_actions * 2, dtype=np.float64).reshape(num_actions, 2)
    proprio = np.zeros((num_actions, 1), dtype=np.float64)
    gripper = np.zeros(num_actions, dtype=np.float64)
    gripper[6:] = 1.0
    config = EventMiningConfig(action_horizon=horizon, nms_radius=0)

    first = mine_episode_events(
        actions,
        proprio,
        gripper,
        config=config,
        start_mode="hybrid",
    )
    second = mine_episode_events(
        actions.copy(),
        proprio.copy(),
        gripper.copy(),
        config=config,
        start_mode="hybrid",
    )

    np.testing.assert_array_equal(first.candidate_indices, second.candidate_indices)
    np.testing.assert_array_equal(first.windows, second.windows)
    np.testing.assert_allclose(first.scores, second.scores)
    assert np.all(first.windows[:, 1] - first.windows[:, 0] == horizon)
    assert np.all(first.windows[:, 0] >= 0)
    assert np.all(first.windows[:, 1] <= num_actions)
    for start, stop in first.windows:
        np.testing.assert_array_equal(actions[start:stop], actions[np.arange(start, stop)])


def test_event_mode_can_return_no_windows_when_no_event_exists() -> None:
    actions, proprio, gripper = _constant_episode(8)
    result = mine_episode_events(
        actions,
        proprio,
        gripper,
        config=EventMiningConfig(action_horizon=4),
        start_mode="event",
    )

    assert result.candidate_indices.shape == (0,)
    assert result.window_starts.shape == (0,)
    assert result.windows.shape == (0, 2)


def test_short_episodes_and_misaligned_inputs_are_rejected() -> None:
    actions, proprio, gripper = _constant_episode(3)
    with pytest.raises(ValueError, match="shorter than action_horizon"):
        mine_episode_events(
            actions,
            proprio,
            gripper,
            config=EventMiningConfig(action_horizon=4),
        )

    actions, proprio, gripper = _constant_episode(8)
    with pytest.raises(ValueError, match=r"semantic_features must have T or T\+1"):
        mine_episode_events(
            actions,
            proprio,
            gripper,
            semantic_features=np.zeros((6, 4)),
            config=EventMiningConfig(action_horizon=4),
        )

    with pytest.raises(ValueError, match="unsupported start mode"):
        select_window_starts(
            num_actions=8,
            candidate_indices=[],
            config=EventMiningConfig(action_horizon=4),
            mode="random",  # type: ignore[arg-type]
        )
