import numpy as np
import pytest

from fastwam.research.libero_table4 import goal_progress, make_plan, to_environment_actions
from fastwam.research.libero_legacy_compat import expanded_signature
from types import SimpleNamespace
from dataclasses import dataclass
from fastwam.research.libero_branches import plain_fields


@pytest.mark.parametrize("before,after,label", [
    ([False, False], [True, False], 1),
    ([True, False], [True, True], 1),
    ([True, False], [False, True], 0),
    ([False, False], [False, False], 0),
    ([True, True], [True, True], 0),
])
def test_independent_goal_progress(before, after, label):
    assert goal_progress(before, after) == label


def test_goal_vector_validation():
    with pytest.raises(ValueError):
        goal_progress([0], [1])
    with pytest.raises(ValueError):
        goal_progress([], [])


def test_plan_covers_complete_suite_without_outcome_selection():
    plan = make_plan()
    assert plan["tasks"] == list(range(10))
    assert len(plan["episodes"])*len(plan["frames"])*len(plan["tasks"]) == 200
    assert 49 not in plan["episodes"]
    with pytest.raises(ValueError):
        make_plan(frames=(13,))


def test_environment_gripper_transform_matches_legacy_order():
    class Evaluator:
        @staticmethod
        def _denormalize_action(action, processor):
            return np.array(action, dtype=np.float32)
    a = np.zeros((2, 32, 7), dtype=np.float32)
    a[0, :, -1] = 0.8
    a[1, :, -1] = 0.2
    result = to_environment_actions(a, None, Evaluator, binarize=True)
    assert np.all(result[0, :, -1] == -1)
    assert np.all(result[1, :, -1] == 1)
    np.testing.assert_array_equal(a[0, :, -1], np.full(32, .8, dtype=np.float32))


def test_legacy_committed_signature_matches_preview_coordinates():
    summary = SimpleNamespace(mean_displacement=np.arange(7), final_displacement=np.arange(7)+7,
                              terminal_gripper_values=np.array([.3]))
    result = expanded_signature(summary)
    np.testing.assert_array_equal(result[:14], np.arange(14))
    np.testing.assert_array_equal(result[14:20], np.zeros(6))
    assert result.shape == (21,) and result[-1] == .3
    summary.terminal_gripper_values = np.zeros(7)
    with pytest.raises(ValueError, match="shape mismatch"):
        expanded_signature(summary)


def test_slots_runtime_pending_actions_copied_without_live_model():
    @dataclass(slots=True)
    class Runtime:
        frame: int
        pending: list
        model: object
    runtime = Runtime(80, [np.ones(7)], object())
    saved = plain_fields(runtime)
    assert set(saved) == {"frame", "pending"}
    runtime.pending[0][:] = 2
    np.testing.assert_array_equal(saved["pending"][0], np.ones(7))
