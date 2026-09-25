"""Small CPU-only protocol functions for the LIBERO Table 4 collector."""
from __future__ import annotations

import numpy as np


def goal_progress(before, after):
    """A completed new BDDL subgoal, with all previously true goals retained.

    This is a deliberately narrow, independent outcome definition. Approaching
    an object without completing a goal is zero under this definition, not an
    unknown observation and not proof that the action is useless in general.
    """
    if not before or len(before) != len(after):
        raise ValueError("nonempty matching goal vectors required")
    if any(type(x) is not bool for x in [*before, *after]):
        raise ValueError("goal truth values must be bool")
    return int(all(not b or a for b, a in zip(before, after))
               and any(not b and a for b, a in zip(before, after)))


def to_environment_actions(normalized, processor, evaluator, *, binarize):
    """Exactly the legacy LIBERO executor's action and gripper conversion."""
    action = evaluator._denormalize_action(normalized, processor)
    action[..., -1] = 1.0 - 2.0 * action[..., -1]
    if binarize:
        action[..., -1] = np.sign(action[..., -1])
    if action.ndim != 3 or action.shape[1:] != (32, 7) or not np.isfinite(action).all():
        raise ValueError("LIBERO adapted proposals must be finite [K,32,7]")
    return action


def make_plan(*, tasks=tuple(range(10)), episodes=tuple(range(10)), frames=(80, 160)):
    if len(set(tasks)) != len(tasks) or any(t not in range(10) for t in tasks):
        raise ValueError("unique LIBERO-10 task ids required")
    if not episodes or len(set(episodes)) != len(episodes) or any(e < 0 for e in episodes):
        raise ValueError("unique nonnegative source episode ids required")
    if not frames or list(frames) != sorted(set(frames)) or any(f < 0 or f % 10 for f in frames):
        raise ValueError("increasing factual query frames must align with replanning")
    return dict(schema="warm.libero.table4.collection.v1", suite="libero_10",
                tasks=list(tasks), episodes=list(episodes), frames=list(frames),
                seed=3407, horizon=32, top_k=32, replan_steps=10,
                outcome="new_BDDL_goal_at_endpoint_without_losing_previously_true_goal",
                outcome_scope="32-step goal progress; not a general usefulness or full-task SR label",
                effect_projection="mean of four DINO spatial token differences",
                policy="legacy_LIBERO_step019100_full_retrospection",
                effect_head="trained historical-effect-plus-warped-action residual; executed proposal includes adaptation",
                bootstrap=10000, metrics_seed=3407,
                candidate_metrics=dict(delta_obs=1e-6, delta_pred=1e-6,
                                       magnitude_weight=.25, top_tolerance=1e-6))
