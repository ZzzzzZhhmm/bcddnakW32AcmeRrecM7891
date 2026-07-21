"""Narrow, attested compatibility repair for the step-019100 evaluator.

Python imports ``sitecustomize`` during interpreter startup when this directory
is placed first on ``PYTHONPATH``.  The ACP launcher enables the repair only
after verifying the exact SHA-256 of the known-buggy source file and records
this file's SHA-256 in the immutable evaluation evidence.

The training commit stored terminal gripper values compactly in committed
``ActionSummary`` signatures but expanded them to the full action width in an
online preview.  The numerical similarity is invariant to those inserted zero
coordinates; only the array widths differed.  Expanding committed signatures
restores the representation used by training and by the old preview code
without changing weights, actions, thresholds, or non-zero feature values.
"""

from __future__ import annotations

import os
import sys


_PATCH_ID = "action-summary-signature-v1"
_AFFECTED_TRAIN_COMMIT = "c4763a975298de6f00939360551616af7902d57a"


def _abort(message: str) -> None:
    # CPython reports and ignores ordinary exceptions raised by sitecustomize.
    # An explicitly requested attested repair must instead fail closed.
    sys.stderr.write(f"fatal evaluation compatibility error: {message}\n")
    sys.stderr.flush()
    os._exit(78)


if os.environ.get("WARM_EVAL_COMPAT_ACTION_SIGNATURE") == _PATCH_ID:
    training_commit = os.environ.get("WARM_EVAL_COMPAT_TRAIN_COMMIT", "")
    if training_commit != _AFFECTED_TRAIN_COMMIT:
        _abort(
            f"{_PATCH_ID} may only repair {_AFFECTED_TRAIN_COMMIT}, "
            f"not {training_commit or '<unset>'}"
        )

    import numpy as np

    from fastwam.memory.episode_memory import ActionSummary

    raw_indices = os.environ.get("WARM_EVAL_COMPAT_GRIPPER_INDICES", "")
    gripper_indices = tuple(
        int(value) for value in raw_indices.split(",") if value != ""
    )
    if gripper_indices != (6,):
        _abort(
            f"{_PATCH_ID} expects the attested LIBERO gripper index (6,), "
            f"got {gripper_indices}"
        )

    def _expanded_signature(self: ActionSummary) -> np.ndarray:
        action_dim = int(self.mean_displacement.shape[0])
        if self.final_displacement.shape != (action_dim,):
            raise RuntimeError("committed action-summary displacement shape changed")
        if self.terminal_gripper_values.shape != (len(gripper_indices),):
            raise RuntimeError("committed action-summary gripper shape changed")
        terminal = np.zeros((action_dim,), dtype=np.float64)
        terminal[list(gripper_indices)] = self.terminal_gripper_values
        return np.ascontiguousarray(
            np.concatenate(
                (
                    self.mean_displacement.astype(np.float64),
                    self.final_displacement.astype(np.float64),
                    terminal,
                )
            )
        )

    ActionSummary.signature = _expanded_signature
