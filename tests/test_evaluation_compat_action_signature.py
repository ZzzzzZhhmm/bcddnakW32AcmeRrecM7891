from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMPATIBILITY_DIR = (
    ROOT / "scripts" / "evaluation_compat" / "action_summary_signature_v1"
)
AFFECTED_COMMIT = "c4763a975298de6f00939360551616af7902d57a"


def test_compatibility_patch_expands_only_terminal_coordinates() -> None:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(
                (str(COMPATIBILITY_DIR), str(ROOT / "src"))
            ),
            "WARM_EVAL_COMPAT_ACTION_SIGNATURE": "action-summary-signature-v1",
            "WARM_EVAL_COMPAT_TRAIN_COMMIT": AFFECTED_COMMIT,
            "WARM_EVAL_COMPAT_GRIPPER_INDICES": "6",
        }
    )
    script = """
import numpy as np
from fastwam.memory.episode_memory import ActionSummary

summary = ActionSummary(
    start_frame=0,
    end_frame=4,
    step_count=4,
    mean_displacement=np.asarray([1, 2, 3, 4, 5, 6, 0], dtype=np.float32),
    final_displacement=np.asarray([2, 4, 6, 8, 10, 12, 0], dtype=np.float32),
    terminal_gripper_values=np.asarray([1], dtype=np.float32),
    gripper_transition_counts=np.asarray([[1, 0]], dtype=np.int64),
    curvature=0.0,
    repetition_similarity=0.0,
    repeated=False,
)
signature = summary.signature()
assert signature.shape == (21,)
np.testing.assert_array_equal(signature[:7], summary.mean_displacement)
np.testing.assert_array_equal(signature[7:14], summary.final_displacement)
np.testing.assert_array_equal(signature[14:20], np.zeros((6,)))
assert signature[20] == 1.0
"""

    subprocess.run([sys.executable, "-c", script], check=True, env=env)


def test_compatibility_patch_rejects_an_unrelated_training_commit() -> None:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(
                (str(COMPATIBILITY_DIR), str(ROOT / "src"))
            ),
            "WARM_EVAL_COMPAT_ACTION_SIGNATURE": "action-summary-signature-v1",
            "WARM_EVAL_COMPAT_TRAIN_COMMIT": "0" * 40,
            "WARM_EVAL_COMPAT_GRIPPER_INDICES": "6",
        }
    )

    result = subprocess.run(
        [sys.executable, "-c", "pass"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode != 0
    assert "may only repair" in result.stderr
