from __future__ import annotations

import os
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
COMPATIBILITY_DIR = (
    ROOT / "scripts" / "evaluation_compat" / "step019100_eval_v2"
)
AFFECTED_COMMIT = "c4763a975298de6f00939360551616af7902d57a"
PATCH_ID = "warm-step019100-eval-v2"


def test_compatibility_patch_expands_only_terminal_coordinates() -> None:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(
                (str(COMPATIBILITY_DIR), str(ROOT / "src"))
            ),
            "WARM_EVAL_COMPAT_ACTION_SIGNATURE": PATCH_ID,
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
            "WARM_EVAL_COMPAT_ACTION_SIGNATURE": PATCH_ID,
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


def _write_fake_numerical_modules(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "torch.py").write_text(
        """
__version__ = "2.7.1+cu128"

class _Version:
    cuda = "12.8"
version = _Version()

class _Matmul:
    allow_tf32 = False
    allow_fp16_reduced_precision_reduction = True
    allow_bf16_reduced_precision_reduction = False

class _CudaBackend:
    matmul = _Matmul()

class _Cudnn:
    enabled = False
    allow_tf32 = False
    benchmark = True
    deterministic = False
    @staticmethod
    def version():
        return 90701

class _Backends:
    cuda = _CudaBackend()
    cudnn = _Cudnn()
backends = _Backends()

_precision = "highest"
_deterministic = False
_warn_only = False

def set_float32_matmul_precision(value):
    global _precision
    _precision = value

def get_float32_matmul_precision():
    return _precision

def use_deterministic_algorithms(value, *, warn_only=False):
    global _deterministic, _warn_only
    _deterministic = bool(value)
    _warn_only = bool(warn_only)

def are_deterministic_algorithms_enabled():
    return _deterministic

def is_deterministic_algorithms_warn_only_enabled():
    return _warn_only
""",
        encoding="utf-8",
    )
    (root / "torchvision.py").write_text(
        '__version__ = "0.22.1+cu128"\n', encoding="utf-8"
    )
    (root / "transformers.py").write_text(
        '__version__ = "4.51.3"\n', encoding="utf-8"
    )


def _runtime_bridge_environment(
    contract_path: Path, fake_modules: Path
) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(
                (
                    str(COMPATIBILITY_DIR),
                    str(fake_modules),
                    str(ROOT / "src"),
                )
            ),
            "WARM_EVAL_COMPAT_ACTION_SIGNATURE": PATCH_ID,
            "WARM_EVAL_COMPAT_TRAIN_COMMIT": AFFECTED_COMMIT,
            "WARM_EVAL_COMPAT_GRIPPER_INDICES": "6",
            "WARM_EVAL_COMPAT_RUNTIME_BRIDGE_ACTIVE": "1",
            "WARM_EVAL_COMPAT_ENCODER_CONTRACT_PATH": str(contract_path),
            "WARM_EVAL_COMPAT_ENCODER_DEVICE": "cpu",
        }
    )
    return env


def _write_encoder_contract(
    path: Path, *, runtime_updates: dict[str, object] | None = None
) -> dict[str, object]:
    runtime: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": "2.7.1+cu128",
        "torchvision": "0.22.1+cu128",
        "transformers": "4.51.3",
        "cuda_runtime": "12.8",
        "cudnn": 90701,
        "float32_matmul_precision": "high",
        "cuda_matmul_allow_tf32": True,
        "cuda_matmul_allow_fp16_reduced_precision_reduction": False,
        "cuda_matmul_allow_bf16_reduced_precision_reduction": True,
        "cudnn_enabled": True,
        "cudnn_allow_tf32": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "deterministic_algorithms": True,
        "deterministic_warn_only": True,
        "cublas_workspace_config": None,
        "nvidia_tf32_override": None,
    }
    runtime.update(runtime_updates or {})
    path.write_text(
        json.dumps(
            {"runtime": runtime, "compute": {"device": "cpu"}},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return runtime


def test_runtime_bridge_allows_only_non_numerical_host_provenance(
    tmp_path: Path,
) -> None:
    contract_path = tmp_path / "encoder_contract.json"
    fake_modules = tmp_path / "fake_modules"
    _write_fake_numerical_modules(fake_modules)
    expected = _write_encoder_contract(
        contract_path,
        runtime_updates={"platform": "attested-m1-host-kernel"},
    )
    script = """
from fastwam.memory.runtime_fingerprint import current_encoder_runtime
runtime = current_encoder_runtime("cpu")
assert runtime["platform"] == "attested-m1-host-kernel"
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=_runtime_bridge_environment(contract_path, fake_modules),
    )

    assert result.returncode == 0, result.stderr
    assert "encoder_runtime_compatibility=" in result.stderr
    assert expected["platform"] in result.stderr


@pytest.mark.parametrize("field", ["numpy", "torch", "cuda_runtime"])
def test_runtime_bridge_rejects_numerical_software_mismatch(
    tmp_path: Path, field: str
) -> None:
    contract_path = tmp_path / "encoder_contract.json"
    fake_modules = tmp_path / "fake_modules"
    _write_fake_numerical_modules(fake_modules)
    _write_encoder_contract(contract_path, runtime_updates={field: "incompatible"})

    result = subprocess.run(
        [sys.executable, "-c", "pass"],
        check=False,
        capture_output=True,
        text=True,
        env=_runtime_bridge_environment(contract_path, fake_modules),
    )

    assert result.returncode != 0
    assert "numerical runtime remains incompatible" in result.stderr
    assert field in result.stderr
