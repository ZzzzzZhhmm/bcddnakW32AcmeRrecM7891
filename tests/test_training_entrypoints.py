from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from fastwam.training_config import validate_training_config


def _valid_config() -> dict[str, object]:
    return {
        "batch_size": 8,
        "num_workers": 4,
        "num_epochs": 10,
        "gradient_accumulation_steps": 4,
        "log_every": 10,
        "save_every": 500,
        "eval_every": 0,
        "eval_num_inference_steps": 10,
        "max_nonfinite_gradient_skips": 3,
        "seed": 42,
        "max_steps": None,
        "run_steps": None,
        "learning_rate": 1.0e-4,
        "weight_decay": 1.0e-2,
        "max_grad_norm": 1.0,
        "mixed_precision": "bf16",
        "lr_scheduler_type": "cosine",
        "wandb": {"enabled": False},
        "allow_unattested_warm_checkpoints": False,
        "model": {
            "_target_": "fastwam.runtime.create_warm_retrospection",
            "source_policy": "fixed_context_top1",
            "mot_checkpoint_mixed_attn": True,
        },
    }


def test_formal_four_gpu_training_config_is_valid() -> None:
    validate_training_config(_valid_config())


def test_legacy_config_uses_trainer_nonfinite_skip_default() -> None:
    config = _valid_config()
    del config["max_nonfinite_gradient_skips"]
    validate_training_config(config)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("batch_size",), 0, "batch_size"),
        (("gradient_accumulation_steps",), True, "gradient_accumulation_steps"),
        (("run_steps",), 0, "run_steps"),
        (("learning_rate",), float("nan"), "learning_rate"),
        (("max_grad_norm",), 0.0, "max_grad_norm"),
        (("mixed_precision",), "BF32", "mixed_precision"),
        (("lr_scheduler_type",), "linear", "lr_scheduler_type"),
        (("wandb", "enabled"), "false", "wandb.enabled"),
        (("allow_unattested_warm_checkpoints",), "false", "allow_unattested"),
        (
            ("model", "mot_checkpoint_mixed_attn"),
            "true",
            "mot_checkpoint_mixed_attn",
        ),
        (("model", "source_policy"), "gaussian_null", "complete WARM"),
    ],
)
def test_invalid_training_overrides_fail_before_model_allocation(
    path: tuple[str, ...], value: object, message: str
) -> None:
    config = deepcopy(_valid_config())
    target = config
    for key in path[:-1]:
        nested = target[key]
        assert isinstance(nested, dict)
        target = nested
    target[path[-1]] = value
    with pytest.raises((TypeError, ValueError), match=message):
        validate_training_config(config)


def test_distributed_launchers_forward_complete_topology() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in ("train_zero1.sh", "train_zero2.sh"):
        source = (root / "scripts" / name).read_text(encoding="utf-8")
        assert "TOTAL_PROCESSES=$((NPROC_PER_NODE * NUM_MACHINES))" in source
        assert '--num_processes "${TOTAL_PROCESSES}"' in source
        assert '--num_machines "${NUM_MACHINES}"' in source
        assert '--machine_rank "${MACHINE_RANK}"' in source
        assert '--main_process_ip "${MAIN_PROCESS_IP}"' in source
        assert '--main_process_port "${MAIN_PROCESS_PORT}"' in source


def test_acp_global_batch_and_runtime_cleanup_are_explicit() -> None:
    root = Path(__file__).resolve().parents[1]
    acp = (root / "scripts/acp_warm_libero.sh").read_text(encoding="utf-8")
    runtime = (root / "src/fastwam/runtime.py").read_text(encoding="utf-8")
    trainer = (root / "src/fastwam/trainer.py").read_text(encoding="utf-8")

    assert "RESOLVED_BATCH_SIZE * NPROC_PER_NODE * NNODES" in acp
    assert "validate_boolean WANDB_ENABLED" in acp
    assert "validate_source_policy_for_kind" in acp
    assert "shlex.split" in acp
    assert "EXTRA_ARGS=(${HYDRA_EXTRA_ARGS})" not in acp
    assert "validate_training_config(cfg)" in runtime
    assert "Wan22Trainer.create(" in runtime
    assert "except BaseException:" in runtime
    assert "trainer.close()" in runtime
    assert "accelerator.end_training()" in trainer
    assert "_run_main_process_step" in trainer
    assert "broadcast_object_list" in trainer
    assert "[done] no optimizer steps remain" in trainer
    assert "single-node orchestrator" in acp
