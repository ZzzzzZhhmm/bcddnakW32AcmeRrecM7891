from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from fastwam.memory.manifest import sha256_file
from fastwam.models.warm.training_attestation import (
    SHARED_RECIPE_IGNORED_PATHS,
    V1_SHARED_RECIPE_IGNORED_PATHS,
    TrainingAttestationError,
    WarmTrainingAttestation,
    WarmTrainingRunContext,
    clean_git_commit,
    load_training_attestation,
    publish_training_attestation,
    prepare_formal_resume_lineage,
    sha256_training_state_tree,
    training_attestation_path,
    training_config_hashes,
    verify_training_attestation,
)


DIGEST = "a" * 64
COMMIT = "b" * 40


def _runtime(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "python_version": "3.11.9",
        "platform": "Linux-test",
        "accelerate_version": "1.7.0",
        "deepspeed_version": None,
        "torch_version": "2.7.0",
        "torch_cuda_version": None,
        "cudnn_version": None,
        "cuda_available": False,
        "gpu_count": 0,
        "gpu_devices": [],
        "distributed_type": "no",
        "deepspeed_config": None,
        "deepspeed_zero_stage": None,
        "float32_matmul_precision": "highest",
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": False,
        "deterministic_algorithms_enabled": False,
        "cublas_workspace_config": None,
    }
    value.update(updates)
    return value


def _value(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema": "warm.training-attestation",
        "version": 1,
        "checkpoint_sha256": DIGEST,
        "checkpoint_step": 17,
        "source_policy": "fixed_context_top1",
        "resolved_train_config_sha256": "1" * 64,
        "shared_recipe_sha256": "2" * 64,
        "shared_recipe_ignored_paths": list(V1_SHARED_RECIPE_IGNORED_PATHS),
        "root_seed": 42,
        "actual_global_step": 17,
        "actual_max_steps": 100,
        "optimizer_name": "AdamW",
        "optimizer_learning_rate": 1.0e-4,
        "optimizer_weight_decay": 1.0e-2,
        "optimizer_beta1": 0.9,
        "optimizer_beta2": 0.95,
        "optimizer_epsilon": 1.0e-8,
        "optimizer_amsgrad": False,
        "optimizer_wrapper_chain": [
            "accelerate.optimizer.AcceleratedOptimizer",
            "torch.optim.adamw.AdamW",
        ],
        "scheduler_type": "cosine",
        "scheduler_total_steps": 100,
        "scheduler_warmup_steps": 5,
        "scheduler_min_learning_rate": 1.0e-6,
        "scheduler_wrapper_chain": [
            "accelerate.scheduler.AcceleratedScheduler",
            "torch.optim.lr_scheduler.SequentialLR",
        ],
        "per_device_batch_size": 16,
        "gradient_accumulation_steps": 2,
        "world_size": 4,
        "effective_batch_size": 128,
        "mixed_precision": "bf16",
        "training_runtime": _runtime(),
        "train_source_contract_sha256": "3" * 64,
        "dev_source_contract_sha256": "4" * 64,
        "base_checkpoint_sha256": "5" * 64,
        "git_commit": COMMIT,
    }
    value.update(updates)
    return value


def _git_repository(path: Path) -> tuple[Path, str]:
    path.mkdir()
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "warm-tests@example.invalid"],
        cwd=path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "WARM Tests"], cwd=path, check=True
    )
    (path / "tracked.txt").write_text("immutable\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=path, check=True)
    subprocess.run(
        ["git", "commit", "-m", "fixture"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    return path, clean_git_commit(path)


def _context(repository: Path) -> WarmTrainingRunContext:
    return WarmTrainingRunContext.create(
        resolved_config={
            "output_dir": "/run/fixed",
            "seed": 42,
            "learning_rate": 1.0e-4,
            "model": {"source_policy": "fixed_context_top1", "width": 8},
            "wandb": {
                "enabled": False,
                "name": "fixed",
                "group": "ablation",
                "project": "warm",
            },
        },
        source_metadata={
            "source_policy": "fixed_context_top1",
            "train_source_contract_sha256": "3" * 64,
            "dev_source_contract_sha256": "4" * 64,
            "base_checkpoint_sha256": "5" * 64,
        },
        root_seed=42,
        actual_max_steps=100,
        optimizer_facts={
            "name": "AdamW",
            "learning_rate": 1.0e-4,
            "weight_decay": 1.0e-2,
            "betas": [0.9, 0.95],
            "epsilon": 1.0e-8,
            "amsgrad": False,
            "wrapper_chain": [
                "accelerate.optimizer.AcceleratedOptimizer",
                "torch.optim.adamw.AdamW",
            ],
        },
        scheduler_type="cosine",
        scheduler_warmup_steps=5,
        scheduler_min_learning_rate=1.0e-6,
        scheduler_wrapper_chain=[
            "accelerate.scheduler.AcceleratedScheduler",
            "torch.optim.lr_scheduler.SequentialLR",
        ],
        per_device_batch_size=16,
        gradient_accumulation_steps=2,
        world_size=4,
        mixed_precision="bf16",
        training_runtime=_runtime(),
        repository_root=repository,
    )


def test_attestation_round_trips_and_hashes_canonically() -> None:
    first = WarmTrainingAttestation.from_dict(_value())
    reversed_value = dict(reversed(list(_value().items())))
    second = WarmTrainingAttestation.from_dict(reversed_value)
    assert first == second
    assert first.sha256 == second.sha256
    assert json.loads(first.encode()) == first.to_dict()


def test_v2_attestation_requires_complete_resume_lineage(
    tmp_path: Path,
) -> None:
    repository, _ = _git_repository(tmp_path / "repository")
    fresh = _context(repository).build(
        checkpoint_sha256=DIGEST, actual_global_step=17
    )
    assert fresh.version == 2
    assert fresh.resume_step is None
    value = fresh.to_dict()
    value["resume_step"] = 10
    with pytest.raises(TrainingAttestationError, match="all null or all populated"):
        WarmTrainingAttestation.from_dict(value)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("checkpoint_sha256", "not-a-hash", "SHA-256"),
        ("checkpoint_step", True, "integer"),
        ("actual_global_step", 18, "checkpoint_step"),
        ("actual_max_steps", 16, "cannot exceed"),
        ("scheduler_total_steps", 99, "scheduler_total_steps"),
        ("effective_batch_size", 127, "effective_batch_size"),
        ("optimizer_learning_rate", float("nan"), "finite"),
        ("optimizer_beta2", 1.0, "beta"),
        ("mixed_precision", "tf32", "mixed_precision"),
    ],
)
def test_attestation_rejects_noncanonical_or_inconsistent_fields(
    field: str, value: object, message: str
) -> None:
    with pytest.raises((TypeError, TrainingAttestationError), match=message):
        WarmTrainingAttestation.from_dict(_value(**{field: value}))


def test_attestation_rejects_missing_and_extra_fields() -> None:
    missing = _value()
    missing.pop("root_seed")
    with pytest.raises(TrainingAttestationError, match="missing"):
        WarmTrainingAttestation.from_dict(missing)
    with pytest.raises(TrainingAttestationError, match="extra"):
        WarmTrainingAttestation.from_dict(_value(unbound="value"))


def test_attestation_binds_distributed_runtime_and_numeric_flags() -> None:
    value = _value()
    runtime = _runtime()
    runtime.pop("cuda_matmul_allow_tf32")
    value["training_runtime"] = runtime
    with pytest.raises(TrainingAttestationError, match="training_runtime fields"):
        WarmTrainingAttestation.from_dict(value)

    value = _value(
        training_runtime=_runtime(cudnn_deterministic="false")
    )
    with pytest.raises(TypeError, match="cudnn_deterministic"):
        WarmTrainingAttestation.from_dict(value)

    value = _value(
        training_runtime=_runtime(
            distributed_type="deepspeed",
            deepspeed_version="0.16.7",
            deepspeed_config={"zero_optimization": {"stage": 3}},
            deepspeed_zero_stage=2,
        )
    )
    with pytest.raises(TrainingAttestationError, match="disagrees"):
        WarmTrainingAttestation.from_dict(value)


def test_loader_rejects_semantically_valid_but_noncanonical_json(
    tmp_path: Path,
) -> None:
    path = tmp_path / "checkpoint.training.json"
    path.write_text(json.dumps(_value(), indent=2), encoding="utf-8")
    with pytest.raises(TrainingAttestationError, match="canonical JSON"):
        load_training_attestation(path)


def test_shared_recipe_ignores_only_declared_pair_display_fields() -> None:
    fixed = {
        "output_dir": "/fixed",
        "learning_rate": 1.0e-4,
        "model": {"source_policy": "fixed_context_top1", "width": 8},
        "wandb": {
            "enabled": False,
            "name": "fixed",
            "group": "fixed-group",
            "project": "warm",
        },
    }
    null = json.loads(json.dumps(fixed))
    null["output_dir"] = "/null"
    null["model"]["source_policy"] = "gaussian_null"
    null["wandb"]["name"] = "null"
    null["wandb"]["group"] = "null-group"
    null["wandb"]["project"] = "warm-null-display"
    fixed.update(
        {
            "resume": "/run/fixed/checkpoints/state/step_000010",
            "run_steps": 10,
            "log_every": 1,
            "save_every": 10,
            "eval_every": 0,
        }
    )
    null.update(
        {
            "resume": "/run/null/checkpoints/state/step_000010",
            "run_steps": None,
            "log_every": 10,
            "save_every": 500,
            "eval_every": 500,
        }
    )

    fixed_full, fixed_shared = training_config_hashes(fixed)
    null_full, null_shared = training_config_hashes(null)
    assert fixed_full != null_full
    assert fixed_shared == null_shared

    null["wandb"]["enabled"] = True
    assert training_config_hashes(null)[1] != fixed_shared
    null["wandb"]["enabled"] = False
    null["learning_rate"] = 2.0e-4
    assert training_config_hashes(null)[1] != fixed_shared


def test_context_requires_clean_git_and_publishes_bound_sidecar(
    tmp_path: Path,
) -> None:
    repository, commit = _git_repository(tmp_path / "repository")
    context = _context(repository)
    assert context.git_commit == commit

    checkpoint = tmp_path / "step_000017.pt"
    checkpoint.write_bytes(b"formal WARM checkpoint bytes")
    output, published = publish_training_attestation(
        checkpoint,
        context=context,
        actual_global_step=17,
    )
    assert output == training_attestation_path(checkpoint.resolve())
    assert output.name == "step_000017.training.json"
    assert published.checkpoint_sha256 == sha256_file(checkpoint)
    assert published.checkpoint_step == 17
    assert published.git_commit == commit
    assert load_training_attestation(output) == published
    assert verify_training_attestation(checkpoint) == published

    original_sidecar = output.read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        publish_training_attestation(
            checkpoint,
            context=context,
            actual_global_step=17,
        )
    assert output.read_bytes() == original_sidecar

    checkpoint.write_bytes(b"mutated after publication")
    with pytest.raises(TrainingAttestationError, match="does not match"):
        verify_training_attestation(checkpoint)


def test_formal_resume_upgrades_v1_parent_and_binds_exact_state_tree(
    tmp_path: Path,
) -> None:
    repository, commit = _git_repository(tmp_path / "repository")
    context = _context(repository)
    checkpoint_root = tmp_path / "run" / "checkpoints"
    weights = checkpoint_root / "weights" / "step_000017.pt"
    state = checkpoint_root / "state" / "step_000017"
    weights.parent.mkdir(parents=True)
    state.mkdir(parents=True)
    weights.write_bytes(b"legacy v1 weights")
    legacy = WarmTrainingAttestation.from_dict(
        _value(
            checkpoint_sha256=sha256_file(weights),
            git_commit=commit,
        )
    )
    training_attestation_path(weights).write_bytes(legacy.encode())
    (state / "trainer_state.json").write_text(
        json.dumps({"global_step": 17, "epoch": 1, "batch_in_epoch": 2}),
        encoding="utf-8",
    )
    nested = state / "pytorch_model"
    nested.mkdir()
    (nested / "mp_rank_00_model_states.pt").write_bytes(b"model state")
    (state / "scheduler.bin").write_bytes(b"scheduler state")

    lineage = prepare_formal_resume_lineage(
        state, current_context=context
    )
    assert lineage == {
        "parent_checkpoint_sha256": sha256_file(weights),
        "parent_training_attestation_sha256": sha256_file(
            training_attestation_path(weights)
        ),
        "resume_state_sha256": sha256_training_state_tree(state),
        "resume_step": 17,
    }
    continued = context.with_resume_lineage(lineage).build(
        checkpoint_sha256="f" * 64,
        actual_global_step=18,
    )
    assert continued.version == 2
    assert continued.parent_checkpoint_sha256 == sha256_file(weights)
    assert continued.resume_step == 17
    assert WarmTrainingAttestation.from_dict(continued.to_dict()) == continued

    (state / "trainer_state.json").write_text(
        json.dumps({"global_step": 16}), encoding="utf-8"
    )
    with pytest.raises(TrainingAttestationError, match="disagrees"):
        prepare_formal_resume_lineage(state, current_context=context)


def test_dirty_git_prevents_context_and_publication(tmp_path: Path) -> None:
    repository, _ = _git_repository(tmp_path / "repository")
    (repository / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(
        TrainingAttestationError,
        match=r"clean Git worktree: repository=.*changes=\?\? untracked\.txt",
    ):
        _context(repository)


def test_git_change_after_context_prevents_sidecar_publication(
    tmp_path: Path,
) -> None:
    repository, _ = _git_repository(tmp_path / "repository")
    context = _context(repository)
    checkpoint = tmp_path / "step_000017.pt"
    checkpoint.write_bytes(b"formal WARM checkpoint bytes")
    (repository / "tracked.txt").write_text("changed\n", encoding="utf-8")
    with pytest.raises(
        TrainingAttestationError,
        match=r"clean Git worktree: repository=.*changes= M tracked\.txt",
    ):
        publish_training_attestation(
            checkpoint,
            context=context,
            actual_global_step=17,
        )
    assert not training_attestation_path(checkpoint).exists()


def test_context_requires_both_train_and_dev_source_contracts(
    tmp_path: Path,
) -> None:
    repository, _ = _git_repository(tmp_path / "repository")
    metadata = {
        "source_policy": "fixed_context_top1",
        "train_source_contract_sha256": "3" * 64,
        "dev_source_contract_sha256": None,
        "base_checkpoint_sha256": "5" * 64,
    }
    with pytest.raises(TrainingAttestationError, match="dev_source_contract"):
        WarmTrainingRunContext.create(
            resolved_config={"seed": 42},
            source_metadata=metadata,
            root_seed=42,
            actual_max_steps=10,
            optimizer_facts={
                "name": "AdamW",
                "learning_rate": 1.0e-4,
                "weight_decay": 0.0,
                "betas": [0.9, 0.95],
                "epsilon": 1.0e-8,
                "amsgrad": False,
                "wrapper_chain": ["torch.optim.adamw.AdamW"],
            },
            scheduler_type="cosine",
            scheduler_warmup_steps=0,
            scheduler_min_learning_rate=1.0e-6,
            scheduler_wrapper_chain=["torch.optim.lr_scheduler.CosineAnnealingLR"],
            per_device_batch_size=1,
            gradient_accumulation_steps=1,
            world_size=1,
            mixed_precision="bf16",
            training_runtime=_runtime(),
            repository_root=repository,
        )


def test_trainer_and_model_expose_attestation_lifecycle() -> None:
    root = Path(__file__).resolve().parents[1]
    trainer = (root / "src/fastwam/trainer.py").read_text(encoding="utf-8")
    runtime = (root / "src/fastwam/runtime.py").read_text(encoding="utf-8")
    model = (
        root / "src/fastwam/models/warm/source_model.py"
    ).read_text(encoding="utf-8")
    attestation = (
        root / "src/fastwam/models/warm/training_attestation.py"
    ).read_text(encoding="utf-8")
    assert "_prepare_warm_training_attestation_context" in trainer
    assert trainer.index("self.accelerator.prepare(") < trainer.index(
        "self._prepare_warm_training_attestation_context()"
    )
    assert "capture_actual_optimizer_facts(self.optimizer)" in trainer
    assert "capture_actual_scheduler_chain(" in trainer
    assert "capture_training_runtime(self.accelerator)" in trainer
    assert "prepare_formal_resume_lineage(" in trainer
    assert "_verify_formal_resume_after_load" in trainer
    assert "publish_training_attestation(" in trainer
    assert "for path in (ckpt_path, sidecar_path)" in trainer
    assert "os.link(temporary, ckpt_path)" in trainer
    assert "os.replace(temporary, ckpt_path)" not in trainer
    assert "checkpoint_saved_this_step" in trainer
    assert "_publish_resolved_training_config(cfg)" in runtime
    assert "config.resume.step_" in runtime
    assert 'open(Path(cfg.output_dir) / "config.yaml", "w")' not in runtime
    assert "os.link(temporary, path)" in attestation
    assert "os.replace(temporary, path)" not in attestation
    assert "training_attestation_metadata" in model
    assert "_warm_loaded_base_checkpoint_sha256" in model


def test_full_retrospection_preserves_closed_source_attestation_schema() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "src/fastwam/models/warm/retrospection_model.py"
    ).read_text(encoding="utf-8")
    start = source.index("    def training_attestation_metadata(self)")
    method = source[start : source.index("\n\n\n__all__", start)]
    assert "return super().training_attestation_metadata()" in method
    assert 'value["retrospection_config_sha256"]' not in method


def test_dirty_debug_mode_is_explicit_and_unattested() -> None:
    root = Path(__file__).resolve().parents[1]
    trainer = (root / "src/fastwam/trainer.py").read_text(encoding="utf-8")
    train_config = (root / "configs/train.yaml").read_text(encoding="utf-8")
    acp_script = (root / "scripts/acp_warm_libero.sh").read_text(
        encoding="utf-8"
    )

    assert "allow_unattested_warm_checkpoints: false" in train_config
    assert "if self.allow_unattested_warm_checkpoints:" in trainer
    assert "saving WARM debug" in trainer
    assert (
        'ALLOW_DIRTY_WARM_TRAINING="${ALLOW_DIRTY_WARM_TRAINING:-false}"'
        in acp_script
    )
    assert '&& "${RUN_KIND}" == "train"' in acp_script
    assert '&& "${TASK_NAME}" == libero_warm_*' in acp_script
    assert (
        'WARM_OVERRIDES+=("allow_unattested_warm_checkpoints=true")'
        in acp_script
    )
