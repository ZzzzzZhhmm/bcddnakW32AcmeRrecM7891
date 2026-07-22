"""Full WARM policy for the pinned official RMBench evaluator.

The three public functions at the bottom are the interface imported by
``script/eval_policy.py``.  All persistent state lives inside
:class:`RMBenchWarmPolicy`; the official simulator checkout remains an
external, read-only dependency.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for _path in (PROJECT_ROOT, SRC_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from .policy_core import (  # noqa: E402
    JsonlEvidenceWriter,
    RecedingHorizonQueue,
    factual_cameras,
    factual_joint_state,
    resolve_task_bundle_paths,
    validate_warm_model_telemetry,
)
from fastwam.benchmarks.rmbench import RMBENCH_TASKS, task_by_name  # noqa: E402
from fastwam.benchmarks.rmbench_runtime import (  # noqa: E402
    build_rmbench_policy_runtime_projection,
)
from fastwam.datasets.lerobot.processors.fastwam_processor import (  # noqa: E402
    FastWAMProcessor,
)
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT  # noqa: E402
from fastwam.datasets.lerobot.utils.normalizer import (  # noqa: E402
    load_dataset_stats_from_json,
)
from fastwam.memory.manifest import (  # noqa: E402
    sha256_array,
    sha256_canonical_json,
    sha256_file,
    sha256_path_tree,
)
from fastwam.memory.online_episode_controller import (  # noqa: E402
    OnlineEpisodeController,
)
from fastwam.memory.online_episode_memory import (  # noqa: E402
    OnlineRetrospectiveEpisodeMemory,
)
from fastwam.memory.online_retrieval import (  # noqa: E402
    FrozenDinoOnlineRetriever,
)
from fastwam.memory.processor_contract import (  # noqa: E402
    extract_m1_robotwin_processor_recipe,
    load_m1_data_config,
    validate_processor_instance,
)
from fastwam.memory.robotwin_artifacts import (  # noqa: E402
    ROBOTWIN_ACTION_DIM,
    ROBOTWIN_GRIPPER_DIMS,
    RobotwinQposZScore,
)
from fastwam.memory.runtime_fingerprint import current_encoder_runtime  # noqa: E402
from fastwam.models.warm.online_contract import WarmOnlineRunContract  # noqa: E402
from fastwam.models.warm.source_contract import WarmSourceRunContract  # noqa: E402
from fastwam.models.warm.training_attestation import (  # noqa: E402
    verify_training_attestation,
)


logger = logging.getLogger(__name__)

_ABLATION_MODES = frozenset(
    {"full", "context_only", "source_only_no_consequence"}
)
_MEMORY_CORRUPTIONS = frozenset(
    {"clean", "wrong_event", "reversed_action", "phase_shift", "effect_mismatch"}
)
_EXPERIMENT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


def _none_like(value: Any) -> bool:
    return value is None or (
        isinstance(value, str) and value.strip().lower() in {"", "none", "null"}
    )


def _bool(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y"}:
            return True
        if normalized in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"{field} must be a boolean")


def _optional_float(value: Any) -> float | None:
    return None if _none_like(value) else float(value)


def _path(value: Any, *, field: str, directory: bool | None = None) -> Path:
    if _none_like(value):
        raise ValueError(f"{field} is required for full WARM evaluation")
    result = Path(os.path.expanduser(os.path.expandvars(str(value))))
    if not result.is_absolute():
        result = PROJECT_ROOT / result
    result = result.resolve()
    if not result.exists():
        raise FileNotFoundError(f"{field} does not exist: {result}")
    if directory is True and not result.is_dir():
        raise NotADirectoryError(f"{field} must be a directory: {result}")
    if directory is False and not result.is_file():
        raise FileNotFoundError(f"{field} must be a file: {result}")
    return result


def _output_path(value: Any, *, field: str) -> Path:
    if _none_like(value):
        raise ValueError(f"{field} is required for auditable RMBench evaluation")
    result = Path(os.path.expanduser(os.path.expandvars(str(value))))
    if not result.is_absolute():
        result = PROJECT_ROOT / result
    result = result.resolve()
    external_checkout = Path.cwd().resolve()
    try:
        result.relative_to(external_checkout)
    except ValueError:
        return result
    raise ValueError("WARM evidence must be outside the official RMBench checkout")


def _planned_file_path(value: Any, *, field: str) -> Path:
    """Resolve one not-yet-written artifact without weakening runtime checks.

    Contract-bundle construction must compose the exact policy configuration
    before the per-task contract exists.  Only that planned contract output is
    allowed to be absent; every model, bank, and processor input continues to
    go through :func:`_path` and must already exist.
    """

    if _none_like(value):
        raise ValueError(f"{field} is required for planned WARM evaluation")
    result = Path(os.path.expanduser(os.path.expandvars(str(value))))
    if not result.is_absolute():
        result = PROJECT_ROOT / result
    result = result.resolve()
    if result.exists() and not result.is_file():
        raise ValueError(f"{field} must name a regular file: {result}")
    if not result.parent.is_dir():
        raise FileNotFoundError(
            f"{field} parent directory does not exist: {result.parent}"
        )
    return result


def _json_snapshot(path: Path, *, label: str) -> tuple[Mapping[str, Any], str]:
    before = sha256_file(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label} JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    if sha256_file(path) != before:
        raise RuntimeError(f"{label} changed while it was loaded")
    return value, before


def _git_identity() -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("cannot attest the WARM Git checkout") from exc
    return commit, bool(status.strip())


def _model_dtype(value: Any) -> torch.dtype:
    precision = str(value).strip().lower()
    mapping = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    if precision not in mapping:
        raise ValueError("mixed_precision must be no, fp16, or bf16")
    return mapping[precision]


def _config_name(sim_cfg_path: Any, sim_cfg_name: Any) -> str:
    configs = (PROJECT_ROOT / "configs").resolve()
    if not _none_like(sim_cfg_path):
        path = Path(str(sim_cfg_path)).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path = path.resolve()
        try:
            return path.relative_to(configs).as_posix()
        except ValueError as exc:
            raise ValueError(f"sim_cfg_path must be under {configs}") from exc
    return "sim_rmbench.yaml" if _none_like(sim_cfg_name) else str(sim_cfg_name)


_ONLINE_ARG_TO_CONFIG = {
    "warm_online_contract_path": "contract_path",
    "warm_training_attestation_path": "training_attestation_path",
    "warm_training_run_contract_path": "training_run_contract_path",
    "warm_validation_run_contract_path": "validation_run_contract_path",
    "warm_base_checkpoint_path": "base_checkpoint_path",
    "warm_bank_directory": "bank_directory",
    "warm_normalizer_contract_path": "normalizer_contract_path",
    "warm_encoder_contract_path": "encoder_contract_path",
    "warm_camera_contract_path": "camera_contract_path",
    "warm_m1_data_config_path": "m1_data_config_path",
    "warm_dino_checkpoint_path": "dino_checkpoint_path",
    "warm_catalog_path": "catalog_path",
    "warm_audit_report_path": "audit_report_path",
}


def _compose_runtime_config(
    usr_args: Mapping[str, Any],
    *,
    checkpoint: Path,
    stats: Path,
    task_name: str,
    task_id: int,
    allow_missing_contract_path: bool = False,
) -> DictConfig:
    task_config = str(
        usr_args.get("sim_task") or "rmbench_warm_online_3cam384_full"
    )
    if task_config != "rmbench_warm_online_3cam384_full":
        raise ValueError(
            "official WARM evaluation requires sim_task="
            "rmbench_warm_online_3cam384_full"
        )
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    configs = (PROJECT_ROOT / "configs").resolve()
    with initialize_config_dir(version_base="1.3", config_dir=str(configs)):
        cfg = compose(
            config_name=_config_name(
                usr_args.get("sim_cfg_path"), usr_args.get("sim_cfg_name")
            ),
            overrides=[f"task={task_config}"],
        )
    with open_dict(cfg):
        cfg.ckpt = str(checkpoint)
        cfg.seed = int(usr_args.get("seed", 0))
        cfg.EVALUATION.dataset_stats_path = str(stats)
        cfg.EVALUATION.task_suite_name = "rmbench"
        cfg.EVALUATION.task_id = task_id
        cfg.EVALUATION.task_description = task_name
        online = cfg.EVALUATION.warm_online
        for argument, config_key in _ONLINE_ARG_TO_CONFIG.items():
            if argument == "warm_online_contract_path" and allow_missing_contract_path:
                configured = _planned_file_path(
                    usr_args.get(argument), field=argument
                )
            else:
                configured = _path(usr_args.get(argument), field=argument)
            online[config_key] = str(configured)
        online.evaluation_namespace = str(
            usr_args.get("warm_evaluation_namespace")
            or "warm-rmbench-full-v1"
        )
        online.top_k = int(usr_args.get("warm_top_k", 32))
        online.recent_event_capacity = int(
            usr_args.get("warm_recent_event_capacity", 6)
        )
        online.action_summary_capacity = int(
            usr_args.get("warm_action_summary_capacity", 2)
        )
        online.dino_device = str(usr_args.get("warm_dino_device") or "cuda")
        online.dino_batch_size = int(usr_args.get("warm_dino_batch_size", 1))
        online.experiment_id = str(usr_args.get("warm_experiment_id") or "")
        online.ablation_mode = str(usr_args.get("warm_ablation_mode") or "full")
        online.memory_corruption = str(
            usr_args.get("warm_memory_corruption") or "clean"
        )
        online.num_inference_steps = int(
            usr_args.get("num_inference_steps", cfg.eval_num_inference_steps)
        )
        cfg.EVALUATION.action_horizon = int(usr_args.get("action_horizon", 32))
        cfg.EVALUATION.replan_steps = int(usr_args.get("replan_steps", 10))
        cfg.model.retrospection.episode_action_chunk_size = int(
            cfg.EVALUATION.replan_steps
        )
        cfg.EVALUATION.num_inference_steps = int(
            usr_args.get("num_inference_steps", cfg.eval_num_inference_steps)
        )
        cfg.EVALUATION.sigma_shift = _optional_float(usr_args.get("sigma_shift"))
        cfg.EVALUATION.text_cfg_scale = float(usr_args.get("text_cfg_scale", 1.0))
        cfg.EVALUATION.negative_prompt = str(usr_args.get("negative_prompt") or "")
        cfg.EVALUATION.rand_device = str(usr_args.get("rand_device") or "cpu")
        cfg.EVALUATION.tiled = _bool(
            usr_args.get("tiled", False), field="tiled"
        )
        cfg.model.run_contract_path = str(
            _path(
                usr_args.get("warm_training_run_contract_path"),
                field="warm_training_run_contract_path",
            )
        )
        cfg.model.validation_run_contract_path = str(
            _path(
                usr_args.get("warm_validation_run_contract_path"),
                field="warm_validation_run_contract_path",
            )
        )
        cfg.model.base_checkpoint_path = str(
            _path(
                usr_args.get("warm_base_checkpoint_path"),
                field="warm_base_checkpoint_path",
            )
        )
        cfg.data.train.pretrained_norm_stats = str(stats)
    OmegaConf.resolve(cfg)
    return cfg


def _stable_stats(path: Path) -> tuple[Mapping[str, Any], str]:
    before = sha256_file(path)
    value = load_dataset_stats_from_json(str(path))
    if sha256_file(path) != before:
        raise RuntimeError("dataset statistics changed while they were loaded")
    if not isinstance(value, Mapping):
        raise TypeError("dataset statistics must decode to a mapping")
    return value, before


def _normalize_state(
    state: np.ndarray, processor: FastWAMProcessor
) -> torch.Tensor:
    state_meta = processor.shape_meta["state"]
    if len(state_meta) != 1:
        raise ValueError("RMBench requires exactly one merged state key")
    key = state_meta[0]["key"]
    batch = {
        "state": {key: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)}
    }
    batch = processor.action_state_transform(batch)
    batch = processor.normalizer.forward(batch)
    result = batch["state"][key]
    if tuple(result.shape) != (1, ROBOTWIN_ACTION_DIM):
        raise ValueError("normalized RMBench proprio must have shape [1,14]")
    return result


def _dino_dtype(path: Path, *, expected_sha256: str) -> torch.dtype:
    value, digest = _json_snapshot(path, label="encoder contract")
    if digest != expected_sha256:
        raise ValueError("encoder contract does not match online contract")
    compute = value.get("compute")
    if not isinstance(compute, Mapping):
        raise ValueError("encoder contract has no compute mapping")
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = compute.get("dtype")
    if dtype not in mapping:
        raise ValueError("unsupported encoder contract compute dtype")
    return mapping[dtype]


def _model_artifact(model: torch.nn.Module, key: str) -> Path:
    paths = getattr(model, "model_paths", None)
    if not isinstance(paths, Mapping) or _none_like(paths.get(key)):
        raise ValueError(f"loaded WARM model has no {key!r} artifact path")
    return _path(paths[key], field=f"model_paths.{key}")


def _query_dict(query_id: Any) -> dict[str, Any]:
    return {
        "dataset_id": str(query_id.dataset_id),
        "dataset_index": int(query_id.dataset_index),
        "episode_index": int(query_id.episode_index),
        "frame_index": int(query_id.frame_index),
    }


def _event_dict(event_id: Any | None) -> dict[str, Any] | None:
    if event_id is None:
        return None
    return {
        "dataset_id": str(event_id.dataset_id),
        "dataset_index": int(event_id.dataset_index),
        "episode_index": int(event_id.episode_index),
        "start_frame": int(event_id.start_frame),
    }


def _text_digest(value: str) -> str:
    return hashlib.sha256(b"warm.rmbench.instruction.v1\0" + value.encode()).hexdigest()


class RMBenchWarmPolicy:
    """One contract-bound, consequence-aligned WARM evaluation process."""

    def __init__(self, usr_args: Mapping[str, Any]) -> None:
        if not isinstance(usr_args, Mapping):
            raise TypeError("usr_args must be a mapping")
        self._usr_args = dict(usr_args)
        self.task_name = str(usr_args.get("task_name") or "")
        task_by_name(self.task_name)
        self.task_id = next(
            index for index, item in enumerate(RMBENCH_TASKS) if item.name == self.task_name
        )
        self.ablation_mode = str(usr_args.get("warm_ablation_mode") or "full")
        self.memory_corruption = str(
            usr_args.get("warm_memory_corruption") or "clean"
        )
        self.experiment_id = str(usr_args.get("warm_experiment_id") or "")
        if self.ablation_mode not in _ABLATION_MODES:
            raise ValueError(
                f"warm_ablation_mode must be one of {sorted(_ABLATION_MODES)}"
            )
        if self.memory_corruption not in _MEMORY_CORRUPTIONS:
            raise ValueError(
                "warm_memory_corruption must be one of "
                f"{sorted(_MEMORY_CORRUPTIONS)}"
            )
        if _EXPERIMENT_ID.fullmatch(self.experiment_id) is None:
            raise ValueError("warm_experiment_id is required and must be normalized")
        if self.ablation_mode != "full" and self.memory_corruption != "clean":
            raise ValueError(
                "memory corruption experiments require warm_ablation_mode=full"
            )
        contract_path, initial_states_path, task_definition_path = (
            resolve_task_bundle_paths(
                _path(
                    usr_args.get("warm_online_contract_path"),
                    field="warm_online_contract_path",
                ),
                task_name=self.task_name,
                experiment_id=self.experiment_id,
                official_root=Path.cwd(),
                initial_states_path=(
                    None
                    if _none_like(usr_args.get("warm_initial_states_path"))
                    else usr_args.get("warm_initial_states_path")
                ),
                task_definition_path=(
                    None
                    if _none_like(usr_args.get("warm_task_definition_path"))
                    else usr_args.get("warm_task_definition_path")
                ),
            )
        )
        self._usr_args["warm_online_contract_path"] = str(contract_path)
        self._usr_args["warm_initial_states_path"] = str(initial_states_path)
        self._usr_args["warm_task_definition_path"] = str(task_definition_path)
        usr_args = self._usr_args
        self.checkpoint = _path(
            usr_args.get("ckpt_setting"), field="ckpt_setting", directory=False
        )
        # A formal checkpoint can be many gigabytes.  Bind its bytes once and
        # reuse the digest across the contract and loaded-model checks.
        self.checkpoint_sha256 = sha256_file(self.checkpoint)
        self.stats_path = _path(
            usr_args.get("dataset_stats_path"),
            field="dataset_stats_path",
            directory=False,
        )
        self.initial_states_path = initial_states_path
        self.task_definition_path = task_definition_path
        if self.task_definition_path.name != f"{self.task_name}.py":
            raise ValueError("warm_task_definition_path does not match task_name")
        self.telemetry_path = _output_path(
            usr_args.get("warm_telemetry_path"), field="warm_telemetry_path"
        )

        self.cfg = _compose_runtime_config(
            usr_args,
            checkpoint=self.checkpoint,
            stats=self.stats_path,
            task_name=self.task_name,
            task_id=self.task_id,
        )
        if not _none_like(usr_args.get("warm_online_pair_contract_path")):
            raise ValueError(
                "full RMBench WARM forbids the source-only online pair contract"
            )
        if (
            str(self.cfg.EVALUATION.warm_online.mode) != "full_retrospection"
            or str(self.cfg.EVALUATION.warm_online.benchmark_profile) != "robotwin"
            or bool(self.cfg.EVALUATION.visualize_future_video)
            or bool(self.cfg.EVALUATION.use_action_ensembler)
        ):
            raise ValueError(
                "RMBench policy requires full_retrospection/robotwin and the "
                "single action-only fast path"
            )
        self.action_horizon = int(self.cfg.EVALUATION.action_horizon)
        self.replan_steps = int(self.cfg.EVALUATION.replan_steps)
        if self.action_horizon != 32:
            raise ValueError("official RMBench WARM requires action_horizon=32")
        self.queue = RecedingHorizonQueue(
            replan_steps=self.replan_steps, action_horizon=self.action_horizon
        )
        self.num_inference_steps = int(self.cfg.EVALUATION.num_inference_steps)
        self.sigma_shift = _optional_float(self.cfg.EVALUATION.sigma_shift)
        self.text_cfg_scale = float(self.cfg.EVALUATION.text_cfg_scale)
        self.negative_prompt = str(self.cfg.EVALUATION.negative_prompt)
        self.rand_device = str(self.cfg.EVALUATION.rand_device)
        self.tiled = bool(self.cfg.EVALUATION.tiled)
        self.timing_enabled = _bool(
            usr_args.get("timing_enabled", True), field="timing_enabled"
        )
        resolved_controls = self.cfg.EVALUATION.warm_online
        if (
            str(resolved_controls.experiment_id) != self.experiment_id
            or str(resolved_controls.ablation_mode) != self.ablation_mode
            or str(resolved_controls.memory_corruption) != self.memory_corruption
            or int(resolved_controls.num_inference_steps)
            != self.num_inference_steps
        ):
            raise ValueError(
                "resolved RMBench experiment controls differ from policy controls"
            )

        online_cfg = self.cfg.EVALUATION.warm_online
        contract_path = _path(online_cfg.contract_path, field="online contract")
        source_path = _path(
            online_cfg.training_run_contract_path, field="training source contract"
        )
        validation_path = _path(
            online_cfg.validation_run_contract_path,
            field="validation source contract",
        )
        attestation_path = _path(
            online_cfg.training_attestation_path, field="training attestation"
        )
        contract_json, self.contract_file_sha256 = _json_snapshot(
            contract_path, label="online run contract"
        )
        source_json, _ = _json_snapshot(source_path, label="training run contract")
        validation_json, _ = _json_snapshot(
            validation_path, label="validation run contract"
        )
        self.contract = WarmOnlineRunContract.from_dict(contract_json)
        self.source_contract = WarmSourceRunContract.from_dict(source_json)
        self.validation_contract = WarmSourceRunContract.from_dict(validation_json)
        self._validate_static_contracts(usr_args, attestation_path)

        stats, stats_sha = _stable_stats(self.stats_path)
        if stats_sha != self.contract.normalization_stats_sha256:
            raise ValueError("RMBench dataset stats do not match online contract")
        self.action_normalizer = RobotwinQposZScore.from_dataset_stats(stats)

        model_dtype = _model_dtype(usr_args.get("mixed_precision", "bf16"))
        device = str(usr_args.get("device") or "cuda")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("formal RMBench WARM evaluation requires CUDA")
        model_cfg = OmegaConf.create(
            OmegaConf.to_container(self.cfg.model, resolve=True)
        )
        model_cfg.load_text_encoder = True
        self.model = instantiate(model_cfg, model_dtype=model_dtype, device=device)
        self.model.load_checkpoint(str(self.checkpoint))
        self.model = self.model.to(device).eval()
        configure_experiment = getattr(
            self.model, "configure_online_experiment", None
        )
        if not callable(configure_experiment):
            raise ValueError(
                "full WARM model lacks configure_online_experiment capability"
            )
        configure_experiment(
            ablation_mode=self.ablation_mode,
            memory_corruption=self.memory_corruption,
            experiment_id=self.experiment_id,
        )

        self.processor: FastWAMProcessor = instantiate(
            self.cfg.data.train.processor
        ).eval()
        self.processor.set_normalizer_from_stats(stats)
        recipe, recipe_sha = load_m1_data_config(
            online_cfg.m1_data_config_path, profile="robotwin"
        )
        if recipe_sha != self.contract.m1_data_config_sha256:
            raise ValueError("M1 RoboTwin data config does not match online contract")
        resolved = OmegaConf.to_container(self.cfg, resolve=True)
        if not isinstance(resolved, Mapping):
            raise ValueError("resolved RMBench evaluation config must be a mapping")
        if extract_m1_robotwin_processor_recipe(resolved) != recipe:
            raise ValueError("online processor recipe differs from M1 RoboTwin recipe")
        validate_processor_instance(self.processor, recipe)
        self._validate_loaded_model(stats_sha)

        encoder_path = _path(online_cfg.encoder_contract_path, field="encoder contract")
        dino_device = str(online_cfg.dino_device)
        if sha256_canonical_json(current_encoder_runtime(dino_device)) != (
            self.contract.encoder_runtime_sha256
        ):
            raise ValueError("online DINO runtime differs from the M1 contract")
        self.retriever = FrozenDinoOnlineRetriever.from_artifacts(
            online_cfg.bank_directory,
            source_run_contract=self.source_contract,
            online_run_contract=self.contract,
            normalizer_contract_path=online_cfg.normalizer_contract_path,
            encoder_contract_path=encoder_path,
            camera_contract_path=online_cfg.camera_contract_path,
            normalization_stats_path=self.stats_path,
            catalog_path=online_cfg.catalog_path,
            audit_report_path=online_cfg.audit_report_path,
            dino_checkpoint_path=online_cfg.dino_checkpoint_path,
            processor=self.processor,
            dino_device=dino_device,
            dino_torch_dtype=_dino_dtype(
                encoder_path, expected_sha256=self.contract.encoder_contract_sha256
            ),
            dino_batch_size=int(online_cfg.dino_batch_size),
            benchmark_profile="robotwin",
        )
        self.model.bind_online_retriever(self.retriever)
        retrospection = getattr(self.model, "warm_retrospection_config", None)
        if retrospection is None:
            raise ValueError("RMBench full WARM checkpoint has no retrospection config")
        if (
            int(retrospection.context_dim) != 777
            or int(retrospection.action_dim) != 14
            or int(retrospection.action_horizon) != 32
            or int(retrospection.semantic_dim) != 768
            or int(retrospection.proprio_dim) != 14
            or int(retrospection.timing_dim) != 4
            or int(retrospection.episode_action_chunk_size) != self.replan_steps
            or int(retrospection.episode_action_summary_dim) != 46
        ):
            raise ValueError("WARM retrospection dimensions differ from RMBench")
        episode_memory = OnlineRetrospectiveEpisodeMemory(
            action_dim=14,
            action_horizon=32,
            semantic_dim=768,
            gripper_indices=ROBOTWIN_GRIPPER_DIMS,
            recent_event_capacity=int(online_cfg.recent_event_capacity),
            action_summary_capacity=int(online_cfg.action_summary_capacity),
            episode_namespace="rmbench-eval",
        )
        self.controller = OnlineEpisodeController(
            contract=self.contract,
            retriever=self.retriever,
            retrospective_episode_memory=episode_memory,
        )
        self.writer = JsonlEvidenceWriter(self.telemetry_path)
        self._next_episode_index = 0
        self._executed_policy_actions = 0
        self._instruction: str | None = None
        self._closed = False
        self.writer.append(self._header())
        atexit.register(self._atexit)

    def _validate_static_contracts(
        self, usr_args: Mapping[str, Any], attestation_path: Path
    ) -> None:
        contract = self.contract
        if contract.task_suite != "rmbench":
            raise ValueError("online contract task_suite must be 'rmbench'")
        if contract.task_id != self.task_id or contract.task_description != self.task_name:
            raise ValueError("online contract task identity differs from RMBench task")
        if contract.root_seed != int(usr_args.get("seed", 0)):
            raise ValueError("online contract root seed differs from official evaluator")
        if contract.action_horizon != 32 or contract.action_dim != 14:
            raise ValueError("online contract must bind RMBench H=32/D=14")
        if contract.top_k != int(usr_args.get("warm_top_k", 32)):
            raise ValueError("online top-k differs from contract")
        for field, minimum in (
            ("warm_recent_event_capacity", 2),
            ("warm_action_summary_capacity", 1),
        ):
            value = usr_args.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < minimum
            ):
                raise ValueError(f"{field} must be an integer >= {minimum}")
        if contract.source_policy != "fixed_context_top1":
            raise ValueError("full RMBench WARM requires fixed_context_top1")
        namespace = str(
            usr_args.get("warm_evaluation_namespace") or "warm-rmbench-full-v1"
        )
        if sha256_canonical_json({"evaluation_namespace": namespace}) != (
            contract.evaluation_namespace_sha256
        ):
            raise ValueError("RMBench evaluation namespace differs from contract")
        initial = np.load(self.initial_states_path, allow_pickle=False)
        if sha256_array(np.ascontiguousarray(initial)) != contract.initial_states_sha256:
            raise ValueError("RMBench seed protocol differs from online contract")
        if sha256_file(self.task_definition_path) != contract.bddl_sha256:
            raise ValueError("RMBench task implementation differs from online contract")
        if self.checkpoint_sha256 != contract.warm_checkpoint_sha256:
            raise ValueError("WARM checkpoint differs from online contract")
        if self.source_contract.sha256 != contract.training_run_contract_sha256:
            raise ValueError("training source contract differs from online contract")
        if self.validation_contract.sha256 != contract.validation_run_contract_sha256:
            raise ValueError("validation source contract differs from online contract")
        if self.source_contract.query_split != "train" or (
            self.validation_contract.query_split != "dev"
        ):
            raise ValueError("RMBench online evaluation requires train/dev contracts")
        attestation_file_sha = sha256_file(attestation_path)
        attestation = verify_training_attestation(self.checkpoint, attestation_path)
        if attestation_file_sha != contract.training_attestation_sha256:
            raise ValueError("training attestation differs from online contract")
        if (
            attestation.source_policy != contract.source_policy
            or attestation.checkpoint_sha256 != contract.warm_checkpoint_sha256
            or attestation.train_source_contract_sha256 != self.source_contract.sha256
            or attestation.dev_source_contract_sha256
            != self.validation_contract.sha256
            or attestation.shared_recipe_sha256 != contract.shared_training_recipe_sha256
            or attestation.training_runtime_sha256 != contract.training_runtime_sha256
            or attestation.git_commit != contract.git_commit
        ):
            raise ValueError("training attestation contradicts online contract")
        resolved = OmegaConf.to_container(self.cfg, resolve=True)
        if not isinstance(resolved, Mapping):
            raise ValueError("resolved RMBench config must be a mapping")
        runtime_projection = build_rmbench_policy_runtime_projection(
            resolved, usr_args
        )
        self.runtime_projection_sha256 = sha256_canonical_json(
            runtime_projection
        )
        if self.runtime_projection_sha256 != contract.resolved_eval_config_sha256:
            raise ValueError(
                "canonical RMBench policy runtime differs from online contract"
            )
        commit, dirty = _git_identity()
        if dirty or commit != contract.git_commit:
            raise ValueError("RMBench runtime requires the contracted clean Git commit")

    def _validate_loaded_model(self, stats_sha: str) -> None:
        contract = self.contract
        if not (
            getattr(self.model, "_warm_loaded_checkpoint_sha256", None)
            == self.checkpoint_sha256
            == contract.warm_checkpoint_sha256
        ):
            raise ValueError("loaded WARM model/checkpoint identity mismatch")
        if getattr(self.model, "warm_source_policy", None) != contract.source_policy:
            raise ValueError("loaded WARM model source policy mismatch")
        if float(getattr(self.model, "warm_memory_sigma", float("nan"))) != (
            contract.memory_sigma
        ):
            raise ValueError("loaded WARM model memory sigma mismatch")
        source = getattr(self.model, "warm_run_contract", None)
        validation = getattr(self.model, "warm_validation_run_contract", None)
        if source is None or source.sha256 != self.source_contract.sha256:
            raise ValueError("loaded WARM model training contract mismatch")
        if validation is None or validation.sha256 != self.validation_contract.sha256:
            raise ValueError("loaded WARM model validation contract mismatch")
        if stats_sha != self.source_contract.normalization_stats_sha256:
            raise ValueError("training source contract normalization mismatch")
        base = _path(
            self.cfg.EVALUATION.warm_online.base_checkpoint_path,
            field="base checkpoint",
            directory=False,
        )
        if sha256_file(base) != self.source_contract.base_checkpoint_sha256:
            raise ValueError("base checkpoint differs from training source contract")
        vae = _model_artifact(self.model, "vae")
        if sha256_file(vae) != contract.vae_checkpoint_sha256:
            raise ValueError("loaded VAE differs from online contract")
        text = _model_artifact(self.model, "text_encoder")
        if sha256_path_tree(text)[0] != contract.text_encoder_tree_sha256:
            raise ValueError("loaded text encoder differs from online contract")
        tokenizer = _model_artifact(self.model, "tokenizer")
        if sha256_path_tree(tokenizer)[0] != contract.tokenizer_tree_sha256:
            raise ValueError("loaded tokenizer differs from online contract")
        explicit_artifacts = (
            ("warm_vae_checkpoint_path", vae),
            ("warm_text_encoder_path", text),
            ("warm_tokenizer_path", tokenizer),
        )
        # The formal policy-runtime projection binds these explicit locations;
        # they must resolve to the exact loaded objects, not equal bytes at a
        # different path.
        for argument, loaded in explicit_artifacts:
            configured = self._usr_args.get(argument)
            if _path(configured, field=argument) != loaded:
                raise ValueError(f"{argument} differs from the loaded model artifact")

    def _header(self) -> dict[str, Any]:
        return {
            "schema": "warm.rmbench-online-evidence",
            "version": 1,
            "kind": "header",
            "task_name": self.task_name,
            "task_id": self.task_id,
            "online_run_contract_sha256": self.contract.sha256,
            "online_contract_file_sha256": self.contract_file_sha256,
            "training_run_contract_sha256": self.source_contract.sha256,
            "validation_run_contract_sha256": self.validation_contract.sha256,
            "checkpoint_sha256": self.contract.warm_checkpoint_sha256,
            "root_seed": self.contract.root_seed,
            "source_policy": self.contract.source_policy,
            "action_horizon": self.action_horizon,
            "replan_steps": self.replan_steps,
            "benchmark_profile": "robotwin",
            "experiment_id": self.experiment_id,
            "ablation_mode": self.ablation_mode,
            "memory_corruption": self.memory_corruption,
            "experiment_controls_sha256": sha256_canonical_json(
                {
                    "experiment_id": self.experiment_id,
                    "ablation_mode": self.ablation_mode,
                    "memory_corruption": self.memory_corruption,
                    "num_inference_steps": self.num_inference_steps,
                }
            ),
            "policy_runtime_projection_sha256": self.runtime_projection_sha256,
        }

    def reset(self) -> None:
        if self._closed:
            raise RuntimeError("cannot reset a closed RMBench WARM policy")
        self.seal_active(reason="next_reset", success=False)
        self.queue.clear()
        self._executed_policy_actions = 0
        self._instruction = None
        self.controller.begin_episode(self._next_episode_index)
        self.writer.append(
            {
                "schema": "warm.rmbench-online-evidence",
                "version": 1,
                "kind": "episode_begin",
                "episode_index": self._next_episode_index,
            }
        )
        self._next_episode_index += 1

    def _instruction_for(self, task_env: Any) -> str:
        value = str(task_env.get_instruction())
        if not value or value.strip() != value or "\x00" in value:
            raise ValueError("RMBench instruction must be normalized and non-empty")
        if self._instruction is None:
            self._instruction = value
        elif value != self._instruction:
            raise RuntimeError("RMBench instruction changed inside one episode")
        return value

    def _validated_model_telemetry(
        self, value: Any, *, online_step: Any
    ) -> dict[str, Any]:
        """Validate experiment semantics before evidence is made durable."""

        return validate_warm_model_telemetry(
            value,
            online_step=online_step,
            experiment_id=self.experiment_id,
            ablation_mode=self.ablation_mode,
            memory_corruption=self.memory_corruption,
            online_contract_sha256=self.contract.sha256,
            training_run_contract_sha256=self.source_contract.sha256,
            validation_run_contract_sha256=self.validation_contract.sha256,
            bank_manifest_sha256=self.contract.bank_manifest_sha256,
            bank_content_sha256=self.contract.bank_content_sha256,
            memory_sigma=self.contract.memory_sigma,
        )

    def _replan(self, observation: Mapping[str, Any], task_env: Any) -> None:
        frame_index = self._executed_policy_actions
        query_id = self.controller.issue_query_id(frame_index)
        instruction = self._instruction_for(task_env)
        prompt = DEFAULT_PROMPT.format(task=instruction)
        raw_cameras = factual_cameras(observation)
        proprio = _normalize_state(factual_joint_state(observation), self.processor)
        start = time.perf_counter()
        online_step = self.retriever.retrieve(
            query_id,
            raw_cameras,
            task_description=self.task_name,
            prompt=prompt,
            proprio=proprio,
        )
        history_kwargs, history_evidence = self.controller.retrospective_history_kwargs()
        infer_kwargs = {
            "online_step": online_step,
            "negative_prompt": self.negative_prompt,
            "text_cfg_scale": self.text_cfg_scale,
            "num_inference_steps": self.num_inference_steps,
            "sigma_shift": self.sigma_shift,
            "rand_device": self.rand_device,
            "tiled": self.tiled,
            **history_kwargs,
        }
        with torch.no_grad():
            output = self.model.infer_action(**infer_kwargs)
        factual_update = self.controller.commit_factual_replan_observation(
            frame_index=frame_index, model_output=output
        )
        action_tensor = output.get("action")
        if action_tensor is None:
            raise RuntimeError("WARM inference returned no action")
        if hasattr(action_tensor, "detach"):
            model_chunk = action_tensor.detach().float().cpu().numpy()
        else:
            model_chunk = np.asarray(action_tensor, dtype=np.float32)
        if model_chunk.ndim == 3 and model_chunk.shape[0] == 1:
            model_chunk = model_chunk[0]
        model_chunk = np.ascontiguousarray(model_chunk, dtype=np.float32)
        if model_chunk.shape != (32, 14) or not np.isfinite(model_chunk).all():
            raise RuntimeError("WARM action output must be finite shape [32,14]")
        environment_chunk = self.action_normalizer.denormalize(model_chunk)
        round_trip = self.action_normalizer.normalize(
            environment_chunk, fail_on_clip=True
        )
        if not np.allclose(round_trip, model_chunk, rtol=1e-5, atol=2e-6):
            raise RuntimeError("RMBench action normalization round trip drifted")
        self.queue.publish(round_trip, environment_chunk)
        model_telemetry = self._validated_model_telemetry(
            output.get("warm_online_telemetry"), online_step=online_step
        )
        candidates = [
            {
                "rank": rank,
                "valid": bool(online_step.candidate_valid_mask[rank]),
                "bank_row": int(online_step.bank_rows[rank]),
                "event_id": _event_dict(online_step.event_ids[rank]),
                "cosine_score": float(online_step.cosine_scores[rank]),
            }
            for rank in range(len(online_step.event_ids))
        ]
        evidence = {
            "schema": "warm.rmbench-online-evidence",
            "version": 1,
            "kind": "replan",
            "episode_index": int(query_id.episode_index),
            "frame_index": frame_index,
            "query_id": _query_dict(query_id),
            "instruction_sha256": _text_digest(instruction),
            "bound_step_sha256": online_step.step_sha256,
            "context_key_sha256": online_step.context_key_sha256,
            "candidate_payload_sha256": online_step.candidate_payload_sha256,
            "raw_camera_sha256": dict(online_step.raw_camera_sha256),
            "processed_camera_sha256": dict(online_step.processed_camera_sha256),
            "model_input_sha256": online_step.model_input_sha256,
            "proprio_sha256": online_step.proprio_sha256,
            "candidate_count": int(sum(online_step.candidate_valid_mask)),
            "candidates": candidates,
            "history_before_replan": history_evidence,
            "factual_update_after_replan": factual_update,
            "model_action_chunk_sha256": sha256_array(model_chunk),
            "environment_action_chunk_sha256": sha256_array(environment_chunk),
            "model": model_telemetry,
        }
        if self.timing_enabled:
            evidence["pipeline_s"] = float(time.perf_counter() - start)
        self.writer.append(evidence)

    def step(self, task_env: Any, observation: Mapping[str, Any]) -> None:
        if self.controller.active_episode_index is None:
            raise RuntimeError("official evaluator must call reset_model before eval")
        if not self.queue:
            self._replan(observation, task_env)
        queued = self.queue.pop()
        task_env.take_action(queued.environment_space, action_type="qpos")
        # Record only after the simulator accepted the command.
        self.controller.note_executed_action(
            queued.environment_space,
            model_space_action=queued.model_space,
        )
        self.writer.append(
            {
                "schema": "warm.rmbench-online-evidence",
                "version": 1,
                "kind": "executed_action",
                "episode_index": self.controller.active_episode_index,
                "frame_index": self._executed_policy_actions,
                "environment_action_sha256": sha256_array(
                    queued.environment_space
                ),
                "model_action_sha256": sha256_array(queued.model_space),
            }
        )
        self._executed_policy_actions += 1
        if bool(getattr(task_env, "eval_success", False)):
            self.seal_active(reason="success", success=True)

    def seal_active(self, *, reason: str, success: bool) -> None:
        if self.controller.active_episode_index is None:
            return
        episode_index = self.controller.active_episode_index
        evidence = self.controller.end_episode()
        self.queue.clear()
        self.writer.append(
            {
                "schema": "warm.rmbench-online-evidence",
                "version": 1,
                "kind": "episode_end",
                "episode_index": episode_index,
                "reason": str(reason),
                "success": bool(success),
                "executed_policy_actions": self._executed_policy_actions,
                "episode_memory": evidence,
            }
        )

    def close(self) -> None:
        if self._closed:
            return
        self.seal_active(reason="process_exit", success=False)
        self.writer.close()
        self._closed = True

    def _atexit(self) -> None:
        try:
            self.close()
        except Exception:
            logger.exception("failed to seal RMBench WARM evidence at process exit")


def encode_obs(observation: Mapping[str, Any]) -> Mapping[str, Any]:
    """Keep the official factual observation unmodified until validation."""

    if not isinstance(observation, Mapping):
        raise TypeError("official RMBench observation must be a mapping")
    return observation


def get_model(usr_args: Mapping[str, Any]) -> RMBenchWarmPolicy:
    """Official RMBench model-construction signature."""

    return RMBenchWarmPolicy(usr_args)


def eval(TASK_ENV: Any, model: RMBenchWarmPolicy, observation: Mapping[str, Any]) -> None:
    """Official RMBench one-policy-command evaluation signature."""

    model.step(TASK_ENV, encode_obs(observation))


def reset_model(model: RMBenchWarmPolicy) -> None:
    """Official RMBench per-episode reset signature."""

    model.reset()


__all__ = ["RMBenchWarmPolicy", "encode_obs", "eval", "get_model", "reset_model"]
