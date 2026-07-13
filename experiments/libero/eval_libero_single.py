import hashlib
import inspect
import json
import logging
import os
import subprocess
import sys
import time
from uuid import uuid4
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import hydra
import numpy as np
import torch
from accelerate import PartialState
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

# try:
#     import rootutils

#     rootutils.setup_root(__file__, indicator=".python-version", pythonpath=True)
# except ModuleNotFoundError:
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.libero.libero_utils import (
    LIBERO_ENV_RESOLUTION,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    invert_gripper_action,
    quat2axisangle,
    save_prediction_video,
    save_rollout_video,
)
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.utils.pytorch_utils import set_global_seed
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from libero.libero import benchmark
from action_ensembler import ActionEnsembler

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("max", lambda x: max(x))
OmegaConf.register_new_resolver("split", lambda s, idx: s.split("/")[int(idx)])

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


@dataclass(slots=True)
class WarmOnlineEvalRuntime:
    """Validated M2.1 online state shared by all replans in one task."""

    contract: Any
    pair_contract: Any
    pair_side: str
    source_contract: Any
    validation_source_contract: Any
    retriever: Any | None
    retrospective_episode_memory: Any | None
    null_image_adapter: Any | None
    normalization_stats_loaded_sha256: str
    pair_contract_file_sha256: str | None
    pair_contract_path: Path | None
    parity_report_file_sha256: str | None
    parity_report_path: Path | None
    m1_data_config_path: Path
    training_attestation_file_sha256: str
    training_attestation_path: Path
    training_runtime_sha256: str
    shared_training_recipe_sha256: str
    model: Any
    checkpoint_path: Path
    bddl_path: Path
    git_commit: str
    _active_episode_index: int | None = field(default=None, init=False, repr=False)
    _last_frame_index: int | None = field(default=None, init=False, repr=False)
    _seen_episode_indices: set[int] = field(
        default_factory=set, init=False, repr=False
    )
    _executed_actions_since_replan: list[np.ndarray] = field(
        default_factory=list, init=False, repr=False
    )
    _executed_environment_actions_since_replan: list[np.ndarray] = field(
        default_factory=list, init=False, repr=False
    )

    @property
    def source_policy(self) -> str:
        return str(self.contract.source_policy)

    def pair_identity(self) -> dict[str, str]:
        """Return the standalone comparison identity for result records."""

        if self.pair_contract is None:
            return {
                "pair_contract_sha256": self.contract.sha256,
                "comparison_kind": "full_retrospection_single_checkpoint",
                "side": "full_retrospection",
            }
        return {
            "pair_contract_sha256": self.pair_contract.sha256,
            "comparison_kind": self.pair_contract.comparison_kind,
            "side": self.pair_side,
        }

    def attest_pair_contract_file(self) -> None:
        from fastwam.memory.manifest import sha256_file

        if self.pair_contract_path is not None:
            if sha256_file(self.pair_contract_path) != self.pair_contract_file_sha256:
                raise RuntimeError("online pair contract changed during evaluation")
        if self.parity_report_path is not None:
            if sha256_file(self.parity_report_path) != self.parity_report_file_sha256:
                raise RuntimeError("online parity report changed during evaluation")
        if sha256_file(self.m1_data_config_path) != self.contract.m1_data_config_sha256:
            raise RuntimeError("M1 data config changed during evaluation")
        if sha256_file(self.training_attestation_path) != (
            self.training_attestation_file_sha256
        ):
            raise RuntimeError("training attestation changed during evaluation")
        loaded_checkpoint_sha256 = getattr(
            self.model, "_warm_loaded_checkpoint_sha256", None
        )
        current_checkpoint_sha256 = sha256_file(self.checkpoint_path)
        if not (
            loaded_checkpoint_sha256
            == current_checkpoint_sha256
            == self.contract.warm_checkpoint_sha256
        ):
            raise RuntimeError(
                "loaded model, checkpoint bytes, and online contract diverged"
            )

    def begin_episode(self, episode_index: int) -> None:
        if isinstance(episode_index, bool) or not isinstance(episode_index, int):
            raise TypeError("episode_index must be a non-negative integer")
        if episode_index < 0:
            raise ValueError("episode_index must be a non-negative integer")
        if episode_index in self._seen_episode_indices:
            raise ValueError(f"episode_index {episode_index} was already evaluated")
        if self.retriever is not None:
            self.retriever.begin_episode(episode_index)
        if self.retrospective_episode_memory is not None:
            self.retrospective_episode_memory.begin_episode(episode_index)
        self._executed_actions_since_replan.clear()
        self._executed_environment_actions_since_replan.clear()
        self._seen_episode_indices.add(episode_index)
        self._active_episode_index = episode_index
        self._last_frame_index = None

    def issue_query_id(self, frame_index: int) -> Any:
        if isinstance(frame_index, bool) or not isinstance(frame_index, int):
            raise TypeError("frame_index must be a non-negative integer")
        if frame_index < 0:
            raise ValueError("frame_index must be a non-negative integer")
        if self._active_episode_index is None:
            raise RuntimeError("begin_episode() is required before online replanning")
        if self._last_frame_index is not None and frame_index <= self._last_frame_index:
            raise ValueError("online frame_index must increase strictly within an episode")
        if self.retriever is not None:
            query_id = self.retriever.make_query_id(frame_index)
        else:
            from fastwam.memory.online_retrieval import make_online_query_id

            query_id = make_online_query_id(
                self.contract, self._active_episode_index, frame_index
            )
        self._last_frame_index = frame_index
        return query_id

    def retrospective_history_kwargs(self) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Return factual history strictly preceding the current replan."""

        if self.retrospective_episode_memory is None:
            return {}, None
        pending = (
            None
            if not self._executed_actions_since_replan
            else np.stack(self._executed_actions_since_replan, axis=0)
        )
        history = self.retrospective_episode_memory.history_inputs(
            executed_actions_since_previous=pending
        )
        if history is None:
            return {}, None
        return history.model_kwargs(), history.evidence()

    def note_executed_action(
        self,
        action: Any,
        *,
        model_space_action: Any,
    ) -> None:
        """Record exactly one command after it was sent to the simulator."""

        if self.retrospective_episode_memory is None:
            return
        array = np.asarray(action, dtype=np.float32)
        if array.ndim != 1 or not array.size or not np.isfinite(array).all():
            raise ValueError("executed WARM action must be one finite vector")
        model_array = np.asarray(model_space_action, dtype=np.float32)
        if model_array.shape != array.shape or not np.isfinite(model_array).all():
            raise ValueError(
                "model-space executed WARM action must match the environment command"
            )
        self._executed_environment_actions_since_replan.append(
            np.ascontiguousarray(array)
        )
        self._executed_actions_since_replan.append(
            np.ascontiguousarray(model_array)
        )

    def commit_factual_replan_observation(
        self,
        *,
        frame_index: int,
        model_output: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Commit model-certified current real features after one inference."""

        if self.retrospective_episode_memory is None:
            return None
        payload = model_output.get("warm_factual_observation")
        if not isinstance(payload, Mapping):
            raise RuntimeError(
                "full WARM inference returned no warm_factual_observation"
            )
        actions: np.ndarray | None
        if self._executed_actions_since_replan:
            actions = np.stack(self._executed_actions_since_replan, axis=0)
        else:
            actions = None
        evidence = self.retrospective_episode_memory.record_factual_observation(
            frame_index=frame_index,
            factual_payload=payload,
            executed_actions_since_previous=actions,
        )
        if self._executed_environment_actions_since_replan:
            from fastwam.memory.manifest import sha256_array

            exact = np.stack(
                self._executed_environment_actions_since_replan, axis=0
            )
            evidence["executed_environment_prefix_sha256"] = sha256_array(exact)
            evidence["executed_environment_prefix_count"] = int(exact.shape[0])
        else:
            evidence["executed_environment_prefix_sha256"] = None
            evidence["executed_environment_prefix_count"] = 0
        self._executed_actions_since_replan.clear()
        self._executed_environment_actions_since_replan.clear()
        return evidence

    def end_episode(self) -> dict[str, Any] | None:
        if self.retrospective_episode_memory is None:
            return None
        evidence = self.retrospective_episode_memory.end_episode()
        # A terminal prefix may have no subsequent observation embedding.  It
        # is reported rather than paired with a fabricated feature write.
        evidence["unpaired_terminal_action_count"] = len(
            self._executed_actions_since_replan
        )
        if self._executed_environment_actions_since_replan:
            from fastwam.memory.manifest import sha256_array

            terminal = np.stack(
                self._executed_environment_actions_since_replan, axis=0
            )
            evidence["unpaired_terminal_environment_actions_sha256"] = (
                sha256_array(terminal)
            )
        else:
            evidence["unpaired_terminal_environment_actions_sha256"] = None
        self._executed_actions_since_replan.clear()
        self._executed_environment_actions_since_replan.clear()
        return evidence

    def result_header(self) -> dict[str, Any]:
        self.attest_pair_contract_file()
        return {
            "schema": "warm.libero-online-evaluation-header",
            "version": 2,
            **self.pair_identity(),
            "online_run_contract_sha256": self.contract.sha256,
            "training_run_contract_sha256": self.source_contract.sha256,
            "validation_run_contract_sha256": (
                self.validation_source_contract.sha256
            ),
            "source_policy": self.source_policy,
            "runtime_attestation": {
                "git_commit": self.git_commit,
                "git_dirty": False,
                "normalization_stats_loaded_sha256": (
                    self.normalization_stats_loaded_sha256
                ),
                "pair_contract_file_sha256": self.pair_contract_file_sha256,
                "parity_report_file_sha256": self.parity_report_file_sha256,
                "m1_data_config_sha256": self.contract.m1_data_config_sha256,
                "training_attestation_file_sha256": (
                    self.training_attestation_file_sha256
                ),
                "shared_training_recipe_sha256": (
                    self.shared_training_recipe_sha256
                ),
                "training_runtime_sha256": self.training_runtime_sha256,
                "model_loaded_checkpoint_sha256": (
                    self.contract.warm_checkpoint_sha256
                ),
                "bddl_sha256_after_environment_load": self.contract.bddl_sha256,
            },
            "contract": self.contract.to_dict(),
            **(
                {"pair_contract": self.pair_contract.to_dict()}
                if self.pair_contract is not None
                else {"pair_contract": None}
            ),
        }


def _read_json_mapping(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {label} JSON at {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _required_online_path(online_cfg: DictConfig, key: str) -> Path:
    value = online_cfg.get(key)
    if value is None or not str(value).strip():
        raise ValueError(f"EVALUATION.warm_online.{key} must be set")
    path = Path(os.path.expanduser(os.path.expandvars(str(value)))).resolve()
    if not path.exists():
        raise FileNotFoundError(f"EVALUATION.warm_online.{key} does not exist: {path}")
    return path


def _resolved_eval_config_sha256(cfg: DictConfig) -> str:
    from fastwam.memory.manifest import sha256_canonical_json

    value = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(value, Mapping):
        raise ValueError("Resolved Hydra evaluation config must be a mapping")
    return sha256_canonical_json(value)


def _git_identity(repository: Path) -> tuple[str, bool]:
    """Return the exact repository revision and dirty state for a formal run."""

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("cannot attest the WARM Git checkout") from exc
    return commit, bool(status.strip())


def _load_dataset_stats_stable(path: Path) -> tuple[Mapping[str, Any], str]:
    """Load normalization statistics while closing the replacement window."""

    from fastwam.memory.manifest import sha256_file

    before = sha256_file(path)
    stats = load_dataset_stats_from_json(str(path))
    after = sha256_file(path)
    if after != before:
        raise RuntimeError(
            "dataset normalization statistics changed while they were loaded"
        )
    if not isinstance(stats, Mapping):
        raise TypeError("dataset normalization statistics must decode to a mapping")
    return stats, after


def _dino_torch_dtype_from_encoder_contract(
    encoder_contract_path: Path,
    *,
    expected_sha256: str,
) -> torch.dtype:
    """Use the exact DINO compute dtype that produced the immutable M1 bank."""

    from fastwam.memory.manifest import sha256_file

    before = sha256_file(encoder_contract_path)
    if before != expected_sha256:
        raise ValueError("encoder contract does not match online run contract")
    value = _read_json_mapping(encoder_contract_path, label="encoder contract")
    after = sha256_file(encoder_contract_path)
    if after != before:
        raise RuntimeError("encoder contract changed while its DINO dtype was read")
    compute = value.get("compute")
    if not isinstance(compute, Mapping):
        raise ValueError("encoder contract has no compute mapping")
    dtype_name = compute.get("dtype")
    dtype_by_name = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in dtype_by_name:
        raise ValueError(
            "encoder contract compute.dtype must be float32, float16, or bfloat16"
        )
    return dtype_by_name[dtype_name]


def _build_null_image_adapter(
    *,
    processor: FastWAMProcessor,
    camera_contract_path: Path,
    expected_sha256: str,
    configured_concat_mode: str,
) -> Any:
    """Build the null image path from the same factual camera contract.

    This deliberately reads no event bank, feature encoder, DINO checkpoint,
    catalog, or audit report.  It only proves that the baseline image adapter
    has the exact camera order and resize/quantization recipe used by fixed.
    """

    from fastwam.memory.manifest import sha256_file
    from fastwam.memory.server_feature_encoders import FastWAMImageAdapter

    before = sha256_file(camera_contract_path)
    if before != expected_sha256:
        raise ValueError("camera contract does not match online run contract")
    value = _read_json_mapping(camera_contract_path, label="camera contract")
    source_keys = value.get("source_camera_keys")
    mapping = value.get("processor_camera_mapping")
    if not isinstance(source_keys, list) or not source_keys:
        raise ValueError("camera contract source_camera_keys must be non-empty")
    if not isinstance(mapping, Mapping) or set(mapping) != set(source_keys):
        raise ValueError(
            "camera contract processor mapping must cover every source camera"
        )
    processor_keys = tuple(mapping[key] for key in source_keys)
    image_meta = tuple(processor.shape_meta["images"])
    configured_keys = tuple(meta["key"] for meta in image_meta)
    if processor_keys != configured_keys:
        raise ValueError(
            "null processor camera order differs from factual camera contract"
        )
    if processor.is_train is not False or int(processor.num_output_cameras) != len(
        processor_keys
    ):
        raise ValueError(
            "null processor must be in eval mode with the contracted camera count"
        )
    camera_signature = tuple(
        (str(meta.get("key")), tuple(int(value) for value in meta.get("shape", ())))
        for meta in image_meta
    )
    expected_signature = tuple(
        (camera_key, (3, 224, 224)) for camera_key in processor_keys
    )
    if camera_signature != expected_signature:
        raise ValueError(
            "null processor camera shapes differ from factual camera contract"
        )
    try:
        processor.normalizer
    except (AttributeError, ValueError) as exc:
        raise ValueError("null processor has no installed normalizer") from exc
    if value.get("concat_mode") != "horizontal" or configured_concat_mode != (
        "horizontal"
    ):
        raise ValueError("formal M2.1 online evaluation requires horizontal cameras")
    exact_recipe = {
        "decoded_range": [0.0, 1.0],
        "baseline_quantization": "validated_0_1_times_255_to_uint8",
        "per_camera_size": [224, 224],
        "vae_model_range": [-1.0, 1.0],
    }
    for field_name, expected in exact_recipe.items():
        if value.get(field_name) != expected:
            raise ValueError(
                f"camera contract {field_name} does not match the M1 recipe"
            )
    if sha256_file(camera_contract_path) != before:
        raise RuntimeError("camera contract changed while the null adapter was built")
    return FastWAMImageAdapter(processor, processor_keys, "horizontal")


def _query_id_dict(query_id: Any) -> dict[str, Any]:
    return {
        "dataset_id": query_id.dataset_id,
        "dataset_index": int(query_id.dataset_index),
        "episode_index": int(query_id.episode_index),
        "frame_index": int(query_id.frame_index),
    }


def _online_prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(
        b"warm.online-prompt.v1\0" + prompt.encode("utf-8")
    ).hexdigest()


def _event_id_dict(event_id: Any | None) -> dict[str, Any] | None:
    if event_id is None:
        return None
    return {
        "dataset_id": event_id.dataset_id,
        "dataset_index": int(event_id.dataset_index),
        "episode_index": int(event_id.episode_index),
        "start_frame": int(event_id.start_frame),
    }


def _model_artifact_path(model: torch.nn.Module, key: str) -> Path:
    model_paths = getattr(model, "model_paths", None)
    if not isinstance(model_paths, Mapping):
        raise ValueError("Online WARM evaluation requires model.model_paths")
    raw = model_paths.get(key)
    if raw is None or not str(raw).strip():
        raise ValueError(f"Online WARM evaluation has no resolved model path for {key}")
    path = Path(os.path.expanduser(os.path.expandvars(str(raw)))).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Resolved model artifact does not exist for {key}: {path}")
    return path


def _validate_online_pair_membership(
    *,
    pair_contract: Any,
    online_contract: Any,
    resolved_eval_config_sha256: str,
    checkpoint_sha256: str,
    training_attestation_sha256: str,
    git_commit: str,
    git_dirty: bool,
) -> str:
    """Prove that this rollout is one declared side of the formal pair."""

    if online_contract.source_policy == "fixed_context_top1":
        pair_side = "fixed"
        expected_online_contract_sha256 = (
            pair_contract.fixed_online_run_contract_sha256
        )
        expected_resolved_config_sha256 = (
            pair_contract.fixed_resolved_eval_config_sha256
        )
        expected_checkpoint_sha256 = pair_contract.fixed_warm_checkpoint_sha256
        expected_training_attestation_sha256 = (
            pair_contract.fixed_training_attestation_sha256
        )
    elif online_contract.source_policy == "gaussian_null":
        pair_side = "gaussian_null"
        expected_online_contract_sha256 = (
            pair_contract.gaussian_null_online_run_contract_sha256
        )
        expected_resolved_config_sha256 = (
            pair_contract.gaussian_null_resolved_eval_config_sha256
        )
        expected_checkpoint_sha256 = (
            pair_contract.gaussian_null_warm_checkpoint_sha256
        )
        expected_training_attestation_sha256 = (
            pair_contract.gaussian_null_training_attestation_sha256
        )
    else:
        raise ValueError(
            "formal online pair has no side for source policy "
            f"{online_contract.source_policy!r}"
        )

    if expected_online_contract_sha256 != online_contract.sha256:
        raise ValueError("online run contract is not the declared pair side")
    if expected_resolved_config_sha256 != resolved_eval_config_sha256:
        raise ValueError("resolved evaluation config is not the declared pair side")
    if expected_checkpoint_sha256 != checkpoint_sha256:
        raise ValueError("WARM checkpoint is not the declared pair side")
    if expected_training_attestation_sha256 != training_attestation_sha256:
        raise ValueError("training attestation is not the declared pair side")
    if git_dirty or git_commit != pair_contract.git_commit:
        raise ValueError(
            "runtime Git checkout does not match the clean online pair contract"
        )
    if online_contract.git_commit != pair_contract.git_commit:
        raise ValueError("online run contract and pair contract disagree on Git commit")
    return pair_side


def _load_warm_online_runtime(
    cfg: DictConfig,
    *,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    dataset_stats_path: Path,
    task: Any,
    initial_states: Any,
    action_horizon: int,
    loaded_dataset_stats_sha256: str,
) -> WarmOnlineEvalRuntime | None:
    online_cfg = cfg.EVALUATION.get("warm_online")
    if online_cfg is None or not bool(online_cfg.get("enabled", False)):
        return None

    from libero.libero import get_libero_path

    from fastwam.memory.manifest import (
        sha256_array,
        sha256_canonical_json,
        sha256_file,
        sha256_path_tree,
    )
    from fastwam.memory.processor_contract import (
        extract_m1_libero_processor_recipe,
        load_m1_data_config,
        validate_processor_instance,
    )
    from fastwam.memory.runtime_fingerprint import current_encoder_runtime
    from fastwam.models.warm.online_contract import WarmOnlineRunContract
    from fastwam.models.warm.online_pair_contract import WarmOnlinePairContract
    from fastwam.models.warm.parity_report import (
        validate_passing_online_parity_report,
    )
    from fastwam.models.warm.source_contract import WarmSourceRunContract
    from fastwam.models.warm.training_attestation import (
        verify_training_attestation,
    )

    contract_path = _required_online_path(online_cfg, "contract_path")
    online_mode = str(online_cfg.get("mode", "source_only"))
    if online_mode not in {"source_only", "full_retrospection"}:
        raise ValueError(
            "EVALUATION.warm_online.mode must be source_only or full_retrospection"
        )
    full_retrospection = online_mode == "full_retrospection"
    pair_contract_path = (
        None
        if full_retrospection
        else _required_online_path(online_cfg, "pair_contract_path")
    )
    parity_report_path = (
        None
        if full_retrospection
        else _required_online_path(online_cfg, "parity_report_path")
    )
    m1_data_config_path = _required_online_path(
        online_cfg, "m1_data_config_path"
    )
    training_attestation_path = _required_online_path(
        online_cfg, "training_attestation_path"
    )
    source_path = _required_online_path(online_cfg, "training_run_contract_path")
    validation_source_path = _required_online_path(
        online_cfg, "validation_run_contract_path"
    )
    contract = WarmOnlineRunContract.from_dict(
        _read_json_mapping(contract_path, label="online run contract")
    )
    pair_contract_file_sha256 = None
    pair_contract = None
    parity_report_file_sha256 = None
    if not full_retrospection:
        assert pair_contract_path is not None and parity_report_path is not None
        pair_contract_file_sha256 = sha256_file(pair_contract_path)
        pair_contract = WarmOnlinePairContract.from_dict(
            _read_json_mapping(pair_contract_path, label="online pair contract")
        )
        if sha256_file(pair_contract_path) != pair_contract_file_sha256:
            raise RuntimeError("online pair contract changed while it was read")
        parity_report_file_sha256 = sha256_file(parity_report_path)
        if parity_report_file_sha256 != pair_contract.parity_report_sha256:
            raise ValueError("online parity report does not match the pair contract")
        parity_report = _read_json_mapping(
            parity_report_path, label="online parity report"
        )
        validate_passing_online_parity_report(
            parity_report,
            fixed_online_contract=contract,
            expected_online_contract_sha256=(
                pair_contract.fixed_online_run_contract_sha256
            ),
            expected_resolved_eval_config_sha256=(
                pair_contract.fixed_resolved_eval_config_sha256
            ),
        )
        if sha256_file(parity_report_path) != parity_report_file_sha256:
            raise RuntimeError("online parity report changed while it was read")
    m1_processor_recipe, m1_data_config_sha256 = load_m1_data_config(
        m1_data_config_path
    )
    if m1_data_config_sha256 != contract.m1_data_config_sha256:
        raise ValueError("M1 data config does not match the online run contract")
    resolved_config_value = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(resolved_config_value, Mapping):
        raise ValueError("resolved evaluation config must be a mapping")
    if extract_m1_libero_processor_recipe(resolved_config_value) != (
        m1_processor_recipe
    ):
        raise ValueError("rollout processor recipe differs from M1 preprocessing")
    validate_processor_instance(processor, m1_processor_recipe)
    if sha256_file(m1_data_config_path) != m1_data_config_sha256:
        raise RuntimeError("M1 data config changed while it was validated")
    configured_runtime_device = str(
        online_cfg.get("dino_device", _resolve_eval_device(cfg))
    )
    if sha256_canonical_json(
        current_encoder_runtime(configured_runtime_device)
    ) != contract.encoder_runtime_sha256:
        raise ValueError(
            "rollout numerical runtime differs from M1 encoding/parity runtime"
        )
    source = WarmSourceRunContract.from_dict(
        _read_json_mapping(source_path, label="training run contract")
    )
    validation_source = WarmSourceRunContract.from_dict(
        _read_json_mapping(
            validation_source_path, label="validation run contract"
        )
    )
    if source.sha256 != contract.training_run_contract_sha256:
        raise ValueError("Online and training run contracts do not match")
    if validation_source.sha256 != contract.validation_run_contract_sha256:
        raise ValueError("Online and validation run contracts do not match")
    if source.query_split != "train" or validation_source.query_split != "dev":
        raise ValueError("Online evaluation requires train/dev source contracts")

    checkpoint_path = Path(
        os.path.expanduser(os.path.expandvars(str(cfg.ckpt)))
    ).resolve()
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if checkpoint_sha256 != contract.warm_checkpoint_sha256:
        raise ValueError("WARM checkpoint does not match online run contract")
    if getattr(model, "_warm_loaded_checkpoint_sha256", None) != checkpoint_sha256:
        raise ValueError(
            "in-memory model checkpoint identity does not match the current bytes"
        )
    training_attestation_file_sha256 = sha256_file(training_attestation_path)
    training_attestation = verify_training_attestation(
        checkpoint_path, training_attestation_path
    )
    if sha256_file(training_attestation_path) != training_attestation_file_sha256:
        raise RuntimeError("training attestation changed while it was verified")
    if training_attestation_file_sha256 != contract.training_attestation_sha256:
        raise ValueError("training attestation does not match online run contract")
    if (
        training_attestation.source_policy != contract.source_policy
        or training_attestation.checkpoint_sha256 != checkpoint_sha256
        or training_attestation.train_source_contract_sha256
        != contract.training_run_contract_sha256
        or training_attestation.dev_source_contract_sha256
        != contract.validation_run_contract_sha256
        or training_attestation.shared_recipe_sha256
        != contract.shared_training_recipe_sha256
        or training_attestation.training_runtime_sha256
        != contract.training_runtime_sha256
        or training_attestation.git_commit != contract.git_commit
    ):
        raise ValueError("training attestation contradicts online run contract")
    stats_sha = sha256_file(dataset_stats_path)
    if stats_sha != loaded_dataset_stats_sha256:
        raise RuntimeError(
            "dataset normalization statistics changed after processor setup"
        )
    if stats_sha != contract.normalization_stats_sha256:
        raise ValueError("dataset stats do not match online run contract")
    if stats_sha != source.normalization_stats_sha256:
        raise ValueError("dataset stats do not match training run contract")

    resolved_eval_config_sha256 = _resolved_eval_config_sha256(cfg)
    if resolved_eval_config_sha256 != contract.resolved_eval_config_sha256:
        raise ValueError("resolved evaluation config does not match online run contract")
    namespace = str(online_cfg.get("evaluation_namespace", ""))
    if sha256_canonical_json({"evaluation_namespace": namespace}) != (
        contract.evaluation_namespace_sha256
    ):
        raise ValueError("evaluation namespace does not match online run contract")
    if int(cfg.seed) != contract.root_seed:
        raise ValueError("root seed does not match online run contract")
    if str(cfg.EVALUATION.task_suite_name) != contract.task_suite:
        raise ValueError("task suite does not match online run contract")
    if int(cfg.EVALUATION.task_id) != contract.task_id:
        raise ValueError("task id does not match online run contract")
    if task.language != contract.task_description:
        raise ValueError("exact LIBERO task language does not match online run contract")
    if int(online_cfg.get("top_k")) != contract.top_k:
        raise ValueError("online top-k does not match online run contract")
    if bool(cfg.EVALUATION.get("visualize_future_video", False)):
        raise ValueError(
            "formal online WARM evaluation is action-only and forbids future-video inference"
        )
    if bool(cfg.EVALUATION.get("use_action_ensembler", False)):
        raise ValueError(
            "formal source-only online WARM evaluation forbids action ensembling"
        )
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    if replan_steps <= 0 or replan_steps > action_horizon:
        raise ValueError(
            "formal online WARM replan_steps must be in [1, action_horizon]"
        )

    git_commit, git_dirty = _git_identity(project_root)
    if git_dirty or git_commit != contract.git_commit:
        raise ValueError(
            "runtime Git checkout does not match the clean online run contract"
        )
    if full_retrospection:
        if contract.source_policy != "fixed_context_top1":
            raise ValueError(
                "full_retrospection requires the fixed-context retrieval policy"
            )
        pair_side = "full_retrospection"
    else:
        assert pair_contract is not None
        pair_side = _validate_online_pair_membership(
            pair_contract=pair_contract,
            online_contract=contract,
            resolved_eval_config_sha256=resolved_eval_config_sha256,
            checkpoint_sha256=checkpoint_sha256,
            training_attestation_sha256=training_attestation_file_sha256,
            git_commit=git_commit,
            git_dirty=git_dirty,
        )
        expected_training_attestation_sha256 = (
            pair_contract.fixed_training_attestation_sha256
            if pair_side == "fixed"
            else pair_contract.gaussian_null_training_attestation_sha256
        )
        if (
            training_attestation_file_sha256
            != expected_training_attestation_sha256
            or training_attestation.shared_recipe_sha256
            != pair_contract.shared_training_recipe_sha256
            or training_attestation.training_runtime_sha256
            != pair_contract.shared_training_runtime_sha256
        ):
            raise ValueError(
                "training attestation does not match the declared online pair side"
            )

    raw_initial_states = np.ascontiguousarray(np.asarray(initial_states))
    if sha256_array(raw_initial_states) != contract.initial_states_sha256:
        raise ValueError("LIBERO initial states do not match online run contract")
    bddl_path = (
        Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    ).resolve()
    if sha256_file(bddl_path) != contract.bddl_sha256:
        raise ValueError("LIBERO BDDL does not match online run contract")

    if action_horizon != contract.action_horizon:
        raise ValueError("evaluation action horizon does not match online run contract")
    action_dim = int(processor.action_output_dim)
    if action_dim != contract.action_dim:
        raise ValueError("processor action dimension does not match online run contract")
    if getattr(model, "warm_source_policy", None) != contract.source_policy:
        raise ValueError("model source policy does not match online run contract")
    if float(getattr(model, "warm_memory_sigma", float("nan"))) != (
        contract.memory_sigma
    ):
        raise ValueError("model memory sigma does not match online run contract")
    model_source = getattr(model, "warm_run_contract", None)
    if model_source is None or model_source.sha256 != source.sha256:
        raise ValueError("model training contract does not match online run contract")
    model_validation_source = getattr(model, "warm_validation_run_contract", None)
    if (
        model_validation_source is None
        or model_validation_source.sha256 != validation_source.sha256
    ):
        raise ValueError("model validation contract does not match online run contract")

    vae_path = _model_artifact_path(model, "vae")
    if not vae_path.is_file() or sha256_file(vae_path) != contract.vae_checkpoint_sha256:
        raise ValueError("loaded VAE checkpoint does not match online run contract")
    text_path = _model_artifact_path(model, "text_encoder")
    if sha256_path_tree(text_path)[0] != contract.text_encoder_tree_sha256:
        raise ValueError("loaded text encoder does not match online run contract")
    tokenizer_path = _model_artifact_path(model, "tokenizer")
    if sha256_path_tree(tokenizer_path)[0] != contract.tokenizer_tree_sha256:
        raise ValueError("loaded tokenizer does not match online run contract")

    retriever = None
    retrospective_episode_memory = None
    null_image_adapter = None
    if contract.source_policy == "fixed_context_top1":
        from fastwam.memory.online_retrieval import FrozenDinoOnlineRetriever

        encoder_contract_path = _required_online_path(
            online_cfg, "encoder_contract_path"
        )
        dino_torch_dtype = _dino_torch_dtype_from_encoder_contract(
            encoder_contract_path,
            expected_sha256=contract.encoder_contract_sha256,
        )

        retriever = FrozenDinoOnlineRetriever.from_artifacts(
            _required_online_path(online_cfg, "bank_directory"),
            source_run_contract=source,
            online_run_contract=contract,
            normalizer_contract_path=_required_online_path(
                online_cfg, "normalizer_contract_path"
            ),
            encoder_contract_path=encoder_contract_path,
            camera_contract_path=_required_online_path(online_cfg, "camera_contract_path"),
            normalization_stats_path=dataset_stats_path,
            catalog_path=_required_online_path(online_cfg, "catalog_path"),
            audit_report_path=_required_online_path(online_cfg, "audit_report_path"),
            dino_checkpoint_path=_required_online_path(online_cfg, "dino_checkpoint_path"),
            processor=processor,
            dino_device=str(online_cfg.get("dino_device", _resolve_eval_device(cfg))),
            dino_torch_dtype=dino_torch_dtype,
            dino_batch_size=int(online_cfg.get("dino_batch_size", 1)),
        )
        model.bind_online_retriever(retriever)
        online_mode = str(online_cfg.get("mode", "source_only"))
        retrospection_config = getattr(model, "warm_retrospection_config", None)
        if online_mode == "full_retrospection":
            if retrospection_config is None:
                raise ValueError(
                    "full_retrospection mode requires WarmRetrospectionFastWAM"
                )
            if (
                int(retrospection_config.action_dim) != action_dim
                or int(retrospection_config.action_horizon) != action_horizon
                or int(retrospection_config.episode_action_summary_dim)
                != 3 * action_dim + 4
                or int(retrospection_config.episode_action_chunk_size)
                != replan_steps
            ):
                raise ValueError(
                    "full WARM episode-memory dimensions differ from the rollout"
                )
            from fastwam.memory.online_episode_memory import (
                OnlineRetrospectiveEpisodeMemory,
            )

            retrospective_episode_memory = OnlineRetrospectiveEpisodeMemory(
                action_dim=action_dim,
                action_horizon=action_horizon,
                semantic_dim=int(retrospection_config.semantic_dim),
                # LIBERO uses the final model-space action channel for the
                # gripper; the evaluator records its exact executed command.
                gripper_indices=(action_dim - 1,),
                recent_event_capacity=6,
            )
        elif retrospection_config is not None:
            raise ValueError(
                "a full WARM checkpoint requires warm_online.mode=full_retrospection"
            )
    else:
        # Null evaluation deliberately reaches no event-bank/DINO path.  It
        # constructs only the contract-verified baseline image transform.
        null_image_adapter = _build_null_image_adapter(
            processor=processor,
            camera_contract_path=_required_online_path(
                online_cfg, "camera_contract_path"
            ),
            expected_sha256=contract.camera_contract_sha256,
            configured_concat_mode=str(
                cfg.data.train.get("concat_multi_camera", "horizontal")
            ),
        )

    if pair_contract_path is not None:
        if sha256_file(pair_contract_path) != pair_contract_file_sha256:
            raise RuntimeError("online pair contract changed during runtime setup")
    if parity_report_path is not None:
        if sha256_file(parity_report_path) != parity_report_file_sha256:
            raise RuntimeError("online parity report changed during runtime setup")
    if sha256_file(m1_data_config_path) != m1_data_config_sha256:
        raise RuntimeError("M1 data config changed during runtime setup")
    if sha256_file(training_attestation_path) != (
        training_attestation_file_sha256
    ):
        raise RuntimeError("training attestation changed during runtime setup")
    if not (
        getattr(model, "_warm_loaded_checkpoint_sha256", None)
        == sha256_file(checkpoint_path)
        == contract.warm_checkpoint_sha256
    ):
        raise RuntimeError(
            "loaded model or checkpoint changed during runtime setup"
        )

    return WarmOnlineEvalRuntime(
        contract=contract,
        pair_contract=pair_contract,
        pair_side=pair_side,
        source_contract=source,
        validation_source_contract=validation_source,
        retriever=retriever,
        retrospective_episode_memory=retrospective_episode_memory,
        null_image_adapter=null_image_adapter,
        normalization_stats_loaded_sha256=loaded_dataset_stats_sha256,
        pair_contract_file_sha256=pair_contract_file_sha256,
        pair_contract_path=pair_contract_path,
        parity_report_file_sha256=parity_report_file_sha256,
        parity_report_path=parity_report_path,
        m1_data_config_path=m1_data_config_path,
        training_attestation_file_sha256=training_attestation_file_sha256,
        training_attestation_path=training_attestation_path,
        training_runtime_sha256=training_attestation.training_runtime_sha256,
        shared_training_recipe_sha256=(
            training_attestation.shared_recipe_sha256
        ),
        model=model,
        checkpoint_path=checkpoint_path,
        bddl_path=bddl_path,
        git_commit=git_commit,
    )


def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _resolve_eval_device(cfg: DictConfig) -> str:
    eval_device = cfg.EVALUATION.get("device")
    if eval_device is not None:
        return str(eval_device)
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_dataset_stats_path(cfg: DictConfig) -> Path:
    explicit = cfg.EVALUATION.get("dataset_stats_path")
    candidates: list[Path] = []

    if explicit is not None:
        candidates.append(Path(os.path.expanduser(os.path.expandvars(str(explicit)))))

    ckpt = Path(os.path.expanduser(os.path.expandvars(str(cfg.ckpt))))
    for parent in list(ckpt.parents)[:4]:
        candidates.append(parent / "dataset_stats.json")

    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved

    msg = (
        "Failed to locate dataset_stats.json. Tried explicit "
        "EVALUATION.dataset_stats_path and checkpoint parent directories. "
        "Please pass EVALUATION.dataset_stats_path=/path/to/dataset_stats.json."
    )
    raise FileNotFoundError(msg)


def _load_model_checkpoint(model: torch.nn.Module, ckpt: str) -> None:
    model.load_checkpoint(ckpt)
    logging.info("Loaded checkpoint via model.load_checkpoint: %s", ckpt)
    return

    # deprecated legacy checkpoint loading
    payload = torch.load(ckpt, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Legacy checkpoint payload must be dict, got: {type(payload)}")

    if "mot" in payload and hasattr(model, "mot"):
        missing, unexpected = model.mot.load_state_dict(payload["mot"], strict=False)
        logging.warning(
            "Loaded fallback `mot` state_dict with strict=False. Missing=%d Unexpected=%d",
            len(missing),
            len(unexpected),
        )
        return

    state_dict = None
    for key in ("model_state_dict", "state_dict", "model"):
        value = payload.get(key)
        if isinstance(value, dict):
            state_dict = value
            break
    if state_dict is None and all(torch.is_tensor(v) for v in payload.values()):
        state_dict = payload
    if state_dict is None:
        raise ValueError(f"Cannot parse legacy checkpoint keys from: {ckpt}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logging.warning(
        "Loaded fallback model state_dict with strict=False. Missing=%d Unexpected=%d",
        len(missing),
        len(unexpected),
    )


def _center_crop_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    pil_image = Image.fromarray(image)
    src_w, src_h = pil_image.size
    scale = max(width / src_w, height / src_h)
    resized = pil_image.resize((round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR)
    rw, rh = resized.size
    left = max((rw - width) // 2, 0)
    top = max((rh - height) // 2, 0)
    cropped = resized.crop((left, top, left + width, top + height))
    return np.asarray(cropped, dtype=np.uint8)


def _normalize_proprio(
    proprio: np.ndarray,
    processor: FastWAMProcessor,
) -> torch.Tensor:
    state_meta = processor.shape_meta["state"]
    if len(state_meta) != 1:
        raise ValueError(
            "LIBERO eval currently expects a single merged state key in shape_meta['state']."
        )
    state_key = state_meta[0]["key"]

    state_batch = {"state": {state_key: torch.as_tensor(proprio, dtype=torch.float32).unsqueeze(0)}}
    state_batch = processor.action_state_transform(state_batch)
    state_batch = processor.normalizer.forward(state_batch)
    return state_batch["state"][state_key]


def _raw_cameras_for_image_adapter(
    raw_cameras: Mapping[str, np.ndarray],
    camera_keys: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """Convert factual LIBERO uint8 HWC cameras to adapter float NCHW."""

    if set(raw_cameras) != set(camera_keys):
        raise ValueError(
            "LIBERO camera keys do not match the bound FastWAM processor"
        )
    converted: dict[str, np.ndarray] = {}
    for key in camera_keys:
        value = raw_cameras[key]
        array = np.asarray(value)
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(f"LIBERO camera {key!r} must be uint8 HWC RGB")
        nchw = np.transpose(array, (2, 0, 1))[None].astype(np.float32) / 255.0
        converted[key] = np.ascontiguousarray(nchw, dtype=np.float32)
    return converted


def _null_online_model_input(
    raw_cameras: Mapping[str, np.ndarray],
    runtime: WarmOnlineEvalRuntime,
    *,
    device: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if runtime.null_image_adapter is None:
        raise RuntimeError("Gaussian-null online runtime has no image adapter")
    from fastwam.memory.manifest import sha256_array

    camera_keys = tuple(runtime.null_image_adapter.camera_keys)
    adapter_input = _raw_cameras_for_image_adapter(raw_cameras, camera_keys)
    prepared = runtime.null_image_adapter.prepare(adapter_input)
    evidence = {
        "raw_camera_sha256": {
            key: sha256_array(np.ascontiguousarray(raw_cameras[key]))
            for key in camera_keys
        },
        "processed_camera_sha256": {
            key: sha256_array(prepared.camera_frames[key]) for key in camera_keys
        },
        "model_input_sha256": sha256_array(prepared.vae_frames),
    }
    return (
        torch.as_tensor(prepared.vae_frames, device=device, dtype=dtype),
        evidence,
    )


def _obs_to_model_input(
    obs: dict,
    cfg: DictConfig,
    processor: FastWAMProcessor,
    width: int,
    height: int,
    device: str,
    dtype: torch.dtype,
):
    imgs = get_libero_image(obs)
    image_meta = processor.shape_meta["images"]
    if len(image_meta) < int(processor.num_output_cameras):
        raise ValueError(
            f"shape_meta.images has {len(image_meta)} entries, "
            f"but num_output_cameras={processor.num_output_cameras}."
        )

    def _meta_to_hw(meta: dict, camera_idx: int) -> tuple[int, int]:
        shape = meta["shape"]
        if len(shape) != 3:
            raise ValueError(f"shape_meta.images[{camera_idx}].shape must be [C,H,W], got {shape}")
        return int(shape[1]), int(shape[2])

    concatenation = cfg.data.train.get("concat_multi_camera", "horizontal")
    num_cameras = processor.num_output_cameras
    if num_cameras == 1:
        primary_h, primary_w = _meta_to_hw(image_meta[0], camera_idx=0)
        rgb = _center_crop_resize(imgs["image"], width=primary_w, height=primary_h)
    elif num_cameras == 2:
        primary_h, primary_w = _meta_to_hw(image_meta[0], camera_idx=0)
        wrist_h, wrist_w = _meta_to_hw(image_meta[1], camera_idx=1)
        primary = _center_crop_resize(imgs["image"], width=primary_w, height=primary_h)
        wrist = _center_crop_resize(imgs["wrist_image"], width=wrist_w, height=wrist_h)
        if concatenation == "horizontal":
            rgb = np.concatenate([primary, wrist], axis=1)
        elif concatenation == "vertical":
            rgb = np.concatenate([primary, wrist], axis=0)
        else:
            raise ValueError(f"Invalid concat_multi_camera: {concatenation}")
    else:
        raise ValueError(f"LIBERO eval currently supports num_output_cameras in [1, 2], got {num_cameras}.")

    actual_h, actual_w = int(rgb.shape[0]), int(rgb.shape[1])
    expected_h, expected_w = int(height), int(width)
    image_shapes = [meta["shape"] for meta in image_meta]
    assert actual_h == expected_h and actual_w == expected_w, (
        "Input image size mismatch after per-camera resize + concat: "
        f"got (H,W)=({actual_h},{actual_w}), expected (H,W)=({expected_h},{expected_w}) "
        f"from data.train.video_size={[expected_h, expected_w]}; "
        f"shape_meta.images={image_shapes}, concat_multi_camera={concatenation}."
    )

    x = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    x = x * (2.0 / 255.0) - 1.0

    proprio = _normalize_proprio(_extract_sim_state(obs), processor)

    return x, proprio, imgs


def _extract_sim_state(obs: dict) -> np.ndarray:
    """Build simulator state from current observation.

    This is used as proprio input for model inference.
    """
    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    ).astype(np.float32)
    return state


def _denormalize_action(action: torch.Tensor, processor: FastWAMProcessor) -> np.ndarray:
    if action.ndim == 2:
        action = action.unsqueeze(0)
    if action.ndim != 3:
        raise ValueError(f"Expected action tensor [B, T, D], got {tuple(action.shape)}")

    action_meta = processor.shape_meta["action"]
    if len(action_meta) != 1:
        raise ValueError(
            "LIBERO eval currently expects a single merged action key in shape_meta['action']."
        )

    action_key = action_meta[0]["key"]
    normalizer = processor.normalizer.normalizers["action"][action_key]
    action = action.to(dtype=torch.float32, device="cpu")
    denorm = normalizer.backward(action)
    return denorm.numpy()


def _executed_action_to_model_space(
    action: Any,
    processor: FastWAMProcessor,
) -> np.ndarray:
    """Invert the LIBERO rollout transform for an action actually executed.

    The returned vector is a deterministic normalized representation of the
    exact environment command.  Episode-memory action summaries therefore use
    the same scale as offline training while telemetry separately hashes the
    unmodified simulator command.
    """

    value = np.asarray(action, dtype=np.float32)
    if value.ndim != 1 or value.size != int(processor.action_output_dim):
        raise ValueError("executed LIBERO action has an invalid shape")
    raw = np.ascontiguousarray(value.copy())
    # Forward rollout uses e = -(2*d - 1) = 1 - 2*d, where d is the factual
    # dataset gripper channel.  Invert both the affine map and sign flip.
    raw[-1] = (np.float32(1.0) - raw[-1]) * np.float32(0.5)
    action_meta = processor.shape_meta["action"]
    if len(action_meta) != 1:
        raise ValueError(
            "LIBERO eval expects one merged action key for executed summaries"
        )
    action_key = action_meta[0]["key"]
    normalizer = processor.normalizer.normalizers["action"][action_key]
    tensor = torch.from_numpy(raw).reshape(1, 1, -1)
    normalized = normalizer.forward(tensor).reshape(-1)
    result = np.ascontiguousarray(normalized.detach().cpu().numpy(), dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("normalized executed action contains non-finite values")
    return result


def _get_num_video_frames(cfg: DictConfig) -> int:
    return (int(cfg.data.train.num_frames) - 1) // int(cfg.data.train.action_video_freq_ratio) + 1


def _validate_visualize_future_video_cfg(cfg: DictConfig) -> None:
    if not bool(cfg.EVALUATION.get("visualize_future_video", False)):
        return

    action_conditioned = cfg.model.video_dit_config.get("action_conditioned", None)
    if action_conditioned is not False:
        raise ValueError(
            "EVALUATION.visualize_future_video=true requires "
            "model.video_dit_config.action_conditioned=false."
        )


def _select_predicted_future_frames(pred_video: list[Image.Image], cfg: DictConfig) -> list[Image.Image]:
    if len(pred_video) == 0:
        raise ValueError("`infer_joint` returned an empty predicted video.")

    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    action_video_freq_ratio = int(cfg.data.train.action_video_freq_ratio)
    num_future_frames = replan_steps // action_video_freq_ratio
    keep_frames = 1 + num_future_frames
    return list(pred_video[:keep_frames])


def _get_future_frame_capture_steps(cfg: DictConfig) -> list[int]:
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    action_video_freq_ratio = int(cfg.data.train.action_video_freq_ratio)
    num_future_frames = replan_steps // action_video_freq_ratio
    return [step_idx * action_video_freq_ratio for step_idx in range(num_future_frames + 1)]


def _frame_to_rgb_array(frame: Any) -> np.ndarray:
    if isinstance(frame, dict):
        images = []
        for value in frame.values():
            value_array = np.array(value) if isinstance(value, Image.Image) else np.array(value, copy=True)
            images.append(value_array)
        return np.concatenate(images, axis=1)
    if isinstance(frame, Image.Image):
        return np.array(frame.convert("RGB"))
    return np.array(frame, copy=True)


def _compute_clip_mean_psnr(
    gt_frames: list[Any],
    pred_frames: list[Any],
    eps: float = 1e-8,
) -> Optional[float]:
    if len(gt_frames) == 0 or len(pred_frames) == 0:
        return None
    assert len(gt_frames) == len(pred_frames), (
        "GT/pred frame count mismatch for PSNR: "
        f"len(gt_frames)={len(gt_frames)} len(pred_frames)={len(pred_frames)}. "
        "This indicates temporal misalignment in future-video capture."
    )
    num_frames = len(gt_frames)

    frame_psnr_values = []
    for gt_frame, pred_frame in zip(gt_frames[:num_frames], pred_frames[:num_frames]):
        gt_image = _frame_to_rgb_array(gt_frame)
        pred_image = _frame_to_rgb_array(pred_frame)
        target_h, target_w = pred_image.shape[:2]
        if gt_image.shape[:2] != (target_h, target_w):
            gt_image = np.array(
                Image.fromarray(gt_image).resize((target_w, target_h), resample=Image.BILINEAR)
            )

        gt_f32 = gt_image.astype(np.float32)
        pred_f32 = pred_image.astype(np.float32)
        mse = float(np.mean((pred_f32 - gt_f32) ** 2))
        psnr = 10.0 * np.log10((255.0 * 255.0) / max(mse, eps))
        frame_psnr_values.append(float(psnr))

    if len(frame_psnr_values) == 0:
        return None
    return float(np.mean(frame_psnr_values))


def _predict_action_chunk(
    obs: dict,
    task_description: str,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
    online_runtime: WarmOnlineEvalRuntime | None = None,
    frame_index: int | None = None,
) -> tuple[
    np.ndarray,
    dict,
    Optional[list[Image.Image]],
    dict[str, Any] | None,
]:
    num_inference_steps_cfg = cfg.EVALUATION.get("num_inference_steps", None)
    if num_inference_steps_cfg is None:
        num_inference_steps = int(cfg.get("eval_num_inference_steps", 20))
    else:
        num_inference_steps = int(num_inference_steps_cfg)
    prompt_template = DEFAULT_PROMPT
    prompt = prompt_template.format(task=task_description)
    common_infer_kwargs = {
        "negative_prompt": str(cfg.EVALUATION.get("negative_prompt", "")),
        "text_cfg_scale": float(cfg.EVALUATION.get("text_cfg_scale", 1.0)),
        "num_inference_steps": num_inference_steps,
        "sigma_shift": (
            None
            if cfg.EVALUATION.get("sigma_shift") is None
            else float(cfg.EVALUATION.get("sigma_shift"))
        ),
        "rand_device": str(cfg.EVALUATION.get("rand_device", "cpu")),
        "tiled": bool(cfg.EVALUATION.get("tiled", False)),
    }
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    predicted_future_frames = None
    online_telemetry: dict[str, Any] | None = None
    online_step = None
    derived_seed: int | None = None
    online_input_evidence: dict[str, Any] | None = None
    online_input_stage_s: float | None = None
    online_pipeline_start: float | None = None
    retrospective_history_evidence: dict[str, Any] | None = None
    retrospective_update_evidence: dict[str, Any] | None = None

    if online_runtime is None:
        image, proprio, imgs = _obs_to_model_input(
            obs,
            cfg=cfg,
            processor=processor,
            width=input_w,
            height=input_h,
            device=model_device,
            dtype=model.torch_dtype,
        )
        infer_kwargs = {
            **common_infer_kwargs,
            "prompt": prompt,
            "input_image": image,
            "action_horizon": action_horizon,
            "proprio": proprio,
            "seed": None if cfg.get("seed") is None else int(cfg.seed),
        }
        if visualize_future_video:
            infer_kwargs["num_video_frames"] = _get_num_video_frames(cfg)
        elif "num_video_frames" in inspect.signature(model.infer_action).parameters:
            infer_kwargs["num_video_frames"] = _get_num_video_frames(cfg)
    else:
        online_pipeline_start = time.perf_counter()
        if visualize_future_video:
            raise ValueError("online WARM evaluation forbids future-video inference")
        if frame_index is None:
            raise ValueError("online WARM replanning requires absolute frame_index")
        imgs = get_libero_image(obs)
        proprio = _normalize_proprio(_extract_sim_state(obs), processor)
        query_id = online_runtime.issue_query_id(frame_index)

        if online_runtime.source_policy == "fixed_context_top1":
            if online_runtime.retriever is None:
                raise RuntimeError("fixed online WARM runtime has no retriever")
            input_stage_start = time.perf_counter()
            online_step = online_runtime.retriever.retrieve(
                query_id,
                imgs,
                task_description=task_description,
                prompt=prompt,
                proprio=proprio,
            )
            online_input_stage_s = time.perf_counter() - input_stage_start
            online_input_evidence = {
                "raw_camera_sha256": dict(online_step.raw_camera_sha256),
                "processed_camera_sha256": dict(
                    online_step.processed_camera_sha256
                ),
                "prompt_sha256": online_step.prompt_sha256,
                "proprio_sha256": online_step.proprio_sha256,
                "model_input_sha256": online_step.model_input_sha256,
            }
            infer_kwargs = {
                **common_infer_kwargs,
                "online_step": online_step,
            }
            history_kwargs, retrospective_history_evidence = (
                online_runtime.retrospective_history_kwargs()
            )
            # Only the full-retrospection runtime owns these learned-history
            # inputs.  Source-only M2.1 checkpoints retain their exact public
            # call surface.
            infer_kwargs.update(history_kwargs)
        elif online_runtime.source_policy == "gaussian_null":
            from fastwam.memory.online_retrieval import derive_online_query_seed
            from fastwam.memory.manifest import sha256_array

            input_stage_start = time.perf_counter()
            image, online_input_evidence = _null_online_model_input(
                imgs,
                online_runtime,
                device=model_device,
                dtype=model.torch_dtype,
            )
            online_input_stage_s = time.perf_counter() - input_stage_start
            normalized_proprio = np.ascontiguousarray(
                proprio.detach().float().cpu().numpy(), dtype=np.float32
            )
            online_input_evidence.update(
                {
                    "prompt_sha256": _online_prompt_sha256(prompt),
                    "proprio_sha256": sha256_array(normalized_proprio),
                }
            )
            derived_seed = derive_online_query_seed(
                online_runtime.contract.root_seed,
                query_id,
                online_runtime.contract.evaluation_namespace_sha256,
            )
            infer_kwargs = {
                **common_infer_kwargs,
                "prompt": prompt,
                "input_image": image,
                "action_horizon": action_horizon,
                "proprio": proprio,
                "seed": derived_seed,
            }
        else:
            raise ValueError(
                f"unsupported online source policy {online_runtime.source_policy!r}"
            )

    model_inference_start = time.perf_counter()
    with torch.no_grad():
        if visualize_future_video:
            pred = model.infer_joint(**infer_kwargs)
            predicted_future_frames = _select_predicted_future_frames(pred["video"], cfg)
        else:
            pred = model.infer_action(**infer_kwargs)
    model_inference_s = time.perf_counter() - model_inference_start
    if online_runtime is not None:
        if online_runtime.retrospective_episode_memory is not None:
            if frame_index is None:
                raise RuntimeError(
                    "full WARM factual memory requires an absolute frame index"
                )
            retrospective_update_evidence = (
                online_runtime.commit_factual_replan_observation(
                    frame_index=frame_index,
                    model_output=pred,
                )
            )
        if (
            online_pipeline_start is None
            or online_input_stage_s is None
            or online_input_evidence is None
        ):
            raise RuntimeError("online evaluator telemetry state is incomplete")
        model_telemetry = pred.get("warm_online_telemetry")
        if not isinstance(model_telemetry, Mapping):
            raise RuntimeError("online WARM inference returned no structured telemetry")
        online_telemetry = {
            **online_runtime.pair_identity(),
            "query_id": _query_id_dict(query_id),
            "absolute_sim_step": int(frame_index),
            "source_policy": online_runtime.source_policy,
            "derived_seed": int(
                online_step.derived_seed if online_step is not None else derived_seed
            ),
            **online_input_evidence,
            "evaluator_latency_s": {
                "input_prepare_or_retrieval_s": float(online_input_stage_s),
                "model_inference_s": float(model_inference_s),
                "online_pipeline_s": float(
                    time.perf_counter() - online_pipeline_start
                ),
            },
            "model": dict(model_telemetry),
        }
        if online_runtime.retrospective_episode_memory is not None:
            online_telemetry["retrospective_episode_memory"] = {
                "history_before_replan": retrospective_history_evidence,
                "factual_update_after_replan": retrospective_update_evidence,
            }
        if online_step is not None:
            candidates = []
            for rank in range(len(online_step.event_ids)):
                candidates.append(
                    {
                        "rank": rank,
                        "valid": bool(online_step.candidate_valid_mask[rank]),
                        "bank_row": int(online_step.bank_rows[rank]),
                        "event_id": _event_id_dict(online_step.event_ids[rank]),
                        "cosine_score": float(online_step.cosine_scores[rank]),
                    }
                )
            online_telemetry.update(
                {
                    "bound_step_sha256": online_step.step_sha256,
                    "context_key_sha256": online_step.context_key_sha256,
                    "candidate_payload_sha256": (
                        online_step.candidate_payload_sha256
                    ),
                    "candidates": candidates,
                }
            )
        else:
            online_telemetry.update(
                {
                    "bound_step_sha256": None,
                    "context_key_sha256": None,
                    "candidate_payload_sha256": None,
                    "candidates": [],
                }
            )
    action = pred["action"]  # [T, D]

    action = _denormalize_action(action, processor)[0]  # [T, D]

    # The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    action[..., -1] = action[..., -1] * 2 - 1
    action = invert_gripper_action(action)
    if bool(cfg.EVALUATION.get("binarize_gripper", False)):
        action[..., -1] = np.sign(action[..., -1])
    return action, imgs, predicted_future_frames, online_telemetry


def _get_max_steps(task_suite_name: str) -> int:
    suite_steps = {
        "libero_spatial": 400,
        "libero_object": 400,
        "libero_goal": 400,
        "libero_10": 700,
        "libero_90": 700,
    }
    if task_suite_name not in suite_steps:
        raise ValueError(f"Unknown task suite: {task_suite_name}")
    return suite_steps[task_suite_name]


def _derive_episode_simulator_seed(
    root_seed: int,
    task_suite: str,
    task_id: int,
    episode_index: int,
) -> int:
    """Derive a policy-independent simulator RNG namespace per trial."""

    payload = json.dumps(
        {
            "root_seed": int(root_seed),
            "task_suite": str(task_suite),
            "task_id": int(task_id),
            "episode_index": int(episode_index),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256(b"warm.libero-episode-seed.v1\0" + payload).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def run_single_episode(
    env,
    initial_state,
    task_description: str,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    episode_idx: int,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
    online_runtime: WarmOnlineEvalRuntime | None = None,
) -> tuple[
    bool,
    list,
    list[dict[str, Any]],
    Optional[float],
    list[dict[str, Any]],
    dict[str, Any],
]:
    max_steps = _get_max_steps(cfg.EVALUATION.task_suite_name)
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    num_steps_wait = int(cfg.EVALUATION.get("num_steps_wait", 5))
    use_action_ensembler = bool(cfg.EVALUATION.get("use_action_ensembler", False))
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    capture_steps = set(_get_future_frame_capture_steps(cfg)[1:])

    if online_runtime is not None:
        online_runtime.begin_episode(episode_idx)
        simulator_seed = _derive_episode_simulator_seed(
            online_runtime.contract.root_seed,
            online_runtime.contract.task_suite,
            online_runtime.contract.task_id,
            episode_idx,
        )
        env.seed(simulator_seed)
        set_global_seed(simulator_seed, get_worker_init_fn=False)
    else:
        simulator_seed = None
    env.reset()
    obs = env.set_init_state(initial_state)
    if use_action_ensembler:
        ensembler = ActionEnsembler()
        ensembler.reset()

    replay_images = []
    predicted_future_video_clips: list[dict[str, Any]] = []
    episode_future_clip_psnr: list[float] = []
    pending_actions: list[list[float]] = []
    current_predicted_future_clip: Optional[dict[str, Any]] = None
    current_replan_step = 0
    current_replan_idx = -1
    online_replans: list[dict[str, Any]] = []

    t = 0
    done = False
    environment_step_count = 0
    policy_action_step_count = 0
    pbar = tqdm(total=max_steps + num_steps_wait, desc=f"Episode {episode_idx + 1}")
    while t < max_steps + num_steps_wait:
        pbar.update(1)
        if t < num_steps_wait:
            obs, _, done, _ = env.step(get_libero_dummy_action())
            environment_step_count += 1
            t += 1
            if done:
                break
            continue

        if len(pending_actions) == 0:
            (
                action_chunk,
                imgs,
                predicted_future_frames,
                online_telemetry,
            ) = _predict_action_chunk(
                obs=obs,
                task_description=task_description,
                model=model,
                processor=processor,
                cfg=cfg,
                action_horizon=action_horizon,
                input_w=input_w,
                input_h=input_h,
                model_device=model_device,
                online_runtime=online_runtime,
                # Query/working-memory time counts only policy actions.  The
                # initial simulator-settling dummy commands are not part of a
                # demonstration and must not phase-shift causal replay.
                frame_index=policy_action_step_count,
            )
            if online_telemetry is not None:
                online_telemetry["replan_index"] = len(online_replans)
                online_replans.append(online_telemetry)
            if predicted_future_frames is not None:
                current_replan_idx += 1
                current_predicted_future_clip = {
                    "replan_idx": current_replan_idx,
                    "gt_frames": [imgs.copy()],
                    "pred_frames": predicted_future_frames,
                }
            else:
                current_predicted_future_clip = None
            current_replan_step = 0
            if use_action_ensembler:
                ensembler.add_actions(action_chunk, t)
                pending_actions = [ensembler.get_action(ts).tolist() for ts in range(t, t + replan_steps)]
            else:
                pending_actions = action_chunk[:replan_steps].tolist()
            replay_images.append(imgs.copy())
        else:
            imgs = get_libero_image(obs)
            replay_images.append(imgs.copy())

        executed_action = pending_actions.pop(0)
        obs, _, done, _ = env.step(executed_action)
        if (
            online_runtime is not None
            and online_runtime.retrospective_episode_memory is not None
        ):
            online_runtime.note_executed_action(
                executed_action,
                model_space_action=_executed_action_to_model_space(
                    executed_action, processor
                ),
            )
        environment_step_count += 1
        policy_action_step_count += 1
        if visualize_future_video and current_predicted_future_clip is not None:
            current_replan_step += 1
            if current_replan_step in capture_steps:
                current_predicted_future_clip["gt_frames"].append(get_libero_image(obs))
            if done or len(pending_actions) == 0:
                expected_frame_count = 1 + sum(
                    1 for capture_step in capture_steps if capture_step <= current_replan_step
                )
                gt_len = len(current_predicted_future_clip["gt_frames"])
                pred_len = len(current_predicted_future_clip["pred_frames"])
                assert gt_len == expected_frame_count, (
                    "GT future frames do not match expected capture count: "
                    f"gt_len={gt_len} expected={expected_frame_count} "
                    f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']} "
                    f"current_replan_step={current_replan_step} capture_steps={sorted(capture_steps)}."
                )
                assert pred_len >= expected_frame_count, (
                    "Predicted future frames shorter than expected capture count: "
                    f"pred_len={pred_len} expected={expected_frame_count} "
                    f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']}."
                )
                if pred_len != expected_frame_count:
                    logging.info(
                        "Align predicted clip length to executed steps: "
                        "episode=%s replan=%s done=%s expected=%s pred_full=%s",
                        episode_idx,
                        current_predicted_future_clip["replan_idx"],
                        done,
                        expected_frame_count,
                        pred_len,
                    )
                current_predicted_future_clip["pred_frames"] = current_predicted_future_clip["pred_frames"][
                    :expected_frame_count
                ]
                assert len(current_predicted_future_clip["gt_frames"]) == len(
                    current_predicted_future_clip["pred_frames"]
                ), (
                    "GT/pred frame count mismatch after alignment: "
                    f"len(gt_frames)={len(current_predicted_future_clip['gt_frames'])} "
                    f"len(pred_frames)={len(current_predicted_future_clip['pred_frames'])} "
                    f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']}."
                )
                clip_psnr = _compute_clip_mean_psnr(
                    current_predicted_future_clip["gt_frames"],
                    current_predicted_future_clip["pred_frames"],
                )
                if clip_psnr is not None:
                    episode_future_clip_psnr.append(clip_psnr)
                predicted_future_video_clips.append(current_predicted_future_clip)
                current_predicted_future_clip = None
        if done:
            break
        t += 1
    pbar.close()

    retrospective_episode_evidence = (
        None if online_runtime is None else online_runtime.end_episode()
    )

    episode_mean_psnr = (
        float(np.mean(episode_future_clip_psnr)) if len(episode_future_clip_psnr) > 0 else None
    )
    episode_evidence = {
        "simulator_seed": simulator_seed,
        "termination_reason": "success" if done else "max_steps",
        "final_frame_index": int(t),
        "environment_step_count": int(environment_step_count),
        "policy_action_step_count": int(policy_action_step_count),
        "configured_wait_steps": int(num_steps_wait),
        "configured_policy_max_steps": int(max_steps),
        "configured_replan_steps": int(replan_steps),
        "replan_count": len(online_replans),
    }
    if retrospective_episode_evidence is not None:
        episode_evidence["retrospective_episode_memory"] = (
            retrospective_episode_evidence
        )
    return (
        bool(done),
        replay_images,
        predicted_future_video_clips,
        episode_mean_psnr,
        online_replans,
        episode_evidence,
    )


def run_single_task(
    task,
    initial_states,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    video_dir: Path,
    predicted_video_dir: Path,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
    online_runtime: WarmOnlineEvalRuntime | None = None,
) -> dict:
    env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, cfg.get("seed"))
    if online_runtime is not None:
        from fastwam.memory.manifest import sha256_file

        if task_description != online_runtime.contract.task_description:
            raise ValueError(
                "LIBERO environment task language changed after contract validation"
            )
        if sha256_file(online_runtime.bddl_path) != online_runtime.contract.bddl_sha256:
            raise RuntimeError(
                "LIBERO BDDL changed while the contracted environment was created"
            )
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    results = {
        "successes": 0,
        "failure_episodes": [],
        "success_episodes": [],
        "task_description": task_description,
    }
    if visualize_future_video:
        results["episode_future_video_psnr"] = []
        results["future_video_psnr_mean"] = None
    if online_runtime is not None:
        results["warm_online_header"] = online_runtime.result_header()
        results["warm_online_episodes"] = []

    for trial_idx in range(int(cfg.EVALUATION.num_trials)):
        (
            success,
            replay_images,
            predicted_future_video_clips,
            episode_mean_psnr,
            online_replans,
            episode_evidence,
        ) = run_single_episode(
            env=env,
            initial_state=initial_states[trial_idx],
            task_description=task_description,
            model=model,
            processor=processor,
            cfg=cfg,
            episode_idx=trial_idx,
            action_horizon=action_horizon,
            input_w=input_w,
            input_h=input_h,
            model_device=model_device,
            online_runtime=online_runtime,
        )
        if success:
            results["successes"] += 1
            results["success_episodes"].append(trial_idx)
        else:
            results["failure_episodes"].append(trial_idx)
        if visualize_future_video:
            results["episode_future_video_psnr"].append(episode_mean_psnr)
        if online_runtime is not None:
            results["warm_online_episodes"].append(
                {
                    **online_runtime.pair_identity(),
                    "episode_index": trial_idx,
                    "success": bool(success),
                    **episode_evidence,
                    "replans": online_replans,
                }
            )

        save_rollout_video(
            video_dir,
            replay_images,
            f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
            success=success,
            task_description=task_description,
        )
        if visualize_future_video:
            if len(predicted_future_video_clips) == 0:
                logging.warning(
                    "No predicted future frames collected for task %s trial %s.",
                    cfg.EVALUATION.task_id,
                    trial_idx,
                )
            else:
                all_gt_frames = []
                all_pred_frames = []
                for clip in predicted_future_video_clips:
                    all_gt_frames.extend(clip["gt_frames"])
                    all_pred_frames.extend(clip["pred_frames"])
                    save_prediction_video(
                        predicted_video_dir,
                        clip["gt_frames"],
                        clip["pred_frames"],
                        f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
                        clip["replan_idx"],
                        success=success,
                        task_description=task_description,
                    )
                save_prediction_video(
                    predicted_video_dir,
                    all_gt_frames,
                    all_pred_frames,
                    f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
                    "all",
                    success=success,
                    task_description=task_description,
                )

    if visualize_future_video:
        valid_episode_psnr = [x for x in results["episode_future_video_psnr"] if x is not None]
        if len(valid_episode_psnr) > 0:
            results["future_video_psnr_mean"] = float(np.mean(valid_episode_psnr))
    if online_runtime is not None:
        online_runtime.attest_pair_contract_file()
    return results


def _write_result_json_atomic(
    path: Path,
    value: Mapping[str, Any],
    *,
    refuse_existing: bool,
) -> None:
    """Publish one complete result; formal jobs never overwrite evidence."""

    if refuse_existing and path.exists():
        raise FileExistsError(f"formal online result already exists: {path}")
    encoded = (
        json.dumps(
            dict(value),
            indent=4,
            ensure_ascii=True,
            allow_nan=False,
            cls=NumpyEncoder,
        ).encode("utf-8")
        + b"\n"
    )
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if refuse_existing:
            # Atomic no-replace publication on both Linux and Windows: the
            # destination hard link can only be created when it is absent.
            os.link(temporary, path)
            temporary.unlink()
        else:
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml")
def eval_single_process(cfg: DictConfig):
    start_time = time.time()
    partial_state = PartialState()
    partial_state.config = cfg

    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)

    if cfg.ckpt is None:
        raise ValueError("cfg.ckpt must not be None.")
    _validate_visualize_future_video_cfg(cfg)

    env_num = int(cfg.EVALUATION.get("env_num", 1))
    if env_num != 1:
        raise ValueError(
            "Only env_num=1 is supported in eval_libero_single.py. "
            "Use run_libero_manager/run_libero_parallel_test.sh for multi-GPU task parallelism."
        )

    model_device = _resolve_eval_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    _load_model_checkpoint(model, str(cfg.ckpt))
    model = model.to(model_device).eval()

    dataset_stats_path = _resolve_dataset_stats_path(cfg)
    dataset_stats, loaded_dataset_stats_sha256 = _load_dataset_stats_stable(
        dataset_stats_path
    )
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)
    logging.info("Using dataset stats: %s", dataset_stats_path)

    action_horizon_cfg = cfg.EVALUATION.get("action_horizon", None)
    if action_horizon_cfg is None:
        action_horizon = int(cfg.data.train.num_frames) - 1
    else:
        action_horizon = int(action_horizon_cfg)
    if action_horizon <= 0:
        raise ValueError(f"EVALUATION.action_horizon must be positive, got {action_horizon}")

    video_size = cfg.data.train.get("video_size", [224, 224])
    if len(video_size) != 2:
        raise ValueError(f"data.train.video_size must be [H, W], got {video_size}")
    input_h = int(video_size[0])
    input_w = int(video_size[1])
    concat_multi_camera = cfg.data.train.get("concat_multi_camera", None)
    shape_meta_images = [meta["shape"] for meta in processor.shape_meta["images"]]

    local_log_dir = Path(cfg.EVALUATION.output_dir)
    local_log_dir.mkdir(parents=True, exist_ok=True)
    output_dir = local_log_dir / cfg.EVALUATION.task_suite_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / (
        f"gpu{cfg.gpu_id}_task{cfg.EVALUATION.task_id}_results.json"
    )
    formal_online_enabled = bool(
        cfg.EVALUATION.get("warm_online", {}).get("enabled", False)
    )
    if formal_online_enabled and output_file.exists():
        raise FileExistsError(
            f"formal online result already exists before rollout: {output_file}"
        )
    video_dir = local_log_dir / cfg.EVALUATION.task_suite_name / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    predicted_video_dir = local_log_dir / cfg.EVALUATION.task_suite_name / "predicted_videos"
    if bool(cfg.EVALUATION.get("visualize_future_video", False)):
        predicted_video_dir.mkdir(parents=True, exist_ok=True)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.EVALUATION.task_suite_name]()
    task = task_suite.get_task(cfg.EVALUATION.task_id)
    initial_states = task_suite.get_task_init_states(cfg.EVALUATION.task_id)

    online_runtime = _load_warm_online_runtime(
        cfg,
        model=model,
        processor=processor,
        dataset_stats_path=dataset_stats_path,
        task=task,
        initial_states=initial_states,
        action_horizon=action_horizon,
        loaded_dataset_stats_sha256=loaded_dataset_stats_sha256,
    )

    while len(initial_states) < int(cfg.EVALUATION.num_trials):
        initial_states.extend(initial_states[: (int(cfg.EVALUATION.num_trials) - len(initial_states))])

    results = {
        "task_suite": cfg.EVALUATION.task_suite_name,
        "task_id": cfg.EVALUATION.task_id,
        "task_description": None,
        "successes": 0,
        "total_episodes": int(cfg.EVALUATION.num_trials),
        "gpu_id": int(cfg.gpu_id),
        "success_episodes": [],
        "failure_episodes": [],
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration": 0,
    }

    logging.info("Running LIBERO evaluation with env_num=1")
    task_results = run_single_task(
        task=task,
        initial_states=initial_states,
        model=model,
        processor=processor,
        cfg=cfg,
        video_dir=video_dir,
        predicted_video_dir=predicted_video_dir,
        action_horizon=action_horizon,
        input_w=input_w,
        input_h=input_h,
        model_device=model_device,
        online_runtime=online_runtime,
    )
    results.update(task_results)

    results["duration"] = time.time() - start_time
    _write_result_json_atomic(
        output_file,
        results,
        refuse_existing=formal_online_enabled,
    )

    print(
        f"Task {cfg.EVALUATION.task_id} completed: "
        f"{results['successes']}/{cfg.EVALUATION.num_trials} successes"
    )
    if results.get("future_video_psnr_mean") is not None:
        print(f"Task {cfg.EVALUATION.task_id} future-video PSNR mean: {results['future_video_psnr_mean']:.4f}")
    print(f"Time taken: {results['duration']:.2f} seconds")
    return results


if __name__ == "__main__":
    eval_single_process()
