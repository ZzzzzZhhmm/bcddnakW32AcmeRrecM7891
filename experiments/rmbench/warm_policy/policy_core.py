"""CPU-only primitives for the official RMBench WARM policy.

The simulator-facing module imports the 5B model stack and is intentionally
server-only.  This module keeps the causal observation/action boundary and
the append-only evidence format independently testable on a CPU workstation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from fastwam.memory.online_episode_controller import (
    FactualObservationBindingError,
    bind_factual_world_tokens,
)


PROCESSOR_CAMERA_KEYS = (
    "cam_high",
    "cam_left_wrist",
    "cam_right_wrist",
)
_OBS_CAMERA_PATHS = (
    ("cam_high", "head_camera"),
    ("cam_left_wrist", "left_camera"),
    ("cam_right_wrist", "right_camera"),
)


class RMBenchPolicyBoundaryError(ValueError):
    """Raised when a simulator value crosses an unproved policy boundary."""


def resolve_task_bundle_paths(
    contract_or_bundle: str | Path,
    *,
    task_name: str,
    experiment_id: str,
    official_root: str | Path,
    initial_states_path: str | Path | None = None,
    task_definition_path: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    """Resolve the exact per-task online contract bundle layout.

    A manager may pass either one concrete ``<task>.json`` or a common bundle
    root.  The latter has the stable layout
    ``<root>/<experiment_id>/<task>.json`` and
    ``<root>/seeds/<task>.seed_protocol.npy``.  The task implementation is never
    copied into the private bundle; by default it is the pinned external
    checkout's ``envs/<task>.py`` and its bytes are validated by the contract.
    """

    if (
        not isinstance(task_name, str)
        or not task_name
        or task_name.strip() != task_name
        or any(char in task_name for char in "/\\\x00")
    ):
        raise RMBenchPolicyBoundaryError("task_name is not a safe bundle component")
    if (
        not isinstance(experiment_id, str)
        or not experiment_id
        or experiment_id.strip() != experiment_id
        or any(char in experiment_id for char in "/\\\x00")
    ):
        raise RMBenchPolicyBoundaryError(
            "experiment_id is not a safe bundle component"
        )
    source = Path(contract_or_bundle).expanduser().resolve()
    if source.is_dir():
        contract = source / experiment_id / f"{task_name}.json"
        default_initial = source / "seeds" / f"{task_name}.seed_protocol.npy"
    elif source.is_file():
        contract = source
        default_initial = (
            source.parent.parent / "seeds" / f"{task_name}.seed_protocol.npy"
        )
    else:
        raise FileNotFoundError(f"online contract/bundle does not exist: {source}")
    initial = (
        default_initial
        if initial_states_path is None or not str(initial_states_path).strip()
        else Path(initial_states_path).expanduser().resolve()
    )
    task_definition = (
        Path(official_root).expanduser().resolve() / "envs" / f"{task_name}.py"
        if task_definition_path is None or not str(task_definition_path).strip()
        else Path(task_definition_path).expanduser().resolve()
    )
    for path, label in (
        (contract, "per-task online contract"),
        (initial, "seed protocol"),
        (task_definition, "pinned task definition"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if contract.name != f"{task_name}.json":
        raise RMBenchPolicyBoundaryError(
            "concrete contract file must be named <task_name>.json"
        )
    if initial.suffix.lower() != ".npy":
        raise RMBenchPolicyBoundaryError("initial-state protocol must be .npy")
    return contract, initial, task_definition


def _uint8_hwc_rgb(value: Any, *, field: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
        raise RMBenchPolicyBoundaryError(f"{field} must be uint8 HWC RGB")
    if array.shape[0] <= 0 or array.shape[1] <= 0:
        raise RMBenchPolicyBoundaryError(f"{field} has an empty spatial axis")
    return np.ascontiguousarray(array)


def factual_cameras(observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Map the official raw observation to the processor's exact camera keys."""

    if not isinstance(observation, Mapping):
        raise TypeError("RMBench observation must be a mapping")
    camera_root = observation.get("observation")
    if not isinstance(camera_root, Mapping):
        raise RMBenchPolicyBoundaryError("observation.observation is missing")
    result: dict[str, np.ndarray] = {}
    for processor_key, raw_key in _OBS_CAMERA_PATHS:
        camera = camera_root.get(raw_key)
        if not isinstance(camera, Mapping) or "rgb" not in camera:
            raise RMBenchPolicyBoundaryError(
                f"observation.observation.{raw_key}.rgb is missing"
            )
        result[processor_key] = _uint8_hwc_rgb(
            camera["rgb"], field=f"{raw_key}.rgb"
        )
    if tuple(result) != PROCESSOR_CAMERA_KEYS:
        raise AssertionError("internal RMBench camera order drifted")
    return result


def factual_joint_state(observation: Mapping[str, Any]) -> np.ndarray:
    """Return the official native 14D bimanual qpos observation."""

    if not isinstance(observation, Mapping):
        raise TypeError("RMBench observation must be a mapping")
    joint = observation.get("joint_action")
    if not isinstance(joint, Mapping) or "vector" not in joint:
        raise RMBenchPolicyBoundaryError("observation.joint_action.vector is missing")
    vector = np.asarray(joint["vector"], dtype=np.float32)
    if vector.shape != (14,) or not np.isfinite(vector).all():
        raise RMBenchPolicyBoundaryError(
            "observation.joint_action.vector must be finite shape [14]"
        )
    return np.ascontiguousarray(vector)


def audit_qpos_execution(
    *,
    before: Any,
    target: Any,
    after: Any,
    command_threshold: float = 1.0e-2,
    motion_threshold: float = 1.0e-5,
) -> dict[str, Any]:
    """Fail closed when RMBench silently drops a commanded arm trajectory.

    The official ``take_action(..., action_type='qpos')`` catches TOPP
    exceptions internally.  On that path the affected arm is never actuated,
    while the evaluator still increments its policy step and returns normally.
    Grippers are excluded because their controller is independent from TOPP.
    """

    arrays: dict[str, np.ndarray] = {}
    for name, value in (("before", before), ("target", target), ("after", after)):
        array = np.asarray(value, dtype=np.float32)
        if array.shape != (14,) or not np.isfinite(array).all():
            raise RMBenchPolicyBoundaryError(
                f"qpos execution audit {name} must be finite shape [14]"
            )
        arrays[name] = np.ascontiguousarray(array)
    if (
        not np.isfinite(command_threshold)
        or not np.isfinite(motion_threshold)
        or command_threshold <= 0.0
        or motion_threshold <= 0.0
        or motion_threshold >= command_threshold
    ):
        raise ValueError("qpos execution audit thresholds are invalid")

    records: dict[str, dict[str, float | bool]] = {}
    silent: list[str] = []
    for name, indices in {"left": slice(0, 6), "right": slice(7, 13)}.items():
        commanded = arrays["target"][indices] - arrays["before"][indices]
        factual = arrays["after"][indices] - arrays["before"][indices]
        command_norm = float(np.linalg.norm(commanded))
        motion_norm = float(np.linalg.norm(factual))
        required = command_norm >= float(command_threshold)
        dropped = bool(required and motion_norm <= float(motion_threshold))
        records[name] = {
            "command_norm": command_norm,
            "motion_norm": motion_norm,
            "before_error": command_norm,
            "after_error": float(
                np.linalg.norm(
                    arrays["target"][indices] - arrays["after"][indices]
                )
            ),
            "motion_required": bool(required),
            "silent_drop": dropped,
        }
        if dropped:
            silent.append(name)
    if silent:
        raise RMBenchPolicyBoundaryError(
            "official RMBench qpos executor silently dropped TOPP motion for "
            + ", ".join(silent)
        )
    return {
        "command_threshold": float(command_threshold),
        "motion_threshold": float(motion_threshold),
        "arms": records,
    }


def replace_bridge_world_tokens_with_factual_dino(
    model_output: Mapping[str, Any],
    factual_world_tokens: Any,
) -> dict[str, Any]:
    """Bind episode memory to factual DINO tokens from online retrieval.

    The complete model emits its learned semantic-bridge tokens for action
    inference.  RMBench working memory, however, is trained on M1's frozen
    DINO 2x2 spatial tokens.  The online retriever already computed those
    tokens from the current real observation; this boundary replaces only the
    world-token field while preserving the model-certified factual VAE latent
    and proprioception.  No prediction or second DINO forward is admitted.
    """

    try:
        return bind_factual_world_tokens(model_output, factual_world_tokens)
    except FactualObservationBindingError as exc:
        raise RMBenchPolicyBoundaryError(str(exc)) from exc


def _event_identity(value: Any | None) -> dict[str, Any] | None:
    if value is None:
        return None
    try:
        return {
            "dataset_id": str(value.dataset_id),
            "dataset_index": int(value.dataset_index),
            "episode_index": int(value.episode_index),
            "start_frame": int(value.start_frame),
        }
    except (AttributeError, TypeError, ValueError) as exc:
        raise RMBenchPolicyBoundaryError("invalid online event identity") from exc


def validate_warm_model_telemetry(
    value: Any,
    *,
    online_step: Any,
    experiment_id: str,
    ablation_mode: str,
    memory_corruption: str,
    online_contract_sha256: str,
    training_run_contract_sha256: str,
    validation_run_contract_sha256: str,
    bank_manifest_sha256: str,
    bank_content_sha256: str,
    memory_sigma: float,
) -> dict[str, Any]:
    """Prove that model telemetry describes the requested factual rollout."""

    if not isinstance(value, Mapping):
        raise RMBenchPolicyBoundaryError(
            "WARM inference returned no structured telemetry"
        )
    experiment = value.get("experiment")
    expected_experiment_fields = {
        "experiment_id",
        "ablation_mode",
        "memory_corruption",
        "corruption_applied",
        "corruption_fallback",
    }
    if not isinstance(experiment, Mapping) or set(experiment) != (
        expected_experiment_fields
    ):
        raise RMBenchPolicyBoundaryError(
            "WARM experiment telemetry schema is incomplete"
        )
    if (
        experiment["experiment_id"] != experiment_id
        or experiment["ablation_mode"] != ablation_mode
        or experiment["memory_corruption"] != memory_corruption
    ):
        raise RMBenchPolicyBoundaryError(
            "WARM model executed different experiment controls"
        )
    applied = experiment["corruption_applied"]
    fallback = experiment["corruption_fallback"]
    if type(applied) is not bool or type(fallback) is not bool:
        raise RMBenchPolicyBoundaryError(
            "WARM corruption telemetry must be boolean"
        )
    candidate_mask = np.asarray(online_step.candidate_valid_mask)
    if candidate_mask.dtype != np.bool_ or candidate_mask.ndim != 1:
        raise RMBenchPolicyBoundaryError("bound candidate mask must be bool [K]")
    valid_count = int(candidate_mask.sum())
    if memory_corruption == "clean":
        expected_applied, expected_fallback = False, False
    elif memory_corruption == "wrong_event":
        expected_applied = valid_count > 0
        expected_fallback = valid_count < 2
    else:
        expected_applied, expected_fallback = valid_count > 0, False
    if (applied, fallback) != (expected_applied, expected_fallback):
        raise RMBenchPolicyBoundaryError(
            "WARM corruption was not applied as contracted"
        )

    retrieval = value.get("retrieval")
    if not isinstance(retrieval, Mapping):
        raise RMBenchPolicyBoundaryError("WARM retrieval telemetry is missing")
    expected_retrieval = {
        "online_contract_sha256": online_contract_sha256,
        "training_run_contract_sha256": training_run_contract_sha256,
        "validation_run_contract_sha256": validation_run_contract_sha256,
        "bank_manifest_sha256": bank_manifest_sha256,
        "bank_content_sha256": bank_content_sha256,
        "step_sha256": online_step.step_sha256,
        "prompt_sha256": online_step.prompt_sha256,
        "proprio_sha256": online_step.proprio_sha256,
        "model_input_sha256": online_step.model_input_sha256,
    }
    if any(
        retrieval.get(key) != expected
        for key, expected in expected_retrieval.items()
    ):
        raise RMBenchPolicyBoundaryError(
            "WARM model telemetry contradicts bound retrieval"
        )
    if retrieval.get("bank_rows") != [
        int(item) for item in online_step.bank_rows.tolist()
    ] or retrieval.get("candidate_valid_mask") != [
        bool(item) for item in candidate_mask.tolist()
    ]:
        raise RMBenchPolicyBoundaryError(
            "WARM telemetry candidate ordering drifted"
        )
    query_id = online_step.query_id
    expected_query = {
        "dataset_id": str(query_id.dataset_id),
        "dataset_index": int(query_id.dataset_index),
        "episode_index": int(query_id.episode_index),
        "frame_index": int(query_id.frame_index),
    }
    if (
        retrieval.get("query_id") != expected_query
        or retrieval.get("ranked_event_ids")
        != [_event_identity(item) for item in online_step.event_ids]
        or retrieval.get("cosine_scores")
        != [float(item) for item in online_step.cosine_scores.tolist()]
    ):
        raise RMBenchPolicyBoundaryError(
            "WARM telemetry ranked retrieval evidence drifted"
        )

    source = value.get("source")
    if not isinstance(source, Mapping):
        raise RMBenchPolicyBoundaryError("WARM source telemetry is missing")
    if source.get("policy") != "consequence_aligned_retrospection":
        raise RMBenchPolicyBoundaryError("WARM source policy telemetry drifted")
    candidate_selected = source.get("candidate_selected")
    memory_selected = source.get("memory_selected")
    if type(candidate_selected) is not bool or type(memory_selected) is not bool:
        raise RMBenchPolicyBoundaryError(
            "WARM source selection telemetry must be boolean"
        )
    selected_rank = source.get("selected_rank")
    if candidate_selected:
        if (
            isinstance(selected_rank, bool)
            or not isinstance(selected_rank, int)
            or not 0 <= selected_rank < len(online_step.event_ids)
            or not bool(candidate_mask[selected_rank])
        ):
            raise RMBenchPolicyBoundaryError(
                "WARM selected an invalid memory candidate"
            )
        expected_event = _event_identity(online_step.event_ids[selected_rank])
    else:
        if selected_rank is not None:
            raise RMBenchPolicyBoundaryError(
                "null WARM selection must not expose a rank"
            )
        expected_event = None
    if source.get("selected_event_id") != expected_event:
        raise RMBenchPolicyBoundaryError(
            "WARM selected-event telemetry is inconsistent"
        )
    if ablation_mode == "context_only":
        if memory_selected:
            raise RMBenchPolicyBoundaryError(
                "context_only ablation used a memory action source"
            )
    elif memory_selected and not candidate_selected:
        raise RMBenchPolicyBoundaryError(
            "WARM memory source has no selected candidate"
        )
    component = source.get("component")
    expected_component = selected_rank + 1 if memory_selected else 0
    if component != expected_component:
        raise RMBenchPolicyBoundaryError(
            "WARM source component telemetry is inconsistent"
        )
    if source.get("derived_seed") != int(online_step.derived_seed):
        raise RMBenchPolicyBoundaryError(
            "WARM sampler seed differs from bound retrieval"
        )
    try:
        reported_sigma = float(source.get("memory_sigma", float("nan")))
    except (TypeError, ValueError) as exc:
        raise RMBenchPolicyBoundaryError("WARM source sigma is invalid") from exc
    if reported_sigma != float(memory_sigma):
        raise RMBenchPolicyBoundaryError(
            "WARM source sigma differs from online contract"
        )
    for name in (
        "gate",
        "memory_relevance_gate",
        "learned_gate",
        "source_quality",
        "selected_probability",
        "normalized_entropy",
        "stagnation_score",
    ):
        try:
            scalar = float(source.get(name, float("nan")))
        except (TypeError, ValueError) as exc:
            raise RMBenchPolicyBoundaryError(
                f"WARM {name} telemetry is invalid"
            ) from exc
        if not np.isfinite(scalar) or not 0.0 <= scalar <= 1.0:
            raise RMBenchPolicyBoundaryError(
                f"WARM {name} telemetry is invalid"
            )
    try:
        probability_margin = float(
            source.get("probability_margin", float("nan"))
        )
    except (TypeError, ValueError) as exc:
        raise RMBenchPolicyBoundaryError(
            "WARM probability_margin telemetry is invalid"
        ) from exc
    if not np.isfinite(probability_margin) or not -1.0 <= probability_margin <= 1.0:
        raise RMBenchPolicyBoundaryError(
            "WARM probability_margin telemetry is invalid"
        )
    try:
        thread_prior = float(source.get("thread_prior", float("nan")))
    except (TypeError, ValueError) as exc:
        raise RMBenchPolicyBoundaryError(
            "WARM thread_prior telemetry is invalid"
        ) from exc
    if not np.isfinite(thread_prior):
        raise RMBenchPolicyBoundaryError(
            "WARM thread_prior telemetry is invalid"
        )
    if type(source.get("thread_source_eligible")) is not bool:
        raise RMBenchPolicyBoundaryError(
            "WARM thread_source_eligible telemetry is invalid"
        )
    try:
        action_offset = int(source.get("thread_action_offset", -1))
    except (TypeError, ValueError) as exc:
        raise RMBenchPolicyBoundaryError("WARM thread_action_offset is invalid") from exc
    if isinstance(source.get("thread_action_offset"), bool) or action_offset < 0:
        raise RMBenchPolicyBoundaryError("WARM thread_action_offset is invalid")
    try:
        phase_elapsed = int(source.get("thread_phase_elapsed_actions", -1))
    except (TypeError, ValueError) as exc:
        raise RMBenchPolicyBoundaryError(
            "WARM thread_phase_elapsed_actions telemetry is invalid"
        ) from exc
    if isinstance(
        source.get("thread_phase_elapsed_actions"), bool
    ) or phase_elapsed < 0:
        raise RMBenchPolicyBoundaryError(
            "WARM thread_phase_elapsed_actions telemetry is invalid"
        )
    try:
        episode_query_delta_norm = float(
            source.get("episode_query_delta_norm", float("nan"))
        )
    except (TypeError, ValueError) as exc:
        raise RMBenchPolicyBoundaryError(
            "WARM episode_query_delta_norm telemetry is invalid"
        ) from exc
    if not np.isfinite(episode_query_delta_norm) or episode_query_delta_norm < 0.0:
        raise RMBenchPolicyBoundaryError(
            "WARM episode_query_delta_norm telemetry is invalid"
        )
    return dict(value)


@dataclass(frozen=True, slots=True)
class QueuedAction:
    """One action with both policy-space and exact simulator-space values."""

    model_space: np.ndarray
    environment_space: np.ndarray

    def __post_init__(self) -> None:
        model = np.ascontiguousarray(np.asarray(self.model_space), dtype=np.float32)
        environment = np.ascontiguousarray(
            np.asarray(self.environment_space), dtype=np.float32
        )
        if model.shape != (14,) or environment.shape != (14,):
            raise RMBenchPolicyBoundaryError(
                "queued model/environment actions must both have shape [14]"
            )
        if not np.isfinite(model).all() or not np.isfinite(environment).all():
            raise RMBenchPolicyBoundaryError("queued actions must be finite")
        object.__setattr__(self, "model_space", model)
        object.__setattr__(self, "environment_space", environment)


class RecedingHorizonQueue:
    """Bounded FIFO that never silently mixes two replans."""

    def __init__(self, *, replan_steps: int, action_horizon: int = 32) -> None:
        if isinstance(replan_steps, bool) or not isinstance(replan_steps, int):
            raise TypeError("replan_steps must be an integer")
        if not 1 <= replan_steps <= action_horizon:
            raise ValueError("replan_steps must lie in [1, action_horizon]")
        self.replan_steps = replan_steps
        self.action_horizon = action_horizon
        self._items: deque[QueuedAction] = deque()

    def __bool__(self) -> bool:
        return bool(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def clear(self) -> None:
        self._items.clear()

    def publish(self, model_chunk: Any, environment_chunk: Any) -> None:
        if self._items:
            raise RuntimeError("cannot replace a non-empty receding-horizon queue")
        model = np.asarray(model_chunk, dtype=np.float32)
        environment = np.asarray(environment_chunk, dtype=np.float32)
        expected = (self.action_horizon, 14)
        if model.shape != expected or environment.shape != expected:
            raise RMBenchPolicyBoundaryError(
                f"action chunks must both have exact shape {expected}"
            )
        if not np.isfinite(model).all() or not np.isfinite(environment).all():
            raise RMBenchPolicyBoundaryError("action chunks must be finite")
        for index in range(self.replan_steps):
            self._items.append(QueuedAction(model[index], environment[index]))

    def pop(self) -> QueuedAction:
        if not self._items:
            raise RuntimeError("cannot pop an empty action queue")
        return self._items.popleft()


class JsonlEvidenceWriter:
    """Append canonical JSON records and fsync each formal evidence boundary."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        if self.path.exists() or self.path.is_symlink():
            raise FileExistsError(
                f"refusing to mix RMBench WARM evidence in existing path: {self.path}"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation closes the check/create race.
        self._stream = self.path.open("x", encoding="utf-8", newline="\n")
        self._closed = False

    def append(self, record: Mapping[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("RMBench WARM evidence stream is closed")
        if not isinstance(record, Mapping):
            raise TypeError("evidence record must be a mapping")
        encoded = json.dumps(
            dict(record),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        self._stream.write(encoded + "\n")
        self._stream.flush()
        os.fsync(self._stream.fileno())

    def close(self) -> None:
        if not self._closed:
            self._stream.flush()
            os.fsync(self._stream.fileno())
            self._stream.close()
            self._closed = True


__all__ = [
    "JsonlEvidenceWriter",
    "PROCESSOR_CAMERA_KEYS",
    "QueuedAction",
    "RMBenchPolicyBoundaryError",
    "RecedingHorizonQueue",
    "factual_cameras",
    "factual_joint_state",
    "audit_qpos_execution",
    "replace_bridge_world_tokens_with_factual_dino",
    "resolve_task_bundle_paths",
    "validate_warm_model_telemetry",
]
