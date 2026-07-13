#!/usr/bin/env python3
"""Fail-closed verification of one formal M2.1 fixed/null result pair.

The LIBERO evaluator writes one ``gpu*_task*_results.json`` file per task.
This verifier accepts either such a file or a directory containing those
files.  It validates policy-specific artifact identities and the complete
episode/replan evidence before publishing an atomic JSON report.

The paired QueryId/seed check is deliberately limited to the common replan
prefix of each episode.  It proves a shared deterministic replan namespace;
it never claims that two closed-loop policies followed the same trajectory.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
from uuid import uuid4

from fastwam.memory.manifest import sha256_canonical_json
from fastwam.memory.online_retrieval import (
    derive_online_query_seed,
    make_online_query_id,
)
from fastwam.models.warm.online_contract import WarmOnlineRunContract
from fastwam.models.warm.online_pair_contract import WarmOnlinePairContract
from fastwam.utils.artifact_claim import artifact_claim


REPORT_SCHEMA = "warm.online-pair-results-verification-report"
REPORT_SCHEMA_VERSION = 1
HEADER_SCHEMA = "warm.libero-online-evaluation-header"
HEADER_VERSION = 2

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_RESULT_PATTERN = "gpu*_task*_results.json"
_PAIR_IDENTITY_FIELDS = frozenset(
    {"pair_contract_sha256", "comparison_kind", "side"}
)
_POLICY_SPECIFIC_ONLINE_FIELDS = frozenset(
    {
        "source_policy",
        "warm_checkpoint_sha256",
        "training_attestation_sha256",
        "resolved_eval_config_sha256",
    }
)
_RESULT_FIELDS = frozenset(
    {
        "task_suite",
        "task_id",
        "task_description",
        "successes",
        "total_episodes",
        "gpu_id",
        "success_episodes",
        "failure_episodes",
        "start_time",
        "duration",
        "warm_online_header",
        "warm_online_episodes",
    }
)
_HEADER_FIELDS = frozenset(
    {
        "schema",
        "version",
        *_PAIR_IDENTITY_FIELDS,
        "online_run_contract_sha256",
        "training_run_contract_sha256",
        "validation_run_contract_sha256",
        "source_policy",
        "runtime_attestation",
        "contract",
        "pair_contract",
    }
)
_RUNTIME_ATTESTATION_FIELDS = frozenset(
    {
        "git_commit",
        "git_dirty",
        "normalization_stats_loaded_sha256",
        "pair_contract_file_sha256",
        "parity_report_file_sha256",
        "m1_data_config_sha256",
        "training_attestation_file_sha256",
        "shared_training_recipe_sha256",
        "training_runtime_sha256",
        "model_loaded_checkpoint_sha256",
        "bddl_sha256_after_environment_load",
    }
)
_EPISODE_FIELDS = frozenset(
    {
        *_PAIR_IDENTITY_FIELDS,
        "episode_index",
        "success",
        "simulator_seed",
        "termination_reason",
        "final_frame_index",
        "environment_step_count",
        "policy_action_step_count",
        "configured_wait_steps",
        "configured_policy_max_steps",
        "configured_replan_steps",
        "replan_count",
        "replans",
    }
)
_REPLAN_FIELDS = frozenset(
    {
        *_PAIR_IDENTITY_FIELDS,
        "query_id",
        "absolute_sim_step",
        "source_policy",
        "derived_seed",
        "raw_camera_sha256",
        "processed_camera_sha256",
        "prompt_sha256",
        "proprio_sha256",
        "model_input_sha256",
        "evaluator_latency_s",
        "model",
        "bound_step_sha256",
        "context_key_sha256",
        "candidate_payload_sha256",
        "candidates",
        "replan_index",
    }
)
_QUERY_FIELDS = frozenset(
    {"dataset_id", "dataset_index", "episode_index", "frame_index"}
)
_EVALUATOR_LATENCY_FIELDS = frozenset(
    {"input_prepare_or_retrieval_s", "model_inference_s", "online_pipeline_s"}
)
_MODEL_FIELDS = frozenset({"retrieval", "source"})
_SOURCE_FIELDS = frozenset(
    {
        "policy",
        "component",
        "selected_rank",
        "selected_event_id",
        "memory_selected",
        "memory_sigma",
        "derived_seed",
    }
)
_RETRIEVAL_FIELDS = frozenset(
    {
        "query_id",
        "online_contract_sha256",
        "training_run_contract_sha256",
        "validation_run_contract_sha256",
        "bank_manifest_sha256",
        "bank_content_sha256",
        "step_sha256",
        "prompt_sha256",
        "proprio_sha256",
        "model_input_sha256",
        "ranked_event_ids",
        "bank_rows",
        "cosine_scores",
        "candidate_valid_mask",
        "latency_s",
    }
)
_RETRIEVAL_LATENCY_FIELDS = frozenset(
    {"preprocess_s", "dino_s", "search_s", "gather_s", "total_s"}
)
_CANDIDATE_FIELDS = frozenset(
    {"rank", "valid", "bank_row", "event_id", "cosine_score"}
)
_EVENT_ID_FIELDS = frozenset(
    {"dataset_id", "dataset_index", "episode_index", "start_frame"}
)


class OnlinePairResultsVerificationError(RuntimeError):
    """Raised when paired rollout evidence is incomplete or inconsistent."""


@dataclass(frozen=True, slots=True)
class _InputSnapshot:
    root: Path
    files: tuple[Path, ...]
    raw_sha256: tuple[str, ...]
    payloads: tuple[Mapping[str, Any], ...]

    def attest_unchanged(self) -> None:
        current = _discover_result_files(self.root)
        if current != self.files:
            raise OnlinePairResultsVerificationError(
                f"result-file set changed during verification: {self.root}"
            )
        for path, expected in zip(self.files, self.raw_sha256, strict=True):
            if _sha256_bytes(_read_bytes(path, label="result file")) != expected:
                raise OnlinePairResultsVerificationError(
                    f"result file changed during verification: {path}"
                )


@dataclass(frozen=True, slots=True)
class _EpisodeEvidence:
    key: tuple[str, int, int]
    success: bool
    simulator_seed: int
    termination_reason: str
    final_frame_index: int
    environment_step_count: int
    policy_action_step_count: int
    configured_wait_steps: int
    configured_policy_max_steps: int
    configured_replan_steps: int
    replans: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class _ValidatedSide:
    policy: str
    result_files: tuple[Path, ...]
    result_file_sha256: tuple[str, ...]
    contracts_by_task: Mapping[tuple[str, int], WarmOnlineRunContract]
    episodes: Mapping[tuple[str, int, int], _EpisodeEvidence]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify formal M2.1 fixed-context/Gaussian-null LIBERO result "
            "evidence and atomically publish a paired report."
        )
    )
    parser.add_argument("--pair-contract", required=True, type=Path)
    parser.add_argument("--fixed-results", required=True, type=Path)
    parser.add_argument("--gaussian-null-results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_bytes(path: Path, *, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise OnlinePairResultsVerificationError(
            f"cannot read {label} at {path}"
        ) from exc


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise OnlinePairResultsVerificationError(
                f"JSON object contains duplicate key {key!r}"
            )
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise OnlinePairResultsVerificationError(
        f"JSON contains non-finite constant {value!r}"
    )


def _assert_finite_json(value: Any, *, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise OnlinePairResultsVerificationError(f"{label} contains non-finite float")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite_json(item, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite_json(item, label=f"{label}[{index}]")


def _decode_json(raw: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except OnlinePairResultsVerificationError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise OnlinePairResultsVerificationError(f"cannot decode {label} JSON") from exc
    if not isinstance(value, Mapping):
        raise OnlinePairResultsVerificationError(f"{label} must be a JSON object")
    _assert_finite_json(value, label=label)
    return value


def _discover_result_files(root: Path) -> tuple[Path, ...]:
    resolved = root.expanduser().resolve()
    if resolved.is_file():
        return (resolved,)
    if not resolved.is_dir():
        raise OnlinePairResultsVerificationError(
            f"result input is neither a file nor a directory: {resolved}"
        )
    files = tuple(
        sorted(
            (path.resolve() for path in resolved.rglob(_RESULT_PATTERN)),
            key=lambda path: str(path).casefold(),
        )
    )
    if not files:
        raise OnlinePairResultsVerificationError(
            f"result directory contains no {_RESULT_PATTERN!r} files: {resolved}"
        )
    if len(set(files)) != len(files):
        raise OnlinePairResultsVerificationError(
            f"result directory resolves duplicate files: {resolved}"
        )
    if any(not path.is_file() for path in files):
        raise OnlinePairResultsVerificationError(
            f"result directory contains a non-regular matched path: {resolved}"
        )
    return files


def _snapshot_results(root: Path) -> _InputSnapshot:
    resolved = root.expanduser().resolve()
    files = _discover_result_files(resolved)
    raw_values = tuple(_read_bytes(path, label="result file") for path in files)
    return _InputSnapshot(
        root=resolved,
        files=files,
        raw_sha256=tuple(_sha256_bytes(raw) for raw in raw_values),
        payloads=tuple(
            _decode_json(raw, label=f"result file {path}")
            for path, raw in zip(files, raw_values, strict=True)
        ),
    )


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OnlinePairResultsVerificationError(f"{label} must be an object")
    return value


def _list(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise OnlinePairResultsVerificationError(f"{label} must be an array")
    return value


def _exact_fields(value: Mapping[str, Any], expected: frozenset[str], *, label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise OnlinePairResultsVerificationError(
            f"{label} fields are not schema-exact; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _int(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise OnlinePairResultsVerificationError(
            f"{label} must be an integer >= {minimum}"
        )
    return value


def _number(value: Any, *, label: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OnlinePairResultsVerificationError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise OnlinePairResultsVerificationError(f"{label} has an invalid value")
    return result


def _bool(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise OnlinePairResultsVerificationError(f"{label} must be a boolean")
    return value


def _string(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or "\x00" in value
    ):
        raise OnlinePairResultsVerificationError(
            f"{label} must be a normalized non-empty string"
        )
    return value


def _derive_episode_simulator_seed(
    root_seed: int,
    task_suite: str,
    task_id: int,
    episode_index: int,
) -> int:
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


def _configured_policy_max_steps(task_suite: str) -> int:
    values = {
        "libero_spatial": 400,
        "libero_object": 400,
        "libero_goal": 400,
        "libero_10": 700,
        "libero_90": 700,
    }
    try:
        return values[task_suite]
    except KeyError as error:
        raise OnlinePairResultsVerificationError(
            f"unsupported task suite in episode evidence: {task_suite!r}"
        ) from error


def _digest(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise OnlinePairResultsVerificationError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return value


def _pair_identity(pair: WarmOnlinePairContract, side: str) -> dict[str, str]:
    return {
        "pair_contract_sha256": pair.sha256,
        "comparison_kind": pair.comparison_kind,
        "side": side,
    }


def _validate_pair_identity(
    value: Mapping[str, Any],
    pair: WarmOnlinePairContract,
    side: str,
    *,
    label: str,
) -> None:
    expected = _pair_identity(pair, side)
    actual = {field: value.get(field) for field in _PAIR_IDENTITY_FIELDS}
    if actual != expected:
        raise OnlinePairResultsVerificationError(
            f"{label} pair identity mismatch: {actual!r} != {expected!r}"
        )


def _validate_hash_map(value: Any, *, label: str) -> dict[str, str]:
    mapping = _mapping(value, label=label)
    if not mapping:
        raise OnlinePairResultsVerificationError(f"{label} must not be empty")
    result: dict[str, str] = {}
    for key, item in mapping.items():
        normalized = _string(key, label=f"{label} key")
        result[normalized] = _digest(item, label=f"{label}.{normalized}")
    return result


def _validate_event_id(value: Any, *, label: str) -> Mapping[str, Any] | None:
    if value is None:
        return None
    event = _mapping(value, label=label)
    _exact_fields(event, _EVENT_ID_FIELDS, label=label)
    _string(event["dataset_id"], label=f"{label}.dataset_id")
    _int(event["dataset_index"], label=f"{label}.dataset_index")
    _int(event["episode_index"], label=f"{label}.episode_index")
    _int(event["start_frame"], label=f"{label}.start_frame")
    return event


def _validate_query(
    value: Any,
    contract: WarmOnlineRunContract,
    *,
    episode_index: int,
    label: str,
) -> Mapping[str, Any]:
    query = _mapping(value, label=label)
    _exact_fields(query, _QUERY_FIELDS, label=label)
    frame_index = _int(query["frame_index"], label=f"{label}.frame_index")
    expected = make_online_query_id(contract, episode_index, frame_index)
    expected_value = {
        "dataset_id": expected.dataset_id,
        "dataset_index": expected.dataset_index,
        "episode_index": expected.episode_index,
        "frame_index": expected.frame_index,
    }
    if dict(query) != expected_value:
        raise OnlinePairResultsVerificationError(
            f"{label} does not match its online contract and episode"
        )
    return query


def _validate_latencies(
    value: Any,
    expected_fields: frozenset[str],
    *,
    label: str,
) -> None:
    mapping = _mapping(value, label=label)
    _exact_fields(mapping, expected_fields, label=label)
    for field in expected_fields:
        _number(mapping[field], label=f"{label}.{field}", minimum=0.0)


def _validate_candidates(
    value: Any,
    contract: WarmOnlineRunContract,
    *,
    label: str,
) -> tuple[Mapping[str, Any], ...]:
    candidates = _list(value, label=label)
    if len(candidates) != contract.top_k:
        raise OnlinePairResultsVerificationError(
            f"{label} length must equal contract.top_k={contract.top_k}"
        )
    result: list[Mapping[str, Any]] = []
    saw_invalid = False
    valid_bank_rows: set[int] = set()
    previous_valid_score: float | None = None
    for rank, raw in enumerate(candidates):
        candidate = _mapping(raw, label=f"{label}[{rank}]")
        _exact_fields(candidate, _CANDIDATE_FIELDS, label=f"{label}[{rank}]")
        if _int(candidate["rank"], label=f"{label}[{rank}].rank") != rank:
            raise OnlinePairResultsVerificationError(
                f"{label}[{rank}].rank is not contiguous"
            )
        valid = _bool(candidate["valid"], label=f"{label}[{rank}].valid")
        if saw_invalid and valid:
            raise OnlinePairResultsVerificationError(
                f"{label} validity mask is not a true prefix"
            )
        saw_invalid = saw_invalid or not valid
        bank_row = _int(
            candidate["bank_row"],
            label=f"{label}[{rank}].bank_row",
            minimum=-1,
        )
        event_id = _validate_event_id(
            candidate["event_id"], label=f"{label}[{rank}].event_id"
        )
        score = _number(
            candidate["cosine_score"], label=f"{label}[{rank}].cosine_score"
        )
        if valid:
            if bank_row < 0 or event_id is None:
                raise OnlinePairResultsVerificationError(
                    f"{label}[{rank}] valid candidate has no factual payload identity"
                )
            if not -1.0 <= score <= 1.0:
                raise OnlinePairResultsVerificationError(
                    f"{label}[{rank}].cosine_score is outside [-1, 1]"
                )
            if bank_row in valid_bank_rows:
                raise OnlinePairResultsVerificationError(
                    f"{label} valid candidates contain a duplicate bank row"
                )
            if previous_valid_score is not None and score > previous_valid_score:
                raise OnlinePairResultsVerificationError(
                    f"{label} valid cosine scores are not non-increasing"
                )
            valid_bank_rows.add(bank_row)
            previous_valid_score = score
        elif bank_row != -1 or event_id is not None or score != 0.0:
            raise OnlinePairResultsVerificationError(
                f"{label}[{rank}] invalid padding is not canonical"
            )
        result.append(candidate)
    return tuple(result)


def _validate_model_telemetry(
    value: Any,
    *,
    side: str,
    contract: WarmOnlineRunContract,
    query: Mapping[str, Any],
    derived_seed: int,
    prompt_sha256: str,
    proprio_sha256: str,
    model_input_sha256: str,
    bound_step_sha256: str | None,
    candidates: tuple[Mapping[str, Any], ...],
    label: str,
) -> None:
    model = _mapping(value, label=label)
    _exact_fields(model, _MODEL_FIELDS, label=label)
    source = _mapping(model["source"], label=f"{label}.source")
    _exact_fields(source, _SOURCE_FIELDS, label=f"{label}.source")
    if source["policy"] != contract.source_policy:
        raise OnlinePairResultsVerificationError(f"{label}.source policy mismatch")
    component = _int(source["component"], label=f"{label}.source.component")
    if component not in {0, 1}:
        raise OnlinePairResultsVerificationError(
            f"{label}.source.component must be zero or one"
        )
    if _bool(source["memory_selected"], label=f"{label}.source.memory_selected") != (
        component == 1
    ):
        raise OnlinePairResultsVerificationError(
            f"{label}.source memory_selected/component mismatch"
        )
    if _number(source["memory_sigma"], label=f"{label}.source.memory_sigma") != (
        contract.memory_sigma
    ):
        raise OnlinePairResultsVerificationError(f"{label}.source memory_sigma mismatch")
    if _int(source["derived_seed"], label=f"{label}.source.derived_seed") != derived_seed:
        raise OnlinePairResultsVerificationError(f"{label}.source derived_seed mismatch")

    if side == "gaussian_null":
        if model["retrieval"] is not None:
            raise OnlinePairResultsVerificationError(
                f"{label}.retrieval proves a forbidden null-side memory read"
            )
        if component != 0:
            raise OnlinePairResultsVerificationError(
                f"{label}.source null component must be zero"
            )
        if source["selected_rank"] is not None or source["selected_event_id"] is not None:
            raise OnlinePairResultsVerificationError(
                f"{label}.source null side must not select memory"
            )
        return

    retrieval = _mapping(model["retrieval"], label=f"{label}.retrieval")
    _exact_fields(retrieval, _RETRIEVAL_FIELDS, label=f"{label}.retrieval")
    if retrieval["query_id"] != query:
        raise OnlinePairResultsVerificationError(f"{label}.retrieval query mismatch")
    expected_hashes = {
        "online_contract_sha256": contract.sha256,
        "training_run_contract_sha256": contract.training_run_contract_sha256,
        "validation_run_contract_sha256": contract.validation_run_contract_sha256,
        "bank_manifest_sha256": contract.bank_manifest_sha256,
        "bank_content_sha256": contract.bank_content_sha256,
        "step_sha256": bound_step_sha256,
        "prompt_sha256": prompt_sha256,
        "proprio_sha256": proprio_sha256,
        "model_input_sha256": model_input_sha256,
    }
    for field, expected in expected_hashes.items():
        if retrieval[field] != expected:
            raise OnlinePairResultsVerificationError(
                f"{label}.retrieval {field} mismatch"
            )
    expected_events = [candidate["event_id"] for candidate in candidates]
    expected_rows = [candidate["bank_row"] for candidate in candidates]
    expected_scores = [candidate["cosine_score"] for candidate in candidates]
    expected_valid = [candidate["valid"] for candidate in candidates]
    if retrieval["ranked_event_ids"] != expected_events:
        raise OnlinePairResultsVerificationError(
            f"{label}.retrieval ranked event identities mismatch"
        )
    if retrieval["bank_rows"] != expected_rows:
        raise OnlinePairResultsVerificationError(
            f"{label}.retrieval bank rows mismatch"
        )
    if retrieval["cosine_scores"] != expected_scores:
        raise OnlinePairResultsVerificationError(
            f"{label}.retrieval cosine scores mismatch"
        )
    if retrieval["candidate_valid_mask"] != expected_valid:
        raise OnlinePairResultsVerificationError(
            f"{label}.retrieval candidate mask mismatch"
        )
    _validate_latencies(
        retrieval["latency_s"],
        _RETRIEVAL_LATENCY_FIELDS,
        label=f"{label}.retrieval.latency_s",
    )
    any_valid = bool(candidates and candidates[0]["valid"])
    if component != int(any_valid):
        raise OnlinePairResultsVerificationError(
            f"{label}.source component does not reflect fixed top-1 availability"
        )
    expected_rank = 0 if any_valid else None
    expected_event = candidates[0]["event_id"] if any_valid else None
    if source["selected_rank"] != expected_rank:
        raise OnlinePairResultsVerificationError(
            f"{label}.source selected rank mismatch"
        )
    if source["selected_event_id"] != expected_event:
        raise OnlinePairResultsVerificationError(
            f"{label}.source selected event mismatch"
        )


def _validate_replan(
    value: Any,
    *,
    side: str,
    pair: WarmOnlinePairContract,
    contract: WarmOnlineRunContract,
    episode_index: int,
    replan_index: int,
    previous_frame_index: int | None,
    label: str,
) -> Mapping[str, Any]:
    replan = _mapping(value, label=label)
    _exact_fields(replan, _REPLAN_FIELDS, label=label)
    _validate_pair_identity(replan, pair, side, label=label)
    if _int(replan["replan_index"], label=f"{label}.replan_index") != replan_index:
        raise OnlinePairResultsVerificationError(f"{label}.replan_index is not contiguous")
    if replan["source_policy"] != contract.source_policy:
        raise OnlinePairResultsVerificationError(f"{label}.source_policy mismatch")
    query = _validate_query(
        replan["query_id"], contract, episode_index=episode_index, label=f"{label}.query_id"
    )
    frame_index = query["frame_index"]
    if previous_frame_index is not None and frame_index <= previous_frame_index:
        raise OnlinePairResultsVerificationError(
            f"{label}.query_id.frame_index is not strictly increasing"
        )
    if _int(replan["absolute_sim_step"], label=f"{label}.absolute_sim_step") != frame_index:
        raise OnlinePairResultsVerificationError(f"{label}.absolute_sim_step mismatch")
    expected_seed = derive_online_query_seed(
        contract.root_seed,
        make_online_query_id(contract, episode_index, frame_index),
        contract.evaluation_namespace_sha256,
    )
    derived_seed = _int(replan["derived_seed"], label=f"{label}.derived_seed")
    if derived_seed != expected_seed:
        raise OnlinePairResultsVerificationError(f"{label}.derived_seed mismatch")
    raw_camera = _validate_hash_map(
        replan["raw_camera_sha256"], label=f"{label}.raw_camera_sha256"
    )
    processed_camera = _validate_hash_map(
        replan["processed_camera_sha256"], label=f"{label}.processed_camera_sha256"
    )
    if set(raw_camera) != set(processed_camera):
        raise OnlinePairResultsVerificationError(
            f"{label} raw/processed camera key sets differ"
        )
    prompt_sha256 = _digest(replan["prompt_sha256"], label=f"{label}.prompt_sha256")
    proprio_sha256 = _digest(
        replan["proprio_sha256"], label=f"{label}.proprio_sha256"
    )
    model_input_sha256 = _digest(
        replan["model_input_sha256"], label=f"{label}.model_input_sha256"
    )
    _validate_latencies(
        replan["evaluator_latency_s"],
        _EVALUATOR_LATENCY_FIELDS,
        label=f"{label}.evaluator_latency_s",
    )
    if side == "gaussian_null":
        if (
            replan["bound_step_sha256"] is not None
            or replan["context_key_sha256"] is not None
            or replan["candidate_payload_sha256"] is not None
            or replan["candidates"] != []
        ):
            raise OnlinePairResultsVerificationError(
                f"{label} contains forbidden Gaussian-null memory-read telemetry"
            )
        bound_step_sha256 = None
        candidates: tuple[Mapping[str, Any], ...] = ()
    else:
        # BoundOnlineStep includes full gathered payload tensors which are not
        # duplicated in this compact result record.  Its digest is therefore
        # treated as opaque emitter evidence: we validate its form and equality
        # with model retrieval telemetry, but do not claim to recompute it here.
        bound_step_sha256 = _digest(
            replan["bound_step_sha256"], label=f"{label}.bound_step_sha256"
        )
        _digest(replan["context_key_sha256"], label=f"{label}.context_key_sha256")
        _digest(
            replan["candidate_payload_sha256"],
            label=f"{label}.candidate_payload_sha256",
        )
        candidates = _validate_candidates(
            replan["candidates"], contract, label=f"{label}.candidates"
        )
    _validate_model_telemetry(
        replan["model"],
        side=side,
        contract=contract,
        query=query,
        derived_seed=derived_seed,
        prompt_sha256=prompt_sha256,
        proprio_sha256=proprio_sha256,
        model_input_sha256=model_input_sha256,
        bound_step_sha256=bound_step_sha256,
        candidates=candidates,
        label=f"{label}.model",
    )
    return replan


def _side_contract_identity(
    pair: WarmOnlinePairContract, side: str
) -> tuple[str, str, str, str, str]:
    if side == "fixed":
        return (
            "fixed_context_top1",
            pair.fixed_online_run_contract_sha256,
            pair.fixed_resolved_eval_config_sha256,
            pair.fixed_warm_checkpoint_sha256,
            pair.fixed_training_attestation_sha256,
        )
    return (
        "gaussian_null",
        pair.gaussian_null_online_run_contract_sha256,
        pair.gaussian_null_resolved_eval_config_sha256,
        pair.gaussian_null_warm_checkpoint_sha256,
        pair.gaussian_null_training_attestation_sha256,
    )


def _validate_header(
    value: Any,
    *,
    side: str,
    pair: WarmOnlinePairContract,
    pair_file_sha256: str,
    label: str,
) -> WarmOnlineRunContract:
    header = _mapping(value, label=label)
    _exact_fields(header, _HEADER_FIELDS, label=label)
    if header["schema"] != HEADER_SCHEMA or header["version"] != HEADER_VERSION:
        raise OnlinePairResultsVerificationError(f"{label} schema/version mismatch")
    _validate_pair_identity(header, pair, side, label=label)
    embedded_pair = WarmOnlinePairContract.from_dict(
        _mapping(header["pair_contract"], label=f"{label}.pair_contract")
    )
    if embedded_pair.to_dict() != pair.to_dict():
        raise OnlinePairResultsVerificationError(
            f"{label}.pair_contract does not match the supplied pair contract"
        )
    contract = WarmOnlineRunContract.from_dict(
        _mapping(header["contract"], label=f"{label}.contract")
    )
    expected_policy, online_hash, config_hash, checkpoint_hash, training_hash = (
        _side_contract_identity(pair, side)
    )
    expected_header = {
        "online_run_contract_sha256": online_hash,
        "training_run_contract_sha256": contract.training_run_contract_sha256,
        "validation_run_contract_sha256": contract.validation_run_contract_sha256,
        "source_policy": expected_policy,
    }
    for field, expected in expected_header.items():
        if header[field] != expected:
            raise OnlinePairResultsVerificationError(f"{label}.{field} mismatch")
    if contract.sha256 != online_hash:
        raise OnlinePairResultsVerificationError(
            f"{label} embedded online contract identity mismatch"
        )
    if contract.source_policy != expected_policy:
        raise OnlinePairResultsVerificationError(f"{label} contract policy mismatch")
    if contract.resolved_eval_config_sha256 != config_hash:
        raise OnlinePairResultsVerificationError(f"{label} resolved config mismatch")
    if contract.warm_checkpoint_sha256 != checkpoint_hash:
        raise OnlinePairResultsVerificationError(f"{label} checkpoint mismatch")
    if contract.training_attestation_sha256 != training_hash:
        raise OnlinePairResultsVerificationError(
            f"{label} training attestation mismatch"
        )
    if (
        contract.shared_training_recipe_sha256
        != pair.shared_training_recipe_sha256
        or contract.training_runtime_sha256
        != pair.shared_training_runtime_sha256
    ):
        raise OnlinePairResultsVerificationError(
            f"{label} shared training fairness identity mismatch"
        )
    attestation = _mapping(
        header["runtime_attestation"], label=f"{label}.runtime_attestation"
    )
    _exact_fields(
        attestation, _RUNTIME_ATTESTATION_FIELDS, label=f"{label}.runtime_attestation"
    )
    expected_attestation = {
        "git_commit": pair.git_commit,
        "git_dirty": False,
        "normalization_stats_loaded_sha256": contract.normalization_stats_sha256,
        "pair_contract_file_sha256": pair_file_sha256,
        "parity_report_file_sha256": pair.parity_report_sha256,
        "m1_data_config_sha256": contract.m1_data_config_sha256,
        "training_attestation_file_sha256": training_hash,
        "shared_training_recipe_sha256": pair.shared_training_recipe_sha256,
        "training_runtime_sha256": pair.shared_training_runtime_sha256,
        "model_loaded_checkpoint_sha256": checkpoint_hash,
        "bddl_sha256_after_environment_load": contract.bddl_sha256,
    }
    if dict(attestation) != expected_attestation:
        raise OnlinePairResultsVerificationError(
            f"{label}.runtime_attestation mismatch"
        )
    if contract.git_commit != pair.git_commit or contract.git_dirty:
        raise OnlinePairResultsVerificationError(f"{label} Git identity mismatch")
    return contract


def _validate_result_payload(
    value: Mapping[str, Any],
    *,
    side: str,
    pair: WarmOnlinePairContract,
    pair_file_sha256: str,
    label: str,
) -> tuple[tuple[str, int], WarmOnlineRunContract, dict[tuple[str, int, int], _EpisodeEvidence]]:
    _exact_fields(value, _RESULT_FIELDS, label=label)
    task_suite = _string(value["task_suite"], label=f"{label}.task_suite")
    task_id = _int(value["task_id"], label=f"{label}.task_id")
    task_description = _string(
        value["task_description"], label=f"{label}.task_description"
    )
    _int(value["gpu_id"], label=f"{label}.gpu_id")
    _string(value["start_time"], label=f"{label}.start_time")
    _number(value["duration"], label=f"{label}.duration", minimum=0.0)
    contract = _validate_header(
        value["warm_online_header"],
        side=side,
        pair=pair,
        pair_file_sha256=pair_file_sha256,
        label=f"{label}.warm_online_header",
    )
    if (
        task_suite != contract.task_suite
        or task_id != contract.task_id
        or task_description != contract.task_description
    ):
        raise OnlinePairResultsVerificationError(
            f"{label} task identity does not match its online contract"
        )
    total = _int(value["total_episodes"], label=f"{label}.total_episodes", minimum=1)
    successes = _int(value["successes"], label=f"{label}.successes")
    success_indices = _list(value["success_episodes"], label=f"{label}.success_episodes")
    failure_indices = _list(value["failure_episodes"], label=f"{label}.failure_episodes")
    success_set = {
        _int(index, label=f"{label}.success_episodes[]") for index in success_indices
    }
    failure_set = {
        _int(index, label=f"{label}.failure_episodes[]") for index in failure_indices
    }
    if len(success_set) != len(success_indices) or len(failure_set) != len(failure_indices):
        raise OnlinePairResultsVerificationError(f"{label} has duplicate outcome indices")
    if success_set & failure_set or success_set | failure_set != set(range(total)):
        raise OnlinePairResultsVerificationError(
            f"{label} success/failure indices are not an exact episode partition"
        )
    if successes != len(success_set):
        raise OnlinePairResultsVerificationError(f"{label}.successes count mismatch")

    raw_episodes = _list(
        value["warm_online_episodes"], label=f"{label}.warm_online_episodes"
    )
    if len(raw_episodes) != total:
        raise OnlinePairResultsVerificationError(
            f"{label}.warm_online_episodes is incomplete"
        )
    episodes: dict[tuple[str, int, int], _EpisodeEvidence] = {}
    for position, raw_episode in enumerate(raw_episodes):
        episode_label = f"{label}.warm_online_episodes[{position}]"
        episode = _mapping(raw_episode, label=episode_label)
        _exact_fields(episode, _EPISODE_FIELDS, label=episode_label)
        _validate_pair_identity(episode, pair, side, label=episode_label)
        episode_index = _int(
            episode["episode_index"], label=f"{episode_label}.episode_index"
        )
        if episode_index >= total:
            raise OnlinePairResultsVerificationError(
                f"{episode_label}.episode_index is out of range"
            )
        success = _bool(episode["success"], label=f"{episode_label}.success")
        if success != (episode_index in success_set):
            raise OnlinePairResultsVerificationError(
                f"{episode_label}.success disagrees with the result partition"
            )
        simulator_seed = _int(
            episode["simulator_seed"], label=f"{episode_label}.simulator_seed"
        )
        expected_simulator_seed = _derive_episode_simulator_seed(
            contract.root_seed,
            contract.task_suite,
            contract.task_id,
            episode_index,
        )
        if simulator_seed != expected_simulator_seed:
            raise OnlinePairResultsVerificationError(
                f"{episode_label}.simulator_seed does not match its policy-independent namespace"
            )
        termination_reason = _string(
            episode["termination_reason"],
            label=f"{episode_label}.termination_reason",
        )
        if termination_reason not in {"success", "max_steps"}:
            raise OnlinePairResultsVerificationError(
                f"{episode_label}.termination_reason is unsupported"
            )
        if success != (termination_reason == "success"):
            raise OnlinePairResultsVerificationError(
                f"{episode_label}.termination_reason disagrees with success"
            )
        final_frame_index = _int(
            episode["final_frame_index"],
            label=f"{episode_label}.final_frame_index",
        )
        environment_steps = _int(
            episode["environment_step_count"],
            label=f"{episode_label}.environment_step_count",
            minimum=1,
        )
        policy_steps = _int(
            episode["policy_action_step_count"],
            label=f"{episode_label}.policy_action_step_count",
        )
        wait_steps = _int(
            episode["configured_wait_steps"],
            label=f"{episode_label}.configured_wait_steps",
        )
        policy_max_steps = _int(
            episode["configured_policy_max_steps"],
            label=f"{episode_label}.configured_policy_max_steps",
            minimum=1,
        )
        replan_steps = _int(
            episode["configured_replan_steps"],
            label=f"{episode_label}.configured_replan_steps",
            minimum=1,
        )
        expected_max_steps = _configured_policy_max_steps(contract.task_suite)
        if policy_max_steps != expected_max_steps:
            raise OnlinePairResultsVerificationError(
                f"{episode_label}.configured_policy_max_steps does not match the evaluator suite"
            )
        wait_steps_executed = environment_steps - policy_steps
        if not 0 <= wait_steps_executed <= wait_steps:
            raise OnlinePairResultsVerificationError(
                f"{episode_label} environment/policy step counts are inconsistent"
            )
        if policy_steps > 0 and wait_steps_executed != wait_steps:
            raise OnlinePairResultsVerificationError(
                f"{episode_label} policy actions began before the configured wait completed"
            )
        if termination_reason == "max_steps":
            if (
                policy_steps != policy_max_steps
                or wait_steps_executed != wait_steps
                or environment_steps != wait_steps + policy_max_steps
                or final_frame_index != environment_steps
            ):
                raise OnlinePairResultsVerificationError(
                    f"{episode_label} max_steps termination counters are inconsistent"
                )
        elif policy_steps == 0:
            if wait_steps == 0 or final_frame_index != environment_steps:
                raise OnlinePairResultsVerificationError(
                    f"{episode_label} wait-phase success counters are inconsistent"
                )
        elif (
            policy_steps > policy_max_steps
            or final_frame_index != environment_steps - 1
        ):
            raise OnlinePairResultsVerificationError(
                f"{episode_label} policy-phase success counters are inconsistent"
            )
        raw_replans = _list(episode["replans"], label=f"{episode_label}.replans")
        replan_count = _int(
            episode["replan_count"], label=f"{episode_label}.replan_count"
        )
        if replan_count != len(raw_replans):
            raise OnlinePairResultsVerificationError(
                f"{episode_label}.replan_count does not match replans"
            )
        expected_replan_count = (
            0
            if policy_steps == 0
            else (policy_steps + replan_steps - 1) // replan_steps
        )
        if replan_count != expected_replan_count:
            raise OnlinePairResultsVerificationError(
                f"{episode_label}.replan_count does not match policy/replan steps"
            )
        if policy_steps == 0 and raw_replans:
            raise OnlinePairResultsVerificationError(
                f"{episode_label} records replans without policy actions"
            )
        if policy_steps > 0 and not raw_replans:
            raise OnlinePairResultsVerificationError(
                f"{episode_label}.replans must contain policy evidence"
            )
        replans: list[Mapping[str, Any]] = []
        previous_frame: int | None = None
        for replan_index, raw_replan in enumerate(raw_replans):
            replan = _validate_replan(
                raw_replan,
                side=side,
                pair=pair,
                contract=contract,
                episode_index=episode_index,
                replan_index=replan_index,
                previous_frame_index=previous_frame,
                label=f"{episode_label}.replans[{replan_index}]",
            )
            previous_frame = replan["query_id"]["frame_index"]
            replans.append(replan)
        if replans:
            frames = [replan["query_id"]["frame_index"] for replan in replans]
            expected_frames = [
                wait_steps + index * replan_steps
                for index in range(replan_count)
            ]
            if frames != expected_frames or frames[-1] > final_frame_index:
                raise OnlinePairResultsVerificationError(
                    f"{episode_label} replan frames disagree with episode counters"
                )
        key = (task_suite, task_id, episode_index)
        if key in episodes:
            raise OnlinePairResultsVerificationError(f"{label} has duplicate episode {key}")
        episodes[key] = _EpisodeEvidence(
            key=key,
            success=success,
            simulator_seed=simulator_seed,
            termination_reason=termination_reason,
            final_frame_index=final_frame_index,
            environment_step_count=environment_steps,
            policy_action_step_count=policy_steps,
            configured_wait_steps=wait_steps,
            configured_policy_max_steps=policy_max_steps,
            configured_replan_steps=replan_steps,
            replans=tuple(replans),
        )
    if {key[2] for key in episodes} != set(range(total)):
        raise OnlinePairResultsVerificationError(f"{label} episode set is incomplete")
    return (task_suite, task_id), contract, episodes


def _validate_side(
    snapshot: _InputSnapshot,
    *,
    side: str,
    pair: WarmOnlinePairContract,
    pair_file_sha256: str,
) -> _ValidatedSide:
    contracts: dict[tuple[str, int], WarmOnlineRunContract] = {}
    episodes: dict[tuple[str, int, int], _EpisodeEvidence] = {}
    for path, payload in zip(snapshot.files, snapshot.payloads, strict=True):
        task_key, contract, task_episodes = _validate_result_payload(
            payload,
            side=side,
            pair=pair,
            pair_file_sha256=pair_file_sha256,
            label=f"{side} result {path}",
        )
        if task_key in contracts:
            raise OnlinePairResultsVerificationError(
                f"{side} results contain duplicate task {task_key}"
            )
        overlap = set(episodes) & set(task_episodes)
        if overlap:
            raise OnlinePairResultsVerificationError(
                f"{side} results contain duplicate episode keys: {sorted(overlap)}"
            )
        contracts[task_key] = contract
        episodes.update(task_episodes)
    policy = "fixed_context_top1" if side == "fixed" else "gaussian_null"
    return _ValidatedSide(
        policy=policy,
        result_files=snapshot.files,
        result_file_sha256=snapshot.raw_sha256,
        contracts_by_task=contracts,
        episodes=episodes,
    )


def _shared_science_identity(contract: WarmOnlineRunContract) -> Mapping[str, Any]:
    value = contract.to_dict()
    for field in _POLICY_SPECIFIC_ONLINE_FIELDS:
        value.pop(field)
    return value


def _query_seed_prefix_record(replan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "replan_index": replan["replan_index"],
        "query_id": replan["query_id"],
        "derived_seed": replan["derived_seed"],
    }


def _pair_sides(
    fixed: _ValidatedSide,
    null: _ValidatedSide,
    pair: WarmOnlinePairContract,
) -> list[dict[str, Any]]:
    if set(fixed.contracts_by_task) != set(null.contracts_by_task):
        raise OnlinePairResultsVerificationError(
            "fixed/null task key sets are not identical"
        )
    if set(fixed.episodes) != set(null.episodes):
        raise OnlinePairResultsVerificationError(
            "fixed/null task/episode key sets are not identical"
        )
    for task_key in sorted(fixed.contracts_by_task):
        fixed_contract = fixed.contracts_by_task[task_key]
        null_contract = null.contracts_by_task[task_key]
        fixed_shared = _shared_science_identity(fixed_contract)
        null_shared = _shared_science_identity(null_contract)
        if fixed_shared != null_shared:
            raise OnlinePairResultsVerificationError(
                f"fixed/null shared science identity differs for task {task_key}"
            )
        if sha256_canonical_json(fixed_shared) != pair.shared_science_identity_sha256:
            raise OnlinePairResultsVerificationError(
                f"task {task_key} shared science identity does not match pair contract"
            )

    report_rows: list[dict[str, Any]] = []
    for key in sorted(fixed.episodes):
        fixed_episode = fixed.episodes[key]
        null_episode = null.episodes[key]
        fixed_shared_episode = (
            fixed_episode.simulator_seed,
            fixed_episode.configured_wait_steps,
            fixed_episode.configured_policy_max_steps,
            fixed_episode.configured_replan_steps,
        )
        null_shared_episode = (
            null_episode.simulator_seed,
            null_episode.configured_wait_steps,
            null_episode.configured_policy_max_steps,
            null_episode.configured_replan_steps,
        )
        if fixed_shared_episode != null_shared_episode:
            raise OnlinePairResultsVerificationError(
                f"fixed/null simulator seed or configured episode limits differ at {key}"
            )
        fixed_has_replans = bool(fixed_episode.replans)
        null_has_replans = bool(null_episode.replans)
        if fixed_has_replans != null_has_replans:
            raise OnlinePairResultsVerificationError(
                f"exactly one policy records a pre-policy replan at {key}"
            )
        if fixed_has_replans:
            initial_fields = (
                "raw_camera_sha256",
                "processed_camera_sha256",
                "model_input_sha256",
                "proprio_sha256",
                "prompt_sha256",
            )
            fixed_initial = {
                field: fixed_episode.replans[0][field] for field in initial_fields
            }
            null_initial = {
                field: null_episode.replans[0][field] for field in initial_fields
            }
            if fixed_initial != null_initial:
                raise OnlinePairResultsVerificationError(
                    f"fixed/null first pre-policy observation evidence differs at {key}"
                )
        common_count = min(len(fixed_episode.replans), len(null_episode.replans))
        common_prefix: list[dict[str, Any]] = []
        for index in range(common_count):
            fixed_record = _query_seed_prefix_record(fixed_episode.replans[index])
            null_record = _query_seed_prefix_record(null_episode.replans[index])
            if fixed_record != null_record:
                raise OnlinePairResultsVerificationError(
                    f"fixed/null QueryId or seed differs in common prefix at {key}, "
                    f"replan {index}"
                )
            common_prefix.append(fixed_record)
        fixed_count = len(fixed_episode.replans)
        null_count = len(null_episode.replans)
        if fixed_count == null_count:
            relation = "equal_replan_count"
        elif fixed_count < null_count:
            relation = "fixed_prefix_ended_first"
        else:
            relation = "gaussian_null_prefix_ended_first"
        report_rows.append(
            {
                "task_suite": key[0],
                "task_id": key[1],
                "episode_index": key[2],
                "simulator_seed": fixed_episode.simulator_seed,
                "configured_wait_steps": fixed_episode.configured_wait_steps,
                "configured_policy_max_steps": (
                    fixed_episode.configured_policy_max_steps
                ),
                "configured_replan_steps": fixed_episode.configured_replan_steps,
                "fixed_success": fixed_episode.success,
                "gaussian_null_success": null_episode.success,
                "fixed_replan_count": fixed_count,
                "gaussian_null_replan_count": null_count,
                "common_replan_prefix_count": common_count,
                "replan_length_relation": relation,
                "common_query_seed_prefix_sha256": sha256_canonical_json(
                    {"records": common_prefix}
                ),
                "first_common_query": (
                    None if not common_prefix else common_prefix[0]
                ),
                "last_common_query": (
                    None if not common_prefix else common_prefix[-1]
                ),
                "trajectory_equivalence_claim": False,
            }
        )
    return report_rows


def _side_report(side: _ValidatedSide) -> dict[str, Any]:
    return {
        "source_policy": side.policy,
        "result_files": [str(path) for path in side.result_files],
        "result_file_sha256": list(side.result_file_sha256),
        "task_artifact_identities": [
            {
                "task_suite": task_key[0],
                "task_id": task_key[1],
                "online_run_contract_sha256": contract.sha256,
                "resolved_eval_config_sha256": (
                    contract.resolved_eval_config_sha256
                ),
                "warm_checkpoint_sha256": contract.warm_checkpoint_sha256,
                "training_run_contract_sha256": (
                    contract.training_run_contract_sha256
                ),
                "validation_run_contract_sha256": (
                    contract.validation_run_contract_sha256
                ),
            }
            for task_key, contract in sorted(side.contracts_by_task.items())
        ],
        "task_count": len(side.contracts_by_task),
        "episode_count": len(side.episodes),
        "replan_count": sum(len(episode.replans) for episode in side.episodes.values()),
    }


def _write_atomic(path: Path, value: Mapping[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"verification report already exists at {path}")
    encoded = (
        json.dumps(
            dict(value),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
            temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pair_path = args.pair_contract.expanduser().resolve()
    if not pair_path.is_file():
        raise OnlinePairResultsVerificationError(
            f"pair contract is not a regular file: {pair_path}"
        )
    pair_raw = _read_bytes(pair_path, label="pair contract")
    pair_file_sha256 = _sha256_bytes(pair_raw)
    pair = WarmOnlinePairContract.from_dict(
        _decode_json(pair_raw, label="pair contract")
    )
    fixed_snapshot = _snapshot_results(args.fixed_results)
    null_snapshot = _snapshot_results(args.gaussian_null_results)
    overlap = set(fixed_snapshot.files) & set(null_snapshot.files)
    if overlap:
        raise OnlinePairResultsVerificationError(
            f"fixed/null inputs overlap: {sorted(str(path) for path in overlap)}"
        )
    output = args.output.expanduser().resolve()
    inputs = {pair_path, *fixed_snapshot.files, *null_snapshot.files}
    if output in inputs:
        raise OnlinePairResultsVerificationError(
            "verification output must not overwrite an input artifact"
        )
    for snapshot in (fixed_snapshot, null_snapshot):
        if snapshot.root.is_dir() and output.is_relative_to(snapshot.root):
            raise OnlinePairResultsVerificationError(
                "verification output must be outside result input directories"
            )

    fixed = _validate_side(
        fixed_snapshot,
        side="fixed",
        pair=pair,
        pair_file_sha256=pair_file_sha256,
    )
    null = _validate_side(
        null_snapshot,
        side="gaussian_null",
        pair=pair,
        pair_file_sha256=pair_file_sha256,
    )
    episode_rows = _pair_sides(fixed, null, pair)
    report = {
        "schema": REPORT_SCHEMA,
        "version": REPORT_SCHEMA_VERSION,
        "status": "verified",
        "pair_contract_path": str(pair_path),
        "pair_contract_file_sha256": pair_file_sha256,
        "pair_contract_sha256": pair.sha256,
        "comparison_kind": pair.comparison_kind,
        "parity_report_sha256": pair.parity_report_sha256,
        "shared_science_identity_sha256": pair.shared_science_identity_sha256,
        "pairing_scope": (
            "first_pre_policy_inputs_and_common_query_id_seed_replan_prefix"
        ),
        "bound_step_verification_scope": (
            "opaque_emitter_digest_and_model_retrieval_equality_only"
        ),
        "trajectory_equivalence_claim": False,
        "fixed": _side_report(fixed),
        "gaussian_null": _side_report(null),
        "task_episode_key_count": len(episode_rows),
        "episodes": episode_rows,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.parent / f".{output.name}.warm-artifact.lock"
    with artifact_claim(lock_path, purpose=f"publish WARM pair verification: {output}"):
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"verification report already exists at {output}")
        if _sha256_bytes(_read_bytes(pair_path, label="pair contract")) != pair_file_sha256:
            raise OnlinePairResultsVerificationError(
                "pair contract changed during verification"
            )
        fixed_snapshot.attest_unchanged()
        null_snapshot.attest_unchanged()
        _write_atomic(output, report, overwrite=args.overwrite)

    print(
        json.dumps(
            {
                "schema": REPORT_SCHEMA,
                "version": REPORT_SCHEMA_VERSION,
                "status": "verified",
                "output": str(output),
                "pair_contract_sha256": pair.sha256,
                "task_episode_key_count": len(episode_rows),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
