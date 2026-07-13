"""Strict validation for the M2.1 online/offline retrieval parity gate."""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any, Mapping

from .online_contract import (
    ONLINE_RETRIEVAL_IMPLEMENTATION,
    WarmOnlineRunContract,
)


ONLINE_PARITY_REPORT_SCHEMA = "warm.online-retrieval-parity-report"
ONLINE_PARITY_REPORT_VERSION = 1
MAX_BFLOAT16_ATOL = 1e-3

_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema",
        "version",
        "status",
        "scope",
        "task",
        "numeric_comparison",
        "contracts",
        "artifacts",
        "counts",
        "implementation",
    }
)
_TASK_FIELDS = frozenset({"suite", "task_id", "description"})
_NUMERIC_FIELDS = frozenset({"mode", "rtol", "atol", "score_exactness"})
_CONTRACT_FIELDS = frozenset(
    {
        "training_run_contract_sha256",
        "validation_run_contract_sha256",
        "online_run_contract_sha256",
        "evaluation_namespace_sha256",
        "encoder_contract_sha256",
        "camera_contract_sha256",
        "normalization_stats_sha256",
        "action_space_contract_sha256",
        "catalog_sha256",
        "audit_sha256",
    }
)
_ARTIFACT_FIELDS = frozenset(
    {
        "bank_manifest_sha256",
        "bank_content_sha256",
        "candidate_manifest_sha256",
        "candidate_payload_sha256",
        "dev_query_corpus_sha256",
        "dev_feature_artifact_set_sha256",
        "dataset_metadata_set_sha256",
        "raw_dev_artifact_set_sha256",
        "raw_dev_artifact_count",
        "dino_checkpoint_tree_sha256",
        "dino_checkpoint_file_count",
        "resolved_eval_config_sha256",
        "resolved_eval_config_file_sha256",
        "data_config_sha256",
    }
)
_COUNT_FIELDS = frozenset(
    {
        "dev_episode_count",
        "task_episode_count",
        "query_count",
        "candidate_slot_count",
        "valid_candidate_count",
        "exact_context_key_count",
        "exact_score_roundtrip_count",
        "max_context_key_abs_error",
        "max_cosine_score_abs_error",
        "parity_transcript_sha256",
    }
)
_IMPLEMENTATION_FIELDS = frozenset(
    {
        "retriever",
        "offline_candidate",
        "search_domain",
        "query_stride",
        "top_k",
        "git_commit",
        "git_dirty",
        "device",
        "encoder_runtime_sha256",
    }
)


class OnlineParityReportError(ValueError):
    """Raised when a purported passing parity report is incomplete."""


def _mapping(value: Any, *, field: str, expected: frozenset[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OnlineParityReportError(f"{field} must be an object")
    actual = set(value)
    if actual != expected:
        raise OnlineParityReportError(
            f"invalid {field} fields; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise OnlineParityReportError(f"{field} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise OnlineParityReportError(f"{field} must be a positive integer")
    return result


def _nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise OnlineParityReportError(f"{field} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise OnlineParityReportError(f"{field} must be a non-negative integer")
    return result


def _finite_nonnegative(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise OnlineParityReportError(f"{field} must be finite and non-negative")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise OnlineParityReportError(f"{field} must be finite and non-negative")
    return result


def _require_equal(actual: Any, expected: Any, *, field: str) -> None:
    if actual != expected:
        raise OnlineParityReportError(
            f"parity report {field} does not match the fixed online contract: "
            f"{actual!r} != {expected!r}"
        )


def validate_passing_online_parity_report(
    value: Mapping[str, Any],
    *,
    fixed_online_contract: WarmOnlineRunContract,
    expected_online_contract_sha256: str | None = None,
    expected_resolved_eval_config_sha256: str | None = None,
) -> None:
    """Fail closed unless ``value`` proves full-task M1/online parity."""

    report = _mapping(value, field="parity report", expected=_TOP_LEVEL_FIELDS)
    _require_equal(report["schema"], ONLINE_PARITY_REPORT_SCHEMA, field="schema")
    _require_equal(report["version"], ONLINE_PARITY_REPORT_VERSION, field="version")
    _require_equal(report["status"], "pass", field="status")
    _require_equal(report["scope"], "catalog-bound-dev-task", field="scope")

    task = _mapping(report["task"], field="task", expected=_TASK_FIELDS)
    _require_equal(task["suite"], fixed_online_contract.task_suite, field="task.suite")
    _require_equal(task["task_id"], fixed_online_contract.task_id, field="task.task_id")
    _require_equal(
        task["description"],
        fixed_online_contract.task_description,
        field="task.description",
    )

    contracts = _mapping(
        report["contracts"], field="contracts", expected=_CONTRACT_FIELDS
    )
    expected_online_sha = (
        fixed_online_contract.sha256
        if expected_online_contract_sha256 is None
        else expected_online_contract_sha256
    )
    expected_eval_sha = (
        fixed_online_contract.resolved_eval_config_sha256
        if expected_resolved_eval_config_sha256 is None
        else expected_resolved_eval_config_sha256
    )
    contract_expectations = {
        "training_run_contract_sha256": fixed_online_contract.training_run_contract_sha256,
        "validation_run_contract_sha256": fixed_online_contract.validation_run_contract_sha256,
        "online_run_contract_sha256": expected_online_sha,
        "evaluation_namespace_sha256": fixed_online_contract.evaluation_namespace_sha256,
        "encoder_contract_sha256": fixed_online_contract.encoder_contract_sha256,
        "camera_contract_sha256": fixed_online_contract.camera_contract_sha256,
        "normalization_stats_sha256": fixed_online_contract.normalization_stats_sha256,
        "action_space_contract_sha256": fixed_online_contract.action_space_contract_sha256,
        "catalog_sha256": fixed_online_contract.catalog_sha256,
        "audit_sha256": fixed_online_contract.audit_sha256,
    }
    for field, expected in contract_expectations.items():
        _require_equal(contracts[field], expected, field=f"contracts.{field}")

    artifacts = _mapping(
        report["artifacts"], field="artifacts", expected=_ARTIFACT_FIELDS
    )
    artifact_expectations = {
        "bank_manifest_sha256": fixed_online_contract.bank_manifest_sha256,
        "bank_content_sha256": fixed_online_contract.bank_content_sha256,
        "dino_checkpoint_tree_sha256": fixed_online_contract.dino_checkpoint_tree_sha256,
        "dino_checkpoint_file_count": fixed_online_contract.dino_checkpoint_file_count,
        "resolved_eval_config_sha256": expected_eval_sha,
        "data_config_sha256": fixed_online_contract.m1_data_config_sha256,
    }
    for field, expected in artifact_expectations.items():
        _require_equal(artifacts[field], expected, field=f"artifacts.{field}")
    _positive_int(artifacts["raw_dev_artifact_count"], field="raw_dev_artifact_count")

    implementation = _mapping(
        report["implementation"],
        field="implementation",
        expected=_IMPLEMENTATION_FIELDS,
    )
    implementation_expectations = {
        "retriever": ONLINE_RETRIEVAL_IMPLEMENTATION,
        "offline_candidate": "exact_cosine_v1",
        "search_domain": "complete_event_bank",
        "query_stride": 1,
        "top_k": fixed_online_contract.top_k,
        "git_commit": fixed_online_contract.git_commit,
        "git_dirty": False,
        "encoder_runtime_sha256": fixed_online_contract.encoder_runtime_sha256,
    }
    for field, expected in implementation_expectations.items():
        _require_equal(implementation[field], expected, field=f"implementation.{field}")
    if not isinstance(implementation["device"], str) or not implementation["device"]:
        raise OnlineParityReportError("implementation.device must be non-empty")

    numeric = _mapping(
        report["numeric_comparison"],
        field="numeric_comparison",
        expected=_NUMERIC_FIELDS,
    )
    _require_equal(numeric["rtol"], 0.0, field="numeric_comparison.rtol")
    _require_equal(
        numeric["score_exactness"],
        "float64-online-to-float32-cache-roundtrip",
        field="numeric_comparison.score_exactness",
    )
    atol = _finite_nonnegative(numeric["atol"], field="numeric_comparison.atol")
    if numeric["mode"] == "exact":
        if atol != 0.0:
            raise OnlineParityReportError("exact parity requires atol=0")
    elif numeric["mode"] == "cuda-bfloat16-atol":
        if atol <= 0.0 or atol > MAX_BFLOAT16_ATOL:
            raise OnlineParityReportError(
                "CUDA bfloat16 parity requires 0 < atol <= 1e-3"
            )
    else:
        raise OnlineParityReportError("unsupported parity numeric mode")

    counts = _mapping(report["counts"], field="counts", expected=_COUNT_FIELDS)
    dev_episodes = _positive_int(counts["dev_episode_count"], field="dev_episode_count")
    task_episodes = _positive_int(
        counts["task_episode_count"], field="task_episode_count"
    )
    queries = _positive_int(counts["query_count"], field="query_count")
    slots = _positive_int(
        counts["candidate_slot_count"], field="candidate_slot_count"
    )
    valid = _nonnegative_int(
        counts["valid_candidate_count"], field="valid_candidate_count"
    )
    exact_keys = _nonnegative_int(
        counts["exact_context_key_count"], field="exact_context_key_count"
    )
    exact_scores = _nonnegative_int(
        counts["exact_score_roundtrip_count"],
        field="exact_score_roundtrip_count",
    )
    if task_episodes > dev_episodes or valid > slots:
        raise OnlineParityReportError("parity report count relationships are invalid")
    if exact_keys > queries or exact_scores > queries:
        raise OnlineParityReportError("parity exact-count relationships are invalid")
    key_error = _finite_nonnegative(
        counts["max_context_key_abs_error"], field="max_context_key_abs_error"
    )
    score_error = _finite_nonnegative(
        counts["max_cosine_score_abs_error"], field="max_cosine_score_abs_error"
    )
    if key_error > atol or score_error > atol:
        raise OnlineParityReportError("parity numeric error exceeds declared atol")
    if numeric["mode"] == "exact" and (
        exact_keys != queries or exact_scores != queries
    ):
        raise OnlineParityReportError(
            "exact parity must report every query key and valid score as exact"
        )
    transcript = counts["parity_transcript_sha256"]
    if (
        not isinstance(transcript, str)
        or len(transcript) != 64
        or any(character not in "0123456789abcdef" for character in transcript)
    ):
        raise OnlineParityReportError(
            "parity_transcript_sha256 must be a lowercase SHA-256 digest"
        )


__all__ = [
    "MAX_BFLOAT16_ATOL",
    "ONLINE_PARITY_REPORT_SCHEMA",
    "ONLINE_PARITY_REPORT_VERSION",
    "OnlineParityReportError",
    "validate_passing_online_parity_report",
]
