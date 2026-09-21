#!/usr/bin/env python3
"""Validate M2.1 online frozen-DINO retrieval against immutable M1 DEV artifacts.

This is a server gate, not an alternative retrieval implementation.  Every
online result is produced by :class:`FrozenDinoOnlineRetriever`; every offline
reference row is resolved by :class:`RuntimeCandidateResolver`.  Both sides
use the M1 ``exact_cosine_v1`` full-bank search semantics: task identity is
part of a task-conditioned context key, never a separate ``TASK_INDEX`` hard
filter.  The validator only compares their contract-bound outputs and
publishes a report after all input artifacts have been re-hashed.

The module remains importable on a CPU-only development machine.  Torch,
Hydra, Transformers, PyArrow, and the video decoder are reached only from the
server execution path in :func:`main`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

import numpy as np

from fastwam.datasets.lerobot.audit import (
    LerobotAuditReport,
    load_audit_report,
    resolve_episode_video_paths,
)
from fastwam.datasets.lerobot.episode_catalog import EpisodeCatalog, EpisodeRecord
from fastwam.memory.candidate_cache import (
    MANIFEST_FILENAME as CANDIDATE_MANIFEST_FILENAME,
    CandidateCacheManifest,
    QueryId,
)
from fastwam.memory.event_bank import MANIFEST_FILENAME as BANK_MANIFEST_FILENAME
from fastwam.memory.feature_cache import (
    FeatureCacheManifest,
    feature_cache_manifest_path,
)
from fastwam.memory.manifest import (
    EventBankManifest,
    sha256_array,
    sha256_canonical_json,
    sha256_file,
    sha256_path_tree,
)
from fastwam.memory.offline_pipeline import (
    FeatureCacheCollection,
    fixed_horizon_query_starts,
    load_feature_cache_collection,
    validate_feature_collection_against_catalog,
)
from fastwam.memory.online_retrieval import FrozenDinoOnlineRetriever
from fastwam.memory.payload_names import MODEL_SPACE_ACTION
from fastwam.memory.runtime_candidates import (
    ResolvedCandidateRow,
    RuntimeCandidateResolver,
)
from fastwam.models.warm.online_contract import WarmOnlineRunContract
from fastwam.models.warm.source_contract import WarmSourceRunContract
from fastwam.utils.artifact_claim import artifact_claim


REPORT_SCHEMA = "warm.online-retrieval-parity-report"
REPORT_VERSION = 1
MAX_BFLOAT16_ATOL = 1e-3


class OnlineParityError(RuntimeError):
    """Raised when online and M1 DEV retrieval cannot be proven equivalent."""


@dataclass(frozen=True, slots=True)
class ParityCounts:
    dev_episode_count: int
    task_episode_count: int
    query_count: int
    candidate_slot_count: int
    valid_candidate_count: int
    exact_context_key_count: int
    exact_score_roundtrip_count: int
    max_context_key_abs_error: float
    max_cosine_score_abs_error: float
    parity_transcript_sha256: str

    def to_dict(self) -> dict[str, int | float | str]:
        return {
            "dev_episode_count": self.dev_episode_count,
            "task_episode_count": self.task_episode_count,
            "query_count": self.query_count,
            "candidate_slot_count": self.candidate_slot_count,
            "valid_candidate_count": self.valid_candidate_count,
            "exact_context_key_count": self.exact_context_key_count,
            "exact_score_roundtrip_count": self.exact_score_roundtrip_count,
            "max_context_key_abs_error": self.max_context_key_abs_error,
            "max_cosine_score_abs_error": self.max_cosine_score_abs_error,
            "parity_transcript_sha256": self.parity_transcript_sha256,
        }


EpisodeLoader = Callable[
    [EpisodeRecord, Path, Any, tuple[str, ...], float, str], Any
]


def _positive_float(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a finite non-negative number") from exc
    if not np.isfinite(result) or result < 0.0:
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the formal M2.1 server parity gate: online frozen-DINO "
            "context keys and M1 full-bank exact-cosine top-K must agree "
            "with the immutable M1 DEV feature/candidate artifacts."
        )
    )
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--dev-candidate-cache", required=True, type=Path)
    parser.add_argument(
        "--dev-feature-list",
        required=True,
        action="append",
        type=Path,
        help="M1 DEV feature list; repeat only when the complete list is sharded.",
    )
    parser.add_argument("--training-run-contract", required=True, type=Path)
    parser.add_argument("--validation-run-contract", required=True, type=Path)
    parser.add_argument("--online-run-contract", required=True, type=Path)
    parser.add_argument("--resolved-eval-config", required=True, type=Path)
    parser.add_argument("--data-config", required=True, type=Path)
    parser.add_argument("--normalizer-contract", required=True, type=Path)
    parser.add_argument("--encoder-contract", required=True, type=Path)
    parser.add_argument("--camera-contract", required=True, type=Path)
    parser.add_argument("--normalization-stats", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--audit-report", required=True, type=Path)
    parser.add_argument(
        "--dataset-root",
        required=True,
        action="append",
        type=Path,
        help="Repeat in exact catalog dataset_index order.",
    )
    parser.add_argument("--dino-checkpoint", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--bfloat16-atol",
        type=_positive_float,
        default=0.0,
        help=(
            "Absolute tolerance for CUDA bfloat16 numeric drift only. "
            "Default 0 requires exact float32 keys and exact float32 score "
            "round-trips; values above 1e-3 are rejected."
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _strict_json(path: str | Path, label: str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        raw = source.read_bytes()
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {constant}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise OnlineParityError(f"cannot read strict JSON {label}: {source}") from exc
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise OnlineParityError(f"{label} must be a JSON object")
    return value


def _load_source_contract(path: Path, label: str) -> WarmSourceRunContract:
    try:
        return WarmSourceRunContract.from_dict(_strict_json(path, label))
    except (TypeError, ValueError) as exc:
        raise OnlineParityError(f"invalid {label}") from exc


def _load_online_contract(path: Path) -> WarmOnlineRunContract:
    try:
        return WarmOnlineRunContract.from_dict(
            _strict_json(path, "online run contract")
        )
    except (TypeError, ValueError) as exc:
        raise OnlineParityError("invalid online run contract") from exc


def _feature_paths(list_paths: Sequence[Path]) -> tuple[Path, ...]:
    resolved: list[Path] = []
    for raw_list in list_paths:
        list_path = raw_list.expanduser().resolve()
        try:
            lines = list_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError) as exc:
            raise OnlineParityError(f"cannot read DEV feature list {list_path}") from exc
        for line_number, raw_line in enumerate(lines, start=1):
            value = raw_line.strip()
            if not value or value.startswith("#"):
                continue
            candidate = Path(value).expanduser()
            if not candidate.is_absolute():
                candidate = list_path.parent / candidate
            candidate = candidate.resolve()
            if candidate.suffix != ".npz":
                raise OnlineParityError(
                    f"DEV feature list {list_path}:{line_number} must name an .npz"
                )
            resolved.append(candidate)
    if not resolved:
        raise OnlineParityError("DEV feature lists contain no feature caches")
    if len(set(resolved)) != len(resolved):
        raise OnlineParityError("DEV feature lists contain duplicate cache paths")
    return tuple(resolved)


def _validate_numeric_atol(value: float, encoder_contract: Mapping[str, Any]) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("bfloat16_atol must be numeric")
    atol = float(value)
    if not np.isfinite(atol) or atol < 0.0 or atol > MAX_BFLOAT16_ATOL:
        raise OnlineParityError(
            f"bfloat16_atol must lie in [0,{MAX_BFLOAT16_ATOL:g}]"
        )
    try:
        compute_dtype = str(encoder_contract["compute"]["dtype"])
    except (KeyError, TypeError) as exc:
        raise OnlineParityError("encoder contract lacks compute.dtype") from exc
    if atol > 0.0 and compute_dtype != "bfloat16":
        raise OnlineParityError(
            "a non-zero parity tolerance is allowed only for bfloat16 DINO compute"
        )
    return atol


def _validate_compute_device(
    requested_device: object, encoder_contract: Mapping[str, Any]
) -> str:
    if (
        not isinstance(requested_device, str)
        or not requested_device
        or requested_device.strip() != requested_device
    ):
        raise OnlineParityError("device must be a normalized non-empty string")
    try:
        expected_device = encoder_contract["compute"]["device"]
    except (KeyError, TypeError) as exc:
        raise OnlineParityError("encoder contract lacks compute.device") from exc
    if not isinstance(expected_device, str) or not expected_device:
        raise OnlineParityError("encoder contract compute.device is invalid")
    if requested_device != expected_device:
        raise OnlineParityError(
            "--device must exactly match encoder_contract.compute.device: "
            f"{requested_device!r} != {expected_device!r}"
        )
    return requested_device


def _assert_shared_contracts(
    *,
    training: WarmSourceRunContract,
    validation: WarmSourceRunContract,
    online: WarmOnlineRunContract,
    resolver: RuntimeCandidateResolver,
    collection: FeatureCacheCollection,
    normalizer_contract_sha256: str,
) -> None:
    if training.query_split != "train":
        raise OnlineParityError("training run contract must be train-bound")
    if validation.query_split != "dev":
        raise OnlineParityError("validation run contract must be DEV-bound")
    if training.sha256 != online.training_run_contract_sha256:
        raise OnlineParityError("online contract binds another training run contract")
    if validation.sha256 != online.validation_run_contract_sha256:
        raise OnlineParityError("online contract binds another validation run contract")

    shared_fields = (
        "bank_manifest_sha256",
        "bank_content_sha256",
        "catalog_sha256",
        "audit_sha256",
        "normalization_stats_sha256",
        "action_space_contract_sha256",
        "base_checkpoint_sha256",
        "global_sample_stride",
        "action_horizon",
        "action_dim",
    )
    for field_name in shared_fields:
        if getattr(training, field_name) != getattr(validation, field_name):
            raise OnlineParityError(
                f"train/DEV source contracts disagree on {field_name}"
            )
    if training.candidate_manifest_sha256 == validation.candidate_manifest_sha256:
        raise OnlineParityError("train and DEV must not reuse one candidate artifact")
    if training.query_corpus_sha256 == validation.query_corpus_sha256:
        raise OnlineParityError("train and DEV must not reuse one query corpus")

    online_pairs = (
        ("bank_manifest_sha256", training.bank_manifest_sha256),
        ("bank_content_sha256", training.bank_content_sha256),
        ("catalog_sha256", training.catalog_sha256),
        ("audit_sha256", training.audit_sha256),
        ("normalization_stats_sha256", training.normalization_stats_sha256),
        ("action_space_contract_sha256", training.action_space_contract_sha256),
        ("action_horizon", training.action_horizon),
        ("action_dim", training.action_dim),
    )
    for field_name, expected in online_pairs:
        if getattr(online, field_name) != expected:
            raise OnlineParityError(f"online/source mismatch for {field_name}")

    if resolver.query_split != "dev":
        raise OnlineParityError("parity resolver must load a DEV candidate cache")
    if resolver.query_stride != 1:
        raise OnlineParityError("DEV candidate cache query_stride must be exactly 1")
    if resolver.candidate_manifest_sha256 != validation.candidate_manifest_sha256:
        raise OnlineParityError("DEV candidate manifest does not match its run contract")
    if resolver.query_corpus_sha256 != validation.query_corpus_sha256:
        raise OnlineParityError("DEV query corpus does not match its run contract")
    if resolver.query_corpus_sha256 != collection.content_hash:
        raise OnlineParityError("DEV feature collection is not the candidate query corpus")
    if resolver.bank_manifest_sha256 != training.bank_manifest_sha256:
        raise OnlineParityError("DEV candidate resolver uses another event bank")
    if resolver.bank_content_sha256 != training.bank_content_sha256:
        raise OnlineParityError("DEV candidate resolver uses other bank content")
    if resolver.query_catalog_sha256 != training.catalog_sha256:
        raise OnlineParityError("DEV candidate catalog binding is inconsistent")
    if resolver.query_audit_sha256 != training.audit_sha256:
        raise OnlineParityError("DEV candidate audit binding is inconsistent")
    if resolver.fixed_k != online.top_k:
        raise OnlineParityError("online top_k differs from DEV candidate top_k")
    if resolver.action_horizon != online.action_horizon:
        raise OnlineParityError("online horizon differs from DEV candidate horizon")
    if resolver.action_space.action_dim != online.action_dim:
        raise OnlineParityError("online action dimension differs from the event bank")
    if sha256_canonical_json(resolver.action_space.to_dict()) != (
        online.action_space_contract_sha256
    ):
        raise OnlineParityError("resolver action-space contract hash is inconsistent")

    if collection.split != "dev":
        raise OnlineParityError("parity feature collection must be DEV")
    if collection.contract.catalog_hash != online.catalog_sha256:
        raise OnlineParityError("DEV feature catalog hash differs from online contract")
    if collection.contract.normalizer_hash != normalizer_contract_sha256:
        raise OnlineParityError("DEV feature normalizer hash differs from runtime")
    if collection.contract.encoder_hash != online.encoder_contract_sha256:
        raise OnlineParityError("DEV feature encoder hash differs from online runtime")
    if collection.contract.camera_hash != online.camera_contract_sha256:
        raise OnlineParityError("DEV feature camera hash differs from online runtime")


def _raw_online_cameras(
    images: Mapping[str, Any],
    frame_index: int,
    *,
    source_camera_keys: tuple[str, ...],
    processor_camera_mapping: Mapping[str, str],
) -> dict[str, np.ndarray]:
    if tuple(images) != source_camera_keys:
        raise OnlineParityError(
            "decoded camera order differs from the camera contract"
        )
    output: dict[str, np.ndarray] = {}
    for source_key in source_camera_keys:
        frames = np.asarray(images[source_key])
        if (
            frames.dtype != np.dtype(np.float32)
            or frames.ndim != 4
            or frames.shape[1] != 3
            or frame_index < 0
            or frame_index >= frames.shape[0]
            or not np.isfinite(frames[frame_index]).all()
            or np.any(frames[frame_index] < 0.0)
            or np.any(frames[frame_index] > 1.0)
        ):
            raise OnlineParityError(
                f"decoded camera {source_key!r} violates factual float32 CHW [0,1]"
            )
        # Exact inverse of the online uint8-HWC boundary and the same baseline
        # quantization used by M1 FastWAMImageAdapter.preprocess().
        uint8_chw = (frames[frame_index] * np.float32(255.0)).astype(np.uint8)
        output[processor_camera_mapping[source_key]] = np.ascontiguousarray(
            np.transpose(uint8_chw, (1, 2, 0))
        )
    return output


def _query_label(query_id: QueryId) -> str:
    return (
        f"{query_id.dataset_id}:{query_id.dataset_index}:"
        f"{query_id.episode_index}:{query_id.frame_index}"
    )


def _compare_query(
    *,
    offline_query_id: QueryId,
    expected_key: np.ndarray,
    online_step: Any,
    offline_row: ResolvedCandidateRow,
    offline_means: np.ndarray,
    atol: float,
) -> tuple[float, float, bool, bool]:
    label = _query_label(offline_query_id)
    actual_key = np.asarray(online_step.context_key)
    expected = np.asarray(expected_key)
    if actual_key.dtype != np.dtype(np.float32) or expected.dtype != np.dtype(np.float32):
        raise OnlineParityError(f"{label}: context keys must be float32")
    if actual_key.shape != expected.shape:
        raise OnlineParityError(f"{label}: context-key shape mismatch")
    key_error = float(np.max(np.abs(actual_key.astype(np.float64) - expected)))
    key_exact = bool(np.array_equal(actual_key, expected))
    if atol == 0.0:
        if not key_exact:
            raise OnlineParityError(
                f"{label}: online context key differs from M1 DEV cache"
            )
    elif not np.allclose(actual_key, expected, rtol=0.0, atol=atol):
        raise OnlineParityError(
            f"{label}: online context key exceeds bfloat16 atol={atol:g}"
        )

    online_mask = np.asarray(online_step.candidate_valid_mask)
    online_rows = np.asarray(online_step.bank_rows)
    if not np.array_equal(online_mask, offline_row.mask):
        raise OnlineParityError(f"{label}: candidate validity mask mismatch")
    if not np.array_equal(online_rows, offline_row.bank_rows):
        raise OnlineParityError(f"{label}: candidate bank-row mismatch")
    if tuple(online_step.event_ids) != tuple(offline_row.event_ids):
        raise OnlineParityError(f"{label}: candidate EventId mismatch")
    if not np.array_equal(np.asarray(online_step.candidate_means), offline_means):
        raise OnlineParityError(f"{label}: candidate action payload mismatch")

    actual_scores = np.asarray(online_step.cosine_scores, dtype=np.float64)
    expected_scores = np.asarray(offline_row.cosine_scores, dtype=np.float64)
    if actual_scores.shape != expected_scores.shape:
        raise OnlineParityError(f"{label}: cosine-score shape mismatch")
    score_error = float(np.max(np.abs(actual_scores - expected_scores)))
    score_roundtrip_exact = bool(
        np.array_equal(actual_scores.astype(np.float32), offline_row.cosine_scores)
    )
    if atol == 0.0:
        # The immutable candidate cache intentionally stores float32.  Exact
        # mode therefore requires an exact float64-online -> float32-cache
        # round-trip, rather than comparing different storage dtypes bitwise.
        if not score_roundtrip_exact:
            raise OnlineParityError(
                f"{label}: online cosine scores do not exactly round-trip to M1"
            )
    elif not np.allclose(actual_scores, expected_scores, rtol=0.0, atol=atol):
        raise OnlineParityError(
            f"{label}: cosine scores exceed bfloat16 atol={atol:g}"
        )
    return key_error, score_error, key_exact, score_roundtrip_exact


def _default_episode_loader(
    record: EpisodeRecord,
    dataset_root: Path,
    proof: Any,
    camera_keys: tuple[str, ...],
    timestamp_tolerance_s: float,
    video_backend: str,
) -> Any:
    from fastwam.datasets.lerobot.full_episode_reader import (
        read_full_lerobot_episode,
    )

    return read_full_lerobot_episode(
        record,
        dataset_root=dataset_root,
        audit_proof=proof,
        camera_keys=camera_keys,
        timestamp_tolerance_s=timestamp_tolerance_s,
        video_backend=video_backend,
    )


def validate_online_parity(
    *,
    retriever: FrozenDinoOnlineRetriever,
    resolver: RuntimeCandidateResolver,
    collection: FeatureCacheCollection,
    catalog: EpisodeCatalog,
    audit: LerobotAuditReport,
    training_run_contract: WarmSourceRunContract,
    validation_run_contract: WarmSourceRunContract,
    normalizer_contract_sha256: str,
    dataset_roots: Sequence[Path],
    source_camera_keys: Sequence[str],
    processor_camera_mapping: Mapping[str, str],
    timestamp_tolerance_s: float,
    video_backend: str,
    bfloat16_atol: float,
    episode_loader: EpisodeLoader = _default_episode_loader,
) -> ParityCounts:
    """Run a task-scoped DEV-query proof of M1 full-bank retrieval parity."""

    if not getattr(retriever, "artifact_verified", False):
        raise OnlineParityError("online retriever did not complete artifact verification")
    online = retriever.online_run_contract
    if online.source_policy != "fixed_context_top1":
        raise OnlineParityError("parity requires fixed_context_top1, never Gaussian-null")
    _assert_shared_contracts(
        training=training_run_contract,
        validation=validation_run_contract,
        online=online,
        resolver=resolver,
        collection=collection,
        normalizer_contract_sha256=normalizer_contract_sha256,
    )
    validate_feature_collection_against_catalog(
        collection, catalog, audit, expected_split="dev"
    )
    if audit.report_sha256 != online.audit_sha256:
        raise OnlineParityError("audited DEV data differs from online contract")

    roots = tuple(Path(path).resolve() for path in dataset_roots)
    descriptors = tuple(sorted(catalog.datasets, key=lambda item: item.dataset_index))
    if [item.dataset_index for item in descriptors] != list(range(len(descriptors))):
        raise OnlineParityError("catalog dataset indices must be contiguous from zero")
    if len(roots) != len(descriptors):
        raise OnlineParityError("dataset-root count must match catalog datasets")

    camera_keys = tuple(source_camera_keys)
    if not camera_keys or tuple(processor_camera_mapping) != camera_keys:
        raise OnlineParityError("camera mapping must preserve exact source-camera order")
    processor_keys = tuple(processor_camera_mapping[key] for key in camera_keys)
    if len(set(processor_keys)) != len(processor_keys):
        raise OnlineParityError("camera mapping collapses processor camera names")
    if tuple(audit.audited_camera_keys) != camera_keys:
        raise OnlineParityError("camera contract differs from audited cameras")
    if not isinstance(video_backend, str) or not video_backend:
        raise OnlineParityError("video_backend must be non-empty")
    tolerance = float(timestamp_tolerance_s)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise OnlineParityError("timestamp tolerance must be finite and non-negative")

    records_by_key = {
        (record.dataset_id, record.dataset_index, record.episode_index): record
        for record in catalog.episodes
    }
    task_records = []
    for loaded in collection.records:
        key = (
            loaded.metadata.dataset_id,
            loaded.metadata.dataset_index,
            loaded.metadata.episode_index,
        )
        record = records_by_key.get(key)
        if record is None:
            raise OnlineParityError(f"feature episode {key!r} is absent from catalog")
        if record.primary_task == online.task_description:
            task_records.append((loaded, record))
    if not task_records:
        raise OnlineParityError(
            "DEV feature corpus contains no episode for online task_description"
        )

    try:
        expected_task_index = tuple(retriever.task_vocabulary).index(
            online.task_description
        )
    except ValueError as exc:
        raise OnlineParityError("online task is absent from retrieval vocabulary") from exc

    query_count = 0
    valid_candidates = 0
    candidate_slots = 0
    exact_keys = 0
    exact_scores = 0
    max_key_error = 0.0
    max_score_error = 0.0
    transcript: list[dict[str, Any]] = []
    proof_index = audit.proof_index

    for parity_episode_index, (loaded, record) in enumerate(task_records):
        if loaded.metadata.split != "dev" or record.split != "dev":
            raise OnlineParityError("task parity accidentally selected a non-DEV episode")
        if loaded.metadata.task_index != expected_task_index:
            raise OnlineParityError(
                "DEV feature task index differs from exact encoder vocabulary"
            )
        key = (record.dataset_id, record.dataset_index, record.episode_index)
        proof = proof_index[key]
        full = episode_loader(
            record,
            roots[record.dataset_index],
            proof,
            camera_keys,
            tolerance,
            video_backend,
        )
        if full.source_episode_sha256 != loaded.metadata.source_episode_sha256:
            raise OnlineParityError(f"DEV raw source identity changed for {key!r}")
        for camera_key in camera_keys:
            frames = np.asarray(full.images[camera_key])
            if frames.shape[0] != record.length:
                raise OnlineParityError(
                    f"decoded camera length differs from catalog for {key!r}"
                )

        retriever.begin_episode(parity_episode_index)
        starts = fixed_horizon_query_starts(
            int(loaded.features.model_actions.shape[0]),
            action_horizon=resolver.action_horizon,
            stride=1,
        )
        if not starts:
            raise OnlineParityError(f"DEV episode {key!r} has no full action query")
        for frame_index in starts:
            offline_query = QueryId(*key, frame_index)
            offline_row = resolver.resolve(offline_query, allow_missing=False)
            offline_means = resolver.gather_payload(
                offline_row, MODEL_SPACE_ACTION, fill_value=0
            )
            raw_cameras = _raw_online_cameras(
                full.images,
                frame_index,
                source_camera_keys=camera_keys,
                processor_camera_mapping=processor_camera_mapping,
            )
            online_query = retriever.make_query_id(frame_index)
            proprio = loaded.features.proprio[frame_index]
            step = retriever.retrieve(
                online_query,
                raw_cameras,
                task_description=online.task_description,
                prompt=online.task_description,
                proprio=proprio,
            )
            if step.task_index != expected_task_index:
                raise OnlineParityError(
                    f"{_query_label(offline_query)}: online task index mismatch"
                )
            key_error, score_error, key_exact, score_exact = _compare_query(
                offline_query_id=offline_query,
                expected_key=loaded.features.context_keys[frame_index],
                online_step=step,
                offline_row=offline_row,
                offline_means=offline_means,
                atol=bfloat16_atol,
            )
            # This is a non-policy diagnostic, but consuming the capability
            # proves the same exactly-once admission checks used by inference.
            retriever.validate_bound_step(
                step,
                prompt=online.task_description,
                proprio=proprio,
                input_image=step.model_input,
            )
            query_count += 1
            candidate_slots += int(offline_row.fixed_k)
            valid_candidates += int(offline_row.valid_count)
            exact_keys += int(key_exact)
            exact_scores += int(score_exact)
            max_key_error = max(max_key_error, key_error)
            max_score_error = max(max_score_error, score_error)
            transcript.append(
                {
                    "query_id": {
                        "dataset_id": offline_query.dataset_id,
                        "dataset_index": offline_query.dataset_index,
                        "episode_index": offline_query.episode_index,
                        "frame_index": offline_query.frame_index,
                    },
                    "raw_camera_sha256": {
                        key: sha256_array(value)
                        for key, value in sorted(raw_cameras.items())
                    },
                    "model_input_sha256": sha256_array(
                        np.asarray(step.model_input)
                    ),
                    "context_key_sha256": sha256_array(
                        np.asarray(step.context_key)
                    ),
                    "event_ids": [
                        None if event_id is None else event_id.to_dict()
                        for event_id in step.event_ids
                    ],
                    "bank_rows_sha256": sha256_array(
                        np.asarray(step.bank_rows)
                    ),
                    "offline_scores_sha256": sha256_array(
                        offline_row.cosine_scores
                    ),
                    "online_scores_sha256": sha256_array(
                        np.asarray(step.cosine_scores)
                    ),
                    "candidate_payload_sha256": sha256_array(
                        np.asarray(step.candidate_means)
                    ),
                }
            )

    return ParityCounts(
        dev_episode_count=len(collection.records),
        task_episode_count=len(task_records),
        query_count=query_count,
        candidate_slot_count=candidate_slots,
        valid_candidate_count=valid_candidates,
        exact_context_key_count=exact_keys,
        exact_score_roundtrip_count=exact_scores,
        max_context_key_abs_error=max_key_error,
        max_cosine_score_abs_error=max_score_error,
        parity_transcript_sha256=sha256_canonical_json(
            {"queries": transcript}
        ),
    )


def _camera_runtime_fields(
    value: Mapping[str, Any],
) -> tuple[tuple[str, ...], dict[str, str], float, str]:
    try:
        sources = tuple(value["source_camera_keys"])
        raw_mapping = value["processor_camera_mapping"]
        tolerance = float(value["timestamp_tolerance_s"])
        backend = str(value["video_backend"])
    except (KeyError, TypeError, ValueError) as exc:
        raise OnlineParityError("camera contract lacks runtime decode fields") from exc
    if (
        not sources
        or any(not isinstance(key, str) or not key for key in sources)
        or not isinstance(raw_mapping, Mapping)
        or tuple(raw_mapping) != sources
    ):
        raise OnlineParityError("camera contract source/mapping fields are invalid")
    mapping = {key: str(raw_mapping[key]) for key in sources}
    return sources, mapping, tolerance, backend


def _git_state(repository_root: Path) -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "0" * 40, False
    return commit, False


def _artifact_snapshot(paths: Mapping[str, Path]) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for label, path in paths.items():
        try:
            snapshot[label] = sha256_file(path)
        except OSError as exc:
            raise OnlineParityError(f"cannot hash parity input {label}: {path}") from exc
    return snapshot


def _assert_snapshot_unchanged(
    paths: Mapping[str, Path], expected: Mapping[str, str]
) -> None:
    actual = _artifact_snapshot(paths)
    if actual != dict(expected):
        changed = sorted(
            label for label in set(actual) | set(expected) if actual.get(label) != expected.get(label)
        )
        raise OnlineParityError(
            f"parity inputs changed while validation was running: {changed!r}"
        )


def _aggregate_hash(snapshot: Mapping[str, str], prefix: str) -> str:
    return sha256_canonical_json(
        {
            label: digest
            for label, digest in sorted(snapshot.items())
            if label.startswith(prefix)
        }
    )


def _write_json_atomic(path: Path, value: Mapping[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"parity report already exists at {path}")
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
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


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _resolve_inside_root(root: Path, relative: str, *, label: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        raise OnlineParityError(f"{label} must be relative to its dataset root")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise OnlineParityError(f"{label} escapes its dataset root") from exc
    return resolved


def _task_raw_artifact_snapshot(
    *,
    catalog: EpisodeCatalog,
    audit: LerobotAuditReport,
    roots: Sequence[Path],
    task_description: str,
) -> tuple[dict[str, Path], dict[str, str]]:
    """Return every audited raw file used by this task's DEV parity proof."""

    root_tuple = tuple(Path(root).resolve() for root in roots)
    proof_index = audit.proof_index
    paths: dict[str, Path] = {}
    expected: dict[str, str] = {}
    selected = tuple(
        record
        for record in catalog.episodes
        if record.split == "dev" and record.primary_task == task_description
    )
    if not selected:
        raise OnlineParityError("catalog has no DEV raw episode for online task")
    for record in selected:
        key = (record.dataset_id, record.dataset_index, record.episode_index)
        proof = proof_index.get(key)
        if proof is None or proof.split != "dev":
            raise OnlineParityError(f"audit lacks DEV proof for {key!r}")
        root = root_tuple[record.dataset_index]
        stem = f"{record.dataset_index:04d}:{record.episode_index:08d}"
        table_label = f"raw_table:{stem}"
        paths[table_label] = _resolve_inside_root(
            root, record.data_relpath, label=f"episode table {key!r}"
        )
        assert proof.table_sha256 is not None
        expected[table_label] = proof.table_sha256
        camera_paths = resolve_episode_video_paths(
            root,
            episode_index=record.episode_index,
            camera_keys=proof.camera_keys,
        )
        if tuple(camera for camera, _ in camera_paths) != proof.camera_keys:
            raise OnlineParityError(f"camera path order differs from audit for {key!r}")
        for rank, ((camera, path), camera_proof) in enumerate(
            zip(camera_paths, proof.ordered_camera_sha256, strict=True)
        ):
            if camera != camera_proof.camera_key:
                raise OnlineParityError(f"camera proof identity mismatch for {key!r}")
            camera_label = f"raw_camera:{stem}:{rank:02d}"
            paths[camera_label] = path
            expected[camera_label] = camera_proof.sha256
    return paths, expected


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repository_root = Path(__file__).resolve().parents[1]
    bank_directory = args.bank.expanduser().resolve()
    candidate_directory = args.dev_candidate_cache.expanduser().resolve()
    feature_list_paths = tuple(
        path.expanduser().resolve() for path in args.dev_feature_list
    )
    feature_paths = _feature_paths(feature_list_paths)
    output = args.output.expanduser().resolve()
    dino_path = args.dino_checkpoint.expanduser().resolve()
    if bank_directory == candidate_directory:
        raise OnlineParityError("bank and DEV candidate cache must be distinct")
    if any(
        _is_within(output, directory)
        for directory in (
            repository_root,
            bank_directory,
            candidate_directory,
            dino_path,
        )
    ):
        raise OnlineParityError(
            "report must be outside the Git worktree and immutable artifact trees"
        )
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"parity report already exists at {output}")

    training_path = args.training_run_contract.expanduser().resolve()
    validation_path = args.validation_run_contract.expanduser().resolve()
    online_path = args.online_run_contract.expanduser().resolve()
    eval_config_path = args.resolved_eval_config.expanduser().resolve()
    data_config_path = args.data_config.expanduser().resolve()
    normalizer_path = args.normalizer_contract.expanduser().resolve()
    encoder_path = args.encoder_contract.expanduser().resolve()
    camera_path = args.camera_contract.expanduser().resolve()
    stats_path = args.normalization_stats.expanduser().resolve()
    catalog_path = args.catalog.expanduser().resolve()
    audit_path = args.audit_report.expanduser().resolve()

    training = _load_source_contract(training_path, "training run contract")
    validation = _load_source_contract(validation_path, "validation run contract")
    online = _load_online_contract(online_path)
    encoder_json = _strict_json(encoder_path, "encoder contract")
    camera_json = _strict_json(camera_path, "camera contract")
    device = _validate_compute_device(args.device, encoder_json)
    from fastwam.memory.runtime_fingerprint import current_encoder_runtime

    encoder_runtime = encoder_json.get("runtime")
    if not isinstance(encoder_runtime, Mapping):
        raise OnlineParityError("encoder contract lacks a runtime fingerprint")
    current_runtime = current_encoder_runtime(device)
    if current_runtime != dict(encoder_runtime):
        raise OnlineParityError(
            "parity runtime differs from the M1 feature-encoding runtime"
        )
    if sha256_canonical_json(current_runtime) != online.encoder_runtime_sha256:
        raise OnlineParityError(
            "runtime fingerprint does not match the online run contract"
        )
    # Reuse the online-contract builder's normalization of resolved Hydra
    # YAML/JSON.  The contract binds canonical content, not file formatting.
    if __package__:
        from scripts.build_warm_online_contract import (
            OnlineContractBuildError,
            _read_resolved_config,
        )
    else:  # Executed as ``python scripts/validate_warm_online_parity.py``.
        from build_warm_online_contract import (
            OnlineContractBuildError,
            _read_resolved_config,
        )

    try:
        resolved_eval_config = _read_resolved_config(eval_config_path)
    except OnlineContractBuildError as exc:
        raise OnlineParityError("cannot normalize resolved evaluation config") from exc
    if sha256_canonical_json(resolved_eval_config) != (
        online.resolved_eval_config_sha256
    ):
        raise OnlineParityError(
            "resolved evaluation config content does not match online contract"
        )
    if sha256_file(data_config_path) != encoder_json.get("data_config_sha256"):
        raise OnlineParityError("data config hash does not match M1 encoder contract")
    atol = _validate_numeric_atol(args.bfloat16_atol, encoder_json)
    try:
        configured_dino_device = resolved_eval_config["EVALUATION"][
            "warm_online"
        ]["dino_device"]
    except (KeyError, TypeError) as exc:
        raise OnlineParityError(
            "resolved evaluation config lacks warm_online.dino_device"
        ) from exc
    if configured_dino_device != args.device:
        raise OnlineParityError(
            "parity --device differs from the contract-bound evaluation "
            f"DINO device: {args.device!r} != {configured_dino_device!r}"
        )

    commit, dirty = _git_state(repository_root)

    catalog = EpisodeCatalog.load(catalog_path)
    audit = load_audit_report(audit_path)
    collection = load_feature_cache_collection(feature_paths)
    validate_feature_collection_against_catalog(
        collection, catalog, audit, expected_split="dev"
    )
    resolver = RuntimeCandidateResolver.from_artifacts(
        bank_directory,
        candidate_directory,
        expected_query_split="dev",
        expected_query_corpus_sha256=validation.query_corpus_sha256,
    )

    # Reuse the exact M1 processor validator/loader.  This import is
    # intentionally server-only because it reaches Hydra and Torch.
    if __package__:
        from scripts.precompute_warm_features import (
            _load_processor,
            _validate_dataset_roots,
            _validate_libero_processor,
        )
    else:  # Executed as ``python scripts/validate_warm_online_parity.py``.
        from precompute_warm_features import (
            _load_processor,
            _validate_dataset_roots,
            _validate_libero_processor,
        )

    roots = _validate_dataset_roots(catalog, args.dataset_root)
    processor = _load_processor(data_config_path, stats_path)
    _validate_libero_processor(processor)
    dino_batch_size = int(encoder_json["compute"]["dino_batch_size"])
    retriever = FrozenDinoOnlineRetriever.from_artifacts(
        bank_directory,
        source_run_contract=training,
        online_run_contract=online,
        normalizer_contract_path=normalizer_path,
        encoder_contract_path=encoder_path,
        camera_contract_path=camera_path,
        normalization_stats_path=stats_path,
        catalog_path=catalog_path,
        audit_report_path=audit_path,
        dino_checkpoint_path=dino_path,
        processor=processor,
        dino_device=device,
        dino_batch_size=dino_batch_size,
    )
    source_cameras, camera_mapping, timestamp_tolerance, video_backend = (
        _camera_runtime_fields(camera_json)
    )

    bank_manifest = EventBankManifest.read(bank_directory / BANK_MANIFEST_FILENAME)
    candidate_manifest = CandidateCacheManifest.read(
        candidate_directory / CANDIDATE_MANIFEST_FILENAME
    )
    snapshot_paths: dict[str, Path] = {
        "bank_manifest": bank_directory / BANK_MANIFEST_FILENAME,
        "bank_payload": bank_directory / bank_manifest.payload_file,
        "candidate_manifest": candidate_directory / CANDIDATE_MANIFEST_FILENAME,
        "candidate_payload": candidate_directory / candidate_manifest.payload_file,
        "training_run_contract": training_path,
        "validation_run_contract": validation_path,
        "online_run_contract": online_path,
        "resolved_eval_config": eval_config_path,
        "data_config": data_config_path,
        "normalizer_contract": normalizer_path,
        "encoder_contract": encoder_path,
        "camera_contract": camera_path,
        "normalization_stats": stats_path,
        "catalog_file": catalog_path,
        "audit_file": audit_path,
    }
    for index, list_path in enumerate(feature_list_paths):
        snapshot_paths[f"feature_list:{index:04d}"] = list_path
    for index, loaded in enumerate(collection.records):
        snapshot_paths[f"feature_payload:{index:06d}"] = loaded.payload_path
        snapshot_paths[f"feature_manifest:{index:06d}"] = feature_cache_manifest_path(
            loaded.payload_path
        )
    for descriptor, root in zip(
        sorted(catalog.datasets, key=lambda item: item.dataset_index), roots, strict=True
    ):
        snapshot_paths[f"dataset_info:{descriptor.dataset_index:04d}"] = (
            root / "meta" / "info.json"
        )
        snapshot_paths[f"dataset_episodes:{descriptor.dataset_index:04d}"] = (
            root / "meta" / "episodes.jsonl"
        )
    raw_snapshot_paths, raw_expected_hashes = _task_raw_artifact_snapshot(
        catalog=catalog,
        audit=audit,
        roots=roots,
        task_description=online.task_description,
    )
    snapshot_paths.update(raw_snapshot_paths)
    if output in set(snapshot_paths.values()):
        raise OnlineParityError("report must not overwrite an input artifact")
    snapshot = _artifact_snapshot(snapshot_paths)
    if _feature_paths(feature_list_paths) != feature_paths:
        raise OnlineParityError(
            "DEV feature list membership changed before parity execution"
        )
    dino_snapshot = sha256_path_tree(dino_path)
    if dino_snapshot != (
        online.dino_checkpoint_tree_sha256,
        online.dino_checkpoint_file_count,
    ):
        raise OnlineParityError("DINO checkpoint tree differs from online contract")
    if snapshot["bank_manifest"] != resolver.bank_manifest_sha256:
        raise OnlineParityError("event-bank manifest changed before parity execution")
    if snapshot["bank_payload"] != bank_manifest.content_hashes[
        bank_manifest.payload_file
    ]:
        raise OnlineParityError("event-bank payload differs from its manifest")
    if snapshot["candidate_manifest"] != resolver.candidate_manifest_sha256:
        raise OnlineParityError("DEV candidate manifest changed before parity execution")
    if snapshot["candidate_payload"] != candidate_manifest.content_hashes[
        candidate_manifest.payload_file
    ]:
        raise OnlineParityError("DEV candidate payload differs from its manifest")
    if _load_source_contract(training_path, "training run contract") != training:
        raise OnlineParityError("training run contract changed before parity execution")
    if _load_source_contract(validation_path, "validation run contract") != validation:
        raise OnlineParityError("validation run contract changed before parity execution")
    if _load_online_contract(online_path) != online:
        raise OnlineParityError("online run contract changed before parity execution")
    for label, expected in (
        ("data_config", str(encoder_json["data_config_sha256"])),
        ("normalizer_contract", collection.contract.normalizer_hash),
        ("encoder_contract", online.encoder_contract_sha256),
        ("camera_contract", online.camera_contract_sha256),
        ("normalization_stats", online.normalization_stats_sha256),
    ):
        if snapshot[label] != expected:
            raise OnlineParityError(f"{label} differs from its bound contract")
    try:
        current_resolved_eval_config = _read_resolved_config(eval_config_path)
    except OnlineContractBuildError as exc:
        raise OnlineParityError(
            "resolved evaluation config changed before parity execution"
        ) from exc
    if current_resolved_eval_config != resolved_eval_config:
        raise OnlineParityError(
            "resolved evaluation config changed before parity execution"
        )
    if EpisodeCatalog.load(catalog_path) != catalog:
        raise OnlineParityError("catalog changed before parity execution")
    if load_audit_report(audit_path) != audit:
        raise OnlineParityError("audit report changed before parity execution")
    for index, loaded in enumerate(collection.records):
        if snapshot[f"feature_payload:{index:06d}"] != loaded.manifest.payload_hash:
            raise OnlineParityError(
                f"DEV feature payload {index} differs from its loaded manifest"
            )
        if FeatureCacheManifest.read(
            snapshot_paths[f"feature_manifest:{index:06d}"]
        ) != loaded.manifest:
            raise OnlineParityError(
                f"DEV feature manifest {index} changed before parity execution"
            )
    for descriptor in sorted(catalog.datasets, key=lambda item: item.dataset_index):
        if snapshot[f"dataset_info:{descriptor.dataset_index:04d}"] != (
            descriptor.info_sha256
        ):
            raise OnlineParityError("dataset info metadata differs from catalog")
        if snapshot[f"dataset_episodes:{descriptor.dataset_index:04d}"] != (
            descriptor.episodes_sha256
        ):
            raise OnlineParityError("dataset episode metadata differs from catalog")
    for label, expected in raw_expected_hashes.items():
        if snapshot[label] != expected:
            raise OnlineParityError(
                f"audited raw DEV artifact differs before parity execution: {label}"
            )

    counts = validate_online_parity(
        retriever=retriever,
        resolver=resolver,
        collection=collection,
        catalog=catalog,
        audit=audit,
        training_run_contract=training,
        validation_run_contract=validation,
        normalizer_contract_sha256=sha256_file(normalizer_path),
        dataset_roots=roots,
        source_camera_keys=source_cameras,
        processor_camera_mapping=camera_mapping,
        timestamp_tolerance_s=timestamp_tolerance,
        video_backend=video_backend,
        bfloat16_atol=atol,
    )

    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "version": REPORT_VERSION,
        "status": "pass",
        "scope": "catalog-bound-dev-task",
        "task": {
            "suite": online.task_suite,
            "task_id": online.task_id,
            "description": online.task_description,
        },
        "numeric_comparison": {
            "mode": "exact" if atol == 0.0 else "cuda-bfloat16-atol",
            "rtol": 0.0,
            "atol": atol,
            "score_exactness": "float64-online-to-float32-cache-roundtrip",
        },
        "contracts": {
            "training_run_contract_sha256": training.sha256,
            "validation_run_contract_sha256": validation.sha256,
            "online_run_contract_sha256": online.sha256,
            "evaluation_namespace_sha256": online.evaluation_namespace_sha256,
            "encoder_contract_sha256": online.encoder_contract_sha256,
            "camera_contract_sha256": online.camera_contract_sha256,
            "normalization_stats_sha256": online.normalization_stats_sha256,
            "action_space_contract_sha256": online.action_space_contract_sha256,
            "catalog_sha256": online.catalog_sha256,
            "audit_sha256": online.audit_sha256,
        },
        "artifacts": {
            "bank_manifest_sha256": resolver.bank_manifest_sha256,
            "bank_content_sha256": resolver.bank_content_sha256,
            "candidate_manifest_sha256": resolver.candidate_manifest_sha256,
            "candidate_payload_sha256": candidate_manifest.content_hashes[
                candidate_manifest.payload_file
            ],
            "dev_query_corpus_sha256": resolver.query_corpus_sha256,
            "dev_feature_artifact_set_sha256": _aggregate_hash(
                snapshot, "feature_"
            ),
            "dataset_metadata_set_sha256": _aggregate_hash(snapshot, "dataset_"),
            "raw_dev_artifact_set_sha256": sha256_canonical_json(
                {
                    label: snapshot[label]
                    for label in sorted(raw_expected_hashes)
                }
            ),
            "raw_dev_artifact_count": len(raw_expected_hashes),
            "dino_checkpoint_tree_sha256": dino_snapshot[0],
            "dino_checkpoint_file_count": dino_snapshot[1],
            "resolved_eval_config_sha256": online.resolved_eval_config_sha256,
            "resolved_eval_config_file_sha256": snapshot["resolved_eval_config"],
            "data_config_sha256": snapshot["data_config"],
        },
        "counts": counts.to_dict(),
        "implementation": {
            "retriever": online.retrieval_implementation,
            "offline_candidate": "exact_cosine_v1",
            "search_domain": "complete_event_bank",
            "query_stride": resolver.query_stride,
            "top_k": resolver.fixed_k,
            "git_commit": commit,
            "git_dirty": False,
            "device": device,
            "encoder_runtime_sha256": online.encoder_runtime_sha256,
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.parent / f".{output.name}.warm-artifact.lock"
    with artifact_claim(lock_path, purpose=f"publish WARM online parity: {output}"):
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"parity report already exists at {output}")
        _assert_snapshot_unchanged(snapshot_paths, snapshot)
        if _feature_paths(feature_list_paths) != feature_paths:
            raise OnlineParityError(
                "DEV feature list membership changed during parity validation"
            )
        _validate_dataset_roots(catalog, roots)
        if sha256_path_tree(dino_path) != dino_snapshot:
            raise OnlineParityError("DINO checkpoint changed during parity validation")
        _write_json_atomic(output, report, overwrite=args.overwrite)

    print(
        json.dumps(
            {
                "status": "pass",
                "output": str(output),
                "report_sha256": sha256_file(output),
                "query_count": counts.query_count,
                "task_episode_count": counts.task_episode_count,
                "max_context_key_abs_error": counts.max_context_key_abs_error,
                "max_cosine_score_abs_error": counts.max_cosine_score_abs_error,
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
