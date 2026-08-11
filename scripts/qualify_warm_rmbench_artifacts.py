#!/usr/bin/env python3
"""Fail-closed, no-training qualification for WARM RMBench artifacts.

This command is intentionally independent of the train/eval launchers.  It
answers three questions before an expensive optimization run is allowed:

* does every bank row belong to a well-formed temporal event chain;
* can exact context retrieval recover a phase-compatible cross-episode event;
* does the materialized candidate cache exactly match the teacher-forced
  online reference search when both receive the same factual context key.

The last check is called *teacher-forced parity*: it does not claim that a
rendered online image equals an offline image.  It proves the downstream
query-to-candidate boundary exactly, leaving image/preprocessor parity as a
separate simulator smoke test.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence
from uuid import uuid4

import numpy as np

from fastwam.memory.candidate_cache import (
    CandidateCache,
    QueryId,
    canonical_event_bank_content_hash,
)
from fastwam.memory.event_bank import MANIFEST_FILENAME, EventBank
from fastwam.memory.manifest import sha256_file
from fastwam.memory.offline_pipeline import (
    FeatureCacheCollection,
    factual_state_query_starts,
    fixed_horizon_query_starts,
    load_feature_cache_collection,
)
from fastwam.memory.payload_names import MODEL_SPACE_ACTION, TASK_INDEX

try:  # Constants are added by the temporal-bank implementation.
    from fastwam.memory.payload_names import (  # type: ignore[attr-defined]
        ACTION_VALID_MASK,
        EVENT_ORDINAL,
        NORMALIZED_PHASE,
        SUCCESSOR_ROW,
        SUCCESSOR_EVENT_START_FRAME,
    )
except ImportError:  # Keep this validator usable while upgrading old checkouts.
    ACTION_VALID_MASK = "action_valid_mask"
    EVENT_ORDINAL = "event_ordinal"
    NORMALIZED_PHASE = "normalized_phase"
    SUCCESSOR_ROW = "successor_row"
    SUCCESSOR_EVENT_START_FRAME = "successor_event_start_frame"


_SCHEMA = "warm.rmbench-artifact-qualification"
_VERSION = 2
_STRATA = (
    ("early", 0.0, 1.0 / 3.0),
    ("middle", 1.0 / 3.0, 2.0 / 3.0),
    ("late", 2.0 / 3.0, 1.0),
)


class QualificationError(ValueError):
    """Raised when an artifact is valid NumPy but unsafe for RMBench training."""


@dataclass(frozen=True, slots=True)
class PhaseRecallThreshold:
    top_k: int
    minimum: float

    def __post_init__(self) -> None:
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int):
            raise TypeError("top_k must be an integer")
        if self.top_k <= 0:
            raise QualificationError("top_k must be positive")
        if not isinstance(self.minimum, (int, float)) or isinstance(self.minimum, bool):
            raise TypeError("minimum phase recall must be numeric")
        value = float(self.minimum)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise QualificationError("minimum phase recall must be in [0, 1]")
        object.__setattr__(self, "minimum", value)


def _require_payload(bank: EventBank, name: str) -> np.ndarray:
    try:
        return bank.payload(name)
    except KeyError as exc:
        raise QualificationError(
            f"event bank lacks required temporal payload {name!r}; rebuild M1"
        ) from exc


def validate_temporal_bank(
    bank: EventBank,
    *,
    max_event_stride: int = 4,
    phase_atol: float = 1e-5,
) -> dict[str, object]:
    """Validate dense phase-aligned rows and exact within-episode successors."""

    if not isinstance(bank, EventBank):
        raise TypeError("bank must be EventBank")
    if isinstance(max_event_stride, bool) or not isinstance(max_event_stride, int):
        raise TypeError("max_event_stride must be an integer")
    if max_event_stride <= 0:
        raise QualificationError("max_event_stride must be positive")
    if not math.isfinite(phase_atol) or phase_atol < 0.0:
        raise QualificationError("phase_atol must be finite and non-negative")
    count = len(bank)
    if count == 0:
        raise QualificationError("event bank must not be empty")

    actions = _require_payload(bank, MODEL_SPACE_ACTION)
    phase = _require_payload(bank, NORMALIZED_PHASE)
    ordinal = _require_payload(bank, EVENT_ORDINAL)
    successor = _require_payload(bank, SUCCESSOR_ROW)
    successor_start = _require_payload(bank, SUCCESSOR_EVENT_START_FRAME)
    valid = _require_payload(bank, ACTION_VALID_MASK)
    horizon = int(actions.shape[1]) if actions.ndim == 3 else -1
    expected = {
        NORMALIZED_PHASE: (np.dtype(np.float32), (count,)),
        EVENT_ORDINAL: (np.dtype(np.int64), (count,)),
        SUCCESSOR_ROW: (np.dtype(np.int64), (count,)),
        SUCCESSOR_EVENT_START_FRAME: (np.dtype(np.int64), (count,)),
        ACTION_VALID_MASK: (np.dtype(np.bool_), (count, horizon)),
    }
    for name, array in (
        (NORMALIZED_PHASE, phase),
        (EVENT_ORDINAL, ordinal),
        (SUCCESSOR_ROW, successor),
        (SUCCESSOR_EVENT_START_FRAME, successor_start),
        (ACTION_VALID_MASK, valid),
    ):
        dtype, shape = expected[name]
        if array.dtype != dtype or array.shape != shape:
            raise QualificationError(
                f"{name} must have dtype/shape {dtype}/{shape}, got "
                f"{array.dtype}/{array.shape}"
            )
    if actions.dtype != np.dtype(np.float32) or actions.ndim != 3 or horizon <= 0:
        raise QualificationError("model_space_action must be float32 [N,H,D]")
    if not np.isfinite(phase).all() or np.any(phase < 0.0) or np.any(phase > 1.0):
        raise QualificationError("normalized_phase must be finite in [0, 1]")
    if np.any(ordinal < 0):
        raise QualificationError("event_ordinal must be non-negative")
    if np.any(successor < -1) or np.any(successor >= count):
        raise QualificationError("successor_row must be -1 or a valid bank row")
    if not valid.all():
        raise QualificationError(
            "RMBench event actions must be factual full horizons; padding is forbidden"
        )

    groups: dict[tuple[str, int, int], list[int]] = {}
    for row, event_id in enumerate(bank.event_ids):
        groups.setdefault(event_id.episode_key, []).append(row)
    max_gap = 0
    terminal_count = 0
    for episode_key, rows in groups.items():
        rows.sort(key=lambda index: bank.event_ids[index].start_frame)
        starts = np.asarray(
            [bank.event_ids[index].start_frame for index in rows], dtype=np.int64
        )
        if starts.size > 1:
            gaps = np.diff(starts)
            if np.any(gaps <= 0):
                raise QualificationError(
                    f"event starts are not strictly increasing in {episode_key!r}"
                )
            episode_max_gap = int(gaps.max())
            max_gap = max(max_gap, episode_max_gap)
            if episode_max_gap > max_event_stride:
                raise QualificationError(
                    f"event chain {episode_key!r} has gap {episode_max_gap}, "
                    f"exceeding max_event_stride={max_event_stride}"
                )
        expected_ordinals = np.arange(len(rows), dtype=np.int64)
        if not np.array_equal(ordinal[rows], expected_ordinals):
            raise QualificationError(
                f"event_ordinal is not contiguous in {episode_key!r}"
            )
        max_start = int(starts[-1])
        expected_phase = (
            np.zeros_like(starts, dtype=np.float32)
            if max_start == 0
            else starts.astype(np.float32) / float(max_start)
        )
        if not np.allclose(phase[rows], expected_phase, rtol=0.0, atol=phase_atol):
            raise QualificationError(
                f"normalized_phase disagrees with start/max_start in {episode_key!r}"
            )
        for position, row in enumerate(rows):
            expected_successor = rows[position + 1] if position + 1 < len(rows) else -1
            if int(successor[row]) != expected_successor:
                raise QualificationError(
                    f"successor_row[{row}]={int(successor[row])}, expected "
                    f"{expected_successor} in {episode_key!r}"
                )
            expected_start = (
                -1
                if expected_successor < 0
                else bank.event_ids[expected_successor].start_frame
            )
            if int(successor_start[row]) != expected_start:
                raise QualificationError(
                    f"successor_event_start_frame[{row}]={int(successor_start[row])}, "
                    f"expected {expected_start} in {episode_key!r}"
                )
        terminal_count += 1

    return {
        "event_count": count,
        "episode_count": len(groups),
        "action_horizon": horizon,
        "max_observed_event_stride": max_gap,
        "max_allowed_event_stride": max_event_stride,
        "terminal_event_count": terminal_count,
        "payload_names": {
            "normalized_phase": NORMALIZED_PHASE,
            "event_ordinal": EVENT_ORDINAL,
            "successor_row": SUCCESSOR_ROW,
            "successor_event_start_frame": SUCCESSOR_EVENT_START_FRAME,
            "action_valid_mask": ACTION_VALID_MASK,
        },
    }


def _record_map(collection: FeatureCacheCollection) -> Mapping[tuple[str, int, int], object]:
    return {
        (
            record.metadata.dataset_id,
            record.metadata.dataset_index,
            record.metadata.episode_index,
        ): record
        for record in collection.records
    }


def _stratum(value: float) -> str:
    for index, (name, low, high) in enumerate(_STRATA):
        if value >= low and (value < high or index == len(_STRATA) - 1):
            return name
    raise AssertionError(value)


def evaluate_phase_recall(
    bank: EventBank,
    collection: FeatureCacheCollection,
    *,
    action_horizon: int,
    query_stride: int,
    thresholds: Sequence[PhaseRecallThreshold],
    phase_tolerance: float = 0.10,
) -> dict[str, object]:
    """Require phase-compatible candidates in every early/middle/late stratum."""

    threshold_rows = tuple(thresholds)
    if not threshold_rows:
        raise QualificationError("at least one phase-recall threshold is required")
    if len({row.top_k for row in threshold_rows}) != len(threshold_rows):
        raise QualificationError("phase-recall top_k values must be unique")
    if action_horizon <= 0 or query_stride <= 0:
        raise QualificationError("action_horizon and query_stride must be positive")
    if not math.isfinite(phase_tolerance) or not 0.0 <= phase_tolerance <= 1.0:
        raise QualificationError("phase_tolerance must be in [0, 1]")
    bank_phase = _require_payload(bank, NORMALIZED_PHASE)
    bank_task = _require_payload(bank, TASK_INDEX)
    if bank_task.dtype != np.dtype(np.int64) or bank_task.shape != (len(bank),):
        raise QualificationError("task_index must be int64 [events]")
    max_k = max(row.top_k for row in threshold_rows)
    counts = {name: 0 for name, _, _ in _STRATA}
    hits = {
        row.top_k: {name: 0 for name, _, _ in _STRATA} for row in threshold_rows
    }
    total = 0
    total_hits = {row.top_k: 0 for row in threshold_rows}
    task_counts: dict[int, dict[str, int]] = {}
    task_hits: dict[int, dict[int, dict[str, int]]] = {}

    for record in collection.records:
        episode = record.features
        steps = int(episode.model_actions.shape[0])
        max_start = steps - action_horizon
        for start in fixed_horizon_query_starts(
            steps, action_horizon=action_horizon, stride=query_stride
        ):
            query_phase = 0.0 if max_start <= 0 else float(start) / float(max_start)
            stratum = _stratum(query_phase)
            task_index = int(episode.task_index)
            task_counts.setdefault(
                task_index, {name: 0 for name, _, _ in _STRATA}
            )
            task_hits.setdefault(
                task_index,
                {
                    row.top_k: {name: 0 for name, _, _ in _STRATA}
                    for row in threshold_rows
                },
            )
            results = bank.search(
                episode.context_keys[start],
                top_k=max_k,
                exclude_episode=(
                    episode.dataset_id,
                    episode.dataset_index,
                    episode.episode_index,
                ),
                exclude_source_episode_sha256=episode.source_episode_sha256,
                exclude_feature_episode_sha256=episode.feature_episode_sha256,
            )
            counts[stratum] += 1
            task_counts[task_index][stratum] += 1
            total += 1
            for threshold in threshold_rows:
                compatible = any(
                    int(bank_task[result.index]) == task_index
                    and abs(float(bank_phase[result.index]) - query_phase)
                    <= phase_tolerance
                    for result in results[: threshold.top_k]
                )
                if compatible:
                    hits[threshold.top_k][stratum] += 1
                    task_hits[task_index][threshold.top_k][stratum] += 1
                    total_hits[threshold.top_k] += 1
    if total == 0:
        raise QualificationError("dev feature collection has no full-horizon queries")
    empty = [name for name, count in counts.items() if count == 0]
    if empty:
        raise QualificationError(
            f"phase qualification has no queries in strata {empty!r}; "
            "use a denser dev query stride"
        )
    for task_index, task_strata in task_counts.items():
        empty_task_strata = [name for name, count in task_strata.items() if count == 0]
        if empty_task_strata:
            raise QualificationError(
                f"task {task_index} has no phase queries in strata "
                f"{empty_task_strata!r}; use a denser dev query stride"
            )

    reports: dict[str, object] = {}
    failures: list[str] = []
    for threshold in threshold_rows:
        per_stratum = {
            name: hits[threshold.top_k][name] / counts[name] for name in counts
        }
        overall = total_hits[threshold.top_k] / total
        reports[str(threshold.top_k)] = {
            "minimum": threshold.minimum,
            "overall": overall,
            "per_stratum": per_stratum,
            "hits": dict(hits[threshold.top_k]),
        }
        if overall + 1e-12 < threshold.minimum:
            failures.append(
                f"recall@{threshold.top_k} overall={overall:.4f} "
                f"< {threshold.minimum:.4f}"
            )
        for name, recall in per_stratum.items():
            if recall + 1e-12 < threshold.minimum:
                failures.append(
                    f"recall@{threshold.top_k}/{name}={recall:.4f} "
                    f"< {threshold.minimum:.4f}"
                )
        for task_index, task_strata in task_counts.items():
            task_total = sum(task_strata.values())
            task_total_hits = sum(task_hits[task_index][threshold.top_k].values())
            task_overall = task_total_hits / task_total
            if task_overall + 1e-12 < threshold.minimum:
                failures.append(
                    f"recall@{threshold.top_k}/task={task_index}="
                    f"{task_overall:.4f} < {threshold.minimum:.4f}"
                )
            for name, count in task_strata.items():
                recall = task_hits[task_index][threshold.top_k][name] / count
                if recall + 1e-12 < threshold.minimum:
                    failures.append(
                        f"recall@{threshold.top_k}/task={task_index}/{name}="
                        f"{recall:.4f} < {threshold.minimum:.4f}"
                    )
    if failures:
        raise QualificationError("phase-compatible candidate gate failed: " + "; ".join(failures))
    per_task: dict[str, object] = {}
    for task_index, task_strata in sorted(task_counts.items()):
        per_task[str(task_index)] = {
            "query_counts": task_strata,
            "reports": {
                str(threshold.top_k): {
                    "overall": sum(task_hits[task_index][threshold.top_k].values())
                    / sum(task_strata.values()),
                    "per_stratum": {
                        name: task_hits[task_index][threshold.top_k][name] / count
                        for name, count in task_strata.items()
                    },
                }
                for threshold in threshold_rows
            },
        }
    return {
        "query_count": total,
        "query_stride": query_stride,
        "phase_tolerance": phase_tolerance,
        "stratum_query_counts": counts,
        "reports": reports,
        "per_task": per_task,
    }


def _sample_positions(size: int, maximum: int) -> tuple[int, ...]:
    if size <= maximum:
        return tuple(range(size))
    # Linspace is deterministic, includes both ends, and covers the whole run.
    return tuple(int(value) for value in np.linspace(0, size - 1, maximum, dtype=np.int64))


def validate_teacher_forced_parity(
    bank: EventBank,
    collection: FeatureCacheCollection,
    cache: CandidateCache,
    *,
    max_queries: int = 2048,
    score_atol: float = 2e-6,
    require_partial_action_queries: bool = False,
) -> dict[str, object]:
    """Compare cached train/eval candidates with exact online-reference search."""

    if max_queries <= 0:
        raise QualificationError("max_queries must be positive")
    if not math.isfinite(score_atol) or score_atol < 0.0:
        raise QualificationError("score_atol must be finite and non-negative")
    if cache.manifest is None:
        raise QualificationError("candidate cache has no loaded manifest")
    recipe = dict(cache.manifest.build_recipe)
    top_k = recipe.get("top_k")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise QualificationError("candidate cache build_recipe lacks positive integer top_k")
    recipe_stride = recipe.get("query_stride")
    if (
        isinstance(recipe_stride, bool)
        or not isinstance(recipe_stride, int)
        or recipe_stride <= 0
    ):
        raise QualificationError(
            "candidate cache build_recipe lacks positive integer query_stride"
        )
    query_frame_policy = recipe.get("query_frame_policy", "full_horizon_v1")
    if query_frame_policy not in {
        "full_horizon_v1",
        "all_factual_states_v1",
    }:
        raise QualificationError(
            "candidate cache has unsupported query_frame_policy"
        )
    if (
        require_partial_action_queries
        and query_frame_policy != "all_factual_states_v1"
    ):
        raise QualificationError(
            "formal RMBench artifacts require terminal partial-action query rows"
        )
    recipe_horizon = recipe.get("action_horizon")
    bank_actions = _require_payload(bank, MODEL_SPACE_ACTION)
    if (
        isinstance(recipe_horizon, bool)
        or not isinstance(recipe_horizon, int)
        or recipe_horizon != int(bank_actions.shape[1])
    ):
        raise QualificationError(
            "candidate cache action_horizon disagrees with the event bank"
        )
    records = _record_map(collection)
    expected_query_ids: set[QueryId] = set()
    partial_query_count = 0
    for record in collection.records:
        episode = record.features
        num_actions = int(episode.model_actions.shape[0])
        starts = (
            factual_state_query_starts(
                int(episode.context_keys.shape[0]), stride=recipe_stride
            )
            if query_frame_policy == "all_factual_states_v1"
            else fixed_horizon_query_starts(
                num_actions,
                action_horizon=recipe_horizon,
                stride=recipe_stride,
            )
        )
        last_full_start = num_actions - recipe_horizon
        for start in starts:
            expected_query_ids.add(
                QueryId(
                    episode.dataset_id,
                    episode.dataset_index,
                    episode.episode_index,
                    start,
                )
            )
            if start > last_full_start:
                partial_query_count += 1
    actual_query_ids = set(cache.query_ids)
    if actual_query_ids != expected_query_ids:
        missing = sorted(expected_query_ids - actual_query_ids)[:5]
        extra = sorted(actual_query_ids - expected_query_ids)[:5]
        raise QualificationError(
            "candidate cache does not exactly cover its declared query-frame "
            f"policy; missing={missing!r}, extra={extra!r}"
        )
    positions = _sample_positions(len(cache), max_queries)
    if not positions:
        raise QualificationError("candidate cache must contain at least one query")
    compared_candidates = 0
    for position in positions:
        query_id = cache.query_ids[position]
        try:
            record = records[query_id.episode_key]
        except KeyError as exc:
            raise QualificationError(
                f"candidate query {query_id!r} is absent from supplied features"
            ) from exc
        episode = record.features
        if query_id.frame_index >= episode.context_keys.shape[0]:
            raise QualificationError(f"candidate query frame is out of range: {query_id!r}")
        expected = bank.search(
            episode.context_keys[query_id.frame_index],
            top_k=top_k,
            exclude_episode=query_id.episode_key,
            exclude_source_episode_sha256=episode.source_episode_sha256,
            exclude_feature_episode_sha256=episode.feature_episode_sha256,
        )
        actual = cache.candidates[position]
        expected_ids = tuple(row.event_id for row in expected)
        actual_ids = tuple(row.event_id for row in actual)
        if actual_ids != expected_ids:
            raise QualificationError(
                f"teacher-forced candidate identity mismatch at {query_id!r}"
            )
        expected_scores = np.asarray([row.score for row in expected], dtype=np.float64)
        actual_scores = np.asarray([row.cosine_score for row in actual], dtype=np.float64)
        if not np.allclose(actual_scores, expected_scores, rtol=0.0, atol=score_atol):
            error = float(np.max(np.abs(actual_scores - expected_scores)))
            raise QualificationError(
                f"teacher-forced cosine mismatch at {query_id!r}: max_abs={error}"
            )
        compared_candidates += len(actual)
    return {
        "cache_query_count": len(cache),
        "checked_query_count": len(positions),
        "checked_candidate_count": compared_candidates,
        "top_k": top_k,
        "query_frame_policy": query_frame_policy,
        "partial_action_query_count": partial_query_count,
        "score_atol": score_atol,
        "scope": "query-to-candidate boundary; factual context keys teacher-forced",
    }


def _read_feature_list(path: Path) -> tuple[Path, ...]:
    root = path.expanduser().resolve()
    rows: list[Path] = []
    for line_number, raw in enumerate(
        root.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        item = Path(value).expanduser()
        if not item.is_absolute():
            item = root.parent / item
        if item.suffix != ".npz":
            raise QualificationError(f"{root}:{line_number}: expected .npz path")
        rows.append(item.resolve())
    if not rows:
        raise QualificationError(f"feature list is empty: {root}")
    return tuple(dict.fromkeys(rows))


def _threshold(value: str) -> PhaseRecallThreshold:
    try:
        raw_k, raw_minimum = value.split("=", 1)
        return PhaseRecallThreshold(int(raw_k), float(raw_minimum))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("threshold must use TOP_K=MINIMUM, e.g. 32=0.85") from exc


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path = path.expanduser().resolve()
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--train-feature-list", required=True, type=Path)
    parser.add_argument("--train-candidate-cache", required=True, type=Path)
    parser.add_argument("--dev-feature-list", required=True, type=Path)
    parser.add_argument("--dev-candidate-cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--query-stride", type=int, default=4)
    parser.add_argument("--max-event-stride", type=int, default=4)
    parser.add_argument("--phase-tolerance", type=float, default=0.10)
    parser.add_argument(
        "--phase-recall-threshold",
        type=_threshold,
        action="append",
        default=None,
        help="Repeat TOP_K=MINIMUM; defaults to 32=0.85 and 128=0.95.",
    )
    parser.add_argument("--parity-max-queries", type=int, default=2048)
    parser.add_argument("--parity-score-atol", type=float, default=2e-6)
    parser.add_argument(
        "--require-partial-action-queries",
        action="store_true",
        help="Require terminal partial-horizon action rows in both caches.",
    )
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help=(
            "Recompute every qualification check and require --output to "
            "contain the identical previously published report."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        bank_root = args.bank.expanduser().resolve()
        bank = EventBank.load(bank_root)
        if bank.manifest is None:
            raise QualificationError("event bank manifest is unavailable")
        train_collection = load_feature_cache_collection(
            _read_feature_list(args.train_feature_list)
        )
        dev_collection = load_feature_cache_collection(
            _read_feature_list(args.dev_feature_list)
        )
        if train_collection.split != "train":
            raise QualificationError(
                "qualification requires train features, got "
                f"split={train_collection.split!r}"
            )
        if dev_collection.split != "dev":
            raise QualificationError(
                f"qualification requires dev features, got split={dev_collection.split!r}"
            )
        bank_manifest_hash = sha256_file(bank_root / MANIFEST_FILENAME)
        bank_content_hash = canonical_event_bank_content_hash(bank.manifest.content_hashes)
        train_cache = CandidateCache.load(
            args.train_candidate_cache.expanduser().resolve(),
            expected_event_bank_manifest_hash=bank_manifest_hash,
            expected_event_bank_content_hash=bank_content_hash,
            expected_query_corpus_hash=train_collection.content_hash,
        )
        dev_cache = CandidateCache.load(
            args.dev_candidate_cache.expanduser().resolve(),
            expected_event_bank_manifest_hash=bank_manifest_hash,
            expected_event_bank_content_hash=bank_content_hash,
            expected_query_corpus_hash=dev_collection.content_hash,
        )
        train_cache.validate_against_event_bank(bank)
        dev_cache.validate_against_event_bank(bank)
        thresholds = args.phase_recall_threshold or [
            PhaseRecallThreshold(32, 0.85),
            PhaseRecallThreshold(128, 0.95),
        ]
        temporal = validate_temporal_bank(
            bank, max_event_stride=args.max_event_stride
        )
        if temporal["action_horizon"] != args.action_horizon:
            raise QualificationError(
                "--action-horizon disagrees with the immutable event bank"
            )
        phase = evaluate_phase_recall(
            bank,
            dev_collection,
            action_horizon=args.action_horizon,
            query_stride=args.query_stride,
            thresholds=thresholds,
            phase_tolerance=args.phase_tolerance,
        )
        train_parity = validate_teacher_forced_parity(
            bank,
            train_collection,
            train_cache,
            max_queries=args.parity_max_queries,
            score_atol=args.parity_score_atol,
            require_partial_action_queries=args.require_partial_action_queries,
        )
        dev_parity = validate_teacher_forced_parity(
            bank,
            dev_collection,
            dev_cache,
            max_queries=args.parity_max_queries,
            score_atol=args.parity_score_atol,
            require_partial_action_queries=args.require_partial_action_queries,
        )
        report: dict[str, object] = {
            "schema": _SCHEMA,
            "version": _VERSION,
            "qualified": True,
            "bank_manifest_sha256": bank_manifest_hash,
            "bank_content_sha256": bank_content_hash,
            "train_feature_collection_sha256": train_collection.content_hash,
            "dev_feature_collection_sha256": dev_collection.content_hash,
            "train_candidate_cache_manifest_sha256": sha256_file(
                args.train_candidate_cache.expanduser().resolve()
                / "candidate_manifest.json"
            ),
            "dev_candidate_cache_manifest_sha256": sha256_file(
                args.dev_candidate_cache.expanduser().resolve()
                / "candidate_manifest.json"
            ),
            "temporal_bank": temporal,
            "phase_recall": phase,
            "teacher_forced_parity": {
                "train": train_parity,
                "dev": dev_parity,
            },
            "limitations": [
                "does not replace rendered-observation preprocessing parity",
                "does not replace simulator expert-action replay",
                "does not measure learned source-gate gradients before training",
            ],
        }
        if args.verify_existing:
            output = args.output.expanduser().resolve()
            try:
                existing = json.loads(output.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise QualificationError(
                    f"cannot read existing qualification report: {output}"
                ) from exc
            if existing != report:
                raise QualificationError(
                    "existing qualification report does not match current "
                    "bank/features/candidates or qualification policy"
                )
        else:
            _atomic_json(args.output, report)
    except (OSError, TypeError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    status = "VERIFIED" if args.verify_existing else "OK"
    print(
        f"RMBENCH_ARTIFACT_QUALIFICATION_{status} "
        f"output={args.output.expanduser().resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
