#!/usr/bin/env python3
"""Stream a formal RMBench evidence file into memory-usage diagnostics.

The evidence stream may be hundreds of megabytes, so this utility never loads
it wholesale.  It is deliberately dependency-free and can run in the base
server Python while an evaluation is still producing records.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


class EvidenceAnalysisError(ValueError):
    pass


class _Mean:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def add(self, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return
        number = float(value)
        if math.isfinite(number):
            self.total += number
            self.count += 1

    def value(self) -> float | None:
        return self.total / self.count if self.count else None


class _UnitHistogram:
    """Bounded-memory approximate quantiles for normalized diagnostics."""

    def __init__(self, bins: int = 1000) -> None:
        self.bins = bins
        self.counts = [0] * bins
        self.count = 0

    def add(self, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return
        number = float(value)
        if not math.isfinite(number):
            return
        number = min(1.0, max(0.0, number))
        index = min(self.bins - 1, int(number * self.bins))
        self.counts[index] += 1
        self.count += 1

    def quantile(self, probability: float) -> float | None:
        if not self.count:
            return None
        target = max(1, math.ceil(probability * self.count))
        cumulative = 0
        for index, count in enumerate(self.counts):
            cumulative += count
            if cumulative >= target:
                return (index + 0.5) / self.bins
        return 1.0


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvidenceAnalysisError(f"{field} must be an object")
    return value


def _event_key(value: object) -> tuple[str, int, int] | None:
    if value is None:
        return None
    event = _mapping(value, "selected_event_id")
    try:
        return (
            str(event["dataset_id"]),
            int(event["episode_index"]),
            int(event["start_frame"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise EvidenceAnalysisError("selected_event_id is malformed") from error


def analyze_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    kinds: Counter[str] = Counter()
    headers = 0
    episode_started: set[int] = set()
    episode_ended: set[int] = set()
    successes = 0
    candidate_selected = 0
    memory_selected = 0
    high_entropy_selected = 0
    entropy_observations = 0
    high_stagnation_selected = 0
    stagnation_observations = 0
    thread_comparable = 0
    thread_continuations = 0
    exact_event_repetitions = 0
    phase_locked_transitions = 0
    maximum_consecutive_phase_replans = 0
    repeated_attempt_max = 0
    input_stats_records = 0
    action_stats_records = 0
    factual_update_records = 0
    event_write_records = 0
    executed_environment_actions = 0
    means = {
        name: _Mean()
        for name in (
            "gate",
            "learned_gate",
            "source_quality",
            "selected_probability",
            "probability_margin",
            "normalized_entropy",
            "stagnation_score",
            "top1_top2_cosine_gap",
        )
    }
    distributions = {name: _UnitHistogram() for name in means}
    previous_event_by_episode: dict[int, tuple[str, int, int]] = {}
    phase_run_by_episode: dict[int, int] = {}
    selected_event_counts: Counter[tuple[str, int, int]] = Counter()

    for line_number, record in enumerate(records, start=1):
        if not isinstance(record, Mapping):
            raise EvidenceAnalysisError(f"record {line_number} is not an object")
        kind = record.get("kind")
        if not isinstance(kind, str):
            raise EvidenceAnalysisError(f"record {line_number} has no kind")
        kinds[kind] += 1
        if kind == "header":
            headers += 1
            continue
        if kind == "episode_begin":
            episode_started.add(int(record["episode_index"]))
            continue
        if kind == "episode_end":
            episode = int(record["episode_index"])
            episode_ended.add(episode)
            successes += int(record.get("success") is True)
            previous_event_by_episode.pop(episode, None)
            phase_run_by_episode.pop(episode, None)
            continue
        if kind != "replan":
            continue

        episode = int(record["episode_index"])
        model = _mapping(record.get("model"), "model")
        source = _mapping(model.get("source"), "model.source")
        selected_candidate = source.get("candidate_selected") is True
        selected_source = source.get("memory_selected") is True
        candidate_selected += int(selected_candidate)
        memory_selected += int(selected_source)
        for name in (
            "gate",
            "learned_gate",
            "source_quality",
            "selected_probability",
            "probability_margin",
            "normalized_entropy",
            "stagnation_score",
        ):
            means[name].add(source.get(name))
            distributions[name].add(source.get(name))
        entropy = source.get("normalized_entropy")
        stagnation = source.get("stagnation_score")
        if (
            selected_source
            and not isinstance(entropy, bool)
            and isinstance(entropy, (int, float))
            and math.isfinite(float(entropy))
        ):
            entropy_observations += 1
            high_entropy_selected += int(float(entropy) >= 0.94)
        if (
            selected_source
            and not isinstance(stagnation, bool)
            and isinstance(stagnation, (int, float))
            and math.isfinite(float(stagnation))
        ):
            stagnation_observations += 1
            high_stagnation_selected += int(float(stagnation) >= 0.75)

        scores = _mapping(model.get("retrieval"), "model.retrieval").get(
            "cosine_scores"
        )
        if isinstance(scores, list) and len(scores) >= 2:
            try:
                means["top1_top2_cosine_gap"].add(
                    float(scores[0]) - float(scores[1])
                )
                distributions["top1_top2_cosine_gap"].add(
                    float(scores[0]) - float(scores[1])
                )
            except (TypeError, ValueError):
                pass

        update = record.get("factual_update_after_replan")
        if isinstance(update, Mapping):
            if int(update.get("observation_updates", 0) or 0) > 0:
                factual_update_records += 1
                event_write_records += int(update.get("event_written") is True)
            try:
                executed_environment_actions += int(
                    update.get("executed_environment_prefix_count", 0) or 0
                )
            except (TypeError, ValueError):
                pass
            try:
                repeated_attempt_max = max(
                    repeated_attempt_max,
                    int(update.get("repeated_attempt_count", 0)),
                )
            except (TypeError, ValueError):
                pass
        input_stats_records += int(
            "raw_camera_stats" in record and "model_input_stats" in record
        )
        action_stats_records += int(
            "model_action_chunk_stats" in record
            and "environment_action_chunk_stats" in record
        )

        event = _event_key(source.get("selected_event_id"))
        if not selected_source or event is None:
            continue
        selected_event_counts[event] += 1
        previous = previous_event_by_episode.get(episode)
        if previous is not None:
            thread_comparable += 1
            same_rollout = event[:2] == previous[:2]
            monotonic = event[2] + 16 >= previous[2]
            thread_continuations += int(same_rollout and monotonic)
            exact_event_repetitions += int(event == previous)
            same_phase = abs(event[2] - previous[2]) <= 16
            phase_locked_transitions += int(same_phase)
            phase_run_by_episode[episode] = (
                phase_run_by_episode.get(episode, 1) + 1 if same_phase else 1
            )
        else:
            phase_run_by_episode[episode] = 1
        maximum_consecutive_phase_replans = max(
            maximum_consecutive_phase_replans,
            phase_run_by_episode[episode],
        )
        previous_event_by_episode[episode] = event

    replans = kinds["replan"]
    ended = len(episode_ended)

    def ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    source_rate = ratio(memory_selected, replans)
    event_rate = ratio(event_write_records, factual_update_records)
    dominant_event_count = (
        selected_event_counts.most_common(1)[0][1]
        if selected_event_counts
        else 0
    )
    dominant_event = (
        selected_event_counts.most_common(1)[0][0]
        if selected_event_counts
        else None
    )
    dominant_event_rate = ratio(dominant_event_count, memory_selected)
    exact_event_repetition_rate = ratio(
        exact_event_repetitions, thread_comparable
    )
    phase_lock_rate = ratio(phase_locked_transitions, thread_comparable)
    failure_signals: list[str] = []
    if source_rate is not None and source_rate < 0.01:
        failure_signals.append("memory_source_starvation")
    if source_rate is not None and source_rate > 0.99:
        failure_signals.append("memory_source_always_on")
    stagnation_median = distributions["stagnation_score"].quantile(0.5)
    if stagnation_median is not None and stagnation_median >= 0.75:
        failure_signals.append("factual_stagnation_saturated")
    if event_rate is not None and event_rate < 0.05:
        failure_signals.append("episode_event_write_starvation")
    if (
        memory_selected >= 100
        and dominant_event_rate is not None
        and dominant_event_rate >= 0.25
    ):
        failure_signals.append("dominant_memory_event_collapse")
    if (
        thread_comparable >= 100
        and phase_lock_rate is not None
        and phase_lock_rate >= 0.80
        and maximum_consecutive_phase_replans >= 16
    ):
        failure_signals.append("memory_event_phase_lock")

    return {
        "schema": "warm.rmbench-evidence-analysis",
        "version": 1,
        "headers": headers,
        "record_counts": dict(sorted(kinds.items())),
        "episodes_started": len(episode_started),
        "episodes_ended": ended,
        "episodes_successful": successes,
        "success_rate": ratio(successes, ended),
        "replans": replans,
        "candidate_selection_rate": ratio(candidate_selected, replans),
        "memory_source_acceptance_rate": source_rate,
        "high_entropy_memory_source_rate": ratio(
            high_entropy_selected, entropy_observations
        ),
        "high_stagnation_memory_source_rate": ratio(
            high_stagnation_selected, stagnation_observations
        ),
        "thread_continuation_rate": ratio(
            thread_continuations, thread_comparable
        ),
        "exact_event_repetition_rate": exact_event_repetition_rate,
        "phase_locked_transition_rate": phase_lock_rate,
        "maximum_consecutive_phase_replans": (
            maximum_consecutive_phase_replans
        ),
        "unique_selected_memory_events": len(selected_event_counts),
        "dominant_selected_memory_event": (
            None
            if dominant_event is None
            else {
                "dataset_id": dominant_event[0],
                "episode_index": dominant_event[1],
                "start_frame": dominant_event[2],
                "count": dominant_event_count,
                "rate": dominant_event_rate,
            }
        ),
        "maximum_repeated_attempt_count": repeated_attempt_max,
        "factual_update_records": factual_update_records,
        "episode_event_write_rate": event_rate,
        "executed_environment_actions": executed_environment_actions,
        "input_stats_coverage": ratio(input_stats_records, replans),
        "action_stats_coverage": ratio(action_stats_records, replans),
        "means": {name: mean.value() for name, mean in means.items()},
        "quantiles": {
            name: {
                "p50": distribution.quantile(0.5),
                "p90": distribution.quantile(0.9),
                "p99": distribution.quantile(0.99),
            }
            for name, distribution in distributions.items()
        },
        "failure_signals": failure_signals,
    }


def analyze_file(path: Path) -> dict[str, Any]:
    def records() -> Iterable[Mapping[str, Any]]:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    value = json.loads(stripped)
                except json.JSONDecodeError as error:
                    raise EvidenceAnalysisError(
                        f"invalid JSON on line {line_number}"
                    ) from error
                yield _mapping(value, f"record {line_number}")

    return analyze_records(records())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    path = args.evidence.expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"evidence file does not exist: {path}")
    try:
        result = analyze_file(path)
    except EvidenceAnalysisError as error:
        raise SystemExit(f"WARM_EVIDENCE_ANALYSIS_ERROR: {error}") from error
    encoded = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
