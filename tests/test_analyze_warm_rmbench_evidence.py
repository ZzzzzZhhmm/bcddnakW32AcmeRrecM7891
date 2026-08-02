from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_warm_rmbench_evidence.py"
SPEC = importlib.util.spec_from_file_location("warm_evidence_analysis", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_evidence_summary_distinguishes_candidate_and_source_selection() -> None:
    event = {"dataset_id": "bank", "episode_index": 3, "start_frame": 10}
    records = [
        {"kind": "header"},
        {"kind": "episode_begin", "episode_index": 0},
        {
            "kind": "replan",
            "episode_index": 0,
            "raw_camera_stats": {},
            "model_input_stats": {},
            "model_action_chunk_stats": {},
            "environment_action_chunk_stats": {},
            "factual_update_after_replan": {
                "repeated_attempt_count": 2,
                "observation_updates": 1,
                "event_written": False,
                "executed_environment_prefix_count": 4,
            },
            "model": {
                "retrieval": {"cosine_scores": [0.9, 0.89]},
                "source": {
                    "candidate_selected": True,
                    "memory_selected": False,
                    "selected_event_id": event,
                    "gate": 0.0,
                    "learned_gate": 0.8,
                    "source_quality": 0.0,
                    "selected_probability": 0.04,
                    "probability_margin": 0.001,
                    "normalized_entropy": 0.99,
                    "stagnation_score": 0.2,
                },
            },
        },
        {
            "kind": "replan",
            "episode_index": 0,
            "factual_update_after_replan": {
                "observation_updates": 1,
                "event_written": True,
                "executed_environment_prefix_count": 4,
            },
            "model": {
                "retrieval": {"cosine_scores": [0.92, 0.80]},
                "source": {
                    "candidate_selected": True,
                    "memory_selected": True,
                    "selected_event_id": event,
                    "gate": 0.6,
                    "learned_gate": 0.8,
                    "source_quality": 0.75,
                    "selected_probability": 0.7,
                    "probability_margin": 0.5,
                    "normalized_entropy": 0.3,
                    "stagnation_score": 0.1,
                },
            },
        },
        {
            "kind": "episode_end",
            "episode_index": 0,
            "success": True,
        },
    ]

    result = MODULE.analyze_records(records)

    assert result["success_rate"] == 1.0
    assert result["candidate_selection_rate"] == 1.0
    assert result["memory_source_acceptance_rate"] == 0.5
    assert result["high_entropy_memory_source_rate"] == 0.0
    assert result["maximum_repeated_attempt_count"] == 2
    assert result["input_stats_coverage"] == 0.5
    assert result["action_stats_coverage"] == 0.5
    assert result["episode_event_write_rate"] == 0.5
    assert result["executed_environment_actions"] == 8
    assert result["quantiles"]["learned_gate"]["p50"] is not None
    assert result["failure_signals"] == []


def test_missing_new_diagnostics_are_reported_as_unknown() -> None:
    records = [
        {"kind": "episode_begin", "episode_index": 0},
        {
            "kind": "replan",
            "episode_index": 0,
            "model": {
                "retrieval": {"cosine_scores": [0.5, 0.49]},
                "source": {
                    "candidate_selected": True,
                    "memory_selected": True,
                    "selected_event_id": {
                        "dataset_id": "old-bank",
                        "episode_index": 1,
                        "start_frame": 4,
                    },
                },
            },
        },
    ]

    result = MODULE.analyze_records(records)

    assert result["memory_source_acceptance_rate"] == 1.0
    assert result["high_entropy_memory_source_rate"] is None
    assert result["high_stagnation_memory_source_rate"] is None
