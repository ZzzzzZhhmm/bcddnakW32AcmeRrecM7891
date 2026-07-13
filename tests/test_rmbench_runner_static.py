"""Dependency-light structural checks for the external RMBench launchers."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SINGLE_SOURCE = (ROOT / "experiments/rmbench/eval_rmbench_single.py").read_text(
    encoding="utf-8"
)
MANAGER_SOURCE = (ROOT / "experiments/rmbench/run_rmbench_manager.py").read_text(
    encoding="utf-8"
)


def test_single_runner_uses_only_external_official_entrypoint_and_fixed_protocol() -> None:
    ast.parse(SINGLE_SOURCE)
    for required in (
        "validate_read_only_checkout",
        "validate_hf_revision_marker",
        "prepare_runtime_overlay",
        "validate_policy_source",
        '"script/eval_policy.py"',
        "RMBENCH_TASK_CONFIG",
        "RMBENCH_EPISODES_PER_TASK",
        "parse_official_result",
        "validate_official_seed_namespace",
        "discover_new_result_file",
    ):
        assert required in SINGLE_SOURCE
    assert "demo_randomized" not in SINGLE_SOURCE
    assert 'env["PYTHONPATH"]' in SINGLE_SOURCE
    assert 'cwd=str(runtime_root)' in SINGLE_SOURCE
    assert 'policy_kind == "fastwam_baseline"' in SINGLE_SOURCE
    assert "actual_accepted_seeds.npy" in SINGLE_SOURCE


def test_manager_forbids_arbitrary_subsets_and_writes_structured_results() -> None:
    ast.parse(MANAGER_SOURCE)
    assert "does not accept arbitrary task subsets" in MANAGER_SOURCE
    assert "assert_exact_task_sequence" in MANAGER_SOURCE
    for output in ("summary.json", "summary.csv", "failures.json", "protocol_manifest.json"):
        assert output in MANAGER_SOURCE
    assert 'suite == "official9"' in MANAGER_SOURCE
    assert "actual_accepted_seed_sha256" in MANAGER_SOURCE
