"""Dependency-free guards for the complete WARM LIBERO rollout path."""

from __future__ import annotations

import ast
from pathlib import Path


SOURCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "libero"
    / "eval_libero_single.py"
)
SOURCE = SOURCE_PATH.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def _function(name: str) -> ast.FunctionDef:
    for node in ast.walk(TREE):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function {name}")


def _source(name: str) -> str:
    value = ast.get_source_segment(SOURCE, _function(name))
    assert value is not None
    return value


def test_full_mode_is_single_checkpoint_and_source_only_keeps_pair_gate() -> None:
    loader = _source("_load_warm_online_runtime")
    assert 'online_mode == "full_retrospection"' in loader
    assert "pair_contract_path = (" in loader
    assert "None\n        if full_retrospection" in loader
    assert "if not full_retrospection:" in loader
    assert "_validate_online_pair_membership(" in loader
    assert 'pair_side = "full_retrospection"' in loader
    assert 'contract.source_policy != "fixed_context_top1"' in loader


def test_full_replan_passes_only_prior_history_then_commits_factual_output() -> None:
    predict = _source("_predict_action_chunk")
    history_pos = predict.index("retrospective_history_kwargs()")
    infer_pos = predict.index("model.infer_action(**infer_kwargs)")
    commit_pos = predict.index("commit_factual_replan_observation(")
    assert history_pos < infer_pos < commit_pos
    for field in (
        "episode_tokens",
        "episode_mask",
        "episode_action_summaries",
        "episode_action_mask",
    ):
        assert field in (
            Path(__file__).resolve().parents[1]
            / "src"
            / "fastwam"
            / "memory"
            / "online_episode_memory.py"
        ).read_text(encoding="utf-8")
    assert 'model_output.get("warm_factual_observation")' in _source(
        "commit_factual_replan_observation"
    )


def test_exact_executed_prefix_is_recorded_after_environment_step() -> None:
    episode = _source("run_single_episode")
    step_pos = episode.index("env.step(executed_action)")
    note_pos = episode.index("online_runtime.note_executed_action(")
    assert step_pos < note_pos
    assert "_executed_action_to_model_space(" in episode
    assert "online_runtime.end_episode()" in episode


def test_gripper_environment_transform_is_inverted_with_the_sign_flip() -> None:
    inverse = _source("_executed_action_to_model_space")
    assert "(np.float32(1.0) - raw[-1]) * np.float32(0.5)" in inverse
    assert "(raw[-1] + np.float32(1.0))" not in inverse


def test_episode_memory_time_excludes_initial_dummy_actions() -> None:
    episode = _source("run_single_episode")
    assert "frame_index=policy_action_step_count" in episode
    assert "frame_index=t," not in episode


def test_episode_lifecycle_resets_memory_and_records_terminal_evidence() -> None:
    begin = _source("begin_episode")
    assert "self.retrospective_episode_memory.begin_episode(episode_index)" in begin
    assert "self._executed_actions_since_replan.clear()" in begin
    end = _source("end_episode")
    assert "self.retrospective_episode_memory.end_episode()" in end
    assert '"unpaired_terminal_action_count"' in end
    assert '"unpaired_terminal_environment_actions_sha256"' in end
