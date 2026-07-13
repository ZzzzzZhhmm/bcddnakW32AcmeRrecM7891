"""Dependency-free contract tests for the formal LIBERO online evaluator.

The developer workstation intentionally has no Torch, Hydra, or LIBERO.  These
tests inspect the evaluator AST so security-critical policy separation and
rollout lifecycle rules still regress locally before the server gate runs.
"""

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
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            assert isinstance(node, ast.FunctionDef)
            return node
    raise AssertionError(f"missing evaluator function {name}")


def _source(node: ast.AST) -> str:
    value = ast.get_source_segment(SOURCE, node)
    assert value is not None
    return value


def _policy_if(function: ast.FunctionDef, policy: str) -> ast.If:
    for node in ast.walk(function):
        if not isinstance(node, ast.If):
            continue
        test = _source(node.test)
        constants = {
            child.value
            for child in ast.walk(node.test)
            if isinstance(child, ast.Constant)
        }
        if "source_policy" in test and policy in constants:
            return node
    raise AssertionError(f"missing {policy} policy branch")


def test_fixed_retriever_uses_offline_encoder_compute_dtype() -> None:
    loader = _function("_load_warm_online_runtime")
    fixed = _policy_if(loader, "fixed_context_top1")
    fixed_source = _source(fixed)
    assert "_dino_torch_dtype_from_encoder_contract" in fixed_source
    assert "dino_torch_dtype=dino_torch_dtype" in fixed_source

    dtype_helper = _source(_function("_dino_torch_dtype_from_encoder_contract"))
    assert 'value.get("compute")' in dtype_helper
    assert '"bfloat16": torch.bfloat16' in dtype_helper
    assert "changed while its DINO dtype was read" in dtype_helper


def test_gaussian_null_branch_cannot_open_retrieval_artifacts() -> None:
    loader = _function("_load_warm_online_runtime")
    fixed = _policy_if(loader, "fixed_context_top1")
    null_source = "\n".join(_source(node) for node in fixed.orelse)
    for forbidden in (
        "FrozenDinoOnlineRetriever",
        "bank_directory",
        "dino_checkpoint_path",
        "encoder_contract_path",
        "catalog_path",
        "audit_report_path",
    ):
        assert forbidden not in null_source
    assert "_build_null_image_adapter" in null_source

    adapter = _source(_function("_build_null_image_adapter"))
    assert "camera_contract_path" in adapter
    assert "expected_sha256" in adapter
    assert 'configured_concat_mode != (\n        "horizontal"' in adapter
    for forbidden in ("event bank", "DINO checkpoint", "catalog", "audit report"):
        assert forbidden in adapter.split('"""', 2)[1]


def test_fixed_model_call_accepts_only_bound_step_and_sampling_parameters() -> None:
    predict = _function("_predict_action_chunk")
    fixed = _policy_if(predict, "fixed_context_top1")
    assignments = [
        node
        for statement in fixed.body
        for node in ast.walk(statement)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "infer_kwargs"
            for target in node.targets
        )
    ]
    assert len(assignments) == 1
    value = assignments[0].value
    assert isinstance(value, ast.Dict)
    literal_keys = {
        key.value for key in value.keys if isinstance(key, ast.Constant)
    }
    assert literal_keys == {"online_step"}
    assert any(key is None for key in value.keys), "sampling kwargs must be unpacked"


def test_online_query_lifecycle_uses_absolute_simulator_step() -> None:
    issue = _source(_function("issue_query_id"))
    assert "frame_index <= self._last_frame_index" in issue
    assert "self._last_frame_index = frame_index" in issue

    episode = _source(_function("run_single_episode"))
    assert "frame_index=t" in episode
    assert "online_runtime.begin_episode(episode_idx)" in episode
    for evidence in (
        '"configured_wait_steps"',
        '"configured_policy_max_steps"',
        '"configured_replan_steps"',
        '"replan_count"',
    ):
        assert evidence in episode


def test_runtime_attests_stats_git_bddl_and_emits_comparable_evidence() -> None:
    loader = _source(_function("_load_warm_online_runtime"))
    assert "loaded_dataset_stats_sha256" in loader
    assert "_git_identity(project_root)" in loader
    assert "git_commit != contract.git_commit" in loader
    assert '_model_artifact_path(model, "tokenizer")' in loader
    assert "contract.tokenizer_tree_sha256" in loader

    task = _source(_function("run_single_task"))
    assert "sha256_file(online_runtime.bddl_path)" in task
    assert "task_description != online_runtime.contract.task_description" in task

    predict = _source(_function("_predict_action_chunk"))
    for evidence in (
        "raw_camera_sha256",
        "processed_camera_sha256",
        "model_input_sha256",
        "prompt_sha256",
        "proprio_sha256",
        "derived_seed",
        "evaluator_latency_s",
    ):
        assert evidence in predict
    assert "evaluation_namespace_sha256" in predict


def test_formal_rollout_is_bound_to_one_strict_online_pair_side() -> None:
    loader = _source(_function("_load_warm_online_runtime"))
    assert '_required_online_path(online_cfg, "pair_contract_path")' in loader
    assert "WarmOnlinePairContract.from_dict" in loader
    assert "pair_contract_file_sha256 = sha256_file(pair_contract_path)" in loader
    assert "online pair contract changed while it was read" in loader
    assert "online pair contract changed during runtime setup" in loader
    assert 'training_attestation_path = _required_online_path' in loader
    assert "verify_training_attestation(" in loader
    assert 'getattr(model, "_warm_loaded_checkpoint_sha256", None)' in loader
    assert "loaded model or checkpoint changed during runtime setup" in loader
    assert "_validate_online_pair_membership" in loader

    membership = _source(_function("_validate_online_pair_membership"))
    for evidence in (
        "fixed_online_run_contract_sha256",
        "gaussian_null_online_run_contract_sha256",
        "fixed_resolved_eval_config_sha256",
        "gaussian_null_resolved_eval_config_sha256",
        "fixed_warm_checkpoint_sha256",
        "gaussian_null_warm_checkpoint_sha256",
        "fixed_training_attestation_sha256",
        "gaussian_null_training_attestation_sha256",
        "pair_contract.git_commit",
        "online_contract.git_commit",
    ):
        assert evidence in membership
    assert 'pair_side = "fixed"' in membership
    assert 'pair_side = "gaussian_null"' in membership

    header = _source(_function("result_header"))
    identity = _source(_function("pair_identity"))
    for field in ("pair_contract_sha256", "comparison_kind", "side"):
        assert f'"{field}"' in identity
    assert '"version": 2' in header
    assert "**self.pair_identity()" in header
    assert '"pair_contract": self.pair_contract.to_dict()' in header
    assert '"pair_contract_file_sha256"' in header

    prediction = _source(_function("_predict_action_chunk"))
    task = _source(_function("run_single_task"))
    assert "**online_runtime.pair_identity()" in prediction
    assert "**online_runtime.pair_identity()" in task
    assert "online_runtime.attest_pair_contract_file()" in task


def test_pair_binding_does_not_change_cross_policy_query_or_seed_identity() -> None:
    issue = _source(_function("issue_query_id"))
    assert "make_online_query_id(" in issue
    assert "pair_contract" not in issue
    assert "pair_side" not in issue

    predict = _source(_function("_predict_action_chunk"))
    assert "derive_online_query_seed(" in predict
    seed_calls = [
        node
        for node in ast.walk(_function("_predict_action_chunk"))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "derive_online_query_seed"
    ]
    assert len(seed_calls) == 1
    assert "pair_contract" not in _source(seed_calls[0])
