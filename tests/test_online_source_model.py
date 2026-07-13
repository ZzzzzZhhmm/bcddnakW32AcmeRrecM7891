from __future__ import annotations

import ast
from pathlib import Path


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "fastwam"
    / "models"
    / "warm"
    / "source_model.py"
)


def _class_node() -> ast.ClassDef:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "WarmSourceFastWAM"
    )


def _method(name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    return next(
        node
        for node in _class_node().body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )


def test_fixed_inference_signature_exposes_only_bound_online_step() -> None:
    method = _method("infer_action")
    keyword_only = {argument.arg for argument in method.args.kwonlyargs}
    ordinary = {argument.arg for argument in method.args.args}
    assert "online_step" in keyword_only
    assert ordinary == {"self"}
    assert {
        "candidate_means",
        "candidate_valid_mask",
        "memory_enabled_mask",
        "action_source_context",
    }.isdisjoint(keyword_only | ordinary)


def test_source_model_contains_capability_validation_and_derived_seed_path() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert "def bind_online_retriever" in source
    assert source.count("retriever.validate_bound_step(") == 1
    assert "retriever.assert_owned_bound_step(" in source
    assert "type(retriever) is not FrozenDinoOnlineRetriever" in source
    assert "retriever.artifact_verified is not True" in source
    assert '"seed": int(online_step.derived_seed)' in source
    assert "warm_checkpoint_sha256" in source
    assert "training_run_contract_sha256" in source
    assert "validation_run_contract_sha256" in source
    assert "warm_online_telemetry" in source


def test_online_telemetry_records_reproducibility_evidence() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    for field in (
        '"query_id"',
        '"ranked_event_ids"',
        '"bank_rows"',
        '"cosine_scores"',
        '"candidate_valid_mask"',
        '"bank_manifest_sha256"',
        '"bank_content_sha256"',
        '"selected_event_id"',
        '"latency_s"',
    ):
        assert field in source


def test_checkpoint_reload_revokes_before_touching_new_path() -> None:
    method = _method("load_checkpoint")
    statements = method.body
    clear_hash = next(
        index
        for index, statement in enumerate(statements)
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and target.attr == "_warm_loaded_checkpoint_sha256"
            for target in statement.targets
        )
    )
    resolve_path = next(
        index
        for index, statement in enumerate(statements)
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "checkpoint_path"
            for target in statement.targets
        )
    )
    assert clear_hash < resolve_path


def test_checkpoint_schema_v2_binds_optional_dev_contract() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    assignments = {
        target.id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    version = assignments["WARM_SOURCE_CHECKPOINT_VERSION"]
    assert isinstance(version, ast.Constant) and version.value == 2
    source = SOURCE.read_text(encoding="utf-8")
    assert "warm_validation_run_contract" in source
    assert '"validation_run_contract_sha256"' in source
    assert "validate_validation_dataset" in source
