from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_episode_index_initialization_remains_inside_constructor() -> None:
    source = (
        ROOT / "src" / "fastwam" / "datasets" / "lerobot" / "base_lerobot_dataset.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BaseLerobotDataset"
    )
    methods = {
        node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)
    }

    def assigns_episode_index(method: ast.FunctionDef) -> bool:
        for node in ast.walk(method):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and target.attr == "episode_data_index"
                for target in targets
            ):
                return True
        return False

    assert assigns_episode_index(methods["__init__"])
    assert not assigns_episode_index(methods["_instruction_for_sample"])


def test_robot_video_dataset_does_not_construct_partial_state() -> None:
    source = (
        ROOT / "src" / "fastwam" / "datasets" / "lerobot" / "robot_video_dataset.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "accelerate":
            imported.extend(alias.name for alias in node.names)
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "PartialState":
                raise AssertionError(
                    "RobotVideoDataset must not construct PartialState before Accelerator"
                )
            if isinstance(func, ast.Attribute) and func.attr == "PartialState":
                raise AssertionError(
                    "RobotVideoDataset must not construct PartialState before Accelerator"
                )
def test_piper_stage_b_acp_writes_tmp_acp_logs() -> None:
    source = (
        ROOT / "scripts" / "real" / "acp_piper_warm_v2_20hz_stage_b.sh"
    ).read_text(encoding="utf-8")
    assert "source \"${PROJECT_DIR}/scripts/real/_acp_log.sh\"" in source
    assert "piper_acp_begin_logs" in source
    assert "piper_resolve_conda_bins" in source
    assert "piper_gpu_preflight" in source
    helper = (ROOT / "scripts" / "real" / "_acp_log.sh").read_text(encoding="utf-8")
    assert "tmp/acp_logs" in helper
    assert "logs-acp-${RUN_ID}.txt.gz" in helper
    assert "do not fall back to PATH" in helper
