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
