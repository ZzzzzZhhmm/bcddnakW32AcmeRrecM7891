from __future__ import annotations

import ast
from pathlib import Path


LOADER = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "fastwam"
    / "models"
    / "wan22"
    / "helpers"
    / "loader.py"
)


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1
    return matches[0]


def _called_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        if isinstance(child.func, ast.Name):
            names.add(child.func.id)
        elif isinstance(child.func, ast.Attribute):
            names.add(child.func.attr)
    return names


def test_public_vae_loader_has_no_full_component_resolution_path() -> None:
    tree = ast.parse(LOADER.read_text(encoding="utf-8"))
    loader = _function(tree, "load_wan22_vae_only")
    calls = _called_names(loader)

    assert "_resolve_vae_config" in calls
    assert "_load_registered_model" in calls
    assert "_resolve_configs" not in calls
    assert "load_wan22_ti2v_5b_components" not in calls
    assert "WanVideoDiT" not in calls
    assert "HuggingfaceTokenizer" not in calls


def test_vae_resolver_mentions_only_vae_checkpoint_patterns() -> None:
    tree = ast.parse(LOADER.read_text(encoding="utf-8"))
    resolver = _function(tree, "_resolve_vae_config")
    literals = {
        node.value
        for node in ast.walk(resolver)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    joined = "\n".join(literals).lower()

    assert "wan2.2_vae" in joined
    assert "diffusion_pytorch_model" not in joined
    assert "umt5" not in joined
