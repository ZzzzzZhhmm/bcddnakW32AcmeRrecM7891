from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "bootstrap_warm_rmbench_assets.py"
SPEC = importlib.util.spec_from_file_location("rmbench_asset_bootstrap", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
assets = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = assets
SPEC.loader.exec_module(assets)


def _complete_asset_tree(root: Path, rmbench_root: Path) -> None:
    aloha = root / "embodiments" / "aloha-agilex"
    aloha.mkdir(parents=True)
    (aloha / "config_tmp.yml").write_text(
        "robot_path: ${ASSETS_PATH}/assets/embodiments/aloha-agilex\n",
        encoding="utf-8",
    )
    (aloha / "robot.urdf").write_text("<robot />", encoding="utf-8")
    (aloha / "robot.srdf").write_text("<robot />", encoding="utf-8")
    (aloha / "mesh.stl").write_bytes(b"mesh")
    for name in assets.REQUIRED_OBJECT_DIRECTORIES:
        directory = root / "objects" / name
        directory.mkdir(parents=True)
        (directory / "model_data0.json").write_text("{}", encoding="utf-8")
    # The production snapshot is much larger; synthetic filler verifies the
    # implausibly-small-tree guard without checking in binary fixtures.
    filler = root / "objects" / "005_button" / "visual"
    filler.mkdir()
    for index in range(100):
        (filler / f"{index:03d}.glb").write_bytes(b"x")
    assets.render_embodiment_configs(root, rmbench_root)


def test_asset_manifest_is_revision_bound_and_self_checking(tmp_path: Path) -> None:
    asset_root = tmp_path / "asset-store"
    rmbench_root = tmp_path / "RMBench-official"
    _complete_asset_tree(asset_root, rmbench_root)
    value = assets.write_asset_manifest(
        asset_root,
        source="unit-test",
        rendered_configs=[
            "embodiments/aloha-agilex/config.yml",
        ],
    )
    assert value["repo_id"] == assets.RMBENCH_HF_REPOSITORY
    assert value["revision"] == assets.RMBENCH_HF_REVISION
    assert value["file_count"] >= 100
    assert assets.validate_asset_manifest(asset_root) == value

    changed = asset_root / "objects" / "005_button" / "visual" / "000.glb"
    changed.write_bytes(b"different-size")
    with pytest.raises(assets.AssetBootstrapError, match="provenance mismatch"):
        assets.validate_asset_manifest(asset_root)


def test_embodiment_template_uses_the_stable_official_checkout(
    tmp_path: Path,
) -> None:
    asset_root = tmp_path / "asset-store"
    rmbench_root = tmp_path / "RMBench-official"
    _complete_asset_tree(asset_root, rmbench_root)
    config = (
        asset_root
        / "embodiments"
        / "aloha-agilex"
        / "config.yml"
    ).read_text(encoding="utf-8")
    assert str(rmbench_root.resolve()) in config
    assert "$ASSETS_PATH" not in config


def test_incomplete_asset_tree_lists_all_missing_official_objects(
    tmp_path: Path,
) -> None:
    aloha = tmp_path / "embodiments" / "aloha-agilex"
    aloha.mkdir(parents=True)
    (aloha / "config.yml").write_text("ok: true\n", encoding="utf-8")
    (tmp_path / "objects").mkdir()
    with pytest.raises(assets.AssetBootstrapError) as captured:
        assets.validate_asset_layout(tmp_path)
    message = str(captured.value)
    assert "objects/005_button" in message
    assert "objects/018_battery" in message
    assert "**/*.urdf" in message


def test_manifest_json_uses_exact_official_patterns(tmp_path: Path) -> None:
    asset_root = tmp_path / "asset-store"
    rmbench_root = tmp_path / "RMBench-official"
    _complete_asset_tree(asset_root, rmbench_root)
    assets.write_asset_manifest(
        asset_root,
        source="unit-test",
        rendered_configs=[],
    )
    payload = json.loads(
        (asset_root / assets.ASSET_MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert payload["allow_patterns"] == [
        "embodiments/aloha-agilex/**",
        "objects/**",
    ]


def test_link_category_recovers_only_fileless_partial_tree(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "payload").write_text("ready", encoding="utf-8")
    target = tmp_path / "target"
    (target / "aloha-agilex" / "meshes").mkdir(parents=True)
    try:
        assets._link_category(source, target)
    except OSError as error:
        pytest.skip(f"local filesystem does not permit symbolic links: {error}")
    assert target.is_symlink()
    assert target.resolve() == source.resolve()

    conflicting = tmp_path / "conflicting"
    conflicting.mkdir()
    (conflicting / "partial.bin").write_bytes(b"partial")
    with pytest.raises(assets.AssetBootstrapError, match="refusing to replace"):
        assets._link_category(source, conflicting)


def test_explicit_download_clears_inherited_offline_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def fake_snapshot_download(**kwargs: object) -> None:
        observed.update(kwargs)
        observed["hf_hub_offline"] = assets.os.environ.get("HF_HUB_OFFLINE")
        observed["datasets_offline"] = assets.os.environ.get("HF_DATASETS_OFFLINE")

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.snapshot_download = fake_snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    assets._download_snapshot(asset_root=tmp_path / "assets", max_workers=7)

    assert observed["revision"] == assets.RMBENCH_HF_REVISION
    assert observed["max_workers"] == 7
    assert observed["hf_hub_offline"] is None
    assert observed["datasets_offline"] is None
