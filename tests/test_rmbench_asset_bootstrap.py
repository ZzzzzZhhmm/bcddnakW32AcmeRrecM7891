from __future__ import annotations

import importlib.util
import io
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
    monkeypatch.setattr(
        assets,
        "_tree_metadata",
        lambda _root: {
            "file_count": assets.EXPECTED_ASSET_FILE_COUNT,
            "total_bytes": assets.EXPECTED_ASSET_TOTAL_BYTES,
            "tree_metadata_sha256": "unused",
        },
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    source = assets._download_snapshot(
        asset_root=tmp_path / "assets",
        max_workers=7,
    )

    assert observed["revision"] == assets.RMBENCH_HF_REVISION
    assert observed["max_workers"] == 7
    assert observed["etag_timeout"] == assets.DEFAULT_NETWORK_TIMEOUT_SECONDS
    assert observed["hf_hub_offline"] is None
    assert observed["datasets_offline"] is None
    assert source.startswith("huggingface_snapshot_download:")


def test_snapshot_metadata_timeout_honors_slow_network_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def fake_snapshot_download(**kwargs: object) -> None:
        observed.update(kwargs)

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.snapshot_download = fake_snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.setattr(
        assets,
        "_tree_metadata",
        lambda _root: {
            "file_count": assets.EXPECTED_ASSET_FILE_COUNT,
            "total_bytes": assets.EXPECTED_ASSET_TOTAL_BYTES,
            "tree_metadata_sha256": "unused",
        },
    )
    monkeypatch.setenv("HF_HUB_ETAG_TIMEOUT", "900")

    assets._download_snapshot(asset_root=tmp_path / "assets", max_workers=3)

    assert observed["etag_timeout"] == 900


def test_endpoint_candidates_are_ordered_and_deduplicated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "RMBENCH_HF_ENDPOINTS",
        "https://example.invalid/, https://huggingface.co, "
        "https://example.invalid",
    )
    assert assets._endpoint_candidates() == (
        "https://example.invalid",
        "https://huggingface.co",
    )


def test_direct_file_download_resumes_and_verifies_lfs_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"pinned-asset-payload"
    destination = tmp_path / "objects" / "005_button" / "model.glb"
    destination.parent.mkdir(parents=True)
    partial = destination.with_name(destination.name + ".warm-partial")
    partial.write_bytes(payload[:7])
    observed_ranges: list[str | None] = []

    class FakeResponse(io.BytesIO):
        status = 206

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            self.close()

        def getcode(self) -> int:
            return self.status

    def fake_urlopen(
        request: object,
        *,
        timeout: int,
    ) -> FakeResponse:
        del timeout
        headers = dict(request.header_items())  # type: ignore[attr-defined]
        value = headers.get("Range")
        observed_ranges.append(value)
        assert value == "bytes=7-"
        return FakeResponse(payload[7:])

    monkeypatch.setattr(assets.urllib.request, "urlopen", fake_urlopen)
    entry = {
        "path": "objects/005_button/model.glb",
        "size": len(payload),
        "algorithm": "sha256",
        "digest": assets.hashlib.sha256(payload).hexdigest(),
    }

    assets._download_file(
        destination=destination,
        entry=entry,
        endpoints=("https://example.invalid",),
        timeout=123,
    )

    assert destination.read_bytes() == payload
    assert observed_ranges == ["bytes=7-"]
    assert not partial.exists()


def test_download_plan_rejects_unexpected_revision_closure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(assets, "EXPECTED_ASSET_FILE_COUNT", 1)
    monkeypatch.setattr(assets, "EXPECTED_ASSET_TOTAL_BYTES", 3)
    plan = {
        "repo_id": assets.RMBENCH_HF_REPOSITORY,
        "revision": assets.RMBENCH_HF_REVISION,
        "files": [
            {
                "path": "objects/005_button/model.json",
                "size": 3,
                "algorithm": "git-sha1",
                "digest": "0" * 40,
            }
        ],
    }
    assert assets._validate_download_plan(plan) == plan["files"]

    plan["files"][0]["size"] = 4
    with pytest.raises(assets.AssetBootstrapError, match="audited revision"):
        assets._validate_download_plan(plan)
