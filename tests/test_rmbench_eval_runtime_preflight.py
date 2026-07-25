from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_warm_rmbench_eval_runtime.py"
SPEC = importlib.util.spec_from_file_location("rmbench_runtime_check", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runtime = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runtime
SPEC.loader.exec_module(runtime)


def test_runtime_manifest_is_exactly_revision_bound(tmp_path: Path) -> None:
    manifest = tmp_path / "warm_rmbench_runtime.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": runtime.RUNTIME_SCHEMA,
                "version": runtime.RUNTIME_VERSION,
                "rmbench_revision": runtime.RMBENCH_REVISION,
                "curobo_revision": runtime.CUROBO_REVISION,
            }
        ),
        encoding="utf-8",
    )
    assert runtime.validate_manifest(manifest)["curobo_revision"] == (
        runtime.CUROBO_REVISION
    )

    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["curobo_revision"] = "0" * 40
    manifest.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(runtime.RuntimeValidationError, match="curobo_revision"):
        runtime.validate_manifest(manifest)


def test_runtime_versions_keep_checkpoint_torch_and_official_simulator() -> None:
    assert runtime.EXPECTED_DISTRIBUTIONS["torch"] == "2.7.1+cu128"
    assert runtime.EXPECTED_DISTRIBUTIONS["numpy"] == "1.26.4"
    assert runtime.EXPECTED_DISTRIBUTIONS["sapien"] == "3.0.0b1"
    assert runtime.EXPECTED_DISTRIBUTIONS["mplib"] == "0.2.1"
    assert runtime.EXPECTED_DISTRIBUTIONS["warp-lang"] == "1.11.1"
    assert runtime.EXPECTED_DISTRIBUTIONS["scikit-image"] == "0.22.0"
    assert runtime.EXPECTED_DISTRIBUTIONS["nvidia_curobo"] == "0.7.8"


def test_official_patch_application_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sapien_root = tmp_path / "sapien"
    mplib_root = tmp_path / "mplib"
    (sapien_root / "wrapper").mkdir(parents=True)
    mplib_root.mkdir()
    sapien_loader = sapien_root / "wrapper" / "urdf_loader.py"
    sapien_loader.write_text(
        'with open(urdf_file, "r") as f:\n'
        '    srdf_file = urdf_file[:-4] + "srdf"\n'
        'with open(srdf_file, "r") as f:\n'
        "    pass\n",
        encoding="utf-8",
    )
    planner = mplib_root / "planner.py"
    planner.write_text(
        "if np.linalg.norm(delta_twist) < 1e-4 or collide "
        'or not within_joint_limit:\n    return {"status": "failed"}\n',
        encoding="utf-8",
    )

    roots = {"sapien": sapien_root, "mplib": mplib_root}
    monkeypatch.setattr(runtime, "_package_root", lambda name: roots[name])

    first = runtime.apply_official_runtime_patches()
    assert all(result.changed for result in first)
    runtime.validate_official_runtime_patches()
    second = runtime.apply_official_runtime_patches()
    assert all(not result.changed for result in second)


def test_required_import_closure_contains_planner_and_renderer_dependencies() -> None:
    required = set(runtime.REQUIRED_IMPORTS)
    assert {
        "sapien.core",
        "sapien.render",
        "mplib.sapien_utils",
        "open3d",
        "curobo.wrap.reacher.motion_gen",
    } <= required
