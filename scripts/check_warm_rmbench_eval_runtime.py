#!/usr/bin/env python3
"""Fail-fast validation for the persistent WARM/RMBench evaluation runtime.

The official RMBench installer pins an older PyTorch release.  WARM evaluation
must instead keep the exact training runtime (PyTorch 2.7.1 + CUDA 12.8) while
adding the simulator stack required by the pinned RMBench checkout.  This
checker validates that combined runtime before any immutable evaluation output
is created.

The optional patch mode applies the two source fixes made by the pinned
RMBench installer to SAPIEN and MPLib.  It is only used by the one-time
bootstrap script; ACP jobs run this file in read-only validation mode.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


RMBENCH_REVISION = "57ee09cbc6267bc36ca0ac2d8d1c5c3b245c112c"
CUROBO_REVISION = "d64c4b005459db10c5dd867d8b30a87d5bda9bdb"
RUNTIME_SCHEMA = "warm.rmbench-eval-runtime"
RUNTIME_VERSION = 3
RMBENCH_ASSET_SCHEMA = "warm.rmbench-simulator-assets"
RMBENCH_ASSET_VERSION = 1
RMBENCH_ASSET_REPOSITORY = "TianxingChen/RMBench"
RMBENCH_ASSET_REVISION = "855e90e1213d150bf4889130e83398f107314681"
RMBENCH_REQUIRED_OBJECTS = (
    "002_breadbasket",
    "003_cover",
    "004_numbercard",
    "005_button",
    "006_check_button",
    "007_T_block",
    "008_shelf",
    "009_toycar",
    "010_mouse",
    "011_stapler",
    "012_bell",
    "013_playingcards",
    "017_battery_slot_gauge",
    "018_battery",
    "cube",
    "sapien-block1",
    "sapien-block2",
    "vis_box",
)

EXPECTED_DISTRIBUTIONS = {
    "numpy": "1.26.4",
    "torch": "2.7.1+cu128",
    "scipy": "1.10.1",
    "transforms3d": "0.4.2",
    "sapien": "3.0.0b1",
    "mplib": "0.2.1",
    "gymnasium": "0.29.1",
    "trimesh": "4.4.3",
    "pyglet": "1.5.31",
    "toppra": "0.6.3",
    "warp-lang": "1.11.1",
    "scikit-image": "0.22.0",
    "imageio": "2.37.0",
    "lazy_loader": "0.4",
    "tifffile": "2024.9.20",
    "pillow": "11.1.0",
    "packaging": "24.2",
    "yourdfpy": "0.0.60",
    "lxml": "5.3.0",
    "six": "1.17.0",
    "numpy-quaternion": "2024.0.13",
    "nvidia_curobo": "0.7.8",
}

REQUIRED_IMPORTS = (
    "sapien",
    "sapien.core",
    "sapien.physx",
    "sapien.render",
    "sapien.sensor",
    "mplib",
    "mplib.sapien_utils",
    "open3d",
    "toppra",
    "transforms3d",
    "trimesh",
    "gymnasium",
    "cv2",
    "h5py",
    "imageio",
    "lazy_loader",
    "tifffile",
    "PIL",
    "packaging",
    "warp",
    "skimage",
    "yourdfpy",
    "quaternion",
    "curobo.types.math",
    "curobo.types.robot",
    "curobo.wrap.reacher.motion_gen",
)

RMBENCH_SENTINELS = (
    "script/eval_policy.py",
    "script/requirements.txt",
    "envs/_base_task.py",
    "envs/robot/planner.py",
    "task_config/_embodiment_config.yml",
)


class RuntimeValidationError(RuntimeError):
    """Raised when the persistent simulator runtime is incomplete."""


@dataclass(frozen=True)
class PatchResult:
    path: Path
    changed: bool


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _package_root(name: str) -> Path:
    spec = importlib.util.find_spec(name)
    if spec is None:
        raise RuntimeValidationError(f"cannot locate installed package {name!r}")
    locations = list(spec.submodule_search_locations or [])
    if locations:
        return Path(locations[0]).resolve()
    if spec.origin is None:
        raise RuntimeValidationError(f"installed package {name!r} has no origin")
    return Path(spec.origin).resolve().parent


def _replace_required(
    path: Path,
    pattern: str,
    replacement: str,
    required_pattern: str,
) -> PatchResult:
    source = path.read_text(encoding="utf-8")
    updated, count = re.subn(pattern, replacement, source)
    if count:
        path.write_text(updated, encoding="utf-8")
        source = updated
    if re.search(required_pattern, source) is None:
        raise RuntimeValidationError(
            f"required official RMBench patch is absent from {path}"
        )
    return PatchResult(path=path, changed=bool(count))


def apply_official_runtime_patches() -> tuple[PatchResult, ...]:
    """Apply the exact compatibility fixes from pinned RMBench ``_install.sh``."""

    sapien_loader = _package_root("sapien") / "wrapper" / "urdf_loader.py"
    mplib_planner = _package_root("mplib") / "planner.py"
    for path in (sapien_loader, mplib_planner):
        if not path.is_file():
            raise RuntimeValidationError(f"required patch target is absent: {path}")

    results = [
        _replace_required(
            sapien_loader,
            r'open\((urdf_file|srdf_file),\s*"r"\)',
            r'open(\1, "r", encoding="utf-8")',
            r'open\((?:urdf_file|srdf_file),\s*"r",\s*encoding="utf-8"\)',
        ),
        _replace_required(
            mplib_planner,
            (
                r"if\s+np\.linalg\.norm\(delta_twist\)\s*<\s*1e-4\s+or\s+"
                r"collide\s+or\s+not\s+within_joint_limit:"
            ),
            (
                "if np.linalg.norm(delta_twist) < 1e-4 "
                "or not within_joint_limit:"
            ),
            (
                r"if\s+np\.linalg\.norm\(delta_twist\)\s*<\s*1e-4\s+or\s+"
                r"not\s+within_joint_limit:"
            ),
        ),
    ]
    sapien_source = sapien_loader.read_text(encoding="utf-8")
    sapien_updated, count = re.subn(
        r'urdf_file\[:-4\]\s*\+\s*"srdf"',
        'urdf_file[:-4] + ".srdf"',
        sapien_source,
    )
    if count:
        sapien_loader.write_text(sapien_updated, encoding="utf-8")
        results[0] = PatchResult(path=sapien_loader, changed=True)
    return tuple(results)


def validate_official_runtime_patches() -> None:
    sapien_source = (
        _package_root("sapien") / "wrapper" / "urdf_loader.py"
    ).read_text(encoding="utf-8")
    if re.search(
        r'open\((?:urdf_file|srdf_file),\s*"r",\s*encoding="utf-8"\)',
        sapien_source,
    ) is None:
        raise RuntimeValidationError(
            "SAPIEN URDF loader is missing RMBench's UTF-8 compatibility patch"
        )
    if re.search(r'urdf_file\[:-4\]\s*\+\s*"\.srdf"', sapien_source) is None:
        raise RuntimeValidationError(
            "SAPIEN URDF loader does not derive the .srdf sidecar correctly"
        )

    mplib_source = (_package_root("mplib") / "planner.py").read_text(
        encoding="utf-8"
    )
    patched = re.search(
        r"if\s+np\.linalg\.norm\(delta_twist\)\s*<\s*1e-4\s+or\s+"
        r"not\s+within_joint_limit:",
        mplib_source,
    )
    stale = re.search(
        r"if\s+np\.linalg\.norm\(delta_twist\)\s*<\s*1e-4\s+or\s+"
        r"collide\s+or\s+not\s+within_joint_limit:",
        mplib_source,
    )
    if patched is None or stale is not None:
        raise RuntimeValidationError(
            "MPLib planner is missing RMBench's screw-planning compatibility patch"
        )


def validate_versions() -> dict[str, str]:
    errors = []
    resolved: dict[str, str] = {}
    for name, expected in EXPECTED_DISTRIBUTIONS.items():
        actual = _distribution_version(name)
        if actual is None:
            errors.append(f"{name} is not installed (expected {expected})")
            continue
        resolved[name] = actual
        if actual != expected:
            errors.append(f"{name}={actual}, expected exactly {expected}")
    if sys.version_info[:2] != (3, 10):
        errors.append(
            f"Python={sys.version_info.major}.{sys.version_info.minor}, expected 3.10"
        )
    if errors:
        raise RuntimeValidationError("; ".join(errors))
    return resolved


def validate_manifest(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise RuntimeValidationError(
            f"runtime provenance manifest is absent: {path}; "
            "run scripts/bootstrap_warm_rmbench_eval_env.sh once"
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema": RUNTIME_SCHEMA,
        "version": RUNTIME_VERSION,
        "rmbench_revision": RMBENCH_REVISION,
        "curobo_revision": CUROBO_REVISION,
        "rmbench_asset_repository": RMBENCH_ASSET_REPOSITORY,
        "rmbench_asset_revision": RMBENCH_ASSET_REVISION,
        "dependency_profile": "rgb-only-minimal-v3",
        "open3d_provider": "warm-rgb-only-import-guard",
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise RuntimeValidationError(
                f"runtime manifest {key}={value.get(key)!r}, expected {expected!r}"
            )
    return value


def _asset_tree_metadata(root: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    count = 0
    total_bytes = 0
    files = []
    for category in ("embodiments", "objects"):
        category_root = root / category
        if category_root.is_dir():
            files.extend(path for path in category_root.rglob("*") if path.is_file())
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\n")
        count += 1
        total_bytes += size
    return {
        "file_count": count,
        "total_bytes": total_bytes,
        "tree_metadata_sha256": digest.hexdigest(),
    }


def validate_asset_provenance(
    runtime_manifest: dict[str, object],
    rmbench_root: Path,
) -> dict[str, object]:
    """Bind deployed simulator files to the pinned HF asset snapshot."""

    raw_path = runtime_manifest.get("rmbench_asset_manifest")
    if not isinstance(raw_path, str) or not raw_path:
        raise RuntimeValidationError(
            "runtime manifest does not identify an RMBench asset manifest"
        )
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise RuntimeValidationError(
            f"RMBench asset provenance manifest is absent: {path}"
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema": RMBENCH_ASSET_SCHEMA,
        "version": RMBENCH_ASSET_VERSION,
        "repo_id": RMBENCH_ASSET_REPOSITORY,
        "revision": RMBENCH_ASSET_REVISION,
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise RuntimeValidationError(
                f"asset manifest {key}={value.get(key)!r}, expected {expected!r}"
            )
    asset_root = path.parent.resolve()
    deployed = rmbench_root / "assets"
    for category in ("embodiments", "objects"):
        link = deployed / category
        if not link.is_symlink():
            raise RuntimeValidationError(
                f"deployed RMBench asset category is not a provenance-bound "
                f"symlink: {link}"
            )
        if link.resolve() != (asset_root / category).resolve():
            raise RuntimeValidationError(
                f"deployed RMBench asset category points outside the attested "
                f"snapshot: {link} -> {link.resolve()}"
            )
    actual = _asset_tree_metadata(asset_root)
    for key in ("file_count", "total_bytes", "tree_metadata_sha256"):
        if value.get(key) != actual[key]:
            raise RuntimeValidationError(
                f"RMBench asset tree {key}={actual[key]!r}, "
                f"manifest={value.get(key)!r}"
            )
    runtime_tree = runtime_manifest.get("rmbench_asset_tree_metadata_sha256")
    if runtime_tree != actual["tree_metadata_sha256"]:
        raise RuntimeValidationError(
            "runtime manifest and deployed RMBench asset tree differ"
        )
    return value


def validate_rmbench_checkout(root: Path, task: str) -> None:
    missing = [str(root / item) for item in RMBENCH_SENTINELS if not (root / item).is_file()]
    task_file = root / "envs" / f"{task}.py"
    if not task_file.is_file():
        missing.append(str(task_file))
    required_asset_dirs = (
        root / "assets" / "embodiments" / "aloha-agilex",
        root / "assets" / "objects",
    )
    for directory in required_asset_dirs:
        if not directory.is_dir() or not any(directory.iterdir()):
            missing.append(f"{directory} (missing or empty)")
    embodiment_config = (
        root / "assets" / "embodiments" / "aloha-agilex" / "config.yml"
    )
    if not embodiment_config.is_file():
        missing.append(str(embodiment_config))
    for object_name in RMBENCH_REQUIRED_OBJECTS:
        directory = root / "assets" / "objects" / object_name
        if not directory.is_dir() or not any(directory.iterdir()):
            missing.append(f"{directory} (official9 asset missing or empty)")
    if task_file.is_file():
        task_source = task_file.read_text(encoding="utf-8")
        model_names = set(
            re.findall(r'modelname\s*=\s*["\']([^"\']+)["\']', task_source)
        )
        for model_name in sorted(model_names):
            directory = root / "assets" / "objects" / model_name
            if not directory.is_dir() or not any(directory.iterdir()):
                missing.append(f"{directory} (task asset missing or empty)")
    if missing:
        raise RuntimeValidationError(
            "pinned RMBench checkout is incomplete: " + ", ".join(missing)
        )


def validate_rgb_only_protocol(root: Path) -> None:
    """Prove that replacing import-only Open3D with a guard is protocol-safe."""

    import yaml

    config_path = root / "task_config" / "demo_clean.yml"
    if not config_path.is_file():
        raise RuntimeValidationError(
            f"official RGB-only protocol config is absent: {config_path}"
        )
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_type = value.get("data_type")
    if not isinstance(data_type, dict):
        raise RuntimeValidationError(
            f"invalid data_type mapping in {config_path}"
        )
    if data_type.get("rgb") is not True:
        raise RuntimeValidationError("formal RMBench protocol must enable RGB")
    forbidden = (
        "depth",
        "pointcloud",
        "mesh_segmentation",
        "actor_segmentation",
    )
    enabled = [name for name in forbidden if data_type.get(name) is not False]
    if enabled:
        raise RuntimeValidationError(
            "minimal Open3D-guard runtime is valid only for the pinned "
            "RGB-only protocol; expected explicit false for: "
            + ", ".join(enabled)
        )


def validate_open3d_guard() -> None:
    module = importlib.import_module("open3d")
    if getattr(module, "__warm_rgb_only_shim__", False) is not True:
        raise RuntimeValidationError(
            "WARM's fail-closed Open3D RGB-only import guard is absent"
        )


def import_required_modules(names: Iterable[str]) -> None:
    errors = []
    for name in names:
        try:
            importlib.import_module(name)
        except Exception as error:  # Import closure errors are the desired signal.
            errors.append(f"{name}: {type(error).__name__}: {error}")
    if errors:
        raise RuntimeValidationError(
            "simulator import closure is incomplete:\n  " + "\n  ".join(errors)
        )


def import_rmbench_task(root: Path, task: str) -> None:
    original = list(sys.path)
    original_cwd = Path.cwd()
    sys.path.insert(0, str(root))
    sys.path.append(str(root / "description" / "utils"))
    try:
        # The pinned official evaluator resolves object manifests and several
        # simulator resources relative to the repository root at import time.
        # Formal evaluation already runs from its root-shaped runtime overlay;
        # make the standalone preflight reproduce that contract as well.
        os.chdir(root)
        for name in ("envs", "envs._base_task", "envs.robot.robot", f"envs.{task}"):
            importlib.import_module(name)
        entrypoint = root / "script" / "eval_policy.py"
        spec = importlib.util.spec_from_file_location(
            "_warm_rmbench_official_eval_preflight",
            entrypoint,
        )
        if spec is None or spec.loader is None:
            raise RuntimeValidationError(
                f"cannot load official evaluation entrypoint: {entrypoint}"
            )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as error:
        raise RuntimeValidationError(
            f"official RMBench evaluator import failed for {task}: "
            f"{type(error).__name__}: {error}"
        ) from error
    finally:
        os.chdir(original_cwd)
        sys.path[:] = original


def validate_system_tools() -> None:
    missing = [name for name in ("ffmpeg",) if shutil.which(name) is None]
    if missing:
        raise RuntimeValidationError(
            "required official-evaluator executables are absent from PATH: "
            + ", ".join(missing)
        )


def validate_cuda_device() -> tuple[str, float]:
    import torch

    count = torch.cuda.device_count()
    if count != 1:
        raise RuntimeValidationError(
            f"expected exactly one visible CUDA device, got {count}"
        )
    properties = torch.cuda.get_device_properties(0)
    gib = properties.total_memory / 1024**3
    if "H100" not in properties.name or gib < 75:
        raise RuntimeValidationError(
            "formal specialist evaluation requires one 80GB H100; "
            f"got {properties.name} ({gib:.1f} GiB)"
        )
    capability = (int(properties.major), int(properties.minor))
    if capability != (9, 0):
        raise RuntimeValidationError(
            f"expected H100 compute capability 9.0, got {capability}"
        )
    return properties.name, gib


def renderer_smoke() -> None:
    """Exercise the same headless ray-tracing path used by official RMBench."""

    import sapien.core as sapien
    from sapien.render import set_global_config

    set_global_config(max_num_materials=50000, max_num_textures=50000)
    engine = sapien.Engine()
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)
    sapien.render.set_camera_shader_dir("rt")
    sapien.render.set_ray_tracing_samples_per_pixel(1)
    sapien.render.set_ray_tracing_path_depth(1)
    sapien.render.set_ray_tracing_denoiser("oidn")
    scene = engine.create_scene(sapien.SceneConfig())
    scene.add_ground(altitude=0.0)
    scene.step()
    scene.update_render()
    del scene
    del renderer
    del engine


def _default_manifest() -> Path:
    return Path(sys.executable).resolve().parent.parent / "warm_rmbench_runtime.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rmbench-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--runtime-manifest", type=Path, default=_default_manifest())
    parser.add_argument("--apply-patches", action="store_true")
    parser.add_argument("--skip-renderer-smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        if args.apply_patches:
            for result in apply_official_runtime_patches():
                print(
                    f"runtime_patch path={result.path} "
                    f"changed={str(result.changed).lower()}"
                )
        versions = validate_versions()
        runtime_manifest = validate_manifest(args.runtime_manifest)
        validate_rmbench_checkout(args.rmbench_root.resolve(), args.task)
        validate_asset_provenance(runtime_manifest, args.rmbench_root.resolve())
        validate_rgb_only_protocol(args.rmbench_root.resolve())
        validate_open3d_guard()
        validate_official_runtime_patches()
        validate_system_tools()
        import_required_modules(REQUIRED_IMPORTS)
        import_rmbench_task(args.rmbench_root.resolve(), args.task)
        gpu_name, gpu_gib = validate_cuda_device()
        if not args.skip_renderer_smoke:
            renderer_smoke()
    except (RuntimeValidationError, json.JSONDecodeError) as error:
        print(f"RMBENCH_EVAL_RUNTIME_ERROR: {error}", file=sys.stderr)
        return 2
    except Exception as error:
        print(
            "RMBENCH_EVAL_RUNTIME_ERROR: "
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2

    print(
        "rmbench_eval_runtime_ok "
        f"python={sys.version.split()[0]} "
        f"torch={versions['torch']} "
        f"sapien={versions['sapien']} "
        f"mplib={versions['mplib']} "
        f"gpu={gpu_name!r} "
        f"gpu_gib={gpu_gib:.1f} "
        f"task={args.task} "
        f"renderer={'skipped' if args.skip_renderer_smoke else 'ok'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
