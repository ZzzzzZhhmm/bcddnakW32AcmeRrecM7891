#!/usr/bin/env python3
"""Install the pinned official RMBench simulator assets into persistent AFS.

RMBench's policy-training data and its SAPIEN simulator assets are different
artifacts.  The official Git checkout intentionally ignores the large asset
trees and downloads them from the RMBench Hugging Face dataset.  This helper:

1. downloads only the ALOHA embodiment and object trees at the protocol-pinned
   dataset revision;
2. keeps the resumable snapshot outside the Git checkout;
3. renders the official ``*_tmp.yml`` embodiment templates deterministically;
4. records a self-checking provenance manifest; and
5. links the two ignored trees into the pinned official checkout.

It never deletes a non-empty directory.  A partial or conflicting deployment
therefore fails closed instead of silently replacing user data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


ASSET_SCHEMA = "warm.rmbench-simulator-assets"
ASSET_VERSION = 1
RMBENCH_HF_REPOSITORY = "TianxingChen/RMBench"
RMBENCH_HF_REVISION = "855e90e1213d150bf4889130e83398f107314681"
ASSET_PATTERNS = (
    "embodiments/aloha-agilex/**",
    "objects/**",
)
ASSET_MANIFEST_NAME = ".warm_rmbench_assets.json"

# Every object required by the official nine-task suite.  The final four are
# common RMBench utility assets used by task helpers rather than appearing as
# literal ``modelname=`` arguments in every task module.
REQUIRED_OBJECT_DIRECTORIES = (
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


class AssetBootstrapError(RuntimeError):
    """Raised when the official asset snapshot cannot be proven complete."""


def _is_nonempty_directory(path: Path) -> bool:
    return path.is_dir() and next(path.iterdir(), None) is not None


def _payload_files(root: Path) -> tuple[Path, ...]:
    files: list[Path] = []
    for category in ("embodiments", "objects"):
        category_root = root / category
        if not category_root.is_dir():
            continue
        files.extend(path for path in category_root.rglob("*") if path.is_file())
    return tuple(sorted(files, key=lambda path: path.relative_to(root).as_posix()))


def _tree_metadata(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    total_bytes = 0
    for path in _payload_files(root):
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


def render_embodiment_configs(asset_root: Path, rmbench_root: Path) -> list[str]:
    """Apply the pinned official path-template operation without prompting."""

    embodiment_root = asset_root / "embodiments"
    rendered: list[str] = []
    replacement = str(rmbench_root.resolve())
    for template in sorted(embodiment_root.rglob("*_tmp.yml")):
        output = template.with_name(template.name.replace("_tmp.yml", ".yml"))
        source = template.read_text(encoding="utf-8")
        value = source.replace("${ASSETS_PATH}", replacement)
        value = value.replace("$ASSETS_PATH", replacement)
        output.write_text(value, encoding="utf-8")
        rendered.append(output.relative_to(asset_root).as_posix())
    return rendered


def validate_asset_layout(asset_root: Path) -> dict[str, Any]:
    """Validate the complete RGB-only official9 asset closure."""

    root = asset_root.resolve()
    aloha = root / "embodiments" / "aloha-agilex"
    objects = root / "objects"
    missing: list[str] = []
    if not _is_nonempty_directory(aloha):
        missing.append("embodiments/aloha-agilex (missing or empty)")
    if not _is_nonempty_directory(objects):
        missing.append("objects (missing or empty)")

    config = aloha / "config.yml"
    if not config.is_file() or config.stat().st_size == 0:
        missing.append("embodiments/aloha-agilex/config.yml")
    else:
        config_source = config.read_text(encoding="utf-8")
        if "$ASSETS_PATH" in config_source:
            missing.append("embodiments/aloha-agilex/config.yml (unresolved template)")

    if aloha.is_dir():
        if not any(aloha.rglob("*.urdf")):
            missing.append("embodiments/aloha-agilex/**/*.urdf")
        if not any(aloha.rglob("*.srdf")):
            missing.append("embodiments/aloha-agilex/**/*.srdf")
        mesh_suffixes = {".dae", ".glb", ".obj", ".stl"}
        if not any(
            path.is_file() and path.suffix.lower() in mesh_suffixes
            for path in aloha.rglob("*")
        ):
            missing.append("embodiments/aloha-agilex mesh closure")

    for name in REQUIRED_OBJECT_DIRECTORIES:
        directory = objects / name
        if not _is_nonempty_directory(directory):
            missing.append(f"objects/{name} (missing or empty)")

    if missing:
        raise AssetBootstrapError(
            "official RMBench simulator assets are incomplete: " + ", ".join(missing)
        )
    metadata = _tree_metadata(root)
    if metadata["file_count"] < 100:
        raise AssetBootstrapError(
            "official RMBench asset snapshot contains implausibly few files: "
            f"{metadata['file_count']}"
        )
    return metadata


def _manifest_path(asset_root: Path) -> Path:
    return asset_root / ASSET_MANIFEST_NAME


def validate_asset_manifest(asset_root: Path) -> dict[str, Any]:
    path = _manifest_path(asset_root)
    if not path.is_file():
        raise AssetBootstrapError(f"asset provenance manifest is absent: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema": ASSET_SCHEMA,
        "version": ASSET_VERSION,
        "repo_id": RMBENCH_HF_REPOSITORY,
        "revision": RMBENCH_HF_REVISION,
        "allow_patterns": list(ASSET_PATTERNS),
    }
    mismatches = [
        f"{key}={value.get(key)!r}, expected {expected!r}"
        for key, expected in required.items()
        if value.get(key) != expected
    ]
    actual = validate_asset_layout(asset_root)
    for key in ("file_count", "total_bytes", "tree_metadata_sha256"):
        if value.get(key) != actual[key]:
            mismatches.append(
                f"{key}={value.get(key)!r}, current tree={actual[key]!r}"
            )
    if mismatches:
        raise AssetBootstrapError(
            "asset provenance mismatch: " + "; ".join(mismatches)
        )
    return value


def write_asset_manifest(
    asset_root: Path,
    *,
    source: str,
    rendered_configs: Iterable[str],
) -> dict[str, Any]:
    metadata = validate_asset_layout(asset_root)
    value: dict[str, Any] = {
        "schema": ASSET_SCHEMA,
        "version": ASSET_VERSION,
        "repo_id": RMBENCH_HF_REPOSITORY,
        "revision": RMBENCH_HF_REVISION,
        "allow_patterns": list(ASSET_PATTERNS),
        "source": source,
        "rendered_configs": sorted(set(rendered_configs)),
        **metadata,
    }
    path = _manifest_path(asset_root)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return value


def _assert_pinned_checkout(root: Path, revision: str) -> None:
    if not (root / ".git").exists():
        raise AssetBootstrapError(f"RMBench Git checkout is absent: {root}")
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    actual = completed.stdout.strip()
    if completed.returncode or actual != revision:
        raise AssetBootstrapError(
            f"RMBench checkout revision={actual or '<unavailable>'}, "
            f"expected {revision}"
        )


def _link_category(source: Path, target: Path) -> None:
    if target.is_symlink():
        if target.resolve() != source.resolve():
            raise AssetBootstrapError(
                f"conflicting asset symlink {target} -> {target.resolve()}; "
                f"expected {source.resolve()}"
            )
        return
    if target.exists():
        if target.is_dir() and not any(
            child.is_file() or child.is_symlink()
            for child in target.rglob("*")
        ):
            # Failed/manual setup often leaves category/aloha-agilex as an
            # empty directory tree.  Removing only fileless directories is
            # safe and keeps retries automatic; any file or link still causes
            # the fail-closed branch below.
            for directory, _, _ in os.walk(target, topdown=False):
                Path(directory).rmdir()
        else:
            raise AssetBootstrapError(
                f"refusing to replace non-empty asset path: {target}; "
                "move it aside or pass it as --asset-source with a valid manifest"
            )
    target.symlink_to(source.resolve(), target_is_directory=True)


def deploy_asset_links(
    *,
    asset_root: Path,
    rmbench_root: Path,
    rmbench_revision: str,
) -> None:
    _assert_pinned_checkout(rmbench_root, rmbench_revision)
    validate_asset_manifest(asset_root)
    checkout_assets = rmbench_root / "assets"
    checkout_assets.mkdir(parents=True, exist_ok=True)
    for category in ("embodiments", "objects"):
        _link_category(asset_root / category, checkout_assets / category)
    validate_asset_layout(checkout_assets)

    completed = subprocess.run(
        [
            "git",
            "-C",
            str(rmbench_root),
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode or completed.stdout.strip():
        raise AssetBootstrapError(
            "asset deployment changed the pinned RMBench Git worktree: "
            f"{completed.stdout.strip() or completed.stderr.strip()}"
        )


def _download_snapshot(
    *,
    asset_root: Path,
    max_workers: int,
) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise AssetBootstrapError(
            "huggingface_hub is required in the base WARM environment"
        ) from error
    asset_root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=RMBENCH_HF_REPOSITORY,
        repo_type="dataset",
        revision=RMBENCH_HF_REVISION,
        allow_patterns=list(ASSET_PATTERNS),
        local_dir=str(asset_root),
        max_workers=max_workers,
        etag_timeout=60,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rmbench-root", type=Path, required=True)
    parser.add_argument("--asset-store", type=Path, required=True)
    parser.add_argument("--asset-source", type=Path)
    parser.add_argument("--rmbench-revision", required=True)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument(
        "--trust-existing-source",
        action="store_true",
        help=(
            "Adopt a complete explicit --asset-source without a WARM manifest. "
            "Use only when that source was independently obtained at the pinned "
            "official Hugging Face revision."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        if args.rmbench_revision != "57ee09cbc6267bc36ca0ac2d8d1c5c3b245c112c":
            raise AssetBootstrapError(
                f"unsupported RMBench code revision: {args.rmbench_revision}"
            )
        if args.max_workers < 1 or args.max_workers > 64:
            raise AssetBootstrapError("--max-workers must be in [1, 64]")

        rmbench_root = args.rmbench_root.expanduser().resolve()
        if args.asset_source is not None:
            asset_root = args.asset_source.expanduser().resolve()
            try:
                previous = validate_asset_manifest(asset_root)
            except AssetBootstrapError:
                if not args.trust_existing_source:
                    raise
                rendered = render_embodiment_configs(asset_root, rmbench_root)
                write_asset_manifest(
                    asset_root,
                    source="trusted-explicit-existing-source",
                    rendered_configs=rendered,
                )
            else:
                rendered = render_embodiment_configs(asset_root, rmbench_root)
                write_asset_manifest(
                    asset_root,
                    source=str(previous["source"]),
                    rendered_configs=rendered,
                )
        else:
            asset_root = args.asset_store.expanduser().resolve()
            try:
                previous = validate_asset_manifest(asset_root)
                print(f"asset_snapshot_reused={asset_root}")
            except (AssetBootstrapError, json.JSONDecodeError):
                _download_snapshot(
                    asset_root=asset_root,
                    max_workers=args.max_workers,
                )
                rendered = render_embodiment_configs(asset_root, rmbench_root)
                write_asset_manifest(
                    asset_root,
                    source="huggingface_snapshot_download",
                    rendered_configs=rendered,
                )
            else:
                rendered = render_embodiment_configs(asset_root, rmbench_root)
                write_asset_manifest(
                    asset_root,
                    source=str(previous["source"]),
                    rendered_configs=rendered,
                )

        manifest = validate_asset_manifest(asset_root)
        deploy_asset_links(
            asset_root=asset_root,
            rmbench_root=rmbench_root,
            rmbench_revision=args.rmbench_revision,
        )
    except (AssetBootstrapError, json.JSONDecodeError, OSError) as error:
        print(f"RMBENCH_ASSET_BOOTSTRAP_ERROR: {error}", file=sys.stderr)
        return 2

    print(f"RMBENCH_ASSETS_READY root={asset_root}")
    print(f"asset_manifest={_manifest_path(asset_root)}")
    print(f"asset_revision={manifest['revision']}")
    print(f"asset_files={manifest['file_count']}")
    print(f"asset_bytes={manifest['total_bytes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
