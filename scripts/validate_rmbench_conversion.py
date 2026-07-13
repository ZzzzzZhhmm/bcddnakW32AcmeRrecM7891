#!/usr/bin/env python3
"""Validate one immutable official RMBench-to-LeRobot artifact tree."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from fastwam.datasets.rmbench.constants import (
    CATALOG_FILENAME,
    CONVERSION_SCHEMA,
    CONVERSION_SCHEMA_VERSION,
    MANIFEST_FILENAME,
    OFFICIAL_EPISODES_PER_TASK,
    OFFICIAL_RMBENCH_TASKS,
    OFFICIAL_TASK_CONFIG,
)
from fastwam.datasets.rmbench.source import canonical_json_bytes, file_sha256


class RMBenchConversionValidationError(RuntimeError):
    """Raised when converted-data provenance or bytes no longer match."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--rmbench-code-revision", required=True)
    parser.add_argument(
        "--skip-artifact-byte-hashes",
        action="store_true",
        help=(
            "Validate manifest/catalog identities without rehashing every MP4. "
            "Only use after a complete audit has been published."
        ),
    )
    return parser


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RMBenchConversionValidationError(f"{label} must be a mapping")
    return value


def _safe_artifact_path(root: Path, relpath: str) -> Path:
    pure = PurePosixPath(relpath)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise RMBenchConversionValidationError(
            f"unsafe converted-artifact path: {relpath!r}"
        )
    path = (root / Path(*pure.parts)).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RMBenchConversionValidationError(
            f"converted-artifact path escapes dataset root: {relpath!r}"
        ) from exc
    return path


def validate_conversion(
    dataset_root: Path,
    *,
    source_revision: str,
    rmbench_code_revision: str,
    hash_artifacts: bool,
) -> Mapping[str, Any]:
    root = dataset_root.expanduser().resolve(strict=True)
    manifest_path = root / "meta" / MANIFEST_FILENAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RMBenchConversionValidationError(
            f"cannot read conversion manifest: {manifest_path}"
        ) from exc
    manifest = _mapping(manifest, "conversion manifest")
    expected_keys = {
        "schema",
        "schema_version",
        "source",
        "output",
        "protocol",
        "episodes",
        "manifest_sha256",
    }
    if set(manifest) != expected_keys:
        raise RMBenchConversionValidationError(
            "conversion manifest has unknown or missing top-level fields"
        )
    if (
        manifest["schema"] != CONVERSION_SCHEMA
        or manifest["schema_version"] != CONVERSION_SCHEMA_VERSION
    ):
        raise RMBenchConversionValidationError("conversion schema mismatch")
    unsigned = dict(manifest)
    claimed_manifest_sha = unsigned.pop("manifest_sha256")
    actual_manifest_sha = sha256(canonical_json_bytes(unsigned)).hexdigest()
    if claimed_manifest_sha != actual_manifest_sha:
        raise RMBenchConversionValidationError("conversion manifest SHA-256 mismatch")

    source = _mapping(manifest["source"], "source")
    if source.get("dataset") != "TianxingChen/RMBench":
        raise RMBenchConversionValidationError("unexpected RMBench source dataset")
    if source.get("revision") != source_revision:
        raise RMBenchConversionValidationError("RMBench dataset revision mismatch")
    if source.get("rmbench_code_revision") != rmbench_code_revision:
        raise RMBenchConversionValidationError("RMBench code revision mismatch")
    if source.get("task_config") != OFFICIAL_TASK_CONFIG:
        raise RMBenchConversionValidationError("RMBench task config mismatch")

    protocol = _mapping(manifest["protocol"], "protocol")
    if tuple(protocol.get("official_task_allow_list", ())) != OFFICIAL_RMBENCH_TASKS:
        raise RMBenchConversionValidationError("official nine-task order mismatch")
    if protocol.get("episodes_per_task") != OFFICIAL_EPISODES_PER_TASK:
        raise RMBenchConversionValidationError("official episode count mismatch")
    episodes = manifest["episodes"]
    if not isinstance(episodes, list) or len(episodes) != (
        len(OFFICIAL_RMBENCH_TASKS) * OFFICIAL_EPISODES_PER_TASK
    ):
        raise RMBenchConversionValidationError("conversion episode inventory mismatch")

    output = _mapping(manifest["output"], "output")
    if output.get("catalog_relpath") != f"meta/{CATALOG_FILENAME}":
        raise RMBenchConversionValidationError("catalog relative path mismatch")
    catalog = root / "meta" / CATALOG_FILENAME
    if file_sha256(catalog) != output.get("catalog_sha256"):
        raise RMBenchConversionValidationError("catalog SHA-256 mismatch")

    artifacts = output.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise RMBenchConversionValidationError("artifact inventory is empty")
    if sha256(canonical_json_bytes(artifacts)).hexdigest() != output.get(
        "artifact_tree_sha256"
    ):
        raise RMBenchConversionValidationError("artifact-tree SHA-256 mismatch")
    seen: set[str] = set()
    for row in artifacts:
        row = _mapping(row, "artifact row")
        if set(row) != {"path", "size", "sha256"}:
            raise RMBenchConversionValidationError("malformed artifact row")
        relpath = row["path"]
        if not isinstance(relpath, str) or relpath in seen:
            raise RMBenchConversionValidationError("duplicate/invalid artifact path")
        seen.add(relpath)
        path = _safe_artifact_path(root, relpath)
        if not path.is_file() or path.stat().st_size != row["size"]:
            raise RMBenchConversionValidationError(
                f"artifact missing or size changed: {relpath}"
            )
        if hash_artifacts and file_sha256(path) != row["sha256"]:
            raise RMBenchConversionValidationError(
                f"artifact SHA-256 changed: {relpath}"
            )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = validate_conversion(
        args.dataset_root,
        source_revision=args.source_revision,
        rmbench_code_revision=args.rmbench_code_revision,
        hash_artifacts=not args.skip_artifact_byte_hashes,
    )
    print(
        json.dumps(
            {
                "dataset_root": str(args.dataset_root.expanduser().resolve()),
                "manifest_sha256": manifest["manifest_sha256"],
                "artifact_tree_sha256": manifest["output"]["artifact_tree_sha256"],
                "artifact_byte_hashes_verified": not args.skip_artifact_byte_hashes,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
