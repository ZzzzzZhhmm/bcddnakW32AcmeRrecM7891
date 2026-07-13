#!/usr/bin/env python3
"""Build one immutable M2 source-only run contract from verified artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

from fastwam.memory.candidate_cache import (
    MANIFEST_FILENAME as CANDIDATE_MANIFEST_FILENAME,
)
from fastwam.memory.event_bank import MANIFEST_FILENAME as BANK_MANIFEST_FILENAME
from fastwam.memory.manifest import sha256_canonical_json, sha256_file
from fastwam.memory.runtime_candidates import RuntimeCandidateResolver
from fastwam.models.warm.source_contract import WarmSourceRunContract
from fastwam.utils.artifact_claim import artifact_claim


SUMMARY_SCHEMA = "warm.source-run-contract-build-summary"
SUMMARY_SCHEMA_VERSION = 1


class SourceRunContractBuildError(RuntimeError):
    """Raised when a source-run contract cannot safely bind its inputs."""


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _sha256(value: str) -> str:
    if (
        len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise argparse.ArgumentTypeError("must be a lowercase SHA-256 digest")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a closed M2 WARM source-run contract from one validated "
            "event bank, candidate cache, and base FastWAM checkpoint."
        )
    )
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--candidate-cache", required=True, type=Path)
    parser.add_argument("--base-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--query-split",
        choices=("train", "dev"),
        default="train",
        help="Must match the immutable candidate-cache build recipe.",
    )
    parser.add_argument(
        "--expected-query-corpus-sha256",
        type=_sha256,
        default=None,
        help="Optional independent assertion for the candidate query corpus.",
    )
    parser.add_argument(
        "--expected-action-horizon",
        type=_positive_int,
        default=None,
        help="Optional independent assertion for the model action horizon.",
    )
    parser.add_argument(
        "--expected-action-dim",
        type=_positive_int,
        default=None,
        help="Optional independent assertion for the model action dimension.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _lock_path(output: Path) -> Path:
    return output.parent / f".{output.name}.warm-artifact.lock"


def _assert_digest(
    path: Path,
    expected: str,
    *,
    label: str,
) -> None:
    try:
        actual = sha256_file(path)
    except OSError as exc:
        raise SourceRunContractBuildError(f"cannot re-read {label} at {path}") from exc
    if actual != expected:
        raise SourceRunContractBuildError(
            f"{label} changed while the source-run contract was being built"
        )


def _write_json_atomic(
    path: Path,
    value: Mapping[str, Any],
    *,
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"source-run contract already exists at {path}")
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    encoded = (
        json.dumps(
            dict(value),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _build_contract(
    resolver: RuntimeCandidateResolver,
    *,
    base_checkpoint_sha256: str,
    expected_action_horizon: int | None,
    expected_action_dim: int | None,
) -> WarmSourceRunContract:
    if resolver.query_stride != 1:
        raise SourceRunContractBuildError(
            "M2 source training requires candidate query_stride=1"
        )
    action_horizon = resolver.action_horizon
    action_dim = resolver.action_space.action_dim
    if (
        expected_action_horizon is not None
        and action_horizon != expected_action_horizon
    ):
        raise SourceRunContractBuildError(
            "event-bank/candidate action horizon does not match "
            f"--expected-action-horizon: artifact={action_horizon}, "
            f"expected={expected_action_horizon}"
        )
    if expected_action_dim is not None and action_dim != expected_action_dim:
        raise SourceRunContractBuildError(
            "event-bank action dimension does not match --expected-action-dim: "
            f"artifact={action_dim}, expected={expected_action_dim}"
        )

    action_contract = resolver.action_space
    return WarmSourceRunContract(
        bank_manifest_sha256=resolver.bank_manifest_sha256,
        bank_content_sha256=resolver.bank_content_sha256,
        candidate_manifest_sha256=resolver.candidate_manifest_sha256,
        query_corpus_sha256=resolver.query_corpus_sha256,
        catalog_sha256=resolver.query_catalog_sha256,
        audit_sha256=resolver.query_audit_sha256,
        normalization_stats_sha256=action_contract.normalization_stats_sha256,
        action_space_contract_sha256=sha256_canonical_json(
            action_contract.to_dict()
        ),
        base_checkpoint_sha256=base_checkpoint_sha256,
        query_split=resolver.query_split,
        global_sample_stride=resolver.query_stride,
        action_horizon=action_horizon,
        action_dim=action_dim,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    bank_directory = args.bank.expanduser().resolve()
    candidate_directory = args.candidate_cache.expanduser().resolve()
    base_checkpoint = args.base_checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()

    if bank_directory == candidate_directory:
        raise SourceRunContractBuildError(
            "--bank and --candidate-cache must be different artifact directories"
        )
    if _is_within(output, bank_directory) or _is_within(
        output, candidate_directory
    ):
        raise SourceRunContractBuildError(
            "--output must be outside the immutable bank and candidate-cache trees"
        )
    if output == base_checkpoint:
        raise SourceRunContractBuildError(
            "--output must not overwrite the base checkpoint"
        )
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"source-run contract already exists at {output}")
    if not base_checkpoint.is_file():
        raise SourceRunContractBuildError(
            f"base checkpoint is not a regular file: {base_checkpoint}"
        )

    resolver = RuntimeCandidateResolver.from_artifacts(
        bank_directory,
        candidate_directory,
        expected_query_split=args.query_split,
        expected_query_corpus_sha256=args.expected_query_corpus_sha256,
    )
    try:
        checkpoint_sha256 = sha256_file(base_checkpoint)
    except OSError as exc:
        raise SourceRunContractBuildError(
            f"cannot hash base checkpoint {base_checkpoint}"
        ) from exc
    contract = _build_contract(
        resolver,
        base_checkpoint_sha256=checkpoint_sha256,
        expected_action_horizon=args.expected_action_horizon,
        expected_action_dim=args.expected_action_dim,
    )

    bank_manifest_path = bank_directory / BANK_MANIFEST_FILENAME
    candidate_manifest_path = candidate_directory / CANDIDATE_MANIFEST_FILENAME
    output.parent.mkdir(parents=True, exist_ok=True)
    with artifact_claim(
        _lock_path(output), purpose=f"publish WARM source-run contract: {output}"
    ):
        if output.exists() and not args.overwrite:
            raise FileExistsError(
                f"source-run contract already exists at {output}"
            )
        _assert_digest(
            bank_manifest_path,
            contract.bank_manifest_sha256,
            label="event-bank manifest",
        )
        _assert_digest(
            candidate_manifest_path,
            contract.candidate_manifest_sha256,
            label="candidate-cache manifest",
        )
        _assert_digest(
            base_checkpoint,
            contract.base_checkpoint_sha256,
            label="base checkpoint",
        )
        _write_json_atomic(output, contract.to_dict(), overwrite=args.overwrite)

    summary: dict[str, Any] = {
        "schema": SUMMARY_SCHEMA,
        "version": SUMMARY_SCHEMA_VERSION,
        "output": str(output),
        "source_run_contract_sha256": contract.sha256,
        "contract": contract.to_dict(),
    }
    print(
        json.dumps(
            summary,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
