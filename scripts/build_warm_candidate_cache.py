#!/usr/bin/env python3
"""Build and self-validate an offline WARM retrieval candidate cache."""

from __future__ import annotations

import argparse
import json
import os
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence
from uuid import uuid4

from fastwam.datasets.lerobot.audit import load_audit_report
from fastwam.datasets.lerobot.episode_catalog import EpisodeCatalog
from fastwam.memory.candidate_cache import (
    MANIFEST_FILENAME as CANDIDATE_CACHE_MANIFEST_FILENAME,
    CandidateCache,
    canonical_event_bank_content_hash,
)
from fastwam.memory.event_bank import (
    MANIFEST_FILENAME as EVENT_BANK_MANIFEST_FILENAME,
    EventBank,
)
from fastwam.memory.manifest import sha256_file
from fastwam.memory.offline_pipeline import (
    FeatureDataBinding,
    build_candidate_cache_from_collection,
    load_feature_cache_collection,
    validate_event_bank_data_binding,
    validate_feature_collection_against_catalog,
    validate_query_collection_against_bank,
)
from fastwam.utils.artifact_claim import artifact_claim


SUMMARY_SCHEMA = "warm.candidate-cache-build-summary"
SUMMARY_SCHEMA_VERSION = 1


class CandidateCacheBuildError(RuntimeError):
    """Raised when artifact paths or the event-bank snapshot are unsafe."""


def _path_order_key(path: Path) -> str:
    return os.path.normcase(str(path))


def _artifact_lock_path(target: Path) -> Path:
    """Return the deterministic sibling lock protecting one artifact path."""

    return target.parent / f".{target.name}.warm-artifact.lock"


@contextmanager
def _claim_publication_targets(
    targets: Sequence[tuple[Path, str]],
) -> Iterator[None]:
    """Claim all publication targets in stable order to avoid lock inversion."""

    target_keys = {_path_order_key(target) for target, _ in targets}
    entries = [
        (_artifact_lock_path(target), target, purpose)
        for target, purpose in targets
    ]
    lock_keys = [_path_order_key(lock_path) for lock_path, _, _ in entries]
    if len(set(lock_keys)) != len(lock_keys):
        raise CandidateCacheBuildError(
            "candidate-cache publication targets resolve to the same sibling lock"
        )
    if target_keys.intersection(lock_keys):
        raise CandidateCacheBuildError(
            "an artifact target collides with a sibling publication lock path"
        )

    entries.sort(key=lambda entry: _path_order_key(entry[0]))
    for lock_path, _, _ in entries:
        lock_path.parent.mkdir(parents=True, exist_ok=True)

    with ExitStack() as stack:
        for lock_path, target, purpose in entries:
            stack.enter_context(
                artifact_claim(
                    lock_path,
                    purpose=f"{purpose}: {target}",
                )
            )
        yield


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build an exact, leave-entire-episode-out candidate cache from a "
            "validated WARM event bank and query feature caches."
        )
    )
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--audit-report", required=True, type=Path)
    parser.add_argument(
        "--feature-cache",
        action="append",
        default=[],
        type=Path,
        help="Episode feature-cache .npz; repeat as needed.",
    )
    parser.add_argument(
        "--feature-list",
        action="append",
        default=[],
        type=Path,
        help=(
            "UTF-8 file containing one feature-cache path per line; repeat as "
            "needed. Blank lines and lines beginning with # are ignored, and "
            "relative entries are resolved from the list file's directory."
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--action-horizon",
        type=_positive_int,
        default=None,
        help="Defaults to the canonical action horizon recorded by the bank.",
    )
    parser.add_argument("--query-stride", required=True, type=_positive_int)
    parser.add_argument("--top-k", required=True, type=_positive_int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Optionally atomically write the printed JSON build summary.",
    )
    return parser


def _feature_paths(
    direct_paths: Sequence[Path],
    list_paths: Sequence[Path],
) -> tuple[Path, ...]:
    paths: list[Path] = [path.expanduser().resolve() for path in direct_paths]
    for list_path_value in list_paths:
        list_path = list_path_value.expanduser().resolve()
        try:
            lines = list_path.read_text(encoding="utf-8-sig").splitlines()
        except (OSError, UnicodeError) as exc:
            raise ValueError(f"cannot read feature list {list_path}") from exc
        for line_number, line in enumerate(lines, start=1):
            entry = line.strip()
            if not entry or entry.startswith("#"):
                continue
            path = Path(entry).expanduser()
            if not path.is_absolute():
                path = list_path.parent / path
            try:
                paths.append(path.resolve())
            except OSError as exc:
                raise ValueError(
                    f"cannot resolve {list_path}:{line_number}: {entry!r}"
                ) from exc
    # Preserve the user's first occurrence while making repeated flags/lists
    # harmless. Duplicate episode content under different paths is still
    # rejected by ``load_feature_cache_collection``.
    unique = tuple(dict.fromkeys(paths))
    if not unique:
        raise ValueError(
            "at least one --feature-cache or non-empty --feature-list is required"
        )
    return unique


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _load_stable_event_bank(
    directory: Path,
    *,
    expected_manifest_hash: str | None = None,
) -> tuple[EventBank, str]:
    """Load a bank only while its manifest remains one exact snapshot."""

    manifest_path = directory / EVENT_BANK_MANIFEST_FILENAME
    before_hash = sha256_file(manifest_path)
    if expected_manifest_hash is not None and before_hash != expected_manifest_hash:
        raise CandidateCacheBuildError(
            "event-bank manifest changed while the candidate cache was being built"
        )
    bank = EventBank.load(directory)
    after_hash = sha256_file(manifest_path)
    if before_hash != after_hash:
        raise CandidateCacheBuildError(
            "event-bank manifest changed while the event bank was being loaded"
        )
    if expected_manifest_hash is not None and after_hash != expected_manifest_hash:
        raise CandidateCacheBuildError(
            "event-bank manifest changed while the candidate cache was being built"
        )
    return bank, after_hash


def _write_json_atomic(
    path: Path,
    value: dict[str, Any],
    *,
    overwrite: bool,
) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"summary already exists at {path}")
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    encoded = (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    try:
        with temporary.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    bank_directory = args.bank.expanduser().resolve()
    output_directory = args.output.expanduser().resolve()
    summary_path = None if args.summary is None else args.summary.expanduser().resolve()
    if _is_within(output_directory, bank_directory):
        raise CandidateCacheBuildError(
            "--output must be outside the immutable event-bank directory"
        )
    if summary_path is not None and (
        _is_within(summary_path, bank_directory)
        or _is_within(summary_path, output_directory)
    ):
        raise CandidateCacheBuildError(
            "--summary must be outside both the event-bank and candidate-cache directories"
        )
    if summary_path is not None and summary_path.exists() and not args.overwrite:
        raise FileExistsError(f"summary already exists at {summary_path}")

    try:
        feature_paths = _feature_paths(args.feature_cache, args.feature_list)
    except ValueError as exc:
        parser.error(str(exc))
    protected_inputs = {
        path
        for payload in feature_paths
        for path in (payload, payload.with_suffix(".manifest.json"))
    }
    protected_inputs.update(
        path.expanduser().resolve() for path in args.feature_list
    )
    protected_inputs.update(
        {
            args.catalog.expanduser().resolve(),
            args.audit_report.expanduser().resolve(),
        }
    )
    if summary_path is not None and summary_path in protected_inputs:
        raise CandidateCacheBuildError(
            "--summary must not overwrite a feature-cache or feature-list input"
        )

    bank, initial_bank_manifest_hash = _load_stable_event_bank(bank_directory)
    collection = load_feature_cache_collection(feature_paths)
    catalog = EpisodeCatalog.load(args.catalog.expanduser().resolve())
    audit = load_audit_report(args.audit_report.expanduser().resolve())
    query_binding = validate_feature_collection_against_catalog(
        collection,
        catalog,
        audit,
        expected_split="dev",
    )
    train_binding = FeatureDataBinding(
        catalog_sha256=query_binding.catalog_sha256,
        audit_report_sha256=query_binding.audit_report_sha256,
        split="train",
    )
    validate_event_bank_data_binding(bank, train_binding)
    bank_summary = validate_query_collection_against_bank(bank, collection)
    action_horizon = (
        bank_summary.action_horizon
        if args.action_horizon is None
        else args.action_horizon
    )

    cache = build_candidate_cache_from_collection(
        bank,
        collection,
        action_horizon=action_horizon,
        query_stride=args.query_stride,
        top_k=args.top_k,
    )
    manifest_path = bank_directory / EVENT_BANK_MANIFEST_FILENAME
    if sha256_file(manifest_path) != initial_bank_manifest_hash:
        raise CandidateCacheBuildError(
            "event-bank manifest changed while the candidate cache was being generated"
        )

    # Reload from disk after candidate generation.  This independently checks
    # the payload hashes and ensures the cache is bound to the same snapshot
    # that was used for retrieval rather than a later in-place overwrite.
    final_bank, event_bank_manifest_hash = _load_stable_event_bank(
        bank_directory,
        expected_manifest_hash=initial_bank_manifest_hash,
    )
    validate_query_collection_against_bank(final_bank, collection)
    validate_event_bank_data_binding(final_bank, train_binding)
    cache.validate_against_event_bank(final_bank)
    if final_bank.manifest is None:  # Defensive; EventBank.load always sets it.
        raise RuntimeError("loaded event bank has no manifest")
    event_bank_content_hash = canonical_event_bank_content_hash(
        final_bank.manifest.content_hashes
    )
    query_key_encoder = dict(final_bank.manifest.encoder)
    build_recipe: dict[str, Any] = {
        "implementation": "exact_cosine_v1",
        "action_horizon": action_horizon,
        "query_stride": args.query_stride,
        "top_k": args.top_k,
        "query_data_binding": query_binding.to_dict(),
        "episode_exclusion": [
            "global_episode_identity",
            "source_episode_sha256",
            "feature_episode_sha256",
        ],
    }
    publication_targets = [
        (output_directory, "publish WARM candidate cache"),
    ]
    if summary_path is not None:
        publication_targets.append((summary_path, "publish WARM candidate summary"))

    with _claim_publication_targets(publication_targets):
        # These checks must run after every publication lock has been acquired.
        # An earlier fail-fast check cannot exclude a competing process that
        # creates the artifact while candidate generation is in progress.
        output_manifest_path = output_directory / CANDIDATE_CACHE_MANIFEST_FILENAME
        if output_manifest_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"candidate cache already exists in {output_directory}"
            )
        if summary_path is not None and summary_path.exists() and not args.overwrite:
            raise FileExistsError(f"summary already exists at {summary_path}")

        cache.save(
            output_directory,
            event_bank_manifest_hash=event_bank_manifest_hash,
            event_bank_content_hash=event_bank_content_hash,
            query_corpus_hash=collection.content_hash,
            query_key_encoder=query_key_encoder,
            build_recipe=build_recipe,
            overwrite=args.overwrite,
        )

        if sha256_file(manifest_path) != event_bank_manifest_hash:
            raise CandidateCacheBuildError(
                "event-bank manifest changed while the candidate cache was being saved"
            )
        confirmed_bank, confirmed_manifest_hash = _load_stable_event_bank(
            bank_directory,
            expected_manifest_hash=event_bank_manifest_hash,
        )
        if confirmed_manifest_hash != event_bank_manifest_hash:
            raise CandidateCacheBuildError("event-bank manifest snapshot is inconsistent")
        if confirmed_bank.manifest is None:  # Defensive; EventBank.load always sets it.
            raise RuntimeError("confirmed event bank has no manifest")
        confirmed_content_hash = canonical_event_bank_content_hash(
            confirmed_bank.manifest.content_hashes
        )
        if confirmed_content_hash != event_bank_content_hash:
            raise CandidateCacheBuildError("event-bank content snapshot is inconsistent")
        validate_query_collection_against_bank(confirmed_bank, collection)
        validate_event_bank_data_binding(confirmed_bank, train_binding)

        restored = CandidateCache.load(
            output_directory,
            expected_event_bank_manifest_hash=event_bank_manifest_hash,
            expected_event_bank_content_hash=event_bank_content_hash,
            expected_query_corpus_hash=collection.content_hash,
            expected_query_key_encoder=query_key_encoder,
            expected_build_recipe=build_recipe,
        )
        restored.validate_against_event_bank(confirmed_bank)
        if restored.manifest is None:  # Defensive; CandidateCache.load always sets it.
            raise RuntimeError("loaded candidate cache has no manifest")

        build_summary: dict[str, Any] = {
            "schema": SUMMARY_SCHEMA,
            "version": SUMMARY_SCHEMA_VERSION,
            "bank": str(bank_directory),
            "output": str(output_directory),
            "feature_cache_count": len(collection.records),
            "event_count": bank_summary.num_events,
            "query_count": len(restored),
            "candidate_count": restored.num_candidates,
            "empty_query_count": sum(not row for row in restored.candidates),
            "action_horizon": action_horizon,
            "query_stride": args.query_stride,
            "top_k": args.top_k,
            "event_bank_manifest_hash": event_bank_manifest_hash,
            "event_bank_content_hash": event_bank_content_hash,
            "query_corpus_hash": collection.content_hash,
            "query_data_binding": query_binding.to_dict(),
            "build_recipe": build_recipe,
            "candidate_payload_file": restored.manifest.payload_file,
            "candidate_payload_hash": restored.manifest.content_hashes[
                restored.manifest.payload_file
            ],
        }
        if summary_path is not None:
            _write_json_atomic(summary_path, build_summary, overwrite=args.overwrite)
    print(
        json.dumps(
            build_summary,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
