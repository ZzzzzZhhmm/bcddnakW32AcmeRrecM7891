#!/usr/bin/env python3
"""Build one leakage-audited WARM event bank from episode feature caches."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence
from uuid import uuid4

from fastwam.datasets.lerobot.audit import load_audit_report
from fastwam.datasets.lerobot.episode_catalog import EpisodeCatalog
from fastwam.memory.action_contract import (
    ActionSpaceContractError,
    validate_action_space_contract,
)
from fastwam.memory.bank_contract import validate_warm_v1_bank
from fastwam.memory.event_bank import MANIFEST_FILENAME, EventBank
from fastwam.memory.event_mining import EventMiningConfig
from fastwam.memory.manifest import sha256_file
from fastwam.memory.offline_pipeline import (
    FeatureCacheCollection,
    build_event_bank_from_collection,
    load_feature_cache_collection,
    validate_feature_collection_against_catalog,
)
from fastwam.utils.artifact_claim import artifact_claim


SUMMARY_SCHEMA = "warm.event-bank-build-summary"
SUMMARY_SCHEMA_VERSION = 1


class BuildWarmEventBankError(ValueError):
    """Raised when CLI inputs do not define one reproducible bank build."""


def _sibling_claim_path(target: Path) -> Path:
    """Return the dedicated cross-process claim beside one publish target."""

    return target.parent / f".{target.name}.warm-build.lock"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an immutable WARM event bank from verified feature caches."
    )
    parser.add_argument(
        "--feature-cache",
        action="append",
        default=[],
        type=Path,
        help="Episode feature-cache .npz path; may be repeated.",
    )
    parser.add_argument(
        "--feature-list",
        action="append",
        default=[],
        type=Path,
        help="UTF-8 file containing one feature-cache path per line; may be repeated.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--audit-report", required=True, type=Path)
    parser.add_argument(
        "--start-mode",
        choices=("uniform", "event", "hybrid"),
        default="hybrid",
    )
    parser.add_argument("--action-horizon", required=True, type=int)
    parser.add_argument("--score-quantile", type=float, default=0.90)
    parser.add_argument("--local-max-radius", type=int, default=2)
    parser.add_argument("--nms-radius", type=int, default=None)
    parser.add_argument("--uniform-stride", type=int, default=None)
    parser.add_argument("--gripper-change-threshold", type=float, default=1e-6)
    parser.add_argument("--mad-epsilon", type=float, default=1e-6)
    parser.add_argument("--robust-clip", type=float, default=10.0)

    parser.add_argument("--normalizer-contract", required=True, type=Path)
    parser.add_argument("--encoder-contract", required=True, type=Path)
    parser.add_argument("--camera-contract", required=True, type=Path)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Permit artifact creation from a dirty Git tree (tests/debug only).",
    )
    return parser


def _read_feature_list(path: Path) -> list[Path]:
    list_path = path.resolve()
    try:
        lines = list_path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        raise BuildWarmEventBankError(f"cannot read feature list {list_path}") from exc

    result: list[Path] = []
    for line_number, line in enumerate(lines, start=1):
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = list_path.parent / candidate
        if candidate.suffix != ".npz":
            raise BuildWarmEventBankError(
                f"feature list {list_path}:{line_number} does not name a .npz payload"
            )
        result.append(candidate)
    return result


def _collect_feature_paths(
    direct: Sequence[Path],
    list_files: Sequence[Path],
) -> tuple[Path, ...]:
    paths = list(direct)
    for list_file in list_files:
        paths.extend(_read_feature_list(list_file))
    if not paths:
        raise BuildWarmEventBankError(
            "at least one --feature-cache or non-empty --feature-list is required"
        )
    for path in paths:
        if path.suffix != ".npz":
            raise BuildWarmEventBankError(
                f"feature-cache payload must use the .npz suffix: {path}"
            )
    return tuple(paths)


def _reject_json_constant(value: str) -> None:
    raise BuildWarmEventBankError(f"contract JSON contains non-finite value {value}")


def _strict_json_object(raw: bytes, *, contract_path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_json_constant,
        )
    except BuildWarmEventBankError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise BuildWarmEventBankError(
            f"cannot read JSON contract {contract_path}"
        ) from exc
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise BuildWarmEventBankError(
            f"contract {contract_path} must be a JSON object with string keys"
        )
    # Re-encode now so nested non-JSON values can never reach the manifest.
    try:
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise BuildWarmEventBankError(
            f"contract {contract_path} must contain finite JSON values"
        ) from exc
    return value


def _load_contract_mapping(
    label: str,
    path: Path,
    expected_hash: str,
) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        raw = resolved.read_bytes()
    except OSError as exc:
        raise BuildWarmEventBankError(
            f"cannot read JSON contract {resolved}"
        ) from exc
    # Hash and parse exactly the same immutable byte snapshot.  Reopening the
    # path here would permit a contract replacement between verification and
    # decoding, leaving provenance bound to different content.
    actual_hash = hashlib.sha256(raw).hexdigest()
    if actual_hash != expected_hash:
        raise BuildWarmEventBankError(
            f"{label} contract SHA-256 mismatch: expected {expected_hash}, "
            f"got {actual_hash} for {resolved}"
        )
    return {
        "file_sha256": actual_hash,
        "contract": _strict_json_object(raw, contract_path=resolved),
    }


def _contract_mappings(
    collection: FeatureCacheCollection,
    *,
    normalizer_path: Path,
    encoder_path: Path,
    camera_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    normalizer = _load_contract_mapping(
        "normalizer", normalizer_path, collection.contract.normalizer_hash
    )
    encoder = _load_contract_mapping(
        "encoder", encoder_path, collection.contract.encoder_hash
    )
    camera = _load_contract_mapping(
        "camera", camera_path, collection.contract.camera_hash
    )
    return normalizer, encoder, camera


def _atomic_write_json(
    path: Path,
    value: Mapping[str, Any],
) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"summary already exists at {path}")
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    try:
        encoded = json.dumps(
            dict(value),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8") + b"\n"
        with temporary.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _summary_payload(
    *,
    bank_manifest_hash: str,
    collection: FeatureCacheCollection,
    summary: Any,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": SUMMARY_SCHEMA,
        "version": SUMMARY_SCHEMA_VERSION,
        "bank_manifest_sha256": bank_manifest_hash,
        "feature_collection_sha256": collection.content_hash,
        "split": provenance["split"],
        "data_binding": provenance["data_binding"],
        "feature_contract": collection.contract.to_dict(),
        "events": {
            "count": summary.num_events,
            "source_episode_count": summary.num_source_episodes,
        },
        "action": {
            "horizon": summary.action_horizon,
            "dimension": summary.action_dim,
        },
        "context": {"dimension": summary.context_dim},
        "effect": {"shape": list(summary.effect_shape)},
        "proprio": {"dimension": summary.proprio_dim},
        "recipe": provenance["build_recipe"],
        "software": provenance["software"],
    }


def _software_provenance() -> dict[str, object]:
    repository = Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "0" * 40
    return {
        "git_commit": commit,
        "git_dirty": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.resolve()
    summary_path = args.summary.resolve()
    try:
        summary_path.relative_to(output)
    except ValueError:
        pass
    else:
        raise BuildWarmEventBankError(
            "--summary must be outside the immutable event-bank directory"
        )
    if output.exists():
        raise FileExistsError(f"event-bank output already exists at {output}")
    if summary_path.exists():
        raise FileExistsError(f"summary already exists at {summary_path}")

    feature_paths = _collect_feature_paths(args.feature_cache, args.feature_list)
    collection = load_feature_cache_collection(feature_paths)
    catalog = EpisodeCatalog.load(args.catalog.resolve())
    audit = load_audit_report(args.audit_report.resolve())
    data_binding = validate_feature_collection_against_catalog(
        collection,
        catalog,
        audit,
        expected_split="train",
    )
    action_normalizer, encoder, camera_layout = _contract_mappings(
        collection,
        normalizer_path=args.normalizer_contract,
        encoder_path=args.encoder_contract,
        camera_path=args.camera_contract,
    )
    try:
        action_space = validate_action_space_contract(
            action_normalizer["contract"]
        )
    except (ActionSpaceContractError, KeyError, TypeError) as exc:
        raise BuildWarmEventBankError(
            "normalizer contract must contain a valid versioned WARM "
            "action-space contract"
        ) from exc
    mining_config = EventMiningConfig(
        action_horizon=args.action_horizon,
        score_quantile=args.score_quantile,
        local_max_radius=args.local_max_radius,
        nms_radius=args.nms_radius,
        uniform_stride=args.uniform_stride,
        gripper_change_threshold=args.gripper_change_threshold,
        mad_epsilon=args.mad_epsilon,
        robust_clip=args.robust_clip,
    )
    bank, bank_summary, provenance = build_event_bank_from_collection(
        collection,
        mining_config=mining_config,
        start_mode=args.start_mode,
        data_binding=data_binding,
    )
    if bank_summary.action_dim != action_space.action_dim:
        raise BuildWarmEventBankError(
            "normalizer action_dim does not match cached model actions: "
            f"contract={action_space.action_dim}, bank={bank_summary.action_dim}"
        )
    software = _software_provenance()
    provenance["software"] = software
    # Each immutable publish target has its own sibling claim.  Acquire both
    # in a stable path order so independent writers cannot deadlock while
    # racing to publish the same bank/summary pair.
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    claim_specs = sorted(
        (
            (
                _sibling_claim_path(output),
                "publish WARM event-bank directory",
            ),
            (
                _sibling_claim_path(summary_path),
                "publish WARM event-bank build summary",
            ),
        ),
        key=lambda item: os.path.normcase(os.fspath(item[0])),
    )
    with ExitStack() as claims:
        for claim_path, purpose in claim_specs:
            claims.enter_context(artifact_claim(claim_path, purpose=purpose))

        # Recheck both targets only after this process owns both claims.  The
        # early exists() checks above are not sufficient across processes.
        if output.exists():
            raise FileExistsError(f"event-bank output already exists at {output}")
        if summary_path.exists():
            raise FileExistsError(f"summary already exists at {summary_path}")

        bank.save(
            output,
            action_normalizer=action_normalizer,
            encoder=encoder,
            camera_layout=camera_layout,
            provenance=provenance,
        )
        restored = EventBank.load(
            output,
            expected_action_normalizer=action_normalizer,
            expected_encoder=encoder,
            expected_camera_layout=camera_layout,
            expected_provenance=provenance,
        )
        restored_summary = validate_warm_v1_bank(
            restored,
            expected_action_horizon=mining_config.action_horizon,
        )
        if restored_summary != bank_summary:
            raise BuildWarmEventBankError(
                "reloaded event bank does not match the in-memory build summary"
            )
        if restored.manifest is None:
            raise BuildWarmEventBankError("reloaded event bank is missing its manifest")
        manifest = restored.manifest
        manifest_path = output / MANIFEST_FILENAME
        manifest_hash = sha256_file(manifest_path)
        _atomic_write_json(
            summary_path,
            _summary_payload(
                bank_manifest_hash=manifest_hash,
                collection=collection,
                summary=bank_summary,
                provenance=provenance,
            ),
        )

    print(f"bank={output}")
    print(f"manifest_sha256={manifest_hash}")
    print(f"feature_collection_sha256={collection.content_hash}")
    print(f"events={manifest.num_events}")
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
