#!/usr/bin/env python3
"""Evaluate the leakage-safe WARM M1 retrieval oracle from cached features."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence
from uuid import uuid4

from fastwam.datasets.lerobot.audit import load_audit_report
from fastwam.datasets.lerobot.episode_catalog import EpisodeCatalog
from fastwam.memory.action_contract import (
    ActionSpaceContractError,
    validate_action_space_contract,
)
from fastwam.memory.candidate_cache import canonical_event_bank_content_hash
from fastwam.memory.event_bank import MANIFEST_FILENAME, EventBank
from fastwam.memory.manifest import sha256_file
from fastwam.memory.offline_pipeline import (
    FeatureDataBinding,
    build_oracle_queries_from_collection,
    load_feature_cache_collection,
    validate_event_bank_data_binding,
    validate_feature_collection_against_catalog,
    validate_query_collection_against_bank,
)
from fastwam.memory.oracle_metrics import (
    ActionDistanceConfig,
    evaluate_oracle_retrieval,
    oracle_metrics_report_to_dict,
)
from fastwam.utils.artifact_claim import artifact_claim


_OUTPUT_SCHEMA = "warm.oracle-evaluation"
_OUTPUT_VERSION = 1


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _parse_top_k(values: Sequence[str]) -> tuple[int, ...]:
    output: list[int] = []
    seen: set[int] = set()
    for value in values:
        parts = value.split(",")
        if not parts or any(not part.strip() for part in parts):
            raise ValueError("--top-k values must be positive comma-separated integers")
        for part in parts:
            try:
                top_k = int(part.strip())
            except ValueError as exc:
                raise ValueError(
                    "--top-k values must be positive comma-separated integers"
                ) from exc
            if top_k <= 0:
                raise ValueError("--top-k values must be positive")
            if top_k not in seen:
                output.append(top_k)
                seen.add(top_k)
    if not output:
        raise ValueError("at least one --top-k value is required")
    return tuple(output)


def _parse_dimensions(value: str, option: str) -> tuple[int, ...]:
    parts = value.split(",")
    if not parts or any(not part.strip() for part in parts):
        raise ValueError(f"{option} must be a non-empty comma-separated integer list")
    dimensions: list[int] = []
    for part in parts:
        try:
            dimension = int(part.strip())
        except ValueError as exc:
            raise ValueError(f"{option} must contain only integers") from exc
        if dimension < 0:
            raise ValueError(f"{option} dimensions must be non-negative")
        dimensions.append(dimension)
    if len(set(dimensions)) != len(dimensions):
        raise ValueError(f"{option} must not contain duplicate dimensions")
    return tuple(dimensions)


def _read_feature_list(path: Path) -> list[Path]:
    path = path.expanduser().resolve()
    rows: list[Path] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        candidate = Path(line).expanduser()
        if not candidate.is_absolute():
            candidate = path.parent / candidate
        if candidate.suffix != ".npz":
            raise ValueError(
                f"{path}:{line_number}: feature-cache path must use the .npz suffix"
            )
        rows.append(candidate.resolve())
    return rows


def _feature_paths(
    direct: Sequence[Path] | None,
    lists: Sequence[Path] | None,
) -> tuple[Path, ...]:
    paths = [path.expanduser().resolve() for path in (direct or ())]
    for list_path in lists or ():
        paths.extend(_read_feature_list(list_path))
    if not paths:
        raise ValueError("provide at least one --feature-cache or non-empty --feature-list")
    # The collection loader performs canonical sorting; retain first occurrence
    # here to make CLI provenance/counts insensitive to repeated arguments.
    return tuple(dict.fromkeys(paths))


def _config_to_dict(config: ActionDistanceConfig) -> dict[str, object]:
    return {
        "action_dim": config.action_dim,
        "arm_dims": list(config.arm_dims),
        "gripper_dims": list(config.gripper_dims),
        "arm_loss": config.arm_loss,
        "huber_delta": config.huber_delta,
        "arm_weight": config.arm_weight,
        "gripper_state_weight": config.gripper_state_weight,
        "gripper_timing_weight": config.gripper_timing_weight,
        "gripper_threshold": config.gripper_threshold,
    }


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"oracle output already exists at {path}")
    encoded = json.dumps(
        payload,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate context top-1 and GT-action oracle top-K using a saved "
            "WARM bank and leakage-checked episode feature caches."
        )
    )
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--audit-report", required=True, type=Path)
    parser.add_argument(
        "--feature-cache",
        action="append",
        type=Path,
        default=None,
        help="Episode feature-cache .npz; repeat for multiple episodes.",
    )
    parser.add_argument(
        "--feature-list",
        action="append",
        type=Path,
        default=None,
        help="UTF-8 newline list of feature-cache paths; relative paths use the list directory.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--query-stride", required=True, type=_positive_int)
    parser.add_argument(
        "--top-k",
        required=True,
        action="append",
        help="Positive K; repeat the option or use comma-separated values.",
    )
    parser.add_argument(
        "--arm-dims",
        help="Optional consistency assertion; canonical dimensions come from the bank.",
    )
    parser.add_argument(
        "--gripper-dims",
        help="Optional consistency assertion; canonical dimensions come from the bank.",
    )
    parser.add_argument("--arm-loss", choices=("mse", "huber"), default="mse")
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--arm-weight", type=float, default=1.0)
    parser.add_argument("--gripper-state-weight", type=float, default=1.0)
    parser.add_argument("--gripper-timing-weight", type=float, default=1.0)
    parser.add_argument(
        "--gripper-threshold",
        type=float,
        default=None,
        help="Optional consistency assertion; canonical threshold comes from the bank.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        top_k_values = _parse_top_k(args.top_k)
        feature_paths = _feature_paths(args.feature_cache, args.feature_list)
        bank_root = args.bank.expanduser().resolve()
        output_path = args.output.expanduser().resolve()
        if output_path == bank_root or bank_root in output_path.parents:
            raise ValueError("--output must be outside the immutable event-bank directory")
        protected_feature_paths = {
            path
            for payload in feature_paths
            for path in (payload, payload.with_suffix(".manifest.json"))
        }
        protected_feature_paths.update(
            {
                args.catalog.expanduser().resolve(),
                args.audit_report.expanduser().resolve(),
            }
        )
        if output_path in protected_feature_paths:
            raise ValueError("--output must not overwrite a feature-cache artifact")

        manifest_path = bank_root / MANIFEST_FILENAME
        manifest_hash_before = sha256_file(manifest_path)
        bank = EventBank.load(bank_root)
        if bank.manifest is None:
            raise ValueError("loaded event bank is missing its manifest")
        bank_content_hash = canonical_event_bank_content_hash(
            bank.manifest.content_hashes
        )
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
        summary = validate_query_collection_against_bank(bank, collection)

        try:
            action_space = validate_action_space_contract(
                bank.manifest.action_normalizer["contract"]
            )
        except (ActionSpaceContractError, KeyError, TypeError) as exc:
            raise ValueError(
                "event bank is missing a valid versioned action-space contract"
            ) from exc
        if action_space.action_dim != summary.action_dim:
            raise ValueError(
                "event-bank action payload disagrees with its action-space contract: "
                f"payload={summary.action_dim}, contract={action_space.action_dim}"
            )
        if args.arm_dims is not None:
            asserted_arm_dims = _parse_dimensions(args.arm_dims, "--arm-dims")
            if asserted_arm_dims != action_space.arm_dims:
                raise ValueError(
                    "--arm-dims does not match the immutable action-space contract"
                )
        if args.gripper_dims is not None:
            asserted_gripper_dims = _parse_dimensions(
                args.gripper_dims, "--gripper-dims"
            )
            if asserted_gripper_dims != action_space.gripper_dims:
                raise ValueError(
                    "--gripper-dims does not match the immutable action-space contract"
                )
        if (
            args.gripper_threshold is not None
            and args.gripper_threshold != action_space.gripper_threshold
        ):
            raise ValueError(
                "--gripper-threshold does not match the immutable action-space contract"
            )
        config = ActionDistanceConfig(
            action_dim=summary.action_dim,
            arm_dims=action_space.arm_dims,
            gripper_dims=action_space.gripper_dims,
            arm_loss=args.arm_loss,
            huber_delta=args.huber_delta,
            arm_weight=args.arm_weight,
            gripper_state_weight=args.gripper_state_weight,
            gripper_timing_weight=args.gripper_timing_weight,
            gripper_threshold=action_space.gripper_threshold,
        )
        queries = build_oracle_queries_from_collection(
            collection,
            action_horizon=summary.action_horizon,
            query_stride=args.query_stride,
        )
        reports = {
            str(top_k): oracle_metrics_report_to_dict(
                evaluate_oracle_retrieval(bank, queries, config, top_k=top_k)
            )
            for top_k in top_k_values
        }

        manifest_hash_after = sha256_file(manifest_path)
        if manifest_hash_after != manifest_hash_before:
            raise ValueError("event-bank manifest changed during oracle evaluation")
        # Reopen every payload through EventBank.load before publishing the
        # report.  A stable manifest filename alone is insufficient because a
        # concurrent process could have replaced payloads while evaluation was
        # running.  The final report is therefore bound to a reproducible,
        # fully revalidated bank snapshot.
        confirmed_bank = EventBank.load(bank_root)
        if confirmed_bank.manifest is None:
            raise ValueError("reloaded event bank is missing its manifest")
        if sha256_file(manifest_path) != manifest_hash_before:
            raise ValueError("event-bank manifest changed before oracle publication")
        confirmed_content_hash = canonical_event_bank_content_hash(
            confirmed_bank.manifest.content_hashes
        )
        if confirmed_content_hash != bank_content_hash:
            raise ValueError("event-bank content changed during oracle evaluation")
        validate_event_bank_data_binding(confirmed_bank, train_binding)
        confirmed_summary = validate_query_collection_against_bank(
            confirmed_bank, collection
        )
        if confirmed_summary != summary:
            raise ValueError("event-bank contract changed during oracle evaluation")
        output: dict[str, object] = {
            "schema": _OUTPUT_SCHEMA,
            "version": _OUTPUT_VERSION,
            "bank_path": str(bank_root),
            "bank_manifest_sha256": manifest_hash_before,
            "bank_content_sha256": bank_content_hash,
            "query_corpus_sha256": collection.content_hash,
            "query_split": collection.split,
            "catalog_sha256": query_binding.catalog_sha256,
            "audit_report_sha256": query_binding.audit_report_sha256,
            "feature_cache_count": len(collection.records),
            "query_stride": args.query_stride,
            "query_count": len(queries),
            "top_k_values": list(top_k_values),
            "bank_summary": {
                "num_events": summary.num_events,
                "num_source_episodes": summary.num_source_episodes,
                "action_horizon": summary.action_horizon,
                "action_dim": summary.action_dim,
                "context_dim": summary.context_dim,
                "effect_shape": list(summary.effect_shape),
                "proprio_dim": summary.proprio_dim,
            },
            "action_distance_config": _config_to_dict(config),
            "action_space_contract": action_space.to_dict(),
            "reports": reports,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        claim_path = output_path.parent / f".{output_path.name}.warm-oracle.lock"
        with artifact_claim(claim_path, purpose="publish WARM oracle report"):
            # Recheck under the cross-process claim.  Without this second
            # check, two writers can both pass the earlier exists() guard and
            # silently produce last-writer-wins output via os.replace().
            if output_path.exists():
                raise FileExistsError(
                    f"oracle output already exists at {output_path}"
                )
            if sha256_file(manifest_path) != manifest_hash_before:
                raise ValueError(
                    "event-bank manifest changed before oracle report publication"
                )
            _atomic_write_json(output_path, output)
    except (OSError, TypeError, ValueError, KeyError) as exc:
        parser.error(str(exc))

    print(f"output={args.output.expanduser().resolve()}")
    print(f"bank_manifest_sha256={output['bank_manifest_sha256']}")
    print(f"query_corpus_sha256={output['query_corpus_sha256']}")
    print(f"queries={len(queries)} top_k={','.join(str(value) for value in top_k_values)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
