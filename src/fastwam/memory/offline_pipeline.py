"""Leakage-safe composition of WARM's M1 offline artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from fastwam.datasets.lerobot.audit import LerobotAuditReport
from fastwam.datasets.lerobot.episode_catalog import EpisodeCatalog

from .bank_builder import EpisodeFeatures, build_event_bank
from .bank_contract import WarmBankSummary, validate_warm_v1_bank
from .candidate_cache import CachedCandidate, CandidateCache, QueryId
from .event_bank import EventBank
from .event_mining import EventMiningConfig, StartMode
from .feature_cache import LoadedFeatureCache, load_episode_feature_cache
from .manifest import sha256_canonical_json
from .oracle_metrics import OracleQuery
from .payload_names import FEATURE_EPISODE_SHA256, SOURCE_EPISODE_SHA256


class OfflinePipelineError(ValueError):
    """Raised when verified offline artifacts cannot be composed safely."""


@dataclass(frozen=True, slots=True)
class FeatureCollectionContract:
    catalog_hash: str
    normalizer_hash: str
    encoder_hash: str
    camera_hash: str

    def to_dict(self) -> dict[str, str]:
        return {
            "catalog_hash": self.catalog_hash,
            "normalizer_hash": self.normalizer_hash,
            "encoder_hash": self.encoder_hash,
            "camera_hash": self.camera_hash,
        }


@dataclass(frozen=True, slots=True)
class FeatureDataBinding:
    """Verified catalog/audit boundary for one homogeneous feature split."""

    catalog_sha256: str
    audit_report_sha256: str
    split: str

    def __post_init__(self) -> None:
        for field in ("catalog_sha256", "audit_report_sha256"):
            value = getattr(self, field)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise OfflinePipelineError(f"{field} must be a lowercase SHA-256 digest")
        if self.split not in {"train", "dev", "test"}:
            raise OfflinePipelineError("binding split must be train, dev, or test")

    def to_dict(self) -> dict[str, str]:
        return {
            "catalog_sha256": self.catalog_sha256,
            "audit_report_sha256": self.audit_report_sha256,
            "split": self.split,
        }


def _collection_hash(records: Sequence[LoadedFeatureCache]) -> str:
    rows = [
        {
            "metadata": record.metadata.to_dict(),
            "episode_content_hash": record.episode_content_hash,
            "payload_hash": record.manifest.payload_hash,
            # Hash the already verified in-memory manifest snapshot. Re-reading
            # the sidecar here would introduce a TOCTOU window after cache load.
            "manifest_hash": sha256_canonical_json(record.manifest.to_dict()),
        }
        for record in records
    ]
    encoded = json.dumps(
        rows,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(b"warm.feature-collection.v1\0" + encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class FeatureCacheCollection:
    records: tuple[LoadedFeatureCache, ...]
    contract: FeatureCollectionContract
    content_hash: str
    split: str

    def __post_init__(self) -> None:
        if not self.records:
            raise OfflinePipelineError("feature-cache collection must not be empty")
        if self.split not in {"train", "dev", "test"}:
            raise OfflinePipelineError("collection split must be train, dev, or test")
        if any(record.metadata.split != self.split for record in self.records):
            raise OfflinePipelineError(
                "feature-cache collection records do not match its declared split"
            )

    @property
    def episodes(self) -> tuple[EpisodeFeatures, ...]:
        return tuple(record.features for record in self.records)

    @property
    def episode_keys(self) -> frozenset[tuple[str, int, int]]:
        return frozenset(
            (
                record.metadata.dataset_id,
                record.metadata.dataset_index,
                record.metadata.episode_index,
            )
            for record in self.records
        )

    @property
    def source_episode_hashes(self) -> frozenset[str]:
        return frozenset(record.metadata.source_episode_sha256 for record in self.records)

    @property
    def episode_content_hashes(self) -> frozenset[str]:
        return frozenset(record.episode_content_hash for record in self.records)

    def provenance(self) -> dict[str, object]:
        return {
            "split": self.split,
            "feature_collection_sha256": self.content_hash,
            "episode_count": len(self.records),
            "feature_contract": self.contract.to_dict(),
        }


def load_feature_cache_collection(
    payload_paths: Iterable[str | Path],
    *,
    reject_duplicate_source_content: bool = True,
) -> FeatureCacheCollection:
    """Load a deterministic collection and enforce one shared model contract."""

    normalized_paths = tuple(sorted({Path(path).resolve() for path in payload_paths}))
    if not normalized_paths:
        raise OfflinePipelineError("feature-cache collection must not be empty")
    records = [load_episode_feature_cache(path) for path in normalized_paths]
    records.sort(
        key=lambda item: (
            item.metadata.dataset_id,
            item.metadata.dataset_index,
            item.metadata.episode_index,
        )
    )
    identities = [
        (
            record.metadata.dataset_id,
            record.metadata.dataset_index,
            record.metadata.episode_index,
        )
        for record in records
    ]
    if len(set(identities)) != len(identities):
        raise OfflinePipelineError("feature-cache collection contains duplicate episode ids")

    splits = {record.metadata.split for record in records}
    if len(splits) != 1:
        raise OfflinePipelineError(
            f"feature-cache collection mixes splits: {sorted(splits)!r}"
        )
    split = next(iter(splits))

    first = records[0].metadata
    contract = FeatureCollectionContract(
        catalog_hash=first.catalog_hash,
        normalizer_hash=first.normalizer_hash,
        encoder_hash=first.encoder_hash,
        camera_hash=first.camera_hash,
    )
    for record in records[1:]:
        actual = FeatureCollectionContract(
            catalog_hash=record.metadata.catalog_hash,
            normalizer_hash=record.metadata.normalizer_hash,
            encoder_hash=record.metadata.encoder_hash,
            camera_hash=record.metadata.camera_hash,
        )
        if actual != contract:
            raise OfflinePipelineError(
                f"feature cache {record.payload_path} has a different model/data contract"
            )

    if reject_duplicate_source_content:
        for label, values in (
            (
                "source episode",
                [record.metadata.source_episode_sha256 for record in records],
            ),
            ("encoded episode", [record.episode_content_hash for record in records]),
        ):
            if len(set(values)) != len(values):
                raise OfflinePipelineError(
                    f"feature-cache collection contains duplicate {label} content"
                )

    record_tuple = tuple(records)
    return FeatureCacheCollection(
        records=record_tuple,
        contract=contract,
        content_hash=_collection_hash(record_tuple),
        split=split,
    )


def assert_feature_collections_disjoint(
    first: FeatureCacheCollection,
    second: FeatureCacheCollection,
) -> None:
    """Reject identity, raw-source, or encoded-content overlap across splits."""

    if not isinstance(first, FeatureCacheCollection) or not isinstance(
        second, FeatureCacheCollection
    ):
        raise TypeError("both values must be FeatureCacheCollection")
    if first.contract != second.contract:
        raise OfflinePipelineError("feature collections use different contracts")
    checks = (
        ("episode identity", first.episode_keys & second.episode_keys),
        (
            "source episode content",
            first.source_episode_hashes & second.source_episode_hashes,
        ),
        (
            "encoded episode content",
            first.episode_content_hashes & second.episode_content_hashes,
        ),
    )
    for label, overlap in checks:
        if overlap:
            raise OfflinePipelineError(
                f"feature collections overlap by {label}: {sorted(overlap)!r}"
            )


def validate_feature_collection_against_catalog(
    collection: FeatureCacheCollection,
    catalog: EpisodeCatalog,
    audit: LerobotAuditReport,
    *,
    expected_split: str,
) -> FeatureDataBinding:
    """Prove feature identities, splits, and raw hashes against audited data.

    Cache metadata alone is not accepted as split authority. The immutable
    catalog supplies membership/split, while the hashed audit supplies the raw
    episode-table SHA-256 for each catalog member.
    """

    if not isinstance(collection, FeatureCacheCollection):
        raise TypeError("collection must be FeatureCacheCollection")
    if not isinstance(catalog, EpisodeCatalog):
        raise TypeError("catalog must be EpisodeCatalog")
    if not isinstance(audit, LerobotAuditReport):
        raise TypeError("audit must be LerobotAuditReport")
    if expected_split not in {"train", "dev", "test"}:
        raise OfflinePipelineError("expected_split must be train, dev, or test")
    if not audit.episode_tables_hashed:
        raise OfflinePipelineError(
            "production feature binding requires an audit with hashed episode tables"
        )
    if audit.cross_split_duplicate_count:
        raise OfflinePipelineError(
            "audited dataset contains raw episode content duplicated across splits"
        )
    if collection.contract.catalog_hash != catalog.content_sha256:
        raise OfflinePipelineError(
            "feature collection catalog hash does not match the supplied catalog"
        )
    if audit.catalog_sha256 != catalog.content_sha256:
        raise OfflinePipelineError(
            "audit report catalog hash does not match the supplied catalog"
        )
    if collection.split != expected_split:
        raise OfflinePipelineError(
            f"feature collection must use split {expected_split!r}, "
            f"got {collection.split!r}"
        )

    catalog_index = {
        (record.dataset_id, record.dataset_index, record.episode_index): record
        for record in catalog.episodes
    }
    proof_index = audit.proof_index
    if set(proof_index) != set(catalog_index):
        missing = sorted(set(catalog_index) - set(proof_index))
        extra = sorted(set(proof_index) - set(catalog_index))
        raise OfflinePipelineError(
            "audit episode proofs do not exactly cover the catalog; "
            f"missing={missing!r}, extra={extra!r}"
        )
    for key, catalog_record in catalog_index.items():
        proof = proof_index[key]
        if proof.split != catalog_record.split:
            raise OfflinePipelineError(
                f"audit split disagrees with catalog for episode {key!r}"
            )

    for loaded in collection.records:
        metadata = loaded.metadata
        key = (metadata.dataset_id, metadata.dataset_index, metadata.episode_index)
        catalog_record = catalog_index.get(key)
        if catalog_record is None:
            raise OfflinePipelineError(
                f"feature cache episode {key!r} is not a member of the supplied catalog"
            )
        proof = proof_index[key]
        if metadata.split != catalog_record.split or metadata.split != proof.split:
            raise OfflinePipelineError(
                f"feature cache split is not catalog-derived for episode {key!r}"
            )
        if metadata.source_episode_sha256 != proof.source_episode_sha256:
            raise OfflinePipelineError(
                f"feature cache source hash is not audit-derived for episode {key!r}"
            )
        factual_states = int(loaded.features.semantic_features.shape[0])
        actions = int(loaded.features.model_actions.shape[0])
        if factual_states != catalog_record.length or actions != catalog_record.length - 1:
            raise OfflinePipelineError(
                f"feature cache time axis disagrees with catalog length for {key!r}: "
                f"states={factual_states}, actions={actions}, "
                f"catalog_length={catalog_record.length}"
            )

    expected_episode_keys = {
        key
        for key, catalog_record in catalog_index.items()
        if catalog_record.split == expected_split
    }
    actual_episode_keys = set(collection.episode_keys)
    if actual_episode_keys != expected_episode_keys:
        missing = sorted(expected_episode_keys - actual_episode_keys)
        extra = sorted(actual_episode_keys - expected_episode_keys)
        raise OfflinePipelineError(
            f"feature collection must exactly cover catalog split "
            f"{expected_split!r}; missing={missing!r}, extra={extra!r}"
        )

    return FeatureDataBinding(
        catalog_sha256=catalog.content_sha256,
        audit_report_sha256=audit.report_sha256,
        split=expected_split,
    )


def validate_event_bank_data_binding(
    bank: EventBank,
    binding: FeatureDataBinding,
) -> None:
    """Require a saved bank to carry the exact train data proof."""

    if not isinstance(bank, EventBank):
        raise TypeError("bank must be EventBank")
    if not isinstance(binding, FeatureDataBinding):
        raise TypeError("binding must be FeatureDataBinding")
    if bank.manifest is None:
        raise OfflinePipelineError("event bank must have a persisted manifest")
    actual = bank.manifest.provenance.get("data_binding")
    if actual != binding.to_dict():
        raise OfflinePipelineError(
            "event-bank data binding does not match the supplied catalog/audit"
        )


def validate_query_collection_against_bank(
    bank: EventBank,
    queries: FeatureCacheCollection,
) -> WarmBankSummary:
    """Bind query features to a saved bank and reject content-level leakage."""

    summary = validate_warm_v1_bank(bank)
    if bank.manifest is None:
        raise OfflinePipelineError("event bank must be loaded from or saved to a manifest")
    provenance_contract = bank.manifest.provenance.get("feature_contract")
    if provenance_contract != queries.contract.to_dict():
        raise OfflinePipelineError(
            "query feature contract does not match the event-bank source contract"
        )
    if bank.manifest.provenance.get("split") != "train":
        raise OfflinePipelineError("query evaluation requires a train-only event bank")
    if queries.split != "dev":
        raise OfflinePipelineError(
            f"M1 query/oracle collection must use split 'dev', got {queries.split!r}"
        )
    for label, manifest_mapping, contract_field in (
        (
            "normalizer",
            bank.manifest.action_normalizer,
            queries.contract.normalizer_hash,
        ),
        ("encoder", bank.manifest.encoder, queries.contract.encoder_hash),
        ("camera", bank.manifest.camera_layout, queries.contract.camera_hash),
    ):
        if manifest_mapping.get("file_sha256") != contract_field:
            raise OfflinePipelineError(
                f"event-bank {label} mapping is not bound to its feature-contract hash"
            )
    if queries.records[0].features.context_keys.shape[1] != summary.context_dim:
        raise OfflinePipelineError(
            "query context-key dimension does not match the event bank"
        )
    bank_hash_rows = bank.payload(SOURCE_EPISODE_SHA256)
    bank_hashes = {
        bytes(memoryview(row).cast("B")).hex() for row in bank_hash_rows
    }
    overlap = bank_hashes & queries.source_episode_hashes
    if overlap:
        raise OfflinePipelineError(
            f"query corpus overlaps bank source episode content: {sorted(overlap)!r}"
        )
    feature_hash_rows = bank.payload(FEATURE_EPISODE_SHA256)
    if feature_hash_rows.dtype != np.dtype(np.uint8) or feature_hash_rows.shape != (
        len(bank),
        32,
    ):
        raise OfflinePipelineError(
            f"event-bank {FEATURE_EPISODE_SHA256} must have dtype uint8 and "
            f"shape {(len(bank), 32)}"
        )
    bank_feature_hashes = {
        bytes(memoryview(row).cast("B")).hex() for row in feature_hash_rows
    }
    feature_overlap = bank_feature_hashes & queries.episode_content_hashes
    if feature_overlap:
        raise OfflinePipelineError(
            "query corpus overlaps bank encoded episode content: "
            f"{sorted(feature_overlap)!r}"
        )
    return summary


def build_event_bank_from_collection(
    collection: FeatureCacheCollection,
    *,
    mining_config: EventMiningConfig,
    start_mode: StartMode,
    data_binding: FeatureDataBinding,
) -> tuple[EventBank, WarmBankSummary, dict[str, object]]:
    if not isinstance(collection, FeatureCacheCollection):
        raise TypeError("collection must be FeatureCacheCollection")
    if collection.split != "train":
        raise OfflinePipelineError(
            f"event-bank source collection must use split 'train', got {collection.split!r}"
        )
    if not isinstance(data_binding, FeatureDataBinding):
        raise TypeError("data_binding must be FeatureDataBinding")
    if data_binding.split != "train":
        raise OfflinePipelineError("event-bank data binding must use split 'train'")
    if data_binding.catalog_sha256 != collection.contract.catalog_hash:
        raise OfflinePipelineError(
            "event-bank data binding catalog does not match feature caches"
        )
    bank = build_event_bank(
        collection.episodes,
        mining_config=mining_config,
        start_mode=start_mode,
    )
    summary = validate_warm_v1_bank(
        bank,
        expected_action_horizon=mining_config.action_horizon,
    )
    provenance = {
        **collection.provenance(),
        "data_binding": data_binding.to_dict(),
        "build_recipe": {
            "start_mode": start_mode,
            "event_mining_config": asdict(mining_config),
            "uses_semantic_change": True,
            "uses_vae_change": any(
                episode.vae_features is not None for episode in collection.episodes
            ),
            "action_resampling": False,
            "padding": False,
        },
    }
    return bank, summary, provenance


def fixed_horizon_query_starts(
    num_actions: int,
    *,
    action_horizon: int,
    stride: int,
) -> tuple[int, ...]:
    if isinstance(num_actions, bool) or not isinstance(num_actions, int):
        raise TypeError("num_actions must be an integer")
    if action_horizon <= 0 or stride <= 0:
        raise OfflinePipelineError("action_horizon and stride must be positive")
    if num_actions < action_horizon:
        return ()
    final_start = num_actions - action_horizon
    starts = list(range(0, final_start + 1, stride))
    if starts[-1] != final_start:
        starts.append(final_start)
    return tuple(starts)


def build_candidate_cache_from_collection(
    bank: EventBank,
    queries: FeatureCacheCollection,
    *,
    action_horizon: int,
    query_stride: int,
    top_k: int,
) -> CandidateCache:
    """Run exact retrieval with identity and source-content episode exclusion."""

    validate_query_collection_against_bank(bank, queries)
    validate_warm_v1_bank(bank, expected_action_horizon=action_horizon)
    if top_k <= 0:
        raise OfflinePipelineError("top_k must be positive")
    query_ids: list[QueryId] = []
    candidate_rows: list[tuple[CachedCandidate, ...]] = []
    for record in queries.records:
        episode = record.features
        for start in fixed_horizon_query_starts(
            int(episode.model_actions.shape[0]),
            action_horizon=action_horizon,
            stride=query_stride,
        ):
            query_ids.append(
                QueryId(
                    episode.dataset_id,
                    episode.dataset_index,
                    episode.episode_index,
                    start,
                )
            )
            results = bank.search(
                episode.context_keys[start],
                top_k=top_k,
                exclude_episode=(
                    episode.dataset_id,
                    episode.dataset_index,
                    episode.episode_index,
                ),
                exclude_source_episode_sha256=episode.source_episode_sha256,
                exclude_feature_episode_sha256=episode.feature_episode_sha256,
            )
            candidate_rows.append(
                tuple(CachedCandidate(result.event_id, result.score) for result in results)
            )
    cache = CandidateCache(query_ids, candidate_rows)
    cache.validate_against_event_bank(bank)
    return cache


def build_oracle_queries_from_collection(
    collection: FeatureCacheCollection,
    *,
    action_horizon: int,
    query_stride: int,
) -> tuple[OracleQuery, ...]:
    """Build fixed-horizon GT queries with an optional previous-chunk baseline."""

    output: list[OracleQuery] = []
    for record in collection.records:
        episode = record.features
        for start in fixed_horizon_query_starts(
            int(episode.model_actions.shape[0]),
            action_horizon=action_horizon,
            stride=query_stride,
        ):
            recent = (
                None
                if start < action_horizon
                else episode.model_actions[start - action_horizon : start]
            )
            output.append(
                OracleQuery(
                    query_key=episode.context_keys[start],
                    gt_model_action=episode.model_actions[
                        start : start + action_horizon
                    ],
                    episode_key=(
                        episode.dataset_id,
                        episode.dataset_index,
                        episode.episode_index,
                    ),
                    recent_action=recent,
                    source_episode_sha256=episode.source_episode_sha256,
                    feature_episode_sha256=episode.feature_episode_sha256,
                )
            )
    if not output:
        raise OfflinePipelineError("query collection contains no full action horizon")
    return tuple(output)


__all__ = [
    "FeatureCacheCollection",
    "FeatureCollectionContract",
    "FeatureDataBinding",
    "OfflinePipelineError",
    "assert_feature_collections_disjoint",
    "build_candidate_cache_from_collection",
    "build_event_bank_from_collection",
    "build_oracle_queries_from_collection",
    "fixed_horizon_query_starts",
    "load_feature_cache_collection",
    "validate_event_bank_data_binding",
    "validate_feature_collection_against_catalog",
    "validate_query_collection_against_bank",
]
