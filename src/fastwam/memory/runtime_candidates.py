"""Strict runtime bridge from immutable candidate caches to bank payload rows.

The offline cache stores stable :class:`EventId` values, not positional bank
indices.  This module resolves those identities once, validates the complete
artifact/action contract, and exposes fixed-width rows for model collation.
Padding uses ``-1`` only as an explicit sentinel; payload gathering first masks
and range-checks every row, so NumPy is never called with a negative index.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from .action_contract import (
    ActionSpaceContract,
    ActionSpaceContractError,
    validate_action_space_contract,
)
from .bank_contract import validate_warm_v1_bank
from .candidate_cache import (
    MANIFEST_FILENAME as CANDIDATE_MANIFEST_FILENAME,
    CandidateCache,
    QueryId,
    canonical_event_bank_content_hash,
)
from .event_bank import MANIFEST_FILENAME as BANK_MANIFEST_FILENAME, EventBank
from .manifest import sha256_file
from .payload_names import MODEL_SPACE_ACTION
from .schema import EventId


INVALID_BANK_ROW = -1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_QUERY_SPLITS = frozenset({"train", "dev"})
_RECIPE_FIELDS = frozenset(
    {
        "implementation",
        "action_horizon",
        "query_stride",
        "top_k",
        "query_split",
        "query_data_binding",
        "episode_exclusion",
    }
)
_BINDING_FIELDS = frozenset(
    {"catalog_sha256", "audit_report_sha256", "split"}
)
_REQUIRED_EXCLUSIONS = (
    "global_episode_identity",
    "source_episode_sha256",
    "feature_episode_sha256",
)


class RuntimeCandidateError(ValueError):
    """Base class for runtime candidate bridge failures."""


class RuntimeCandidateContractError(RuntimeCandidateError):
    """Raised when bank/cache/runtime contracts do not describe one snapshot."""


class RuntimeCandidateGatherError(RuntimeCandidateError):
    """Raised before an unsafe or inconsistent bank payload gather."""


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RuntimeCandidateContractError(
            f"{field} must be a lowercase SHA-256 digest"
        )
    return value


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise RuntimeCandidateContractError(f"{field} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise RuntimeCandidateContractError(f"{field} must be a positive integer")
    return result


def _query_split(value: object, field: str = "query_split") -> str:
    if not isinstance(value, str) or value not in _QUERY_SPLITS:
        raise RuntimeCandidateContractError(f"{field} must be 'train' or 'dev'")
    return str(value)


def _binding(value: object, field: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != _BINDING_FIELDS:
        raise RuntimeCandidateContractError(
            f"{field} must contain exactly {sorted(_BINDING_FIELDS)!r}"
        )
    split = value["split"]
    if not isinstance(split, str) or split not in {"train", "dev", "test"}:
        raise RuntimeCandidateContractError(
            f"{field}.split must be train, dev, or test"
        )
    return {
        "catalog_sha256": _sha256(
            value["catalog_sha256"], f"{field}.catalog_sha256"
        ),
        "audit_report_sha256": _sha256(
            value["audit_report_sha256"], f"{field}.audit_report_sha256"
        ),
        "split": str(split),
    }


def _runtime_recipe(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _RECIPE_FIELDS:
        actual = set(value) if isinstance(value, Mapping) else set()
        raise RuntimeCandidateContractError(
            "candidate build_recipe fields are not the closed runtime schema; "
            f"missing={sorted(_RECIPE_FIELDS - actual)!r}, "
            f"extra={sorted(actual - _RECIPE_FIELDS)!r}"
        )
    if value["implementation"] != "exact_cosine_v1":
        raise RuntimeCandidateContractError(
            "candidate build_recipe implementation must be 'exact_cosine_v1'"
        )
    split = _query_split(value["query_split"], "build_recipe.query_split")
    action_horizon = _positive_int(
        value["action_horizon"], "build_recipe.action_horizon"
    )
    query_stride = _positive_int(
        value["query_stride"], "build_recipe.query_stride"
    )
    top_k = _positive_int(value["top_k"], "build_recipe.top_k")
    if split == "train" and query_stride != 1:
        raise RuntimeCandidateContractError(
            "train candidate cache build_recipe must use query_stride=1"
        )
    query_binding = _binding(
        value["query_data_binding"], "build_recipe.query_data_binding"
    )
    if query_binding["split"] != split:
        raise RuntimeCandidateContractError(
            "candidate query_split disagrees with query_data_binding.split"
        )
    exclusions = value["episode_exclusion"]
    if not isinstance(exclusions, list) or tuple(exclusions) != _REQUIRED_EXCLUSIONS:
        raise RuntimeCandidateContractError(
            "candidate build_recipe must prove identity, source-hash, and "
            "feature-hash episode exclusion in canonical order"
        )
    return {
        "implementation": "exact_cosine_v1",
        "action_horizon": action_horizon,
        "query_stride": query_stride,
        "top_k": top_k,
        "query_split": split,
        "query_data_binding": query_binding,
        "episode_exclusion": list(_REQUIRED_EXCLUSIONS),
    }


def _readonly_array(name: str, value: object, dtype: np.dtype) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != dtype or array.ndim != 1 or array.size <= 0:
        raise RuntimeCandidateContractError(
            f"{name} must have dtype {dtype} and non-empty shape [K]"
        )
    result = np.array(array, copy=True, order="C")
    result.flags.writeable = False
    return result


@dataclass(frozen=True, slots=True)
class ResolvedCandidateRow:
    """One fixed-width, mask-bearing cache row resolved to bank positions."""

    query_id: QueryId
    bank_rows: np.ndarray
    mask: np.ndarray
    cosine_scores: np.ndarray
    event_ids: tuple[EventId | None, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.query_id, QueryId):
            raise TypeError("query_id must be QueryId")
        rows = _readonly_array("bank_rows", self.bank_rows, np.dtype(np.int64))
        mask = _readonly_array("mask", self.mask, np.dtype(np.bool_))
        scores = _readonly_array(
            "cosine_scores", self.cosine_scores, np.dtype(np.float32)
        )
        if rows.shape != mask.shape or rows.shape != scores.shape:
            raise RuntimeCandidateContractError(
                "bank_rows, mask, and cosine_scores must have identical [K] shape"
            )
        events = tuple(self.event_ids)
        if len(events) != rows.size:
            raise RuntimeCandidateContractError(
                "event_ids must contain exactly one value per fixed-width slot"
            )
        valid_count = int(mask.sum())
        if valid_count and not np.all(mask[:valid_count]):
            raise RuntimeCandidateContractError(
                "valid candidate slots must form a contiguous prefix"
            )
        if np.any(mask[valid_count:]):
            raise RuntimeCandidateContractError(
                "valid candidate slots must form a contiguous prefix"
            )
        if np.any(rows[mask] < 0):
            raise RuntimeCandidateContractError(
                "valid candidate slots must contain non-negative bank rows"
            )
        if np.any(rows[~mask] != INVALID_BANK_ROW):
            raise RuntimeCandidateContractError(
                f"masked candidate slots must use row sentinel {INVALID_BANK_ROW}"
            )
        if np.any(scores[~mask] != 0.0):
            raise RuntimeCandidateContractError(
                "masked candidate slots must use zero cosine score"
            )
        valid_scores = scores[mask]
        if not np.all(np.isfinite(valid_scores)) or np.any(valid_scores < -1.0) or np.any(
            valid_scores > 1.0
        ):
            raise RuntimeCandidateContractError(
                "valid cosine scores must be finite values in [-1, 1]"
            )
        if valid_scores.size > 1 and np.any(valid_scores[:-1] < valid_scores[1:]):
            raise RuntimeCandidateContractError(
                "candidate cosine scores must be sorted in non-increasing order"
            )
        valid_rows = rows[mask]
        if len(set(int(value) for value in valid_rows.tolist())) != valid_count:
            raise RuntimeCandidateContractError(
                "valid candidate bank rows must be unique"
            )
        for index, (is_valid, event_id) in enumerate(zip(mask, events, strict=True)):
            if bool(is_valid) and not isinstance(event_id, EventId):
                raise RuntimeCandidateContractError(
                    f"valid slot {index} must contain an EventId"
                )
            if not bool(is_valid) and event_id is not None:
                raise RuntimeCandidateContractError(
                    f"masked slot {index} must contain event_id=None"
                )

        object.__setattr__(self, "bank_rows", rows)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "cosine_scores", scores)
        object.__setattr__(self, "event_ids", events)

    @property
    def fixed_k(self) -> int:
        return int(self.bank_rows.size)

    @property
    def valid_count(self) -> int:
        return int(self.mask.sum())

    @property
    def valid_bank_rows(self) -> np.ndarray:
        rows = np.ascontiguousarray(self.bank_rows[self.mask])
        rows.flags.writeable = False
        return rows


class RuntimeCandidateResolver:
    """Resolve verified candidate identities into safe fixed-width bank rows."""

    def __init__(
        self,
        bank: EventBank,
        cache: CandidateCache,
        *,
        event_bank_manifest_sha256: str,
        candidate_manifest_sha256: str | None = None,
        expected_query_split: str = "dev",
        expected_query_corpus_sha256: str | None = None,
        expected_action_space: ActionSpaceContract | Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(bank, EventBank):
            raise TypeError("bank must be EventBank")
        if not isinstance(cache, CandidateCache):
            raise TypeError("cache must be CandidateCache")
        if bank.manifest is None:
            raise RuntimeCandidateContractError(
                "runtime event bank must carry a persisted manifest"
            )
        if cache.manifest is None:
            raise RuntimeCandidateContractError(
                "runtime candidate cache must carry a persisted manifest"
            )

        bank_manifest_hash = _sha256(
            event_bank_manifest_sha256, "event_bank_manifest_sha256"
        )
        candidate_manifest_hash = (
            None
            if candidate_manifest_sha256 is None
            else _sha256(
                candidate_manifest_sha256,
                "candidate_manifest_sha256",
            )
        )
        split = _query_split(expected_query_split, "expected_query_split")
        if cache.manifest.event_bank_manifest_hash != bank_manifest_hash:
            raise RuntimeCandidateContractError(
                "candidate cache is bound to a different event-bank manifest"
            )
        bank_content_hash = canonical_event_bank_content_hash(
            bank.manifest.content_hashes
        )
        if cache.manifest.event_bank_content_hash != bank_content_hash:
            raise RuntimeCandidateContractError(
                "candidate cache is bound to different event-bank content"
            )
        if dict(cache.manifest.query_key_encoder) != dict(bank.manifest.encoder):
            raise RuntimeCandidateContractError(
                "candidate query encoder does not match the event-bank encoder"
            )
        if expected_query_corpus_sha256 is not None:
            expected_corpus = _sha256(
                expected_query_corpus_sha256, "expected_query_corpus_sha256"
            )
            if cache.manifest.query_corpus_hash != expected_corpus:
                raise RuntimeCandidateContractError(
                    "candidate query corpus does not match the requested snapshot"
                )

        recipe = _runtime_recipe(cache.manifest.build_recipe)
        if recipe["query_split"] != split:
            raise RuntimeCandidateContractError(
                "candidate cache query split does not match runtime mode"
            )
        bank_binding = _binding(
            bank.manifest.provenance.get("data_binding"),
            "event_bank.provenance.data_binding",
        )
        if bank_binding["split"] != "train":
            raise RuntimeCandidateContractError(
                "runtime event bank must be built from the train split"
            )
        query_binding = recipe["query_data_binding"]
        for field in ("catalog_sha256", "audit_report_sha256"):
            if query_binding[field] != bank_binding[field]:
                raise RuntimeCandidateContractError(
                    f"candidate query binding {field} does not match the event bank"
                )
        if split == "train":
            bank_collection_hash = _sha256(
                bank.manifest.provenance.get("feature_collection_sha256"),
                "event_bank.provenance.feature_collection_sha256",
            )
            if cache.manifest.query_corpus_hash != bank_collection_hash:
                raise RuntimeCandidateContractError(
                    "train candidate cache must cover the exact feature collection "
                    "used to build the event bank"
                )

        summary = validate_warm_v1_bank(bank)
        if recipe["action_horizon"] != summary.action_horizon:
            raise RuntimeCandidateContractError(
                "candidate action_horizon does not match event-bank payload"
            )
        try:
            action_contract = validate_action_space_contract(
                bank.manifest.action_normalizer["contract"]
            )
        except (KeyError, TypeError, ActionSpaceContractError) as exc:
            raise RuntimeCandidateContractError(
                "event bank lacks a valid versioned action-space contract"
            ) from exc
        if action_contract.action_dim != summary.action_dim:
            raise RuntimeCandidateContractError(
                "event-bank action payload disagrees with its action-space contract"
            )
        if expected_action_space is not None:
            if isinstance(expected_action_space, ActionSpaceContract):
                expected_contract = expected_action_space
            else:
                try:
                    expected_contract = validate_action_space_contract(
                        expected_action_space
                    )
                except (TypeError, ActionSpaceContractError) as exc:
                    raise RuntimeCandidateContractError(
                        "expected_action_space is not a valid action contract"
                    ) from exc
            if expected_contract != action_contract:
                raise RuntimeCandidateContractError(
                    "event-bank action-space contract does not match runtime"
                )

        cache.validate_against_event_bank(bank)
        event_to_row = {
            event_id: index for index, event_id in enumerate(bank.event_ids)
        }
        fixed_k = int(recipe["top_k"])
        rows_by_query: dict[QueryId, tuple[tuple[int, EventId, float], ...]] = {}
        for query_id, candidates in zip(
            cache.query_ids, cache.candidates, strict=True
        ):
            if len(candidates) > fixed_k:
                raise RuntimeCandidateContractError(
                    f"candidate row for {query_id!r} exceeds configured top_k={fixed_k}"
                )
            resolved: list[tuple[int, EventId, float]] = []
            previous_score = float("inf")
            for candidate in candidates:
                bank_row = event_to_row.get(candidate.event_id)
                if bank_row is None:
                    raise RuntimeCandidateContractError(
                        f"candidate event {candidate.event_id!r} is absent from bank"
                    )
                if candidate.cosine_score > previous_score:
                    raise RuntimeCandidateContractError(
                        f"candidate row for {query_id!r} is not score sorted"
                    )
                previous_score = candidate.cosine_score
                resolved.append(
                    (bank_row, candidate.event_id, candidate.cosine_score)
                )
            rows_by_query[query_id] = tuple(resolved)

        self._bank = bank
        self._cache = cache
        self._event_to_row = MappingProxyType(event_to_row)
        self._rows_by_query = MappingProxyType(rows_by_query)
        self._fixed_k = fixed_k
        self._query_split = split
        self._recipe = MappingProxyType(
            {
                **recipe,
                "query_data_binding": MappingProxyType(
                    dict(recipe["query_data_binding"])
                ),
                "episode_exclusion": tuple(recipe["episode_exclusion"]),
            }
        )
        self._action_space = action_contract
        self._bank_manifest_sha256 = bank_manifest_hash
        self._bank_content_sha256 = bank_content_hash
        self._candidate_manifest_sha256 = candidate_manifest_hash

    @classmethod
    def from_artifacts(
        cls,
        bank_directory: str | Path,
        candidate_directory: str | Path,
        *,
        expected_query_split: str = "dev",
        expected_query_corpus_sha256: str | None = None,
        expected_action_space: ActionSpaceContract | Mapping[str, Any] | None = None,
    ) -> "RuntimeCandidateResolver":
        """Load both immutable artifacts while pinning one bank snapshot."""

        bank_directory = Path(bank_directory).expanduser().resolve()
        candidate_directory = Path(candidate_directory).expanduser().resolve()
        bank_manifest_path = bank_directory / BANK_MANIFEST_FILENAME
        before_hash = sha256_file(bank_manifest_path)
        bank = EventBank.load(bank_directory)
        if sha256_file(bank_manifest_path) != before_hash:
            raise RuntimeCandidateContractError(
                "event-bank manifest changed while runtime artifacts were loading"
            )
        if bank.manifest is None:  # Defensive; EventBank.load always sets it.
            raise RuntimeCandidateContractError("loaded event bank has no manifest")
        content_hash = canonical_event_bank_content_hash(
            bank.manifest.content_hashes
        )
        candidate_manifest_path = candidate_directory / CANDIDATE_MANIFEST_FILENAME
        candidate_manifest_hash = sha256_file(candidate_manifest_path)
        cache = CandidateCache.load(
            candidate_directory,
            expected_event_bank_manifest_hash=before_hash,
            expected_event_bank_content_hash=content_hash,
            expected_query_corpus_hash=expected_query_corpus_sha256,
            expected_query_key_encoder=bank.manifest.encoder,
        )
        if sha256_file(bank_manifest_path) != before_hash:
            raise RuntimeCandidateContractError(
                "event-bank manifest changed while candidate cache was loading"
            )
        # CandidateCache.load already requires this filename, but referencing it
        # here makes the two-artifact dependency explicit and fail-fast.
        if not candidate_manifest_path.is_file():
            raise RuntimeCandidateContractError(
                "candidate cache manifest disappeared during runtime load"
            )
        if sha256_file(candidate_manifest_path) != candidate_manifest_hash:
            raise RuntimeCandidateContractError(
                "candidate-cache manifest changed while runtime artifacts were loading"
            )
        return cls(
            bank,
            cache,
            event_bank_manifest_sha256=before_hash,
            candidate_manifest_sha256=candidate_manifest_hash,
            expected_query_split=expected_query_split,
            expected_query_corpus_sha256=expected_query_corpus_sha256,
            expected_action_space=expected_action_space,
        )

    @property
    def fixed_k(self) -> int:
        return self._fixed_k

    @property
    def query_split(self) -> str:
        return self._query_split

    @property
    def query_stride(self) -> int:
        return int(self._recipe["query_stride"])

    @property
    def action_horizon(self) -> int:
        return int(self._recipe["action_horizon"])

    @property
    def action_space(self) -> ActionSpaceContract:
        return self._action_space

    @property
    def query_catalog_sha256(self) -> str:
        binding = self._recipe["query_data_binding"]
        assert isinstance(binding, Mapping)
        value = binding["catalog_sha256"]
        assert isinstance(value, str)
        return value

    @property
    def query_audit_sha256(self) -> str:
        binding = self._recipe["query_data_binding"]
        assert isinstance(binding, Mapping)
        value = binding["audit_report_sha256"]
        assert isinstance(value, str)
        return value

    @property
    def bank_manifest_sha256(self) -> str:
        return self._bank_manifest_sha256

    @property
    def bank_content_sha256(self) -> str:
        return self._bank_content_sha256

    @property
    def candidate_manifest_sha256(self) -> str:
        if self._candidate_manifest_sha256 is None:
            raise RuntimeCandidateContractError(
                "candidate manifest digest was not pinned; load resolver from artifacts"
            )
        return self._candidate_manifest_sha256

    @property
    def build_recipe(self) -> Mapping[str, Any]:
        """Return the recursively immutable, validated cache recipe."""

        return self._recipe

    @property
    def event_to_row(self) -> Mapping[EventId, int]:
        return self._event_to_row

    @property
    def query_corpus_sha256(self) -> str:
        assert self._cache.manifest is not None
        return self._cache.manifest.query_corpus_hash

    def _empty_row(self, query_id: QueryId) -> ResolvedCandidateRow:
        return ResolvedCandidateRow(
            query_id=query_id,
            bank_rows=np.full(
                (self._fixed_k,), INVALID_BANK_ROW, dtype=np.int64
            ),
            mask=np.zeros((self._fixed_k,), dtype=np.bool_),
            cosine_scores=np.zeros((self._fixed_k,), dtype=np.float32),
            event_ids=(None,) * self._fixed_k,
        )

    def resolve(
        self,
        query_id: QueryId,
        *,
        allow_missing: bool = False,
    ) -> ResolvedCandidateRow:
        """Resolve exactly one QueryId; missing rows fail unless explicit."""

        if not isinstance(query_id, QueryId):
            raise TypeError("query_id must be QueryId")
        if not isinstance(allow_missing, (bool, np.bool_)):
            raise TypeError("allow_missing must be a boolean")
        candidates = self._rows_by_query.get(query_id)
        if candidates is None:
            if not bool(allow_missing):
                raise KeyError(f"candidate cache has no exact row for {query_id!r}")
            return self._empty_row(query_id)

        rows = np.full((self._fixed_k,), INVALID_BANK_ROW, dtype=np.int64)
        mask = np.zeros((self._fixed_k,), dtype=np.bool_)
        scores = np.zeros((self._fixed_k,), dtype=np.float32)
        events: list[EventId | None] = [None] * self._fixed_k
        for index, (bank_row, event_id, score) in enumerate(candidates):
            rows[index] = bank_row
            mask[index] = True
            scores[index] = score
            events[index] = event_id
        return ResolvedCandidateRow(
            query_id=query_id,
            bank_rows=rows,
            mask=mask,
            cosine_scores=scores,
            event_ids=tuple(events),
        )

    def resolve_many(
        self,
        query_ids: Sequence[QueryId],
        *,
        allow_missing: bool = False,
    ) -> tuple[ResolvedCandidateRow, ...]:
        """Resolve a batch without changing caller order."""

        if isinstance(query_ids, (str, bytes)):
            raise TypeError("query_ids must be a sequence of QueryId values")
        return tuple(
            self.resolve(query_id, allow_missing=allow_missing)
            for query_id in query_ids
        )

    def gather_payload(
        self,
        resolved: ResolvedCandidateRow,
        payload_name: str,
        *,
        fill_value: int | float | bool = 0,
    ) -> np.ndarray:
        """Gather valid rows into ``[K,...]`` without indexing the sentinel."""

        if not isinstance(resolved, ResolvedCandidateRow):
            raise TypeError("resolved must be ResolvedCandidateRow")
        if resolved.fixed_k != self._fixed_k:
            raise RuntimeCandidateGatherError(
                "resolved row width does not match this resolver"
            )
        if not isinstance(payload_name, str) or not payload_name:
            raise TypeError("payload_name must be a non-empty string")

        valid_positions = np.flatnonzero(resolved.mask)
        valid_rows = resolved.bank_rows[valid_positions]
        if np.any(valid_rows < 0) or np.any(valid_rows >= len(self._bank)):
            raise RuntimeCandidateGatherError(
                "resolved row contains an out-of-range valid bank index"
            )
        for position, bank_row in zip(
            valid_positions.tolist(), valid_rows.tolist(), strict=True
        ):
            event_id = resolved.event_ids[position]
            if not isinstance(event_id, EventId):
                raise RuntimeCandidateGatherError(
                    "valid resolved slot has no EventId"
                )
            expected_row = self._event_to_row.get(event_id)
            if expected_row != int(bank_row):
                raise RuntimeCandidateGatherError(
                    "resolved EventId-to-row binding is inconsistent"
                )

        try:
            payload = self._bank.payload(payload_name)
        except KeyError as exc:
            raise RuntimeCandidateGatherError(
                f"event bank has no payload {payload_name!r}"
            ) from exc
        try:
            output = np.full(
                (self._fixed_k, *payload.shape[1:]),
                fill_value,
                dtype=payload.dtype,
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeCandidateGatherError(
                f"fill_value is incompatible with payload {payload_name!r}"
            ) from exc
        if valid_rows.size:
            # Only the already mask- and range-checked non-negative rows reach
            # NumPy indexing. The -1 padding sentinel is never gathered.
            output[valid_positions] = payload[valid_rows]
        output = np.ascontiguousarray(output)
        output.flags.writeable = False
        return output

    def gather_model_actions(
        self, resolved: ResolvedCandidateRow
    ) -> np.ndarray:
        """Safely gather the canonical model-space source action payload."""

        return self.gather_payload(resolved, MODEL_SPACE_ACTION)


__all__ = [
    "INVALID_BANK_ROW",
    "ResolvedCandidateRow",
    "RuntimeCandidateContractError",
    "RuntimeCandidateError",
    "RuntimeCandidateGatherError",
    "RuntimeCandidateResolver",
]
