"""Factual training payloads for the complete WARM retrospective path.

The M2 candidate adapter intentionally exposes only action-source tensors.
This wrapper adds the state/action/effect evidence required by consequence
alignment, using the already verified event bank and immutable per-episode
DINO feature caches.  It never runs a new encoder, invents labels, or reads a
future from the model: all future semantic values are offline teachers and are
masked before padded episode tails.
"""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from fastwam.memory.candidate_cache import QueryId
from fastwam.memory.episode_memory import action_summary_signature
from fastwam.memory.offline_pipeline import (
    FeatureCacheCollection,
    load_feature_cache_collection,
)
from fastwam.memory.payload_names import (
    EFFECT_POST,
    EFFECT_PRE,
    OBSERVED_GRIPPER_STATE,
    START_PROPRIO,
)
from fastwam.memory.online_episode_memory import (
    OnlineRetrospectiveEpisodeMemory,
)
from fastwam.memory.runtime_candidates import RuntimeCandidateResolver
from fastwam.utils.logging_config import get_logger

from .warm_candidates import (
    RuntimeCandidateDatasetAdapter,
    RuntimeCandidateDatasetContractError,
    WARM_CANDIDATE_MASK,
)


WARM_CANDIDATE_CONTEXT = "warm_candidate_context"
WARM_CANDIDATE_EFFECT_PRE = "warm_candidate_effect_pre"
WARM_CANDIDATE_EFFECT_POST = "warm_candidate_effect_post"
WARM_CANDIDATE_EFFECT_DELTA = "warm_candidate_effect_delta"
WARM_CANDIDATE_START_PROPRIO = "warm_candidate_start_proprio"
WARM_CANDIDATE_GRIPPER = "warm_candidate_gripper"
WARM_CANDIDATE_TIMING = "warm_candidate_timing"
WARM_CANDIDATE_SUPPORT = "warm_candidate_support"
WARM_CURRENT_CONTEXT = "warm_current_context"
WARM_CURRENT_SEMANTIC = "warm_current_semantic"
WARM_FUTURE_SEMANTIC = "warm_future_semantic"
WARM_TARGET_EFFECT = "warm_target_effect"
WARM_FUTURE_VALID = "warm_future_valid"
WARM_EPISODE_TOKENS = "warm_episode_tokens"
WARM_EPISODE_MASK = "warm_episode_mask"
WARM_EPISODE_ACTION_SUMMARIES = "warm_episode_action_summaries"
WARM_EPISODE_ACTION_MASK = "warm_episode_action_mask"

logger = get_logger(__name__)

WARM_RETROSPECTIVE_FIELDS = (
    WARM_CANDIDATE_CONTEXT,
    WARM_CANDIDATE_EFFECT_PRE,
    WARM_CANDIDATE_EFFECT_POST,
    WARM_CANDIDATE_EFFECT_DELTA,
    WARM_CANDIDATE_START_PROPRIO,
    WARM_CANDIDATE_GRIPPER,
    WARM_CANDIDATE_TIMING,
    WARM_CANDIDATE_SUPPORT,
    WARM_CURRENT_CONTEXT,
    WARM_CURRENT_SEMANTIC,
    WARM_FUTURE_SEMANTIC,
    WARM_TARGET_EFFECT,
    WARM_FUTURE_VALID,
    WARM_EPISODE_TOKENS,
    WARM_EPISODE_MASK,
    WARM_EPISODE_ACTION_SUMMARIES,
    WARM_EPISODE_ACTION_MASK,
)


class RetrospectiveFeatureStoreError(ValueError):
    """Raised when factual feature caches cannot support a WARM query."""


def collect_feature_payloads(directory: str | Path) -> tuple[Path, ...]:
    """Collect only NPZ files that have the strict feature-cache sidecar."""

    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"feature cache directory not found: {root}")
    paths = tuple(
        path
        for path in sorted(root.rglob("*.npz"))
        if path.with_suffix(".manifest.json").is_file()
    )
    if not paths:
        raise RetrospectiveFeatureStoreError(
            f"no episode feature caches found below {root}"
        )
    return paths


def collect_feature_payloads_from_list(list_path: str | Path) -> tuple[Path, ...]:
    """Load an immutable split-specific feature list.

    Relative entries are resolved against the list file's parent, matching the
    lists emitted by ``precompute_warm_features.py``.  A list is preferable to
    recursively scanning a mixed train/dev feature root because its collection
    digest is exactly the one bound by the candidate-cache contract.
    """

    source = Path(list_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"feature cache list not found: {source}")
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise RetrospectiveFeatureStoreError(
            f"cannot read feature cache list: {source}"
        ) from exc
    paths: list[Path] = []
    seen: set[Path] = set()
    for line_number, raw in enumerate(lines, start=1):
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = source.parent / candidate
        candidate = candidate.resolve()
        if candidate.suffix.lower() != ".npz" or not candidate.is_file():
            raise RetrospectiveFeatureStoreError(
                f"feature list entry {line_number} is not an NPZ file: {candidate}"
            )
        if not candidate.with_suffix(".manifest.json").is_file():
            raise RetrospectiveFeatureStoreError(
                f"feature list entry {line_number} has no manifest sidecar: {candidate}"
            )
        if candidate in seen:
            raise RetrospectiveFeatureStoreError(
                f"feature list contains duplicate entry: {candidate}"
            )
        seen.add(candidate)
        paths.append(candidate)
    if not paths:
        raise RetrospectiveFeatureStoreError(
            f"feature cache list is empty: {source}"
        )
    return tuple(paths)


def _action_summary_vector(
    actions: np.ndarray,
    *,
    action_horizon: int,
    gripper_indices: tuple[int, ...],
    previous_signatures: tuple[np.ndarray, ...],
    repetition_cosine_threshold: float = 0.97,
    repetition_distance_threshold: float = 0.20,
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror the evaluator's compact factual executed-action summary."""

    values = np.asarray(actions, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] <= 0 or values.shape[1] <= 0:
        raise RetrospectiveFeatureStoreError(
            "episode action summary requires non-empty [T,Da] actions"
        )
    action_dim = int(values.shape[1])
    movement = tuple(index for index in range(action_dim) if index not in gripper_indices)
    mean = np.zeros((action_dim,), dtype=np.float32)
    displacement = np.zeros((action_dim,), dtype=np.float32)
    if movement:
        mean[list(movement)] = values[:, movement].mean(axis=0, dtype=np.float32)
        displacement[list(movement)] = values[:, movement].sum(axis=0, dtype=np.float32)
    terminal = np.zeros((action_dim,), dtype=np.float32)
    if gripper_indices:
        terminal[list(gripper_indices)] = values[-1, list(gripper_indices)]

    curvature_values: list[float] = []
    if movement and values.shape[0] > 1:
        vectors = values[:, movement].astype(np.float64, copy=False)
        for left, right in zip(vectors[:-1], vectors[1:], strict=True):
            denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
            if denominator > 1.0e-8:
                cosine = float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))
                curvature_values.append((1.0 - cosine) * 0.5)
    curvature = float(np.mean(curvature_values)) if curvature_values else 0.0
    terminal_gripper_values = (
        values[-1, list(gripper_indices)]
        if gripper_indices
        else np.empty((0,), dtype=np.float32)
    )
    signature = action_summary_signature(
        mean,
        displacement,
        terminal_gripper_values,
    )
    repetition = 0.0
    best_distance = np.inf
    repeated = 0.0
    if previous_signatures:
        current64 = signature.astype(np.float64, copy=False)
        repetition = -1.0
        for previous_signature in previous_signatures:
            previous64 = np.asarray(previous_signature, dtype=np.float64)
            denominator = float(
                np.linalg.norm(current64) * np.linalg.norm(previous64)
            )
            if denominator > 1.0e-8:
                similarity = float(
                    np.clip(np.dot(current64, previous64) / denominator, -1.0, 1.0)
                )
            elif not np.any(current64) and not np.any(previous64):
                similarity = 1.0
            else:
                similarity = 0.0
            distance_denominator = float(np.linalg.norm(current64)) + float(
                np.linalg.norm(previous64)
            )
            distance = (
                0.0
                if distance_denominator <= 1.0e-8
                else float(
                    np.linalg.norm(current64 - previous64)
                    / (distance_denominator + 1.0e-8)
                )
            )
            if similarity > repetition or (
                np.isclose(similarity, repetition) and distance < best_distance
            ):
                repetition = similarity
                best_distance = distance
        repeated = float(
            repetition >= repetition_cosine_threshold
            and best_distance <= repetition_distance_threshold
        )
    scalars = np.asarray(
        [
            min(1.0, float(values.shape[0]) / float(action_horizon)),
            curvature,
            repetition,
            repeated,
        ],
        dtype=np.float32,
    )
    feature_vector = np.concatenate((mean, displacement, terminal, scalars))
    return np.ascontiguousarray(feature_vector), signature


def _factual_payload(features: Any, frame: int) -> dict[str, np.ndarray]:
    vae = features.vae_features
    if vae is None:
        raise RetrospectiveFeatureStoreError(
            "complete WARM episode replay requires factual VAE features"
        )
    return {
        "world_tokens": np.asarray(
            features.semantic_features[frame], dtype=np.float32
        ),
        "vae_latent": np.asarray(vae[frame], dtype=np.float32),
        "proprio": np.asarray(features.proprio[frame], dtype=np.float32),
    }


def _causal_event_frames_by_query(
    features: Any,
    *,
    action_horizon: int,
    action_chunk_size: int,
    action_dim: int,
    semantic_dim: int,
    gripper_indices: tuple[int, ...],
    recent_event_capacity: int,
) -> Mapping[int, tuple[int, ...]]:
    """Replay the exact online state machine without reading a future suffix.

    One independent replay is used for every possible replan-phase residue so
    stride-one training queries still see the same chunk cadence as rollout.
    Only event frame identities are retained; compact tensors are materialized
    lazily from the factual feature cache.
    """

    observation_count = int(features.semantic_features.shape[0])
    result: dict[int, tuple[int, ...]] = {}
    for residue in range(min(action_chunk_size, observation_count)):
        memory = OnlineRetrospectiveEpisodeMemory(
            action_dim=action_dim,
            action_horizon=action_horizon,
            semantic_dim=semantic_dim,
            gripper_indices=gripper_indices,
            recent_event_capacity=recent_event_capacity,
        )
        memory.begin_episode(residue)
        writes: list[int] = []
        last_recorded: int | None = None
        if residue > 0:
            memory.record_factual_observation(
                frame_index=0,
                factual_payload=_factual_payload(features, 0),
                executed_actions_since_previous=None,
                include_snapshot_sha256=False,
            )
            last_recorded = 0
        for frame in range(residue, observation_count, action_chunk_size):
            result[frame] = tuple(writes)
            actions = (
                None
                if last_recorded is None
                else np.asarray(
                    features.model_actions[last_recorded:frame],
                    dtype=np.float32,
                )
            )
            evidence = memory.record_factual_observation(
                frame_index=frame,
                factual_payload=_factual_payload(features, frame),
                executed_actions_since_previous=actions,
                include_snapshot_sha256=False,
            )
            if bool(evidence["event_written"]):
                writes.append(frame)
            last_recorded = frame
        memory.end_episode(include_snapshot_sha256=False)
    if set(result) != set(range(observation_count)):
        raise RetrospectiveFeatureStoreError(
            "causal episode replay did not cover every factual query frame"
        )
    return MappingProxyType(result)


def _compressed_event_end_frames(
    event_frames: Iterable[int],
    semantic_features: np.ndarray,
    *,
    capacity: int,
) -> list[int]:
    """Mirror EpisodeWorkingMemory's protected-latest merge policy."""

    entries = [
        {
            "end": int(frame),
            "representative": np.asarray(
                semantic_features[int(frame)], dtype=np.float64
            ),
            "mass": 1,
        }
        for frame in event_frames
    ]
    while len(entries) > capacity:
        candidate_count = len(entries) - 2
        if candidate_count < 1:
            raise RetrospectiveFeatureStoreError(
                "no legal causal episode-event merge pair"
            )
        similarities: list[float] = []
        for index in range(candidate_count):
            left = entries[index]["representative"].reshape(-1)
            right = entries[index + 1]["representative"].reshape(-1)
            denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
            similarity = (
                1.0
                if denominator <= 1.0e-8 and not np.any(left) and not np.any(right)
                else 0.0
                if denominator <= 1.0e-8
                else float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))
            )
            similarities.append(similarity)
        merge_index = int(np.argmax(np.asarray(similarities, dtype=np.float64)))
        older = entries[merge_index]
        newer = entries[merge_index + 1]
        total = int(older["mass"]) + int(newer["mass"])
        merged = {
            "end": int(newer["end"]),
            "representative": (
                older["representative"] * int(older["mass"])
                + newer["representative"] * int(newer["mass"])
            )
            / total,
            "mass": total,
        }
        entries[merge_index : merge_index + 2] = [merged]
    return [int(entry["end"]) for entry in entries]


class RetrospectiveFeatureStore:
    """Immutable QueryId-to-factual-semantic lookup for one data split."""

    def __init__(
        self,
        collection: FeatureCacheCollection,
        *,
        expected_split: str,
        expected_collection_sha256: str,
        expected_catalog_sha256: str,
        action_horizon: int,
        recent_event_capacity: int = 6,
        action_summary_capacity: int = 2,
        action_summary_chunk_size: int | None = None,
        gripper_indices: Iterable[int] = (),
    ) -> None:
        if not isinstance(collection, FeatureCacheCollection):
            raise TypeError("collection must be FeatureCacheCollection")
        if expected_split not in {"train", "dev"}:
            raise ValueError("expected_split must be 'train' or 'dev'")
        if collection.split != expected_split:
            raise RetrospectiveFeatureStoreError(
                f"feature split {collection.split!r} != {expected_split!r}"
            )
        if collection.content_hash != expected_collection_sha256:
            raise RetrospectiveFeatureStoreError(
                "feature collection does not match the candidate query corpus"
            )
        if collection.contract.catalog_hash != expected_catalog_sha256:
            raise RetrospectiveFeatureStoreError(
                "feature collection does not match the episode catalog"
            )
        if isinstance(action_horizon, bool) or not isinstance(action_horizon, int):
            raise TypeError("action_horizon must be a positive integer")
        if action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        if (
            isinstance(recent_event_capacity, bool)
            or not isinstance(recent_event_capacity, int)
            or recent_event_capacity <= 0
        ):
            raise ValueError("recent_event_capacity must be a positive integer")
        if (
            isinstance(action_summary_capacity, bool)
            or not isinstance(action_summary_capacity, int)
            or action_summary_capacity <= 0
        ):
            raise ValueError("action_summary_capacity must be a positive integer")
        if action_summary_chunk_size is None:
            action_summary_chunk_size = action_horizon
        if (
            isinstance(action_summary_chunk_size, bool)
            or not isinstance(action_summary_chunk_size, int)
            or action_summary_chunk_size <= 0
            or action_summary_chunk_size > action_horizon
        ):
            raise ValueError(
                "action_summary_chunk_size must lie in [1, action_horizon]"
            )
        parsed_gripper: list[int] = []
        for position, value in enumerate(gripper_indices):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(
                    f"gripper_indices[{position}] must be a non-negative integer"
                )
            if int(value) < 0:
                raise ValueError("gripper_indices must be non-negative")
            parsed_gripper.append(int(value))
        if len(set(parsed_gripper)) != len(parsed_gripper):
            raise ValueError("gripper_indices must be unique")

        records = {
            (
                record.metadata.dataset_id,
                record.metadata.dataset_index,
                record.metadata.episode_index,
            ): record
            for record in collection.records
        }
        action_dim = int(collection.records[0].features.model_actions.shape[1])
        parsed_gripper_tuple = tuple(sorted(parsed_gripper))
        if any(index >= action_dim for index in parsed_gripper_tuple):
            raise ValueError("gripper_indices must lie within the action dimension")
        for record in collection.records:
            if int(record.features.model_actions.shape[1]) != action_dim:
                raise RetrospectiveFeatureStoreError(
                    "feature collection contains inconsistent action dimensions"
                )
            if record.features.vae_features is None:
                raise RetrospectiveFeatureStoreError(
                    "complete WARM feature caches must include factual VAE features"
                )

        semantic_shape = tuple(
            int(value)
            for value in collection.records[0].features.semantic_features.shape[1:]
        )
        if len(semantic_shape) != 2:
            raise RetrospectiveFeatureStoreError(
                "complete WARM semantic features must have shape [T,N,D]"
            )
        logger.info(
            "Building causal retrospective replay index for %d episodes",
            len(records),
        )
        event_frames: dict[tuple[str, int, int], Mapping[int, tuple[int, ...]]] = {}
        for record_index, (key, record) in enumerate(records.items(), start=1):
            event_frames[key] = _causal_event_frames_by_query(
                record.features,
                action_horizon=int(action_horizon),
                action_chunk_size=int(action_summary_chunk_size),
                action_dim=action_dim,
                semantic_dim=semantic_shape[-1],
                gripper_indices=parsed_gripper_tuple,
                recent_event_capacity=int(recent_event_capacity),
            )
            if record_index % 100 == 0 or record_index == len(records):
                logger.info(
                    "Causal retrospective replay index: %d/%d episodes",
                    record_index,
                    len(records),
                )
        self._collection = collection
        self._records = MappingProxyType(records)
        self._event_frames = MappingProxyType(event_frames)
        self._action_horizon = int(action_horizon)
        self._recent_event_capacity = int(recent_event_capacity)
        self._action_summary_capacity = int(action_summary_capacity)
        self._action_summary_chunk_size = int(action_summary_chunk_size)
        self._gripper_indices = parsed_gripper_tuple

    @classmethod
    def from_paths(
        cls,
        paths: Iterable[str | Path],
        **kwargs: Any,
    ) -> "RetrospectiveFeatureStore":
        return cls(load_feature_cache_collection(paths), **kwargs)

    @property
    def collection_sha256(self) -> str:
        return self._collection.content_hash

    @property
    def semantic_shape(self) -> tuple[int, ...]:
        shape = self._collection.records[0].features.semantic_features.shape[1:]
        return tuple(int(value) for value in shape)

    @property
    def context_dim(self) -> int:
        return int(self._collection.records[0].features.context_keys.shape[1])

    @property
    def max_episode_tokens(self) -> int:
        # Four DINO spatial tokens for the immutable initial anchor and for
        # each bounded factual recent chunk boundary.
        return int(self.semantic_shape[0]) * (1 + self._recent_event_capacity)

    @property
    def action_dim(self) -> int:
        return int(self._collection.records[0].features.model_actions.shape[1])

    @property
    def action_summary_dim(self) -> int:
        return 3 * self.action_dim + 4

    @property
    def action_summary_capacity(self) -> int:
        return self._action_summary_capacity

    @property
    def action_summary_chunk_size(self) -> int:
        return self._action_summary_chunk_size

    def _features(self, query_id: QueryId):
        if not isinstance(query_id, QueryId):
            raise TypeError("query_id must be QueryId")
        try:
            loaded = self._records[
                (
                    query_id.dataset_id,
                    query_id.dataset_index,
                    query_id.episode_index,
                )
            ]
        except KeyError as exc:
            raise RetrospectiveFeatureStoreError(
                f"no feature episode for query {query_id!r}"
            ) from exc
        return loaded.features

    def query_payload(
        self, query_id: QueryId
    ) -> dict[str, np.ndarray | np.bool_]:
        features = self._features(query_id)
        frame = int(query_id.frame_index)
        if frame < 0 or frame >= features.semantic_features.shape[0]:
            raise RetrospectiveFeatureStoreError(
                f"query frame lies outside feature episode: {query_id!r}"
            )
        current = np.asarray(features.semantic_features[frame], dtype=np.float32)
        current_context = np.asarray(features.context_keys[frame], dtype=np.float32)
        future_index = frame + self._action_horizon
        future_valid = future_index < features.semantic_features.shape[0]
        if future_valid:
            future = np.asarray(
                features.semantic_features[future_index], dtype=np.float32
            )
            target_effect = future - current
        else:
            future = np.zeros_like(current)
            target_effect = np.zeros_like(current)

        # Training uses the same factual semantics as online working memory:
        # the first observation is permanent; bounded recent snapshots precede
        # the query and never include the future target.
        token_width = int(current.shape[-1])
        memory = np.zeros(
            (self.max_episode_tokens, token_width), dtype=np.float32
        )
        memory_mask = np.zeros((self.max_episode_tokens,), dtype=np.bool_)
        # At frame zero the online model has not yet committed its first
        # factual observation, so the initial query must have empty history.
        snapshots: list[int] = [] if frame == 0 else [0]
        record_key = (
            query_id.dataset_id,
            query_id.dataset_index,
            query_id.episode_index,
        )
        # Online inference can only retain observations that existed before
        # the current replan.  Protect the latest preceding replan snapshot
        # even when it did not cross the event-write threshold, mirroring
        # OnlineRetrospectiveEpisodeMemory._snapshot_inputs exactly.
        latest_observed = max(0, frame - self._action_summary_chunk_size)
        factual_events = self._event_frames[record_key][frame]
        recent_snapshots = _compressed_event_end_frames(
            factual_events,
            features.semantic_features,
            capacity=self._recent_event_capacity,
        )
        if latest_observed > 0 and latest_observed not in recent_snapshots:
            recent_snapshots = [*recent_snapshots, latest_observed]
        recent_snapshots = sorted(set(recent_snapshots))[
            -self._recent_event_capacity :
        ]
        snapshots.extend(recent_snapshots)
        cursor = 0
        for index in snapshots:
            tokens = np.asarray(features.semantic_features[index], dtype=np.float32)
            stop = cursor + int(tokens.shape[0])
            memory[cursor:stop] = tokens
            memory_mask[cursor:stop] = True
            cursor = stop

        summaries = np.zeros(
            (self._action_summary_capacity, self.action_summary_dim),
            dtype=np.float32,
        )
        summary_mask = np.zeros(
            (self._action_summary_capacity,), dtype=np.bool_
        )
        chunks: list[np.ndarray] = []
        end = frame
        # Keep one additional predecessor while constructing the tensors.  It
        # may be evicted from the returned bounded history, but the newest
        # preview summary must still compare against every summary that was in
        # the online memory immediately before that preview was appended.
        while end > 0 and len(chunks) < self._action_summary_capacity + 1:
            start = max(0, end - self._action_summary_chunk_size)
            chunk = np.asarray(features.model_actions[start:end], dtype=np.float32)
            if chunk.shape[0] > 0:
                chunks.append(chunk)
            end = start
        chunks.reverse()
        previous_signatures: list[np.ndarray] = []
        summary_vectors: list[np.ndarray] = []
        for chunk in chunks:
            vector, signature = _action_summary_vector(
                chunk,
                action_horizon=self._action_horizon,
                gripper_indices=self._gripper_indices,
                previous_signatures=tuple(previous_signatures),
            )
            summary_vectors.append(vector)
            previous_signatures.append(signature)
            previous_signatures = previous_signatures[
                -self._action_summary_capacity :
            ]
        summary_vectors = summary_vectors[-self._action_summary_capacity :]
        offset = self._action_summary_capacity - len(summary_vectors)
        for position, vector in enumerate(summary_vectors, start=offset):
            summaries[position] = vector
            summary_mask[position] = True

        return {
            WARM_CURRENT_CONTEXT: np.ascontiguousarray(current_context),
            WARM_CURRENT_SEMANTIC: np.ascontiguousarray(current),
            WARM_FUTURE_SEMANTIC: np.ascontiguousarray(future),
            WARM_TARGET_EFFECT: np.ascontiguousarray(target_effect),
            WARM_FUTURE_VALID: np.bool_(future_valid),
            WARM_EPISODE_TOKENS: np.ascontiguousarray(memory),
            WARM_EPISODE_MASK: np.ascontiguousarray(memory_mask),
            WARM_EPISODE_ACTION_SUMMARIES: np.ascontiguousarray(summaries),
            WARM_EPISODE_ACTION_MASK: np.ascontiguousarray(summary_mask),
        }


def _gripper_timing(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Return close/open phase plus validity flags for every candidate."""

    if values.ndim != 2 or valid.ndim != 1 or values.shape[0] != valid.shape[0]:
        raise RetrospectiveFeatureStoreError(
            "candidate gripper/mask shapes are inconsistent"
        )
    output = np.zeros((values.shape[0], 4), dtype=np.float32)
    denominator = max(values.shape[1] - 1, 1)
    for row in np.flatnonzero(valid).tolist():
        delta = np.diff(values[row].astype(np.float64, copy=False))
        close = np.flatnonzero(delta < 0.0)
        opened = np.flatnonzero(delta > 0.0)
        if close.size:
            output[row, 0] = float(close[0] + 1) / denominator
            output[row, 1] = 1.0
        if opened.size:
            output[row, 2] = float(opened[0] + 1) / denominator
            output[row, 3] = 1.0
    return output


class RuntimeRetrospectiveDatasetAdapter(torch.utils.data.Dataset):
    """Add complete WARM evidence to an M2 candidate-adapted dataset."""

    def __init__(
        self,
        dataset: RuntimeCandidateDatasetAdapter,
        feature_store: RetrospectiveFeatureStore,
    ) -> None:
        if not isinstance(dataset, RuntimeCandidateDatasetAdapter):
            raise TypeError(
                "dataset must first be RuntimeCandidateDatasetAdapter"
            )
        if not isinstance(feature_store, RetrospectiveFeatureStore):
            raise TypeError("feature_store must be RetrospectiveFeatureStore")
        if feature_store.collection_sha256 != dataset.resolver.query_corpus_sha256:
            raise RetrospectiveFeatureStoreError(
                "feature store and candidate resolver bind different query corpora"
            )
        if feature_store.semantic_shape != dataset.resolver.semantic_effect_shape:
            raise RetrospectiveFeatureStoreError(
                "feature semantic shape does not match event-bank effect payload"
            )
        if feature_store.context_dim != dataset.resolver.context_dim:
            raise RetrospectiveFeatureStoreError(
                "feature context width does not match event-bank context keys"
            )
        if feature_store.action_dim != dataset.resolver.action_space.action_dim:
            raise RetrospectiveFeatureStoreError(
                "feature action dimension does not match event-bank action contract"
            )
        self._dataset = dataset
        self._feature_store = feature_store
        self.lerobot_dataset = dataset.lerobot_dataset

    @property
    def resolver(self) -> RuntimeCandidateResolver:
        return self._dataset.resolver

    @property
    def feature_store(self) -> RetrospectiveFeatureStore:
        return self._feature_store

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._dataset, name)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self._dataset[index])
        collisions = sorted(set(sample).intersection(WARM_RETROSPECTIVE_FIELDS))
        if collisions:
            raise RuntimeCandidateDatasetContractError(
                f"sample already contains WARM retrospective fields: {collisions}"
            )
        record, frame_index = self._dataset._episode_record(sample)
        query_id = QueryId(
            record.dataset_id,
            record.dataset_index,
            record.episode_index,
            frame_index,
        )
        padded_tail = self._dataset._is_explicit_padded_tail(
            sample, record=record, frame_index=frame_index
        )
        resolved = self.resolver.resolve(query_id, allow_missing=padded_tail)
        valid = np.asarray(resolved.mask, dtype=np.bool_)
        effect_pre = self.resolver.gather_payload(resolved, EFFECT_PRE)
        effect_post = self.resolver.gather_payload(resolved, EFFECT_POST)
        gripper = self.resolver.gather_payload(
            resolved, OBSERVED_GRIPPER_STATE
        )
        payloads: dict[str, np.ndarray | np.bool_] = {
            WARM_CANDIDATE_CONTEXT: self.resolver.gather_context_keys(resolved),
            WARM_CANDIDATE_EFFECT_PRE: effect_pre,
            WARM_CANDIDATE_EFFECT_POST: effect_post,
            WARM_CANDIDATE_EFFECT_DELTA: np.ascontiguousarray(
                effect_post - effect_pre, dtype=np.float32
            ),
            WARM_CANDIDATE_START_PROPRIO: self.resolver.gather_payload(
                resolved, START_PROPRIO
            ),
            WARM_CANDIDATE_GRIPPER: gripper,
            WARM_CANDIDATE_TIMING: _gripper_timing(gripper, valid),
            # V1 stores exemplars rather than learned clusters.  Every factual
            # exemplar therefore has support one; event_score remains factual
            # change metadata and is not mislabeled as cluster support.
            WARM_CANDIDATE_SUPPORT: valid.astype(np.float32),
        }
        payloads.update(self._feature_store.query_payload(query_id))
        for key, value in payloads.items():
            array = np.array(value, copy=True, order="C")
            sample[key] = torch.from_numpy(array) if array.ndim else torch.tensor(array.item())
        if not torch.equal(
            sample[WARM_CANDIDATE_MASK],
            torch.from_numpy(np.array(valid, copy=True)),
        ):
            raise RuntimeCandidateDatasetContractError(
                "retrospective and M2 candidate masks disagree"
            )
        return sample


__all__ = [
    "RetrospectiveFeatureStore",
    "RetrospectiveFeatureStoreError",
    "RuntimeRetrospectiveDatasetAdapter",
    "WARM_CANDIDATE_CONTEXT",
    "WARM_CANDIDATE_EFFECT_DELTA",
    "WARM_CANDIDATE_EFFECT_POST",
    "WARM_CANDIDATE_EFFECT_PRE",
    "WARM_CANDIDATE_GRIPPER",
    "WARM_CANDIDATE_START_PROPRIO",
    "WARM_CANDIDATE_SUPPORT",
    "WARM_CANDIDATE_TIMING",
    "WARM_CURRENT_CONTEXT",
    "WARM_CURRENT_SEMANTIC",
    "WARM_EPISODE_MASK",
    "WARM_EPISODE_ACTION_MASK",
    "WARM_EPISODE_ACTION_SUMMARIES",
    "WARM_EPISODE_TOKENS",
    "WARM_FUTURE_SEMANTIC",
    "WARM_FUTURE_VALID",
    "WARM_RETROSPECTIVE_FIELDS",
    "WARM_TARGET_EFFECT",
    "collect_feature_payloads",
    "collect_feature_payloads_from_list",
]
