"""Contract-bound frozen-DINO retrieval for online WARM rollouts.

This module is the online counterpart of the immutable M1 feature/cache
pipeline.  It deliberately has no module-level Torch, Transformers, Hydra, or
LIBERO imports.  A formal fixed-source rollout loads the exact train event bank
and the exact processor/DINO contracts, converts one factual observation with
the same validation preprocessing used by M1, and performs deterministic
full-bank exact-cosine retrieval.  In the task-conditioned M1 recipe, task
identity is already encoded in the context key; it is not a second hard
eligibility filter over ``TASK_INDEX``.

The public output is :class:`BoundOnlineStep`.  It owns immutable NumPy arrays,
content hashes, stable event identities, and a process-local capability.  The
capability is intentionally not serializable: a model must retain the
``FrozenDinoOnlineRetriever`` and call :meth:`validate_bound_step` rather than
accepting caller-supplied action tensors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from numbers import Integral, Real
from pathlib import Path
import re
import threading
import time
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from fastwam.datasets.lerobot.audit import load_audit_report
from fastwam.datasets.lerobot.episode_catalog import EpisodeCatalog
from fastwam.models.warm.online_contract import (
    ONLINE_RETRIEVAL_IMPLEMENTATION,
    WarmOnlineRunContract,
)
from fastwam.models.warm.source_contract import WarmSourceRunContract

from .action_contract import (
    ActionSpaceContractError,
    validate_action_space_contract,
)
from .bank_contract import validate_warm_v1_bank
from .candidate_cache import QueryId, canonical_event_bank_content_hash
from .event_bank import MANIFEST_FILENAME as BANK_MANIFEST_FILENAME, EventBank
from .feature_precompute import build_m1_context_keys
from .manifest import (
    sha256_array,
    sha256_canonical_json,
    sha256_file,
    sha256_path_tree,
)
from .payload_names import MODEL_SPACE_ACTION, TASK_INDEX
from .schema import EventId


INVALID_BANK_ROW = -1
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ENCODER_SCHEMA = "warm.feature-encoder"
_ENCODER_VERSION = 2
_CAMERA_SCHEMA = "warm.camera-layout"
_CAMERA_VERSION = 1
_MAPPING_FIELDS = frozenset({"file_sha256", "contract"})
_TELEMETRY_FIELDS = (
    "preprocess_s",
    "dino_s",
    "search_s",
    "gather_s",
    "total_s",
)
_MAX_TORCH_SEED = (1 << 63) - 1
_ARTIFACT_VERIFICATION_CAPABILITY = object()


class OnlineRetrievalError(ValueError):
    """Base class for online retrieval contract failures."""


class OnlineArtifactContractError(OnlineRetrievalError):
    """Raised when online artifacts do not describe one immutable snapshot."""


class OnlineEpisodeStateError(OnlineRetrievalError):
    """Raised when QueryId issuance/consumption is ambiguous or non-monotonic."""


class OnlineBoundStepError(OnlineRetrievalError):
    """Raised when a bound step was forged, mutated, or used by another bridge."""


def _digest(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise OnlineArtifactContractError(
            f"{field_name} must be a lowercase SHA-256 digest"
        )
    return value


def _nonnegative_int(value: object, field_name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Integral, np.integer)
    ):
        raise TypeError(f"{field_name} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return result


def _positive_int(value: object, field_name: str) -> int:
    result = _nonnegative_int(value, field_name)
    if result == 0:
        raise ValueError(f"{field_name} must be positive")
    return result


def _normalized_text(value: object, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or "\x00" in value
    ):
        raise OnlineRetrievalError(
            f"{field_name} must be a normalized non-empty string"
        )
    return value


def _text_sha256(value: str, domain: bytes) -> str:
    return hashlib.sha256(domain + b"\0" + value.encode("utf-8")).hexdigest()


def _readonly_array(
    value: Any,
    *,
    dtype: np.dtype[Any],
    field_name: str,
    rank: int,
) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != dtype or array.ndim != rank:
        raise OnlineBoundStepError(
            f"{field_name} must have dtype {dtype} and rank {rank}, "
            f"got {array.dtype}/{array.shape}"
        )
    if any(int(size) <= 0 for size in array.shape):
        raise OnlineBoundStepError(f"{field_name} must have non-empty dimensions")
    if array.dtype.kind == "f" and not np.isfinite(array).all():
        raise OnlineBoundStepError(f"{field_name} must contain finite values")
    contiguous = np.ascontiguousarray(array, dtype=dtype)
    # Backing by immutable bytes prevents callers from re-enabling write access.
    frozen = np.frombuffer(contiguous.tobytes(order="C"), dtype=dtype)
    return frozen.reshape(contiguous.shape)


def _optional_proprio(value: Any | None) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise OnlineBoundStepError("proprio must be numeric")
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or array.shape[0] != 1 or array.shape[1] <= 0:
        raise OnlineBoundStepError("proprio must have shape [D] or [1,D]")
    if not np.isfinite(array).all():
        raise OnlineBoundStepError("proprio must be finite")
    contiguous = np.ascontiguousarray(array, dtype=np.float32)
    frozen = np.frombuffer(contiguous.tobytes(order="C"), dtype=np.float32)
    return frozen.reshape(contiguous.shape)


def _immutable_digest_mapping(
    value: Mapping[str, str], field_name: str
) -> Mapping[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise OnlineBoundStepError(f"{field_name} must be a non-empty mapping")
    result: dict[str, str] = {}
    for key, digest in value.items():
        if not isinstance(key, str) or not key or key.strip() != key:
            raise OnlineBoundStepError(
                f"{field_name} keys must be normalized non-empty strings"
            )
        if key in result:
            raise OnlineBoundStepError(f"{field_name} keys must be unique")
        result[key] = _digest(digest, f"{field_name}[{key!r}]")
    return MappingProxyType(result)


def _immutable_telemetry(value: Mapping[str, Any]) -> Mapping[str, float]:
    if not isinstance(value, Mapping) or tuple(value) != _TELEMETRY_FIELDS:
        raise OnlineBoundStepError(
            f"telemetry fields must be exactly {_TELEMETRY_FIELDS!r} in order"
        )
    result: dict[str, float] = {}
    for key in _TELEMETRY_FIELDS:
        raw = value[key]
        if isinstance(raw, (bool, np.bool_)) or not isinstance(raw, Real):
            raise OnlineBoundStepError(f"telemetry[{key!r}] must be numeric")
        number = float(raw)
        if not np.isfinite(number) or number < 0.0:
            raise OnlineBoundStepError(
                f"telemetry[{key!r}] must be finite and non-negative"
            )
        result[key] = number
    if result["total_s"] + 1e-12 < sum(
        result[key] for key in _TELEMETRY_FIELDS[:-1]
    ):
        raise OnlineBoundStepError(
            "telemetry total_s cannot be smaller than its measured stages"
        )
    return MappingProxyType(result)


def _query_payload(query_id: QueryId) -> dict[str, Any]:
    return {
        "dataset_id": query_id.dataset_id,
        "dataset_index": query_id.dataset_index,
        "episode_index": query_id.episode_index,
        "frame_index": query_id.frame_index,
    }


def online_query_dataset_id(contract: WarmOnlineRunContract) -> str:
    """Return the source-policy-independent QueryId namespace.

    Fixed and Gaussian-null contracts differ by policy and therefore by their
    own contract hash.  The explicit ``evaluation_namespace_sha256`` is shared
    across paired runs and is the only digest admitted into QueryId identity.
    """

    if not isinstance(contract, WarmOnlineRunContract):
        raise TypeError("contract must be WarmOnlineRunContract")
    return (
        f"warm-online/libero/{contract.task_suite}/"
        f"{contract.evaluation_namespace_sha256}"
    )


def make_online_query_id(
    contract: WarmOnlineRunContract,
    episode_index: int,
    frame_index: int,
) -> QueryId:
    """Construct one deterministic rollout QueryId without loading memory."""

    if not isinstance(contract, WarmOnlineRunContract):
        raise TypeError("contract must be WarmOnlineRunContract")
    return QueryId(
        online_query_dataset_id(contract),
        contract.task_id,
        _nonnegative_int(episode_index, "episode_index"),
        _nonnegative_int(frame_index, "frame_index"),
    )


def derive_online_query_seed(
    root_seed: int,
    query_id: QueryId,
    evaluation_namespace_sha256: str,
) -> int:
    """Derive a paired per-query Torch seed independent of source policy."""

    seed = _nonnegative_int(root_seed, "root_seed")
    if not isinstance(query_id, QueryId):
        raise TypeError("query_id must be QueryId")
    namespace = _digest(
        evaluation_namespace_sha256, "evaluation_namespace_sha256"
    )
    encoded = json.dumps(
        {
            "root_seed": seed,
            "query_id": _query_payload(query_id),
            "evaluation_namespace_sha256": namespace,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256(b"warm.online-query-seed.v1\0" + encoded).digest()
    return int.from_bytes(digest[:8], "big") & _MAX_TORCH_SEED


def _step_digest(step: "BoundOnlineStep") -> str:
    event_rows: list[dict[str, Any] | None] = []
    for event_id in step.event_ids:
        if event_id is None:
            event_rows.append(None)
        else:
            event_rows.append(
                {
                    "dataset_id": event_id.dataset_id,
                    "dataset_index": event_id.dataset_index,
                    "episode_index": event_id.episode_index,
                    "start_frame": event_id.start_frame,
                }
            )
    metadata = {
        "query_id": _query_payload(step.query_id),
        "online_contract_sha256": step.online_contract_sha256,
        "training_run_contract_sha256": step.training_run_contract_sha256,
        "validation_run_contract_sha256": step.validation_run_contract_sha256,
        "bank_manifest_sha256": step.bank_manifest_sha256,
        "bank_content_sha256": step.bank_content_sha256,
        "encoder_contract_sha256": step.encoder_contract_sha256,
        "camera_contract_sha256": step.camera_contract_sha256,
        "normalization_stats_sha256": step.normalization_stats_sha256,
        "action_space_contract_sha256": step.action_space_contract_sha256,
        "catalog_sha256": step.catalog_sha256,
        "audit_sha256": step.audit_sha256,
        "task_description": step.task_description,
        "task_index": step.task_index,
        "prompt_sha256": step.prompt_sha256,
        "proprio_sha256": step.proprio_sha256,
        "raw_camera_sha256": dict(step.raw_camera_sha256),
        "processed_camera_sha256": dict(step.processed_camera_sha256),
        "model_input_sha256": step.model_input_sha256,
        "context_key_sha256": step.context_key_sha256,
        "event_ids": event_rows,
        "bank_rows_sha256": sha256_array(step.bank_rows),
        "cosine_scores_sha256": sha256_array(step.cosine_scores),
        "candidate_valid_mask_sha256": sha256_array(
            step.candidate_valid_mask
        ),
        "candidate_payload_sha256": step.candidate_payload_sha256,
        "derived_seed": step.derived_seed,
        "telemetry": dict(step.telemetry),
    }
    encoded = json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(b"warm.bound-online-step.v1\0" + encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class BoundOnlineStep:
    """One immutable, capability-bound online retrieval result."""

    query_id: QueryId
    online_contract_sha256: str
    training_run_contract_sha256: str
    validation_run_contract_sha256: str
    bank_manifest_sha256: str
    bank_content_sha256: str
    encoder_contract_sha256: str
    camera_contract_sha256: str
    normalization_stats_sha256: str
    action_space_contract_sha256: str
    catalog_sha256: str
    audit_sha256: str
    task_description: str
    task_index: int
    prompt: str
    proprio: np.ndarray | None
    raw_camera_sha256: Mapping[str, str]
    processed_camera_sha256: Mapping[str, str]
    model_input: np.ndarray
    context_key: np.ndarray
    event_ids: tuple[EventId | None, ...]
    bank_rows: np.ndarray
    cosine_scores: np.ndarray
    candidate_valid_mask: np.ndarray
    candidate_means: np.ndarray
    derived_seed: int
    telemetry: Mapping[str, float]
    _capability: object = field(repr=False, compare=False)
    prompt_sha256: str = field(init=False)
    proprio_sha256: str | None = field(init=False)
    model_input_sha256: str = field(init=False)
    context_key_sha256: str = field(init=False)
    candidate_payload_sha256: str = field(init=False)
    step_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.query_id, QueryId):
            raise TypeError("query_id must be QueryId")
        if self._capability is None:
            raise OnlineBoundStepError("bound step capability must not be None")
        for field_name in (
            "online_contract_sha256",
            "training_run_contract_sha256",
            "validation_run_contract_sha256",
            "bank_manifest_sha256",
            "bank_content_sha256",
            "encoder_contract_sha256",
            "camera_contract_sha256",
            "normalization_stats_sha256",
            "action_space_contract_sha256",
            "catalog_sha256",
            "audit_sha256",
        ):
            object.__setattr__(
                self, field_name, _digest(getattr(self, field_name), field_name)
            )
        task = _normalized_text(self.task_description, "task_description")
        prompt = _normalized_text(self.prompt, "prompt")
        task_index = _nonnegative_int(self.task_index, "task_index")
        proprio = _optional_proprio(self.proprio)
        model_input = _readonly_array(
            self.model_input,
            dtype=np.dtype(np.float32),
            field_name="model_input",
            rank=4,
        )
        if model_input.shape[0] != 1 or model_input.shape[1] != 3:
            raise OnlineBoundStepError(
                "model_input must have shape [1,3,H,W]"
            )
        if np.any(model_input < -1.0) or np.any(model_input > 1.0):
            raise OnlineBoundStepError("model_input must lie in [-1,1]")
        context_key = _readonly_array(
            self.context_key,
            dtype=np.dtype(np.float32),
            field_name="context_key",
            rank=1,
        )
        rows = _readonly_array(
            self.bank_rows,
            dtype=np.dtype(np.int64),
            field_name="bank_rows",
            rank=1,
        )
        scores = _readonly_array(
            self.cosine_scores,
            dtype=np.dtype(np.float64),
            field_name="cosine_scores",
            rank=1,
        )
        valid = _readonly_array(
            self.candidate_valid_mask,
            dtype=np.dtype(np.bool_),
            field_name="candidate_valid_mask",
            rank=1,
        )
        means = _readonly_array(
            self.candidate_means,
            dtype=np.dtype(np.float32),
            field_name="candidate_means",
            rank=3,
        )
        if rows.shape != scores.shape or rows.shape != valid.shape:
            raise OnlineBoundStepError(
                "bank_rows, cosine_scores, and candidate_valid_mask must share [K]"
            )
        if means.shape[0] != rows.size:
            raise OnlineBoundStepError(
                "candidate_means leading dimension must equal K"
            )
        events = tuple(self.event_ids)
        if len(events) != rows.size:
            raise OnlineBoundStepError("event_ids must contain exactly K entries")
        valid_count = int(valid.sum())
        if valid_count and not bool(valid[:valid_count].all()):
            raise OnlineBoundStepError("valid candidates must form a prefix")
        if bool(valid[valid_count:].any()):
            raise OnlineBoundStepError("valid candidates must form a prefix")
        if np.any(rows[valid] < 0):
            raise OnlineBoundStepError(
                "valid candidates must have non-negative bank rows"
            )
        if np.any(rows[~valid] != INVALID_BANK_ROW):
            raise OnlineBoundStepError(
                f"invalid candidates must use bank row {INVALID_BANK_ROW}"
            )
        if np.any(scores[~valid] != 0.0):
            raise OnlineBoundStepError("invalid candidate scores must be zero")
        if np.any(means[~valid] != 0.0):
            raise OnlineBoundStepError("invalid candidate action payload must be zero")
        if valid_count > 1 and np.any(scores[: valid_count - 1] < scores[1:valid_count]):
            raise OnlineBoundStepError(
                "valid candidate scores must be non-increasing"
            )
        if np.any(scores[valid] < -1.0) or np.any(scores[valid] > 1.0):
            raise OnlineBoundStepError("cosine scores must lie in [-1,1]")
        if len(set(int(row) for row in rows[valid].tolist())) != valid_count:
            raise OnlineBoundStepError("valid candidate rows must be unique")
        for position, (is_valid, event_id) in enumerate(
            zip(valid, events, strict=True)
        ):
            if bool(is_valid) and not isinstance(event_id, EventId):
                raise OnlineBoundStepError(
                    f"valid candidate slot {position} must contain EventId"
                )
            if not bool(is_valid) and event_id is not None:
                raise OnlineBoundStepError(
                    f"invalid candidate slot {position} must contain None"
                )
        seed = _nonnegative_int(self.derived_seed, "derived_seed")
        if seed > _MAX_TORCH_SEED:
            raise OnlineBoundStepError("derived_seed exceeds signed 63-bit range")

        object.__setattr__(self, "task_description", task)
        object.__setattr__(self, "task_index", task_index)
        object.__setattr__(self, "prompt", prompt)
        object.__setattr__(self, "proprio", proprio)
        object.__setattr__(
            self,
            "raw_camera_sha256",
            _immutable_digest_mapping(
                self.raw_camera_sha256, "raw_camera_sha256"
            ),
        )
        object.__setattr__(
            self,
            "processed_camera_sha256",
            _immutable_digest_mapping(
                self.processed_camera_sha256, "processed_camera_sha256"
            ),
        )
        object.__setattr__(self, "model_input", model_input)
        object.__setattr__(self, "context_key", context_key)
        object.__setattr__(self, "event_ids", events)
        object.__setattr__(self, "bank_rows", rows)
        object.__setattr__(self, "cosine_scores", scores)
        object.__setattr__(self, "candidate_valid_mask", valid)
        object.__setattr__(self, "candidate_means", means)
        object.__setattr__(self, "derived_seed", seed)
        object.__setattr__(self, "telemetry", _immutable_telemetry(self.telemetry))
        object.__setattr__(
            self,
            "prompt_sha256",
            _text_sha256(prompt, b"warm.online-prompt.v1"),
        )
        object.__setattr__(
            self,
            "proprio_sha256",
            None if proprio is None else sha256_array(proprio),
        )
        object.__setattr__(self, "model_input_sha256", sha256_array(model_input))
        object.__setattr__(self, "context_key_sha256", sha256_array(context_key))
        object.__setattr__(
            self, "candidate_payload_sha256", sha256_array(means)
        )
        object.__setattr__(self, "step_sha256", _step_digest(self))

    @property
    def fixed_k(self) -> int:
        return int(self.bank_rows.size)

    @property
    def valid_count(self) -> int:
        return int(self.candidate_valid_mask.sum())

    @property
    def selected_event_id(self) -> EventId | None:
        return self.event_ids[0] if self.valid_count else None

    def _assert_integrity(self) -> None:
        for name, array in (
            ("model_input", self.model_input),
            ("context_key", self.context_key),
            ("bank_rows", self.bank_rows),
            ("cosine_scores", self.cosine_scores),
            ("candidate_valid_mask", self.candidate_valid_mask),
            ("candidate_means", self.candidate_means),
        ):
            if array.flags.writeable:
                raise OnlineBoundStepError(f"bound step array {name} became writable")
        if sha256_array(self.model_input) != self.model_input_sha256:
            raise OnlineBoundStepError("bound model_input content changed")
        if sha256_array(self.context_key) != self.context_key_sha256:
            raise OnlineBoundStepError("bound context_key content changed")
        if sha256_array(self.candidate_means) != self.candidate_payload_sha256:
            raise OnlineBoundStepError("bound candidate action content changed")
        if _text_sha256(self.prompt, b"warm.online-prompt.v1") != self.prompt_sha256:
            raise OnlineBoundStepError("bound prompt content changed")
        actual_proprio_sha = (
            None if self.proprio is None else sha256_array(self.proprio)
        )
        if actual_proprio_sha != self.proprio_sha256:
            raise OnlineBoundStepError("bound proprio content changed")
        if _step_digest(self) != self.step_sha256:
            raise OnlineBoundStepError("bound online step digest is invalid")


def _coerce_source_contract(
    value: WarmSourceRunContract | Mapping[str, Any]
) -> WarmSourceRunContract:
    if isinstance(value, WarmSourceRunContract):
        return value
    if isinstance(value, Mapping):
        return WarmSourceRunContract.from_dict(value)
    raise TypeError("source_run_contract must be WarmSourceRunContract or mapping")


def _coerce_online_contract(
    value: WarmOnlineRunContract | Mapping[str, Any]
) -> WarmOnlineRunContract:
    if isinstance(value, WarmOnlineRunContract):
        return value
    if isinstance(value, Mapping):
        return WarmOnlineRunContract.from_dict(value)
    raise TypeError("online_run_contract must be WarmOnlineRunContract or mapping")


def _strict_json_snapshot(path: str | Path, field_name: str) -> tuple[dict[str, Any], str]:
    source = Path(path).expanduser().resolve()
    try:
        raw = source.read_bytes()
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {constant}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise OnlineArtifactContractError(
            f"cannot read strict JSON artifact {field_name}: {source}"
        ) from exc
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise OnlineArtifactContractError(f"{field_name} must be a JSON object")
    return value, hashlib.sha256(raw).hexdigest()


def _manifest_contract(
    value: Mapping[str, Any],
    *,
    field_name: str,
    external_value: Mapping[str, Any],
    external_sha256: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _MAPPING_FIELDS:
        raise OnlineArtifactContractError(
            f"event-bank {field_name} must contain exactly {sorted(_MAPPING_FIELDS)!r}"
        )
    if value["file_sha256"] != external_sha256:
        raise OnlineArtifactContractError(
            f"{field_name} file SHA-256 does not match the event bank"
        )
    embedded = value["contract"]
    if not isinstance(embedded, Mapping):
        raise OnlineArtifactContractError(
            f"event-bank {field_name} contract must be a JSON object"
        )
    if dict(embedded) != dict(external_value):
        raise OnlineArtifactContractError(
            f"{field_name} JSON does not match the event bank"
        )
    return dict(external_value)


def _dtype_label(value: Any | None) -> str:
    if value is None:
        return "float32"
    label = str(value).strip().lower()
    if label.startswith("torch."):
        label = label[6:]
    aliases = {"float": "float32", "half": "float16"}
    return aliases.get(label, label)


def _validate_encoder_contract(
    value: Mapping[str, Any],
) -> tuple[str, tuple[str, ...], str, str]:
    if value.get("schema") != _ENCODER_SCHEMA or value.get("version") != _ENCODER_VERSION:
        raise OnlineArtifactContractError("unsupported online feature-encoder contract")
    if value.get("official_complete") is not True:
        raise OnlineArtifactContractError(
            "online retrieval requires an official complete feature encoder"
        )
    dino = value.get("dino")
    context = value.get("context")
    compute = value.get("compute")
    runtime = value.get("runtime")
    if (
        not isinstance(dino, Mapping)
        or not isinstance(context, Mapping)
        or not isinstance(compute, Mapping)
        or not isinstance(runtime, Mapping)
    ):
        raise OnlineArtifactContractError(
            "encoder contract must contain dino, context, compute, and runtime objects"
        )
    _digest(value.get("data_config_sha256"), "encoder data_config_sha256")
    compute_dtype = _dtype_label(compute.get("dtype"))
    if compute_dtype not in {"float32", "float16", "bfloat16"}:
        raise OnlineArtifactContractError(
            "encoder compute.dtype must be float32, float16, or bfloat16"
        )
    compute_device = _normalized_text(
        compute.get("device"), "encoder compute.device"
    )
    if value.get("output_dtype") != "float32":
        raise OnlineArtifactContractError(
            "online retrieval requires float32 stored DINO features"
        )
    mode = context.get("mode")
    if mode not in {"task-conditioned", "visual-only"}:
        raise OnlineArtifactContractError(
            "encoder context mode must be task-conditioned or visual-only"
        )
    vocabulary_raw = context.get("task_vocabulary")
    if not isinstance(vocabulary_raw, list) or not vocabulary_raw:
        raise OnlineArtifactContractError(
            "encoder context task_vocabulary must be a non-empty list"
        )
    vocabulary = tuple(
        _normalized_text(item, f"task_vocabulary[{index}]")
        for index, item in enumerate(vocabulary_raw)
    )
    if len(set(vocabulary)) != len(vocabulary) or vocabulary != tuple(sorted(vocabulary)):
        raise OnlineArtifactContractError(
            "encoder task_vocabulary must be unique and sorted"
        )
    energy = context.get("visual_task_energy")
    expected_energy = [0.5, 0.5] if mode == "task-conditioned" else [1.0, 0.0]
    if (
        not isinstance(energy, list)
        or len(energy) != 2
        or any(
            isinstance(item, (bool, np.bool_)) or not isinstance(item, Real)
            for item in energy
        )
        or [float(item) for item in energy] != expected_energy
    ):
        raise OnlineArtifactContractError(
            "encoder context visual_task_energy does not match the exact M1 recipe"
        )
    required_dino = {
        "model_id",
        "revision",
        "checkpoint_tree_sha256",
        "checkpoint_file_count",
        "hidden_size",
        "patch_size",
        "patch_grid_size",
        "register_token_count",
        "image_size",
        "image_mean",
        "image_std",
        "semantic_pool",
        "resize_in_encoder",
    }
    if not required_dino.issubset(dino):
        raise OnlineArtifactContractError(
            "encoder dino contract is missing required fields"
        )
    if dino["semantic_pool"] != "adaptive_avg_pool_2x2_row_major" or dino[
        "resize_in_encoder"
    ] is not False:
        raise OnlineArtifactContractError(
            "online DINO preprocessing does not match M1"
        )
    return str(mode), vocabulary, compute_dtype, compute_device


def _validate_camera_contract(
    value: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...], str, str]:
    if value.get("schema") != _CAMERA_SCHEMA or value.get("version") != _CAMERA_VERSION:
        raise OnlineArtifactContractError("unsupported online camera-layout contract")
    source_raw = value.get("source_camera_keys")
    mapping_raw = value.get("processor_camera_mapping")
    semantic_source = value.get("semantic_camera")
    concat_mode = value.get("concat_mode")
    if not isinstance(source_raw, list) or not source_raw:
        raise OnlineArtifactContractError("camera source keys must be non-empty")
    source_keys = tuple(
        _normalized_text(item, f"source_camera_keys[{index}]")
        for index, item in enumerate(source_raw)
    )
    if len(set(source_keys)) != len(source_keys):
        raise OnlineArtifactContractError("camera source keys must be unique")
    if not isinstance(mapping_raw, Mapping) or set(mapping_raw) != set(source_keys):
        raise OnlineArtifactContractError(
            "processor_camera_mapping must cover every source camera exactly"
        )
    processor_keys = tuple(
        _normalized_text(mapping_raw[key], f"processor_camera_mapping[{key!r}]")
        for key in source_keys
    )
    if len(set(processor_keys)) != len(processor_keys):
        raise OnlineArtifactContractError("processor camera keys must be unique")
    if semantic_source not in source_keys:
        raise OnlineArtifactContractError("semantic_camera is not a source camera")
    if concat_mode != "horizontal":
        raise OnlineArtifactContractError(
            "official M1 LIBERO online retrieval requires horizontal camera concat"
        )
    if value.get("per_camera_size") != [224, 224]:
        raise OnlineArtifactContractError("online cameras must use M1 size [224,224]")
    if value.get("decoded_range") != [0.0, 1.0] or value.get(
        "vae_model_range"
    ) != [-1.0, 1.0]:
        raise OnlineArtifactContractError("camera value ranges do not match M1")
    if value.get("baseline_quantization") != "validated_0_1_times_255_to_uint8":
        raise OnlineArtifactContractError("camera quantization does not match M1")
    semantic_processor = processor_keys[source_keys.index(str(semantic_source))]
    return source_keys, processor_keys, semantic_processor, str(concat_mode)


def validate_online_encoder_contract(
    value: Mapping[str, Any],
) -> tuple[str, tuple[str, ...], str, str]:
    """Validate and summarize the exact M1 encoder recipe used online."""

    return _validate_encoder_contract(value)


def validate_online_camera_contract(
    value: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...], str, str]:
    """Validate and summarize the exact M1 LIBERO camera recipe."""

    return _validate_camera_contract(value)


def _validate_processor(processor: Any, processor_keys: tuple[str, ...]) -> None:
    try:
        if processor.is_train is not False:
            raise OnlineArtifactContractError("online processor must be in eval mode")
        if int(processor.num_output_cameras) != len(processor_keys):
            raise OnlineArtifactContractError(
                "processor output-camera count does not match camera contract"
            )
        images = tuple(processor.shape_meta["images"])
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, OnlineArtifactContractError):
            raise
        raise OnlineArtifactContractError(
            "processor lacks required eval/camera metadata"
        ) from exc
    signature = tuple(
        (str(item.get("key")), tuple(int(v) for v in item.get("shape", ())))
        for item in images
    )
    expected = tuple((key, (3, 224, 224)) for key in processor_keys)
    if signature != expected:
        raise OnlineArtifactContractError(
            f"processor camera metadata {signature!r} != {expected!r}"
        )
    # Accessing this property proves that a normalizer was installed.  Numeric
    # identity is separately bound to the exact stats bytes and action contract.
    try:
        processor.normalizer
    except (AttributeError, ValueError) as exc:
        raise OnlineArtifactContractError(
            "processor must have its contract-bound normalizer installed"
        ) from exc


def _validate_dino_instance(
    dino: Any,
    contract: Mapping[str, Any],
    *,
    expected_compute_dtype: str,
    expected_compute_device: str,
) -> None:
    expected_pairs = (
        ("model_id", str(contract["model_id"])),
        ("revision", str(contract["revision"]).lower()),
        ("hidden_size", int(contract["hidden_size"])),
        ("patch_size", tuple(int(v) for v in contract["patch_size"])),
        ("patch_grid_size", tuple(int(v) for v in contract["patch_grid_size"])),
        ("register_token_count", int(contract["register_token_count"])),
        ("image_size", tuple(int(v) for v in contract["image_size"])),
        ("image_mean", tuple(float(v) for v in contract["image_mean"])),
        ("image_std", tuple(float(v) for v in contract["image_std"])),
    )
    for field_name, expected in expected_pairs:
        actual = getattr(dino, field_name, None)
        if isinstance(expected, tuple) and actual is not None:
            actual = tuple(actual)
        if actual != expected:
            raise OnlineArtifactContractError(
                f"DINO {field_name} does not match encoder contract: "
                f"{actual!r} != {expected!r}"
            )
    if not callable(getattr(dino, "encode", None)):
        raise OnlineArtifactContractError("DINO encoder must expose encode()")
    if getattr(dino, "compute_dtype", None) != expected_compute_dtype:
        raise OnlineArtifactContractError(
            "DINO compute dtype does not match the encoder contract: "
            f"{getattr(dino, 'compute_dtype', None)!r} != "
            f"{expected_compute_dtype!r}"
        )
    if getattr(dino, "compute_device", None) != expected_compute_device:
        raise OnlineArtifactContractError(
            "DINO compute device does not match the encoder contract: "
            f"{getattr(dino, 'compute_device', None)!r} != "
            f"{expected_compute_device!r}"
        )


def _external_array(value: Any, field_name: str) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    try:
        detached = value.detach()
        # bfloat16 has no NumPy dtype; validation intentionally compares the
        # factual float32 model input retained in BoundOnlineStep.
        try:
            detached = detached.float()
        except (AttributeError, TypeError, RuntimeError):
            pass
        return detached.cpu().numpy()
    except (AttributeError, TypeError, RuntimeError) as exc:
        raise OnlineBoundStepError(
            f"{field_name} must be a NumPy array or CPU-convertible tensor"
        ) from exc


class FrozenDinoOnlineRetriever:
    """Stateful, contract-bound online frozen-DINO retrieval bridge."""

    def __init__(
        self,
        *,
        bank: EventBank,
        source_run_contract: WarmSourceRunContract,
        online_run_contract: WarmOnlineRunContract,
        image_adapter: Any,
        dino_encoder: Any,
        semantic_processor_camera: str,
        processor_camera_keys: tuple[str, ...],
        context_mode: str,
        task_vocabulary: tuple[str, ...],
        dino_batch_size: int,
        _artifact_verification_capability: object,
    ) -> None:
        if _artifact_verification_capability is not _ARTIFACT_VERIFICATION_CAPABILITY:
            raise OnlineArtifactContractError(
                "FrozenDinoOnlineRetriever must be constructed by from_artifacts()"
            )
        self._bank = bank
        self._source_run_contract = source_run_contract
        self._online_run_contract = online_run_contract
        self._image_adapter = image_adapter
        self._dino = dino_encoder
        self._semantic_processor_camera = semantic_processor_camera
        self._processor_camera_keys = processor_camera_keys
        self._context_mode = context_mode
        self._task_vocabulary = task_vocabulary
        self._dino_batch_size = _positive_int(dino_batch_size, "dino_batch_size")
        self._summary = validate_warm_v1_bank(
            bank,
            expected_action_horizon=online_run_contract.action_horizon,
            expected_action_dim=online_run_contract.action_dim,
        )
        # TASK_INDEX is validated as immutable bank metadata, but it must not
        # filter retrieval.  M1 exact_cosine_v1 searches the complete bank and
        # carries task identity inside each task-conditioned context key.
        self._task_indices = bank.payload(TASK_INDEX)
        if np.any(self._task_indices < 0) or np.any(
            self._task_indices >= len(task_vocabulary)
        ):
            raise OnlineArtifactContractError(
                "event bank contains task indices outside encoder vocabulary"
            )
        self._capability = object()
        self._lock = threading.RLock()
        self._active_episode_index: int | None = None
        self._last_frame_index: int | None = None
        self._seen_episode_indices: set[int] = set()
        self._issued: set[QueryId] = set()
        self._inflight: set[QueryId] = set()
        self._consumed: set[QueryId] = set()
        # Retrieval and model consumption are intentionally distinct.  A
        # successfully retrieved step is registered by digest, then may be
        # admitted to the policy exactly once.  This prevents replaying a
        # stale capability later in the same episode.
        self._delivered_step_digests: dict[QueryId, str] = {}
        self._model_consumed: set[QueryId] = set()

    @classmethod
    def from_artifacts(
        cls,
        bank_directory: str | Path,
        *,
        source_run_contract: WarmSourceRunContract | Mapping[str, Any],
        online_run_contract: WarmOnlineRunContract | Mapping[str, Any],
        normalizer_contract_path: str | Path,
        encoder_contract_path: str | Path,
        camera_contract_path: str | Path,
        normalization_stats_path: str | Path,
        catalog_path: str | Path,
        audit_report_path: str | Path,
        dino_checkpoint_path: str | Path,
        processor: Any,
        dino_encoder: Any | None = None,
        dino_device: str = "cuda",
        dino_torch_dtype: Any | None = None,
        dino_batch_size: int = 1,
        tensor_factory: Callable[[np.ndarray], Any] | None = None,
    ) -> "FrozenDinoOnlineRetriever":
        source_contract = _coerce_source_contract(source_run_contract)
        online_contract = _coerce_online_contract(online_run_contract)
        if online_contract.source_policy != "fixed_context_top1":
            raise OnlineArtifactContractError(
                "FrozenDinoOnlineRetriever may only be constructed for "
                "fixed_context_top1; Gaussian-null must not load memory"
            )
        if online_contract.retrieval_implementation != ONLINE_RETRIEVAL_IMPLEMENTATION:
            raise OnlineArtifactContractError("unsupported online retrieval recipe")
        if online_contract.training_run_contract_sha256 != source_contract.sha256:
            raise OnlineArtifactContractError(
                "online contract does not bind the supplied training run contract"
            )
        if source_contract.query_split != "train":
            raise OnlineArtifactContractError(
                "online retrieval requires a train-bound source run contract"
            )
        paired_fields = (
            ("bank_manifest_sha256", source_contract.bank_manifest_sha256),
            ("bank_content_sha256", source_contract.bank_content_sha256),
            ("normalization_stats_sha256", source_contract.normalization_stats_sha256),
            ("action_space_contract_sha256", source_contract.action_space_contract_sha256),
            ("catalog_sha256", source_contract.catalog_sha256),
            ("audit_sha256", source_contract.audit_sha256),
            ("action_horizon", source_contract.action_horizon),
            ("action_dim", source_contract.action_dim),
        )
        for field_name, expected in paired_fields:
            if getattr(online_contract, field_name) != expected:
                raise OnlineArtifactContractError(
                    f"online/source contract mismatch for {field_name}"
                )

        bank_directory = Path(bank_directory).expanduser().resolve()
        bank_manifest_path = bank_directory / BANK_MANIFEST_FILENAME
        bank_manifest_before = sha256_file(bank_manifest_path)
        if bank_manifest_before != online_contract.bank_manifest_sha256:
            raise OnlineArtifactContractError(
                "event-bank manifest hash does not match online contract"
            )
        bank = EventBank.load(bank_directory)
        if bank.manifest is None:
            raise OnlineArtifactContractError("loaded event bank has no manifest")
        bank_content = canonical_event_bank_content_hash(bank.manifest.content_hashes)
        if bank_content != online_contract.bank_content_sha256:
            raise OnlineArtifactContractError(
                "event-bank content hash does not match online contract"
            )
        summary = validate_warm_v1_bank(
            bank,
            expected_action_horizon=online_contract.action_horizon,
            expected_action_dim=online_contract.action_dim,
        )

        normalizer_json, normalizer_sha = _strict_json_snapshot(
            normalizer_contract_path, "normalizer_contract"
        )
        encoder_json, encoder_sha = _strict_json_snapshot(
            encoder_contract_path, "encoder_contract"
        )
        camera_json, camera_sha = _strict_json_snapshot(
            camera_contract_path, "camera_contract"
        )
        if encoder_sha != online_contract.encoder_contract_sha256:
            raise OnlineArtifactContractError(
                "encoder contract hash does not match online contract"
            )
        if camera_sha != online_contract.camera_contract_sha256:
            raise OnlineArtifactContractError(
                "camera contract hash does not match online contract"
            )
        normalizer_contract = _manifest_contract(
            bank.manifest.action_normalizer,
            field_name="action_normalizer",
            external_value=normalizer_json,
            external_sha256=normalizer_sha,
        )
        encoder_contract = _manifest_contract(
            bank.manifest.encoder,
            field_name="encoder",
            external_value=encoder_json,
            external_sha256=encoder_sha,
        )
        camera_contract = _manifest_contract(
            bank.manifest.camera_layout,
            field_name="camera_layout",
            external_value=camera_json,
            external_sha256=camera_sha,
        )
        try:
            action_contract = validate_action_space_contract(normalizer_contract)
        except (ActionSpaceContractError, TypeError) as exc:
            raise OnlineArtifactContractError(
                "normalizer contract is not a valid action-space contract"
            ) from exc
        action_contract_sha = sha256_canonical_json(action_contract.to_dict())
        if action_contract_sha != online_contract.action_space_contract_sha256:
            raise OnlineArtifactContractError(
                "action-space contract hash does not match online contract"
            )
        if action_contract.action_dim != summary.action_dim:
            raise OnlineArtifactContractError(
                "action-space contract dimension does not match event bank"
            )

        stats_path = Path(normalization_stats_path).expanduser().resolve()
        stats_before = sha256_file(stats_path)
        if stats_before != online_contract.normalization_stats_sha256 or (
            action_contract.normalization_stats_sha256 != stats_before
        ):
            raise OnlineArtifactContractError(
                "normalization statistics do not match action/online contracts"
            )

        catalog_file = Path(catalog_path).expanduser().resolve()
        audit_file = Path(audit_report_path).expanduser().resolve()
        catalog_file_before = sha256_file(catalog_file)
        audit_file_before = sha256_file(audit_file)
        catalog = EpisodeCatalog.load(catalog_file)
        audit = load_audit_report(audit_file)
        if catalog.content_sha256 != online_contract.catalog_sha256:
            raise OnlineArtifactContractError(
                "episode catalog content does not match online contract"
            )
        if audit.report_sha256 != online_contract.audit_sha256:
            raise OnlineArtifactContractError(
                "audit report does not match online contract"
            )
        if audit.catalog_sha256 != catalog.content_sha256:
            raise OnlineArtifactContractError(
                "audit report is bound to a different episode catalog"
            )
        binding = bank.manifest.provenance.get("data_binding")
        if not isinstance(binding, Mapping) or binding != {
            "catalog_sha256": catalog.content_sha256,
            "audit_report_sha256": audit.report_sha256,
            "split": "train",
        }:
            raise OnlineArtifactContractError(
                "event-bank train data binding does not match catalog/audit"
            )

        (
            context_mode,
            vocabulary,
            compute_dtype,
            compute_device,
        ) = _validate_encoder_contract(encoder_contract)
        if encoder_contract.get("data_config_sha256") != (
            online_contract.m1_data_config_sha256
        ):
            raise OnlineArtifactContractError(
                "encoder data config differs from online contract"
            )
        if sha256_canonical_json(dict(encoder_contract["runtime"])) != (
            online_contract.encoder_runtime_sha256
        ):
            raise OnlineArtifactContractError(
                "encoder runtime fingerprint differs from online contract"
            )
        source_keys, processor_keys, semantic_processor, concat_mode = (
            _validate_camera_contract(camera_contract)
        )
        if audit.audited_camera_keys != source_keys:
            raise OnlineArtifactContractError(
                "camera contract does not match the audited camera order"
            )
        if online_contract.task_description not in vocabulary:
            raise OnlineArtifactContractError(
                "online task_description is absent from encoder vocabulary"
            )
        _validate_processor(processor, processor_keys)

        dino_contract = encoder_contract["dino"]
        assert isinstance(dino_contract, Mapping)
        if dino_device != compute_device:
            raise OnlineArtifactContractError(
                "requested DINO device differs from encoder compute contract: "
                f"{dino_device!r} != {compute_device!r}"
            )
        dino_path = Path(dino_checkpoint_path).expanduser().resolve()
        dino_tree_before, dino_files_before = sha256_path_tree(dino_path)
        if (
            dino_tree_before != online_contract.dino_checkpoint_tree_sha256
            or dino_files_before != online_contract.dino_checkpoint_file_count
            or dino_tree_before != dino_contract["checkpoint_tree_sha256"]
            or dino_files_before != int(dino_contract["checkpoint_file_count"])
        ):
            raise OnlineArtifactContractError(
                "DINO checkpoint tree does not match encoder/online contracts"
            )
        if dino_encoder is None:
            # Heavy imports remain behind this production-only boundary.
            from .server_feature_encoders import DinoV2FactualEncoder
            import torch

            expected_torch_dtype = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }[compute_dtype]
            if (
                dino_torch_dtype is not None
                and _dtype_label(dino_torch_dtype) != compute_dtype
            ):
                raise OnlineArtifactContractError(
                    "requested DINO dtype differs from encoder compute contract"
                )

            dino_encoder = DinoV2FactualEncoder.from_pretrained(
                dino_path,
                model_id=str(dino_contract["model_id"]),
                revision=str(dino_contract["revision"]),
                device=dino_device,
                torch_dtype=expected_torch_dtype,
                expected_image_size=tuple(dino_contract["image_size"]),
                register_token_count=int(dino_contract["register_token_count"]),
            )
        elif (
            dino_torch_dtype is not None
            and _dtype_label(dino_torch_dtype) != compute_dtype
        ):
            raise OnlineArtifactContractError(
                "requested DINO dtype differs from encoder compute contract"
            )
        _validate_dino_instance(
            dino_encoder,
            dino_contract,
            expected_compute_dtype=compute_dtype,
            expected_compute_device=compute_device,
        )

        from .server_feature_encoders import FastWAMImageAdapter

        image_adapter = FastWAMImageAdapter(
            processor,
            processor_keys,
            concat_mode,
            tensor_factory=tensor_factory,
        )

        # Close ordinary replacement windows after every loader/constructor.
        if sha256_file(bank_manifest_path) != bank_manifest_before:
            raise OnlineArtifactContractError(
                "event-bank manifest changed during online bridge construction"
            )
        payload_path = bank_directory / bank.manifest.payload_file
        if sha256_file(payload_path) != bank.manifest.content_hashes[
            bank.manifest.payload_file
        ]:
            raise OnlineArtifactContractError(
                "event-bank payload changed during online bridge construction"
            )
        for path, expected, label in (
            (Path(normalizer_contract_path).expanduser().resolve(), normalizer_sha, "normalizer"),
            (Path(encoder_contract_path).expanduser().resolve(), encoder_sha, "encoder"),
            (Path(camera_contract_path).expanduser().resolve(), camera_sha, "camera"),
            (stats_path, stats_before, "normalization statistics"),
            (catalog_file, catalog_file_before, "catalog"),
            (audit_file, audit_file_before, "audit"),
        ):
            if sha256_file(path) != expected:
                raise OnlineArtifactContractError(
                    f"{label} artifact changed during online bridge construction"
                )
        dino_tree_after, dino_files_after = sha256_path_tree(dino_path)
        if (dino_tree_after, dino_files_after) != (
            dino_tree_before,
            dino_files_before,
        ):
            raise OnlineArtifactContractError(
                "DINO checkpoint tree changed while its encoder was loading"
            )

        return cls(
            bank=bank,
            source_run_contract=source_contract,
            online_run_contract=online_contract,
            image_adapter=image_adapter,
            dino_encoder=dino_encoder,
            semantic_processor_camera=semantic_processor,
            processor_camera_keys=processor_keys,
            context_mode=context_mode,
            task_vocabulary=vocabulary,
            dino_batch_size=dino_batch_size,
            _artifact_verification_capability=_ARTIFACT_VERIFICATION_CAPABILITY,
        )

    @property
    def source_run_contract(self) -> WarmSourceRunContract:
        return self._source_run_contract

    @property
    def online_run_contract(self) -> WarmOnlineRunContract:
        return self._online_run_contract

    @property
    def artifact_verified(self) -> bool:
        """Attest that construction completed the closed-world artifact loader."""

        return True

    @property
    def task_vocabulary(self) -> tuple[str, ...]:
        return self._task_vocabulary

    def begin_episode(self, episode_index: int) -> None:
        episode = _nonnegative_int(episode_index, "episode_index")
        with self._lock:
            if self._inflight:
                raise OnlineEpisodeStateError(
                    "cannot begin an episode while a query is in flight"
                )
            outstanding = self._issued - self._consumed
            if outstanding:
                raise OnlineEpisodeStateError(
                    "cannot begin an episode with unconsumed issued QueryIds"
                )
            unconsumed_steps = set(self._delivered_step_digests) - self._model_consumed
            if unconsumed_steps:
                raise OnlineEpisodeStateError(
                    "cannot begin an episode with an unconsumed BoundOnlineStep"
                )
            if episode in self._seen_episode_indices:
                raise OnlineEpisodeStateError(
                    f"episode_index {episode} was already used by this retriever"
                )
            self._seen_episode_indices.add(episode)
            self._active_episode_index = episode
            self._last_frame_index = None
            self._issued.clear()
            self._consumed.clear()
            self._delivered_step_digests.clear()
            self._model_consumed.clear()

    def make_query_id(self, frame_index: int) -> QueryId:
        frame = _nonnegative_int(frame_index, "frame_index")
        with self._lock:
            if self._active_episode_index is None:
                raise OnlineEpisodeStateError(
                    "begin_episode() is required before issuing QueryIds"
                )
            if self._last_frame_index is not None and frame <= self._last_frame_index:
                raise OnlineEpisodeStateError(
                    "online QueryId frame_index must increase strictly"
                )
            query_id = make_online_query_id(
                self._online_run_contract,
                self._active_episode_index,
                frame,
            )
            if query_id in self._issued or query_id in self._consumed:
                raise OnlineEpisodeStateError("duplicate online QueryId")
            self._last_frame_index = frame
            self._issued.add(query_id)
            return query_id

    def _claim_query(self, query_id: QueryId) -> None:
        if not isinstance(query_id, QueryId):
            raise TypeError("query_id must be QueryId")
        with self._lock:
            if query_id not in self._issued:
                raise OnlineEpisodeStateError(
                    "query_id was not issued by this retriever"
                )
            if query_id in self._consumed or query_id in self._inflight:
                raise OnlineEpisodeStateError(
                    "query_id was already consumed or is in flight"
                )
            self._inflight.add(query_id)

    def _release_query(self, query_id: QueryId, *, success: bool) -> None:
        with self._lock:
            self._inflight.discard(query_id)
            if success:
                self._consumed.add(query_id)

    def _register_delivered_step(self, step: BoundOnlineStep) -> None:
        """Register the one immutable result allowed for an issued QueryId."""

        with self._lock:
            if step.query_id not in self._issued or step.query_id in self._delivered_step_digests:
                raise OnlineEpisodeStateError(
                    "online QueryId already has a delivered retrieval result"
                )
            self._delivered_step_digests[step.query_id] = step.step_sha256

    def _prepare_raw_cameras(
        self, raw_cameras: Mapping[str, Any]
    ) -> tuple[dict[str, np.ndarray], Mapping[str, str]]:
        if not isinstance(raw_cameras, Mapping):
            raise TypeError("raw_cameras must be a camera-keyed mapping")
        if set(raw_cameras) != set(self._processor_camera_keys):
            raise OnlineRetrievalError(
                "raw camera keys must exactly match the processor camera contract"
            )
        prepared: dict[str, np.ndarray] = {}
        hashes: dict[str, str] = {}
        for key in self._processor_camera_keys:
            raw = np.asarray(raw_cameras[key])
            if raw.dtype != np.dtype(np.uint8) or raw.ndim != 3 or raw.shape[2] != 3:
                raise OnlineRetrievalError(
                    f"raw camera {key!r} must be uint8 HWC RGB"
                )
            if raw.shape[0] <= 0 or raw.shape[1] <= 0:
                raise OnlineRetrievalError(
                    f"raw camera {key!r} must have non-empty spatial dimensions"
                )
            contiguous = np.ascontiguousarray(raw)
            hashes[key] = sha256_array(contiguous)
            chw = np.transpose(contiguous, (2, 0, 1))[None, ...]
            prepared[key] = np.ascontiguousarray(
                chw.astype(np.float32) / np.float32(255.0)
            )
        return prepared, MappingProxyType(hashes)

    def _search_full_bank(
        self, context_key: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run the exact M1 full-bank cosine search in stable bank-row order.

        ``EventBank.search`` is the reference backend used by
        ``exact_cosine_v1`` candidate-cache construction.  Calling it here
        prevents online inference from silently introducing a ``TASK_INDEX``
        hard filter that the immutable train/dev caches never used.
        """

        query = np.asarray(context_key, dtype=np.float64)
        query_norm = float(np.linalg.norm(query))
        if query.ndim != 1 or query.shape[0] != self._summary.context_dim:
            raise OnlineRetrievalError(
                "online context key dimension does not match event bank"
            )
        if not np.isfinite(query).all() or query_norm <= 0.0:
            raise OnlineRetrievalError(
                "online context key must be finite with non-zero norm"
            )
        results = self._bank.search(
            query,
            top_k=self._online_run_contract.top_k,
        )
        rows = np.ascontiguousarray(
            [result.index for result in results], dtype=np.int64
        )
        scores = np.ascontiguousarray(
            np.clip(
                [result.score for result in results], -1.0, 1.0
            ),
            dtype=np.float64,
        )
        return rows, scores

    def retrieve(
        self,
        query_id: QueryId,
        raw_cameras: Mapping[str, Any],
        *,
        task_description: str,
        prompt: str,
        proprio: Any | None = None,
    ) -> BoundOnlineStep:
        """Consume one issued QueryId and return its immutable bound step."""

        self._claim_query(query_id)
        succeeded = False
        start_total = time.perf_counter()
        try:
            task = _normalized_text(task_description, "task_description")
            if task != self._online_run_contract.task_description:
                raise OnlineRetrievalError(
                    "task_description does not match the online run contract"
                )
            try:
                task_index = self._task_vocabulary.index(task)
            except ValueError as exc:
                raise OnlineRetrievalError(
                    "task_description is absent from exact encoder vocabulary"
                ) from exc
            prompt_value = _normalized_text(prompt, "prompt")
            proprio_value = _optional_proprio(proprio)
            if (
                proprio_value is not None
                and proprio_value.shape[1] != self._summary.proprio_dim
            ):
                raise OnlineRetrievalError(
                    "proprio dimension does not match event-bank contract"
                )

            stage = time.perf_counter()
            raw_batch, raw_hashes = self._prepare_raw_cameras(raw_cameras)
            prepared = self._image_adapter.prepare(raw_batch)
            preprocess_s = time.perf_counter() - stage
            processed_hashes = MappingProxyType(
                {
                    key: sha256_array(prepared.camera_frames[key])
                    for key in self._processor_camera_keys
                }
            )

            stage = time.perf_counter()
            semantic_frames = prepared.camera_frames[
                self._semantic_processor_camera
            ]
            dino_features = self._dino.encode(
                semantic_frames, batch_size=self._dino_batch_size
            )
            dino_s = time.perf_counter() - stage
            cls_features = np.asarray(getattr(dino_features, "cls", None))
            if (
                cls_features.dtype != np.dtype(np.float32)
                or cls_features.ndim != 2
                or cls_features.shape[0] != 1
            ):
                raise OnlineRetrievalError(
                    "DINO encoder must return float32 CLS features [1,D]"
                )
            keys = build_m1_context_keys(
                cls_features,
                task_index=task_index,
                catalog_task_count=len(self._task_vocabulary),
                visual_only=self._context_mode == "visual-only",
            )
            context_key = np.ascontiguousarray(keys[0], dtype=np.float32)

            stage = time.perf_counter()
            selected_rows, selected_scores = self._search_full_bank(context_key)
            search_s = time.perf_counter() - stage

            stage = time.perf_counter()
            top_k = self._online_run_contract.top_k
            rows = np.full((top_k,), INVALID_BANK_ROW, dtype=np.int64)
            scores = np.zeros((top_k,), dtype=np.float64)
            valid = np.zeros((top_k,), dtype=np.bool_)
            actions = np.zeros(
                (
                    top_k,
                    self._online_run_contract.action_horizon,
                    self._online_run_contract.action_dim,
                ),
                dtype=np.float32,
            )
            events: list[EventId | None] = [None] * top_k
            count = int(selected_rows.size)
            if count:
                if np.any(selected_rows < 0) or np.any(selected_rows >= len(self._bank)):
                    raise OnlineRetrievalError(
                        "search returned an out-of-range event-bank row"
                    )
                # Only already range-checked non-negative rows are gathered.
                payload = self._bank.payload(MODEL_SPACE_ACTION)
                selected_actions = payload[selected_rows]
                rows[:count] = selected_rows
                scores[:count] = selected_scores
                valid[:count] = True
                actions[:count] = selected_actions
                for rank, row in enumerate(selected_rows.tolist()):
                    events[rank] = self._bank.event_ids[int(row)]
            gather_s = time.perf_counter() - stage
            stage_sum = preprocess_s + dino_s + search_s + gather_s
            elapsed = time.perf_counter() - start_total
            total_s = max(elapsed, stage_sum)
            telemetry = {
                "preprocess_s": preprocess_s,
                "dino_s": dino_s,
                "search_s": search_s,
                "gather_s": gather_s,
                "total_s": total_s,
            }
            step = BoundOnlineStep(
                query_id=query_id,
                online_contract_sha256=self._online_run_contract.sha256,
                training_run_contract_sha256=self._source_run_contract.sha256,
                validation_run_contract_sha256=(
                    self._online_run_contract.validation_run_contract_sha256
                ),
                bank_manifest_sha256=self._online_run_contract.bank_manifest_sha256,
                bank_content_sha256=self._online_run_contract.bank_content_sha256,
                encoder_contract_sha256=self._online_run_contract.encoder_contract_sha256,
                camera_contract_sha256=self._online_run_contract.camera_contract_sha256,
                normalization_stats_sha256=(
                    self._online_run_contract.normalization_stats_sha256
                ),
                action_space_contract_sha256=(
                    self._online_run_contract.action_space_contract_sha256
                ),
                catalog_sha256=self._online_run_contract.catalog_sha256,
                audit_sha256=self._online_run_contract.audit_sha256,
                task_description=task,
                task_index=task_index,
                prompt=prompt_value,
                proprio=proprio_value,
                raw_camera_sha256=raw_hashes,
                processed_camera_sha256=processed_hashes,
                model_input=np.ascontiguousarray(prepared.vae_frames, dtype=np.float32),
                context_key=context_key,
                event_ids=tuple(events),
                bank_rows=rows,
                cosine_scores=scores,
                candidate_valid_mask=valid,
                candidate_means=actions,
                derived_seed=derive_online_query_seed(
                    self._online_run_contract.root_seed,
                    query_id,
                    self._online_run_contract.evaluation_namespace_sha256,
                ),
                telemetry=telemetry,
                _capability=self._capability,
            )
            # Validate before publication without consuming the model-use
            # capability.  Public validate_bound_step() performs the one-time
            # policy admission later.
            self._assert_owned_bound_step(step)
            self._register_delivered_step(step)
            succeeded = True
            return step
        finally:
            self._release_query(query_id, success=succeeded)

    def _assert_owned_bound_step(
        self,
        step: BoundOnlineStep,
        *,
        prompt: str | None = None,
        proprio: Any | None = None,
        input_image: Any | None = None,
    ) -> BoundOnlineStep:
        """Prove ownership, contracts, payloads, and optional model inputs."""

        if not isinstance(step, BoundOnlineStep):
            raise TypeError("step must be BoundOnlineStep")
        if step._capability is not self._capability:
            raise OnlineBoundStepError(
                "bound step was not created by this online retriever"
            )
        expected = {
            "online_contract_sha256": self._online_run_contract.sha256,
            "training_run_contract_sha256": self._source_run_contract.sha256,
            "validation_run_contract_sha256": (
                self._online_run_contract.validation_run_contract_sha256
            ),
            "bank_manifest_sha256": self._online_run_contract.bank_manifest_sha256,
            "bank_content_sha256": self._online_run_contract.bank_content_sha256,
            "encoder_contract_sha256": self._online_run_contract.encoder_contract_sha256,
            "camera_contract_sha256": self._online_run_contract.camera_contract_sha256,
            "normalization_stats_sha256": self._online_run_contract.normalization_stats_sha256,
            "action_space_contract_sha256": self._online_run_contract.action_space_contract_sha256,
            "catalog_sha256": self._online_run_contract.catalog_sha256,
            "audit_sha256": self._online_run_contract.audit_sha256,
        }
        for field_name, expected_value in expected.items():
            if getattr(step, field_name) != expected_value:
                raise OnlineBoundStepError(
                    f"bound step {field_name} does not match this retriever"
                )
        expected_query = make_online_query_id(
            self._online_run_contract,
            step.query_id.episode_index,
            step.query_id.frame_index,
        )
        if step.query_id != expected_query:
            raise OnlineBoundStepError("bound step QueryId namespace is invalid")
        if step.task_description != self._online_run_contract.task_description:
            raise OnlineBoundStepError("bound step task does not match online contract")
        expected_seed = derive_online_query_seed(
            self._online_run_contract.root_seed,
            step.query_id,
            self._online_run_contract.evaluation_namespace_sha256,
        )
        if step.derived_seed != expected_seed:
            raise OnlineBoundStepError("bound step derived seed is invalid")
        valid_rows = step.bank_rows[step.candidate_valid_mask]
        if np.any(valid_rows < 0) or np.any(valid_rows >= len(self._bank)):
            raise OnlineBoundStepError("bound step contains an out-of-range bank row")
        for rank, row in enumerate(valid_rows.tolist()):
            if self._bank.event_ids[int(row)] != step.event_ids[rank]:
                raise OnlineBoundStepError(
                    "bound step EventId-to-bank-row relation is invalid"
                )
        expected_actions = self._bank.payload(MODEL_SPACE_ACTION)[valid_rows]
        if not np.array_equal(
            step.candidate_means[: step.valid_count], expected_actions
        ):
            raise OnlineBoundStepError(
                "bound step candidate actions do not match event-bank payload"
            )
        step._assert_integrity()
        if prompt is not None and prompt != step.prompt:
            raise OnlineBoundStepError("model prompt does not match bound step")
        if proprio is not None:
            actual_proprio = _optional_proprio(_external_array(proprio, "proprio"))
            if step.proprio is None or not np.array_equal(actual_proprio, step.proprio):
                raise OnlineBoundStepError("model proprio does not match bound step")
        if input_image is not None:
            actual_input = np.asarray(
                _external_array(input_image, "input_image"), dtype=np.float32
            )
            if actual_input.ndim == 3:
                actual_input = actual_input[None, ...]
            if not np.array_equal(actual_input, step.model_input):
                raise OnlineBoundStepError("model input_image does not match bound step")
        return step

    def assert_owned_bound_step(
        self,
        step: BoundOnlineStep,
        *,
        prompt: str | None = None,
        proprio: Any | None = None,
        input_image: Any | None = None,
    ) -> BoundOnlineStep:
        """Validate a capability without consuming its one model-use token.

        Production policy code should call :meth:`validate_bound_step` instead.
        This non-consuming variant exists for preflight checks and integration
        diagnostics; it still requires that the retriever actually delivered
        this exact step in the currently active episode.
        """

        result = self._assert_owned_bound_step(
            step,
            prompt=prompt,
            proprio=proprio,
            input_image=input_image,
        )
        with self._lock:
            if self._active_episode_index != step.query_id.episode_index:
                raise OnlineBoundStepError(
                    "bound step does not belong to the active episode"
                )
            delivered = self._delivered_step_digests.get(step.query_id)
            if delivered != step.step_sha256:
                raise OnlineBoundStepError(
                    "bound step is not the registered result for its QueryId"
                )
        return result

    def validate_bound_step(
        self,
        step: BoundOnlineStep,
        *,
        prompt: str | None = None,
        proprio: Any | None = None,
        input_image: Any | None = None,
    ) -> BoundOnlineStep:
        """Validate and consume one retriever-produced policy capability.

        Consumption is atomic and exactly-once per QueryId.  The content
        hashes for prompt, proprioception, model input, context key, and bank
        actions are recomputed before the capability is marked consumed.
        """

        result = self.assert_owned_bound_step(
            step,
            prompt=prompt,
            proprio=proprio,
            input_image=input_image,
        )
        with self._lock:
            if step.query_id in self._model_consumed:
                raise OnlineBoundStepError(
                    "bound step capability was already consumed by the policy"
                )
            # Recheck the active registration while holding the same lock used
            # to mark consumption, closing concurrent double-use races.
            if self._active_episode_index != step.query_id.episode_index or (
                self._delivered_step_digests.get(step.query_id) != step.step_sha256
            ):
                raise OnlineBoundStepError(
                    "bound step registration changed before policy consumption"
                )
            self._model_consumed.add(step.query_id)
        return result


__all__ = [
    "BoundOnlineStep",
    "FrozenDinoOnlineRetriever",
    "INVALID_BANK_ROW",
    "OnlineArtifactContractError",
    "OnlineBoundStepError",
    "OnlineEpisodeStateError",
    "OnlineRetrievalError",
    "derive_online_query_seed",
    "make_online_query_id",
    "online_query_dataset_id",
    "validate_online_camera_contract",
    "validate_online_encoder_contract",
]
