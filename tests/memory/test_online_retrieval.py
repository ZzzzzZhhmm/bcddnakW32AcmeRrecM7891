from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from fastwam.datasets.lerobot.audit import (
    AUDIT_SCHEMA,
    AUDIT_VERSION,
    write_audit_report,
)
from fastwam.datasets.lerobot.episode_catalog import (
    DatasetDescriptor,
    EpisodeCatalog,
    EpisodeRecord,
)
from fastwam.memory.candidate_cache import canonical_event_bank_content_hash
from fastwam.memory.event_bank import EventBank
from fastwam.memory.feature_precompute import build_m1_context_keys
from fastwam.memory.manifest import (
    sha256_array,
    sha256_canonical_json,
    sha256_file,
    sha256_path_tree,
)
from fastwam.memory.online_retrieval import (
    INVALID_BANK_ROW,
    FrozenDinoOnlineRetriever,
    OnlineArtifactContractError,
    OnlineBoundStepError,
    OnlineCandidateFacts,
    OnlineEpisodeStateError,
    OnlineRetrievalError,
    derive_online_query_seed,
    make_online_query_id,
    online_query_dataset_id,
)
from fastwam.memory.payload_names import (
    CONTAINS_FORCED_GRIPPER,
    EFFECT_POST,
    EFFECT_PRE,
    EVENT_SCORE,
    FEATURE_EPISODE_SHA256,
    MODEL_SPACE_ACTION,
    OBSERVED_GRIPPER_STATE,
    SOURCE_EPISODE_SHA256,
    START_PROPRIO,
    TASK_INDEX,
)
from fastwam.memory.schema import EventId
from fastwam.models.warm.online_contract import (
    ONLINE_RETRIEVAL_IMPLEMENTATION,
    WarmOnlineRunContract,
)
from fastwam.models.warm.source_contract import WarmSourceRunContract


TASK_A = "task-a"
TASK_B = "task-b"
CAMERA_KEYS = ("image", "wrist_image")
SOURCE_CAMERA_KEYS = (
    "observation.images.image",
    "observation.images.wrist_image",
)
HORIZON = 2
ACTION_DIM = 3


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class _ScaleToFloat:
    def __call__(self, value: Any) -> np.ndarray:
        return np.asarray(value, dtype=np.float32) / np.float32(255.0)


class _NumpyProcessor:
    is_train = False
    num_output_cameras = 2
    shape_meta = {
        "images": [
            {"key": "image", "shape": [3, 224, 224]},
            {"key": "wrist_image", "shape": [3, 224, 224]},
        ]
    }
    normalizer = object()
    val_transforms = [_ScaleToFloat()]


class _FrozenNumpyDino:
    def __init__(self, contract: dict[str, Any]) -> None:
        dino = contract["dino"]
        self.compute_device = contract["compute"]["device"]
        self.compute_dtype = contract["compute"]["dtype"]
        self.model_id = dino["model_id"]
        self.revision = dino["revision"]
        self.hidden_size = dino["hidden_size"]
        self.patch_size = tuple(dino["patch_size"])
        self.patch_grid_size = tuple(dino["patch_grid_size"])
        self.register_token_count = dino["register_token_count"]
        self.image_size = tuple(dino["image_size"])
        self.image_mean = tuple(dino["image_mean"])
        self.image_std = tuple(dino["image_std"])
        self.seen_frames: list[np.ndarray] = []

    def encode(self, frames: Any, *, batch_size: int) -> SimpleNamespace:
        array = np.asarray(frames)
        assert batch_size == 1
        assert array.dtype == np.float32
        assert array.shape == (1, 3, 224, 224)
        self.seen_frames.append(np.array(array, copy=True))
        return SimpleNamespace(
            cls=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)
        )


@dataclass(frozen=True)
class _Artifacts:
    bank_directory: Path
    normalizer_contract_path: Path
    encoder_contract_path: Path
    camera_contract_path: Path
    normalization_stats_path: Path
    catalog_path: Path
    audit_report_path: Path
    dino_checkpoint_path: Path
    source_contract: WarmSourceRunContract
    online_contract: WarmOnlineRunContract
    processor: _NumpyProcessor
    dino: _FrozenNumpyDino
    actions: np.ndarray


def _event_hash_rows(prefix: str, count: int) -> np.ndarray:
    return np.stack(
        [
            np.frombuffer(
                hashlib.sha256(f"{prefix}-{index}".encode("utf-8")).digest(),
                dtype=np.uint8,
            )
            for index in range(count)
        ],
        axis=0,
    )


def _write_catalog_and_audit(root: Path) -> tuple[Path, Path, EpisodeCatalog, str]:
    descriptor = DatasetDescriptor(
        dataset_id="synthetic",
        dataset_index=0,
        fps=20.0,
        total_episodes=4,
        chunks_size=1000,
        data_path_template=(
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        ),
        info_sha256=_digest("info"),
        episodes_sha256=_digest("episodes"),
    )
    episodes = tuple(
        EpisodeRecord(
            dataset_id="synthetic",
            dataset_index=0,
            episode_index=index,
            length=8,
            fps=20.0,
            tasks=(TASK_B if index == 2 else TASK_A,),
            data_relpath=f"data/chunk-000/episode_{index:06d}.parquet",
            split="train",
        )
        for index in range(4)
    )
    catalog = EpisodeCatalog((descriptor,), episodes)
    catalog_path = root / "catalog.json"
    catalog.save(catalog_path)

    audit_without_hash: dict[str, Any] = {
        "schema": AUDIT_SCHEMA,
        "version": AUDIT_VERSION,
        "catalog_sha256": catalog.content_sha256,
        "audited_camera_keys": list(SOURCE_CAMERA_KEYS),
        "datasets": [],
        "summary": {
            "dataset_count": 1,
            "episode_count": 4,
            "task_count": 2,
            "split_counts": {"train": 4},
            "episode_tables_hashed": False,
            "cross_split_duplicate_count": 0,
            "cross_split_table_duplicate_count": 0,
            "cross_split_video_duplicate_count": 0,
            "cross_split_camera_bundle_duplicate_count": 0,
            "cross_split_source_bundle_duplicate_count": 0,
        },
        "cross_split_table_duplicates": [],
        "cross_split_video_duplicates": [],
        "cross_split_camera_bundle_duplicates": [],
        "cross_split_source_bundle_duplicates": [],
        "episode_source_hashes": [],
    }
    audit_hash = _canonical_hash(audit_without_hash)
    audit = {**audit_without_hash, "report_sha256": audit_hash}
    audit_path = root / "audit.json"
    write_audit_report(audit, audit_path)
    return catalog_path, audit_path, catalog, audit_hash


def _make_artifacts(tmp_path: Path) -> _Artifacts:
    root = tmp_path / "artifacts"
    root.mkdir()
    stats_path = root / "dataset_stats.json"
    stats_path.write_bytes(b'{"mean":[0.0,0.0,0.0],"std":[1.0,1.0,1.0]}\n')
    stats_sha = sha256_file(stats_path)

    normalizer = {
        "schema": "warm.action-space",
        "version": 1,
        "action_dim": ACTION_DIM,
        "arm_dims": [0, 1],
        "gripper_dims": [2],
        "gripper_threshold": 0.0,
        "normalization_mode": "standard-score",
        "normalization_stats_sha256": stats_sha,
        "control_mode": "ee-delta",
        "embodiment": "libero-panda",
    }
    normalizer_path = root / "normalizer_contract.json"
    _write_json(normalizer_path, normalizer)

    dino_path = root / "dino"
    dino_path.mkdir()
    (dino_path / "config.json").write_bytes(b'{"model_type":"dinov2"}\n')
    dino_tree_sha, dino_file_count = sha256_path_tree(dino_path)
    encoder = {
        "schema": "warm.feature-encoder",
        "version": 2,
        "official_complete": True,
        "runtime": {"python": "test", "device": "fake-cuda"},
        "data_config_sha256": _digest("m1-data-config"),
        "output_dtype": "float32",
        "compute": {
            "device": "cuda",
            "dtype": "bfloat16",
            "dino_batch_size": 1,
            "vae_batch_size": None,
        },
        "dino": {
            "model_id": "facebook/dinov2-small",
            "revision": "abcdef0",
            "checkpoint_tree_sha256": dino_tree_sha,
            "checkpoint_file_count": dino_file_count,
            "hidden_size": 3,
            "patch_size": [14, 14],
            "patch_grid_size": [16, 16],
            "register_token_count": 0,
            "image_size": [224, 224],
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
            "semantic_pool": "adaptive_avg_pool_2x2_row_major",
            "resize_in_encoder": False,
        },
        "context": {
            "mode": "task-conditioned",
            "task_vocabulary": [TASK_A, TASK_B],
            "visual_task_energy": [0.5, 0.5],
        },
    }
    encoder_path = root / "encoder_contract.json"
    _write_json(encoder_path, encoder)

    camera = {
        "schema": "warm.camera-layout",
        "version": 1,
        "source_camera_keys": list(SOURCE_CAMERA_KEYS),
        "processor_camera_mapping": {
            SOURCE_CAMERA_KEYS[0]: CAMERA_KEYS[0],
            SOURCE_CAMERA_KEYS[1]: CAMERA_KEYS[1],
        },
        "semantic_camera": SOURCE_CAMERA_KEYS[0],
        "concat_mode": "horizontal",
        "per_camera_size": [224, 224],
        "decoded_range": [0.0, 1.0],
        "vae_model_range": [-1.0, 1.0],
        "baseline_quantization": "validated_0_1_times_255_to_uint8",
    }
    camera_path = root / "camera_contract.json"
    _write_json(camera_path, camera)

    catalog_path, audit_path, catalog, audit_sha = _write_catalog_and_audit(root)
    visual_vectors = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    task_indices = np.asarray([0, 0, 1, 0], dtype=np.int64)
    context_keys = np.concatenate(
        [
            build_m1_context_keys(
                visual_vectors[index : index + 1],
                task_index=int(task_indices[index]),
                catalog_task_count=2,
            )
            for index in range(4)
        ],
        axis=0,
    )
    actions = np.stack(
        [
            np.full((HORIZON, ACTION_DIM), index + 1, dtype=np.float32)
            for index in range(4)
        ]
    )
    event_ids = tuple(EventId("synthetic", 0, index, 1) for index in range(4))
    effect_pre = np.arange(4 * 4 * 3, dtype=np.float32).reshape(4, 4, 3)
    bank = EventBank(
        event_ids,
        np.ascontiguousarray(context_keys, dtype=np.float32),
        {
            MODEL_SPACE_ACTION: actions,
            EFFECT_PRE: effect_pre,
            EFFECT_POST: effect_pre + np.float32(2.0),
            START_PROPRIO: np.arange(8, dtype=np.float32).reshape(4, 2),
            OBSERVED_GRIPPER_STATE: np.asarray(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0],
                    [0.5, 0.5, 0.5],
                    [1.0, 1.0, 0.0],
                ],
                dtype=np.float32,
            ),
            TASK_INDEX: task_indices,
            EVENT_SCORE: np.ones((4,), dtype=np.float32),
            CONTAINS_FORCED_GRIPPER: np.zeros((4,), dtype=np.bool_),
            SOURCE_EPISODE_SHA256: _event_hash_rows("source", 4),
            FEATURE_EPISODE_SHA256: _event_hash_rows("feature", 4),
        },
    )
    bank_directory = root / "bank"
    manifest = bank.save(
        bank_directory,
        action_normalizer={
            "file_sha256": sha256_file(normalizer_path),
            "contract": normalizer,
        },
        encoder={
            "file_sha256": sha256_file(encoder_path),
            "contract": encoder,
        },
        camera_layout={
            "file_sha256": sha256_file(camera_path),
            "contract": camera,
        },
        provenance={
            "data_binding": {
                "catalog_sha256": catalog.content_sha256,
                "audit_report_sha256": audit_sha,
                "split": "train",
            }
        },
    )
    manifest_sha = sha256_file(bank_directory / "manifest.json")
    bank_content_sha = canonical_event_bank_content_hash(manifest.content_hashes)
    action_contract_sha = sha256_canonical_json(normalizer)
    source = WarmSourceRunContract(
        bank_manifest_sha256=manifest_sha,
        bank_content_sha256=bank_content_sha,
        candidate_manifest_sha256=_digest("candidate-manifest"),
        query_corpus_sha256=_digest("query-corpus"),
        catalog_sha256=catalog.content_sha256,
        audit_sha256=audit_sha,
        normalization_stats_sha256=stats_sha,
        action_space_contract_sha256=action_contract_sha,
        base_checkpoint_sha256=_digest("base-checkpoint"),
        query_split="train",
        global_sample_stride=1,
        action_horizon=HORIZON,
        action_dim=ACTION_DIM,
    )
    online = WarmOnlineRunContract(
        training_run_contract_sha256=source.sha256,
        validation_run_contract_sha256=_digest("dev-source-contract"),
        warm_checkpoint_sha256=_digest("warm-checkpoint"),
        training_attestation_sha256=_digest("training-attestation"),
        shared_training_recipe_sha256=_digest("shared-training-recipe"),
        training_runtime_sha256=_digest("training-runtime"),
        bank_manifest_sha256=manifest_sha,
        bank_content_sha256=bank_content_sha,
        encoder_contract_sha256=sha256_file(encoder_path),
        encoder_runtime_sha256=sha256_canonical_json(encoder["runtime"]),
        camera_contract_sha256=sha256_file(camera_path),
        m1_data_config_sha256=encoder["data_config_sha256"],
        dino_checkpoint_tree_sha256=dino_tree_sha,
        dino_checkpoint_file_count=dino_file_count,
        normalization_stats_sha256=stats_sha,
        action_space_contract_sha256=action_contract_sha,
        catalog_sha256=catalog.content_sha256,
        audit_sha256=audit_sha,
        resolved_eval_config_sha256=_digest("eval-config"),
        vae_checkpoint_sha256=_digest("vae"),
        text_encoder_tree_sha256=_digest("text"),
        tokenizer_tree_sha256=_digest("tokenizer"),
        evaluation_namespace_sha256=_digest("shared-eval-namespace"),
        task_suite="libero_10",
        task_id=0,
        task_description=TASK_A,
        root_seed=17,
        initial_states_sha256=_digest("initial-states"),
        bddl_sha256=_digest("bddl"),
        retrieval_implementation=ONLINE_RETRIEVAL_IMPLEMENTATION,
        top_k=5,
        source_policy="fixed_context_top1",
        memory_sigma=0.2,
        action_horizon=HORIZON,
        action_dim=ACTION_DIM,
        git_commit="a" * 40,
        git_dirty=False,
    )
    dino = _FrozenNumpyDino(encoder)
    return _Artifacts(
        bank_directory=bank_directory,
        normalizer_contract_path=normalizer_path,
        encoder_contract_path=encoder_path,
        camera_contract_path=camera_path,
        normalization_stats_path=stats_path,
        catalog_path=catalog_path,
        audit_report_path=audit_path,
        dino_checkpoint_path=dino_path,
        source_contract=source,
        online_contract=online,
        processor=_NumpyProcessor(),
        dino=dino,
        actions=actions,
    )


def _load_retriever(artifacts: _Artifacts) -> FrozenDinoOnlineRetriever:
    return FrozenDinoOnlineRetriever.from_artifacts(
        artifacts.bank_directory,
        source_run_contract=artifacts.source_contract,
        online_run_contract=artifacts.online_contract,
        normalizer_contract_path=artifacts.normalizer_contract_path,
        encoder_contract_path=artifacts.encoder_contract_path,
        camera_contract_path=artifacts.camera_contract_path,
        normalization_stats_path=artifacts.normalization_stats_path,
        catalog_path=artifacts.catalog_path,
        audit_report_path=artifacts.audit_report_path,
        dino_checkpoint_path=artifacts.dino_checkpoint_path,
        processor=artifacts.processor,
        dino_encoder=artifacts.dino,
        dino_batch_size=1,
        tensor_factory=lambda value: value,
    )


def _raw_cameras() -> dict[str, np.ndarray]:
    return {
        "image": np.full((224, 224, 3), 64, dtype=np.uint8),
        "wrist_image": np.full((224, 224, 3), 192, dtype=np.uint8),
    }


def test_artifact_bound_retrieval_matches_m1_full_bank_stable_search_and_padding(
    tmp_path: Path,
) -> None:
    artifacts = _make_artifacts(tmp_path)
    retriever = _load_retriever(artifacts)
    assert type(retriever) is FrozenDinoOnlineRetriever
    assert retriever.artifact_verified is True
    retriever.begin_episode(3)
    query_id = retriever.make_query_id(7)

    with pytest.raises(OnlineRetrievalError, match="does not match"):
        retriever.retrieve(
            query_id,
            _raw_cameras(),
            task_description=TASK_B,
            prompt=TASK_A,
            proprio=np.asarray([0.25, -0.25], dtype=np.float32),
        )

    step = retriever.retrieve(
        query_id,
        _raw_cameras(),
        task_description=TASK_A,
        prompt=TASK_A,
        proprio=np.asarray([0.25, -0.25], dtype=np.float32),
    )
    assert step.query_id == make_online_query_id(artifacts.online_contract, 3, 7)
    assert step.validation_run_contract_sha256 == (
        artifacts.online_contract.validation_run_contract_sha256
    )
    # M1 exact_cosine_v1 searches the complete bank.  Row 2 belongs to TASK_B,
    # but remains an eligible neighbor because task identity is already one
    # component of the task-conditioned context key.  Equal-score rows retain
    # stable bank-row order, exactly as EventBank.search does offline.
    assert step.bank_rows.tolist() == [0, 3, 1, 2, INVALID_BANK_ROW]
    assert step.candidate_valid_mask.tolist() == [True, True, True, True, False]
    np.testing.assert_allclose(step.cosine_scores, [1.0, 1.0, 0.5, 0.5, 0.0])
    np.testing.assert_array_equal(
        step.candidate_means[:4], artifacts.actions[[0, 3, 1, 2]]
    )
    np.testing.assert_array_equal(
        step.candidate_means[4], np.zeros((HORIZON, ACTION_DIM), dtype=np.float32)
    )
    assert step.event_ids == tuple(
        EventId("synthetic", 0, row, 1) for row in (0, 3, 1, 2)
    ) + (None,)
    assert step.model_input.shape == (1, 3, 224, 448)
    np.testing.assert_allclose(
        step.model_input[:, :, :, :224], np.float32(2.0 * 64.0 / 255.0 - 1.0)
    )
    np.testing.assert_allclose(
        step.model_input[:, :, :, 224:], np.float32(2.0 * 192.0 / 255.0 - 1.0)
    )
    np.testing.assert_allclose(
        artifacts.dino.seen_frames[-1], np.float32(64.0 / 255.0)
    )
    assert step.raw_camera_sha256["image"] == sha256_array(_raw_cameras()["image"])
    assert not step.model_input.flags.writeable
    assert not step.candidate_means.flags.writeable
    assert step.derived_seed == derive_online_query_seed(
        17, query_id, artifacts.online_contract.evaluation_namespace_sha256
    )
    assert tuple(step.telemetry) == (
        "preprocess_s",
        "dino_s",
        "search_s",
        "gather_s",
        "total_s",
    )
    assert retriever.assert_owned_bound_step(step) is step
    assert (
        retriever.validate_bound_step(
            step,
            prompt=TASK_A,
            proprio=np.asarray([0.25, -0.25], dtype=np.float32),
            input_image=step.model_input,
        )
        is step
    )
    with pytest.raises(OnlineBoundStepError, match="already consumed"):
        retriever.validate_bound_step(step)


def test_candidate_facts_are_owned_immutable_and_strictly_zero_padded(
    tmp_path: Path,
) -> None:
    artifacts = _make_artifacts(tmp_path)
    retriever = _load_retriever(artifacts)
    retriever.begin_episode(0)
    step = retriever.retrieve(
        retriever.make_query_id(2),
        _raw_cameras(),
        task_description=TASK_A,
        prompt=TASK_A,
        proprio=np.asarray([0.25, -0.25], dtype=np.float32),
    )

    facts = retriever.gather_candidate_facts(
        step,
        prompt=TASK_A,
        proprio=np.asarray([0.25, -0.25], dtype=np.float32),
        input_image=step.model_input,
    )
    assert isinstance(facts, OnlineCandidateFacts)
    assert facts.step_sha256 == step.step_sha256
    assert facts.candidate_valid_mask.tolist() == [True, True, True, True, False]
    assert facts.context_keys.shape == (5, 5)
    assert facts.effect_pre.shape == (5, 4, 3)
    assert facts.effect_post.shape == (5, 4, 3)
    assert facts.effect_delta.shape == (5, 4, 3)
    assert facts.start_proprio.shape == (5, 2)
    assert facts.observed_gripper.shape == (5, HORIZON + 1)
    assert facts.gripper_timing.shape == (5, 4)
    assert facts.support.tolist() == [1.0, 1.0, 1.0, 1.0, 0.0]

    selected_rows = np.asarray([0, 3, 1, 2], dtype=np.int64)
    expected_pre = np.arange(4 * 4 * 3, dtype=np.float32).reshape(4, 4, 3)
    np.testing.assert_array_equal(facts.effect_pre[:4], expected_pre[selected_rows])
    np.testing.assert_array_equal(
        facts.effect_post[:4], expected_pre[selected_rows] + np.float32(2.0)
    )
    np.testing.assert_array_equal(
        facts.effect_delta[:4], np.full((4, 4, 3), 2.0, dtype=np.float32)
    )
    np.testing.assert_array_equal(
        facts.start_proprio[:4],
        np.arange(8, dtype=np.float32).reshape(4, 2)[selected_rows],
    )
    np.testing.assert_array_equal(
        facts.observed_gripper[:4],
        np.asarray(
            [
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [0.5, 0.5, 0.5],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        facts.gripper_timing[:4],
        np.asarray(
            [
                [0.5, 1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.5, 1.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    for array in facts.as_mapping().values():
        assert not array.flags.writeable
        np.testing.assert_array_equal(array[4], np.zeros_like(array[4]))
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(TypeError):
        facts.as_mapping()["support"] = facts.support  # type: ignore[index]

    # Gathering alone is deliberately non-consuming, so the existing policy
    # boundary remains responsible for the exactly-once capability use.
    assert retriever.validate_bound_step(step) is step


def test_validate_and_gather_candidate_facts_consumes_exactly_once(
    tmp_path: Path,
) -> None:
    artifacts = _make_artifacts(tmp_path)
    retriever = _load_retriever(artifacts)
    retriever.begin_episode(1)
    step = retriever.retrieve(
        retriever.make_query_id(3),
        _raw_cameras(),
        task_description=TASK_A,
        prompt=TASK_A,
        proprio=np.asarray([0.0, 0.0], dtype=np.float32),
    )

    facts = retriever.validate_and_gather_candidate_facts(step, prompt=TASK_A)
    assert facts.step_sha256 == step.step_sha256
    with pytest.raises(OnlineBoundStepError, match="already consumed"):
        retriever.validate_and_gather_candidate_facts(step)


def test_candidate_facts_reject_forged_negative_valid_bank_row_before_gather(
    tmp_path: Path,
) -> None:
    artifacts = _make_artifacts(tmp_path)
    retriever = _load_retriever(artifacts)
    retriever.begin_episode(1)
    step = retriever.retrieve(
        retriever.make_query_id(3),
        _raw_cameras(),
        task_description=TASK_A,
        prompt=TASK_A,
        proprio=np.asarray([0.0, 0.0], dtype=np.float32),
    )
    forged = np.array(step.bank_rows, copy=True)
    forged[0] = INVALID_BANK_ROW
    object.__setattr__(step, "bank_rows", forged)

    with pytest.raises(OnlineBoundStepError, match="out-of-range bank row"):
        retriever.gather_candidate_facts(step)


def test_query_namespace_and_seed_are_policy_independent_but_ids_are_strict(
    tmp_path: Path,
) -> None:
    artifacts = _make_artifacts(tmp_path)
    null_value = artifacts.online_contract.to_dict()
    null_value["source_policy"] = "gaussian_null"
    null_contract = WarmOnlineRunContract.from_dict(null_value)
    fixed_id = make_online_query_id(artifacts.online_contract, 1, 9)
    null_id = make_online_query_id(null_contract, 1, 9)
    assert fixed_id == null_id
    assert online_query_dataset_id(artifacts.online_contract) == (
        online_query_dataset_id(null_contract)
    )
    assert derive_online_query_seed(
        17, fixed_id, artifacts.online_contract.evaluation_namespace_sha256
    ) == derive_online_query_seed(
        17, null_id, null_contract.evaluation_namespace_sha256
    )

    retriever = _load_retriever(artifacts)
    retriever.begin_episode(0)
    assert retriever.make_query_id(2).frame_index == 2
    with pytest.raises(OnlineEpisodeStateError, match="increase strictly"):
        retriever.make_query_id(2)
    with pytest.raises(OnlineEpisodeStateError, match="increase strictly"):
        retriever.make_query_id(1)
    with pytest.raises(OnlineEpisodeStateError, match="unconsumed issued"):
        retriever.begin_episode(1)


def test_episode_transition_rejects_a_delivered_but_unconsumed_step(
    tmp_path: Path,
) -> None:
    artifacts = _make_artifacts(tmp_path)
    retriever = _load_retriever(artifacts)
    retriever.begin_episode(0)
    step = retriever.retrieve(
        retriever.make_query_id(4),
        _raw_cameras(),
        task_description=TASK_A,
        prompt=TASK_A,
        proprio=np.asarray([0.0, 0.0], dtype=np.float32),
    )
    with pytest.raises(OnlineEpisodeStateError, match="unconsumed BoundOnlineStep"):
        retriever.begin_episode(1)
    retriever.validate_bound_step(step)
    retriever.begin_episode(1)
    with pytest.raises(OnlineBoundStepError, match="active episode"):
        retriever.assert_owned_bound_step(step)


def test_bound_step_recomputes_prompt_and_proprio_hashes_before_consumption(
    tmp_path: Path,
) -> None:
    artifacts = _make_artifacts(tmp_path)
    retriever = _load_retriever(artifacts)
    retriever.begin_episode(0)
    step = retriever.retrieve(
        retriever.make_query_id(1),
        _raw_cameras(),
        task_description=TASK_A,
        prompt=TASK_A,
        proprio=np.asarray([0.0, 0.0], dtype=np.float32),
    )
    object.__setattr__(step, "prompt", "tampered-prompt")
    with pytest.raises(OnlineBoundStepError, match="prompt content changed"):
        retriever.validate_bound_step(step)


@pytest.mark.parametrize(
    ("artifact_name", "mutate", "message"),
    [
        (
            "encoder",
            lambda value: {
                **value,
                "context": {
                    **value["context"],
                    "visual_task_energy": [1.0, 0.0],
                },
            },
            "visual_task_energy",
        ),
        (
            "encoder",
            lambda value: {
                **value,
                "context": {
                    **value["context"],
                    "mode": "visual-only",
                    "visual_task_energy": [0.5, 0.5],
                },
            },
            "visual_task_energy",
        ),
        (
            "encoder",
            lambda value: {
                **value,
                "compute": {**value["compute"], "dtype": "float16"},
            },
            "compute dtype",
        ),
        (
            "encoder",
            lambda value: {
                **value,
                "compute": {**value["compute"], "device": "cpu"},
            },
            "DINO device",
        ),
        (
            "camera",
            lambda value: {**value, "concat_mode": "vertical"},
            "horizontal camera concat",
        ),
    ],
)
def test_exact_m1_encoder_and_camera_invariants_fail_closed(
    tmp_path: Path,
    artifact_name: str,
    mutate: Any,
    message: str,
) -> None:
    artifacts = _make_artifacts(tmp_path)
    path = (
        artifacts.encoder_contract_path
        if artifact_name == "encoder"
        else artifacts.camera_contract_path
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    mutated = mutate(value)
    _write_json(path, mutated)

    # Rebind only the outer online/file hashes and the manifest wrapper so the
    # test reaches semantic validation instead of stopping at a stale digest.
    manifest_path = artifacts.bank_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    field = "encoder" if artifact_name == "encoder" else "camera_layout"
    manifest[field] = {
        "file_sha256": sha256_file(path),
        "contract": mutated,
    }
    _write_json(manifest_path, manifest)
    online_value = artifacts.online_contract.to_dict()
    online_value[f"{artifact_name}_contract_sha256"] = sha256_file(path)
    online_value["bank_manifest_sha256"] = sha256_file(manifest_path)
    online = WarmOnlineRunContract.from_dict(online_value)
    source_value = artifacts.source_contract.to_dict()
    source_value["bank_manifest_sha256"] = sha256_file(manifest_path)
    source = WarmSourceRunContract.from_dict(source_value)
    online_value = online.to_dict()
    online_value["training_run_contract_sha256"] = source.sha256
    online = WarmOnlineRunContract.from_dict(online_value)

    rebound = _Artifacts(
        **{
            **artifacts.__dict__,
            "source_contract": source,
            "online_contract": online,
        }
    )
    with pytest.raises(OnlineArtifactContractError, match=message):
        _load_retriever(rebound)


def test_post_contract_artifact_tamper_is_rejected_before_dino_use(
    tmp_path: Path,
) -> None:
    artifacts = _make_artifacts(tmp_path)
    artifacts.encoder_contract_path.write_bytes(
        artifacts.encoder_contract_path.read_bytes() + b" "
    )
    with pytest.raises(OnlineArtifactContractError, match="encoder contract hash"):
        _load_retriever(artifacts)
    assert artifacts.dino.seen_frames == []
