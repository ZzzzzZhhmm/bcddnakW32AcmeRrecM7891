from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from fastwam.datasets.lerobot.audit import (
    CameraVideoAuditProof,
    EpisodeAuditProof,
    LerobotAuditReport,
    compute_source_bundle_sha256,
)
from fastwam.datasets.lerobot.episode_catalog import (
    DatasetDescriptor,
    EpisodeCatalog,
    EpisodeRecord,
)
from fastwam.memory.action_contract import ActionSpaceContract
from fastwam.memory.bank_builder import EpisodeFeatures
from fastwam.memory.candidate_cache import QueryId
from fastwam.memory.feature_cache import (
    FeatureCacheMetadata,
    load_episode_feature_cache,
    save_episode_feature_cache,
)
from fastwam.memory.manifest import sha256_canonical_json
from fastwam.memory.offline_pipeline import load_feature_cache_collection
from fastwam.memory.runtime_candidates import ResolvedCandidateRow
from fastwam.memory.schema import EventId
from fastwam.models.warm.online_contract import (
    ONLINE_RETRIEVAL_IMPLEMENTATION,
    WarmOnlineRunContract,
)
from fastwam.models.warm.source_contract import WarmSourceRunContract
from scripts.validate_warm_online_parity import (
    MAX_BFLOAT16_ATOL,
    OnlineParityError,
    _artifact_snapshot,
    _assert_shared_contracts,
    _task_raw_artifact_snapshot,
    _validate_compute_device,
    _validate_numeric_atol,
    _write_json_atomic,
    build_parser,
    validate_online_parity,
)


TASK = "move the red block"
SOURCE_CAMERAS = (
    "observation.images.image",
    "observation.images.wrist_image",
)
CAMERA_MAPPING = {
    SOURCE_CAMERAS[0]: "image",
    SOURCE_CAMERAS[1]: "wrist_image",
}


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _source_digest(label: str) -> tuple[str, tuple[CameraVideoAuditProof, ...]]:
    cameras = tuple(
        CameraVideoAuditProof(key, _digest(f"{label}:{key}"))
        for key in SOURCE_CAMERAS
    )
    return compute_source_bundle_sha256(_digest(f"{label}:table"), cameras), cameras


@dataclass
class _Fixture:
    collection: object
    catalog: EpisodeCatalog
    audit: LerobotAuditReport
    training: WarmSourceRunContract
    validation: WarmSourceRunContract
    online: WarmOnlineRunContract
    resolver: object
    retriever: object
    dev_source_sha256: str
    full_images: MappingProxyType
    normalizer_sha256: str


class _Resolver:
    def __init__(
        self,
        *,
        training: WarmSourceRunContract,
        validation: WarmSourceRunContract,
        collection_hash: str,
        action_space: ActionSpaceContract,
        rows: dict[QueryId, ResolvedCandidateRow],
        means: dict[QueryId, np.ndarray],
    ) -> None:
        self.query_split = "dev"
        self.query_stride = 1
        self.candidate_manifest_sha256 = validation.candidate_manifest_sha256
        self.query_corpus_sha256 = collection_hash
        self.bank_manifest_sha256 = training.bank_manifest_sha256
        self.bank_content_sha256 = training.bank_content_sha256
        self.query_catalog_sha256 = training.catalog_sha256
        self.query_audit_sha256 = training.audit_sha256
        self.fixed_k = 2
        self.action_horizon = 2
        self.action_space = action_space
        self._rows = rows
        self._means = means

    def resolve(self, query_id: QueryId, *, allow_missing: bool) -> ResolvedCandidateRow:
        assert allow_missing is False
        return self._rows[query_id]

    def gather_payload(
        self, row: ResolvedCandidateRow, payload_name: str, *, fill_value: int
    ) -> np.ndarray:
        assert payload_name == "model_space_action"
        assert fill_value == 0
        return np.array(self._means[row.query_id], copy=True)


class _Retriever:
    def __init__(
        self,
        online: WarmOnlineRunContract,
        keys: dict[int, np.ndarray],
        rows: dict[int, ResolvedCandidateRow],
        means: dict[int, np.ndarray],
        *,
        mutation: str | None = None,
    ) -> None:
        self.artifact_verified = True
        self.online_run_contract = online
        self.task_vocabulary = (TASK,)
        self._keys = keys
        self._rows = rows
        self._means = means
        self._episode = -1
        self._mutation = mutation
        self.consumed = 0

    def begin_episode(self, episode_index: int) -> None:
        self._episode = episode_index

    def make_query_id(self, frame_index: int) -> QueryId:
        return QueryId("synthetic-online", 0, self._episode, frame_index)

    def retrieve(
        self,
        query_id: QueryId,
        raw_cameras: object,
        *,
        task_description: str,
        prompt: str,
        proprio: np.ndarray,
    ) -> object:
        assert set(raw_cameras) == {"image", "wrist_image"}
        assert task_description == TASK and prompt == TASK
        frame = query_id.frame_index
        row = self._rows[frame]
        context_key = np.array(self._keys[frame], copy=True)
        bank_rows = np.array(row.bank_rows, copy=True)
        event_ids = tuple(row.event_ids)
        scores = row.cosine_scores.astype(np.float64)
        means = np.array(self._means[frame], copy=True)
        if self._mutation == "key":
            context_key[0] += np.float32(0.01)
        elif self._mutation == "row":
            bank_rows[0], bank_rows[1] = bank_rows[1], bank_rows[0]
        elif self._mutation == "event":
            event_ids = (event_ids[1], event_ids[0])
        elif self._mutation == "score":
            scores[0] += 0.01
        elif self._mutation == "payload":
            means[0, 0, 0] += np.float32(1.0)
        return SimpleNamespace(
            context_key=context_key,
            candidate_valid_mask=np.array(row.mask, copy=True),
            bank_rows=bank_rows,
            event_ids=event_ids,
            cosine_scores=scores,
            candidate_means=means,
            task_index=0,
            model_input=np.zeros((1, 3, 224, 448), dtype=np.float32),
        )

    def validate_bound_step(self, step: object, **kwargs: object) -> object:
        self.consumed += 1
        return step


def _make_fixture(tmp_path: Path, *, mutation: str | None = None) -> _Fixture:
    descriptor = DatasetDescriptor(
        dataset_id="synthetic",
        dataset_index=0,
        fps=20.0,
        total_episodes=2,
        chunks_size=1000,
        data_path_template="data/{episode_index}.parquet",
        info_sha256=_digest("info"),
        episodes_sha256=_digest("episodes"),
    )
    train_source, train_cameras = _source_digest("train")
    dev_source, dev_cameras = _source_digest("dev")
    train_record = EpisodeRecord(
        "synthetic", 0, 0, 4, 20.0, (TASK,), "data/0.parquet", "train"
    )
    dev_record = EpisodeRecord(
        "synthetic", 0, 1, 4, 20.0, (TASK,), "data/1.parquet", "dev"
    )
    catalog = EpisodeCatalog((descriptor,), (train_record, dev_record))
    audit = LerobotAuditReport(
        catalog_sha256=catalog.content_sha256,
        report_sha256=_digest("audit-report"),
        episode_tables_hashed=True,
        audited_camera_keys=SOURCE_CAMERAS,
        cross_split_duplicate_count=0,
        cross_split_table_duplicate_count=0,
        cross_split_video_duplicate_count=0,
        cross_split_camera_bundle_duplicate_count=0,
        cross_split_source_bundle_duplicate_count=0,
        episode_proofs=(
            EpisodeAuditProof(
                "synthetic",
                0,
                0,
                "train",
                train_source,
                _digest("train:table"),
                train_cameras,
            ),
            EpisodeAuditProof(
                "synthetic",
                0,
                1,
                "dev",
                dev_source,
                _digest("dev:table"),
                dev_cameras,
            ),
        ),
    )

    keys = np.asarray(
        [[1.0, 0.0], [0.8, 0.2], [0.5, 0.5], [0.0, 1.0]],
        dtype=np.float32,
    )
    features = EpisodeFeatures(
        dataset_id="synthetic",
        dataset_index=0,
        episode_index=1,
        task_index=0,
        source_episode_sha256=dev_source,
        model_actions=np.arange(6, dtype=np.float32).reshape(3, 2),
        proprio=np.zeros((4, 2), dtype=np.float32),
        gripper=np.zeros((4,), dtype=np.float32),
        context_keys=keys,
        semantic_features=np.ones((4, 4, 2), dtype=np.float32),
    )
    normalizer_sha = _digest("normalizer-file")
    encoder_sha = _digest("encoder-file")
    camera_sha = _digest("camera-file")
    payload = tmp_path / "dev.npz"
    save_episode_feature_cache(
        payload,
        features,
        metadata=FeatureCacheMetadata.for_episode(
            features,
            split="dev",
            catalog_hash=catalog.content_sha256,
            normalizer_hash=normalizer_sha,
            encoder_hash=encoder_sha,
            camera_hash=camera_sha,
        ),
    )
    assert load_episode_feature_cache(payload).metadata.split == "dev"
    collection = load_feature_cache_collection((payload,))

    stats_sha = _digest("stats")
    action_space = ActionSpaceContract(
        action_dim=2,
        arm_dims=(0,),
        gripper_dims=(1,),
        gripper_threshold=0.0,
        normalization_mode="standard-score",
        normalization_stats_sha256=stats_sha,
        control_mode="delta-ee",
        embodiment="synthetic",
    )
    action_sha = sha256_canonical_json(action_space.to_dict())
    shared = dict(
        bank_manifest_sha256=_digest("bank-manifest"),
        bank_content_sha256=_digest("bank-content"),
        catalog_sha256=catalog.content_sha256,
        audit_sha256=audit.report_sha256,
        normalization_stats_sha256=stats_sha,
        action_space_contract_sha256=action_sha,
        base_checkpoint_sha256=_digest("base"),
        global_sample_stride=1,
        action_horizon=2,
        action_dim=2,
    )
    training = WarmSourceRunContract(
        candidate_manifest_sha256=_digest("train-candidates"),
        query_corpus_sha256=_digest("train-query-corpus"),
        query_split="train",
        **shared,
    )
    validation = WarmSourceRunContract(
        candidate_manifest_sha256=_digest("dev-candidates"),
        query_corpus_sha256=collection.content_hash,
        query_split="dev",
        **shared,
    )
    online = WarmOnlineRunContract(
        training_run_contract_sha256=training.sha256,
        validation_run_contract_sha256=validation.sha256,
        warm_checkpoint_sha256=_digest("warm"),
        training_attestation_sha256=_digest("training-attestation"),
        shared_training_recipe_sha256=_digest("shared-training-recipe"),
        training_runtime_sha256=_digest("training-runtime"),
        bank_manifest_sha256=training.bank_manifest_sha256,
        bank_content_sha256=training.bank_content_sha256,
        encoder_contract_sha256=encoder_sha,
        encoder_runtime_sha256=_digest("encoder-runtime"),
        camera_contract_sha256=camera_sha,
        m1_data_config_sha256=_digest("m1-data-config"),
        dino_checkpoint_tree_sha256=_digest("dino"),
        dino_checkpoint_file_count=1,
        normalization_stats_sha256=stats_sha,
        action_space_contract_sha256=action_sha,
        catalog_sha256=catalog.content_sha256,
        audit_sha256=audit.report_sha256,
        resolved_eval_config_sha256=_digest("eval"),
        vae_checkpoint_sha256=_digest("vae"),
        text_encoder_tree_sha256=_digest("text"),
        tokenizer_tree_sha256=_digest("tokenizer"),
        evaluation_namespace_sha256=_digest("namespace"),
        task_suite="synthetic_suite",
        task_id=0,
        task_description=TASK,
        root_seed=7,
        initial_states_sha256=_digest("initial"),
        bddl_sha256=_digest("bddl"),
        retrieval_implementation=ONLINE_RETRIEVAL_IMPLEMENTATION,
        top_k=2,
        source_policy="fixed_context_top1",
        memory_sigma=0.2,
        action_horizon=2,
        action_dim=2,
        git_commit="a" * 40,
        git_dirty=False,
    )

    event_ids = (EventId("synthetic", 0, 0, 0), EventId("synthetic", 0, 0, 1))
    rows: dict[QueryId, ResolvedCandidateRow] = {}
    means_by_query: dict[QueryId, np.ndarray] = {}
    rows_by_frame: dict[int, ResolvedCandidateRow] = {}
    means_by_frame: dict[int, np.ndarray] = {}
    for frame in (0, 1):
        query = QueryId("synthetic", 0, 1, frame)
        row = ResolvedCandidateRow(
            query_id=query,
            bank_rows=np.asarray([0, 1], dtype=np.int64),
            mask=np.asarray([True, True], dtype=np.bool_),
            cosine_scores=np.asarray([0.9 - frame * 0.1, 0.4], dtype=np.float32),
            event_ids=event_ids,
        )
        means = np.asarray(
            [
                [[1.0, 0.0], [1.0, 0.0]],
                [[0.0, 1.0], [0.0, 1.0]],
            ],
            dtype=np.float32,
        )
        rows[query] = row
        means_by_query[query] = means
        rows_by_frame[frame] = row
        means_by_frame[frame] = means
    resolver = _Resolver(
        training=training,
        validation=validation,
        collection_hash=collection.content_hash,
        action_space=action_space,
        rows=rows,
        means=means_by_query,
    )
    retriever = _Retriever(
        online,
        {0: keys[0], 1: keys[1]},
        rows_by_frame,
        means_by_frame,
        mutation=mutation,
    )
    images = MappingProxyType(
        {
            SOURCE_CAMERAS[0]: np.full(
                (4, 3, 8, 8), np.float32(64.0 / 255.0), dtype=np.float32
            ),
            SOURCE_CAMERAS[1]: np.full(
                (4, 3, 8, 8), np.float32(192.0 / 255.0), dtype=np.float32
            ),
        }
    )
    return _Fixture(
        collection=collection,
        catalog=catalog,
        audit=audit,
        training=training,
        validation=validation,
        online=online,
        resolver=resolver,
        retriever=retriever,
        dev_source_sha256=dev_source,
        full_images=images,
        normalizer_sha256=normalizer_sha,
    )


def _run(fixture: _Fixture, tmp_path: Path):
    def loader(*args: object) -> object:
        return SimpleNamespace(
            source_episode_sha256=fixture.dev_source_sha256,
            images=fixture.full_images,
        )

    return validate_online_parity(
        retriever=fixture.retriever,
        resolver=fixture.resolver,
        collection=fixture.collection,
        catalog=fixture.catalog,
        audit=fixture.audit,
        training_run_contract=fixture.training,
        validation_run_contract=fixture.validation,
        normalizer_contract_sha256=fixture.normalizer_sha256,
        dataset_roots=(tmp_path,),
        source_camera_keys=SOURCE_CAMERAS,
        processor_camera_mapping=CAMERA_MAPPING,
        timestamp_tolerance_s=1e-4,
        video_backend="pyav",
        bfloat16_atol=0.0,
        episode_loader=loader,
    )


def test_full_synthetic_dev_parity_passes_and_consumes_every_query(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    counts = _run(fixture, tmp_path)
    assert counts.dev_episode_count == 1
    assert counts.task_episode_count == 1
    assert counts.query_count == 2
    assert counts.candidate_slot_count == 4
    assert counts.valid_candidate_count == 4
    assert counts.exact_context_key_count == 2
    assert counts.exact_score_roundtrip_count == 2
    assert counts.max_context_key_abs_error == 0.0
    assert len(counts.parity_transcript_sha256) == 64
    assert fixture.retriever.consumed == 2


def test_parity_cli_declares_m1_full_bank_search_semantics() -> None:
    description = build_parser().description
    assert description is not None
    assert "M1 full-bank exact-cosine top-K" in description
    assert "task-filtered" not in description


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("key", "context key"),
        ("row", "bank-row"),
        ("event", "EventId"),
        ("score", "cosine scores"),
        ("payload", "action payload"),
    ],
)
def test_parity_fails_closed_on_every_query_identity_or_numeric_mismatch(
    tmp_path: Path, mutation: str, message: str
) -> None:
    fixture = _make_fixture(tmp_path, mutation=mutation)
    with pytest.raises(OnlineParityError, match=message):
        _run(fixture, tmp_path)


def test_train_or_non_stride_one_reference_is_rejected_before_decode(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    fixture.resolver.query_stride = 2
    with pytest.raises(OnlineParityError, match="query_stride"):
        _run(fixture, tmp_path)

    fixture = _make_fixture(tmp_path / "second")
    fake_train_collection = SimpleNamespace(
        split="train",
        contract=fixture.collection.contract,
        content_hash=fixture.collection.content_hash,
    )
    with pytest.raises(OnlineParityError, match="must be DEV"):
        _assert_shared_contracts(
            training=fixture.training,
            validation=fixture.validation,
            online=fixture.online,
            resolver=fixture.resolver,
            collection=fake_train_collection,
            normalizer_contract_sha256=fixture.normalizer_sha256,
        )


def test_contract_identity_mismatch_is_rejected(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    fixture.resolver.candidate_manifest_sha256 = _digest("wrong-dev-candidates")
    with pytest.raises(OnlineParityError, match="candidate manifest"):
        _run(fixture, tmp_path)


def test_numeric_tolerance_is_only_available_for_tight_bfloat16_drift() -> None:
    bf16 = {"compute": {"dtype": "bfloat16"}}
    fp32 = {"compute": {"dtype": "float32"}}
    assert _validate_numeric_atol(0.0, fp32) == 0.0
    assert _validate_numeric_atol(1e-5, bf16) == 1e-5
    with pytest.raises(OnlineParityError, match="only for bfloat16"):
        _validate_numeric_atol(1e-5, fp32)
    with pytest.raises(OnlineParityError, match="must lie"):
        _validate_numeric_atol(MAX_BFLOAT16_ATOL * 2.0, bf16)


def test_server_device_must_exactly_match_encoder_compute_device() -> None:
    contract = {"compute": {"device": "cuda:0", "dtype": "bfloat16"}}
    assert _validate_compute_device("cuda:0", contract) == "cuda:0"
    with pytest.raises(OnlineParityError, match="exactly match"):
        _validate_compute_device("cuda", contract)


def test_raw_dev_snapshot_covers_table_and_every_camera_for_post_run_rehash(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    info_path = root / "meta" / "info.json"
    episodes_path = root / "meta" / "episodes.jsonl"
    info_path.parent.mkdir(parents=True)
    info_path.write_text(
        """{
          "chunks_size": 1000,
          "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
          "features": {
            "observation.images.image": {"dtype": "video"},
            "observation.images.wrist_image": {"dtype": "video"}
          }
        }\n""",
        encoding="utf-8",
    )
    episodes_path.write_text("{}\n", encoding="utf-8")
    table = root / "data" / "episode_000000.parquet"
    table.parent.mkdir(parents=True)
    table.write_bytes(b"table")
    camera_paths: list[Path] = []
    camera_proofs: list[CameraVideoAuditProof] = []
    for index, camera in enumerate(SOURCE_CAMERAS):
        path = root / "videos" / "chunk-000" / camera / "episode_000000.mp4"
        path.parent.mkdir(parents=True)
        payload = f"camera-{index}".encode("ascii")
        path.write_bytes(payload)
        camera_paths.append(path)
        camera_proofs.append(CameraVideoAuditProof(camera, hashlib.sha256(payload).hexdigest()))
    table_sha = hashlib.sha256(b"table").hexdigest()
    source_sha = compute_source_bundle_sha256(table_sha, tuple(camera_proofs))
    descriptor = DatasetDescriptor(
        "synthetic",
        0,
        20.0,
        1,
        1000,
        "data/episode_{episode_index:06d}.parquet",
        hashlib.sha256(info_path.read_bytes()).hexdigest(),
        hashlib.sha256(episodes_path.read_bytes()).hexdigest(),
    )
    record = EpisodeRecord(
        "synthetic",
        0,
        0,
        4,
        20.0,
        (TASK,),
        "data/episode_000000.parquet",
        "dev",
    )
    catalog = EpisodeCatalog((descriptor,), (record,))
    audit = LerobotAuditReport(
        catalog_sha256=catalog.content_sha256,
        report_sha256=_digest("raw-snapshot-audit"),
        episode_tables_hashed=True,
        audited_camera_keys=SOURCE_CAMERAS,
        cross_split_duplicate_count=0,
        cross_split_table_duplicate_count=0,
        cross_split_video_duplicate_count=0,
        cross_split_camera_bundle_duplicate_count=0,
        cross_split_source_bundle_duplicate_count=0,
        episode_proofs=(
            EpisodeAuditProof(
                "synthetic",
                0,
                0,
                "dev",
                source_sha,
                table_sha,
                tuple(camera_proofs),
            ),
        ),
    )
    paths, expected = _task_raw_artifact_snapshot(
        catalog=catalog,
        audit=audit,
        roots=(root,),
        task_description=TASK,
    )
    assert len(paths) == 3
    assert _artifact_snapshot(paths) == expected
    camera_paths[0].write_bytes(b"changed-after-read")
    assert _artifact_snapshot(paths) != expected


def test_report_writer_is_atomic_and_requires_explicit_overwrite(tmp_path: Path) -> None:
    report = tmp_path / "reports" / "parity.json"
    report.parent.mkdir()
    _write_json_atomic(report, {"status": "pass", "count": 2}, overwrite=False)
    assert report.read_text(encoding="utf-8").endswith("\n")
    with pytest.raises(FileExistsError):
        _write_json_atomic(report, {"status": "new"}, overwrite=False)
    _write_json_atomic(report, {"status": "new"}, overwrite=True)
    assert '"status": "new"' in report.read_text(encoding="utf-8")
    assert not list(report.parent.glob(".*.tmp"))
