from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

if os.environ.get("WARM_REQUIRE_TORCH_TESTS") == "1":
    try:
        import torch
    except ModuleNotFoundError as error:  # pragma: no cover - server contract
        raise RuntimeError(
            "WARM_REQUIRE_TORCH_TESTS=1 but PyTorch is unavailable"
        ) from error
else:
    torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader

from fastwam.datasets.lerobot.episode_catalog import (
    DatasetDescriptor,
    EpisodeCatalog,
    EpisodeRecord,
)
from fastwam.datasets.lerobot.audit import (
    audit_lerobot_catalog,
    write_audit_report,
)
from fastwam.datasets.warm_candidates import (
    RuntimeCandidateDatasetAdapter,
    RuntimeCandidateDatasetContractError,
    RuntimeCandidateMissingRowError,
    WARM_CANDIDATE_EVENT_INDEX,
    WARM_CANDIDATE_MASK,
    WARM_CANDIDATE_MU,
    WARM_CANDIDATE_SCORE,
    WARM_ORACLE_CANDIDATE_INDEX,
)
from fastwam.memory.action_contract import ActionSpaceContract
from fastwam.memory.manifest import sha256_file
from fastwam.memory.runtime_candidates import INVALID_BANK_ROW, RuntimeCandidateResolver
from tests.test_runtime_candidates import (
    AUDIT_HASH,
    QUERY_CORPUS_HASH,
    _action_contract,
    _write_artifacts,
)


def _catalog(*, split: str = "train", length: int = 8) -> EpisodeCatalog:
    descriptor = DatasetDescriptor(
        dataset_id="libero",
        dataset_index=0,
        fps=10.0,
        total_episodes=3,
        chunks_size=1000,
        data_path_template="data/episode_{episode_index:06d}.parquet",
        info_sha256="7" * 64,
        episodes_sha256="8" * 64,
    )
    episodes = tuple(
        EpisodeRecord(
            dataset_id="libero",
            dataset_index=0,
            episode_index=episode_index,
            length=length,
            fps=10.0,
            tasks=("test task",),
            data_relpath=f"data/episode_{episode_index:06d}.parquet",
            split=split,
        )
        for episode_index in range(3)
    )
    return EpisodeCatalog((descriptor,), episodes)


def _sample(frame_index: int, *, length: int = 8) -> dict[str, object]:
    horizon = 4
    action_offsets = torch.arange(horizon) + frame_index
    state_offsets = torch.arange(horizon + 1) + frame_index
    return {
        "action": torch.zeros((horizon, 3), dtype=torch.float32),
        "action_is_pad": action_offsets >= length,
        "image_is_pad": state_offsets >= length,
        "proprio_is_pad": state_offsets >= length,
        "dataset_index": torch.tensor(0, dtype=torch.int64),
        "episode_index": torch.tensor(0, dtype=torch.int64),
        "frame_index": torch.tensor(frame_index, dtype=torch.int64),
        "marker": torch.tensor(frame_index, dtype=torch.int64),
    }


class _LerobotDatasetContract:
    def __init__(self, *, stride: int = 1) -> None:
        self.global_sample_stride = stride
        self.action_size = 4
        self.processor = object()


class _BaseDataset(torch.utils.data.Dataset):
    def __init__(self, samples: list[dict[str, object]], *, stride: int = 1) -> None:
        self.samples = samples
        self.lerobot_dataset = _LerobotDatasetContract(stride=stride)
        self.compatibility_marker = "delegated"

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.samples[index]


class ConcatLeftAlign:
    __module__ = "fastwam.datasets.lerobot.transforms.action_state_merger"

    def __init__(self) -> None:
        self.action_target_dim = None
        self.state_target_dim = None


class LinearNormalizer:
    __module__ = "fastwam.datasets.lerobot.utils.normalizer"


class FastWAMProcessor:
    __module__ = "fastwam.datasets.lerobot.processors.fastwam_processor"

    def __init__(self, *, gripper_is_delta: bool = False) -> None:
        self.action_output_dim = 7
        self.proprio_output_dim = 8
        self.num_obs_steps = 5
        self.num_output_cameras = 2
        self.use_stepwise_action_norm = False
        self.norm_default_mode = "min/max"
        self.norm_exception_mode = None
        self.action_state_transforms = None
        self.shape_meta = {
            "images": [
                {
                    "key": "image",
                    "raw_shape": [3, 512, 512],
                    "shape": [3, 224, 224],
                },
                {
                    "key": "wrist_image",
                    "raw_shape": [3, 512, 512],
                    "shape": [3, 224, 224],
                },
            ],
            "action": [
                {"key": "default", "raw_shape": 7, "shape": 7}
            ],
            "state": [
                {"key": "default", "raw_shape": 8, "shape": 8}
            ],
        }
        self.delta_action_dim_mask = {
            "default": torch.tensor(
                [True, True, True, True, True, True, gripper_is_delta],
                dtype=torch.bool,
            )
        }
        self.action_state_merger = ConcatLeftAlign()
        self.normalizer = LinearNormalizer()


_BoundProcessor = FastWAMProcessor


class _BoundLerobotDataset:
    def __init__(
        self,
        roots: list[Path],
        catalog: EpisodeCatalog,
        *,
        processor: object | None = None,
    ) -> None:
        self.global_sample_stride = 1
        self.action_size = 4
        self.strict_sample_loading = True
        self.skip_padding_as_possible = False
        self.dataset_dirs = [str(root) for root in roots]
        descriptors = tuple(
            sorted(catalog.datasets, key=lambda item: item.dataset_index)
        )
        self.multi_dataset = SimpleNamespace(
            _datasets=[
                SimpleNamespace(
                    meta=SimpleNamespace(total_episodes=item.total_episodes),
                    episodes=None,
                )
                for item in descriptors
            ]
        )
        self.processor = _BoundProcessor() if processor is None else processor


class _BoundDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        sample: dict[str, object],
        lerobot_dataset: _BoundLerobotDataset,
        stats_path: Path,
    ) -> None:
        self.samples = [sample]
        self.lerobot_dataset = lerobot_dataset
        self.strict_sample_loading = True
        self.skip_padding_as_possible = False
        self.num_frames = 5
        self.video_size = [224, 448]
        self.concat_multi_camera = "horizontal"
        self.pretrained_norm_stats_path = str(stats_path.resolve())
        self.pretrained_norm_stats_sha256 = sha256_file(stats_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.samples[index]


def _bound_catalog_and_roots(
    tmp_path: Path,
    *,
    dataset_count: int = 1,
) -> tuple[EpisodeCatalog, list[Path]]:
    descriptors: list[DatasetDescriptor] = []
    records: list[EpisodeRecord] = []
    roots: list[Path] = []
    for dataset_index in range(dataset_count):
        dataset_id = "libero" if dataset_index == 0 else f"libero-{dataset_index}"
        root = tmp_path / f"dataset-{dataset_index}"
        meta = root / "meta"
        meta.mkdir(parents=True)
        info = meta / "info.json"
        episodes = meta / "episodes.jsonl"
        info.write_text(
            json.dumps(
                {
                    "features": {},
                    "total_frames": 24,
                    "total_tasks": 1,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        episodes.write_text(
            "".join(
                json.dumps(
                    {"episode_index": episode_index, "length": 8},
                    sort_keys=True,
                )
                + "\n"
                for episode_index in range(3)
            ),
            encoding="utf-8",
        )
        data = root / "data"
        data.mkdir()
        for episode_index in range(3):
            (data / f"episode_{episode_index:06d}.parquet").write_bytes(
                f"{dataset_id}:{episode_index}\n".encode("utf-8")
            )
        descriptors.append(
            DatasetDescriptor(
                dataset_id=dataset_id,
                dataset_index=dataset_index,
                fps=10.0,
                total_episodes=3,
                chunks_size=1000,
                data_path_template="data/episode_{episode_index:06d}.parquet",
                info_sha256=sha256_file(info),
                episodes_sha256=sha256_file(episodes),
            )
        )
        records.extend(
            EpisodeRecord(
                dataset_id=dataset_id,
                dataset_index=dataset_index,
                episode_index=episode_index,
                length=8,
                fps=10.0,
                tasks=("test task",),
                data_relpath=f"data/episode_{episode_index:06d}.parquet",
                split="train",
            )
            for episode_index in range(3)
        )
        roots.append(root)
    return EpisodeCatalog(tuple(descriptors), tuple(records)), roots


def _bound_fixture(
    tmp_path: Path,
    *,
    dataset_count: int = 1,
    processor: object | None = None,
) -> tuple[
    RuntimeCandidateResolver,
    EpisodeCatalog,
    list[Path],
    Path,
    Path,
    _BoundDataset,
]:
    catalog, roots = _bound_catalog_and_roots(
        tmp_path, dataset_count=dataset_count
    )
    audit_document = audit_lerobot_catalog(
        catalog,
        roots,
        hash_episode_tables=True,
    )
    audit_path = tmp_path / "audit.json"
    write_audit_report(audit_document, audit_path)
    stats_path = tmp_path / "dataset_stats.json"
    stats_path.write_text('{"fixture":"normalizer"}\n', encoding="utf-8")
    action_contract = ActionSpaceContract(
        action_dim=7,
        arm_dims=(0, 1, 2, 3, 4, 5),
        gripper_dims=(6,),
        gripper_threshold=0.0,
        normalization_mode="global:min/max",
        normalization_stats_sha256=sha256_file(stats_path),
        control_mode="libero_delta_eef_axis_angle_plus_gripper",
        embodiment="libero_panda",
    )
    bank_actions = torch.arange(3 * 4 * 7, dtype=torch.float32).reshape(
        3, 4, 7
    ).numpy()
    bank_path, cache_path, _ = _write_artifacts(
        tmp_path / "artifacts",
        catalog_hash=catalog.content_sha256,
        audit_hash=str(audit_document["report_sha256"]),
        action_contract=action_contract,
        actions=bank_actions,
    )
    resolver = RuntimeCandidateResolver.from_artifacts(
        bank_path,
        cache_path,
        expected_query_split="train",
        expected_query_corpus_sha256=QUERY_CORPUS_HASH,
        expected_action_space=action_contract,
    )
    lerobot_dataset = _BoundLerobotDataset(
        roots,
        catalog,
        processor=processor,
    )
    sample = _sample(0)
    sample["action"] = torch.zeros((4, 7), dtype=torch.float32)
    dataset = _BoundDataset(sample, lerobot_dataset, stats_path)
    return (
        resolver,
        catalog,
        roots,
        stats_path,
        audit_path,
        dataset,
    )


def _resolver_and_catalog(
    tmp_path: Path,
) -> tuple[RuntimeCandidateResolver, EpisodeCatalog]:
    catalog = _catalog()
    bank_path, cache_path, _ = _write_artifacts(
        tmp_path,
        catalog_hash=catalog.content_sha256,
    )
    resolver = RuntimeCandidateResolver.from_artifacts(
        bank_path,
        cache_path,
        expected_query_split="train",
        expected_query_corpus_sha256=QUERY_CORPUS_HASH,
        expected_action_space=_action_contract(),
    )
    assert resolver.query_audit_sha256 == AUDIT_HASH
    return resolver, catalog


def test_adapter_resolves_exact_query_and_default_collates_fixed_shapes(
    tmp_path: Path,
) -> None:
    resolver, catalog = _resolver_and_catalog(tmp_path)
    base = _BaseDataset([_sample(0), _sample(1)])
    adapter = RuntimeCandidateDatasetAdapter(base, resolver, catalog)

    first = adapter[0]
    assert first[WARM_CANDIDATE_MU].shape == (3, 4, 3)
    assert first[WARM_CANDIDATE_MU].dtype == torch.float32
    assert first[WARM_CANDIDATE_MASK].tolist() == [True, True, False]
    assert first[WARM_CANDIDATE_SCORE].tolist() == pytest.approx([0.8, 0.7, 0.0])
    assert first[WARM_CANDIDATE_EVENT_INDEX].tolist() == [
        1,
        2,
        INVALID_BANK_ROW,
    ]
    assert first[WARM_ORACLE_CANDIDATE_INDEX].item() == 0
    assert adapter.lerobot_dataset is base.lerobot_dataset
    assert adapter.lerobot_dataset.processor is base.lerobot_dataset.processor
    assert adapter.compatibility_marker == "delegated"
    assert WARM_CANDIDATE_MU not in base.samples[0]

    batch = next(iter(DataLoader(adapter, batch_size=2, shuffle=False)))
    assert batch[WARM_CANDIDATE_MU].shape == (2, 3, 4, 3)
    assert batch[WARM_CANDIDATE_MASK].shape == (2, 3)
    assert batch[WARM_CANDIDATE_SCORE].shape == (2, 3)
    assert batch[WARM_CANDIDATE_EVENT_INDEX].shape == (2, 3)
    assert batch[WARM_ORACLE_CANDIDATE_INDEX].shape == (2,)
    assert batch[WARM_ORACLE_CANDIDATE_INDEX].tolist() == [0, -1]
    # Frame one has a factual cache row whose candidate list is genuinely empty.
    assert not bool(batch[WARM_CANDIDATE_MASK][1].any().item())


def test_missing_non_padded_row_is_a_hard_error(tmp_path: Path) -> None:
    resolver, catalog = _resolver_and_catalog(tmp_path)
    adapter = RuntimeCandidateDatasetAdapter(
        _BaseDataset([_sample(2)]),
        resolver,
        catalog,
    )

    with pytest.raises(RuntimeCandidateMissingRowError, match="non-padded"):
        adapter[0]


def test_only_catalog_verified_padded_tail_may_use_null_row(tmp_path: Path) -> None:
    resolver, catalog = _resolver_and_catalog(tmp_path)
    adapter = RuntimeCandidateDatasetAdapter(
        _BaseDataset([_sample(4)]),
        resolver,
        catalog,
    )

    sample = adapter[0]
    assert not bool(sample[WARM_CANDIDATE_MASK].any().item())
    assert sample[WARM_CANDIDATE_EVENT_INDEX].tolist() == [INVALID_BANK_ROW] * 3
    assert torch.count_nonzero(sample[WARM_CANDIDATE_MU]).item() == 0
    assert torch.count_nonzero(sample[WARM_CANDIDATE_SCORE]).item() == 0
    assert sample[WARM_ORACLE_CANDIDATE_INDEX].item() == -1


def test_adapter_rejects_false_padding_claim_and_nonunit_stride(tmp_path: Path) -> None:
    resolver, catalog = _resolver_and_catalog(tmp_path)
    false_tail = _sample(2)
    false_tail["image_is_pad"] = torch.tensor(
        [False, False, False, False, True], dtype=torch.bool
    )
    adapter = RuntimeCandidateDatasetAdapter(
        _BaseDataset([false_tail]),
        resolver,
        catalog,
    )
    with pytest.raises(
        RuntimeCandidateDatasetContractError,
        match="padding masks disagree",
    ):
        adapter[0]

    with pytest.raises(
        RuntimeCandidateDatasetContractError,
        match="global_sample_stride=1",
    ):
        RuntimeCandidateDatasetAdapter(
            _BaseDataset([_sample(0)], stride=2),
            resolver,
            catalog,
        )


def test_adapter_rejects_catalog_not_bound_to_candidate_cache(tmp_path: Path) -> None:
    resolver, _ = _resolver_and_catalog(tmp_path)
    different_catalog = _catalog(length=9)

    with pytest.raises(
        RuntimeCandidateDatasetContractError,
        match="does not match the candidate-cache",
    ):
        RuntimeCandidateDatasetAdapter(
            _BaseDataset([_sample(0, length=9)]),
            resolver,
            different_catalog,
        )


def test_formal_binding_accepts_exact_stats_audit_and_dataset_roots(
    tmp_path: Path,
) -> None:
    resolver, catalog, _, stats_path, audit_path, dataset = _bound_fixture(
        tmp_path
    )

    adapter = RuntimeCandidateDatasetAdapter(
        dataset,
        resolver,
        catalog,
        normalization_stats_path=stats_path,
        audit_report_path=audit_path,
    )

    assert adapter[0][WARM_ORACLE_CANDIDATE_INDEX].item() == 0


def test_normalization_stats_tamper_fails_closed(tmp_path: Path) -> None:
    resolver, catalog, _, stats_path, audit_path, dataset = _bound_fixture(
        tmp_path
    )
    stats_path.write_text('{"fixture":"tampered"}\n', encoding="utf-8")

    with pytest.raises(
        RuntimeCandidateDatasetContractError,
        match="normalization stats file does not match",
    ):
        RuntimeCandidateDatasetAdapter(
            dataset,
            resolver,
            catalog,
            normalization_stats_path=stats_path,
            audit_report_path=audit_path,
        )


def test_loaded_normalization_snapshot_mismatch_fails_closed(
    tmp_path: Path,
) -> None:
    resolver, catalog, _, stats_path, audit_path, dataset = _bound_fixture(
        tmp_path
    )
    dataset.pretrained_norm_stats_sha256 = "0" * 64

    with pytest.raises(
        RuntimeCandidateDatasetContractError,
        match="did not load the exact contract-bound",
    ):
        RuntimeCandidateDatasetAdapter(
            dataset,
            resolver,
            catalog,
            normalization_stats_path=stats_path,
            audit_report_path=audit_path,
        )


def test_audit_report_tamper_fails_closed(tmp_path: Path) -> None:
    resolver, catalog, _, stats_path, audit_path, dataset = _bound_fixture(
        tmp_path
    )
    document = json.loads(audit_path.read_text(encoding="utf-8"))
    document["summary"]["dataset_count"] += 1
    audit_path.write_text(json.dumps(document) + "\n", encoding="utf-8")

    with pytest.raises(
        RuntimeCandidateDatasetContractError,
        match="cannot validate WARM audit report",
    ):
        RuntimeCandidateDatasetAdapter(
            dataset,
            resolver,
            catalog,
            normalization_stats_path=stats_path,
            audit_report_path=audit_path,
        )


def test_dataset_root_reordering_fails_closed(tmp_path: Path) -> None:
    resolver, catalog, roots, stats_path, audit_path, dataset = _bound_fixture(
        tmp_path,
        dataset_count=2,
    )
    dataset.lerobot_dataset.dataset_dirs = [
        str(root) for root in reversed(roots)
    ]

    with pytest.raises(
        RuntimeCandidateDatasetContractError,
        match="root/order metadata does not match",
    ):
        RuntimeCandidateDatasetAdapter(
            dataset,
            resolver,
            catalog,
            normalization_stats_path=stats_path,
            audit_report_path=audit_path,
        )


def test_processor_action_semantics_mismatch_fails_closed(tmp_path: Path) -> None:
    bad_processor = _BoundProcessor(gripper_is_delta=True)
    resolver, catalog, _, stats_path, audit_path, dataset = _bound_fixture(
        tmp_path,
        processor=bad_processor,
    )

    with pytest.raises(
        RuntimeCandidateDatasetContractError,
        match="arm/gripper delta semantics",
    ):
        RuntimeCandidateDatasetAdapter(
            dataset,
            resolver,
            catalog,
            normalization_stats_path=stats_path,
            audit_report_path=audit_path,
        )


def test_runtime_oracle_uses_m1_action_distance_not_uniform_mse(
    tmp_path: Path,
) -> None:
    catalog = _catalog()
    actions = torch.zeros((3, 4, 3), dtype=torch.float32).numpy()
    actions[1, :, 2] = -0.001
    actions[2, :, :2] = 0.2
    actions[2, :, 2] = 1.0
    bank_path, cache_path, _ = _write_artifacts(
        tmp_path,
        catalog_hash=catalog.content_sha256,
        actions=actions,
    )
    resolver = RuntimeCandidateResolver.from_artifacts(
        bank_path,
        cache_path,
        expected_query_split="train",
        expected_query_corpus_sha256=QUERY_CORPUS_HASH,
        expected_action_space=_action_contract(),
    )
    sample = _sample(0)
    sample["action"][:, 2] = 0.01
    adapter = RuntimeCandidateDatasetAdapter(
        _BaseDataset([sample]), resolver, catalog
    )

    selected = adapter[0][WARM_ORACLE_CANDIDATE_INDEX].item()

    # Uniform all-dimension MSE prefers slot 0 because its gripper magnitude
    # is numerically closer.  The M1 metric correctly prefers slot 1: its arm
    # error is small and its thresholded gripper state is semantically right.
    assert selected == 1
