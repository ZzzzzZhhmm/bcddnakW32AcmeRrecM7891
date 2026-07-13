from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from types import MappingProxyType, ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from fastwam.datasets.lerobot.audit import (
    audit_lerobot_catalog,
    load_audit_report,
    write_audit_report,
)
from fastwam.datasets.lerobot.episode_catalog import (
    EpisodeCatalog,
    assign_task_stratified_dev_split,
    scan_lerobot_datasets,
)
import scripts.precompute_warm_features as precompute_cli
from scripts.precompute_warm_features import main
from fastwam.memory.train_stats import (
    TRAIN_STATS_FILENAME,
    TRAIN_STATS_MANIFEST_FILENAME,
    FastWAMLiberoTrainStats,
    TrainEpisodeSource,
    TrainStatsManifest,
    compute_train_episode_set_sha256,
    encode_train_stats_json,
    encode_train_stats_manifest_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CAMERAS = (
    "observation.images.image",
    "observation.images.wrist_image",
)


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _write_dataset(root: Path) -> None:
    meta = root / "meta"
    data = root / "data" / "chunk-000"
    meta.mkdir(parents=True)
    data.mkdir(parents=True)
    info = {
        "fps": 20,
        "total_episodes": 2,
        "total_frames": 8,
        "total_tasks": 1,
        "chunks_size": 1000,
        "data_path": (
            "data/chunk-{episode_chunk:03d}/"
            "episode_{episode_index:06d}.parquet"
        ),
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/"
            "episode_{episode_index:06d}.mp4"
        ),
        "features": {
            "observation.images.image": {"dtype": "video"},
            "observation.images.wrist_image": {"dtype": "video"},
            "observation.state": {"dtype": "float32"},
            "action": {"dtype": "float32"},
        },
    }
    (meta / "info.json").write_text(
        json.dumps(info, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = [
        {"episode_index": index, "length": 4, "tasks": ["pick object"]}
        for index in range(2)
    ]
    (meta / "episodes.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    for index in range(2):
        (data / f"episode_{index:06d}.parquet").write_bytes(
            f"synthetic-parquet-{index}".encode("utf-8")
        )
        for camera in SOURCE_CAMERAS:
            video = (
                root / "videos" / "chunk-000" / camera / f"episode_{index:06d}.mp4"
            )
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(f"synthetic-video:{camera}:{index}".encode("utf-8"))


def _write_audit(catalog: EpisodeCatalog, root: Path, path: Path) -> None:
    report = audit_lerobot_catalog(
        catalog,
        [root],
        hash_episode_tables=True,
        camera_keys=SOURCE_CAMERAS,
    )
    write_audit_report(report, path)


def _fixture(tmp_path: Path) -> dict[str, Path | list[str]]:
    root = tmp_path / "dataset"
    _write_dataset(root)
    catalog = assign_task_stratified_dev_split(
        scan_lerobot_datasets([root], dataset_ids=["synthetic-libero"]),
        dev_per_task=1,
        seed=17,
    )
    assert {record.split for record in catalog.episodes} == {"train", "dev"}
    catalog_path = tmp_path / "catalog.json"
    catalog.save(catalog_path)
    audit_path = tmp_path / "audit.json"
    _write_audit(catalog, root, audit_path)

    data_config = tmp_path / "libero.yaml"
    data_config.write_text(
        "train:\n  processor:\n    _target_: synthetic.NotLoadedInPlanOnly\n",
        encoding="utf-8",
    )
    train_stats_root = tmp_path / "train-stats"
    train_stats_root.mkdir()
    dataset_stats = train_stats_root / TRAIN_STATS_FILENAME
    stats_value = FastWAMLiberoTrainStats(
        action_min=(-1.0,) * 7,
        action_max=(1.0,) * 7,
        state_min=(-1.0,) * 8,
        state_max=(1.0,) * 8,
        num_episodes=1,
        num_transition=4,
    )
    stats_payload = encode_train_stats_json(stats_value)
    dataset_stats.write_bytes(stats_payload)
    audit = load_audit_report(audit_path)
    train_sources = tuple(
        TrainEpisodeSource(
            record.dataset_id,
            record.dataset_index,
            record.episode_index,
            audit.proof_index[
                (record.dataset_id, record.dataset_index, record.episode_index)
            ].source_episode_sha256,
        )
        for record in catalog.episodes
        if record.split == "train"
    )
    stats_manifest = TrainStatsManifest(
        stats_file_sha256=sha256(stats_payload).hexdigest(),
        catalog_sha256=catalog.content_sha256,
        audit_report_sha256=audit.report_sha256,
        data_config_sha256=sha256(data_config.read_bytes()).hexdigest(),
        train_episode_set_sha256=compute_train_episode_set_sha256(train_sources),
        train_episode_count=1,
        train_frame_count=4,
        action_dim=7,
        state_dim=8,
        git_commit="1" * 40,
    )
    dataset_stats_manifest = train_stats_root / TRAIN_STATS_MANIFEST_FILENAME
    dataset_stats_manifest.write_bytes(
        encode_train_stats_manifest_json(stats_manifest)
    )
    dino = tmp_path / "dinov2-local"
    dino.mkdir()
    (dino / "config.json").write_text(
        '{"architectures":["Dinov2Model"]}\n', encoding="utf-8"
    )
    (dino / "model.safetensors").write_bytes(b"not-loaded-in-plan-only")
    output = tmp_path / "features"
    args = [
        "--data-config",
        str(data_config),
        "--catalog",
        str(catalog_path),
        "--audit-report",
        str(audit_path),
        "--dataset-root",
        str(root),
        "--dataset-stats",
        str(dataset_stats),
        "--dataset-stats-manifest",
        str(dataset_stats_manifest),
        "--dino-checkpoint",
        str(dino),
        "--dino-revision",
        "0123456789abcdef0123456789abcdef01234567",
        "--output",
        str(output),
        "--plan-only",
    ]
    return {
        "root": root,
        "catalog": catalog_path,
        "audit": audit_path,
        "data_config": data_config,
        "dataset_stats": dataset_stats,
        "dataset_stats_manifest": dataset_stats_manifest,
        "dino": dino,
        "output": output,
        "args": args,
    }


def _replace_option(args: list[str], option: str, value: Path) -> list[str]:
    changed = list(args)
    changed[changed.index(option) + 1] = str(value)
    return changed


def _assert_cli_rejected(args: list[str], *, output: Path) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(args)
    assert exc_info.value.code == 2
    assert not output.exists()


def _rewrite_hashed_audit(
    source: Path,
    target: Path,
    mutation: Any,
) -> None:
    document = json.loads(source.read_text(encoding="utf-8"))
    document.pop("report_sha256")
    mutation(document)
    document["report_sha256"] = _canonical_hash(document)
    write_audit_report(document, target)


def test_module_import_blocks_all_server_only_dependencies() -> None:
    script = """
import importlib.abc
import sys

HEAVY = {'torch', 'pyarrow', 'transformers', 'hydra'}

class BlockHeavyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in HEAVY:
            raise ImportError(f'blocked dependency: {fullname}')
        return None

sys.meta_path.insert(0, BlockHeavyImports())
import scripts.precompute_warm_features as module
for name in HEAVY:
    assert name not in module.__dict__, name
    assert name not in sys.modules, name
"""
    pythonpath = os.pathsep.join(
        (str(PROJECT_ROOT), str(PROJECT_ROOT / "src"))
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": pythonpath},
    )

    assert completed.returncode == 0, completed.stderr


def test_plan_only_validates_full_local_contract_without_writing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _fixture(tmp_path)
    args = fixture["args"]
    assert isinstance(args, list)

    assert main(args) == 0

    captured = capsys.readouterr()
    plan = json.loads(captured.out)
    assert captured.err == ""
    assert plan["schema"] == "warm.feature-precompute-plan"
    assert plan["version"] == 1
    assert plan["splits"] == ["train", "dev"]
    assert plan["split_episode_counts"] == {"dev": 1, "train": 1}
    assert plan["episode_count"] == 2
    assert plan["dino"] == {
        "model_id": "facebook/dinov2-base",
        "revision": "0123456789abcdef0123456789abcdef01234567",
        "checkpoint_sha256": plan["dino"]["checkpoint_sha256"],
        "checkpoint_file_count": 2,
    }
    assert len(plan["dino"]["checkpoint_sha256"]) == 64
    assert plan["vae"] == {"enabled": False, "checkpoint_sha256": None}
    assert plan["incomplete_smoke"] is False
    output = fixture["output"]
    assert isinstance(output, Path)
    assert plan["output"] == str(output.resolve())
    assert not output.exists()
    assert not list(tmp_path.glob(".*.warm-feature-precompute.lock"))
    assert not list(tmp_path.glob(".*.staging"))


def test_plan_only_rejects_test_split(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    args = fixture["args"]
    output = fixture["output"]
    assert isinstance(args, list) and isinstance(output, Path)

    _assert_cli_rejected([*args, "--split", "test"], output=output)


def test_plan_only_rejects_feature_output_inside_git_worktree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    args = fixture["args"]
    output = fixture["output"]
    assert isinstance(args, list) and isinstance(output, Path)
    monkeypatch.setattr(precompute_cli, "_repository_root", lambda: tmp_path.resolve())

    _assert_cli_rejected(args, output=output)


def test_task_vocabulary_is_identical_for_separate_train_and_dev_runs(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    catalog_path = fixture["catalog"]
    assert isinstance(catalog_path, Path)

    catalog = EpisodeCatalog.load(catalog_path)
    renamed = replace(
        catalog,
        episodes=tuple(
            replace(
                record,
                tasks=("train-only task",) if record.split == "train" else ("dev-only task",),
            )
            for record in catalog.episodes
        ),
    )
    expected = ("dev-only task", "train-only task")
    assert precompute_cli._task_vocabulary(renamed) == expected


def test_plan_only_rejects_audit_bound_to_a_different_catalog(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    audit = fixture["audit"]
    output = fixture["output"]
    args = fixture["args"]
    assert isinstance(audit, Path) and isinstance(output, Path)
    assert isinstance(args, list)
    mismatched = tmp_path / "audit-mismatched-catalog.json"
    _rewrite_hashed_audit(
        audit,
        mismatched,
        lambda document: document.__setitem__("catalog_sha256", "f" * 64),
    )

    _assert_cli_rejected(
        _replace_option(args, "--audit-report", mismatched),
        output=output,
    )


def test_plan_only_rejects_tampered_audit_content_hash(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    audit = fixture["audit"]
    output = fixture["output"]
    args = fixture["args"]
    assert isinstance(audit, Path) and isinstance(output, Path)
    assert isinstance(args, list)
    document = json.loads(audit.read_text(encoding="utf-8"))
    document["summary"]["task_count"] += 1
    tampered = tmp_path / "audit-tampered.json"
    tampered.write_text(
        json.dumps(document, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )

    _assert_cli_rejected(
        _replace_option(args, "--audit-report", tampered),
        output=output,
    )


def test_plan_only_rejects_incomplete_audit_episode_coverage(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    audit = fixture["audit"]
    output = fixture["output"]
    args = fixture["args"]
    assert isinstance(audit, Path) and isinstance(output, Path)
    assert isinstance(args, list)
    incomplete = tmp_path / "audit-incomplete.json"

    def drop_one_proof(document: dict[str, Any]) -> None:
        document["episode_source_hashes"].pop()
        document["summary"]["episode_count"] -= 1

    _rewrite_hashed_audit(audit, incomplete, drop_one_proof)

    _assert_cli_rejected(
        _replace_option(args, "--audit-report", incomplete),
        output=output,
    )


@pytest.mark.parametrize("metadata_name", ["info.json", "episodes.jsonl"])
def test_plan_only_rejects_root_metadata_changed_after_cataloging(
    tmp_path: Path,
    metadata_name: str,
) -> None:
    fixture = _fixture(tmp_path)
    root = fixture["root"]
    output = fixture["output"]
    args = fixture["args"]
    assert isinstance(root, Path) and isinstance(output, Path)
    assert isinstance(args, list)
    metadata = root / "meta" / metadata_name
    metadata.write_bytes(metadata.read_bytes() + b"\n")

    _assert_cli_rejected(args, output=output)


def test_plan_only_rejects_include_vae_without_local_checkpoint(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    args = fixture["args"]
    output = fixture["output"]
    assert isinstance(args, list) and isinstance(output, Path)

    _assert_cli_rejected([*args, "--include-vae"], output=output)


def test_plan_only_rejects_truncation_without_explicit_smoke_opt_in(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    args = fixture["args"]
    output = fixture["output"]
    assert isinstance(args, list) and isinstance(output, Path)

    _assert_cli_rejected(
        [*args, "--max-episodes-per-split", "1"],
        output=output,
    )


def _immutable_array(value: Any, dtype: np.dtype[Any]) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    frozen = np.frombuffer(contiguous.tobytes(order="C"), dtype=contiguous.dtype)
    return frozen.reshape(contiguous.shape)


@pytest.mark.parametrize("include_vae", [False, True], ids=("dino", "dino+vae"))
def test_server_precompute_cpu_integration_publishes_atomic_feature_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    include_vae: bool,
) -> None:
    """Exercise the real non-plan orchestration without a GPU or model load.

    Only environmental boundaries are replaced: the Hydra processor factory,
    audited episode reader, DINO model, optional VAE model, dtype selection,
    and Git provenance.  The real FastWAM preprocessing adapter, image adapter,
    episode assembler, feature-cache writer/reader, contracts, artifact claim,
    and final staging rename all remain active.
    """

    from fastwam.datasets.lerobot.full_episode_reader import FullLerobotEpisode
    import fastwam.datasets.lerobot.full_episode_reader as reader_module
    from fastwam.memory.feature_cache import (
        feature_cache_manifest_path,
        load_episode_feature_cache,
        save_episode_feature_cache,
    )
    from fastwam.memory.feature_precompute import (
        DinoFactualFeatures,
        FastWAMProcessorAdapter as RealProcessorAdapter,
    )
    import fastwam.memory.feature_precompute as feature_precompute_module
    import fastwam.memory.feature_precompute_vae as vae_feature_module
    from fastwam.memory.manifest import sha256_file
    import fastwam.memory.server_feature_encoders as encoder_module

    fixture = _fixture(tmp_path)
    raw_args = fixture["args"]
    output = fixture["output"]
    root = fixture["root"]
    dataset_stats = fixture["dataset_stats"]
    dataset_stats_manifest = fixture["dataset_stats_manifest"]
    dino_checkpoint = fixture["dino"]
    assert isinstance(raw_args, list)
    assert isinstance(output, Path)
    assert isinstance(root, Path)
    assert isinstance(dataset_stats, Path)
    assert isinstance(dataset_stats_manifest, Path)
    assert isinstance(dino_checkpoint, Path)

    arguments = [item for item in raw_args if item != "--plan-only"]
    arguments.extend(("--split", "train"))
    vae_checkpoint = tmp_path / "Wan2.2_VAE.pth"
    if include_vae:
        vae_checkpoint.write_bytes(b"local-mocked-vae-checkpoint")
        arguments.extend(
            ("--include-vae", "--vae-checkpoint", str(vae_checkpoint))
        )
    args = precompute_cli.build_parser().parse_args(arguments)
    plan = precompute_cli._build_plan(args)
    assert len(plan.records) == 1
    record = plan.records[0]
    assert record.split == "train"
    assert record.length == 4

    calls: dict[str, Any] = {
        "processor": [],
        "image_transforms": [],
        "reader": [],
        "dino_load": [],
        "dino_encode": [],
        "vae_load": [],
        "vae_encode": [],
        "rename": [],
    }

    class _Stage:
        def __init__(self, name: str, transform: Any) -> None:
            self._name = name
            self._transform = transform

        def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
            calls["processor"].append(self._name)
            return self._transform(batch)

    class ConcatLeftAlign(_Stage):
        __module__ = "fastwam.datasets.lerobot.transforms.action_state_merger"

        def __init__(self, name: str, transform: Any) -> None:
            super().__init__(name, transform)
            self.action_target_dim = None
            self.state_target_dim = None

    def _normalizer(batch: dict[str, Any]) -> dict[str, Any]:
        batch["action"]["default"] *= 2.0
        batch["state"]["default"] *= 3.0
        return batch

    def _merger(batch: dict[str, Any]) -> dict[str, Any]:
        batch["action"] = batch["action"]["default"]
        batch["state"] = batch["state"]["default"]
        return batch

    def _validation_transform(camera: str):
        def transform(value: np.ndarray) -> np.ndarray:
            assert value.dtype == np.uint8
            assert value.shape == (4, 3, 224, 224)
            calls["image_transforms"].append((camera, float(value.mean())))
            return value.astype(np.float32) / np.float32(255.0)

        return transform

    class _CPUProcessor:
        action_output_dim = 7
        proprio_output_dim = 8
        shape_meta = {
            "action": [{"key": "default", "raw_shape": 7, "shape": 7}],
            "state": [{"key": "default", "raw_shape": 8, "shape": 8}],
        }
        action_state_transforms = None
        use_stepwise_action_norm = False
        norm_default_mode = "min/max"
        norm_exception_mode = None
        delta_action_dim_mask = {
            "default": np.asarray(
                [True, True, True, True, True, True, False], dtype=np.bool_
            )
        }

        def __init__(self) -> None:
            self.normalizer = _Stage("normalizer.forward", _normalizer)
            self.action_state_merger = ConcatLeftAlign("merger.forward", _merger)
            self.val_transforms = {
                "image": (_validation_transform("image"),),
                "wrist_image": (_validation_transform("wrist_image"),),
            }

        def action_state_transform(
            self, batch: dict[str, Any]
        ) -> dict[str, Any]:
            calls["processor"].append("action_state_transform")
            # The terminal raw action was retained by the reader and dropped
            # exactly once by the real FastWAMProcessorAdapter.
            assert batch["action"]["default"].shape == (3, 7)
            assert batch["state"]["default"].shape == (4, 8)
            batch["action"]["default"] += 1.0
            batch["state"]["default"] += 2.0
            return batch

    processor = _CPUProcessor()
    monkeypatch.setattr(
        precompute_cli,
        "_load_processor",
        lambda data_config, stats_path: processor,
    )
    monkeypatch.setattr(
        precompute_cli,
        "_software_provenance",
        lambda: {"git_commit": "c" * 40, "git_dirty": False},
    )
    monkeypatch.setattr(
        precompute_cli,
        "_runtime_provenance",
        lambda device: {
            "python": "3.test",
            "platform": "test-platform",
            "numpy": np.__version__,
            "torch": "test-torch",
            "transformers": "test-transformers",
            "cuda_runtime": "test-cuda",
            "cudnn": 9999,
            "cuda_device_name": "test-gpu",
            "cuda_capability": [9, 9],
        },
    )
    monkeypatch.setattr(precompute_cli, "_torch_dtype", lambda name: f"mock:{name}")

    # Preserve the real adapter's exact preprocessing sequence while forcing
    # its array boundary to NumPy so no torch import or CUDA operation occurs.
    monkeypatch.setattr(
        feature_precompute_module,
        "FastWAMProcessorAdapter",
        lambda processor, tensor_backend: RealProcessorAdapter(
            processor, tensor_backend="numpy"
        ),
    )

    raw_actions = np.arange(4 * 7, dtype=np.float32).reshape(4, 7) / 10.0
    raw_states = np.arange(4 * 8, dtype=np.float32).reshape(4, 8) / 20.0
    raw_states[:, -2:] = np.asarray(
        [[-0.1, 0.2], [-0.2, 0.3], [0.4, -0.1], [-0.5, -0.5]],
        dtype=np.float32,
    )
    front_value = np.float32(0.125)
    wrist_value = np.float32(0.75)
    front = np.full((4, 3, 224, 224), front_value, dtype=np.float32)
    wrist = np.full((4, 3, 224, 224), wrist_value, dtype=np.float32)

    def _mock_full_reader(
        requested_record: Any,
        *,
        dataset_root: Path,
        audit_proof: Any,
        camera_keys: tuple[str, ...],
        timestamp_tolerance_s: float,
        video_backend: str,
    ) -> FullLerobotEpisode:
        assert requested_record == record
        assert Path(dataset_root).resolve() == root.resolve()
        assert audit_proof.episode_key == (
            record.dataset_id,
            record.dataset_index,
            record.episode_index,
        )
        assert camera_keys == (
            "observation.images.image",
            "observation.images.wrist_image",
        )
        assert timestamp_tolerance_s == pytest.approx(1e-4)
        assert video_backend == "pyav"
        calls["reader"].append(camera_keys)
        return FullLerobotEpisode(
            record=record,
            source_episode_sha256=audit_proof.source_episode_sha256,
            data_path=(root / record.data_relpath).resolve(),
            actions=_immutable_array(raw_actions, np.dtype(np.float32)),
            states=_immutable_array(raw_states, np.dtype(np.float32)),
            images=MappingProxyType(
                {
                    camera_keys[0]: _immutable_array(front, np.dtype(np.float32)),
                    camera_keys[1]: _immutable_array(wrist, np.dtype(np.float32)),
                }
            ),
            timestamps=_immutable_array(
                np.arange(4, dtype=np.float64) / 20.0, np.dtype(np.float64)
            ),
            task_indices=_immutable_array(
                np.zeros(4, dtype=np.int64), np.dtype(np.int64)
            ),
        )

    monkeypatch.setattr(
        reader_module, "read_full_lerobot_episode", _mock_full_reader
    )

    RealImageAdapter = encoder_module.FastWAMImageAdapter

    class _CPUImageAdapter(RealImageAdapter):
        def __init__(self, processor: Any, camera_keys: Any, concat_mode: str) -> None:
            super().__init__(
                processor,
                camera_keys,
                concat_mode,
                tensor_factory=lambda value: value,
            )

    class _MockDino:
        hidden_size = 3
        patch_size = (16, 16)
        patch_grid_size = (14, 14)
        register_token_count = 0
        image_size = (224, 224)
        image_mean = (0.485, 0.456, 0.406)
        image_std = (0.229, 0.224, 0.225)

        @classmethod
        def from_pretrained(
            cls,
            local_checkpoint: Path,
            *,
            model_id: str,
            revision: str,
            device: str,
            torch_dtype: Any,
        ) -> "_MockDino":
            calls["dino_load"].append(
                (Path(local_checkpoint), model_id, revision, device, torch_dtype)
            )
            return cls()

        def encode(
            self, frames: np.ndarray, *, batch_size: int
        ) -> DinoFactualFeatures:
            assert frames.dtype == np.float32
            assert not frames.flags.writeable
            assert frames.shape == (4, 3, 224, 224)
            calls["dino_encode"].append((np.array(frames, copy=True), batch_size))
            frame_mean = frames.mean(axis=(1, 2, 3), dtype=np.float64).astype(
                np.float32
            )
            cls = np.stack(
                (frame_mean, np.ones(4, dtype=np.float32), frame_mean + 0.25),
                axis=1,
            )
            spatial = np.stack(
                tuple(cls + np.float32(index / 10.0) for index in range(4)),
                axis=1,
            )
            return DinoFactualFeatures(cls=cls, spatial=spatial)

    monkeypatch.setattr(encoder_module, "FastWAMImageAdapter", _CPUImageAdapter)
    monkeypatch.setattr(encoder_module, "DinoV2FactualEncoder", _MockDino)

    vae_sentinel = object()
    if include_vae:
        loader_module = ModuleType("fastwam.models.wan22.helpers.loader")

        def _mock_load_vae_only(**kwargs: Any) -> SimpleNamespace:
            calls["vae_load"].append(dict(kwargs))
            return SimpleNamespace(vae=vae_sentinel, vae_path=str(vae_checkpoint))

        loader_module.load_wan22_vae_only = _mock_load_vae_only  # type: ignore[attr-defined]
        loader_module.Wan22LoadedVAE = SimpleNamespace  # type: ignore[attr-defined]
        helpers_module = ModuleType("fastwam.models.wan22.helpers")
        helpers_module.__path__ = []  # type: ignore[attr-defined]
        helpers_module.loader = loader_module  # type: ignore[attr-defined]
        wan22_module = ModuleType("fastwam.models.wan22")
        wan22_module.__path__ = []  # type: ignore[attr-defined]
        wan22_module.helpers = helpers_module  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "fastwam.models.wan22", wan22_module)
        monkeypatch.setitem(
            sys.modules, "fastwam.models.wan22.helpers", helpers_module
        )
        monkeypatch.setitem(
            sys.modules,
            "fastwam.models.wan22.helpers.loader",
            loader_module,
        )

        def _mock_encode_vae(
            vae: Any,
            frames: np.ndarray,
            *,
            device: str,
            batch_size: int,
            torch_dtype: Any,
        ) -> np.ndarray:
            assert vae is vae_sentinel
            assert frames.dtype == np.float32
            assert not frames.flags.writeable
            assert frames.shape == (4, 3, 224, 448)
            calls["vae_encode"].append(
                (np.array(frames, copy=True), device, batch_size, torch_dtype)
            )
            return np.arange(4 * 2 * 4 * 8, dtype=np.float32).reshape(
                4, 2, 4, 8
            )

        monkeypatch.setattr(
            vae_feature_module,
            "encode_wan22_factual_frames",
            _mock_encode_vae,
        )

    real_rename = os.rename

    def _recording_rename(source: Path, destination: Path) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        assert source_path.is_dir()
        assert not destination_path.exists()
        calls["rename"].append((source_path, destination_path))
        real_rename(source_path, destination_path)

    monkeypatch.setattr(precompute_cli.os, "rename", _recording_rename)

    precompute_cli._run_server_precompute(args, plan)

    assert output.is_dir()
    assert calls["processor"] == [
        "action_state_transform",
        "normalizer.forward",
        "merger.forward",
    ]
    assert calls["reader"] == [
        ("observation.images.image", "observation.images.wrist_image")
    ]
    assert calls["image_transforms"] == [
        ("image", 31.0),
        ("wrist_image", 191.0),
    ]
    assert len(calls["dino_load"]) == 1
    assert calls["dino_load"][0] == (
        dino_checkpoint.resolve(),
        "facebook/dinov2-base",
        "0123456789abcdef0123456789abcdef01234567",
        "cuda",
        "mock:bfloat16",
    )
    assert len(calls["dino_encode"]) == 1
    dino_input, dino_batch_size = calls["dino_encode"][0]
    assert dino_batch_size == 64
    np.testing.assert_allclose(dino_input, np.float32(31.0 / 255.0))

    assert len(calls["rename"]) == 1
    staging_path, published_path = calls["rename"][0]
    assert staging_path.parent == output.parent
    assert staging_path.name.startswith(f".{output.name}.")
    assert staging_path.name.endswith(".staging")
    assert published_path == output
    assert not staging_path.exists()
    assert not list(output.parent.glob(f".{output.name}.*.staging"))
    assert not (output.parent / f".{output.name}.warm-feature-precompute.lock").exists()

    expected_relative = (
        f"dataset_{record.dataset_index:03d}/"
        f"episode_{record.episode_index:06d}.npz"
    )
    assert (output / "train_features.list").read_text(encoding="utf-8") == (
        expected_relative + "\n"
    )
    assert not (output / "dev_features.list").exists()
    payload = output / expected_relative
    manifest_path = feature_cache_manifest_path(payload)
    assert payload.is_file()
    assert manifest_path.is_file()

    loaded = load_episode_feature_cache(payload)
    features = loaded.features
    assert features.model_actions.shape == (3, 7)
    assert features.proprio.shape == (4, 8)
    assert features.context_keys.shape == (4, 4)
    assert features.semantic_features.shape == (4, 4, 3)
    np.testing.assert_allclose(features.model_actions, (raw_actions[:-1] + 1.0) * 2.0)
    np.testing.assert_allclose(features.proprio, (raw_states + 2.0) * 3.0)
    np.testing.assert_allclose(features.gripper, [0.3, 0.5, 0.5, 1.0])
    np.testing.assert_allclose(
        np.linalg.norm(features.context_keys, axis=1), 1.0, atol=1e-6
    )
    for value in (
        features.model_actions,
        features.proprio,
        features.gripper,
        features.context_keys,
        features.semantic_features,
    ):
        assert value.dtype == np.float32
        assert not value.flags.writeable

    if include_vae:
        assert len(calls["vae_load"]) == 1
        assert calls["vae_load"][0] == {
            "device": "cuda",
            "torch_dtype": "mock:bfloat16",
            "vae_path": vae_checkpoint.resolve(),
        }
        assert len(calls["vae_encode"]) == 1
        vae_input, vae_device, vae_batch_size, vae_dtype = calls["vae_encode"][0]
        assert (vae_device, vae_batch_size, vae_dtype) == (
            "cuda",
            4,
            "mock:bfloat16",
        )
        assert vae_input.shape == (4, 3, 224, 448)
        np.testing.assert_allclose(
            vae_input[:, :, :, :224], np.float32(2.0 * 31.0 / 255.0 - 1.0)
        )
        np.testing.assert_allclose(
            vae_input[:, :, :, 224:], np.float32(2.0 * 191.0 / 255.0 - 1.0)
        )
        assert features.vae_features is not None
        assert features.vae_features.shape == (4, 2, 4, 8)
        assert not features.vae_features.flags.writeable
    else:
        assert calls["vae_load"] == []
        assert calls["vae_encode"] == []
        assert features.vae_features is None

    contracts = output / "contracts"
    normalizer_contract = json.loads(
        (contracts / "normalizer_contract.json").read_text(encoding="utf-8")
    )
    encoder_contract = json.loads(
        (contracts / "encoder_contract.json").read_text(encoding="utf-8")
    )
    camera_contract = json.loads(
        (contracts / "camera_contract.json").read_text(encoding="utf-8")
    )
    assert (contracts / "dataset_stats.source.json").read_bytes() == (
        dataset_stats.read_bytes()
    )
    assert (contracts / "train_stats_manifest.source.json").read_bytes() == (
        dataset_stats_manifest.read_bytes()
    )
    assert normalizer_contract["action_dim"] == 7
    assert normalizer_contract["arm_dims"] == [0, 1, 2, 3, 4, 5]
    assert normalizer_contract["gripper_dims"] == [6]
    assert normalizer_contract["normalization_stats_sha256"] == sha256_file(
        dataset_stats
    )
    assert encoder_contract["dino"]["hidden_size"] == 3
    assert encoder_contract["normalization"] == {
        "stats_sha256": sha256_file(dataset_stats),
        "train_stats_manifest_file_sha256": sha256_file(dataset_stats_manifest),
        "train_stats_manifest_content_sha256": json.loads(
            dataset_stats_manifest.read_text(encoding="utf-8")
        )["content_sha256"],
    }
    assert encoder_contract["context"] == {
        "mode": "task-conditioned",
        "task_vocabulary": ["pick object"],
        "visual_task_energy": [0.5, 0.5],
    }
    assert encoder_contract["vae"]["enabled"] is include_vae
    assert encoder_contract["vae"]["checkpoint_sha256"] == (
        sha256_file(vae_checkpoint) if include_vae else None
    )
    assert camera_contract["source_camera_keys"] == [
        "observation.images.image",
        "observation.images.wrist_image",
    ]
    assert camera_contract["processor_camera_mapping"] == {
        "observation.images.image": "image",
        "observation.images.wrist_image": "wrist_image",
    }
    assert camera_contract["semantic_camera"] == "observation.images.image"
    assert camera_contract["concat_mode"] == "horizontal"

    summary = json.loads(
        (output / "precompute_summary.json").read_text(encoding="utf-8")
    )
    assert summary["schema"] == "warm.feature-precompute-summary"
    assert summary["version"] == 1
    assert summary["split_counts"] == {"train": 1}
    assert summary["official_complete"] is True
    assert summary["software"] == {"git_commit": "c" * 40, "git_dirty": False}
    assert summary["feature_shapes_excluding_time"] == {
        "model_actions": [7],
        "proprio": [8],
        "context_keys": [4],
        "semantic_features": [4, 3],
        "vae_features": [2, 4, 8] if include_vae else [],
    }
    assert summary["contracts"]["normalizer_hash"] == sha256_file(
        contracts / "normalizer_contract.json"
    )
    assert summary["contracts"]["encoder_hash"] == sha256_file(
        contracts / "encoder_contract.json"
    )
    assert summary["contracts"]["camera_hash"] == sha256_file(
        contracts / "camera_contract.json"
    )
    assert loaded.metadata.normalizer_hash == summary["contracts"]["normalizer_hash"]
    assert loaded.metadata.encoder_hash == summary["contracts"]["encoder_hash"]
    assert loaded.metadata.camera_hash == summary["contracts"]["camera_hash"]
    assert loaded.metadata.catalog_hash == plan.catalog.content_sha256
    assert loaded.metadata.source_episode_sha256 == (
        plan.audit.proof_index[
            (record.dataset_id, record.dataset_index, record.episode_index)
        ].source_episode_sha256
    )

    # Both the episode cache API and the top-level artifact publisher reject
    # replacement, which is the operational immutability contract.
    with pytest.raises(FileExistsError, match="already exists"):
        save_episode_feature_cache(
            payload,
            features,
            metadata=loaded.metadata,
        )
    with pytest.raises(FileExistsError, match="feature output already exists"):
        precompute_cli._run_server_precompute(args, plan)
    assert len(calls["rename"]) == 1
    assert not (output.parent / f".{output.name}.warm-feature-precompute.lock").exists()
