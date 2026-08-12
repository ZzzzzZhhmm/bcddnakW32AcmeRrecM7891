from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest

from fastwam.datasets.lerobot.episode_catalog import EpisodeCatalog
from fastwam.datasets.lerobot.episode_catalog import scan_lerobot_datasets
from fastwam.datasets.rmbench.constants import (
    MANIFEST_FILENAME,
    OFFICIAL_CAMERA_KEYS,
    OFFICIAL_RMBENCH_TASKS,
)
from fastwam.datasets.rmbench.converter import (
    RMBenchConversionConfig,
    _atomic_publish_directory,
    convert_rmbench_dataset,
)
from fastwam.datasets.rmbench.source import (
    RMBenchSourceError,
    canonical_json_bytes,
    decode_official_jpeg,
    file_sha256,
    read_episode_instruction,
)
from fastwam.datasets.rmbench.split import deterministic_task_split


def test_official_allow_list_is_exactly_the_nine_paper_tasks() -> None:
    assert OFFICIAL_RMBENCH_TASKS == (
        "observe_and_pickup",
        "rearrange_blocks",
        "put_back_block",
        "swap_blocks",
        "swap_T",
        "blocks_ranking_try",
        "press_button",
        "cover_blocks",
        "battery_try",
    )


def test_episode_split_is_deterministic_task_stratified_and_disjoint() -> None:
    episodes = {task: range(50) for task in OFFICIAL_RMBENCH_TASKS}
    first = deterministic_task_split(episodes, dev_per_task=5, seed=42)
    second = deterministic_task_split(episodes, dev_per_task=5, seed=42)
    changed = deterministic_task_split(episodes, dev_per_task=5, seed=43)

    assert first == second
    assert first != changed
    assert len(first) == 450
    for task in OFFICIAL_RMBENCH_TASKS:
        train = {
            index for index in range(50) if first[(task, index)] == "train"
        }
        dev = {index for index in range(50) if first[(task, index)] == "dev"}
        assert len(train) == 45
        assert len(dev) == 5
        assert train.isdisjoint(dev)
        assert train | dev == set(range(50))


def test_production_config_rejects_partial_task_or_episode_sets(tmp_path: Path) -> None:
    kwargs = {
        "source_root": tmp_path / "source",
        "output_root": tmp_path / "output",
        "source_revision": "dataset-commit",
        "data_revision": "warm-rmbench-v1",
        "rmbench_code_revision": "code-commit",
    }
    with pytest.raises(ValueError, match="nine-task"):
        RMBenchConversionConfig(**kwargs, tasks=(OFFICIAL_RMBENCH_TASKS[0],))
    with pytest.raises(ValueError, match="exactly 50"):
        RMBenchConversionConfig(**kwargs, episodes_per_task=49)


def test_instruction_reader_is_exact_deterministic_and_hashed(tmp_path: Path) -> None:
    path = tmp_path / "episode0.json"
    payload = b'{"seen":["  Pick   the block.  ","Alternative"],"unseen":["Novel"]}\n'
    path.write_bytes(payload)

    instruction, variants, digest = read_episode_instruction(path)

    assert instruction == "Pick the block."
    assert variants == {
        "seen": ("Pick the block.", "Alternative"),
        "unseen": ("Novel",),
    }
    assert digest == sha256(payload).hexdigest()

    path.write_text('{"seen":["ok"],"extra":[]}', encoding="utf-8")
    with pytest.raises(RMBenchSourceError, match="unsupported keys"):
        read_episode_instruction(path)


def test_catalog_prefers_nine_way_identity_without_replacing_language_tasks(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "fps": 15,
                "total_episodes": 18,
                "chunks_size": 1000,
                "data_path": (
                    "data/chunk-{episode_chunk:03d}/"
                    "episode_{episode_index:06d}.parquet"
                ),
            }
        ),
        encoding="utf-8",
    )
    rows = []
    language_tasks = []
    for task in OFFICIAL_RMBENCH_TASKS:
        for variant in range(2):
            episode_index = len(rows)
            instruction = f"Natural instruction {episode_index} for {task}."
            language_tasks.append(instruction)
            rows.append(
                {
                    "episode_index": episode_index,
                    "length": 3,
                    "tasks": [instruction],
                    "warm_task_identity": task,
                }
            )
            parquet = (
                root
                / "data"
                / "chunk-000"
                / f"episode_{episode_index:06d}.parquet"
            )
            parquet.parent.mkdir(parents=True, exist_ok=True)
            parquet.write_bytes(f"fixture-{episode_index}".encode("ascii"))
    (root / "meta" / "episodes.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    (root / "meta" / "tasks.jsonl").write_text(
        "".join(
            json.dumps({"task_index": index, "task": instruction}) + "\n"
            for index, instruction in enumerate(language_tasks)
        ),
        encoding="utf-8",
    )

    catalog = scan_lerobot_datasets([root], dataset_ids=["rmbench"])

    assert tuple(sorted({episode.primary_task for episode in catalog.episodes})) == tuple(
        sorted(OFFICIAL_RMBENCH_TASKS)
    )
    assert len({episode.primary_task for episode in catalog.episodes}) == 9
    stored_language = {
        row["task"]
        for row in (
            json.loads(line)
            for line in (root / "meta" / "tasks.jsonl").read_text().splitlines()
        )
    }
    assert stored_language == set(language_tasks)
    assert stored_language.isdisjoint(OFFICIAL_RMBENCH_TASKS)


def test_strict_jpeg_decoder_accepts_only_zero_padding() -> None:
    cv2 = pytest.importorskip("cv2")
    original = np.zeros((8, 12, 3), dtype=np.uint8)
    original[:, :, 0] = 30
    original[:, :, 1] = 120
    original[:, :, 2] = 220
    ok, encoded = cv2.imencode(".jpg", original)
    assert ok
    jpeg = encoded.tobytes()

    decoded, exact = decode_official_jpeg(
        jpeg + b"\0" * 31,
        label="fixture",
        expected_height=8,
        expected_width=12,
    )
    assert decoded.shape == original.shape
    assert decoded.dtype == np.uint8
    assert exact == jpeg

    with pytest.raises(RMBenchSourceError, match="non-zero"):
        decode_official_jpeg(
            jpeg + b"\0bad",
            label="fixture",
            expected_height=8,
            expected_width=12,
        )


def _write_source_episode(root: Path, task: str, episode_index: int) -> np.ndarray:
    h5py = pytest.importorskip("h5py")
    cv2 = pytest.importorskip("cv2")
    task_root = root / task / "demo_clean"
    data_dir = task_root / "data"
    instruction_dir = task_root / "instructions"
    data_dir.mkdir(parents=True, exist_ok=True)
    instruction_dir.mkdir(parents=True, exist_ok=True)

    observations = 4
    qpos = (
        np.arange(observations * 14, dtype=np.float32).reshape(observations, 14)
        + episode_index * 100
    )
    encoded_per_camera: dict[str, list[bytes]] = {}
    for camera_offset, camera in enumerate(OFFICIAL_CAMERA_KEYS):
        encoded: list[bytes] = []
        for frame in range(observations):
            image = np.full(
                (8, 12, 3),
                fill_value=20 + 30 * camera_offset + frame,
                dtype=np.uint8,
            )
            ok, jpeg = cv2.imencode(".jpg", image)
            assert ok
            encoded.append(jpeg.tobytes())
        encoded_per_camera[camera] = encoded

    hdf5_path = data_dir / f"episode{episode_index}.hdf5"
    with h5py.File(hdf5_path, "w") as episode:
        episode.create_dataset("/joint_action/vector", data=qpos)
        source_names = {
            "cam_high": "head_camera",
            "cam_left_wrist": "left_camera",
            "cam_right_wrist": "right_camera",
        }
        for camera in OFFICIAL_CAMERA_KEYS:
            encoded = encoded_per_camera[camera]
            max_len = max(len(item) for item in encoded) + 7
            episode.create_dataset(
                f"/observation/{source_names[camera]}/rgb",
                data=encoded,
                dtype=f"S{max_len}",
            )
    (instruction_dir / f"episode{episode_index}.json").write_text(
        json.dumps(
            {
                "seen": [f"Perform {task} fixture {episode_index}."],
                "unseen": [f"Unseen {task} fixture {episode_index}."],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return qpos


def _require_conversion_dependencies() -> None:
    pytest.importorskip("h5py")
    pytest.importorskip("pyarrow")
    av = pytest.importorskip("av")
    try:
        av.codec.Codec("libx264", "w")
    except Exception:
        pytest.skip("PyAV build has no libx264 encoder")


def test_synthetic_conversion_is_factual_hashed_atomic_and_split_safe(
    tmp_path: Path,
) -> None:
    _require_conversion_dependencies()
    import av
    import pyarrow.parquet as pq

    task = OFFICIAL_RMBENCH_TASKS[0]
    source = tmp_path / "source"
    expected_qpos = {
        index: _write_source_episode(source, task, index) for index in range(2)
    }
    output = tmp_path / "converted"
    config = RMBenchConversionConfig(
        source_root=source,
        output_root=output,
        source_revision="fixture-source-sha",
        data_revision="fixture-data-v1",
        rmbench_code_revision="fixture-code-sha",
        dataset_id="rmbench_fixture",
        fps=15,
        dev_per_task=1,
        split_seed=7,
        workers=2,
        tasks=(task,),
        episodes_per_task=2,
        expected_image_height=8,
        expected_image_width=12,
        strict_official_contract=False,
    )

    manifest = convert_rmbench_dataset(config)

    assert output.is_dir()
    assert manifest["manifest_sha256"] == sha256(
        canonical_json_bytes(
            {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        )
    ).hexdigest()
    assert manifest["protocol"]["split"]["train_episodes"] == 1
    assert manifest["protocol"]["split"]["dev_episodes"] == 1
    assert manifest["protocol"]["terminal_visual_excluded_from_training"]
    assert manifest["protocol"]["terminal_observation_count"] == 2
    assert {item["split"] for item in manifest["episodes"]} == {"train", "dev"}
    task_rows = [
        json.loads(line)
        for line in (output / "meta" / "tasks.jsonl").read_text().splitlines()
    ]
    instruction_by_index = {
        int(row["task_index"]): row["task"] for row in task_rows
    }
    instruction_variant_rows = [
        json.loads(line)
        for line in (
            output / "meta" / "warm_instruction_variants.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert [row["episode_index"] for row in instruction_variant_rows] == [0, 1]
    for row in instruction_variant_rows:
        assert row["primary"]
        assert row["seen"]
        assert row["primary"] in row["seen"]
        assert isinstance(row["unseen"], list)
        assert not (set(row["seen"]) & set(row["unseen"]))

    for episode in manifest["episodes"]:
        output_index = episode["output_episode_index"]
        source_index = episode["source_episode_index"]
        parquet = output / episode["output_parquet"]["relpath"]
        table = pq.read_table(parquet)
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        parquet_task_indices = set(table["task_index"].to_pylist())
        assert len(parquet_task_indices) == 1
        parquet_task_index = int(next(iter(parquet_task_indices)))
        assert instruction_by_index[parquet_task_index] == episode["instruction"]
        variants = instruction_variant_rows[output_index]
        assert variants["primary"] == episode["instruction"]
        assert variants["seen"] == episode["instruction_variants"]["seen"]
        assert variants["unseen"] == episode["instruction_variants"]["unseen"]
        np.testing.assert_array_equal(states, expected_qpos[source_index][:-1])
        np.testing.assert_array_equal(actions, expected_qpos[source_index][1:])
        assert table.num_rows == 3
        assert episode["source_observation_count"] == 4
        assert episode["factual_transition_count"] == 3
        assert episode["terminal_observation_count"] == 1
        assert episode["terminal_visual_excluded_from_training"]
        assert not episode["terminal_observation_published_as_training_row"]
        terminal_digest_payload = {
            "state_sha256": episode["terminal_state_sha256"],
            "jpeg_sha256": episode["terminal_jpeg_sha256"],
        }
        assert episode["terminal_observation_digest"] == sha256(
            canonical_json_bytes(terminal_digest_payload)
        ).hexdigest()

        for camera in OFFICIAL_CAMERA_KEYS:
            video = output / episode["output_videos"][camera]["relpath"]
            with av.open(str(video)) as container:
                frames = list(container.decode(video=0))
            assert len(frames) == 3
            assert all((frame.height, frame.width) == (8, 12) for frame in frames)

    catalog = EpisodeCatalog.load(output / "meta" / "warm_episode_catalog.json")
    assert {episode.split for episode in catalog.episodes} == {"train", "dev"}
    assert len({episode.global_episode_id for episode in catalog.episodes}) == 2
    assert {episode.primary_task for episode in catalog.episodes} == {task}
    assert {row["task"] for row in task_rows} == {
        f"Perform {task} fixture 0.",
        f"Perform {task} fixture 1.",
    }
    assert all(row["task"] != task for row in task_rows)

    for artifact in manifest["output"]["artifacts"]:
        path = output / artifact["relpath"]
        assert path.stat().st_size == artifact["size_bytes"]
        assert file_sha256(path) == artifact["sha256"]
    written = json.loads(
        (output / "meta" / MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert written == manifest

    second_output = tmp_path / "converted-again"
    second_manifest = convert_rmbench_dataset(
        replace(config, output_root=second_output)
    )
    assert second_manifest == manifest
    for artifact in manifest["output"]["artifacts"]:
        relpath = artifact["relpath"]
        assert (output / relpath).read_bytes() == (second_output / relpath).read_bytes()

    with pytest.raises(FileExistsError, match="already exists"):
        convert_rmbench_dataset(config)


def test_atomic_publish_falls_back_to_rename_on_einval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ctypes
    import errno
    import sys

    if not sys.platform.startswith("linux"):
        pytest.skip("renameat2 fallback is Linux-specific")

    staging = tmp_path / "staging"
    output = tmp_path / "output"
    staging.mkdir()
    (staging / "marker").write_text("ok", encoding="utf-8")

    class FakeLibC:
        @staticmethod
        def renameat2(*_args, **_kwargs) -> int:
            ctypes.set_errno(errno.EINVAL)
            return -1

    monkeypatch.setattr(
        "ctypes.CDLL",
        lambda *_args, **_kwargs: FakeLibC(),
    )

    _atomic_publish_directory(staging, output)
    assert not staging.exists()
    assert output.is_dir()
    assert (output / "marker").read_text(encoding="utf-8") == "ok"


def test_failed_conversion_does_not_publish_partial_destination(tmp_path: Path) -> None:
    _require_conversion_dependencies()
    h5py = pytest.importorskip("h5py")

    task = OFFICIAL_RMBENCH_TASKS[0]
    source = tmp_path / "source"
    _write_source_episode(source, task, 0)
    _write_source_episode(source, task, 1)
    corrupt = source / task / "demo_clean" / "data" / "episode1.hdf5"
    with h5py.File(corrupt, "r+") as episode:
        cells = episode["/observation/head_camera/rgb"]
        cells[0] = b"not-a-jpeg"

    output = tmp_path / "must-not-appear"
    config = RMBenchConversionConfig(
        source_root=source,
        output_root=output,
        source_revision="fixture-source-sha",
        data_revision="fixture-data-v1",
        rmbench_code_revision="fixture-code-sha",
        tasks=(task,),
        episodes_per_task=2,
        dev_per_task=1,
        workers=2,
        expected_image_height=8,
        expected_image_width=12,
        strict_official_contract=False,
    )
    with pytest.raises(RMBenchSourceError, match="JPEG"):
        convert_rmbench_dataset(config)
    assert not output.exists()
    assert not list(tmp_path.glob(f".{output.name}.tmp-*"))
