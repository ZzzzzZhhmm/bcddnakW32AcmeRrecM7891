"""Publish adapter episodes in the repository's LeRobot v2.1 format."""
from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from fastwam.datasets.lerobot.episode_catalog import EpisodeCatalog, scan_lerobot_datasets
from .contracts import PreparationError, file_sha256, inside, write_json


DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _stats(array: np.ndarray) -> dict[str, Any]:
    return {**{k: getattr(np, k)(array, axis=0).tolist() for k in ("min", "max", "mean", "std")},
            "count": [len(array)]}


def write_video(path: Path, images: np.ndarray, fps: int) -> None:
    import av
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=Fraction(fps), options={"crf": "18", "preset": "medium", "threads": "1"})
        stream.width, stream.height = images.shape[2], images.shape[1]
        stream.pix_fmt = "yuv420p"
        stream.codec_context.max_b_frames = 0
        stream.time_base = stream.codec_context.time_base = Fraction(1, fps)
        for index, image in enumerate(images):
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, fps)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)


def convert_episodes(adapter: Any, entries: list[dict[str, Any]], output: Path,
                     profile: Mapping[str, Any], dataset_id: str) -> EpisodeCatalog:
    import pyarrow as pa
    import pyarrow.parquet as pq
    output.mkdir(parents=True, exist_ok=False)
    (output / "meta").mkdir()
    ordered = sorted(entries, key=lambda x: ({"train": 0, "dev": 1, "test": 2}[x["split"]], x["id"]))
    episode_rows, stat_rows, source_rows = [], [], []
    tasks: dict[str, int] = {}
    fingerprints: dict[str, str] = {}
    camera_shapes: dict[str, list[int]] = {}
    global_index = 0
    split_counts = {split: 0 for split in ("train", "dev", "test")}
    for index, entry in enumerate(ordered):
        source_path = inside(adapter.root, entry["path"])
        source_files = sorted(p for p in source_path.rglob("*") if p.is_file()) if source_path.is_dir() else [source_path]
        if "instructions" in entry:
            source_files.append(inside(adapter.root, entry["instructions"]))
        before = {str(p.resolve()): file_sha256(p) for p in source_files}
        episode = adapter.read(entry)
        episode.validate(profile)
        after = {str(p.resolve()): file_sha256(p) for p in episode.source_files}
        if any(before.get(path) != digest for path, digest in after.items()):
            raise PreparationError("Raw source changed while reading, or referenced an unlisted source")
        fingerprint = episode.content_sha256()
        if fingerprint in fingerprints:
            raise PreparationError(f"Duplicate raw episode content: {fingerprints[fingerprint]} and {episode.source_id}")
        fingerprints[fingerprint] = episode.source_id
        task_index = tasks.setdefault(episode.instruction, len(tasks))
        n = len(episode.states)
        arrays = {"action": episode.actions, "observation.state": episode.states,
                  "timestamp": (np.arange(n, dtype=np.float64) / profile["fps"]).astype(np.float32),
                  "frame_index": np.arange(n, dtype=np.int64),
                  "episode_index": np.full(n, index, dtype=np.int64),
                  "index": np.arange(global_index, global_index + n, dtype=np.int64),
                  "task_index": np.full(n, task_index, dtype=np.int64)}
        table = pa.table({name: pa.array(value.tolist(), type=pa.list_(pa.float32(), value.shape[1]))
                          if value.ndim == 2 else pa.array(value) for name, value in arrays.items()})
        parquet = output / DATA_PATH.format(episode_chunk=index // 1000, episode_index=index)
        parquet.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, parquet)
        for key, images in episode.images.items():
            shape = [3, images.shape[1], images.shape[2]]
            if camera_shapes.setdefault(key, shape) != shape:
                raise PreparationError(f"Camera resolution changed within dataset: {key}")
            video = output / VIDEO_PATH.format(episode_chunk=index // 1000, episode_index=index,
                                               video_key="observation.images." + key)
            write_video(video, images, profile["fps"])
        episode_rows.append({"episode_index": index, "length": n, "tasks": [episode.instruction],
                             "warm_task_identity": episode.task})
        stat_rows.append({"episode_index": index, "stats": {k: _stats(arrays[k]) for k in ("action", "observation.state")}})
        source_rows.append({"episode_index": index, "source_id": episode.source_id, "split": episode.split,
                            "raw_content_sha256": fingerprint,
                            "sources": [{"path": Path(path).relative_to(adapter.root).as_posix(), "sha256": digest}
                                        for path, digest in sorted(after.items())],
                            "rows": n, "memory_transition_count": n - 1,
                            "annotations": episode.annotations})
        global_index += n
        split_counts[episode.split] += 1
        print(f"converted {index + 1}/{len(ordered)}: {episode.source_id} ({n} rows)", flush=True)
    features = {name: {"dtype": "float32", "shape": [profile[dim]], "names": [[f"{name}_{i}" for i in range(profile[dim])]]}
                for name, dim in (("action", "action_dim"), ("observation.state", "state_dim"))}
    features.update({name: {"dtype": "float32" if name == "timestamp" else "int64", "shape": [1], "names": None}
                     for name in ("timestamp", "frame_index", "episode_index", "index", "task_index")})
    features.update({"observation.images." + key: {"dtype": "video", "shape": shape,
                                                  "names": ["channels", "height", "width"]}
                     for key, shape in camera_shapes.items()})
    splits, offset = {}, 0
    for split, count in split_counts.items():
        if count:
            splits[split] = f"{offset}:{offset + count}"
        offset += count
    write_json(output / "meta/info.json", {"codebase_version": "v2.1", "robot_type": profile["embodiment"],
               "total_episodes": len(ordered), "total_frames": global_index, "total_tasks": len(tasks),
               "total_videos": len(ordered) * len(camera_shapes), "total_chunks": (len(ordered) + 999) // 1000,
               "chunks_size": 1000, "fps": profile["fps"], "splits": splits, "data_path": DATA_PATH,
               "video_path": VIDEO_PATH, "features": features})
    _jsonl(output / "meta/episodes.jsonl", episode_rows)
    _jsonl(output / "meta/tasks.jsonl", [{"task_index": i, "task": task} for task, i in tasks.items()])
    _jsonl(output / "meta/episodes_stats.jsonl", stat_rows)
    write_json(output / "meta/conversion.json", {"schema": "warm.adapter-conversion.v1", "profile": dict(profile),
               "episodes": source_rows, "alignment": "N command-bearing rows; memory uses N-1 factual transitions; no padding"})
    scanned = scan_lerobot_datasets([output], dataset_ids=[dataset_id])
    catalog = EpisodeCatalog(scanned.datasets, tuple(replace(record, split=ordered[record.episode_index]["split"])
                                                     for record in scanned.episodes))
    catalog.save(output / "meta/warm_episode_catalog.json")
    return catalog
