#!/usr/bin/env python3
"""Copy measured-20Hz Piper episodes into one training-ready raw tree.

Nominal ``nominal_action_hz`` is ignored: v2 mixed 16.67Hz streams that still
declared 20.  Selection uses mean command dt.  Documented timestamp repairs
are applied on the copy only: JSON metadata is never hardlinked, so the
collection dumps stay untouched.

Repairs:
- terminal observation arrived ~25ms early while the command clock is 20Hz
- a camera/robot stamp slightly ahead of the observation stamp (sub-ms)
- a camera stamp that did not advance (duplicate last frame, +1 ns)

The last episode of each session is rewritten as a holdout DEV session so one
physical collection cannot cross splits.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path


TARGET_HZ = 20.0
HZ_TOLERANCE = 1.0
TICK_NS = int(round(1e9 / TARGET_HZ))
SHORT_DT_NS = (20_000_000, 30_000_000)
JSON_SUFFIXES = {".json", ".jsonl"}


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".packtmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def _write_json(path: Path, value) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    _write_bytes(path, payload.encode("utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    payload = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows)
    _write_bytes(path, payload.encode("utf-8"))


def _command_hz(episode_dir: Path) -> float | None:
    stamps = [int(row["timestamp_ns"]) for row in _read_jsonl(episode_dir / "commands.jsonl")]
    if len(stamps) < 2:
        return None
    mean = (stamps[-1] - stamps[0]) / (len(stamps) - 1) / 1e9
    if mean <= 0:
        return None
    return 1.0 / mean


def _copytree(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise FileExistsError(dest)

    def _copy(src_file: str, dest_file: str) -> str:
        if Path(src_file).suffix.lower() in JSON_SUFFIXES:
            return shutil.copy2(src_file, dest_file)
        try:
            os.link(src_file, dest_file)
            return dest_file
        except OSError:
            return shutil.copy2(src_file, dest_file)

    shutil.copytree(src, dest, copy_function=_copy)


def _add_ns(record: dict, delta_ns: int) -> dict:
    updated = json.loads(json.dumps(record))
    updated["timestamp_ns"] = int(updated["timestamp_ns"]) + delta_ns
    robot = updated.get("robot")
    if isinstance(robot, dict) and "timestamp_ns" in robot:
        robot["timestamp_ns"] = int(robot["timestamp_ns"]) + delta_ns
    cameras = updated.get("cameras")
    if isinstance(cameras, dict):
        for frame in cameras.values():
            if isinstance(frame, dict) and "timestamp_ns" in frame:
                frame["timestamp_ns"] = int(frame["timestamp_ns"]) + delta_ns
    return updated


def _camera_order(episode_dir: Path, observations: list[dict]) -> list[str]:
    meta_path = episode_dir / "episode.json"
    if meta_path.is_file():
        order = _read_json(meta_path).get("camera_order")
        if isinstance(order, list) and order:
            return [str(item) for item in order]
    cameras = observations[0].get("cameras") if observations else None
    if isinstance(cameras, dict):
        return list(cameras)
    return []


def _fix_terminal_observation(observations: list[dict]) -> dict | None:
    if len(observations) < 2:
        return None
    last_dt = int(observations[-1]["timestamp_ns"]) - int(observations[-2]["timestamp_ns"])
    if not SHORT_DT_NS[0] <= last_dt <= SHORT_DT_NS[1]:
        return None
    delta_ns = TICK_NS - last_dt
    observations[-1] = _add_ns(observations[-1], delta_ns)
    return {
        "repair": "shift_terminal_observation_to_20hz_tick",
        "observation_seq": observations[-1]["seq"],
        "original_last_dt_ns": last_dt,
        "applied_delta_ns": delta_ns,
        "reason": (
            "command clock is 20Hz; the terminal observation arrived ~25ms "
            "after the previous observation instead of 50ms"
        ),
    }


def _fix_future_sensors(observations: list[dict], camera_order: list[str]) -> dict | None:
    frames = []
    for obs in observations:
        stamp = int(obs["timestamp_ns"])
        latest = stamp
        robot = obs.get("robot")
        if isinstance(robot, dict) and "timestamp_ns" in robot:
            latest = max(latest, int(robot["timestamp_ns"]))
        cameras = obs.get("cameras") if isinstance(obs.get("cameras"), dict) else {}
        for key in camera_order:
            frame = cameras.get(key)
            if isinstance(frame, dict) and "timestamp_ns" in frame:
                latest = max(latest, int(frame["timestamp_ns"]))
        if latest > stamp:
            obs["timestamp_ns"] = latest
            frames.append(
                {
                    "seq": obs["seq"],
                    "original_timestamp_ns": stamp,
                    "applied_timestamp_ns": latest,
                    "delta_ns": latest - stamp,
                }
            )
    if not frames:
        return None
    return {
        "repair": "raise_observation_stamp_to_latest_sensor",
        "frames": frames,
        "max_delta_ns": max(item["delta_ns"] for item in frames),
        "reason": (
            "a wrist/external/robot stamp was slightly ahead of observation.timestamp_ns; "
            "only the observation stamp is raised"
        ),
    }


def _fix_non_advancing_cameras(observations: list[dict], camera_order: list[str]) -> dict | None:
    frames = []
    for previous, current in zip(observations, observations[1:]):
        prev_cams = previous.get("cameras") if isinstance(previous.get("cameras"), dict) else {}
        cur_cams = current.get("cameras") if isinstance(current.get("cameras"), dict) else {}
        for key in camera_order:
            prev_frame = prev_cams.get(key)
            cur_frame = cur_cams.get(key)
            if not isinstance(prev_frame, dict) or not isinstance(cur_frame, dict):
                continue
            if "timestamp_ns" not in prev_frame or "timestamp_ns" not in cur_frame:
                continue
            prev_ts = int(prev_frame["timestamp_ns"])
            cur_ts = int(cur_frame["timestamp_ns"])
            if cur_ts <= prev_ts:
                cur_frame["timestamp_ns"] = prev_ts + 1
                frames.append(
                    {
                        "seq": current["seq"],
                        "camera": key,
                        "original_timestamp_ns": cur_ts,
                        "applied_timestamp_ns": prev_ts + 1,
                    }
                )
    if not frames:
        return None
    return {
        "repair": "bump_non_advancing_camera_timestamp",
        "frames": frames,
        "reason": "camera timestamp did not advance; add 1 ns on the copy only",
    }


def _apply_clock_repairs(episode_dir: Path) -> dict | None:
    observations_path = episode_dir / "observations.jsonl"
    observations = _read_jsonl(observations_path)
    camera_order = _camera_order(episode_dir, observations)
    repairs = []
    for fixer in (
        lambda: _fix_terminal_observation(observations),
        lambda: _fix_future_sensors(observations, camera_order),
        lambda: _fix_non_advancing_cameras(observations, camera_order),
    ):
        report = fixer()
        if report is not None:
            repairs.append(report)
    if not repairs:
        return None
    _write_jsonl(observations_path, observations)
    payload = {"schema": "warm.real.clock-postprocess.v1", "repairs": repairs}
    _write_json(episode_dir / "clock_postprocess.json", payload)
    return payload


def _max_obs_grid_error_s(episode_dir: Path, fps: float = TARGET_HZ) -> float:
    stamps = [int(row["timestamp_ns"]) for row in _read_jsonl(episode_dir / "observations.jsonl")]
    if len(stamps) < 2:
        return 0.0
    elapsed = [(stamp - stamps[0]) / 1e9 for stamp in stamps]
    expected = [i / fps for i in range(len(stamps))]
    return max(abs(a - b) for a, b in zip(elapsed, expected, strict=True))


def _discover(sources: list[Path]) -> list[dict]:
    found = []
    for source in sources:
        for meta_path in sorted(source.rglob("episode.json")):
            episode_dir = meta_path.parent
            hz = _command_hz(episode_dir)
            if hz is None or abs(hz - TARGET_HZ) > HZ_TOLERANCE:
                continue
            meta = _read_json(meta_path)
            found.append(
                {
                    "source_dir": episode_dir,
                    "origin": source.name,
                    "session_id": meta["session_id"],
                    "episode_id": meta["episode_id"],
                    "task_id": meta["task_id"],
                    "command_hz": hz,
                }
            )
    if not found:
        raise SystemExit("no measured-20Hz episodes found")
    return found


def pack(sources: list[Path], dest: Path) -> dict:
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"destination is not empty: {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    discovered = _discover(sources)
    by_session: dict[str, list[dict]] = defaultdict(list)
    for item in discovered:
        by_session[item["session_id"]].append(item)
    packed = []
    for session_id, items in sorted(by_session.items()):
        items = sorted(items, key=lambda row: row["episode_id"])
        holdout = items[-1]
        for item in items:
            is_dev = item is holdout
            session = f"{session_id}_holdout" if is_dev else session_id
            new_id = f"{session}__{item['episode_id']}"
            dest_dir = dest / session / item["episode_id"]
            _copytree(item["source_dir"], dest_dir)
            meta = _read_json(dest_dir / "episode.json")
            meta["episode_id"] = new_id
            meta["session_id"] = session
            meta["split"] = "dev" if is_dev else "train"
            _write_json(dest_dir / "episode.json", meta)
            repair = _apply_clock_repairs(dest_dir)
            packed.append(
                {
                    "id": new_id,
                    "path": dest_dir.relative_to(dest).as_posix(),
                    "split": meta["split"],
                    "task_id": item["task_id"],
                    "origin": item["origin"],
                    "source_session_id": session_id,
                    "command_hz": item["command_hz"],
                    "clock_repair": repair,
                    "obs_grid_error_s": _max_obs_grid_error_s(dest_dir),
                }
            )
    summary = {
        "schema": "warm.real.pack-20hz.v1",
        "target_hz": TARGET_HZ,
        "episodes": packed,
        "counts": {
            "total": len(packed),
            "train": sum(row["split"] == "train" for row in packed),
            "dev": sum(row["split"] == "dev" for row in packed),
            "clock_repaired": sum(row["clock_repair"] is not None for row in packed),
        },
    }
    _write_json(dest / "PACK.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        action="append",
        required=True,
        help="raw collection root (repeatable), e.g. tmp/WARM_real/pilot_v2",
    )
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = pack([path.expanduser().resolve() for path in args.source], args.dest.expanduser().resolve())
    print(json.dumps(summary["counts"], ensure_ascii=False, indent=2))
    print(f"wrote {args.dest / 'PACK.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
