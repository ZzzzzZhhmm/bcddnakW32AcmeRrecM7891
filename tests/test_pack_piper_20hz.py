from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "real"))
import pack_piper_20hz as pack  # noqa: E402


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _episode_meta(session_id: str, episode_id: str, task_id: str) -> dict:
    return {
        "schema": "warm.real.episode.v1",
        "episode_id": episode_id,
        "session_id": session_id,
        "task_id": task_id,
        "split": "train",
        "camera_order": ["external", "wrist"],
        "nominal_action_hz": 20.0,
    }


def _obs(seq: int, stamp: int, *, wrist: int | None = None, external: int | None = None) -> dict:
    wrist_ts = stamp if wrist is None else wrist
    ext_ts = stamp if external is None else external
    return {
        "seq": seq,
        "timestamp_ns": stamp,
        "robot": {"timestamp_ns": stamp},
        "cameras": {
            "external": {"timestamp_ns": ext_ts, "path": f"external/{seq:06d}.jpg"},
            "wrist": {"timestamp_ns": wrist_ts, "path": f"wrist/{seq:06d}.jpg"},
        },
    }


def _commands(n: int, dt_ns: int, start: int = 1_000_000_000) -> list[dict]:
    return [{"seq": i, "timestamp_ns": start + i * dt_ns} for i in range(n)]


def _make_episode(
    root: Path,
    *,
    session_id: str,
    episode_id: str,
    task_id: str,
    command_dt_ns: int,
    observations: list[dict],
    n_commands: int | None = None,
) -> Path:
    dest = root / session_id / episode_id
    dest.mkdir(parents=True)
    (dest / "external").mkdir()
    (dest / "wrist").mkdir()
    (dest / "external" / "frame.jpg").write_bytes(b"img")
    _write_json(dest / "episode.json", _episode_meta(session_id, episode_id, task_id))
    _write_jsonl(dest / "observations.jsonl", observations)
    n_commands = len(observations) - 1 if n_commands is None else n_commands
    _write_jsonl(dest / "commands.jsonl", _commands(max(n_commands, 2), command_dt_ns))
    return dest


def test_keeps_measured_20hz_and_drops_16p67(tmp_path: Path) -> None:
    src = tmp_path / "src"
    start = 1_000_000_000
    keep = _make_episode(
        src,
        session_id="sess20",
        episode_id="episode_000",
        task_id="keep",
        command_dt_ns=50_000_000,
        observations=[_obs(0, start), _obs(1, start + 50_000_000), _obs(2, start + 100_000_000)],
    )
    _make_episode(
        src,
        session_id="sess16",
        episode_id="episode_000",
        task_id="drop",
        command_dt_ns=60_000_000,
        observations=[_obs(0, start), _obs(1, start + 60_000_000), _obs(2, start + 120_000_000)],
    )
    dest = tmp_path / "dest"
    summary = pack.pack([src], dest)
    ids = {row["id"] for row in summary["episodes"]}
    assert ids == {"sess20_holdout__episode_000"}
    assert summary["counts"]["total"] == 1
    assert summary["episodes"][0]["split"] == "dev"
    source_meta = json.loads((keep / "episode.json").read_text())
    assert source_meta["episode_id"] == "episode_000"
    assert source_meta["session_id"] == "sess20"
    assert source_meta["split"] == "train"


def test_last_of_session_is_holdout_dev(tmp_path: Path) -> None:
    src = tmp_path / "src"
    start = 1_000_000_000
    obs = [_obs(0, start), _obs(1, start + 50_000_000)]
    _make_episode(src, session_id="sessA", episode_id="episode_000", task_id="a", command_dt_ns=50_000_000, observations=obs)
    _make_episode(src, session_id="sessA", episode_id="episode_001", task_id="a", command_dt_ns=50_000_000, observations=obs)
    summary = pack.pack([src], tmp_path / "dest")
    by_id = {row["id"]: row for row in summary["episodes"]}
    assert by_id["sessA__episode_000"]["split"] == "train"
    assert by_id["sessA_holdout__episode_001"]["split"] == "dev"
    assert summary["counts"] == {"total": 2, "train": 1, "dev": 1, "clock_repaired": 0}


def test_terminal_observation_and_future_wrist_repairs(tmp_path: Path) -> None:
    src = tmp_path / "src"
    start = 1_000_000_000
    observations = [
        _obs(0, start),
        _obs(1, start + 50_000_000),
        _obs(2, start + 75_000_000, wrist=start + 75_000_000 + 250_000),
    ]
    source = _make_episode(
        src,
        session_id="sessB",
        episode_id="episode_000",
        task_id="b",
        command_dt_ns=50_000_000,
        observations=observations,
    )
    dest = tmp_path / "dest"
    summary = pack.pack([src], dest)
    packed = dest / "sessB_holdout" / "episode_000"
    rows = [json.loads(line) for line in (packed / "observations.jsonl").read_text().splitlines() if line.strip()]
    last_dt = rows[-1]["timestamp_ns"] - rows[-2]["timestamp_ns"]
    assert abs(last_dt - 50_000_000) <= 1_000_000
    assert rows[-1]["timestamp_ns"] >= rows[-1]["cameras"]["wrist"]["timestamp_ns"]
    assert rows[-1]["timestamp_ns"] >= rows[-1]["cameras"]["wrist"]["timestamp_ns"]
    assert rows[-1]["cameras"]["wrist"]["timestamp_ns"] - rows[-2]["cameras"]["wrist"]["timestamp_ns"] > 0
    source_rows = [json.loads(line) for line in (source / "observations.jsonl").read_text().splitlines() if line.strip()]
    assert source_rows[-1]["timestamp_ns"] == start + 75_000_000
    repairs = {item["repair"] for item in summary["episodes"][0]["clock_repair"]["repairs"]}
    assert "shift_terminal_observation_to_20hz_tick" in repairs
    assert "raise_observation_stamp_to_latest_sensor" in repairs


def test_json_rewrite_does_not_mutate_hardlinked_source(tmp_path: Path) -> None:
    src = tmp_path / "src"
    start = 1_000_000_000
    source = _make_episode(
        src,
        session_id="sessC",
        episode_id="episode_000",
        task_id="c",
        command_dt_ns=50_000_000,
        observations=[_obs(0, start), _obs(1, start + 50_000_000)],
    )
    dest = tmp_path / "dest"
    pack.pack([src], dest)
    packed_json = dest / "sessC_holdout" / "episode_000" / "episode.json"
    assert os.stat(packed_json).st_ino != os.stat(source / "episode.json").st_ino
    packed_json.write_text('{"mutated": true}\n', encoding="utf-8")
    source_meta = json.loads((source / "episode.json").read_text())
    assert source_meta["episode_id"] == "episode_000"
    assert source_meta["split"] == "train"
