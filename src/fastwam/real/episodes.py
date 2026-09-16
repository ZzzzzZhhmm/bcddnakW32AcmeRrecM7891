"""Raw single-arm Cartesian demonstration interchange, without robot I/O.

Targets are controller commands in metres/quaternion xyzw, NOT normalized
LIBERO actions. Conversion into a training profile is a separate validated step.
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path, PurePosixPath


class ContractError(ValueError):
    """A recorded episode does not satisfy the agreed interchange contract."""


def require(condition, message):
    if not condition:
        raise ContractError(message)


def read_json(path):
    def reject_constant(value):
        raise ContractError(f"non-finite JSON constant: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      parse_constant=reject_constant)


def write_json(path, value):
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(encoded + "\n")


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, f"{name}: expected integer >= {minimum}")
    return value


def number(value, name, minimum=None):
    require(type(value) in (int, float) and math.isfinite(value), f"{name}: expected finite number")
    if minimum is not None:
        require(value >= minimum, f"{name}: must be >= {minimum}")
    return value


def vector(value, size, name):
    require(isinstance(value, list) and len(value) == size, f"{name}: expected {size} values")
    for item in value:
        number(item, name)


def pose(value, name):
    vector(value["tcp_position_m"], 3, name + ".tcp_position_m")
    quat = value["tcp_quaternion_xyzw"]
    vector(quat, 4, name + ".tcp_quaternion_xyzw")
    require(abs(sum(x*x for x in quat) - 1.0) < 0.002, name + ": quaternion must have unit norm")
    number(value["gripper_width_m"], name + ".gripper_width_m", 0)


def local_file(root, relative):
    require(isinstance(relative, str) and bool(relative), "image path must be a nonempty string")
    path = PurePosixPath(relative)
    require(not path.is_absolute() and ".." not in path.parts and ":" not in relative
            and "\\" not in relative, "image path must be relative POSIX without traversal")
    resolved = (Path(root) / relative).resolve()
    require(resolved.is_relative_to(Path(root).resolve()), "image symlink escapes episode")
    require(resolved.is_file() and resolved.stat().st_size > 0, f"missing/empty image: {relative}")
    return resolved


def validate_manifest(meta):
    require(meta["schema"] == "warm.real.episode.v1", "unsupported episode schema")
    for field in ("episode_id", "session_id", "task_id", "instruction", "calibration_id",
                  "control_frame", "tcp_frame", "clock_id"):
        require(isinstance(meta[field], str) and bool(meta[field].strip()), f"missing {field}")
    require(meta["split"] in ("train", "dev", "test"), "invalid split")
    require(type(meta["synthetic"]) is bool, "synthetic must be boolean")
    require(meta["camera_order"] == ["external", "wrist"], "v1 requires external then wrist")
    require(meta["action_semantics"] == "commanded_absolute_tcp_pose_and_gripper_width",
            "v1 requires actual commanded Cartesian targets; no pose-difference substitution")
    require(meta["clock_domain"] == "client_monotonic_ns", "all timestamps must share client clock")
    number(meta["nominal_action_hz"], "nominal_action_hz", 0.001)
    integer(meta["max_observation_gap_ns"], "max_observation_gap_ns", 1)
    integer(meta["max_sensor_skew_ns"], "max_sensor_skew_ns", 1)
    return meta


def validate_observation(root, meta, obs, expected_seq, previous=None):
    require(obs["seq"] == expected_seq and type(obs["seq"]) is int, "noncontiguous observation sequence")
    stamp = integer(obs["timestamp_ns"], "observation timestamp")
    if previous is not None:
        gap = stamp - previous["timestamp_ns"]
        require(0 < gap <= meta["max_observation_gap_ns"], "observation time reversed or gap too large")
    require(set(obs["cameras"]) == set(meta["camera_order"]), "incorrect camera set")
    sensors = [obs["robot"]] + [obs["cameras"][key] for key in meta["camera_order"]]
    for sensor in sensors:
        ts = integer(sensor["timestamp_ns"], "sensor timestamp")
        require(0 <= stamp-ts <= meta["max_sensor_skew_ns"], "stale sensor or future timestamp")
    for key in meta["camera_order"]:
        frame = obs["cameras"][key]
        require(frame["color_space"] == "RGB", "image color_space must be RGB")
        integer(frame["width"], "width", 1)
        integer(frame["height"], "height", 1)
        local_file(root, frame["path"])
        if previous is not None:
            require(frame["timestamp_ns"] > previous["cameras"][key]["timestamp_ns"],
                    "camera timestamp did not advance")
    robot = obs["robot"]
    vector(robot["joint_position_rad"], 6, "joint_position_rad")
    pose(robot, "robot")
    if previous is not None:
        require(robot["timestamp_ns"] > previous["robot"]["timestamp_ns"], "robot feedback did not advance")


def validate_command(command, obs, successor):
    require(type(command["seq"]) is int and command["seq"] == obs["seq"], "noncontiguous command sequence")
    require(type(command["observation_seq"]) is int and command["observation_seq"] == obs["seq"],
            "command does not reference preceding observation")
    ts = integer(command["timestamp_ns"], "command timestamp")
    require(obs["timestamp_ns"] <= ts < successor["timestamp_ns"], "command outside observation interval")
    require(successor["robot"]["timestamp_ns"] > ts, "successor feedback predates command")
    require(type(command["accepted"]) is bool, "accepted must be boolean")
    require(command["accepted"], "rejected command: keep diagnostic record, exclude from continuous training segment")
    pose(command, "command")


def read_records(path):
    result = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            require(bool(line.strip()), "empty JSONL record")
            def reject(value):
                raise ContractError(f"non-finite JSON constant: {value}")
            result.append(json.loads(line, parse_constant=reject))
    return result


def load_episode(root, allow_synthetic=False):
    root = Path(root)
    meta = validate_manifest(read_json(root / "episode.json"))
    require(allow_synthetic or not meta["synthetic"], "synthetic data are not real demonstrations")
    observations = read_records(root / "observations.jsonl")
    commands = read_records(root / "commands.jsonl")
    require(len(commands) >= 1 and len(observations) == len(commands)+1,
            "need N commands and N+1 observations including the terminal observation")
    for i, obs in enumerate(observations):
        validate_observation(root, meta, obs, i, observations[i-1] if i else None)
        if i:
            validate_command(commands[i-1], observations[i-1], obs)
    outcome = read_json(root / "outcome.json")
    require(outcome["status"] in ("success", "failure", "aborted"), "invalid outcome")
    integer(outcome["interventions"], "interventions")
    require(isinstance(outcome["notes"], str), "outcome notes must be text")
    return meta, observations, commands, outcome


def audit_episode(root, allow_synthetic=False):
    report = {"schema": "warm.real.episode-audit.v1", "ok": False,
              "qualification": "structural_only_not_training_or_motion_approval",
              "errors": [], "warnings": []}
    try:
        meta, obs, commands, outcome = load_episode(root, allow_synthetic)
        report.update(ok=True, episode_id=meta["episode_id"], split=meta["split"],
                      synthetic=meta["synthetic"], observations=len(obs), commands=len(commands),
                      complete_h32_windows=max(0, len(commands)-32+1), outcome=outcome["status"])
        if len(commands) < 32:
            report["warnings"].append("too short for one H=32 event")
        if outcome["status"] != "success" or outcome["interventions"]:
            report["warnings"].append("requires supervision-mask review before training/bank admission")
        report["warnings"].append("timestamps and image declarations checked; decoding, visual quality, calibration and controller semantics need separate review")
    except (ContractError, KeyError, TypeError, ValueError, OSError) as exc:
        report["errors"].append(str(exc))
    return report


class EpisodeWriter:
    """Append durable records from an ALREADY validated vendor recorder.

    Caller saves camera images first, in this new episode directory. This class
    cannot acquire cameras, connect CAN, or send movement. Incomplete episodes
    retain their files and cannot pass audit without a valid outcome.
    """
    def __init__(self, root, metadata):
        self.meta = copy.deepcopy(validate_manifest(metadata))
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        write_json(self.root / "episode.json", self.meta)
        self.previous = None
        self.count = 0
        self.finished = False

    def _append(self, filename, record):
        encoded = json.dumps(record, ensure_ascii=False, allow_nan=False)
        with (self.root / filename).open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def start(self, observation):
        require(self.previous is None and not self.finished, "episode already started or finished")
        validate_observation(self.root, self.meta, observation, 0)
        self._append("observations.jsonl", observation)
        self.previous = copy.deepcopy(observation)

    def append_transition(self, command, successor):
        require(self.previous is not None and not self.finished, "episode not active")
        validate_observation(self.root, self.meta, successor, self.count+1, self.previous)
        validate_command(command, self.previous, successor)
        self._append("commands.jsonl", command)
        self._append("observations.jsonl", successor)
        self.previous = copy.deepcopy(successor)
        self.count += 1

    def finish(self, status, interventions=0, notes=""):
        require(self.count > 0 and not self.finished, "empty or already finished episode")
        require(status in ("success", "failure", "aborted"), "invalid status")
        integer(interventions, "interventions")
        require(isinstance(notes, str), "notes must be text")
        write_json(self.root / "outcome.json", dict(status=status, interventions=interventions, notes=notes))
        self.finished = True
