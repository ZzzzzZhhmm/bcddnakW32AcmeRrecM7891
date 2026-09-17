"""Convert recorded commanded TCP targets, never finite-difference measured actions."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from fastwam.real.episodes import load_episode
from fastwam.preprocessing.contracts import PreparationError, RawEpisode, inside, required_text


def quaternion_rotvec(value: Any) -> np.ndarray:
    q = np.asarray(value, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q) - 1) > 1e-4:
        raise PreparationError("Expected a unit xyzw quaternion")
    q = q / np.linalg.norm(q)
    # q and -q must produce exactly the same principal rotation, including pi.
    pivot = 3 if abs(q[3]) > 1e-12 else int(np.argmax(np.abs(q[:3])))
    if q[pivot] < 0:
        q = -q
    norm = np.linalg.norm(q[:3])
    return 2 * q[:3] if norm < 1e-10 else q[:3] * (2 * np.arctan2(norm, max(0, q[3])) / norm)


def base_rotation_delta(target: Any, current: Any) -> np.ndarray:
    """Log(R_target R_current^T), expressed in the common base/control frame."""
    a, b = np.asarray(target, dtype=np.float64), np.asarray(current, dtype=np.float64)
    quaternion_rotvec(a)
    quaternion_rotvec(b)
    b = b * np.array([-1, -1, -1, 1])
    xyz = a[3] * b[:3] + b[3] * a[:3] + np.cross(a[:3], b[:3])
    return quaternion_rotvec(np.r_[xyz, a[3] * b[3] - np.dot(a[:3], b[:3])])


class PiperTeleopAdapter:
    def __init__(self, source: Mapping[str, Any], profile: Mapping[str, Any]):
        self.root = Path(source["root"]).resolve()
        self.source, self.profile = source, profile
        self.sessions: dict[str, str] = {}
        if (profile["action_dim"], profile["state_dim"]) != (7, 7):
            raise PreparationError("Piper single-active-arm profile requires action=7, state=7")
        if profile["control_mode"] != "base_delta_tcp_rotvec_plus_absolute_gripper_width_m":
            raise PreparationError("Piper profile requires base-frame Cartesian deltas and absolute width")
        if profile["camera_keys"] != ["external", "wrist"]:
            raise PreparationError("Piper profile requires external then active-arm wrist camera")
        for name in ("calibration_id", "control_frame", "tcp_frame"):
            required_text(source.get(name), name)
        tolerance = source.get("timestamp_tolerance_s")
        if not isinstance(tolerance, (float, int)) or not 0 < tolerance < 0.5 / profile["fps"]:
            raise PreparationError("Declare timestamp_tolerance_s > 0 and less than half a tick")

    def read(self, entry: Mapping[str, Any]) -> RawEpisode:
        from PIL import Image
        root = inside(self.root, entry["path"])
        meta, observations, commands, outcome = load_episode(
            root, allow_synthetic=self.source.get("allow_synthetic", False))
        if meta["episode_id"] != entry["id"] or meta["split"] != entry["split"]:
            raise PreparationError("Raw episode identity/split must match the source manifest")
        for name in ("calibration_id", "control_frame", "tcp_frame"):
            if meta[name] != self.source[name]:
                raise PreparationError(f"Mixed or unverified {name}; use a separate dataset version")
        if meta["camera_order"] != self.profile["camera_keys"]:
            raise PreparationError("Camera order changed between collection and preprocessing")
        if meta["nominal_action_hz"] != self.profile["fps"]:
            raise PreparationError("Piper nominal action rate differs from configured fps")
        if outcome["status"] not in self.source["include_outcomes"]:
            raise PreparationError("Episode outcome not in include_outcomes; curate manifest explicitly")
        previous = self.sessions.setdefault(meta["session_id"], meta["split"])
        if previous != meta["split"]:
            raise PreparationError("A real collection session cannot cross train/dev/test splits")
        # Preserve physical timing. Do not convert a dropped/irregular stream into
        # a fictitious constant-rate trajectory by reindexing or interpolation.
        stamps = np.array([o["timestamp_ns"] for o in observations], dtype=np.int64)
        elapsed = (stamps - stamps[0]) / 1e9
        expected = np.arange(len(stamps)) / self.profile["fps"]
        if np.max(np.abs(elapsed - expected)) > self.source["timestamp_tolerance_s"]:
            raise PreparationError("Nonuniform observation clock; recollect or explicitly segment upstream")
        command_times = np.array([c["timestamp_ns"] for c in commands], dtype=np.int64)
        if len(command_times) > 1:
            jitter = np.diff(command_times) / 1e9 - 1 / self.profile["fps"]
            if np.max(np.abs(jitter)) > self.source["timestamp_tolerance_s"]:
                raise PreparationError("Nonuniform accepted-command clock")
        states, actions = [], []
        for obs, command in zip(observations[:-1], commands, strict=True):
            robot = obs["robot"]
            xyz, quat = robot["tcp_position_m"], robot["tcp_quaternion_xyzw"]
            states.append([*xyz, *quaternion_rotvec(quat), robot["gripper_width_m"]])
            actions.append([*(np.array(command["tcp_position_m"]) - xyz),
                            *base_rotation_delta(command["tcp_quaternion_xyzw"], quat),
                            command["gripper_width_m"]])
        images, files = {}, [root / name for name in ("episode.json", "observations.jsonl", "commands.jsonl", "outcome.json")]
        for camera in meta["camera_order"]:
            decoded = []
            for obs in observations:
                record = obs["cameras"][camera]
                path = inside(root, record["path"])
                files.append(path)
                with Image.open(path) as image:
                    if image.mode != "RGB" or image.size != (record["width"], record["height"]):
                        raise PreparationError("Decoded RGB image does not match logged camera metadata")
                    decoded.append(np.array(image, dtype=np.uint8))
            images[camera] = np.stack(decoded[:-1])
        annotations = {"session_id": meta["session_id"], "calibration_id": meta["calibration_id"],
                       "control_frame": meta["control_frame"], "tcp_frame": meta["tcp_frame"],
                       "synthetic": meta["synthetic"], "outcome": outcome,
                       "observation_timestamp_ns": stamps.tolist(),
                       "command_timestamp_ns": command_times.tolist(),
                       "terminal_observation": observations[-1],
                       "action_provenance": "accepted_command_target_relative_to_factual_tcp",
                       "rotation_composition": "R_command = Exp(delta_rotvec) @ R_observed"}
        episode = RawEpisode(meta["episode_id"], meta["task_id"], meta["instruction"], meta["split"],
                             np.asarray(states), np.asarray(actions), images, tuple(dict.fromkeys(files)), annotations)
        episode.validate(self.profile)
        return episode
