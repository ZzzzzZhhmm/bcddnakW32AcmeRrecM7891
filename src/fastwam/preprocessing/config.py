"""Versioned explicit preprocessing configuration and fail-closed stage outputs."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from uuid import uuid4
import os

from .contracts import PreparationError, read_json, required_text, write_json


ADAPTERS = ("libero_lerobot", "rmbench", "robotwin2", "piper_teleop")


def load_config(path: Path) -> dict:
    cfg = read_json(path)
    if cfg.get("schema") != "warm.preprocessing.v1" or cfg.get("adapter") not in ADAPTERS:
        raise PreparationError(f"Expected warm.preprocessing.v1 and adapter in {ADAPTERS}")
    if type(cfg.get("test_only", False)) is not bool:
        raise PreparationError("test_only must be a JSON boolean")
    if type(cfg.get("source", {}).get("allow_synthetic", False)) is not bool:
        raise PreparationError("allow_synthetic must be a JSON boolean")
    for key in ("dataset_id", "output"):
        required_text(cfg.get(key), key)
    profile = cfg["profile"]
    for key in ("action_dim", "state_dim", "action_horizon", "action_video_freq_ratio"):
        if type(profile.get(key)) is not int or profile[key] <= 0:
            raise PreparationError(f"profile.{key} must be a positive integer")
    raw_fps = profile.get("fps")
    if type(raw_fps) is int and raw_fps > 0:
        pass
    elif (isinstance(raw_fps, list) and len(raw_fps) == 2
          and all(type(item) is int and item > 0 for item in raw_fps)):
        profile["fps"] = raw_fps[0] / raw_fps[1]
    else:
        raise PreparationError("profile.fps must be a positive integer or [numerator, denominator]")
    horizon, ratio = profile["action_horizon"], profile["action_video_freq_ratio"]
    if horizon % ratio or (horizon // ratio) % 4:
        raise PreparationError("FastWAM horizon / action_video_freq_ratio must be divisible by 4")
    for key in ("control_mode", "embodiment"):
        required_text(profile.get(key), "profile." + key)
    if profile["normalization"] not in {"min/max", "z-score"}:
        raise PreparationError("Only the existing min/max and z-score normalizers are supported")
    cameras = profile["camera_keys"]
    if not isinstance(cameras, list) or not cameras or len(cameras) != len(set(cameras)):
        raise PreparationError("camera_keys must be an ordered unique list")
    for camera in cameras:
        required_text(camera, "camera key")
        if not all(char.isalnum() or char == "_" for char in camera):
            raise PreparationError("Use short camera names without paths or observation.images prefix")
    if profile["semantic_camera"] not in cameras:
        raise PreparationError("semantic_camera must be in camera_keys")
    if profile["image_layout"] not in {"horizontal_224", "robotwin_3cam"}:
        raise PreparationError("Choose a verified image layout")
    if profile["image_layout"] == "robotwin_3cam" and len(cameras) != 3:
        raise PreparationError("robotwin_3cam requires exactly three cameras")
    for field, dim in (("gripper_action_indices", profile["action_dim"]), ("gripper_state_indices", profile["state_dim"])):
        indices = profile[field]
        if not indices or indices != sorted(set(indices)) or any(type(i) is not int or not 0 <= i < dim for i in indices):
            raise PreparationError(f"Invalid {field}")
    mask = profile["delta_action_mask"]
    if len(mask) != profile["action_dim"] or any(type(x) is not bool for x in mask):
        raise PreparationError("delta_action_mask must explicitly describe every action channel")
    if cfg["adapter"] in {"robotwin2", "rmbench"}:
        expected = {"action_dim": 14, "state_dim": 14, "image_layout": "robotwin_3cam",
                    "normalization": "z-score", "gripper_action_indices": [6, 13], "gripper_state_indices": [6, 13],
                    "delta_action_mask": [False] * 14, "control_mode": "robotwin_bimanual_qpos_plus_grippers",
                    "embodiment": "robotwin_aloha_agilex", "camera_keys": ["cam_high", "cam_left_wrist", "cam_right_wrist"]}
    else:
        real = cfg["adapter"] == "piper_teleop"
        expected = {"action_dim": 7, "state_dim": 7 if real else 8, "image_layout": "horizontal_224",
                    "normalization": "min/max", "gripper_action_indices": [6], "gripper_state_indices": [6] if real else [6, 7],
                    "delta_action_mask": [True] * 6 + [False],
                    "control_mode": "base_delta_tcp_rotvec_plus_absolute_gripper_width_m" if real else "libero_delta_eef_axis_angle_plus_gripper",
                    "embodiment": "piper_single_active_6dof" if real else "libero_panda",
                    "camera_keys": ["external", "wrist"] if real else ["image", "wrist_image"]}
    for key, value in expected.items():
        if profile[key] != value:
            raise PreparationError(f"{cfg['adapter']} profile.{key} must be {value!r}")
    # Paths are relative to the config file, never a shell-dependent working directory.
    cfg = deepcopy(cfg)
    def resolve(value: str) -> str:
        required_text(value, "path")
        return str((path.resolve().parent / value).resolve())
    cfg["output"] = resolve(cfg["output"])
    source = cfg["source"]
    for key in ("root", "manifest", "catalog"):
        if key in source:
            source[key] = resolve(source[key])
    if "roots" in source:
        source["roots"] = [resolve(root) for root in source["roots"]]
    output = Path(cfg["output"])
    for raw in ([source["root"]] if "root" in source else source.get("roots", [])):
        root = Path(raw)
        if output == root or output.is_relative_to(root) or root.is_relative_to(output):
            raise PreparationError("Raw source and output must be separate non-nested directories")
    for key in ("dino_checkpoint", "vae_checkpoint"):
        if cfg.get("encoders", {}).get(key):
            cfg["encoders"][key] = resolve(cfg["encoders"][key])
    return cfg


@contextmanager
def stage_directory(target: Path):
    """Single-writer publication; failed staging stays available for diagnosis."""
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = target.parent / ("." + target.name + ".lock")
    fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    staging = target.parent / ("." + target.name + ".staging-" + uuid4().hex)
    try:
        os.close(fd)
        if target.exists():
            raise FileExistsError(f"Immutable stage already exists: {target}")
        staging.mkdir()
        try:
            yield staging
            if target.exists():
                raise FileExistsError(target)
            staging.rename(target)
        except Exception as exc:
            write_json(staging / "FAILED.json", {"error": type(exc).__name__, "message": str(exc)})
            raise
    finally:
        lock.unlink(missing_ok=True)
