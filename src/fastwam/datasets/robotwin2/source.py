"""Read explicit RoboTwin layouts; do not infer action meaning from vector size.

Upstream contracts: RoboTwin envs/utils/pkl2hdf5.py and XPolicyLab
utils/process_data.py. Native actions are already shifted next-observed qpos.
They must not be shifted a second time or described as measured commands.
"""
from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from fastwam.preprocessing.contracts import (
    PreparationError, RawEpisode, inside, read_json, required_text,
)


def decode_rgb(cell: Any, encoding: str) -> np.ndarray:
    """Decode known producers, with the upstream JPEG COM color marker.

    Pillow reports the XPL-RGB1 COM segment. Unmarked XPolicyLab/legacy JPEGs
    were produced by passing RGB directly to cv2.imencode: reverse the
    standards-compliant decoder's BGR-valued result exactly once.
    """
    from PIL import Image
    array = np.asarray(cell)
    if encoding == "rgb_array":
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
            raise PreparationError("rgb_array requires an actual uint8 HWC RGB image")
        return array.copy()
    if encoding not in {"legacy_opencv_rgb_jpeg", "xpolicylab_jpeg"}:
        raise PreparationError(f"Unsupported explicit image encoding: {encoding}")
    if isinstance(cell, (bytes, np.bytes_)):
        payload = bytes(cell)
    elif array.dtype == np.uint8 and array.ndim == 1:
        payload = array.tobytes()
    else:
        raise PreparationError("Expected an encoded JPEG byte string or uint8 vector")
    with Image.open(BytesIO(payload)) as image:
        if image.format != "JPEG":
            raise PreparationError("Encoded RoboTwin profile requires JPEG; select rgb_array for arrays")
        # Inspect all COM segments, not just Pillow's last-comment convenience field.
        marked = any(kind == "COM" and data == b"XPL-RGB1" for kind, data in image.applist)
        if encoding == "legacy_opencv_rgb_jpeg" and marked:
            raise PreparationError("Found a new RGB marker in an explicitly legacy JPEG profile")
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if not marked:
            rgb = rgb[..., ::-1]
        return np.ascontiguousarray(rgb)


class RoboTwin2Adapter:
    def __init__(self, source: Mapping[str, Any], profile: Mapping[str, Any]):
        self.root = Path(source["root"]).resolve()
        self.layout = source["layout"]
        self.encoding = source["image_encoding"]
        self.profile = profile
        if self.layout not in {"legacy", "xpolicylab_v1"}:
            raise PreparationError("Select legacy or xpolicylab_v1; formats are never guessed")
        if profile["action_dim"] != 14 or profile["state_dim"] != 14:
            raise PreparationError("This RoboTwin adapter supports the dual 6-joint + 1-gripper layout only")
        if profile["control_mode"] != "robotwin_bimanual_qpos_plus_grippers":
            raise PreparationError("RoboTwin labels are next-observed qpos reference trajectories")
        if profile["camera_keys"] != ["cam_high", "cam_left_wrist", "cam_right_wrist"]:
            raise PreparationError("RoboTwin camera order must be head, left wrist, right wrist")

    def read(self, entry: Mapping[str, Any]) -> RawEpisode:
        import h5py
        path = inside(self.root, entry["path"])
        files = [path]
        with h5py.File(path, "r") as h5:
            if self.layout == "legacy":
                qpos = np.asarray(h5["joint_action/vector"], dtype=np.float32)
                if qpos.ndim != 2 or qpos.shape[1] != 14 or len(qpos) < 3:
                    raise PreparationError("Legacy joint_action/vector must be [N>=3,14]")
                states, actions = qpos[:-1], qpos[1:]
                camera_paths = [f"observation/{k}/rgb" for k in ("head_camera", "left_camera", "right_camera")]
                paths_rows = [len(qpos)] * 3
                instruction_path = inside(self.root, entry["instructions"])
                files.append(instruction_path)
                instructions = read_json(instruction_path)
                if not isinstance(instructions, dict) or not instructions.get("seen"):
                    raise PreparationError("Legacy instructions JSON requires a nonempty seen list")
                variants = instructions["seen"]
                terminal = {"state": qpos[-1].tolist(), "raw_frame_index": len(qpos) - 1}
            else:
                version = h5["data_format_version"][()]
                if isinstance(version, bytes):
                    version = version.decode("utf-8")
                if version != "v1.0":
                    raise PreparationError(f"Unsupported XPolicyLab data_format_version: {version!r}")
                fps = float(np.asarray(h5["additional_info/frequency"]))
                if fps != self.profile["fps"]:
                    raise PreparationError("Native collection frequency differs from configured fps")
                fields = [("left_arm_joint_states", 6), ("left_ee_joint_states", 1),
                          ("right_arm_joint_states", 6), ("right_ee_joint_states", 1)]
                packed = {}
                for group in ("state", "action"):
                    arrays = []
                    for name, dim in fields:
                        value = np.asarray(h5[f"{group}/{name}"], dtype=np.float32)
                        if value.ndim != 2 or value.shape[1] != dim:
                            raise PreparationError(f"{group}/{name} must have shape [N,{dim}]")
                        arrays.append(value)
                    packed[group] = np.concatenate(arrays, axis=1)
                states, actions = packed["state"], packed["action"]
                if states.shape != actions.shape or not np.allclose(actions[:-1], states[1:], atol=1e-6, rtol=0):
                    raise PreparationError("Native next-observed-qpos alignment is inconsistent")
                camera_paths = [f"vision/{k}/colors" for k in ("cam_head", "cam_left_wrist", "cam_right_wrist")]
                paths_rows = [len(states)] * 3
                variants = json.loads(h5["instructions"][()])
                terminal = {"state": actions[-1].tolist(), "rgb_available": False}
            if not isinstance(variants, list) or not variants or any(not isinstance(x, str) for x in variants):
                raise PreparationError("Instruction variants must be a nonempty list of strings")
            instruction = required_text(variants[0], "primary instruction")
            images = {}
            for camera, key, expected in zip(self.profile["camera_keys"], camera_paths, paths_rows, strict=True):
                source_images = h5[key]
                if len(source_images) != expected:
                    raise PreparationError(f"{key} is not aligned with states")
                # Validate/decode the legacy terminal image too, then retain it in raw HDF5.
                decoded = [decode_rgb(cell, self.encoding) for cell in source_images]
                images[camera] = np.stack(decoded[:len(states)])
        episode = RawEpisode(entry["id"], required_text(entry["task"], "task"), instruction,
                             entry["split"], states, actions, images, tuple(files),
                             {"layout": self.layout, "image_encoding": self.encoding,
                              "instruction_variants_seen": variants, "terminal": terminal,
                              "action_provenance": "next_observed_qpos_not_recorded_command"})
        episode.validate(self.profile)
        return episode
