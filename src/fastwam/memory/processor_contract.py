"""Closed-world processor recipes shared by M1 and online rollout.

Profiles are explicit rather than permissive.  LIBERO keeps its original
two-camera/7D delta-action contract; RoboTwin/RMBench uses the native
three-camera/14D bimanual qpos contract.  A caller must name the latter and
there is no shape-based auto-detection.
"""

from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path
from typing import Any, Mapping

from .manifest import sha256_canonical_json, sha256_file


M1_LIBERO_PROCESSOR_RECIPE = "warm.m1-libero-processor-recipe"
M1_LIBERO_PROCESSOR_RECIPE_VERSION = 1
M1_ROBOTWIN_PROCESSOR_RECIPE = "warm.m1-robotwin-processor-recipe"
M1_ROBOTWIN_PROCESSOR_RECIPE_VERSION = 1


class ProcessorContractError(ValueError):
    """Raised when online preprocessing can differ from M1 preprocessing."""


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProcessorContractError(f"{field} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise ProcessorContractError(f"{field} keys must be strings")
    return value


def _list(value: Any, field: str) -> list[Any]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
    ):
        raise ProcessorContractError(f"{field} must be a list")
    return list(value)


def _exact_keys(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ProcessorContractError(
            f"invalid {field} fields; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _meta(value: Any, *, field: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, raw in enumerate(_list(value, field)):
        item = _mapping(raw, f"{field}[{index}]")
        required = {"key", "raw_shape", "shape"}
        if not required.issubset(item):
            raise ProcessorContractError(f"{field}[{index}] is incomplete")
        raw_shape = item["raw_shape"]
        shape = item["shape"]
        output.append(
            {
                "key": str(item["key"]),
                "raw_shape": (
                    [int(v) for v in raw_shape]
                    if isinstance(raw_shape, (list, tuple))
                    else int(raw_shape)
                ),
                "shape": (
                    [int(v) for v in shape]
                    if isinstance(shape, (list, tuple))
                    else int(shape)
                ),
            }
        )
    return output


def extract_m1_libero_processor_recipe(config: Mapping[str, Any]) -> dict[str, Any]:
    """Extract and validate every field that can affect factual online input."""

    root = _mapping(config, "data config")
    if "data" in root:
        root = _mapping(root["data"], "data")
    train = _mapping(root.get("train"), "data.train")
    shape_meta = _mapping(train.get("shape_meta"), "data.train.shape_meta")
    images = _meta(shape_meta.get("images"), field="shape_meta.images")
    actions = _meta(shape_meta.get("action"), field="shape_meta.action")
    states = _meta(shape_meta.get("state"), field="shape_meta.state")
    expected_images = [
        {"key": "image", "raw_shape": [3, 512, 512], "shape": [3, 224, 224]},
        {
            "key": "wrist_image",
            "raw_shape": [3, 512, 512],
            "shape": [3, 224, 224],
        },
    ]
    if images != expected_images:
        raise ProcessorContractError("M1 requires the exact two-camera 512->224 metadata")
    if actions != [{"key": "default", "raw_shape": 7, "shape": 7}]:
        raise ProcessorContractError("M1 requires exactly one default 7D action")
    if states != [{"key": "default", "raw_shape": 8, "shape": 8}]:
        raise ProcessorContractError("M1 requires exactly one default 8D state")
    if [int(v) for v in _list(train.get("video_size"), "video_size")] != [224, 448]:
        raise ProcessorContractError("M1 requires video_size=[224,448]")
    if train.get("concat_multi_camera") != "horizontal":
        raise ProcessorContractError("M1 requires horizontal camera concatenation")

    processor = _mapping(train.get("processor"), "data.train.processor")
    processor_fields = {
        "_target_",
        "shape_meta",
        "num_obs_steps",
        "num_output_cameras",
        "action_output_dim",
        "proprio_output_dim",
        "delta_action_dim_mask",
        "action_state_transforms",
        "use_stepwise_action_norm",
        "norm_default_mode",
        "norm_exception_mode",
        "action_state_merger",
        "train_transforms",
        "val_transforms",
    }
    _exact_keys(processor, processor_fields, "data.train.processor")
    if processor["_target_"] != (
        "fastwam.datasets.lerobot.processors.fastwam_processor.FastWAMProcessor"
    ):
        raise ProcessorContractError("M1 requires FastWAMProcessor")
    num_frames = int(train.get("num_frames"))
    if int(processor["num_obs_steps"]) != num_frames or num_frames <= 1:
        raise ProcessorContractError("processor.num_obs_steps must equal num_frames")
    scalar_expected = {
        "num_output_cameras": 2,
        "action_output_dim": 7,
        "proprio_output_dim": 8,
        "action_state_transforms": None,
        "use_stepwise_action_norm": False,
        "norm_default_mode": "min/max",
        "norm_exception_mode": None,
    }
    for field, expected in scalar_expected.items():
        if processor[field] != expected:
            raise ProcessorContractError(
                f"M1 processor {field} must equal {expected!r}"
            )
    if processor["shape_meta"] != shape_meta:
        raise ProcessorContractError("processor.shape_meta must equal train.shape_meta")
    mask = _mapping(processor["delta_action_dim_mask"], "delta_action_dim_mask")
    if set(mask) != {"default"} or list(mask["default"]) != [
        True,
        True,
        True,
        True,
        True,
        True,
        False,
    ]:
        raise ProcessorContractError("M1 delta-action mask is not exact")
    merger = _mapping(processor["action_state_merger"], "action_state_merger")
    if set(merger) != {"_target_"} or merger["_target_"] != (
        "fastwam.datasets.lerobot.transforms.action_state_merger.ConcatLeftAlign"
    ):
        raise ProcessorContractError("M1 requires unpadded ConcatLeftAlign")

    def transforms(raw: Any, field: str) -> list[dict[str, Any]]:
        items = [_mapping(item, field) for item in _list(raw, field)]
        if len(items) != 2:
            raise ProcessorContractError(f"{field} must contain exactly two transforms")
        _exact_keys(items[0], {"_target_"}, f"{field}[0]")
        _exact_keys(items[1], {"_target_", "size"}, f"{field}[1]")
        result = [
            {"_target_": items[0]["_target_"]},
            {
                "_target_": items[1]["_target_"],
                "size": [int(v) for v in _list(items[1]["size"], "resize size")],
            },
        ]
        if result != [
            {
                "_target_": "fastwam.datasets.lerobot.transforms.image.ToTensor"
            },
            {"_target_": "torchvision.transforms.Resize", "size": [224, 224]},
        ]:
            raise ProcessorContractError(
                f"{field} must be exactly ToTensor -> Resize([224,224])"
            )
        return result

    return {
        "schema": M1_LIBERO_PROCESSOR_RECIPE,
        "version": M1_LIBERO_PROCESSOR_RECIPE_VERSION,
        "num_frames": num_frames,
        "video_size": [224, 448],
        "concat_multi_camera": "horizontal",
        "shape_meta": {"images": images, "action": actions, "state": states},
        "processor": {
            "target": processor["_target_"],
            "num_obs_steps": num_frames,
            "num_output_cameras": 2,
            "action_output_dim": 7,
            "proprio_output_dim": 8,
            "delta_action_dim_mask": {"default": list(mask["default"])},
            "action_state_transforms": None,
            "use_stepwise_action_norm": False,
            "norm_default_mode": "min/max",
            "norm_exception_mode": None,
            "action_state_merger": dict(merger),
            "train_transforms": transforms(
                processor["train_transforms"], "train_transforms"
            ),
            "val_transforms": transforms(
                processor["val_transforms"], "val_transforms"
            ),
        },
    }


def extract_m1_robotwin_processor_recipe(config: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the exact native-14D RoboTwin/RMBench processor contract."""

    root = _mapping(config, "data config")
    if "data" in root:
        root = _mapping(root["data"], "data")
    train = _mapping(root.get("train"), "data.train")
    shape_meta = _mapping(train.get("shape_meta"), "data.train.shape_meta")
    images = _meta(shape_meta.get("images"), field="shape_meta.images")
    actions = _meta(shape_meta.get("action"), field="shape_meta.action")
    states = _meta(shape_meta.get("state"), field="shape_meta.state")
    keys = ("cam_high", "cam_left_wrist", "cam_right_wrist")
    allowed_raw_sizes = ([3, 480, 640], [3, 240, 320])
    valid_images = (
        len(images) == len(keys)
        and all(item.get("key") == key for item, key in zip(images, keys))
        and all(item.get("shape") == [3, 240, 320] for item in images)
        and len({tuple(item.get("raw_shape", ())) for item in images}) == 1
        and images[0].get("raw_shape") in allowed_raw_sizes
    )
    if not valid_images:
        raise ProcessorContractError(
            "RoboTwin/RMBench M1 requires exact head/left-wrist/right-wrist "
            "metadata with one factual raw resolution (480x640 RoboTwin or "
            "240x320 RMBench) and 240x320 processor output"
        )
    expected_vector = [{"key": "default", "raw_shape": 14, "shape": 14}]
    if actions != expected_vector or states != expected_vector:
        raise ProcessorContractError(
            "RoboTwin M1 requires one native default 14D action and state"
        )
    if [int(v) for v in _list(train.get("video_size"), "video_size")] != [384, 320]:
        raise ProcessorContractError("RoboTwin M1 requires video_size=[384,320]")
    if train.get("concat_multi_camera") != "robotwin":
        raise ProcessorContractError(
            "RoboTwin M1 requires concat_multi_camera='robotwin'"
        )

    processor = _mapping(train.get("processor"), "data.train.processor")
    required_fields = {
        "_target_",
        "shape_meta",
        "num_obs_steps",
        "num_output_cameras",
        "action_output_dim",
        "proprio_output_dim",
        "action_state_transforms",
        "use_stepwise_action_norm",
        "norm_default_mode",
        "norm_exception_mode",
        "action_state_merger",
        "train_transforms",
        "val_transforms",
    }
    missing = required_fields - set(processor)
    if missing:
        raise ProcessorContractError(
            f"RoboTwin processor is incomplete; missing={sorted(missing)}"
        )
    if processor["_target_"] != (
        "fastwam.datasets.lerobot.processors.fastwam_processor.FastWAMProcessor"
    ):
        raise ProcessorContractError("RoboTwin M1 requires FastWAMProcessor")
    num_frames = int(train.get("num_frames"))
    if num_frames != 33 or int(processor["num_obs_steps"]) != num_frames:
        raise ProcessorContractError("RoboTwin M1 requires exactly 33 observation frames")
    scalar_expected = {
        "num_output_cameras": 3,
        "action_output_dim": 14,
        "proprio_output_dim": 14,
        "action_state_transforms": None,
        "use_stepwise_action_norm": False,
        "norm_default_mode": "z-score",
        "norm_exception_mode": None,
    }
    for field, expected in scalar_expected.items():
        if processor[field] != expected:
            raise ProcessorContractError(
                f"RoboTwin M1 processor {field} must equal {expected!r}"
            )
    if processor["shape_meta"] != shape_meta:
        raise ProcessorContractError("processor.shape_meta must equal train.shape_meta")
    if processor.get("delta_action_dim_mask") not in (None, {}):
        raise ProcessorContractError(
            "native RoboTwin qpos must not use a LIBERO delta-action mask"
        )
    merger = _mapping(processor["action_state_merger"], "action_state_merger")
    if set(merger) != {"_target_"} or merger["_target_"] != (
        "fastwam.datasets.lerobot.transforms.action_state_merger.ConcatLeftAlign"
    ):
        raise ProcessorContractError("RoboTwin M1 requires unpadded ConcatLeftAlign")

    def transforms(raw: Any, field: str) -> list[dict[str, Any]]:
        items = [_mapping(item, field) for item in _list(raw, field)]
        if len(items) != 2:
            raise ProcessorContractError(f"{field} must contain exactly two transforms")
        _exact_keys(items[0], {"_target_"}, f"{field}[0]")
        _exact_keys(items[1], {"_target_", "size"}, f"{field}[1]")
        result = [
            {"_target_": items[0]["_target_"]},
            {
                "_target_": items[1]["_target_"],
                "size": [int(v) for v in _list(items[1]["size"], "resize size")],
            },
        ]
        expected = [
            {"_target_": "fastwam.datasets.lerobot.transforms.image.ToTensor"},
            {"_target_": "torchvision.transforms.Resize", "size": [240, 320]},
        ]
        if result != expected:
            raise ProcessorContractError(
                f"{field} must be exactly ToTensor -> Resize([240,320])"
            )
        return result

    return {
        "schema": M1_ROBOTWIN_PROCESSOR_RECIPE,
        "version": M1_ROBOTWIN_PROCESSOR_RECIPE_VERSION,
        "num_frames": num_frames,
        "video_size": [384, 320],
        "concat_multi_camera": "robotwin",
        "shape_meta": {"images": images, "action": actions, "state": states},
        "processor": {
            "target": processor["_target_"],
            "num_obs_steps": num_frames,
            "num_output_cameras": 3,
            "action_output_dim": 14,
            "proprio_output_dim": 14,
            "delta_action_dim_mask": None,
            "action_state_transforms": None,
            "use_stepwise_action_norm": False,
            "norm_default_mode": "z-score",
            "norm_exception_mode": None,
            "action_state_merger": dict(merger),
            "train_transforms": transforms(
                processor["train_transforms"], "train_transforms"
            ),
            "val_transforms": transforms(
                processor["val_transforms"], "val_transforms"
            ),
        },
    }


def extract_m1_processor_recipe(
    config: Mapping[str, Any], *, profile: str = "libero"
) -> dict[str, Any]:
    if profile == "libero":
        return extract_m1_libero_processor_recipe(config)
    if profile == "robotwin":
        return extract_m1_robotwin_processor_recipe(config)
    raise ProcessorContractError("processor profile must be 'libero' or 'robotwin'")


def load_m1_data_config(
    path: str | Path, *, profile: str = "libero"
) -> tuple[dict[str, Any], str]:
    """Load the exact M1 YAML/JSON bytes and return its validated recipe/hash."""

    source = Path(path).expanduser().resolve()
    digest = sha256_file(source)
    try:
        raw_bytes = source.read_bytes()
        try:
            raw = json.loads(raw_bytes)
            value = {"data": raw}
        except (UnicodeError, json.JSONDecodeError):
            from omegaconf import OmegaConf

            raw = OmegaConf.load(source)
            wrapped = OmegaConf.create({"data": raw})
            OmegaConf.resolve(wrapped)
            value = OmegaConf.to_container(wrapped, resolve=True)
    except Exception as exc:
        raise ProcessorContractError(f"cannot resolve M1 data config {source}") from exc
    if sha256_file(source) != digest:
        raise ProcessorContractError("M1 data config changed while it was loaded")
    if not isinstance(value, Mapping):
        raise ProcessorContractError("M1 data config must resolve to a mapping")
    return extract_m1_processor_recipe(value, profile=profile), digest


def validate_processor_instance(processor: Any, recipe: Mapping[str, Any]) -> None:
    """Verify Hydra instantiated exactly the processor described by ``recipe``."""

    expected = _mapping(recipe, "processor recipe")
    if expected.get("schema") == M1_ROBOTWIN_PROCESSOR_RECIPE:
        _validate_robotwin_processor_instance(processor, expected)
        return
    if sha256_canonical_json(dict(expected)) != sha256_canonical_json(
        extract_m1_libero_processor_recipe(
            {
                "train": {
                    "num_frames": expected["num_frames"],
                    "video_size": expected["video_size"],
                    "concat_multi_camera": expected["concat_multi_camera"],
                    "shape_meta": expected["shape_meta"],
                    "processor": {
                        "_target_": expected["processor"]["target"],
                        "shape_meta": expected["shape_meta"],
                        "num_obs_steps": expected["processor"]["num_obs_steps"],
                        "num_output_cameras": expected["processor"]["num_output_cameras"],
                        "action_output_dim": expected["processor"]["action_output_dim"],
                        "proprio_output_dim": expected["processor"]["proprio_output_dim"],
                        "delta_action_dim_mask": expected["processor"]["delta_action_dim_mask"],
                        "action_state_transforms": None,
                        "use_stepwise_action_norm": False,
                        "norm_default_mode": "min/max",
                        "norm_exception_mode": None,
                        "action_state_merger": expected["processor"]["action_state_merger"],
                        "train_transforms": expected["processor"]["train_transforms"],
                        "val_transforms": expected["processor"]["val_transforms"],
                    },
                }
            }
        )
    ):
        raise ProcessorContractError("processor recipe is not canonical")
    if processor.__class__.__module__ + "." + processor.__class__.__name__ != (
        expected["processor"]["target"]
    ):
        raise ProcessorContractError("instantiated processor class differs from M1")
    checks = {
        "num_obs_steps": expected["processor"]["num_obs_steps"],
        "num_output_cameras": 2,
        "action_output_dim": 7,
        "proprio_output_dim": 8,
        "action_state_transforms": None,
        "use_stepwise_action_norm": False,
        "norm_default_mode": "min/max",
        "norm_exception_mode": None,
    }
    for field, wanted in checks.items():
        if getattr(processor, field, object()) != wanted:
            raise ProcessorContractError(f"instantiated processor {field} differs from M1")
    merger = processor.action_state_merger
    if merger.__class__.__module__ + "." + merger.__class__.__name__ != (
        expected["processor"]["action_state_merger"]["_target_"]
    ) or merger.action_target_dim is not None or merger.state_target_dim is not None:
        raise ProcessorContractError("instantiated action/state merger differs from M1")
    transforms = processor.val_transforms
    if isinstance(transforms, Mapping):
        raise ProcessorContractError("M1 validation transforms must be one shared list")
    if (
        not isinstance(transforms, Sequence)
        or isinstance(transforms, (str, bytes, bytearray))
        or len(transforms) != 2
    ):
        raise ProcessorContractError("instantiated validation transform count differs")
    transforms = list(transforms)
    targets = [item.__class__.__module__ + "." + item.__class__.__name__ for item in transforms]
    if targets != [
        "fastwam.datasets.lerobot.transforms.image.ToTensor",
        "torchvision.transforms.transforms.Resize",
    ]:
        raise ProcessorContractError("instantiated validation transforms differ from M1")
    if list(transforms[1].size) != [224, 224]:
        raise ProcessorContractError("instantiated Resize size differs from M1")


def _validate_robotwin_processor_instance(
    processor: Any, expected: Mapping[str, Any]
) -> None:
    canonical = extract_m1_robotwin_processor_recipe(
        {
            "train": {
                "num_frames": expected["num_frames"],
                "video_size": expected["video_size"],
                "concat_multi_camera": expected["concat_multi_camera"],
                "shape_meta": expected["shape_meta"],
                "processor": {
                    "_target_": expected["processor"]["target"],
                    "shape_meta": expected["shape_meta"],
                    "num_obs_steps": expected["processor"]["num_obs_steps"],
                    "num_output_cameras": 3,
                    "action_output_dim": 14,
                    "proprio_output_dim": 14,
                    "delta_action_dim_mask": None,
                    "action_state_transforms": None,
                    "use_stepwise_action_norm": False,
                    "norm_default_mode": "z-score",
                    "norm_exception_mode": None,
                    "action_state_merger": expected["processor"][
                        "action_state_merger"
                    ],
                    "train_transforms": expected["processor"]["train_transforms"],
                    "val_transforms": expected["processor"]["val_transforms"],
                },
            }
        }
    )
    if sha256_canonical_json(canonical) != sha256_canonical_json(dict(expected)):
        raise ProcessorContractError("RoboTwin processor recipe is not canonical")
    checks = {
        "num_obs_steps": 33,
        "num_output_cameras": 3,
        "action_output_dim": 14,
        "proprio_output_dim": 14,
        "action_state_transforms": None,
        "use_stepwise_action_norm": False,
        "norm_default_mode": "z-score",
        "norm_exception_mode": None,
    }
    actual_target = processor.__class__.__module__ + "." + processor.__class__.__name__
    if actual_target != expected["processor"]["target"]:
        raise ProcessorContractError("instantiated processor class differs from RoboTwin M1")
    for field, wanted in checks.items():
        if getattr(processor, field, object()) != wanted:
            raise ProcessorContractError(
                f"instantiated RoboTwin processor {field} differs from M1"
            )
    merger = processor.action_state_merger
    if merger.__class__.__module__ + "." + merger.__class__.__name__ != (
        expected["processor"]["action_state_merger"]["_target_"]
    ) or merger.action_target_dim is not None or merger.state_target_dim is not None:
        raise ProcessorContractError(
            "instantiated RoboTwin action/state merger differs from M1"
        )
    transforms = processor.val_transforms
    if isinstance(transforms, Mapping) or not isinstance(transforms, Sequence):
        raise ProcessorContractError("RoboTwin validation transforms must be one list")
    transforms = list(transforms)
    if len(transforms) != 2 or list(transforms[1].size) != [240, 320]:
        raise ProcessorContractError("RoboTwin validation Resize differs from M1")


__all__ = [
    "M1_LIBERO_PROCESSOR_RECIPE",
    "M1_LIBERO_PROCESSOR_RECIPE_VERSION",
    "M1_ROBOTWIN_PROCESSOR_RECIPE",
    "M1_ROBOTWIN_PROCESSOR_RECIPE_VERSION",
    "ProcessorContractError",
    "extract_m1_libero_processor_recipe",
    "extract_m1_processor_recipe",
    "extract_m1_robotwin_processor_recipe",
    "load_m1_data_config",
    "validate_processor_instance",
]
