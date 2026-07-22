"""Frozen data contract for the official RMBench demonstration release."""

from __future__ import annotations


OFFICIAL_RMBENCH_TASKS: tuple[str, ...] = (
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

OFFICIAL_TASK_CONFIG = "demo_clean"
OFFICIAL_EPISODES_PER_TASK = 50
OFFICIAL_CAMERA_KEYS: tuple[str, ...] = (
    "cam_high",
    "cam_left_wrist",
    "cam_right_wrist",
)
SOURCE_CAMERA_PATHS: dict[str, str] = {
    "cam_high": "/observation/head_camera/rgb",
    "cam_left_wrist": "/observation/left_camera/rgb",
    "cam_right_wrist": "/observation/right_camera/rgb",
}
SOURCE_QPOS_PATH = "/joint_action/vector"

ACTION_DIM = 14
DEFAULT_FPS = 15
# RMBench's official ``demo_clean.yml`` selects the ``LargeView`` profile,
# whose frozen task_config/_camera_config.yml contract is 320x240 (W x H).
DEFAULT_IMAGE_HEIGHT = 240
DEFAULT_IMAGE_WIDTH = 320
DEFAULT_DEV_PER_TASK = 5
DEFAULT_SPLIT_SEED = 42
LEROBOT_CODEBASE_VERSION = "v2.1"
LEROBOT_CHUNK_SIZE = 1000

DATA_PATH_TEMPLATE = (
    "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
)
VIDEO_PATH_TEMPLATE = (
    "videos/chunk-{episode_chunk:03d}/{video_key}/"
    "episode_{episode_index:06d}.mp4"
)

CONVERSION_SCHEMA = "warm.rmbench-to-lerobot"
CONVERSION_SCHEMA_VERSION = 2
CATALOG_FILENAME = "warm_episode_catalog.json"
MANIFEST_FILENAME = "rmbench_conversion_manifest.json"

MOTOR_NAMES: tuple[str, ...] = (
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_gripper",
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
    "right_gripper",
)


__all__ = [
    "ACTION_DIM",
    "CATALOG_FILENAME",
    "CONVERSION_SCHEMA",
    "CONVERSION_SCHEMA_VERSION",
    "DATA_PATH_TEMPLATE",
    "DEFAULT_DEV_PER_TASK",
    "DEFAULT_FPS",
    "DEFAULT_IMAGE_HEIGHT",
    "DEFAULT_IMAGE_WIDTH",
    "DEFAULT_SPLIT_SEED",
    "LEROBOT_CHUNK_SIZE",
    "LEROBOT_CODEBASE_VERSION",
    "MANIFEST_FILENAME",
    "MOTOR_NAMES",
    "OFFICIAL_CAMERA_KEYS",
    "OFFICIAL_EPISODES_PER_TASK",
    "OFFICIAL_RMBENCH_TASKS",
    "OFFICIAL_TASK_CONFIG",
    "SOURCE_CAMERA_PATHS",
    "SOURCE_QPOS_PATH",
    "VIDEO_PATH_TEMPLATE",
]
