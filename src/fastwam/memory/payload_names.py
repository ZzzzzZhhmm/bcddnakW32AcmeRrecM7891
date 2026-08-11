"""Canonical numeric payload names shared by WARM offline artifacts."""

from __future__ import annotations


MODEL_SPACE_ACTION = "model_space_action"
EFFECT_PRE = "effect_pre"
EFFECT_POST = "effect_post"
START_PROPRIO = "start_proprio"
OBSERVED_GRIPPER_STATE = "observed_gripper_state"
TASK_INDEX = "task_index"
EVENT_SCORE = "event_score"
CONTAINS_FORCED_GRIPPER = "contains_forced_gripper"
SOURCE_EPISODE_SHA256 = "source_episode_sha256"
FEATURE_EPISODE_SHA256 = "feature_episode_sha256"
NORMALIZED_PHASE = "normalized_phase"
EVENT_ORDINAL = "event_ordinal"
SUCCESSOR_ROW = "successor_row"
SUCCESSOR_EVENT_START_FRAME = "successor_event_start_frame"
ACTION_VALID_MASK = "action_valid_mask"


__all__ = [
    "ACTION_VALID_MASK",
    "CONTAINS_FORCED_GRIPPER",
    "EFFECT_POST",
    "EFFECT_PRE",
    "EVENT_ORDINAL",
    "EVENT_SCORE",
    "FEATURE_EPISODE_SHA256",
    "MODEL_SPACE_ACTION",
    "NORMALIZED_PHASE",
    "OBSERVED_GRIPPER_STATE",
    "SOURCE_EPISODE_SHA256",
    "START_PROPRIO",
    "SUCCESSOR_EVENT_START_FRAME",
    "SUCCESSOR_ROW",
    "TASK_INDEX",
]
