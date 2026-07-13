"""Strict RoboTwin/RMBench image-action artifact profile primitives.

The benchmark executes a 14-dimensional dual-arm qpos command.  Dimensions
6 and 13 are the two grippers; all other dimensions are arm joints.  FastWAM
trains in globally z-scored model space, so online factual action histories
must use the exact same affine map rather than re-normalizing rollouts ad hoc.

This module is NumPy-only and import-safe on a CPU development machine.  It
mirrors :class:`SingleFieldLinearNormalizer`'s z-score path, including its
``1e-8`` denominator regularizer and ``[-5, 5]`` forward clamp.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any, Mapping

import numpy as np

from .action_contract import ActionSpaceContract


ROBOTWIN_ACTION_DIM = 14
ROBOTWIN_GRIPPER_DIMS = (6, 13)
ROBOTWIN_ARM_DIMS = tuple(
    index for index in range(ROBOTWIN_ACTION_DIM) if index not in ROBOTWIN_GRIPPER_DIMS
)
ROBOTWIN_CONTROL_MODE = "robotwin_bimanual_qpos_plus_grippers"
ROBOTWIN_EMBODIMENT = "robotwin_aloha_agilex"
ROBOTWIN_NORMALIZATION_MODE = "global:z-score"
ROBOTWIN_ZSCORE_EPSILON = np.float32(1e-8)
ROBOTWIN_ZSCORE_CLIP = np.float32(5.0)


class RobotwinArtifactContractError(ValueError):
    """Raised when a RoboTwin artifact cannot prove the exact baseline layout."""


def _immutable_float32(value: Any, *, field: str) -> np.ndarray:
    try:
        if isinstance(value, np.ndarray):
            array = value
        else:
            try:
                array = value.detach().cpu().numpy()
            except (AttributeError, TypeError, RuntimeError):
                array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise RobotwinArtifactContractError(f"{field} must be numeric") from exc
    if not np.issubdtype(array.dtype, np.number):
        raise RobotwinArtifactContractError(f"{field} must be numeric")
    contiguous = np.ascontiguousarray(array, dtype=np.float32)
    if not np.isfinite(contiguous).all():
        raise RobotwinArtifactContractError(f"{field} must be finite")
    frozen = np.frombuffer(contiguous.tobytes(order="C"), dtype=np.float32)
    return frozen.reshape(contiguous.shape)


def _action_array(value: Any, *, field: str) -> np.ndarray:
    array = _immutable_float32(value, field=field)
    if array.ndim < 1 or array.shape[-1] != ROBOTWIN_ACTION_DIM:
        raise RobotwinArtifactContractError(
            f"{field} must end in the exact 14D RoboTwin qpos action dimension"
        )
    return array


@dataclass(frozen=True, slots=True)
class RobotwinQposZScore:
    """Exact global z-score affine map used by the RoboTwin FastWAM processor."""

    mean: np.ndarray
    std: np.ndarray
    epsilon: float = float(ROBOTWIN_ZSCORE_EPSILON)
    clip: float = float(ROBOTWIN_ZSCORE_CLIP)

    def __post_init__(self) -> None:
        mean = _immutable_float32(self.mean, field="mean")
        std = _immutable_float32(self.std, field="std")
        if mean.shape != (ROBOTWIN_ACTION_DIM,) or std.shape != (
            ROBOTWIN_ACTION_DIM,
        ):
            raise RobotwinArtifactContractError(
                "RoboTwin action mean/std must both have shape [14]"
            )
        if np.any(std < 0.0):
            raise RobotwinArtifactContractError("RoboTwin action std cannot be negative")
        if isinstance(self.epsilon, bool) or not isinstance(self.epsilon, Real):
            raise RobotwinArtifactContractError("epsilon must be a finite positive real")
        if isinstance(self.clip, bool) or not isinstance(self.clip, Real):
            raise RobotwinArtifactContractError("clip must be a finite positive real")
        epsilon = float(self.epsilon)
        clip = float(self.clip)
        if not np.isfinite(epsilon) or epsilon <= 0.0:
            raise RobotwinArtifactContractError("epsilon must be a finite positive real")
        if not np.isfinite(clip) or clip <= 0.0:
            raise RobotwinArtifactContractError("clip must be a finite positive real")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)
        object.__setattr__(self, "epsilon", epsilon)
        object.__setattr__(self, "clip", clip)

    @classmethod
    def from_dataset_stats(
        cls,
        dataset_stats: Mapping[str, Any],
        *,
        action_key: str = "default",
    ) -> "RobotwinQposZScore":
        """Read the non-stepwise action statistics emitted by FastWAM.

        The closed path is exactly ``action.<key>.global_mean/global_std``.
        Stepwise statistics and inferred fallback keys are intentionally not
        accepted because they define a different model-action space.
        """

        if not isinstance(dataset_stats, Mapping):
            raise RobotwinArtifactContractError("dataset_stats must be a mapping")
        action = dataset_stats.get("action")
        if not isinstance(action, Mapping):
            raise RobotwinArtifactContractError("dataset_stats.action must be a mapping")
        stats = action.get(action_key)
        if not isinstance(stats, Mapping):
            raise RobotwinArtifactContractError(
                f"dataset_stats.action[{action_key!r}] must be a mapping"
            )
        if "global_mean" not in stats or "global_std" not in stats:
            raise RobotwinArtifactContractError(
                "RoboTwin z-score stats require global_mean and global_std"
            )
        return cls(mean=stats["global_mean"], std=stats["global_std"])

    def normalize(self, qpos_actions: Any, *, fail_on_clip: bool = False) -> np.ndarray:
        """Map executed qpos actions to the exact FastWAM model space."""

        raw = _action_array(qpos_actions, field="qpos_actions")
        denominator = self.std + np.float32(self.epsilon)
        # Preserve SingleFieldLinearNormalizer's operation ordering rather
        # than relying only on the algebraically equivalent (x-mean)/std.
        scale = np.float32(1.0) / denominator
        offset = -self.mean / denominator
        unbounded = raw * scale + offset
        if fail_on_clip and np.any(np.abs(unbounded) > np.float32(self.clip)):
            raise RobotwinArtifactContractError(
                "qpos action exceeds the reversible [-5,5] model-space range"
            )
        normalized = np.clip(
            unbounded,
            -np.float32(self.clip),
            np.float32(self.clip),
        )
        return _immutable_float32(normalized, field="normalized_actions")

    def denormalize(self, model_actions: Any) -> np.ndarray:
        """Map model actions back to the simulator's 14D qpos command."""

        model = _action_array(model_actions, field="model_actions")
        denominator = self.std + np.float32(self.epsilon)
        scale = np.float32(1.0) / denominator
        offset = -self.mean / denominator
        raw = (model - offset) / scale
        return _immutable_float32(raw, field="denormalized_actions")

    def checked_round_trip(
        self,
        qpos_actions: Any,
        *,
        rtol: float = 1e-5,
        atol: float = 1e-6,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Normalize and invert one unclipped action tensor, failing on drift."""

        raw = _action_array(qpos_actions, field="qpos_actions")
        model = self.normalize(raw, fail_on_clip=True)
        restored = self.denormalize(model)
        if not np.allclose(restored, raw, rtol=rtol, atol=atol):
            raise RobotwinArtifactContractError(
                "RoboTwin qpos z-score round trip exceeded tolerance"
            )
        return model, restored


def robotwin_qpos_action_contract(
    *,
    normalization_stats_sha256: str,
    gripper_threshold: float = 0.0,
    embodiment: str = ROBOTWIN_EMBODIMENT,
) -> ActionSpaceContract:
    """Build the canonical WARM action-space contract for RoboTwin/RMBench."""

    return ActionSpaceContract(
        action_dim=ROBOTWIN_ACTION_DIM,
        arm_dims=ROBOTWIN_ARM_DIMS,
        gripper_dims=ROBOTWIN_GRIPPER_DIMS,
        gripper_threshold=gripper_threshold,
        normalization_mode=ROBOTWIN_NORMALIZATION_MODE,
        normalization_stats_sha256=normalization_stats_sha256,
        control_mode=ROBOTWIN_CONTROL_MODE,
        embodiment=embodiment,
    )


__all__ = [
    "ROBOTWIN_ACTION_DIM",
    "ROBOTWIN_ARM_DIMS",
    "ROBOTWIN_CONTROL_MODE",
    "ROBOTWIN_EMBODIMENT",
    "ROBOTWIN_GRIPPER_DIMS",
    "ROBOTWIN_NORMALIZATION_MODE",
    "RobotwinArtifactContractError",
    "RobotwinQposZScore",
    "robotwin_qpos_action_contract",
]
