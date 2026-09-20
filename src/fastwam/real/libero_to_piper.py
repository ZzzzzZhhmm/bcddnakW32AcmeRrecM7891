"""Migrate a LIBERO FastWAM checkpoint onto the Piper 7D proprio interface.

This does not pad Piper state to 8D.  It shrinks the LIBERO input projection
so a 7D Piper processor can load the released FastWAM weights.  Pose columns
are copied as an initialization only; Panda EE and Piper TCP frames remain
unverified.  The two LIBERO gripper channels are averaged into one width.
"""
from __future__ import annotations

from typing import Any, Mapping

LIBERO_PROPRIO_DIM = 8
PIPER_PROPRIO_DIM = 7
LIBERO_POSE_DIM = 6
MIGRATION_SCHEMA = "warm.real.libero-to-piper.v1"


class LiberoToPiperMigrationError(ValueError):
    """Raised when a LIBERO FastWAM checkpoint cannot be projected to Piper."""


def migrate_proprio_encoder_state(
    state: Mapping[str, Any],
    *,
    expected_out_dim: int | None = None,
) -> dict[str, Any]:
    """Return a 7D proprio_encoder state_dict from a LIBERO 8D state_dict."""

    if "weight" not in state or "bias" not in state:
        raise LiberoToPiperMigrationError(
            "proprio_encoder state must contain weight and bias"
        )
    weight = state["weight"]
    bias = state["bias"]
    if getattr(weight, "ndim", None) != 2:
        raise LiberoToPiperMigrationError(
            f"proprio_encoder.weight must be 2D, got shape {tuple(getattr(weight, 'shape', ()))}"
        )
    out_dim, in_dim = tuple(int(size) for size in weight.shape)
    if in_dim != LIBERO_PROPRIO_DIM:
        raise LiberoToPiperMigrationError(
            "LIBERO proprio_encoder input dim must be "
            f"{LIBERO_PROPRIO_DIM}, got {in_dim}"
        )
    if expected_out_dim is not None and out_dim != int(expected_out_dim):
        raise LiberoToPiperMigrationError(
            f"proprio_encoder output dim must be {expected_out_dim}, got {out_dim}"
        )
    if tuple(int(size) for size in bias.shape) != (out_dim,):
        raise LiberoToPiperMigrationError(
            f"proprio_encoder.bias must have shape {(out_dim,)}, got {tuple(bias.shape)}"
        )
    migrated_weight = weight.new_empty((out_dim, PIPER_PROPRIO_DIM))
    migrated_weight[:, :LIBERO_POSE_DIM] = weight[:, :LIBERO_POSE_DIM]
    migrated_weight[:, LIBERO_POSE_DIM] = 0.5 * (
        weight[:, LIBERO_POSE_DIM] + weight[:, LIBERO_POSE_DIM + 1]
    )
    return {"weight": migrated_weight, "bias": bias.clone()}


def migrate_fastwam_checkpoint_payload(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Rewrite one FastWAM payload in place of the 8D proprio head."""

    if not isinstance(payload, Mapping) or "mot" not in payload:
        raise LiberoToPiperMigrationError(
            "LIBERO FastWAM checkpoint must be a mapping with a mot state"
        )
    if "warm_source" in payload:
        raise LiberoToPiperMigrationError(
            "refuse to migrate a WARM checkpoint as a FastWAM base; "
            "use the official LIBERO FastWAM weights and let WARM heads "
            "initialize on the Piper 770-D context"
        )
    proprio = payload.get("proprio_encoder")
    if not isinstance(proprio, Mapping):
        raise LiberoToPiperMigrationError(
            "LIBERO FastWAM checkpoint is missing proprio_encoder"
        )
    migrated = dict(payload)
    migrated["proprio_encoder"] = migrate_proprio_encoder_state(proprio)
    report = {
        "schema": MIGRATION_SCHEMA,
        "source_proprio_in": LIBERO_PROPRIO_DIM,
        "target_proprio_in": PIPER_PROPRIO_DIM,
        "pose_columns": list(range(LIBERO_POSE_DIM)),
        "gripper_rule": "mean(libero_gripper_channels_6_7)",
        "padded_piper_to_8d": False,
        "warm_heads_copied": False,
    }
    return migrated, report
