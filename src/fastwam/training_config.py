"""Side-effect-free validation for the distributed training entry point."""

from __future__ import annotations

import math
from typing import Any


def validate_training_config(cfg: Any) -> None:
    """Validate optimizer-loop inputs before allocating the 6B model.

    Hydra resolves most scalar overrides, but it does not enforce the semantic
    constraints required by the trainer.  This module intentionally imports no
    CUDA, PyTorch, or Accelerate code so launch preflight can remain cheap.
    """

    def require_int(name: str, *, minimum: int, default: Any = None) -> int:
        value = cfg.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer, got {value!r}")
        resolved = int(value)
        if resolved < minimum:
            raise ValueError(f"{name} must be >= {minimum}, got {resolved}")
        return resolved

    def require_finite(name: str, *, minimum: float, inclusive: bool) -> float:
        value = cfg.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a finite number, got {value!r}")
        resolved = float(value)
        valid_bound = resolved >= minimum if inclusive else resolved > minimum
        if not math.isfinite(resolved) or not valid_bound:
            relation = ">=" if inclusive else ">"
            raise ValueError(
                f"{name} must be finite and {relation} {minimum}, got {resolved}"
            )
        return resolved

    require_int("batch_size", minimum=1)
    require_int("num_workers", minimum=0)
    require_int("num_epochs", minimum=1)
    require_int("gradient_accumulation_steps", minimum=1)
    require_int("log_every", minimum=0)
    require_int("save_every", minimum=0)
    require_int("eval_every", minimum=0)
    require_int("eval_num_inference_steps", minimum=1)
    # Zero means a deterministic full-DEV pass.  Positive values request a
    # deterministic task/event-stratified subset.
    require_int("eval_num_samples", minimum=0, default=1)
    require_int("max_nonfinite_gradient_skips", minimum=1, default=3)
    seed = require_int("seed", minimum=1)
    if seed >= 2**32 - 1:
        raise ValueError("seed must be smaller than uint32 max")

    for optional_name in ("max_steps", "run_steps"):
        value = cfg.get(optional_name)
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"{optional_name} must be a positive integer or null, got {value!r}"
                )
            if int(value) <= 0:
                raise ValueError(
                    f"{optional_name} must be a positive integer or null, got {value!r}"
                )

    require_finite("learning_rate", minimum=0.0, inclusive=False)
    require_finite("weight_decay", minimum=0.0, inclusive=True)
    require_finite("max_grad_norm", minimum=0.0, inclusive=False)

    precision = cfg.get("mixed_precision")
    if not isinstance(precision, str) or precision.strip().lower() not in {
        "no",
        "fp16",
        "bf16",
    }:
        raise ValueError(
            "mixed_precision must be one of ['no', 'fp16', 'bf16'], "
            f"got {precision!r}"
        )
    scheduler = cfg.get("lr_scheduler_type")
    if not isinstance(scheduler, str) or scheduler.strip().lower() not in {
        "cosine",
        "constant",
    }:
        raise ValueError(
            "lr_scheduler_type must be one of ['cosine', 'constant'], "
            f"got {scheduler!r}"
        )

    wandb_cfg = cfg.get("wandb")
    wandb_enabled = None if wandb_cfg is None else wandb_cfg.get("enabled")
    if not isinstance(wandb_enabled, bool):
        raise TypeError(
            f"wandb.enabled must resolve to a boolean, got {wandb_enabled!r}"
        )
    allow_unattested = cfg.get("allow_unattested_warm_checkpoints", False)
    if not isinstance(allow_unattested, bool):
        raise TypeError(
            "allow_unattested_warm_checkpoints must resolve to a boolean"
        )
    initialization_checkpoint = cfg.get("initialization_checkpoint")
    initialization_manifest = cfg.get("initialization_fork_manifest")
    has_initialization_checkpoint = initialization_checkpoint not in (None, "", False)
    has_initialization_manifest = initialization_manifest not in (None, "", False)
    if has_initialization_checkpoint != has_initialization_manifest:
        raise ValueError(
            "initialization_checkpoint and initialization_fork_manifest must "
            "be either both null or both populated"
        )
    if has_initialization_checkpoint and cfg.get("resume") not in (None, "", False):
        raise ValueError("weights-only initialization is mutually exclusive with resume")
    stage = cfg.get("rmbench_training_stage")
    if stage not in (None, "shared", "specialist"):
        raise ValueError("rmbench_training_stage must be null, shared, or specialist")
    if stage == "shared" and has_initialization_checkpoint:
        raise ValueError("shared RMBench training cannot use specialist initialization")
    allow_direct_base = cfg.get("allow_direct_base_specialist", False)
    if not isinstance(allow_direct_base, bool):
        raise TypeError("allow_direct_base_specialist must be a boolean")
    if stage == "specialist" and not has_initialization_checkpoint and not allow_direct_base:
        raise ValueError(
            "formal RMBench specialist training requires an attested shared-WARM fork"
        )
    if allow_direct_base and stage != "specialist":
        raise ValueError(
            "allow_direct_base_specialist is valid only for RMBench specialist training"
        )

    sampler = cfg.get("sampler", {})
    if sampler is None or not hasattr(sampler, "get"):
        raise TypeError("sampler must be a mapping")
    sampler_mode = sampler.get("mode", "random")
    if sampler_mode not in {"random", "rmbench_task_event_balanced"}:
        raise ValueError(
            "sampler.mode must be random or rmbench_task_event_balanced"
        )
    event_boost = sampler.get("event_boost", 1.5)
    if (
        isinstance(event_boost, bool)
        or not isinstance(event_boost, (int, float))
        or not math.isfinite(float(event_boost))
        or not 1.0 <= float(event_boost) <= 4.0
    ):
        raise ValueError("sampler.event_boost must be finite and lie in [1, 4]")

    model_cfg = cfg.get("model")
    if model_cfg is not None:
        mot_checkpoint = model_cfg.get("mot_checkpoint_mixed_attn")
        if mot_checkpoint is not None and not isinstance(mot_checkpoint, bool):
            raise TypeError(
                "model.mot_checkpoint_mixed_attn must resolve to a boolean, "
                f"got {mot_checkpoint!r}"
            )
        source_policy = model_cfg.get("source_policy")
        if source_policy is not None and source_policy not in {
            "fixed_context_top1",
            "gaussian_null",
            "oracle_action_top1",
        }:
            raise ValueError(
                f"unsupported model.source_policy {source_policy!r}"
            )
        target = str(model_cfg.get("_target_", ""))
        if target.endswith("create_warm_retrospection"):
            if source_policy != "fixed_context_top1":
                raise ValueError(
                    "complete WARM requires "
                    "model.source_policy=fixed_context_top1"
                )
            if mot_checkpoint is not True:
                raise ValueError(
                    "complete WARM requires "
                    "model.mot_checkpoint_mixed_attn=true"
                )


__all__ = ["validate_training_config"]
