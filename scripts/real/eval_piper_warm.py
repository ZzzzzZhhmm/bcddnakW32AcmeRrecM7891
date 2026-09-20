#!/usr/bin/env python3
"""Holdout loss-only eval of a Piper complete-WARM checkpoint.

Loads the Phase-1 FastWAM base for contract hashing, then restores the trained
WARM weights.  Formal WARM evaluation is loss-only (no closed-loop infer).
This is not a real-robot success metric.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from hydra.utils import instantiate

from fastwam.datasets.warm_candidates import (
    WARM_CANDIDATE_MASK,
    WARM_CANDIDATE_MU,
    WARM_ORACLE_CANDIDATE_INDEX,
)
from fastwam.runtime import (
    _mixed_precision_to_model_dtype,
    _normalize_mixed_precision,
    _publish_resolved_training_config,
    _resolve_train_device,
    build_datasets,
)
from fastwam.trainer import Wan22Trainer
from fastwam.training_config import validate_training_config
from fastwam.utils import misc
from fastwam.utils.logging_config import setup_logging
from fastwam.utils.pytorch_utils import set_global_seed

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_piper_warm as piper_train  # noqa: E402

REPO = piper_train.REPO
DEFAULT_CHECKPOINT = (
    REPO
    / "real/piper/processed/pilot_v1/warm_from_piper_fastwam"
    / "run_20260920_040507/checkpoints/weights/step_000400.pt"
)
logger = logging.getLogger("eval_piper_warm")


def _jsonable(value):
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _retrieval_diagnostics(dataset, *, max_samples: int | None = None) -> dict:
    """Normalized GT-action vs retrieved candidate action, holdout only."""

    n = len(dataset) if max_samples is None else min(int(max_samples), len(dataset))
    oracle_sq: list[float] = []
    top1_sq: list[float] = []
    skipped = 0
    for index in range(n):
        sample = dataset[index]
        action = sample["action"]
        mask = sample[WARM_CANDIDATE_MASK]
        mu = sample[WARM_CANDIDATE_MU]
        oracle = int(sample[WARM_ORACLE_CANDIDATE_INDEX].item())
        valid = torch.nonzero(mask, as_tuple=False).flatten()
        if valid.numel() == 0:
            skipped += 1
            continue
        top1 = int(valid[0].item())
        top1_sq.append(float((mu[top1] - action).square().mean().item()))
        if oracle < 0 or not bool(mask[oracle].item()):
            skipped += 1
            continue
        oracle_sq.append(float((mu[oracle] - action).square().mean().item()))
    def _mean(values: list[float]) -> float | None:
        if not values:
            return None
        return float(sum(values) / len(values))

    return {
        "num_windows": n,
        "skipped_no_oracle": skipped,
        "normalized_oracle_action_mse": _mean(oracle_sq),
        "normalized_top1_action_mse": _mean(top1_sq),
        "oracle_count": len(oracle_sq),
        "top1_count": len(top1_sq),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=piper_train.DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--eval-num-samples",
        type=int,
        default=0,
        help="0 = every DEV window; positive = stratified subset",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args(argv)

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"WARM checkpoint not found: {checkpoint}")

    cfg, dataset_dir, cache_dir = piper_train.build_cfg(
        args.config.expanduser().resolve(),
        overwrite_contracts=False,
    )
    if args.output_dir is not None:
        cfg.output_dir = str(args.output_dir.expanduser().resolve())
    else:
        stamp = checkpoint.stem
        cfg.output_dir = str(
            (checkpoint.parents[3] / f"eval_{stamp}").resolve()
        )
    cfg.num_workers = int(args.num_workers)
    cfg.eval_every = 1
    cfg.eval_num_samples = int(args.eval_num_samples)
    cfg.run_steps = 1
    cfg.save_every = 0
    cfg.wandb = {"enabled": False}
    output_dir = Path(str(cfg.output_dir)).expanduser().resolve()
    cfg.output_dir = str(output_dir)
    piper_train._assert_fresh_output_dir(output_dir)
    piper_train.ensure_text_embeds(dataset_dir, cache_dir)

    validate_training_config(cfg)
    setup_logging(log_level=logging.INFO, is_main_process=True)
    misc.register_work_dir(str(output_dir))
    resolved_config_path = _publish_resolved_training_config(cfg)
    set_global_seed(int(cfg.seed), rank_offset=False)
    model_device = _resolve_train_device()
    mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)

    print(f"eval output: {output_dir}")
    print(f"checkpoint: {checkpoint}")
    print(f"eval_num_samples={cfg.eval_num_samples} device={model_device}")
    print(f"resolved config: {resolved_config_path}")

    train_ds, val_ds = build_datasets(cfg.data, build_validation=True)
    # Avoid a second full video scan before GPU eval.  32 windows is enough to
    # see whether holdout retrieval is in the same numeric ballpark as GT.
    retrieval_n = (
        len(val_ds)
        if int(cfg.eval_num_samples) == 0
        else int(cfg.eval_num_samples)
    )
    retrieval = _retrieval_diagnostics(val_ds, max_samples=min(32, retrieval_n))
    print(
        "holdout retrieval (first "
        f"{retrieval['num_windows']} windows): "
        f"oracle_mse={retrieval['normalized_oracle_action_mse']} "
        f"top1_mse={retrieval['normalized_top1_action_mse']}"
    )

    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    print(f"loading WARM checkpoint {checkpoint}")
    model.load_checkpoint(str(checkpoint))
    model.validate_training_dataset(train_ds)
    model.validate_validation_dataset(val_ds)

    trainer = Wan22Trainer.create(
        cfg=cfg,
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
    )
    try:
        metrics = trainer.evaluate()
    finally:
        trainer.close()
    if not isinstance(metrics, dict):
        raise RuntimeError("WARM evaluate() returned no metrics")

    payload = {
        "schema": "warm.real.piper-holdout-eval.v1",
        "checkpoint": str(checkpoint),
        "output_dir": str(output_dir),
        "eval_num_samples": int(cfg.eval_num_samples),
        "val_dataset_size": int(len(val_ds)),
        "retrieval": retrieval,
        "metrics": {str(key): _jsonable(value) for key, value in sorted(metrics.items())},
    }
    report_path = output_dir / "eval_metrics.json"
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {report_path}")
    print(
        f"val_loss={metrics.get('val_loss')} "
        f"val_num_samples={metrics.get('val_num_samples')} "
        f"mode={metrics.get('evaluation_mode')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
