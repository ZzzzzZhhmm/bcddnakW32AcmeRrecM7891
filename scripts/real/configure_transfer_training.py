#!/usr/bin/env python3
"""Compose a portable plain FastWAM training config from prepared Piper data.

This does not load a model, connect hardware, or start training. WARM-specific
memory heads use train_piper_warm.py instead. Run from the reference repo root.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import _bootstrap  # noqa: F401
from omegaconf import OmegaConf

from fastwam.preprocessing.contracts import file_sha256, read_json
from fastwam.real.preprocessing.contract import PIPER_JOINT_CONTROL_MODE
from fastwam.utils.config_resolvers import register_default_resolvers

REPO = Path(__file__).resolve().parents[2]


def compose(*, processed: Path, run_dir: Path, text_cache: Path, model_config: Path,
            checkpoint: Path, resume_state: bool = False, epochs: int = 20,
            batch_size: int = 1, accumulation: int = 1, learning_rate: float = 1e-4,
            save_every: int = 400, eval_every: int = 1000, run_steps: int | None = None):
    if min(epochs, batch_size, accumulation, save_every, eval_every) <= 0 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Epochs, batch sizes, intervals and learning rate must be positive")
    if run_steps is not None and run_steps <= 0:
        raise ValueError("run_steps must be positive")
    prepared = processed.resolve() / "prepared"
    marker = read_json(prepared / "COMPLETE.json")
    if marker.get("synthetic"):
        raise ValueError("Synthetic prepared data must not be used as a real training release")
    for name, digest in marker["files"].items():
        path = (prepared / name).resolve()
        if not path.is_relative_to(prepared) or file_sha256(path) != digest:
            raise ValueError(f"Prepared artifact missing or changed: {name}")
    profile = marker["profile"]
    if (profile["action_dim"], profile["state_dim"], profile["control_mode"]) != (7, 7, PIPER_JOINT_CONTROL_MODE):
        raise ValueError("This transfer recipe requires absolute joint targets, TCP proprio, both 7D")
    if profile["camera_keys"] != ["external", "wrist"] or profile["action_horizon"] != 32 or profile["action_video_freq_ratio"] != 4 or profile['fps'] != 20:
        raise ValueError("This recipe requires external/wrist, 20 Hz, H32 and action/video ratio 4")
    checkpoint = checkpoint.expanduser().resolve()
    if not (checkpoint.is_dir() if resume_state else checkpoint.is_file()):
        raise FileNotFoundError(checkpoint)
    if resume_state and not (checkpoint / "trainer_state.json").is_file():
        raise ValueError("Resume needs a full state directory with trainer_state.json")
    if run_dir.exists() and any(run_dir.iterdir()) and not resume_state:
        raise FileExistsError("Use a fresh run directory for weight-only initialization")
    register_default_resolvers()
    base = OmegaConf.load(REPO / "configs/train.yaml")
    for key in ("defaults", "hydra"):
        base.pop(key, None)
    model = OmegaConf.load(model_config)
    if "defaults" in model:
        raise ValueError("Pass a standalone model config with inherited defaults already composed")
    data = OmegaConf.load(prepared / "data_config.yaml")
    if "train" not in data or "val" not in data:
        raise ValueError("Provide explicit train and dev episodes before preparing training")
    if "warm_candidates" in data:
        raise ValueError("Plain FastWAM must use prepared/data_config.yaml without WARM candidates")
    for split in ("train", "val"):
        data[split].text_embedding_cache_dir = str(text_cache.resolve())
    cfg = OmegaConf.merge(base, {"model": model, "data": data}, {
        "output_dir": str(run_dir.resolve()), "resume": str(checkpoint),
        "num_epochs": epochs, "batch_size": batch_size,
        "gradient_accumulation_steps": accumulation, "learning_rate": learning_rate,
        "weight_decay": 1e-2, "num_workers": 0, "mixed_precision": "bf16",
        "save_every": save_every, "eval_every": eval_every, "eval_num_samples": 0,
        "log_every": 20, "run_steps": run_steps, "max_steps": None,
        "wandb": {"enabled": False},
        "hydra": {"job": {"chdir": False}, "run": {"dir": "."}, "output_subdir": None},
    })
    OmegaConf.resolve(cfg)
    return cfg


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--processed-root", required=True, type=Path)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--text-cache", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--model-config", type=Path, default=REPO / "configs/model/fastwam.yaml")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--init-checkpoint", type=Path)
    g.add_argument("--resume-state", type=Path)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--accumulation", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--save-every", type=int, default=400)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--run-steps", type=int)
    a = p.parse_args(argv)
    cfg = compose(processed=a.processed_root, run_dir=a.run_dir, text_cache=a.text_cache,
                  model_config=a.model_config, checkpoint=a.init_checkpoint or a.resume_state,
                  resume_state=a.resume_state is not None, epochs=a.epochs,
                  batch_size=a.batch_size, accumulation=a.accumulation,
                  learning_rate=a.learning_rate, save_every=a.save_every,
                  eval_every=a.eval_every, run_steps=a.run_steps)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
    print(json.dumps({"config": str(a.output.resolve()), "model_factory": cfg.model._target_,
                      "data_config": str(a.processed_root.resolve() / "prepared/data_config.yaml"),
                      "checkpoint": cfg.resume, "mode": "full-state resume" if a.resume_state else "weights only",
                      "model_loaded": False, "training_started": False}, indent=2))


if __name__ == "__main__":
    main()
