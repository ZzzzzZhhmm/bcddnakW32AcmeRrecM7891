#!/usr/bin/env python3
"""Single-GPU Piper smoke: FastWAM backward on processed warm_data_config."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf

from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.runtime import run_training
from fastwam.utils.config_resolvers import register_default_resolvers

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "configs/real/piper_fastwam_smoke.local.yaml"
DEFAULT_PROCESSED = REPO / "real/piper/processed/pilot_v1"
CONTEXT_LEN = 128
TEXT_CACHE_SUFFIX = f"t5_len{CONTEXT_LEN}.wan22ti2v5b.pt"


def _resolve_against(path: str | Path, anchor: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = (anchor / candidate).resolve()
    return candidate


def _strip_compose_keys(cfg):
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    for key in ("defaults", "hydra"):
        if key in cfg:
            del cfg[key]
    return cfg


def _task_cache_path(cache_dir: Path, task: str) -> Path:
    prompt = DEFAULT_PROMPT.format(task=task)
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.{TEXT_CACHE_SUFFIX}"


def _load_tasks(dataset_dir: Path) -> list[str]:
    tasks_path = dataset_dir / "meta" / "tasks.jsonl"
    tasks: list[str] = []
    seen: set[str] = set()
    for line in tasks_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        task = str(json.loads(line)["task"])
        if task not in seen:
            seen.add(task)
            tasks.append(task)
    if not tasks:
        raise RuntimeError(f"no tasks found in {tasks_path}")
    return tasks


def ensure_text_embeds(dataset_dir: Path, cache_dir: Path) -> None:
    tasks = _load_tasks(dataset_dir)
    missing = [task for task in tasks if not _task_cache_path(cache_dir, task).is_file()]
    if not missing:
        print(f"text embeds already present for {len(tasks)} tasks in {cache_dir}")
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(REPO / "scripts/precompute_text_embeds.py"),
        "task=libero_uncond_2cam224_1e-4",
        f"data.train.dataset_dirs=[{dataset_dir}]",
        f"data.train.text_embedding_cache_dir={cache_dir}",
        f"data.train.context_len={CONTEXT_LEN}",
        "overwrite=false",
    ]
    print("encoding missing text embeds:", ", ".join(missing))
    subprocess.run(command, cwd=str(REPO), check=True)
    still_missing = [task for task in tasks if not _task_cache_path(cache_dir, task).is_file()]
    if still_missing:
        raise RuntimeError(f"text embed cache still missing: {still_missing}")


def build_cfg(config_path: Path):
    register_default_resolvers()
    smoke = OmegaConf.load(config_path)
    anchor = config_path.parent
    processed = _resolve_against(
        smoke.get("processed_root", DEFAULT_PROCESSED), anchor
    )
    cache_dir = _resolve_against(smoke.text_embedding_cache_dir, anchor)
    output_dir = _resolve_against(smoke.output_dir, anchor)
    data_path = processed / "memory" / "warm_data_config.yaml"
    if not data_path.is_file():
        raise FileNotFoundError(f"missing warm data config: {data_path}")

    data = OmegaConf.load(data_path)
    if not bool(smoke.get("attach_warm_candidates", True)):
        if "warm_candidates" in data:
            del data["warm_candidates"]
    data.train.text_embedding_cache_dir = str(cache_dir)
    if "val" in data:
        data.val.text_embedding_cache_dir = str(cache_dir)

    dataset_dir = Path(str(data.train.dataset_dirs[0])).expanduser().resolve()
    train_base = _strip_compose_keys(OmegaConf.load(REPO / "configs/train.yaml"))
    model = _strip_compose_keys(OmegaConf.load(REPO / "configs/model/fastwam.yaml"))
    model.mot_checkpoint_mixed_attn = bool(
        smoke.get("mot_checkpoint_mixed_attn", True)
    )
    model.action_dit_pretrained_path = str(
        (REPO / "checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt").resolve()
    )

    overrides = OmegaConf.create(
        {
            "output_dir": str(output_dir),
            "batch_size": int(smoke.batch_size),
            "num_workers": int(smoke.num_workers),
            "run_steps": None if smoke.get("run_steps") is None else int(smoke.run_steps),
            "max_steps": None if smoke.get("max_steps") is None else int(smoke.max_steps),
            "num_epochs": int(smoke.num_epochs),
            "eval_every": int(smoke.eval_every),
            "eval_num_samples": int(smoke.get("eval_num_samples", 0)),
            "save_every": int(smoke.save_every),
            "log_every": int(smoke.log_every),
            "learning_rate": float(smoke.learning_rate),
            "weight_decay": float(smoke.weight_decay),
            "gradient_accumulation_steps": int(smoke.gradient_accumulation_steps),
            "mixed_precision": str(smoke.mixed_precision),
            "seed": int(smoke.seed),
            "allow_unattested_warm_checkpoints": bool(
                smoke.get("allow_unattested_warm_checkpoints", True)
            ),
            "wandb": {"enabled": False},
        }
    )
    cfg = OmegaConf.merge(train_base, {"model": model, "data": data}, overrides)
    OmegaConf.resolve(cfg)
    return cfg, dataset_dir, cache_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="local smoke yaml under configs/real",
    )
    parser.add_argument(
        "--skip-text-embeds",
        action="store_true",
        help="do not encode T5 caches; fail if they are missing",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Piper-shaped FastWAM .pt (after 8D-to-7D migration); loaded as resume weights",
    )
    parser.add_argument("--run-steps", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    config_path = args.config.expanduser().resolve()
    cfg, dataset_dir, cache_dir = build_cfg(config_path)
    if args.run_steps is not None:
        cfg.run_steps = int(args.run_steps)
    if args.num_epochs is not None:
        cfg.num_epochs = int(args.num_epochs)
    if args.save_every is not None:
        cfg.save_every = int(args.save_every)
    if args.eval_every is not None:
        cfg.eval_every = int(args.eval_every)
    if args.output_dir is not None:
        cfg.output_dir = str(args.output_dir.expanduser().resolve())
    if args.init_checkpoint is not None:
        init_path = args.init_checkpoint.expanduser().resolve()
        if not init_path.is_file():
            raise FileNotFoundError(f"init checkpoint not found: {init_path}")
        cfg.resume = str(init_path)
    if args.skip_text_embeds:
        tasks = _load_tasks(dataset_dir)
        missing = [
            task for task in tasks if not _task_cache_path(cache_dir, task).is_file()
        ]
        if missing:
            raise FileNotFoundError(f"missing text embeds: {missing}")
    else:
        ensure_text_embeds(dataset_dir, cache_dir)
    print(f"fastwam output: {cfg.output_dir}")
    print(
        f"num_epochs={cfg.num_epochs} run_steps={cfg.run_steps} "
        f"batch_size={cfg.batch_size} save_every={cfg.save_every} "
        f"eval_every={cfg.eval_every} eval_num_samples={cfg.eval_num_samples} "
        f"init_checkpoint={cfg.get('resume')}"
    )
    run_training(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
