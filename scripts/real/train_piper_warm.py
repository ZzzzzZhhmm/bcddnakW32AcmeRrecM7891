#!/usr/bin/env python3
"""Single-GPU Piper WARM smoke on processed bank/candidates and a 7D FastWAM base."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf

from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.runtime import run_training
from fastwam.utils.config_resolvers import register_default_resolvers

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "configs/real/piper_warm_smoke.local.yaml"
CONTEXT_LEN = 128
TEXT_CACHE_SUFFIX = f"t5_len{CONTEXT_LEN}.wan22ti2v5b.pt"
PIPER_CONTEXT_DIM = 770


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
    tasks: list[str] = []
    seen: set[str] = set()
    for line in (dataset_dir / "meta" / "tasks.jsonl").read_text(
        encoding="utf-8"
    ).splitlines():
        if not line.strip():
            continue
        task = str(json.loads(line)["task"])
        if task not in seen:
            seen.add(task)
            tasks.append(task)
    if not tasks:
        raise RuntimeError(f"no tasks found under {dataset_dir}")
    return tasks


_DIST_ENV_KEYS = (
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "GROUP_RANK",
    "ROLE_RANK",
    "ROLE_NAME",
    "GROUP_WORLD_SIZE",
    "ROLE_WORLD_SIZE",
    "TORCHELASTIC_RUN_ID",
    "ACCELERATE_USE_DEEPSPEED",
)


def _is_launch_main_process() -> bool:
    for key in ("RANK", "SLURM_PROCID", "LOCAL_RANK"):
        if key in os.environ:
            return os.environ.get(key, "0").strip() in {"", "0"}
    return True


def _isolated_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in _DIST_ENV_KEYS:
        env.pop(key, None)
    return env


def _drop_dead_artifact_claim(lock_path: Path) -> None:
    """Remove a contract lock only when its recorded pid is gone.

    Artifact claims are not stolen from a live owner.  A crashed 4-rank
    launch can leave `.warm-artifact.lock` behind; that lock must not block
    the next rank-0 rebuild.
    """

    if not lock_path.is_file():
        return
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        pid = int(payload.get("pid", -1))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        print(f"WARNING: leaving unreadable artifact claim in place: {lock_path}")
        return
    if pid <= 0:
        return
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        print(f"removing dead artifact claim pid={pid} path={lock_path}")
        lock_path.unlink(missing_ok=True)
    except PermissionError:
        return
    except OSError:
        return


def _wait_for_source_contracts(
    outputs: tuple[Path, Path],
    ready: Path,
    expected: str,
    *,
    timeout_s: float = 1800,
) -> None:
    deadline = time.time() + timeout_s
    while True:
        if ready.is_file() and all(path.is_file() for path in outputs):
            if ready.read_text(encoding="utf-8").strip() == expected:
                return
        if time.time() >= deadline:
            raise RuntimeError(
                f"timed out waiting for source-run contracts {outputs[0]} "
                f"{outputs[1]} ready={ready}"
            )
        time.sleep(2)


def ensure_text_embeds(dataset_dir: Path, cache_dir: Path) -> None:
    tasks = _load_tasks(dataset_dir)
    missing = [task for task in tasks if not _task_cache_path(cache_dir, task).is_file()]
    if not missing:
        print(f"text embeds already present for {len(tasks)} tasks in {cache_dir}")
        return
    if _is_launch_main_process():
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
        subprocess.run(
            command, cwd=str(REPO), check=True, env=_isolated_subprocess_env()
        )
    deadline = time.time() + 1800
    while True:
        still_missing = [
            task for task in tasks if not _task_cache_path(cache_dir, task).is_file()
        ]
        if not still_missing:
            return
        if time.time() >= deadline:
            raise RuntimeError(f"text embed cache still missing: {still_missing}")
        time.sleep(2)


def build_contracts(
    *,
    processed: Path,
    base_checkpoint: Path,
    contract_dir: Path,
    overwrite: bool,
) -> tuple[Path, Path]:
    bindings = json.loads(
        (processed / "memory" / "training_bindings.json").read_text(encoding="utf-8")
    )
    bank = Path(bindings["event_bank"])
    contract_dir.mkdir(parents=True, exist_ok=True)
    train_contract = contract_dir / "train_source.json"
    dev_contract = contract_dir / "dev_source.json"
    outputs = (train_contract, dev_contract)
    expected = str(Path(base_checkpoint).expanduser().resolve())
    ready = contract_dir / ".source_contracts.ready"
    if not _is_launch_main_process():
        print(
            "waiting for rank 0 to publish source-run contracts: "
            f"{train_contract} {dev_contract}"
        )
        _wait_for_source_contracts(outputs, ready, expected)
        return train_contract, dev_contract

    if overwrite and ready.is_file():
        ready.unlink()
    for split, cache, output in (
        ("train", Path(bindings["splits"]["train"]["candidates"]), train_contract),
        ("dev", Path(bindings["splits"]["dev"]["candidates"]), dev_contract),
    ):
        _drop_dead_artifact_claim(
            output.parent / f".{output.name}.warm-artifact.lock"
        )
        if output.is_file() and not overwrite:
            print(f"reuse {split} contract: {output}")
            continue
        command = [
            sys.executable,
            str(REPO / "scripts/build_warm_source_run_contract.py"),
            "--bank",
            str(bank),
            "--candidate-cache",
            str(cache),
            "--base-checkpoint",
            expected,
            "--output",
            str(output),
            "--query-split",
            split,
            "--expected-action-horizon",
            "32",
            "--expected-action-dim",
            "7",
            "--expected-query-corpus-sha256",
            str(bindings["splits"][split]["query_corpus_sha256"]),
        ]
        if overwrite:
            command.append("--overwrite")
        print(f"building {split} source-run contract")
        subprocess.run(
            command, cwd=str(REPO), check=True, env=_isolated_subprocess_env()
        )
    ready.write_text(expected + "\n", encoding="utf-8")
    return train_contract, dev_contract


def _load_warm_model_cfg(context_dim: int):
    fastwam = _strip_compose_keys(OmegaConf.load(REPO / "configs/model/fastwam.yaml"))
    source = _strip_compose_keys(OmegaConf.load(REPO / "configs/model/warm_source.yaml"))
    warm = _strip_compose_keys(OmegaConf.load(REPO / "configs/model/warm.yaml"))
    model = OmegaConf.merge(fastwam, source, warm)
    model.retrospection.context_dim = int(context_dim)
    model.mot_checkpoint_mixed_attn = True
    model.skip_dit_load_from_pretrain = True
    model.action_dit_pretrained_path = None
    model.source_policy = "fixed_context_top1"
    return model


def _assert_fresh_output_dir(output_dir: Path) -> None:
    """Refuse a leftover failed launch that already published config.yaml.

    Runtime publishes config.yaml immutably.  A previous crash leaves that
    file plus empty checkpoint directories, so a retry with any config change
    dies before training.  Completed or in-progress runs with metrics/weights
    are also left untouched.

    Only the launch main process may run this check.  Rank 0 writes
    config.yaml during ``run_training`` while other ranks can still be in
    ``main()``; those ranks must not treat the in-flight file as a leftover.
    """

    if not _is_launch_main_process():
        return
    config_path = output_dir / "config.yaml"
    if not config_path.is_file():
        return
    metrics = output_dir / "training_metrics.jsonl"
    weights_dir = output_dir / "checkpoints" / "weights"
    has_weights = weights_dir.is_dir() and any(weights_dir.glob("*.pt"))
    if metrics.is_file() or has_weights:
        raise RuntimeError(
            f"output dir already has a WARM run: {output_dir}. "
            "Pass a new --output-dir instead of overwriting."
        )
    raise RuntimeError(
        f"refusing leftover failed run at {output_dir} "
        "(config.yaml exists, but no training_metrics.jsonl or weights). "
        "Choose a new --output-dir."
    )


def build_cfg(
    config_path: Path,
    *,
    overwrite_contracts: bool,
    base_checkpoint: Path | None = None,
):
    register_default_resolvers()
    smoke = OmegaConf.load(config_path)
    anchor = config_path.parent
    processed = _resolve_against(smoke.processed_root, anchor)
    cache_dir = _resolve_against(smoke.text_embedding_cache_dir, anchor)
    output_dir = _resolve_against(smoke.output_dir, anchor)
    if base_checkpoint is None:
        base_checkpoint = _resolve_against(smoke.base_checkpoint, anchor)
    else:
        base_checkpoint = base_checkpoint.expanduser().resolve()
    contract_dir = _resolve_against(smoke.contract_dir, anchor)
    if not base_checkpoint.is_file():
        raise FileNotFoundError(f"Piper FastWAM base not found: {base_checkpoint}")
    data_path = processed / "memory" / "warm_data_config.yaml"
    if not data_path.is_file():
        raise FileNotFoundError(f"missing warm data config: {data_path}")

    train_contract, dev_contract = build_contracts(
        processed=processed,
        base_checkpoint=base_checkpoint,
        contract_dir=contract_dir,
        overwrite=overwrite_contracts,
    )
    data = OmegaConf.load(data_path)
    data.train.text_embedding_cache_dir = str(cache_dir)
    if "val" in data:
        data.val.text_embedding_cache_dir = str(cache_dir)
    dataset_dir = Path(str(data.train.dataset_dirs[0])).expanduser().resolve()

    model = _load_warm_model_cfg(int(smoke.get("context_dim", PIPER_CONTEXT_DIM)))
    model.base_checkpoint_path = str(base_checkpoint)
    model.run_contract_path = str(train_contract)
    model.validation_run_contract_path = str(dev_contract)

    train_base = _strip_compose_keys(OmegaConf.load(REPO / "configs/train.yaml"))
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
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-steps", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--base-checkpoint", type=Path, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--overwrite-contracts",
        action="store_true",
        help="rebuild train/dev source-run contracts even if they exist",
    )
    args = parser.parse_args(argv)
    cfg, dataset_dir, cache_dir = build_cfg(
        args.config.expanduser().resolve(),
        overwrite_contracts=bool(args.overwrite_contracts),
        base_checkpoint=args.base_checkpoint,
    )
    if args.run_steps is not None:
        cfg.run_steps = int(args.run_steps)
    if args.num_epochs is not None:
        cfg.num_epochs = int(args.num_epochs)
    if args.output_dir is not None:
        cfg.output_dir = str(args.output_dir.expanduser().resolve())
    if args.save_every is not None:
        cfg.save_every = int(args.save_every)
    if args.eval_every is not None:
        cfg.eval_every = int(args.eval_every)
    if args.num_workers is not None:
        cfg.num_workers = int(args.num_workers)
    output_dir = Path(str(cfg.output_dir)).expanduser().resolve()
    cfg.output_dir = str(output_dir)
    _assert_fresh_output_dir(output_dir)
    ensure_text_embeds(dataset_dir, cache_dir)
    print(f"warm output: {cfg.output_dir}")
    print(
        f"num_epochs={cfg.num_epochs} run_steps={cfg.run_steps} "
        f"batch_size={cfg.batch_size} save_every={cfg.save_every} "
        f"eval_every={cfg.eval_every} eval_num_samples={cfg.eval_num_samples} "
        f"num_workers={cfg.num_workers} "
        f"base={cfg.model.base_checkpoint_path} "
        f"context_dim={cfg.model.retrospection.context_dim} "
        f"mot_checkpoint_mixed_attn={cfg.model.mot_checkpoint_mixed_attn}"
    )
    run_training(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
