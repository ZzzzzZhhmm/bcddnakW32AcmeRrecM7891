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

from fastwam.training_config import validate_training_config
from fastwam.utils.config_resolvers import register_default_resolvers

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "configs/real/piper_warm_smoke.local.yaml"
CONTEXT_LEN = 128
TEXT_CACHE_SUFFIX = f"t5_len{CONTEXT_LEN}.wan22ti2v5b.pt"
PIPER_CONTEXT_DIM = 770
DEFAULT_PROMPT = (
    "A video recorded from a robot's point of view executing the following "
    "instruction: {task}"
)


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


def _is_distributed_launch() -> bool:
    world = os.environ.get("WORLD_SIZE", "1").strip() or "1"
    return world != "1"


def _distributed_env_keys_present() -> list[str]:
    found: list[str] = []
    for key in (
        "RANK",
        "LOCAL_RANK",
        "MASTER_ADDR",
        "ACCELERATE_USE_DEEPSPEED",
        "TORCHELASTIC_RUN_ID",
    ):
        if os.environ.get(key, "").strip():
            found.append(key)
    world = os.environ.get("WORLD_SIZE", "").strip()
    if world and world != "1":
        found.append("WORLD_SIZE")
    return found


def _clear_distributed_env() -> None:
    for key in _DIST_ENV_KEYS:
        os.environ.pop(key, None)


def _require_single_process_job(flag: str) -> None:
    leftover = _distributed_env_keys_present()
    if leftover:
        raise RuntimeError(
            f"{flag} must run as a single process before accelerate; "
            f"leftover distributed env: {leftover}"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _wait_for_source_contracts(
    outputs: tuple[Path, Path],
    ready: Path,
    expected: str,
    *,
    timeout_s: float = 120,
) -> None:
    deadline = time.time() + timeout_s
    while True:
        try:
            if ready.is_file() and all(path.is_file() for path in outputs):
                if ready.read_text(encoding="utf-8").strip() == expected:
                    return
        except OSError:
            pass
        if time.time() >= deadline:
            raise RuntimeError(
                f"timed out waiting for source-run contracts {outputs[0]} "
                f"{outputs[1]} ready={ready}"
            )
        remaining = deadline - time.time()
        time.sleep(min(2.0, max(0.01, remaining)))


def ensure_text_embeds(dataset_dir: Path, cache_dir: Path) -> None:
    tasks = _load_tasks(dataset_dir)
    missing = [task for task in tasks if not _task_cache_path(cache_dir, task).is_file()]
    if not missing:
        print(f"text embeds already present for {len(tasks)} tasks in {cache_dir}")
        return
    if _is_distributed_launch():
        raise FileNotFoundError(
            "text embeds must be prepared in a single process before "
            f"accelerate launch: missing {missing} in {cache_dir}"
        )
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
    subprocess.run(command, cwd=str(REPO), check=True, env=_isolated_subprocess_env())
    still_missing = [
        task for task in tasks if not _task_cache_path(cache_dir, task).is_file()
    ]
    if still_missing:
        raise RuntimeError(f"text embed cache still missing: {still_missing}")


def build_contracts(
    *,
    processed: Path,
    base_checkpoint: Path,
    contract_dir: Path,
    overwrite: bool,
    wait_timeout_s: float = 120,
) -> tuple[Path, Path]:
    bindings = json.loads(
        (processed / "memory" / "training_bindings.json").read_text(encoding="utf-8")
    )
    bank = Path(bindings["event_bank"])
    contract_dir.mkdir(parents=True, exist_ok=True)
    train_contract = contract_dir / "train_source.json"
    dev_contract = contract_dir / "dev_source.json"
    expected = str(Path(base_checkpoint).expanduser().resolve())
    ready = contract_dir / ".source_contracts.ready"
    outputs = (train_contract, dev_contract)
    present = train_contract.is_file() and dev_contract.is_file()

    # LIBERO/RMBench prepare contracts in a single process, then every
    # accelerate rank only reads the paths.  Piper must do the same: never
    # unlink/rebuild under WORLD_SIZE>1.
    if _is_distributed_launch() or not _is_launch_main_process():
        if not present:
            raise FileNotFoundError(
                "Piper source-run contracts must be built in a single process "
                "before accelerate launch (same as LIBERO prepare_artifacts). "
                f"missing={train_contract} {dev_contract}"
            )
        _wait_for_source_contracts(
            outputs, ready, expected, timeout_s=wait_timeout_s
        )
        return train_contract, dev_contract

    ready_matches = False
    if ready.is_file():
        try:
            ready_matches = ready.read_text(encoding="utf-8").strip() == expected
        except OSError:
            ready_matches = False
    rebuild = bool(overwrite or not present or (ready.is_file() and not ready_matches))
    if rebuild and ready.is_file():
        ready.unlink()
    for split, cache, output in (
        ("train", Path(bindings["splits"]["train"]["candidates"]), train_contract),
        ("dev", Path(bindings["splits"]["dev"]["candidates"]), dev_contract),
    ):
        _drop_dead_artifact_claim(
            output.parent / f".{output.name}.warm-artifact.lock"
        )
        if output.is_file() and not rebuild:
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
        if rebuild:
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
    wait_timeout_s: float = 120,
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
        wait_timeout_s=wait_timeout_s,
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
    return cfg, dataset_dir, cache_dir, processed


def _apply_cli_overrides(cfg, args) -> Path:
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
    return output_dir


def _assert_feature_list(list_path: Path) -> int:
    if not list_path.is_file():
        raise FileNotFoundError(f"feature list missing: {list_path}")
    count = 0
    for line_number, raw in enumerate(list_path.read_text(encoding="utf-8").splitlines(), 1):
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = list_path.parent / candidate
        candidate = candidate.resolve()
        if not candidate.is_file():
            raise FileNotFoundError(
                f"feature list entry {line_number} missing: {candidate}"
            )
        sidecar = candidate.with_suffix(".manifest.json")
        if not sidecar.is_file():
            raise FileNotFoundError(
                f"feature list entry {line_number} missing manifest: {sidecar}"
            )
        count += 1
    if count <= 0:
        raise RuntimeError(f"feature list is empty: {list_path}")
    return count


def _preflight_static_artifacts(cfg, processed: Path, *, verify_checkpoint_hash: bool) -> None:
    complete = processed / "memory" / "COMPLETE.json"
    if not complete.is_file():
        raise FileNotFoundError(f"missing memory COMPLETE.json: {complete}")
    manifest = json.loads(
        (processed / "memory" / "event_bank" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    context_width = int(manifest["arrays"]["context_key"]["shape"][1])
    expected_context = int(cfg.model.retrospection.context_dim)
    if context_width != expected_context:
        raise RuntimeError(
            "event-bank context_dim does not match WARM config: "
            f"bank={context_width} cfg={expected_context}"
        )
    action_dim = int(cfg.data.train.processor.action_output_dim)
    proprio_dim = int(cfg.data.train.processor.proprio_output_dim)
    if action_dim != 7 or proprio_dim != 7:
        raise RuntimeError(
            f"Piper WARM requires 7D action/proprio, got action={action_dim} "
            f"proprio={proprio_dim}"
        )
    if int(cfg.model.retrospection.action_dim) != 7:
        raise RuntimeError(
            "resolved retrospection.action_dim must be 7 for Piper, got "
            f"{cfg.model.retrospection.action_dim}"
        )
    if int(cfg.model.action_dit_config.action_dim) != 7:
        raise RuntimeError(
            "resolved Action DiT action_dim must be 7 for Piper, got "
            f"{cfg.model.action_dit_config.action_dim}"
        )
    warm_cfg = cfg.data.warm_candidates
    train_features = _assert_feature_list(
        Path(str(warm_cfg.train.retrospective_feature_list)).expanduser()
    )
    dev_features = _assert_feature_list(
        Path(str(warm_cfg.val.retrospective_feature_list)).expanduser()
    )
    for split_name, split_cfg in (("train", warm_cfg.train), ("val", warm_cfg.val)):
        for field in (
            "bank_directory",
            "candidate_directory",
            "catalog_path",
            "normalization_stats_path",
            "audit_report_path",
        ):
            path = Path(str(split_cfg[field])).expanduser()
            if not path.exists():
                raise FileNotFoundError(
                    f"warm_candidates.{split_name}.{field} missing: {path}"
                )
    train_contract = Path(str(cfg.model.run_contract_path))
    dev_contract = Path(str(cfg.model.validation_run_contract_path))
    train_payload = json.loads(train_contract.read_text(encoding="utf-8"))
    dev_payload = json.loads(dev_contract.read_text(encoding="utf-8"))
    if train_payload.get("query_split") != "train":
        raise RuntimeError(f"train contract query_split is {train_payload.get('query_split')}")
    if dev_payload.get("query_split") != "dev":
        raise RuntimeError(f"dev contract query_split is {dev_payload.get('query_split')}")
    if int(train_payload.get("action_dim", -1)) != 7:
        raise RuntimeError(f"train contract action_dim is {train_payload.get('action_dim')}")
    if int(dev_payload.get("action_dim", -1)) != 7:
        raise RuntimeError(f"dev contract action_dim is {dev_payload.get('action_dim')}")
    checkpoint = Path(str(cfg.model.base_checkpoint_path))
    if verify_checkpoint_hash:
        print(f"preflight hashing base checkpoint {checkpoint}")
        actual = _sha256_file(checkpoint)
        expected = str(train_payload["base_checkpoint_sha256"])
        if actual != expected:
            raise RuntimeError(
                "base checkpoint SHA256 does not match train source-run contract: "
                f"{actual} != {expected}"
            )
        if actual != str(dev_payload["base_checkpoint_sha256"]):
            raise RuntimeError(
                "base checkpoint SHA256 does not match dev source-run contract"
            )
        print(f"preflight checkpoint sha256 ok {actual}")
    wan_root = Path(
        os.environ.get("DIFFSYNTH_MODEL_BASE_PATH", str(REPO / "checkpoints"))
    )
    vae = (
        wan_root
        / "DiffSynth-Studio"
        / "Wan-Series-Converted-Safetensors"
        / "Wan2.2_VAE.safetensors"
    )
    if not vae.is_file():
        raise FileNotFoundError(f"Wan2.2 VAE missing: {vae}")
    ds_config = REPO / "scripts" / "ds_configs" / "ds_zero1_config.json"
    if not ds_config.is_file():
        raise FileNotFoundError(f"DeepSpeed ZeRO-1 json missing: {ds_config}")
    print(
        "preflight artifacts ok: "
        f"context_dim={expected_context} action_dim=7 "
        f"train_feature_files={train_features} dev_feature_files={dev_features} "
        f"vae={vae}"
    )


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
    parser.add_argument(
        "--prepare-contracts",
        action="store_true",
        help=(
            "single-process prepare of source-run contracts and text embeds, "
            "then exit; required before multi-GPU accelerate launch"
        ),
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help=(
            "single-process launch audit: contracts, artifacts, resolved "
            "config, and checkpoint hash; does not load the 5B model"
        ),
    )
    parser.add_argument(
        "--skip-checkpoint-hash",
        action="store_true",
        help="preflight without hashing the 12G FastWAM checkpoint",
    )
    args = parser.parse_args(argv)
    exclusive = [
        name
        for name, enabled in (
            ("--prepare-contracts", args.prepare_contracts),
            ("--preflight", args.preflight),
        )
        if enabled
    ]
    if len(exclusive) > 1:
        raise RuntimeError("use only one of --prepare-contracts and --preflight")
    if args.prepare_contracts or args.preflight:
        _require_single_process_job(
            "--prepare-contracts" if args.prepare_contracts else "--preflight"
        )
        _clear_distributed_env()
    cfg, dataset_dir, cache_dir, processed = build_cfg(
        args.config.expanduser().resolve(),
        overwrite_contracts=bool(args.overwrite_contracts),
        base_checkpoint=args.base_checkpoint,
    )
    output_dir = _apply_cli_overrides(cfg, args)
    ensure_text_embeds(dataset_dir, cache_dir)
    if args.prepare_contracts:
        print(
            "prepared Piper WARM contracts and text embeds; "
            "launch training without --overwrite-contracts"
        )
        return 0
    if args.preflight:
        validate_training_config(cfg)
        _assert_fresh_output_dir(output_dir)
        _preflight_static_artifacts(
            cfg,
            processed,
            verify_checkpoint_hash=not args.skip_checkpoint_hash,
        )
        dump_a = OmegaConf.to_yaml(cfg, resolve=True)
        dump_b = OmegaConf.to_yaml(cfg, resolve=True)
        if dump_a != dump_b:
            raise RuntimeError("resolved training config is not stable across dumps")
        print(
            "preflight ok: "
            f"output={cfg.output_dir} epochs={cfg.num_epochs} "
            f"save_every={cfg.save_every} eval_every={cfg.eval_every} "
            f"batch_size={cfg.batch_size} num_workers={cfg.num_workers} "
            f"context_dim={cfg.model.retrospection.context_dim} "
            f"base={cfg.model.base_checkpoint_path}"
        )
        return 0
    _assert_fresh_output_dir(output_dir)
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
    from fastwam.runtime import run_training

    run_training(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
