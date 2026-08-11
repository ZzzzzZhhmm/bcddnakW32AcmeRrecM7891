import logging
import hashlib
import os
import inspect
import json
from pathlib import Path
import re
from tempfile import NamedTemporaryFile

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from PIL import Image
import numpy as np
from einops import repeat
from omegaconf import OmegaConf

from .trainer import Wan22Trainer
from .training_config import validate_training_config
from .utils.logging_config import get_logger, setup_logging
from .utils.video_io import save_mp4
from .utils import misc

logger = get_logger(__name__)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    if not isinstance(mixed_precision, str):
        raise ValueError(f"`mixed_precision` must be str, got {type(mixed_precision)}")
    key = mixed_precision.strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def create_wan22_model(
    model_id: str,
    tokenizer_model_id: str,
    dit_config,
    tokenizer_max_len: int = 512,
    train_shift: float = 5.0,
    infer_shift: float = 5.0,
    num_train_timesteps: int = 1000,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    from .models.wan22.wan22 import Wan22Core

    if isinstance(dit_config, DictConfig):
        dit_config = OmegaConf.to_container(dit_config, resolve=True)
    if not isinstance(dit_config, dict):
        raise ValueError(f"`dit_config` must resolve to a dict, got {type(dit_config)}")

    return Wan22Core.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        redirect_common_files=bool(redirect_common_files),
        dit_config=dit_config,
        train_shift=float(train_shift),
        infer_shift=float(infer_shift),
        num_train_timesteps=int(num_train_timesteps),
    )


def create_fastwam(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    mot_checkpoint_mixed_attn: bool = True,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    _model_class=None,
    _from_pretrained_extra: dict | None = None,
):
    if _model_class is None:
        from .models.wan22.fastwam import FastWAM
    else:
        FastWAM = _model_class

    if isinstance(video_dit_config, DictConfig):
        video_dit_config = OmegaConf.to_container(video_dit_config, resolve=True)
    if not isinstance(video_dit_config, dict):
        raise ValueError(f"`video_dit_config` must resolve to a dict, got {type(video_dit_config)}")

    if isinstance(action_dit_config, DictConfig):
        action_dit_config = OmegaConf.to_container(action_dit_config, resolve=True)
    if action_dit_config is None:
        action_dit_config = {}
    if not isinstance(action_dit_config, dict):
        raise ValueError(f"`action_dit_config` must resolve to a dict, got {type(action_dit_config)}")

    if isinstance(video_scheduler, DictConfig):
        video_scheduler = OmegaConf.to_container(video_scheduler, resolve=True)
    if video_scheduler is None:
        video_scheduler = {}
    if not isinstance(video_scheduler, dict):
        raise ValueError(f"`video_scheduler` must be dict-like, got {type(video_scheduler)}")

    if isinstance(action_scheduler, DictConfig):
        action_scheduler = OmegaConf.to_container(action_scheduler, resolve=True)
    if action_scheduler is None:
        raise ValueError("`action_scheduler` is required for FastWAM.")
    if not isinstance(action_scheduler, dict):
        raise ValueError(f"`action_scheduler` must be dict-like, got {type(action_scheduler)}")
    required_action_scheduler_keys = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing_keys = required_action_scheduler_keys - set(action_scheduler.keys())
    if missing_keys:
        raise ValueError(
            f"`action_scheduler` missing required keys: {sorted(missing_keys)}. "
            "Expected keys: train_shift, infer_shift, num_train_timesteps."
        )

    if isinstance(loss, DictConfig):
        loss = OmegaConf.to_container(loss, resolve=True)
    if loss is None:
        loss = {}
    if not isinstance(loss, dict):
        raise ValueError(f"`loss` must be dict-like, got {type(loss)}")

    extra = {} if _from_pretrained_extra is None else dict(_from_pretrained_extra)
    return FastWAM.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
        **extra,
    )


def create_warm_source(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    mot_checkpoint_mixed_attn: bool = True,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    source_policy: str = "fixed_context_top1",
    memory_sigma: float = 0.2,
    run_contract=None,
    run_contract_path: str | None = None,
    validation_run_contract=None,
    validation_run_contract_path: str | None = None,
    base_checkpoint_path: str | None = None,
    _warm_model_class=None,
    _warm_pretrained_extra: dict | None = None,
):
    """Create the base-FastWAM M2 model with strict source-only semantics."""

    from .memory.manifest import sha256_file
    from .models.warm.source_contract import WarmSourceRunContract
    from .models.warm.source_model import WarmSourceFastWAM

    if run_contract is not None and run_contract_path is not None:
        raise ValueError(
            "run_contract and run_contract_path are mutually exclusive"
        )
    if run_contract_path is not None:
        contract_path = Path(run_contract_path).expanduser().resolve()
        with open(contract_path, "r", encoding="utf-8") as file:
            run_contract = json.load(file)
    if (
        validation_run_contract is not None
        and validation_run_contract_path is not None
    ):
        raise ValueError(
            "validation_run_contract and validation_run_contract_path are "
            "mutually exclusive"
        )
    if validation_run_contract_path is not None:
        validation_contract_path = Path(
            validation_run_contract_path
        ).expanduser().resolve()
        with open(validation_contract_path, "r", encoding="utf-8") as file:
            validation_run_contract = json.load(file)
    if isinstance(run_contract, DictConfig):
        run_contract = OmegaConf.to_container(run_contract, resolve=True)
    if isinstance(validation_run_contract, DictConfig):
        validation_run_contract = OmegaConf.to_container(
            validation_run_contract, resolve=True
        )
    if run_contract is None:
        contract = None
    elif isinstance(run_contract, WarmSourceRunContract):
        contract = run_contract
    else:
        contract = WarmSourceRunContract.from_dict(run_contract)
    if validation_run_contract is None:
        validation_contract = None
    elif isinstance(validation_run_contract, WarmSourceRunContract):
        validation_contract = validation_run_contract
    else:
        validation_contract = WarmSourceRunContract.from_dict(
            validation_run_contract
        )

    if source_policy not in {
        "gaussian_null",
        "fixed_context_top1",
        "oracle_action_top1",
    }:
        raise ValueError(f"unsupported WARM source_policy {source_policy!r}")
    resolved_action_config = action_dit_config
    if isinstance(resolved_action_config, DictConfig):
        resolved_action_config = OmegaConf.to_container(
            resolved_action_config, resolve=True
        )
    if contract is not None and isinstance(resolved_action_config, dict):
        configured_action_dim = resolved_action_config.get("action_dim")
        if (
            configured_action_dim is not None
            and int(configured_action_dim) != contract.action_dim
        ):
            raise ValueError(
                "Action DiT config does not match WarmSourceRunContract "
                f"action_dim: {configured_action_dim} != {contract.action_dim}"
            )

    # The public Hydra/runtime factory is the formal experiment path.  Even
    # the Gaussian-null ablation must start from the exact same immutable
    # FastWAM checkpoint and artifact contract as fixed/oracle runs; otherwise
    # warm_source.yaml's skipped pretrained loads could leave random experts
    # and create a meaningless "null" comparison.  Tiny unit fixtures may
    # still construct WarmSourceFastWAM directly without a contract.
    if contract is None:
        raise ValueError(
            "formal WARM runtime requires run_contract or run_contract_path "
            "for every source policy, including gaussian_null"
        )
    if base_checkpoint_path is None:
        raise ValueError(
            "a run contract requires base_checkpoint_path for hash verification"
        )

    base_path = None
    if base_checkpoint_path is not None:
        base_path = Path(base_checkpoint_path).expanduser().resolve()
        if not base_path.is_file():
            raise FileNotFoundError(f"base checkpoint not found: {base_path}")
        if contract is None:
            raise ValueError(
                "base_checkpoint_path requires a run contract binding its hash"
            )
        actual_sha256 = sha256_file(base_path)
        if actual_sha256 != contract.base_checkpoint_sha256:
            raise ValueError(
                "base checkpoint SHA256 does not match WarmSourceRunContract: "
                f"{actual_sha256} != {contract.base_checkpoint_sha256}"
            )

    model_class = WarmSourceFastWAM if _warm_model_class is None else _warm_model_class
    pretrained_extra = {
        "warm_source_policy": source_policy,
        "memory_sigma": memory_sigma,
        "warm_run_contract": contract,
        "warm_validation_run_contract": validation_contract,
    }
    if _warm_pretrained_extra:
        collisions = set(pretrained_extra).intersection(_warm_pretrained_extra)
        if collisions:
            raise ValueError(
                f"duplicate WARM pretrained arguments: {sorted(collisions)}"
            )
        pretrained_extra.update(_warm_pretrained_extra)
    model = create_fastwam(
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        video_dit_config=video_dit_config,
        tokenizer_max_len=tokenizer_max_len,
        load_text_encoder=load_text_encoder,
        proprio_dim=proprio_dim,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
        video_scheduler=video_scheduler,
        action_scheduler=action_scheduler,
        loss=loss,
        mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        redirect_common_files=redirect_common_files,
        model_dtype=model_dtype,
        device=device,
        _model_class=model_class,
        _from_pretrained_extra=pretrained_extra,
    )
    if base_path is not None:
        model.load_base_checkpoint(str(base_path))
    return model


def create_warm_retrospection(*, retrospection, **kwargs):
    """Create the complete consequence-aligned WARM model.

    The immutable M2 source contracts still bind the bank, train/dev candidate
    corpora, and baseline checkpoint.  ``fixed_context_top1`` here describes
    the coarse retrieval family; final selection is learned and consequence
    aligned inside :class:`WarmRetrospectionFastWAM`.
    """

    from .models.warm.retrospection_config import WarmRetrospectionConfig
    from .models.warm.retrospection_model import WarmRetrospectionFastWAM

    if isinstance(retrospection, DictConfig):
        retrospection = OmegaConf.to_container(retrospection, resolve=True)
    config = (
        retrospection
        if isinstance(retrospection, WarmRetrospectionConfig)
        else WarmRetrospectionConfig.from_dict(retrospection)
    )
    source_policy = kwargs.pop("source_policy", "fixed_context_top1")
    if source_policy != "fixed_context_top1":
        raise ValueError(
            "complete WARM uses fixed_context_top1 only for coarse retrieval; "
            "learned consequence selection supplies the final source"
        )
    return create_warm_source(
        source_policy=source_policy,
        _warm_model_class=WarmRetrospectionFastWAM,
        _warm_pretrained_extra={"warm_retrospection_config": config},
        **kwargs,
    )


def create_fastwam_joint(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    mot_checkpoint_mixed_attn: bool = True,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    from .models.wan22.fastwam_joint import FastWAMJoint

    if isinstance(video_dit_config, DictConfig):
        video_dit_config = OmegaConf.to_container(video_dit_config, resolve=True)
    if not isinstance(video_dit_config, dict):
        raise ValueError(f"`video_dit_config` must resolve to a dict, got {type(video_dit_config)}")

    if isinstance(action_dit_config, DictConfig):
        action_dit_config = OmegaConf.to_container(action_dit_config, resolve=True)
    if action_dit_config is None:
        action_dit_config = {}
    if not isinstance(action_dit_config, dict):
        raise ValueError(f"`action_dit_config` must resolve to a dict, got {type(action_dit_config)}")

    if isinstance(video_scheduler, DictConfig):
        video_scheduler = OmegaConf.to_container(video_scheduler, resolve=True)
    if video_scheduler is None:
        video_scheduler = {}
    if not isinstance(video_scheduler, dict):
        raise ValueError(f"`video_scheduler` must be dict-like, got {type(video_scheduler)}")

    if isinstance(action_scheduler, DictConfig):
        action_scheduler = OmegaConf.to_container(action_scheduler, resolve=True)
    if action_scheduler is None:
        raise ValueError("`action_scheduler` is required for FastWAM.")
    if not isinstance(action_scheduler, dict):
        raise ValueError(f"`action_scheduler` must be dict-like, got {type(action_scheduler)}")
    required_action_scheduler_keys = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing_keys = required_action_scheduler_keys - set(action_scheduler.keys())
    if missing_keys:
        raise ValueError(
            f"`action_scheduler` missing required keys: {sorted(missing_keys)}. "
            "Expected keys: train_shift, infer_shift, num_train_timesteps."
        )

    if isinstance(loss, DictConfig):
        loss = OmegaConf.to_container(loss, resolve=True)
    if loss is None:
        loss = {}
    if not isinstance(loss, dict):
        raise ValueError(f"`loss` must be dict-like, got {type(loss)}")

    return FastWAMJoint.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
    )


def create_fastwam_idm(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    mot_checkpoint_mixed_attn: bool = True,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    from .models.wan22.fastwam_idm import (
        FastWAMIDM,
    )

    if isinstance(video_dit_config, DictConfig):
        video_dit_config = OmegaConf.to_container(video_dit_config, resolve=True)
    if not isinstance(video_dit_config, dict):
        raise ValueError(f"`video_dit_config` must resolve to a dict, got {type(video_dit_config)}")

    if isinstance(action_dit_config, DictConfig):
        action_dit_config = OmegaConf.to_container(action_dit_config, resolve=True)
    if action_dit_config is None:
        action_dit_config = {}
    if not isinstance(action_dit_config, dict):
        raise ValueError(f"`action_dit_config` must resolve to a dict, got {type(action_dit_config)}")

    if isinstance(video_scheduler, DictConfig):
        video_scheduler = OmegaConf.to_container(video_scheduler, resolve=True)
    if video_scheduler is None:
        video_scheduler = {}
    if not isinstance(video_scheduler, dict):
        raise ValueError(f"`video_scheduler` must be dict-like, got {type(video_scheduler)}")

    if isinstance(action_scheduler, DictConfig):
        action_scheduler = OmegaConf.to_container(action_scheduler, resolve=True)
    if action_scheduler is None:
        raise ValueError("`action_scheduler` is required for FastWAM.")
    if not isinstance(action_scheduler, dict):
        raise ValueError(f"`action_scheduler` must be dict-like, got {type(action_scheduler)}")
    required_action_scheduler_keys = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing_keys = required_action_scheduler_keys - set(action_scheduler.keys())
    if missing_keys:
        raise ValueError(
            f"`action_scheduler` missing required keys: {sorted(missing_keys)}. "
            "Expected keys: train_shift, infer_shift, num_train_timesteps."
        )

    if isinstance(loss, DictConfig):
        loss = OmegaConf.to_container(loss, resolve=True)
    if loss is None:
        loss = {}
    if not isinstance(loss, dict):
        raise ValueError(f"`loss` must be dict-like, got {type(loss)}")

    return FastWAMIDM.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
    )


def build_datasets(data_cfg: DictConfig, *, build_validation: bool = True):
    train_ds = instantiate(data_cfg.train)
    warm_cfg = data_cfg.get("warm_candidates")
    if warm_cfg is not None:
        train_candidate_cfg = warm_cfg.get("train")
        if train_candidate_cfg is None:
            raise ValueError(
                "data.warm_candidates.train is required when WARM candidates are enabled"
            )
        train_ds = _wrap_warm_candidate_dataset(
            train_ds,
            train_candidate_cfg,
            expected_query_split="train",
        )
    if not build_validation or data_cfg.get("val") is None:
        val_ds = train_ds
    else:
        train_stats_path = data_cfg.train.get("pretrained_norm_stats")
        default_stats_path = os.path.join(misc.get_work_dir(), "dataset_stats.json")
        val_stats_path = data_cfg.val.get("pretrained_norm_stats")
        pretrained_norm_stats = val_stats_path or train_stats_path or default_stats_path
        logger.info("Building val dataset with pretrained_norm_stats: %s", pretrained_norm_stats)
        val_ds = instantiate(data_cfg.val, pretrained_norm_stats=pretrained_norm_stats)
        if warm_cfg is not None:
            val_candidate_cfg = warm_cfg.get("val")
            if val_candidate_cfg is None:
                raise ValueError(
                    "data.warm_candidates.val is required when a separate "
                    "validation dataset is configured"
                )
            val_ds = _wrap_warm_candidate_dataset(
                val_ds,
                val_candidate_cfg,
                expected_query_split="dev",
            )
    return train_ds, val_ds


def _wrap_warm_candidate_dataset(
    dataset,
    candidate_cfg,
    *,
    expected_query_split: str,
):
    """Attach one immutable candidate-cache row to each exact dataset sample."""

    from .datasets.warm_candidates import RuntimeCandidateDatasetAdapter
    from .memory.runtime_candidates import RuntimeCandidateResolver

    if isinstance(candidate_cfg, DictConfig):
        candidate_cfg = OmegaConf.to_container(
            candidate_cfg, resolve=True
        )
    if not isinstance(candidate_cfg, dict):
        raise TypeError("WARM candidate dataset config must resolve to a dict")
    allowed = {
        "bank_directory",
        "candidate_directory",
        "catalog_path",
        "normalization_stats_path",
        "audit_report_path",
        "expected_query_corpus_sha256",
        "retrospective_feature_directory",
        "retrospective_feature_list",
        "retrospective_recent_event_capacity",
        "retrospective_action_summary_capacity",
        "retrospective_action_summary_chunk_size",
        "retrospective_gripper_indices",
    }
    extra = set(candidate_cfg) - allowed
    if extra:
        raise ValueError(
            f"unsupported WARM candidate dataset config keys: {sorted(extra)}"
        )
    missing = [
        field
        for field in (
            "bank_directory",
            "candidate_directory",
            "catalog_path",
            "normalization_stats_path",
            "audit_report_path",
        )
        if not candidate_cfg.get(field)
    ]
    if missing:
        raise ValueError(
            f"WARM candidate dataset paths are required: {sorted(missing)}"
        )
    resolver = RuntimeCandidateResolver.from_artifacts(
        candidate_cfg["bank_directory"],
        candidate_cfg["candidate_directory"],
        expected_query_split=expected_query_split,
        expected_query_corpus_sha256=candidate_cfg.get(
            "expected_query_corpus_sha256"
        ),
    )
    adapted = RuntimeCandidateDatasetAdapter(
        dataset,
        resolver,
        candidate_cfg["catalog_path"],
        normalization_stats_path=candidate_cfg["normalization_stats_path"],
        audit_report_path=candidate_cfg["audit_report_path"],
    )
    feature_directory = candidate_cfg.get("retrospective_feature_directory")
    feature_list = candidate_cfg.get("retrospective_feature_list")
    if feature_directory is not None and feature_list is not None:
        raise ValueError(
            "warm_candidates retrospective_feature_directory and "
            "retrospective_feature_list are mutually exclusive"
        )
    if feature_directory is None and feature_list is None:
        return adapted
    from .datasets.warm_retrospective import (
        RetrospectiveFeatureStore,
        RuntimeRetrospectiveDatasetAdapter,
        collect_feature_payloads,
        collect_feature_payloads_from_list,
    )

    recent_capacity = int(
        candidate_cfg.get("retrospective_recent_event_capacity", 6)
    )
    feature_paths = (
        collect_feature_payloads(feature_directory)
        if feature_directory is not None
        else collect_feature_payloads_from_list(feature_list)
    )
    feature_store = RetrospectiveFeatureStore.from_paths(
        feature_paths,
        expected_split=expected_query_split,
        expected_collection_sha256=resolver.query_corpus_sha256,
        expected_catalog_sha256=resolver.query_catalog_sha256,
        action_horizon=resolver.action_horizon,
        recent_event_capacity=recent_capacity,
        action_summary_capacity=int(
            candidate_cfg.get("retrospective_action_summary_capacity", 2)
        ),
        action_summary_chunk_size=int(
            candidate_cfg.get(
                "retrospective_action_summary_chunk_size",
                resolver.action_horizon,
            )
        ),
        gripper_indices=tuple(
            int(value)
            for value in candidate_cfg.get("retrospective_gripper_indices", ())
        ),
        # The action-space contract, not a free Hydra override, determines
        # how factual executed chunks are summarized.  RMBench/RoboTwin uses
        # absolute qpos targets; LIBERO uses delta commands.
        action_mode=(
            "absolute_target"
            if resolver.action_space.control_mode
            == "robotwin_bimanual_qpos_plus_grippers"
            else "delta"
        ),
        # DINO/VAE relative changes in the 3-camera RMBench bridge are an
        # order of magnitude smaller than the original LIBERO defaults.  A
        # benchmark-bound floor preserves contact/progress events without an
        # online mask or manual event annotation.
        change_threshold_floor=(
            0.015
            if resolver.action_space.control_mode
            == "robotwin_bimanual_qpos_plus_grippers"
            else 0.08
        ),
        change_threshold_ceiling=(
            0.25
            if resolver.action_space.control_mode
            == "robotwin_bimanual_qpos_plus_grippers"
            else 0.85
        ),
    )
    return RuntimeRetrospectiveDatasetAdapter(adapted, feature_store)


def _resolve_train_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    device_count = torch.cuda.device_count()
    if device_count <= 1:
        return "cuda:0"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank < 0 or local_rank >= device_count:
        return "cuda:0"
    return f"cuda:{local_rank}"


def _publish_resolved_training_config(cfg: DictConfig) -> Path:
    """Publish, never overwrite, the exact config for this launch.

    A continuation commonly reuses the parent's output directory.  Replacing
    ``config.yaml`` would destroy the original run record, and every rank used
    to race that replacement before Accelerate initialized distributed state.
    Resume launches therefore receive an immutable step-specific config; all
    ranks may race the same atomic hard-link publication only when their bytes
    are identical.
    """

    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    resume = cfg.get("resume")
    payload = OmegaConf.to_yaml(cfg, resolve=True).encode("utf-8")
    if resume not in (None, "", False):
        match = re.fullmatch(r"step_(\d{6,})", Path(str(resume)).name)
        suffix = match.group(1) if match is not None else "unresolved"
        identity = hashlib.sha256(payload).hexdigest()[:12]
        target = output / f"config.resume.step_{suffix}.{identity}.yaml"
    else:
        target = output / "config.yaml"
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.read_bytes() != payload:
                raise RuntimeError(
                    f"resolved training config already exists with different bytes: {target}"
                )
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


def run_training(cfg: DictConfig):
    # Fail before loading the 5B Video DiT and 1B Action DiT when an override
    # cannot define a valid optimizer loop.
    validate_training_config(cfg)
    setup_logging(
        log_level=logging.INFO,
        is_main_process=torch.distributed.get_rank() == 0 if torch.distributed.is_initialized() else True,
    )
    misc.register_work_dir(cfg.output_dir)
    resolved_config_path = _publish_resolved_training_config(cfg)
    logger.info("Resolved training config published at %s", resolved_config_path)

    # Accelerate launches one Python process per rank before DeepSpeed is
    # initialized.  Seed model construction identically here; otherwise WARM
    # modules that do not exist in the immutable FastWAM checkpoint can be
    # initialized differently on every rank.  Wan22Trainer deliberately
    # reseeds with a rank offset after construction for independent training
    # RNG streams and dataloader workers.
    from .utils.pytorch_utils import set_global_seed

    set_global_seed(int(cfg.seed), rank_offset=False)
    model_device = _resolve_train_device()
    logger.info(
        "Using rank-invariant model initialization seed=%d on device=%s",
        int(cfg.seed),
        model_device,
    )
    mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    evaluation_enabled = int(cfg.get("eval_every", 0)) > 0
    validate_validation_dataset = getattr(
        model, "validate_validation_dataset", None
    )
    # Formal WARM always binds an independent DEV source contract.  Construct
    # and validate that dataset even if a debug launch disables periodic
    # metrics; otherwise a bad DEV cache can survive until checkpoint eval.
    validation_required = evaluation_enabled or callable(
        validate_validation_dataset
    )
    train_ds, val_ds = build_datasets(
        cfg.data, build_validation=validation_required
    )
    validate_training_dataset = getattr(model, "validate_training_dataset", None)
    if callable(validate_training_dataset):
        validate_training_dataset(train_ds)
    if callable(validate_validation_dataset):
        if val_ds is train_ds:
            raise ValueError(
                "formal WARM training requires a separate catalog-bound "
                "dev dataset/cache; reusing the train dataset is forbidden"
            )
        if val_ds is not train_ds:
            validate_validation_dataset(val_ds)

    trainer = Wan22Trainer.create(
        cfg=cfg,
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
    )
    try:
        trainer.train()
    except BaseException:
        # Teardown must not replace the original forward/backward/checkpoint
        # exception with a secondary NCCL or tracker shutdown failure.
        try:
            trainer.close()
        except Exception:
            logger.exception("Training cleanup failed after the primary error")
        raise
    else:
        trainer.close()

def run_inference(cfg: DictConfig):
    setup_logging(log_level=logging.INFO)
    inference_cfg = cfg.inference
    mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)

    model = instantiate(cfg.model, model_dtype=model_dtype, device=str(inference_cfg.device))
    checkpoint_path = inference_cfg.get("checkpoint_path")
    if checkpoint_path:
        ckpt = Path(checkpoint_path)
        if ckpt.exists():
            logger.info("Loading finetuned checkpoint: %s", checkpoint_path)
            model.load_checkpoint(checkpoint_path)
        else:
            logger.warning("Checkpoint not found, skipping load: %s", checkpoint_path)
    model.eval()
    
    def center_crop_resize(img: Image, width: int, height: int) -> Image.Image:
        src_w, src_h = img.size
        scale = max(width / src_w, height / src_h)
        resized = img.resize((round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR)
        rw, rh = resized.size
        left = max((rw - width) // 2, 0)
        top = max((rh - height) // 2, 0)
        return resized.crop((left, top, left + width, top + height))

    input_image = Image.open(str(inference_cfg.input_image_path)).convert("RGB")
    input_image = center_crop_resize(input_image, width=inference_cfg.width, height=inference_cfg.height)
    arr = np.array(input_image, dtype=np.float32)
    x = torch.from_numpy(arr)
    x = x.to(device=model.device, dtype=model.torch_dtype)
    x = x * (2.0 / 255.0) - 1.0
    x = repeat(x, "H W C -> B C H W", B=1)
    output_mp4 = str(inference_cfg.output_mp4)

    infer_kwargs = {
        "prompt": str(inference_cfg.prompt),
        "negative_prompt": str(inference_cfg.negative_prompt),
        "text_cfg_scale": float(inference_cfg.text_cfg_scale),
        "action_cfg_scale": float(inference_cfg.action_cfg_scale),
        "input_image": x,
        "num_frames": int(inference_cfg.num_frames),
        "num_inference_steps": int(inference_cfg.num_inference_steps),
        "sigma_shift": None if inference_cfg.get("sigma_shift") is None else float(inference_cfg.sigma_shift),
        "seed": int(inference_cfg.seed),
        "rand_device": str(inference_cfg.rand_device),
        "tiled": bool(inference_cfg.tiled),
    }

    infer_out = model.infer(**infer_kwargs)
    video = infer_out["video"]
    save_mp4(video, output_mp4, fps=15)
    logger.info("Saved inference video to %s", output_mp4)
    return output_mp4
