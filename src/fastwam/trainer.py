import logging
import json
import inspect
import os
import re
from math import ceil
from pathlib import Path
import time
from uuid import uuid4

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedType
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from .utils.fs import ensure_dir
from .utils.logging_config import get_logger, setup_logging
from .utils.pytorch_utils import set_global_seed
from .utils.samplers import (
    ResumableEpochSampler,
    ResumableTaskEventBalancedSampler,
)
from .utils.video_io import save_mp4
from .utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim
from .training_config import validate_training_config

logger = get_logger(__name__)


class Wan22Trainer:
    @classmethod
    def create(cls, model, train_dataset, val_dataset=None, *, cfg: DictConfig):
        """Construct a trainer while retaining access for failure teardown.

        Python does not return an object whose ``__init__`` raised.  Allocating
        first lets us close an Accelerator/DeepSpeed process group when a late
        initialization check (for example formal-resume verification) fails.
        """

        trainer = cls.__new__(cls)
        trainer._closed = False
        try:
            cls.__init__(
                trainer,
                model,
                train_dataset,
                val_dataset,
                cfg=cfg,
            )
        except BaseException:
            try:
                trainer.close()
            except Exception:
                logger.exception(
                    "Trainer cleanup failed after an initialization error"
                )
            raise
        return trainer

    def __init__(self, model, train_dataset, val_dataset=None, *, cfg: DictConfig):
        self._closed = False
        validate_training_config(cfg)
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg
        self.output_dir = str(cfg.output_dir)
        self.learning_rate = float(cfg.learning_rate)
        self.weight_decay = float(cfg.weight_decay)
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        self.num_epochs = int(cfg.num_epochs)
        max_steps = cfg.max_steps
        self.max_steps = int(max_steps) if max_steps is not None else None
        run_steps = cfg.get("run_steps", None)
        self.run_steps = int(run_steps) if run_steps is not None else None
        if self.run_steps is not None and self.run_steps <= 0:
            raise ValueError("run_steps must be a positive integer or null")
        self.log_every = int(cfg.log_every)
        self.save_every = int(cfg.save_every)
        self.eval_every = int(cfg.eval_every)
        self.eval_num_inference_steps = int(cfg.eval_num_inference_steps)
        self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.max_nonfinite_gradient_skips = int(
            cfg.get("max_nonfinite_gradient_skips", 3)
        )
        if self.max_nonfinite_gradient_skips <= 0:
            raise ValueError("max_nonfinite_gradient_skips must be positive")
        self._consecutive_nonfinite_gradient_skips = 0
        self.seed = int(cfg.seed)
        sampler_cfg = cfg.get("sampler", {})
        self.sampler_mode = str(sampler_cfg.get("mode", "random")).strip()
        self.sampler_event_boost = float(sampler_cfg.get("event_boost", 1.5))
        
        self.resume = cfg.resume
        allow_unattested = cfg.get("allow_unattested_warm_checkpoints", False)
        if not isinstance(allow_unattested, bool):
            raise TypeError("allow_unattested_warm_checkpoints must be a boolean")
        self.allow_unattested_warm_checkpoints = allow_unattested
        self.mixed_precision = str(cfg.mixed_precision).strip().lower()
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(
                f"Unsupported mixed_precision: {cfg.mixed_precision}. "
                "Expected one of: ['no', 'fp16', 'bf16']."
            )
        self.wandb_enabled = bool(cfg.wandb.enabled)

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            step_scheduler_with_optimizer=False,
        )
        
        logger.info(
            "Accelerate training: distributed_type=%s zero_stage=%s world_size=%d process_index=%d cfg_mixed_precision=%s accelerator_mixed_precision=%s grad_accum=%d grad_clip=%.4f",
            self.accelerator.distributed_type,
            (
                getattr(self.accelerator.state, "deepspeed_plugin", None)
                and self.accelerator.state.deepspeed_plugin.deepspeed_config.get(
                    "zero_optimization", {}
                ).get("stage", "unknown")
            )
            or "none",
            self.accelerator.num_processes,
            self.accelerator.process_index,
            self.mixed_precision,
            self.accelerator.mixed_precision,
            self.gradient_accumulation_steps,
            self.max_grad_norm,
        )
        logger.info("using accelerator.device=%s", self.accelerator.device)
        worker_init_fn = set_global_seed(self.seed, get_worker_init_fn=True)
        self._assert_dataset_length_consistent(self.train_dataset, "train_dataset")
        if self.val_dataset is not None:
            self._assert_dataset_length_consistent(self.val_dataset, "val_dataset")

        # Apply the model-owned trainable-module contract before
        # optimizer/DeepSpeed initialization.  Complete WARM returns Action
        # DiT, compact retrospective modules, selected video adapters, and
        # the optional proprio bridge while leaving the 5B backbone frozen.
        trainable_params = self._apply_dit_only_train_mode(self.model)
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )
        
        self.train_loader = self._build_loader(self.train_dataset, worker_init_fn=worker_init_fn)
        total_train_steps = self._estimate_total_train_steps()
        self.max_steps = total_train_steps
        warmup_steps = int(total_train_steps * 0.05)
        self.scheduler_type = str(cfg.lr_scheduler_type).strip().lower()
        self.scheduler_warmup_steps = warmup_steps
        self.scheduler_min_learning_rate = (
            self.learning_rate * 0.01
            if self.scheduler_type == "cosine"
            else self.learning_rate
        )
        self.scheduler = self._build_scheduler(
            scheduler_type=self.scheduler_type,
            total_train_steps=total_train_steps,
            warmup_steps=warmup_steps,
        )
        self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0

        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        self.weights_dir = os.path.join(self.checkpoint_root, "weights")
        self.state_dir = os.path.join(self.checkpoint_root, "state")
        self.eval_dir = os.path.join(self.output_dir, "eval")

        ensure_dir(self.output_dir)
        ensure_dir(self.checkpoint_root)
        ensure_dir(self.weights_dir)
        ensure_dir(self.state_dir)
        ensure_dir(self.eval_dir)

        self._last_training_attestation_path: str | None = None

        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler
        )
        self._validate_deepspeed_numerics_contract()
        # Formal provenance must describe the actual objects and distributed
        # runtime returned by Accelerate/DeepSpeed, not only the pre-prepare
        # Python recipe.
        self._warm_training_attestation_context = (
            self._prepare_warm_training_attestation_context()
        )
        self.optimizer.zero_grad(set_to_none=True)
        self.wandb_run = None
        self._init_wandb()
        self._resume_or_load_checkpoint()
        # Validate the state that will actually enter the first forward.  Keep
        # this after resume loading so a corrupt or rank-divergent checkpoint
        # cannot bypass the audit.
        self._validate_prepared_trainable_state()

        val_size = len(self.val_dataset) if self.val_dataset is not None else len(self.train_dataset)
        logger.info("Train/val dataset size: %d/%d", len(self.train_dataset), val_size)

    def _is_deepspeed(self) -> bool:
        return self.accelerator.distributed_type == DistributedType.DEEPSPEED

    def _validate_deepspeed_numerics_contract(self) -> None:
        """Fail early unless BF16 DeepSpeed can block a corrupt update.

        Accelerate's DeepSpeed wrapper performs ``engine.step()`` from inside
        ``accelerator.backward`` on a synchronized microbatch.  Consequently,
        a gradient check performed after ``backward`` is too late.  DeepSpeed
        BF16 gradient-overflow checking must therefore be an explicit runtime
        contract, not merely a JSON-file intention.
        """

        if not self._is_deepspeed():
            return
        runtime = getattr(self.model, "_config", None)
        if runtime is None:
            raise RuntimeError("DeepSpeed engine has no resolved runtime config")
        bf16 = getattr(runtime, "bfloat16_config", None)
        overflow_guard = getattr(bf16, "check_grad_overflow", None)
        gradient_clipping = getattr(runtime, "gradient_clipping", None)
        engine_optimizer = getattr(self.model, "optimizer", None)
        if self.mixed_precision == "bf16" and overflow_guard is not True:
            raise RuntimeError(
                "BF16 DeepSpeed requires bf16.check_grad_overflow=true; "
                f"resolved value={overflow_guard!r}"
            )
        if self.mixed_precision == "bf16" and not hasattr(
            engine_optimizer, "overflow"
        ):
            raise RuntimeError(
                "BF16 DeepSpeed optimizer does not expose the overflow flag "
                "required by Accelerator.optimizer_step_was_skipped"
            )
        optimizer_guard = getattr(
            engine_optimizer, "check_grad_overflow", None
        )
        if self.mixed_precision == "bf16" and optimizer_guard is not True:
            raise RuntimeError(
                "BF16 DeepSpeed optimizer did not enable gradient-overflow "
                f"checking; resolved optimizer value={optimizer_guard!r}"
            )
        if gradient_clipping is None or not np.isclose(
            float(gradient_clipping), self.max_grad_norm, rtol=0.0, atol=1.0e-12
        ):
            raise RuntimeError(
                "DeepSpeed gradient_clipping disagrees with max_grad_norm: "
                f"resolved={gradient_clipping!r} expected={self.max_grad_norm}"
            )
        if self.accelerator.is_main_process:
            logger.info(
                "DeepSpeed numerics contract passed: bf16_overflow_guard=%s "
                "gradient_clipping=%.4f step_owner=deepspeed_backward",
                overflow_guard,
                float(gradient_clipping),
            )

    def _validate_prepared_trainable_state(self) -> None:
        """Audit finite and synchronized trainable weights after DeepSpeed.

        Base checkpoints do not contain newly introduced WARM modules.  This
        check makes rank divergence or an optimizer-wrapper initialization
        fault fail before the first expensive data forward.
        """

        model = self.accelerator.unwrap_model(self.model)
        trainable = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        if not trainable:
            raise RuntimeError("prepared model has no trainable parameters")

        finite_checks: list[torch.Tensor] = []
        probes: list[torch.Tensor] = []
        trainable_numel = 0
        for name, parameter in trainable:
            detached = parameter.detach()
            trainable_numel += int(detached.numel())
            finite_checks.append(torch.isfinite(detached).all())
            flat = detached.reshape(-1)
            if flat.numel() == 0:
                continue
            positions = sorted({0, int(flat.numel() // 2), int(flat.numel() - 1)})
            probes.append(flat[positions].to(dtype=torch.float32))

        # Stack all device checks before synchronizing once.  Calling .item()
        # per tensor is very expensive for a billion-parameter action expert.
        local_checks = torch.stack(finite_checks)
        local_finite = local_checks.all().to(dtype=torch.int32)
        finite_by_rank = self.accelerator.gather(local_finite.reshape(1))
        if not bool(torch.all(finite_by_rank != 0).item()):
            check_values = local_checks.detach().cpu().tolist()
            bad_names = [
                name
                for (name, _), is_finite in zip(trainable, check_values)
                if not is_finite
            ]
            raise FloatingPointError(
                "prepared trainable parameters contain non-finite values: "
                f"local_bad={bad_names[:20]} finite_flags="
                f"{finite_by_rank.detach().cpu().tolist()}"
            )

        if not probes:
            raise RuntimeError("prepared trainable parameters are all empty")
        probe = torch.cat(probes)
        gathered = self.accelerator.gather(probe.unsqueeze(0))
        world_size = int(self.accelerator.num_processes)
        gathered = gathered.reshape(world_size, -1)
        # ``torch.equal`` returns Python bool; keep the comparison explicit so
        # diagnostics report the exact divergent ranks.
        divergent_ranks = [
            rank
            for rank in range(1, world_size)
            if not torch.equal(gathered[0], gathered[rank])
        ]
        if divergent_ranks:
            raise RuntimeError(
                "trainable parameter initialization differs across ranks after "
                f"Accelerate/DeepSpeed prepare; divergent_ranks={divergent_ranks}"
            )
        if self.accelerator.is_main_process:
            logger.info(
                "Prepared trainable-state audit passed: tensors=%d params=%d "
                "world_size=%d finite=true synchronized=true",
                len(trainable),
                trainable_numel,
                world_size,
            )

    def _validate_critical_trainable_state(self, *, stage: str) -> None:
        """Cheap post-update finite audit for newly introduced WARM modules."""

        model = self.accelerator.unwrap_model(self.model)
        critical_fragments = (
            "proprio_encoder",
            "semantic_bridge",
            "retrospective_gist",
            "retrospective_event_adapter",
            "utility_reranker",
            "source_confidence_gate",
            "episode_action_projection",
            "gist_to_text",
            "action_context_to_text",
            "video_layer_adapters",
        )
        critical = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
            and any(fragment in name for fragment in critical_fragments)
        ]
        if not critical:
            return
        checks = torch.stack(
            [torch.isfinite(parameter.detach()).all() for _, parameter in critical]
        )
        local_finite = checks.all().to(dtype=torch.int32).reshape(1)
        finite_by_rank = self.accelerator.gather(local_finite)
        if bool(torch.all(finite_by_rank != 0).item()):
            return
        local_values = checks.detach().cpu().tolist()
        bad_names = [
            name
            for (name, _), finite in zip(critical, local_values)
            if not finite
        ]
        raise FloatingPointError(
            "critical trainable parameters became non-finite after an update: "
            f"stage={stage} rank={self.accelerator.process_index} "
            f"local_bad={bad_names[:32]} "
            f"finite_flags={finite_by_rank.detach().cpu().tolist()}"
        )

    def _record_skipped_update(self, *, reason: str, sample) -> None:
        self._consecutive_nonfinite_gradient_skips += 1
        self.optimizer.zero_grad(set_to_none=True)
        if self.accelerator.is_main_process:
            logger.warning(
                "Skipping optimizer update at global_step=%d: reason=%s "
                "consecutive_skips=%d/%d sample_identity=%s",
                self.global_step,
                reason,
                self._consecutive_nonfinite_gradient_skips,
                self.max_nonfinite_gradient_skips,
                self._sample_identity(sample),
            )
        if (
            self._consecutive_nonfinite_gradient_skips
            >= self.max_nonfinite_gradient_skips
        ):
            raise FloatingPointError(
                "repeated non-finite distributed gradients; DeepSpeed/AMP "
                "blocked the optimizer update before parameter corruption"
            )

    @staticmethod
    def _sample_identity(sample) -> dict[str, object]:
        identity: dict[str, object] = {}
        if not isinstance(sample, dict):
            return identity
        for key in ("idx", "dataset_index", "episode_index", "frame_index"):
            value = sample.get(key)
            if isinstance(value, torch.Tensor):
                flat = value.detach().cpu().reshape(-1)
                identity[key] = flat[:32].tolist()
            elif isinstance(value, (str, int, float, bool)):
                identity[key] = value
        return identity

    def _prepare_warm_training_attestation_context(self):
        """Create immutable provenance from the actual live trainer state."""

        attested_model = self.accelerator.unwrap_model(self.model)
        metadata_fn = getattr(
            attested_model, "training_attestation_metadata", None
        )
        if not callable(metadata_fn):
            return None

        if self.allow_unattested_warm_checkpoints:
            logger.warning(
                "allow_unattested_warm_checkpoints=true: saving WARM debug "
                "checkpoints without formal .training.json attestations"
            )
            return None

        from .models.warm.training_attestation import (
            WarmTrainingRunContext,
            capture_actual_optimizer_facts,
            capture_actual_scheduler_chain,
            capture_training_runtime,
            prepare_formal_resume_lineage,
        )

        resolved_config = OmegaConf.to_container(self.cfg, resolve=True)
        if not isinstance(resolved_config, dict):
            raise TypeError("resolved training config must be a mapping")
        actual_precision = str(self.accelerator.mixed_precision).strip().lower()
        if actual_precision != self.mixed_precision:
            raise ValueError(
                "Accelerate mixed precision differs from the resolved trainer "
                f"config: {actual_precision!r} != {self.mixed_precision!r}"
            )
        repository_root = Path(__file__).resolve().parents[2]
        context = WarmTrainingRunContext.create(
            resolved_config=resolved_config,
            source_metadata=metadata_fn(),
            root_seed=self.seed,
            actual_max_steps=self.max_steps,
            optimizer_facts=capture_actual_optimizer_facts(self.optimizer),
            scheduler_type=self.scheduler_type,
            scheduler_warmup_steps=self.scheduler_warmup_steps,
            scheduler_min_learning_rate=self.scheduler_min_learning_rate,
            scheduler_wrapper_chain=capture_actual_scheduler_chain(
                self.scheduler,
                scheduler_type=self.scheduler_type,
                total_steps=self.max_steps,
                warmup_steps=self.scheduler_warmup_steps,
                minimum_learning_rate=self.scheduler_min_learning_rate,
            ),
            per_device_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            world_size=int(self.accelerator.num_processes),
            mixed_precision=actual_precision,
            training_runtime=capture_training_runtime(self.accelerator),
            repository_root=repository_root,
        )
        if self.resume in (None, "", False):
            return context

        resume_path = Path(str(self.resume)).expanduser().resolve()
        envelope: list[object] = [None]
        if self.accelerator.is_main_process:
            try:
                logger.info(
                    "Hashing and validating formal WARM resume state before "
                    "load (large DeepSpeed states may take several minutes): %s",
                    resume_path,
                )
                envelope[0] = {
                    "ok": True,
                    "lineage": prepare_formal_resume_lineage(
                        resume_path,
                        current_context=context,
                    ),
                }
            except Exception as error:
                envelope[0] = {
                    "ok": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
        if torch.distributed.is_initialized():
            torch.distributed.broadcast_object_list(envelope, src=0)
        result = envelope[0]
        if not isinstance(result, dict) or result.get("ok") is not True:
            if isinstance(result, dict):
                detail = f"{result.get('error_type')}: {result.get('error')}"
            else:
                detail = "rank 0 did not publish a resume-lineage result"
            raise ValueError(f"formal WARM resume validation failed: {detail}")
        lineage = result.get("lineage")
        if not isinstance(lineage, dict):
            raise ValueError("formal WARM resume returned invalid lineage")
        bound = context.with_resume_lineage(lineage)
        logger.info(
            "Bound formal WARM resume lineage: step=%d parent_checkpoint=%s "
            "resume_state=%s",
            int(bound.resume_step or 0),
            str(bound.parent_checkpoint_sha256)[:12],
            str(bound.resume_state_sha256)[:12],
        )
        return bound

    def _init_wandb(self):
        if not self.wandb_enabled or not self.accelerator.is_main_process:
            return
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb logging is enabled in config (`wandb.enabled=true`) but wandb is not installed."
            ) from e

        self.wandb_run = wandb.init(
            entity=self.cfg.wandb.workspace,
            project=self.cfg.wandb.project,
            name=self.cfg.wandb.name,
            group=None if self.cfg.wandb.group in (None, "null", "") else str(self.cfg.wandb.group),
            mode=self.cfg.wandb.mode,
            dir=self.output_dir,
        )
        logger.info(
            "Initialized wandb run: workspace=%s project=%s name=%s",
            self.cfg.wandb.workspace,
            self.cfg.wandb.project,
            self.cfg.wandb.name,
        )

    def _wandb_log(self, payload: dict):
        if self.wandb_run is None:
            return
        self.wandb_run.log(payload, step=self.global_step)

    def _finish_wandb(self):
        if getattr(self, "wandb_run", None) is None:
            return
        self.wandb_run.finish()
        self.wandb_run = None

    def close(self) -> None:
        """Flush process-local logging and tear down distributed state once."""

        if getattr(self, "_closed", False):
            return
        self._closed = True
        try:
            self._finish_wandb()
        finally:
            # Accelerator owns the process group created by Accelerate/
            # DeepSpeed.  Explicit teardown prevents NCCL resources from
            # leaking at normal completion and on Python-level exceptions.
            accelerator = getattr(self, "accelerator", None)
            if accelerator is not None:
                accelerator.end_training()

    def _run_main_process_step(self, label: str, callback):
        """Run a filesystem publication on rank zero and share its outcome.

        A rank-zero exception immediately followed by a barrier leaves every
        other rank waiting forever.  Broadcasting a small result envelope
        makes checkpoint/metadata failures deterministic on all ranks.
        """

        envelope: list[object] = [None]
        if self.accelerator.is_main_process:
            try:
                envelope[0] = {"ok": True, "value": callback()}
            except Exception as error:
                logger.exception("%s failed on rank 0", label)
                envelope[0] = {
                    "ok": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
        if torch.distributed.is_initialized():
            torch.distributed.broadcast_object_list(envelope, src=0)
        result = envelope[0]
        if not isinstance(result, dict) or result.get("ok") is not True:
            if isinstance(result, dict):
                detail = f"{result.get('error_type')}: {result.get('error')}"
            else:
                detail = "rank 0 did not publish a result"
            raise RuntimeError(f"{label} failed on rank 0: {detail}")
        return result.get("value")

    def _build_loader(self, dataset, worker_init_fn=None):
        sampler_kwargs = {
            "dataset": dataset,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "num_processes": self.accelerator.num_processes,
        }
        if self.sampler_mode == "random":
            self.train_sampler = ResumableEpochSampler(**sampler_kwargs)
        elif self.sampler_mode == "rmbench_task_event_balanced":
            self.train_sampler = ResumableTaskEventBalancedSampler(
                **sampler_kwargs,
                event_boost=self.sampler_event_boost,
            )
        else:  # guarded by validate_training_config; retained fail-closed.
            raise ValueError(f"unsupported sampler mode {self.sampler_mode!r}")
        logger.info(
            "Training sampler: mode=%s event_boost=%.3f samples=%d",
            self.sampler_mode,
            self.sampler_event_boost,
            len(dataset),
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
        )

    def _assert_dataset_length_consistent(self, dataset, dataset_name: str):
        if not hasattr(dataset, "__len__"):
            raise TypeError(f"`{dataset_name}` must implement __len__ for rank consistency checks.")

        local_length = len(dataset)
        gathered_lengths = self.accelerator.gather(
            torch.tensor([local_length], device=self.accelerator.device, dtype=torch.int64)
        ).reshape(-1)
        if torch.all(gathered_lengths == gathered_lengths[0]):
            return

        if self.accelerator.is_main_process:
            print(f"[dataset-check] {dataset_name} length mismatch across ranks after initialization:")
            for rank, rank_length in enumerate(gathered_lengths.cpu().tolist()):
                print(f"rank {rank}: {rank_length}")
        self.accelerator.wait_for_everyone()
        raise RuntimeError(
            f"{dataset_name} length mismatch across ranks: {gathered_lengths.cpu().tolist()}"
        )

    def _estimate_total_train_steps(self) -> int:
        if self.max_steps is not None:
            return max(int(self.max_steps), 1)

        if not hasattr(self.train_dataset, "__len__"):
            raise TypeError("`train_dataset` must implement __len__ when `max_steps` is None.")

        num_processes = max(int(self.accelerator.num_processes), 1)
        global_batch_size = max(self.batch_size * num_processes, 1)
        micro_steps_per_epoch = max(ceil(len(self.train_dataset) / global_batch_size), 1)
        opt_steps_per_epoch = max(
            ceil(micro_steps_per_epoch / self.gradient_accumulation_steps),
            1,
        )
        return max(opt_steps_per_epoch * self.num_epochs, 1)

    def _build_scheduler(self, scheduler_type, total_train_steps: int, warmup_steps: int = 0):
        scheduler_type = str(scheduler_type).strip().lower()
        total_train_steps = max(int(total_train_steps), 1)
        warmup_steps = min(max(int(warmup_steps), 0), total_train_steps - 1)

        remaining_steps = max(total_train_steps - warmup_steps, 1)
        if scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=remaining_steps,
                eta_min=self.learning_rate * 0.01,
            )
        elif scheduler_type == "constant":
            main_scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=remaining_steps)
        else:
            raise ValueError(
                f"Unsupported lr_scheduler_type: {scheduler_type}. "
                "Expected one of: ['cosine', 'constant']."
            )

        if warmup_steps <= 0:
            return main_scheduler

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )
    
    def _estimate_eta(self):
        elapsed = max(time.perf_counter() - self.run_start_time, 1e-6)
        done_steps = max(self.global_step - self.run_start_step, 1)
        steps_per_sec = done_steps / elapsed
        remaining_steps = max(self.max_steps - self.global_step, 0)
        eta_seconds = int(remaining_steps / max(steps_per_sec, 1e-9))
        eta_h, eta_rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(eta_rem, 60)
        return f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}", steps_per_sec

    def _resume_or_load_checkpoint(self):
        resume = self.resume
        if not resume:
            return
        resume_path = Path(str(resume))
        if resume_path.is_dir():
            logger.info("Resuming full training state from directory: %s", resume)
            self.load_training_state(str(resume_path))
            return
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        logger.info("Loading weight checkpoint only: %s", resume)
        self.accelerator.unwrap_model(self.model).load_checkpoint(str(resume_path), optimizer=None)
        logger.warning(
            "Loaded .pt weights only; distributed optimizer/scheduler/step "
            "state was not restored."
        )

    def _verify_formal_resume_after_load(self, state_dir: str) -> None:
        """Close the validation/load TOCTOU window for attested resumes."""

        context = self._warm_training_attestation_context
        if context is None or context.resume_step is None:
            return
        if int(self.global_step) != int(context.resume_step):
            raise ValueError(
                "loaded global_step disagrees with the formally attested "
                f"resume step: {self.global_step} != {context.resume_step}"
            )

        from .models.warm.training_attestation import sha256_training_state_tree

        envelope: list[object] = [None]
        if self.accelerator.is_main_process:
            try:
                logger.info(
                    "Rehashing formal WARM resume state after load to close "
                    "the validation/load race: %s",
                    state_dir,
                )
                actual = sha256_training_state_tree(state_dir)
                envelope[0] = {
                    "ok": actual == context.resume_state_sha256,
                    "actual": actual,
                }
            except Exception as error:
                envelope[0] = {
                    "ok": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
        if torch.distributed.is_initialized():
            torch.distributed.broadcast_object_list(envelope, src=0)
        result = envelope[0]
        if not isinstance(result, dict) or result.get("ok") is not True:
            if isinstance(result, dict) and result.get("error"):
                detail = f"{result.get('error_type')}: {result.get('error')}"
            elif isinstance(result, dict):
                detail = (
                    f"state SHA-256 {result.get('actual')} != "
                    f"{context.resume_state_sha256}"
                )
            else:
                detail = "rank 0 did not publish post-load state verification"
            raise ValueError(
                "formal WARM resume state changed during load: " + detail
            )
        logger.info(
            "Verified formal WARM resume after load: step=%d state_sha256=%s",
            self.global_step,
            str(context.resume_state_sha256)[:12],
        )

    def _set_dit_only_train_mode(self):
        logger.info(
            "Applying model trainable-module contract and freezing all other components."
        )
        model = self.accelerator.unwrap_model(self.model)
        self._apply_dit_only_train_mode(model)

    @staticmethod
    def _apply_dit_only_train_mode(model):
        configure = getattr(model, "configure_trainable_modules", None)
        if callable(configure):
            configured = configure()
            if configured is None:
                trainable = [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ]
            else:
                trainable = list(configured)
            if not trainable:
                raise ValueError(
                    "model.configure_trainable_modules() returned no trainable parameters"
                )
            model_parameter_ids = {id(parameter) for parameter in model.parameters()}
            seen: set[int] = set()
            for index, parameter in enumerate(trainable):
                if not isinstance(parameter, torch.nn.Parameter):
                    raise TypeError(
                        "configure_trainable_modules() must return torch Parameters; "
                        f"item {index} has type {type(parameter)}"
                    )
                if id(parameter) not in model_parameter_ids:
                    raise ValueError(
                        "configure_trainable_modules() returned a parameter not owned by model"
                    )
                if id(parameter) in seen:
                    raise ValueError(
                        "configure_trainable_modules() returned duplicate parameters"
                    )
                if not parameter.requires_grad:
                    raise ValueError(
                        "configure_trainable_modules() returned a frozen parameter"
                    )
                seen.add(id(parameter))
            return trainable

        model.eval()
        model.requires_grad_(False)
        model.dit.train()
        model.dit.requires_grad_(True)
        trainable = list(model.dit.parameters())
        proprio_encoder = getattr(model, "proprio_encoder", None)
        if proprio_encoder is not None:
            proprio_encoder.train()
            proprio_encoder.requires_grad_(True)
            trainable.extend(list(proprio_encoder.parameters()))
        return trainable

    @staticmethod
    def _to_batched_eval_sample(sample):
        video = sample["video"]
        prompt = sample["prompt"]
        action = sample.get("action", None)
        proprio = sample.get("proprio", None)
        context = sample.get("context", None)
        context_mask = sample.get("context_mask", None)

        if not isinstance(video, torch.Tensor):
            raise TypeError(
                f"Expected tensor video for evaluation, got {type(video)}. "
                "Evaluation now expects `video` with shape [3,T,H,W] or [B,3,T,H,W]."
            )
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5:
            raise ValueError(f"Expected video shape [3,T,H,W] or [B,3,T,H,W], got {tuple(video.shape)}")
        num_video_frames = video.shape[2]
        if num_video_frames <= 1:
            raise ValueError(f"`sample['video']` must have at least 2 frames for action evaluation, got {num_video_frames}")

        if isinstance(prompt, str):
            prompt = [prompt]
        elif isinstance(prompt, tuple):
            prompt = list(prompt)
        elif not isinstance(prompt, list):
            raise TypeError(f"Expected prompt type str/list[str], got {type(prompt)}")
        if len(prompt) != video.shape[0]:
            raise ValueError(f"Prompt batch mismatch: len(prompt)={len(prompt)} vs video batch={video.shape[0]}")
        
        action_horizon = None
        action = None
        if "action" in sample:
            action = sample["action"]
            if not isinstance(action, torch.Tensor):
                raise TypeError(
                    f"`sample['action']` must be a torch.Tensor, got {type(action)}"
                )
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3:
                raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
            if action.shape[1] % (num_video_frames - 1) != 0:
                raise ValueError(f"`sample['action']` temporal dimension must be divisible by video frames-1={num_video_frames - 1}, got {action.shape[1]}")
            action_horizon = int(action.shape[1])

        proprio = None
        if "proprio" in sample:
            proprio = sample["proprio"]
            if not isinstance(proprio, torch.Tensor):
                raise TypeError(f"`sample['proprio']` must be a torch.Tensor, got {type(proprio)}")
            if proprio.ndim == 2:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")

        if context is not None or context_mask is not None:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must both exist in eval sample.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )

        batched = {
            "video": video,
            "prompt": prompt,
            "action": action,
            "proprio": proprio,
            "context": context,
            "context_mask": context_mask,
            "action_horizon": action_horizon,
        }
        extra_tensor_ranks = {
            "action_is_pad": 1,
            "warm_candidate_mu": 3,
            "warm_candidate_mask": 1,
            "warm_candidate_score": 1,
            "warm_candidate_event_index": 1,
            "warm_oracle_candidate_index": 0,
            "warm_memory_enabled": 0,
            "warm_candidate_context": 2,
            "warm_candidate_effect_pre": 3,
            "warm_candidate_effect_post": 3,
            "warm_candidate_effect_delta": 3,
            "warm_candidate_start_proprio": 2,
            "warm_candidate_gripper": 2,
            "warm_candidate_timing": 2,
            "warm_candidate_support": 1,
            "warm_current_context": 1,
            "warm_current_semantic": 2,
            "warm_future_semantic": 2,
            "warm_target_effect": 2,
            "warm_future_valid": 0,
            "warm_episode_tokens": 2,
            "warm_episode_mask": 1,
            "warm_episode_action_summaries": 2,
            "warm_episode_action_mask": 1,
        }
        for key, unbatched_rank in extra_tensor_ranks.items():
            if key not in sample:
                continue
            value = sample[key]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"`sample[{key!r}]` must be a torch.Tensor")
            if value.ndim == unbatched_rank:
                value = value.unsqueeze(0)
            elif value.ndim != unbatched_rank + 1 or value.shape[0] != 1:
                raise ValueError(
                    f"`sample[{key!r}]` must have unbatched rank {unbatched_rank} "
                    f"or leading singleton batch, got {tuple(value.shape)}"
                )
            batched[key] = value
        if "warm_query_split" in sample:
            split = sample["warm_query_split"]
            if not isinstance(split, str):
                raise TypeError("`sample['warm_query_split']` must be a string")
            batched["warm_query_split"] = split
        for key in ("dataset_index", "episode_index", "frame_index"):
            if key in sample:
                batched[key] = sample[key]
        return batched

    @torch.no_grad()
    def evaluate(self):
        if self.val_dataset is None:
            return None

        model = self.accelerator.unwrap_model(self.model)
        was_train_scope_active = any(
            module.training
            and any(
                parameter.requires_grad
                for parameter in module.parameters(recurse=False)
            )
            for module in model.modules()
        )
        model.eval()

        # eval_index = (self.global_step + self.accelerator.process_index) % len(self.val_dataset)
        rng = torch.Generator(device="cpu").manual_seed(self.global_step + self.accelerator.process_index)
        eval_index = torch.randint(0, len(self.val_dataset), (1,), generator=rng).item()
        sample = self._to_batched_eval_sample(self.val_dataset[eval_index])

        # 1. training loss
        with self.accelerator.autocast():
            val_loss_tensor, _ = model.training_loss(sample)
        val_loss_tensor = val_loss_tensor.detach().float().reshape(1)
        val_loss = float(val_loss_tensor.item())

        if getattr(model, "trainer_evaluation_mode", "full") == "loss_only":
            gathered = self.accelerator.gather_for_metrics(val_loss_tensor)
            result = {
                "val_loss": float(gathered.mean().item()),
                "evaluation_mode": "loss_only",
            }
            if was_train_scope_active:
                self._set_dit_only_train_mode()
            return result
        
        prompt = sample["prompt"][0]
        video0 = sample["video"][0] # Tensor [3, T, H, W] in (-1, 1)
        action = sample["action"][0] if "action" in sample and sample["action"] is not None else None
        proprio = sample["proprio"][0, 0] if "proprio" in sample and sample["proprio"] is not None else None # from [1, T, d] to [d]
        input_image = video0[:, 0].unsqueeze(0)
        _, num_frames, _, _ = video0.shape

        # 2. inference and video saving
        infer_kwargs = {
            "input_image": input_image,
            "num_frames": num_frames,
            "action": action,
            "action_horizon": sample['action_horizon'],
            "proprio": proprio,
            "text_cfg_scale": 1.0,
            "action_cfg_scale": 1.0,
            "num_inference_steps": self.eval_num_inference_steps,
            "seed": 42,
            "tiled": False,
        }
        if sample["context"] is not None:
            infer_kwargs["prompt"] = None
            infer_kwargs["context"] = sample["context"][0]
            infer_kwargs["context_mask"] = sample["context_mask"][0]
        else:
            infer_kwargs["prompt"] = prompt

        pred = model.infer(
            **infer_kwargs,
        )
        
        pred_video = pred["video"]
        pred_action = pred.get("action", None)

        # 3. inference metrics against GT video
        pred_video_tensor = pil_frames_to_video_tensor(pred_video)
        gt_video_tensor = ((video0.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()

        assert pred_video_tensor.shape == gt_video_tensor.shape, (
            "Eval infer prediction/GT shape mismatch: "
            f"pred={tuple(pred_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_rollout_vs_gt = video_psnr(pred=pred_video_tensor, target=gt_video_tensor)
        ssim_rollout_vs_gt = video_ssim(pred=pred_video_tensor, target=gt_video_tensor)

        action_l1 = None
        action_l2 = None
        if action is not None and pred_action is not None:
            if sample["proprio"] is None:
                raise ValueError("Eval sample must contain `proprio` for action denormalization.")
            proprio = sample["proprio"].detach().to(device="cpu", dtype=torch.float32)
            
            processor = self.val_dataset.lerobot_dataset.processor

            denorm_actions = {}
            action_meta = processor.shape_meta["action"]
            state_meta = processor.shape_meta["state"]
            for action_name, raw_action in (("pred", pred_action), ("gt", action)):
                if not isinstance(raw_action, torch.Tensor):
                    raise TypeError(f"{action_name} action must be a torch.Tensor, got {type(raw_action)}")
                if raw_action.ndim == 2:
                    action_btd = raw_action.unsqueeze(0)
                elif raw_action.ndim == 3 and raw_action.shape[0] == 1:
                    action_btd = raw_action
                else:
                    raise ValueError(
                        f"{action_name} action must have shape [T, D] or [1, T, D], got {tuple(raw_action.shape)}"
                    )
                action_btd = action_btd.detach().to(device="cpu", dtype=torch.float32)

                batch = {
                    "action": action_btd,
                    "state": proprio,
                }
                batch = processor.action_state_merger.backward(batch)
                batch = processor.normalizer.backward(batch)
                merged_batch = {
                    "action": {meta["key"]: batch["action"][meta["key"]].squeeze(0) for meta in action_meta},
                    "state": {meta["key"]: batch["state"][meta["key"]].squeeze(0) for meta in state_meta},
                }
                merged_batch = processor.action_state_merger.forward(merged_batch)
                denorm_action = merged_batch["action"].unsqueeze(0)
                if denorm_action.ndim != 3 or denorm_action.shape[0] != 1:
                    raise ValueError(
                        f"Denormalized {action_name} action must have shape [1, T, D], got {tuple(denorm_action.shape)}"
                    )
                denorm_actions[action_name] = denorm_action

            pred_action_denorm = denorm_actions["pred"]
            gt_action_denorm = denorm_actions["gt"]

            if pred_action_denorm.shape != gt_action_denorm.shape:
                raise ValueError(
                    "Predicted action/GT action shape mismatch after denormalization: "
                    f"pred={tuple(pred_action_denorm.shape)} vs gt={tuple(gt_action_denorm.shape)}"
                )
            action_diff = pred_action_denorm - gt_action_denorm
            action_l1 = action_diff.abs().mean().item()
            action_l2 = action_diff.pow(2).mean().item()

        # 4. VAE reconstruction metrics against GT video
        gt_video_batch = video0.unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
        vae_latents = model._encode_video_latents(gt_video_batch, tiled=False)
        vae_recon_video = model._decode_latents(vae_latents, tiled=False)
        vae_video_tensor = pil_frames_to_video_tensor(vae_recon_video)

        assert vae_video_tensor.shape == gt_video_tensor.shape, (
            "Eval VAE reconstruction/GT shape mismatch: "
            f"vae={tuple(vae_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_decode_vs_gt = video_psnr(pred=vae_video_tensor, target=gt_video_tensor)
        ssim_decode_vs_gt = video_ssim(pred=vae_video_tensor, target=gt_video_tensor)

        psnr_rollout_vs_decode = video_psnr(pred=pred_video_tensor, target=vae_video_tensor)
        ssim_rollout_vs_decode = video_ssim(pred=pred_video_tensor, target=vae_video_tensor)

        stitched_video_tensor = torch.cat(
            [pred_video_tensor, vae_video_tensor, gt_video_tensor],
            dim=2,
        ).contiguous()
        stitched_frames = []
        for t in range(stitched_video_tensor.shape[1]):
            frame = (stitched_video_tensor[:, t].permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
            stitched_frames.append(Image.fromarray(frame))

        video_path = os.path.join(
            self.eval_dir,
            f"step_{self.global_step:06d}_rank_{self.accelerator.process_index:03d}.mp4",
        )
        save_mp4(stitched_frames, video_path, fps=8)

        local_metrics = torch.tensor(
            [
                float(val_loss),
                float(psnr_rollout_vs_gt),
                float(ssim_rollout_vs_gt),
                float(psnr_rollout_vs_decode),
                float(ssim_rollout_vs_decode),
                float(psnr_decode_vs_gt),
                float(ssim_decode_vs_gt),
                float(action_l2) if action_l2 is not None else -1.0,
                float(action_l1) if action_l1 is not None else -1.0,
            ],
            device=self.accelerator.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        gathered_metrics = self.accelerator.gather_for_metrics(local_metrics)
        mean_metrics = gathered_metrics[:, :7].mean(dim=0)
        action_l2_mean = gathered_metrics[:, 7].mean().item() if action_l2 is not None else None
        action_l1_mean = gathered_metrics[:, 8].mean().item() if action_l1 is not None else None

        if was_train_scope_active:
            self._set_dit_only_train_mode()

        result = {
            "val_loss": float(mean_metrics[0].item()),
            "psnr_rg": float(mean_metrics[1].item()),
            "ssim_rg": float(mean_metrics[2].item()),
            "psnr_rd": float(mean_metrics[3].item()),
            "ssim_rd": float(mean_metrics[4].item()),
            "psnr_dg": float(mean_metrics[5].item()),
            "ssim_dg": float(mean_metrics[6].item()),
            "video_path": video_path,
        }
        if action_l2_mean is not None:
            result["action_l2"] = float(action_l2_mean)
        if action_l1_mean is not None:
            result["action_l1"] = float(action_l1_mean)
        return result

    def _save_weights_checkpoint(self, step_tag: str):
        model = self.accelerator.unwrap_model(self.model)
        ckpt_path = Path(self.weights_dir) / f"{step_tag}.pt"
        self._last_training_attestation_path = None
        context = self._warm_training_attestation_context
        if context is None:
            model.save_checkpoint(
                str(ckpt_path), optimizer=None, step=self.global_step
            )
            return str(ckpt_path)

        from .models.warm.training_attestation import (
            clean_git_commit,
            publish_training_attestation,
            training_attestation_path,
        )

        # Formal WARM checkpoint bytes are staged and atomically replaced.  If
        # the subsequent attestation cannot be published, neither file remains
        # under its formal name.
        if clean_git_commit(context.repository_root) != context.git_commit:
            raise ValueError(
                "Git state changed after formal WARM training started"
            )
        sidecar_path = training_attestation_path(ckpt_path)
        occupied = [
            path
            for path in (ckpt_path, sidecar_path)
            if path.exists() or path.is_symlink()
        ]
        if occupied:
            raise FileExistsError(
                "formal WARM checkpoint publication never replaces an "
                f"existing weights/attestation path: {occupied}"
            )
        temporary = ckpt_path.parent / f".{ckpt_path.name}.{uuid4().hex}.tmp"
        try:
            model.save_checkpoint(
                str(temporary), optimizer=None, step=self.global_step
            )
            # Windows requires a writable descriptor for fsync.
            with temporary.open("r+b") as handle:
                handle.flush()
                os.fsync(handle.fileno())
            if clean_git_commit(context.repository_root) != context.git_commit:
                raise ValueError(
                    "Git state changed while the WARM checkpoint was saved"
                )
            # Atomic no-replace publication.  os.link fails if another writer
            # claimed this exact step after the preflight above.
            os.link(temporary, ckpt_path)
            temporary.unlink()
            try:
                attestation_path, _ = publish_training_attestation(
                    ckpt_path,
                    context=context,
                    actual_global_step=self.global_step,
                )
            except Exception:
                ckpt_path.unlink(missing_ok=True)
                raise
            self._last_training_attestation_path = str(attestation_path)
        finally:
            temporary.unlink(missing_ok=True)
        return str(ckpt_path)

    def _save_trainer_state(self, state_path: str):
        state_file = os.path.join(state_path, "trainer_state.json")
        payload = {
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "batch_in_epoch": int(self.batch_in_epoch),
            "consecutive_nonfinite_gradient_skips": int(
                self._consecutive_nonfinite_gradient_skips
            ),
        }
        model = self.accelerator.unwrap_model(self.model)
        metadata = getattr(model, "trainer_state_metadata", None)
        if callable(metadata):
            value = metadata()
            if not isinstance(value, dict):
                raise TypeError("model.trainer_state_metadata() must return a dict")
            payload["model_contract"] = value
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def save_checkpoint(self):
        step_tag = f"step_{self.global_step:06d}"

        self.accelerator.wait_for_everyone()
        ckpt_path = self._run_main_process_step(
            "weights/attestation publication",
            lambda: self._save_weights_checkpoint(step_tag=step_tag),
        )

        state_path = os.path.join(self.state_dir, step_tag)
        ensure_dir(state_path)
        self.accelerator.save_state(output_dir=state_path)
        self._run_main_process_step(
            "trainer-state metadata publication",
            lambda: self._save_trainer_state(state_path),
        )

        return {
            "weights_path": ckpt_path,
            "state_path": state_path,
            "training_attestation_path": self._last_training_attestation_path,
        }

    def load_training_state(self, state_dir: str):
        state_file = Path(state_dir) / "trainer_state.json"
        payload = None
        model = self.accelerator.unwrap_model(self.model)
        validate_metadata = getattr(model, "validate_trainer_state_metadata", None)
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if callable(validate_metadata):
                validate_metadata(payload.get("model_contract"))
        elif callable(validate_metadata):
            raise ValueError(
                "model requires trainer-state contract metadata, but "
                f"{state_file} does not exist"
            )

        self.accelerator.load_state(input_dir=state_dir)
        if payload is not None:
            self.global_step = int(payload["global_step"])
            self._consecutive_nonfinite_gradient_skips = int(
                payload.get("consecutive_nonfinite_gradient_skips", 0)
            )

            if "epoch" in payload and "batch_in_epoch" in payload:
                self.epoch = int(payload["epoch"])
                self.batch_in_epoch = int(payload["batch_in_epoch"])
                self.train_sampler.set_epoch_offset(self.epoch)
                self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
                logger.info(
                    "Restored dataloader progress: epoch=%d batch_in_epoch=%d sample_offset=%d",
                    self.epoch,
                    self.batch_in_epoch,
                    self.batch_in_epoch * self.batch_size * self.accelerator.num_processes,
                )
            else:
                self.epoch = 0
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                logger.warning(
                    "State file does not contain `epoch`/`batch_in_epoch`; "
                    "optimizer/scheduler were restored, but dataloader progress resume is skipped."
                )
            self._verify_formal_resume_after_load(state_dir)
            self.accelerator.wait_for_everyone()
            return

        match = re.search(r"step[_-](\d+)$", str(state_dir).rstrip("/"))
        if match:
            self.global_step = int(match.group(1))
        else:
            self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.train_sampler.clear_resume_batch_offset()
        self._verify_formal_resume_after_load(state_dir)
        self.accelerator.wait_for_everyone()
        logger.info("Loaded accelerate training state from %s at step=%d", state_dir, self.global_step)
        logger.warning(
            "State file `%s` is missing; dataloader progress resume is skipped.",
            state_file,
        )

    def train(self):
        self._set_dit_only_train_mode()

        if self.max_steps is None:
            raise ValueError("`max_steps` must be set before entering the while-step training loop.")

        run_target_step = self.max_steps
        if self.run_steps is not None:
            run_target_step = min(
                self.max_steps,
                self.global_step + self.run_steps,
            )
        logger.info(
            "Starting training with scheduler_max_steps=%d run_target_step=%d "
            "run_steps=%s warmup_steps=%d initial_lr=%.3e.",
            self.max_steps,
            run_target_step,
            self.run_steps,
            self.scheduler_warmup_steps,
            float(self.optimizer.param_groups[0]["lr"]),
        )
        if self.global_step >= run_target_step:
            # A completed full-state resume is a valid idempotent invocation.
            # Re-publishing the same formal checkpoint would correctly trip
            # the no-overwrite guard, so return without touching artifacts.
            logger.info(
                "[done] no optimizer steps remain: current_step=%d "
                "run_target_step=%d scheduler_max_steps=%d",
                self.global_step,
                run_target_step,
                self.max_steps,
            )
            return
        data_iter = iter(self.train_loader)
        self.run_start_step = self.global_step
        self.run_start_time = time.perf_counter()

        while self.global_step < run_target_step:
            try:
                sample = next(data_iter)
                self.batch_in_epoch += 1
            except StopIteration:
                self.epoch += 1
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                data_iter = iter(self.train_loader)
                continue

            with self.accelerator.accumulate(self.model):
                train_model = self.model if hasattr(self.model, "training_loss") else self.accelerator.unwrap_model(self.model)

                try:
                    with self.accelerator.autocast():
                        loss, loss_dict = train_model.training_loss(sample)
                except Exception:
                    logger.exception(
                        "Training forward failed before backward: rank=%d "
                        "global_step=%d epoch=%d batch_in_epoch=%d sample_identity=%s",
                        int(self.accelerator.process_index),
                        self.global_step,
                        self.epoch,
                        self.batch_in_epoch,
                        self._sample_identity(sample),
                    )
                    raise
                if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
                    raise TypeError("training_loss must return a scalar loss tensor")
                if not bool(torch.isfinite(loss.detach()).item()):
                    raise FloatingPointError(
                        "non-finite loss blocked before backward: "
                        f"rank={self.accelerator.process_index} "
                        f"global_step={self.global_step} "
                        f"sample_identity={self._sample_identity(sample)} "
                        f"loss_dict={loss_dict}"
                    )
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    grad_norm_tensor = torch.as_tensor(
                        grad_norm,
                        device=loss.device,
                        dtype=torch.float32,
                    ).detach().reshape(1)
                    gathered_grad_norms = self.accelerator.gather(
                        grad_norm_tensor
                    ).detach()
                    finite_grad_flags = torch.isfinite(gathered_grad_norms)
                    if self._is_deepspeed():
                        # In Accelerate DeepSpeed mode engine.step() already
                        # ran inside accelerator.backward().  The BF16
                        # overflow guard is the only pre-update protection.
                        local_skipped = torch.tensor(
                            [int(self.accelerator.optimizer_step_was_skipped)],
                            device=loss.device,
                            dtype=torch.int32,
                        )
                        skipped_by_rank = self.accelerator.gather(local_skipped)
                        if not bool(
                            torch.all(skipped_by_rank == skipped_by_rank[0]).item()
                        ):
                            raise RuntimeError(
                                "DeepSpeed overflow decision differs across ranks: "
                                f"{skipped_by_rank.detach().cpu().tolist()}"
                            )
                        if bool(skipped_by_rank[0].item()):
                            self._record_skipped_update(
                                reason="deepspeed_bf16_gradient_overflow",
                                sample=sample,
                            )
                            continue
                        if not bool(torch.all(finite_grad_flags).item()):
                            raise FloatingPointError(
                                "DeepSpeed applied an update despite a non-finite "
                                "global gradient norm; parameters may be corrupt: "
                                f"norms={gathered_grad_norms.cpu().tolist()}"
                            )
                        self._validate_critical_trainable_state(
                            stage=f"global_step_{self.global_step + 1}"
                        )
                        self.scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)
                    elif not bool(torch.all(finite_grad_flags).item()):
                        self._record_skipped_update(
                            reason=(
                                "nonfinite_gradient_norms="
                                f"{gathered_grad_norms.cpu().tolist()}"
                            ),
                            sample=sample,
                        )
                        continue
                    else:
                        self.optimizer.step()
                        local_skipped = torch.tensor(
                            [int(self.accelerator.optimizer_step_was_skipped)],
                            device=loss.device,
                            dtype=torch.int32,
                        )
                        skipped_by_rank = self.accelerator.gather(local_skipped)
                        if not bool(
                            torch.all(skipped_by_rank == skipped_by_rank[0]).item()
                        ):
                            raise RuntimeError(
                                "AMP overflow decision differs across ranks: "
                                f"{skipped_by_rank.detach().cpu().tolist()}"
                            )
                        if bool(skipped_by_rank[0].item()):
                            self._record_skipped_update(
                                reason="amp_gradient_overflow",
                                sample=sample,
                            )
                            continue
                        self.scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        self._validate_critical_trainable_state(
                            stage=f"global_step_{self.global_step + 1}"
                        )

                    self._consecutive_nonfinite_gradient_skips = 0
                    self.global_step += 1
                    checkpoint_saved_this_step = None
                    global_loss = float(
                        self.accelerator.gather(loss.detach().float().reshape(1)).mean().item()
                    )
                    global_loss_metrics = {}
                    for key, value in loss_dict.items():
                        metric_tensor = torch.tensor(float(value), device=loss.device, dtype=torch.float32).reshape(1)
                        global_loss_metrics[key] = float(
                            self.accelerator.gather(metric_tensor).mean().item()
                        )
                    global_grad_norm = float(gathered_grad_norms.mean().item())

                    current_lr = float(self.optimizer.param_groups[0]["lr"])

                    if self.log_every > 0 and self.global_step % self.log_every == 0 and self.accelerator.is_main_process:
                        eta_str, steps_per_sec = self._estimate_eta()
                        description = "[train] epoch=%d step=%d/%d loss=%.4f " % (
                            self.epoch,
                            self.global_step,
                            self.max_steps,
                            global_loss,
                        )
                        if global_loss_metrics:
                            detail_str = " ".join([f"{k}={v:.4f}" for k, v in sorted(global_loss_metrics.items())])
                            description += detail_str + " "
                        description += "grad_norm=%.4f lr=%.2e speed=%.2f step/s, %.2f samples/s eta=%s" % (
                            global_grad_norm,
                            current_lr,
                            steps_per_sec,
                            steps_per_sec
                            * self.batch_size
                            * self.accelerator.num_processes
                            * self.gradient_accumulation_steps,
                            eta_str,
                        )
                        logger.info(description)

                        wandb_payload = {
                            "train/loss": global_loss,
                            "train/grad_norm": global_grad_norm,
                            "train/lr": current_lr,
                            "performance/steps_per_sec": steps_per_sec,
                            "performance/samples_per_sec": (
                                steps_per_sec
                                * self.batch_size
                                * self.accelerator.num_processes
                                * self.gradient_accumulation_steps
                            ),
                        }
                        for key, value in global_loss_metrics.items():
                            wandb_payload[f"train/{key}"] = value
                        self._wandb_log(wandb_payload)

                    if (
                        self.eval_every > 0
                        and self.val_dataset is not None
                        and self.global_step % self.eval_every == 0
                    ):
                        metrics = self.evaluate()
                        self.accelerator.wait_for_everyone()
                        if metrics is not None and self.accelerator.is_main_process:
                            if metrics.get("evaluation_mode") == "loss_only":
                                description = "[eval] step=%d val_loss=%.4f mode=loss_only" % (
                                    self.global_step,
                                    metrics["val_loss"],
                                )
                                eval_payload = {
                                    "eval/val_loss": float(metrics["val_loss"]),
                                }
                            else:
                                description = "[eval] step=%d val_loss=%.4f infer_psnr=%.4f infer_ssim=%.4f" % (
                                    self.global_step,
                                    metrics["val_loss"],
                                    metrics["psnr_rd"],
                                    metrics["ssim_rd"],
                                )
                                eval_payload = {
                                    "eval/val_loss": float(metrics["val_loss"]),
                                    "eval/psnr_rg": float(metrics["psnr_rg"]),
                                    "eval/ssim_rg": float(metrics["ssim_rg"]),
                                    "eval/psnr_rd": float(metrics["psnr_rd"]),
                                    "eval/ssim_rd": float(metrics["ssim_rd"]),
                                    "eval/psnr_dg": float(metrics["psnr_dg"]),
                                    "eval/ssim_dg": float(metrics["ssim_dg"]),
                                }
                            if "action_l2" in metrics:
                                description += " action_l2=%.4f" % metrics["action_l2"]
                            if "action_l1" in metrics:
                                description += " action_l1=%.4f" % metrics["action_l1"]
                            logger.info(description)
                            if "action_l2" in metrics:
                                eval_payload["eval/action_l2"] = float(metrics["action_l2"])
                            if "action_l1" in metrics:
                                eval_payload["eval/action_l1"] = float(metrics["action_l1"])
                            self._wandb_log(eval_payload)

                    if self.save_every > 0 and self.global_step % self.save_every == 0:
                        ckpt_info = self.save_checkpoint()
                        checkpoint_saved_this_step = ckpt_info
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[ckpt] step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )

                    if self.global_step >= run_target_step:
                        ckpt_info = (
                            checkpoint_saved_this_step
                            if checkpoint_saved_this_step is not None
                            else self.save_checkpoint()
                        )
                        if self.accelerator.is_main_process:
                            reason = (
                                "max_steps reached"
                                if self.global_step >= self.max_steps
                                else "run_steps reached"
                            )
                            logger.info(
                                "[done] %s step=%d scheduler_max_steps=%d "
                                "weights=%s state=%s",
                                reason,
                                self.global_step,
                                self.max_steps,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )
                        return

        ckpt_info = self.save_checkpoint()
        if self.accelerator.is_main_process:
            logger.info(
                "[done] training finished step=%d weights=%s state=%s",
                self.global_step,
                ckpt_info["weights_path"],
                ckpt_info["state_path"],
            )
        
