"""M2 source-only WARM model on the base FastWAM action fast path."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import torch

from fastwam.models.wan22.fastwam import FastWAM

from .source_contract import WarmSourceRunContract
from .source_transport import (
    ActionSourceContext,
    SourcePolicy,
    SourceTransportError,
    select_source_components,
)


WARM_SOURCE_CHECKPOINT_SCHEMA = "warm.source-only-checkpoint"
WARM_SOURCE_CHECKPOINT_VERSION = 1

WARM_CANDIDATE_MEANS = "warm_candidate_mu"
WARM_CANDIDATE_MASK = "warm_candidate_mask"
WARM_ORACLE_CANDIDATE_INDEX = "warm_oracle_candidate_index"
WARM_MEMORY_ENABLED = "warm_memory_enabled"


class WarmSourceFastWAM(FastWAM):
    """Action-only FastWAM using fixed retrieved stochastic sources.

    This class implements M2 mechanism identification only.  It has no
    consequence encoder, learned candidate scorer, action context injection, or
    learned null gate.  Those belong to later milestones and cannot silently
    enter source-only comparisons.
    """

    trainer_evaluation_mode = "loss_only"

    @classmethod
    def from_wan22_pretrained(
        cls,
        *args,
        warm_source_policy: SourcePolicy = "fixed_context_top1",
        memory_sigma: float = 0.2,
        warm_run_contract: WarmSourceRunContract | Mapping[str, Any] | None = None,
        **kwargs,
    ) -> "WarmSourceFastWAM":
        model = super().from_wan22_pretrained(*args, **kwargs)
        if not isinstance(model, cls):
            raise TypeError("FastWAM factory did not preserve the WARM subclass")
        model.configure_warm_source(
            policy=warm_source_policy,
            memory_sigma=memory_sigma,
            run_contract=warm_run_contract,
        )
        return model

    def configure_warm_source(
        self,
        *,
        policy: SourcePolicy,
        memory_sigma: float,
        run_contract: WarmSourceRunContract | Mapping[str, Any] | None,
    ) -> None:
        if policy not in (
            "gaussian_null",
            "fixed_context_top1",
            "oracle_action_top1",
        ):
            raise SourceTransportError(f"unsupported source policy {policy!r}")
        if isinstance(memory_sigma, bool) or not isinstance(
            memory_sigma, (int, float)
        ):
            raise TypeError("memory_sigma must be a finite positive number")
        sigma = float(memory_sigma)
        if not math.isfinite(sigma) or sigma <= 0.0:
            raise SourceTransportError(
                "memory_sigma must be a finite positive number"
            )
        if run_contract is None:
            contract = None
        elif isinstance(run_contract, WarmSourceRunContract):
            contract = run_contract
        elif isinstance(run_contract, Mapping):
            contract = WarmSourceRunContract.from_dict(run_contract)
        else:
            raise TypeError(
                "warm_run_contract must be WarmSourceRunContract, mapping, or None"
            )
        if policy != "gaussian_null" and contract is None:
            raise SourceTransportError(
                f"{policy} requires a complete WarmSourceRunContract"
            )
        if contract is not None and contract.action_dim != int(
            self.action_expert.action_dim
        ):
            raise SourceTransportError(
                "run-contract action_dim does not match Action Expert: "
                f"{contract.action_dim} != {int(self.action_expert.action_dim)}"
            )

        self.warm_source_policy: SourcePolicy = policy
        self.warm_memory_sigma = sigma
        self.warm_run_contract = contract

    def load_checkpoint(self, path, optimizer=None):
        """Strictly restore a complete, contract-bound WARM checkpoint.

        Unlike the backward-compatible FastWAM loader, this public entry point
        deliberately exposes no switches that could weaken model strictness,
        proprio matching, legacy-checkpoint rejection, or metadata preflight.
        The separate :meth:`load_base_checkpoint` method is the only supported
        way to initialize WARM from a non-WARM baseline.
        """

        return super().load_checkpoint(
            path,
            optimizer=optimizer,
            load_extra_state=True,
            strict_model_state=True,
            exact_proprio_state=True,
            allow_legacy_dit=False,
        )

    def load_base_checkpoint(self, path: str | Path) -> dict[str, Any]:
        """Load an immutable FastWAM baseline, never a prior WARM checkpoint."""

        self._require_warm_configuration()
        if self.warm_run_contract is None:
            raise SourceTransportError(
                "base checkpoint loading requires a WarmSourceRunContract"
            )
        from fastwam.memory.manifest import sha256_file

        checkpoint_path = Path(path).expanduser().resolve()
        actual_sha256 = sha256_file(checkpoint_path)
        expected_sha256 = self.warm_run_contract.base_checkpoint_sha256
        if actual_sha256 != expected_sha256:
            raise SourceTransportError(
                "base checkpoint SHA256 changed or does not match the run "
                f"contract: {actual_sha256} != {expected_sha256}"
            )
        return super().load_checkpoint(
            checkpoint_path,
            load_extra_state=False,
            forbid_extra_keys=("warm_source",),
            strict_model_state=True,
            exact_proprio_state=True,
            allow_legacy_dit=False,
        )

    def _require_warm_configuration(self) -> None:
        if not hasattr(self, "warm_source_policy") or not hasattr(
            self, "warm_memory_sigma"
        ):
            raise RuntimeError("WarmSourceFastWAM has not been configured")

    def validate_training_dataset(self, dataset: object) -> None:
        """Prove that DataLoader artifacts are the ones bound by the model."""

        self._require_warm_configuration()
        contract = self.warm_run_contract
        if contract is None:
            if self.warm_source_policy != "gaussian_null":
                raise SourceTransportError(
                    "non-null source policy has no WarmSourceRunContract"
                )
            return

        from fastwam.memory.manifest import sha256_canonical_json
        from fastwam.memory.runtime_candidates import RuntimeCandidateResolver

        resolver = getattr(dataset, "resolver", None)
        if not isinstance(resolver, RuntimeCandidateResolver):
            raise SourceTransportError(
                "contract-bound WARM training requires "
                "RuntimeCandidateDatasetAdapter"
            )
        actual = {
            "bank_manifest_sha256": resolver.bank_manifest_sha256,
            "bank_content_sha256": resolver.bank_content_sha256,
            "candidate_manifest_sha256": resolver.candidate_manifest_sha256,
            "query_corpus_sha256": resolver.query_corpus_sha256,
            "catalog_sha256": resolver.query_catalog_sha256,
            "audit_sha256": resolver.query_audit_sha256,
            "normalization_stats_sha256": (
                resolver.action_space.normalization_stats_sha256
            ),
            "action_space_contract_sha256": sha256_canonical_json(
                resolver.action_space.to_dict()
            ),
            "query_split": resolver.query_split,
            "global_sample_stride": resolver.query_stride,
            "action_horizon": resolver.action_horizon,
            "action_dim": resolver.action_space.action_dim,
        }
        mismatches = {
            field: (getattr(contract, field), value)
            for field, value in actual.items()
            if getattr(contract, field) != value
        }
        if mismatches:
            details = ", ".join(
                f"{field}: contract={expected!r}, dataset={observed!r}"
                for field, (expected, observed) in sorted(mismatches.items())
            )
            raise SourceTransportError(
                "training dataset does not match WarmSourceRunContract; "
                + details
            )

    def configure_trainable_modules(self):
        """Freeze world modeling and train only Action Expert/proprio projection."""

        self._require_warm_configuration()
        self.eval()
        self.requires_grad_(False)
        self.action_expert.train()
        self.action_expert.requires_grad_(True)
        if self.proprio_encoder is not None:
            self.proprio_encoder.train()
            self.proprio_encoder.requires_grad_(True)
        return tuple(
            parameter for parameter in self.parameters() if parameter.requires_grad
        )

    def _source_context_from_batch(
        self,
        sample: Mapping[str, Any],
    ) -> ActionSourceContext:
        self._require_warm_configuration()
        if not isinstance(sample, Mapping):
            raise TypeError("sample must be a mapping")
        action = sample.get("action")
        if not isinstance(action, torch.Tensor) or action.ndim != 3:
            raise ValueError("sample['action'] must be a [B,H,D] torch tensor")
        batch_size = int(action.shape[0])
        if (
            self.warm_run_contract is not None
            and self.warm_run_contract.query_split != "train"
        ):
            raise SourceTransportError(
                "training requires a run contract bound to a train candidate cache"
            )
        if self.warm_run_contract is not None and tuple(action.shape[1:]) != (
            self.warm_run_contract.action_horizon,
            self.warm_run_contract.action_dim,
        ):
            raise SourceTransportError(
                "training action shape does not match WarmSourceRunContract: "
                f"{tuple(action.shape[1:])} != "
                f"{(self.warm_run_contract.action_horizon, self.warm_run_contract.action_dim)}"
            )

        if self.warm_source_policy == "gaussian_null":
            return ActionSourceContext(
                component_indices=torch.zeros(
                    (batch_size,), dtype=torch.long, device=self.device
                ),
                candidate_means=None,
                candidate_valid_mask=None,
            )

        means = sample.get(WARM_CANDIDATE_MEANS)
        valid = sample.get(WARM_CANDIDATE_MASK)
        if not isinstance(means, torch.Tensor) or not isinstance(
            valid, torch.Tensor
        ):
            raise ValueError(
                f"{self.warm_source_policy} requires {WARM_CANDIDATE_MEANS!r} "
                f"and {WARM_CANDIDATE_MASK!r} tensors"
            )
        means = means.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        valid = valid.to(device=self.device, dtype=torch.bool, non_blocking=True)

        memory_enabled = sample.get(WARM_MEMORY_ENABLED)
        if memory_enabled is not None:
            if not isinstance(memory_enabled, torch.Tensor):
                raise TypeError(f"{WARM_MEMORY_ENABLED} must be a torch tensor")
            memory_enabled = memory_enabled.to(
                device=self.device, dtype=torch.bool, non_blocking=True
            )

        oracle = sample.get(WARM_ORACLE_CANDIDATE_INDEX)
        if oracle is not None:
            if not isinstance(oracle, torch.Tensor):
                raise TypeError(
                    f"{WARM_ORACLE_CANDIDATE_INDEX} must be a torch tensor"
                )
            oracle = oracle.to(device=self.device, dtype=torch.long, non_blocking=True)

        components = select_source_components(
            valid,
            policy=self.warm_source_policy,
            phase="train",
            oracle_candidate_indices=oracle,
            memory_enabled_mask=memory_enabled,
        )
        return ActionSourceContext(
            component_indices=components,
            candidate_means=means,
            candidate_valid_mask=valid,
        )

    def training_loss(self, sample, tiled: bool = False):
        source_context = self._source_context_from_batch(sample)
        return self.training_loss_action_only(
            sample,
            action_source_context=source_context,
            memory_sigma=self.warm_memory_sigma,
            tiled=tiled,
        )

    def build_inference_source_context(
        self,
        *,
        candidate_means: torch.Tensor | None = None,
        candidate_valid_mask: torch.Tensor | None = None,
        memory_enabled_mask: torch.Tensor | None = None,
        batch_size: int = 1,
    ) -> ActionSourceContext:
        """Build a deployment-safe context; oracle policy is rejected here."""

        self._require_warm_configuration()
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be a positive integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.warm_source_policy == "oracle_action_top1":
            raise SourceTransportError(
                "oracle_action_top1 is forbidden for inference/rollout"
            )
        if self.warm_source_policy == "gaussian_null":
            return ActionSourceContext(
                component_indices=torch.zeros(
                    (batch_size,), dtype=torch.long, device=self.device
                ),
                candidate_means=None,
                candidate_valid_mask=None,
            )
        if candidate_means is None or candidate_valid_mask is None:
            raise SourceTransportError(
                "fixed_context_top1 inference requires candidate means and mask"
            )
        means = candidate_means.to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True
        )
        valid = candidate_valid_mask.to(
            device=self.device, dtype=torch.bool, non_blocking=True
        )
        if means.ndim != 4 or valid.ndim != 2:
            raise SourceTransportError(
                "inference candidates must have shapes [B,K,H,D] and [B,K]"
            )
        if means.shape[:2] != valid.shape or means.shape[0] != batch_size:
            raise SourceTransportError(
                "inference candidate batch/rank dimensions do not match"
            )
        if self.warm_run_contract is not None and (
            means.ndim != 4
            or tuple(means.shape[2:])
            != (
                self.warm_run_contract.action_horizon,
                self.warm_run_contract.action_dim,
            )
        ):
            raise SourceTransportError(
                "inference candidate action shape does not match WarmSourceRunContract"
            )
        enabled = None
        if memory_enabled_mask is not None:
            enabled = memory_enabled_mask.to(
                device=self.device, dtype=torch.bool, non_blocking=True
            )
        components = select_source_components(
            valid,
            policy="fixed_context_top1",
            phase="infer",
            memory_enabled_mask=enabled,
        )
        return ActionSourceContext(
            component_indices=components,
            candidate_means=means,
            candidate_valid_mask=valid,
        )

    @torch.no_grad()
    def infer_action(
        self,
        *args,
        candidate_means: torch.Tensor | None = None,
        candidate_valid_mask: torch.Tensor | None = None,
        memory_enabled_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        """Run the action-only path using only policy-resolved raw candidates.

        A caller cannot provide an ``ActionSourceContext`` because that object
        already contains component indices and could therefore bypass the
        checkpoint-bound source policy.  Fixed-source deployment must provide
        the raw rank-ordered candidate tensors returned by online retrieval.
        """

        self._require_warm_configuration()
        if "action_source_context" in kwargs:
            raise SourceTransportError(
                "prebuilt ActionSourceContext is forbidden for WARM inference; "
                "provide raw candidate_means/candidate_valid_mask instead"
            )
        if self.warm_source_policy == "oracle_action_top1":
            raise SourceTransportError(
                "oracle_action_top1 checkpoints cannot run infer_action"
            )
        if self.warm_source_policy == "gaussian_null":
            # Deliberately do not inspect caller candidate payloads.  The null
            # checkpoint always reconstructs the explicit Gaussian component.
            action_source_context = self.build_inference_source_context(batch_size=1)
        else:
            if candidate_means is None or candidate_valid_mask is None:
                raise SourceTransportError(
                    "fixed_context_top1 infer_action requires raw candidate "
                    "means and mask from runtime retrieval"
                )
            if not isinstance(candidate_means, torch.Tensor) or not isinstance(
                candidate_valid_mask, torch.Tensor
            ):
                raise TypeError(
                    "candidate_means and candidate_valid_mask must be torch tensors"
                )
            if candidate_means.ndim == 3:
                candidate_means = candidate_means.unsqueeze(0)
            if candidate_valid_mask.ndim == 1:
                candidate_valid_mask = candidate_valid_mask.unsqueeze(0)
            if candidate_means.ndim != 4 or candidate_means.shape[0] != 1:
                raise SourceTransportError(
                    "infer_action candidate_means must have shape [K,H,D] or "
                    "[1,K,H,D]"
                )
            if (
                candidate_valid_mask.ndim != 2
                or candidate_valid_mask.shape[0] != 1
            ):
                raise SourceTransportError(
                    "infer_action candidate_valid_mask must have shape [K] or [1,K]"
                )
            if memory_enabled_mask is not None:
                if not isinstance(memory_enabled_mask, torch.Tensor):
                    raise TypeError("memory_enabled_mask must be a torch tensor")
                if memory_enabled_mask.ndim == 0:
                    memory_enabled_mask = memory_enabled_mask.reshape(1)
                if memory_enabled_mask.shape != (1,):
                    raise SourceTransportError(
                        "infer_action memory_enabled_mask must be scalar or shape [1]"
                    )
            action_source_context = self.build_inference_source_context(
                candidate_means=candidate_means,
                candidate_valid_mask=candidate_valid_mask,
                memory_enabled_mask=memory_enabled_mask,
                batch_size=1,
            )
        kwargs["action_source_context"] = action_source_context
        kwargs["memory_sigma"] = self.warm_memory_sigma
        return super().infer_action(*args, **kwargs)

    @torch.no_grad()
    def infer_joint(self, *args, **kwargs):
        """Forbid joint paths that cannot consume the fixed retrieval payload."""

        self._require_warm_configuration()
        if self.warm_source_policy == "oracle_action_top1":
            raise SourceTransportError(
                "oracle_action_top1 checkpoints cannot run infer_joint"
            )
        if self.warm_source_policy == "fixed_context_top1":
            raise SourceTransportError(
                "fixed_context_top1 joint inference is unavailable until the "
                "online retrieval bridge is implemented; use infer_action with "
                "raw retrieved candidates"
            )
        return super().infer_joint(*args, **kwargs)

    @torch.no_grad()
    def infer(self, *args, **kwargs):
        """Keep the generic video+action API from bypassing source policy."""

        self._require_warm_configuration()
        if self.warm_source_policy == "oracle_action_top1":
            raise SourceTransportError(
                "oracle_action_top1 checkpoints cannot run infer"
            )
        if self.warm_source_policy == "fixed_context_top1":
            raise SourceTransportError(
                "fixed_context_top1 generic inference is unavailable until the "
                "online retrieval bridge is implemented; use infer_action with "
                "raw retrieved candidates"
            )
        return super().infer(*args, **kwargs)

    def _checkpoint_extra_state(self) -> dict[str, Any]:
        self._require_warm_configuration()
        contract = (
            None
            if self.warm_run_contract is None
            else self.warm_run_contract.to_dict()
        )
        contract_sha256 = (
            None if self.warm_run_contract is None else self.warm_run_contract.sha256
        )
        return {
            "warm_source": {
                "schema": WARM_SOURCE_CHECKPOINT_SCHEMA,
                "version": WARM_SOURCE_CHECKPOINT_VERSION,
                "policy": self.warm_source_policy,
                "memory_sigma": self.warm_memory_sigma,
                "run_contract": contract,
                "run_contract_sha256": contract_sha256,
            }
        }

    def trainer_state_metadata(self) -> dict[str, Any]:
        """Bind Accelerate resume state to the same immutable artifacts."""

        return dict(self._checkpoint_extra_state()["warm_source"])

    def validate_trainer_state_metadata(self, value: object) -> None:
        if not isinstance(value, Mapping):
            raise ValueError(
                "WARM trainer state is missing its model_contract metadata"
            )
        self._validate_warm_source_state(value)

    def _validate_warm_source_state(self, value: Mapping[str, Any]) -> None:
        self._require_warm_configuration()
        expected_fields = {
            "schema",
            "version",
            "policy",
            "memory_sigma",
            "run_contract",
            "run_contract_sha256",
        }
        if set(value) != expected_fields:
            raise ValueError(
                "invalid warm_source checkpoint fields; "
                f"missing={sorted(expected_fields - set(value))}, "
                f"extra={sorted(set(value) - expected_fields)}"
            )
        if value["schema"] != WARM_SOURCE_CHECKPOINT_SCHEMA or value[
            "version"
        ] != WARM_SOURCE_CHECKPOINT_VERSION:
            raise ValueError("unsupported warm_source checkpoint schema/version")
        if value["policy"] != self.warm_source_policy:
            raise ValueError(
                "checkpoint source policy does not match the configured model"
            )
        if float(value["memory_sigma"]) != self.warm_memory_sigma:
            raise ValueError(
                "checkpoint memory_sigma does not match the configured model"
            )
        loaded_contract_value = value["run_contract"]
        loaded_contract = (
            None
            if loaded_contract_value is None
            else WarmSourceRunContract.from_dict(loaded_contract_value)
        )
        loaded_hash = None if loaded_contract is None else loaded_contract.sha256
        if value["run_contract_sha256"] != loaded_hash:
            raise ValueError("checkpoint run_contract_sha256 is invalid")
        configured = self.warm_run_contract
        configured_hash = None if configured is None else configured.sha256
        if loaded_hash != configured_hash:
            raise ValueError(
                "checkpoint run contract does not match the configured artifacts"
            )

    def _preflight_checkpoint_extra_state(self, payload: dict[str, Any]) -> None:
        self._require_warm_configuration()
        value = payload.get("warm_source")
        if not isinstance(value, Mapping):
            raise ValueError("WARM checkpoint is missing the warm_source state")
        self._validate_warm_source_state(value)

    def _load_checkpoint_extra_state(self, payload: dict[str, Any]) -> None:
        # All WARM state is immutable metadata and was validated before any
        # parameters were loaded.  There is no mutable post-load state.
        del payload


__all__ = [
    "WARM_CANDIDATE_MASK",
    "WARM_CANDIDATE_MEANS",
    "WARM_MEMORY_ENABLED",
    "WARM_ORACLE_CANDIDATE_INDEX",
    "WARM_SOURCE_CHECKPOINT_SCHEMA",
    "WARM_SOURCE_CHECKPOINT_VERSION",
    "WarmSourceFastWAM",
]
