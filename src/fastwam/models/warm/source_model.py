"""M2 source-only WARM model on the base FastWAM action fast path."""

from __future__ import annotations

import json
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
WARM_SOURCE_CHECKPOINT_VERSION = 2

WARM_CANDIDATE_MEANS = "warm_candidate_mu"
WARM_CANDIDATE_MASK = "warm_candidate_mask"
WARM_ORACLE_CANDIDATE_INDEX = "warm_oracle_candidate_index"
WARM_MEMORY_ENABLED = "warm_memory_enabled"
WARM_QUERY_SPLIT = "warm_query_split"

_FORBIDDEN_RAW_INFERENCE_FIELDS = frozenset(
    {
        "action_source_context",
        "candidate_means",
        "candidate_valid_mask",
        "memory_enabled_mask",
    }
)


def _coerce_run_contract(
    value: WarmSourceRunContract | Mapping[str, Any] | None,
    *,
    field: str,
) -> WarmSourceRunContract | None:
    if value is None:
        return None
    if isinstance(value, WarmSourceRunContract):
        return value
    if isinstance(value, Mapping):
        return WarmSourceRunContract.from_dict(value)
    raise TypeError(f"{field} must be WarmSourceRunContract, mapping, or None")


def _json_safe(value: object, *, field: str) -> object:
    """Return a detached JSON value or fail before rollout telemetry escapes."""

    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise SourceTransportError(
            f"{field} must contain only JSON-safe finite values"
        ) from error
    return json.loads(encoded)


def _event_id_json(value: Any | None) -> dict[str, object] | None:
    if value is None:
        return None
    return {
        "dataset_id": value.dataset_id,
        "dataset_index": int(value.dataset_index),
        "episode_index": int(value.episode_index),
        "start_frame": int(value.start_frame),
    }


def _fixed_retrieval_telemetry(online_step: Any) -> dict[str, object]:
    """Emit the compact factual evidence needed to audit one retrieval."""

    query_id = online_step.query_id
    return {
        "query_id": {
            "dataset_id": query_id.dataset_id,
            "dataset_index": int(query_id.dataset_index),
            "episode_index": int(query_id.episode_index),
            "frame_index": int(query_id.frame_index),
        },
        "online_contract_sha256": online_step.online_contract_sha256,
        "training_run_contract_sha256": (
            online_step.training_run_contract_sha256
        ),
        "validation_run_contract_sha256": (
            online_step.validation_run_contract_sha256
        ),
        "bank_manifest_sha256": online_step.bank_manifest_sha256,
        "bank_content_sha256": online_step.bank_content_sha256,
        "step_sha256": online_step.step_sha256,
        "prompt_sha256": online_step.prompt_sha256,
        "proprio_sha256": online_step.proprio_sha256,
        "model_input_sha256": online_step.model_input_sha256,
        "ranked_event_ids": [
            _event_id_json(value) for value in online_step.event_ids
        ],
        "bank_rows": [int(value) for value in online_step.bank_rows.tolist()],
        "cosine_scores": [
            float(value) for value in online_step.cosine_scores.tolist()
        ],
        "candidate_valid_mask": [
            bool(value) for value in online_step.candidate_valid_mask.tolist()
        ],
        "latency_s": dict(online_step.telemetry),
    }


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
        warm_validation_run_contract: (
            WarmSourceRunContract | Mapping[str, Any] | None
        ) = None,
        **kwargs,
    ) -> "WarmSourceFastWAM":
        model = super().from_wan22_pretrained(*args, **kwargs)
        if not isinstance(model, cls):
            raise TypeError("FastWAM factory did not preserve the WARM subclass")
        model.configure_warm_source(
            policy=warm_source_policy,
            memory_sigma=memory_sigma,
            run_contract=warm_run_contract,
            validation_run_contract=warm_validation_run_contract,
        )
        return model

    def configure_warm_source(
        self,
        *,
        policy: SourcePolicy,
        memory_sigma: float,
        run_contract: WarmSourceRunContract | Mapping[str, Any] | None,
        validation_run_contract: (
            WarmSourceRunContract | Mapping[str, Any] | None
        ) = None,
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
        contract = _coerce_run_contract(run_contract, field="warm_run_contract")
        validation_contract = _coerce_run_contract(
            validation_run_contract,
            field="warm_validation_run_contract",
        )
        if policy != "gaussian_null" and contract is None:
            raise SourceTransportError(
                f"{policy} requires a complete WarmSourceRunContract"
            )
        if contract is not None and contract.query_split != "train":
            raise SourceTransportError(
                "warm_run_contract must bind the train candidate/cache split"
            )
        if validation_contract is not None:
            if contract is None:
                raise SourceTransportError(
                    "warm_validation_run_contract requires a train run contract"
                )
            if validation_contract.query_split != "dev":
                raise SourceTransportError(
                    "warm_validation_run_contract must bind the dev split"
                )
            shared_fields = (
                "bank_manifest_sha256",
                "bank_content_sha256",
                "catalog_sha256",
                "audit_sha256",
                "normalization_stats_sha256",
                "action_space_contract_sha256",
                "base_checkpoint_sha256",
                "global_sample_stride",
                "action_horizon",
                "action_dim",
            )
            mismatches = {
                name: (getattr(contract, name), getattr(validation_contract, name))
                for name in shared_fields
                if getattr(contract, name) != getattr(validation_contract, name)
            }
            if mismatches:
                details = ", ".join(
                    f"{name}: train={train!r}, dev={dev!r}"
                    for name, (train, dev) in sorted(mismatches.items())
                )
                raise SourceTransportError(
                    "train/dev WARM run contracts disagree on shared artifacts; "
                    + details
                )
            if (
                validation_contract.candidate_manifest_sha256
                == contract.candidate_manifest_sha256
                or validation_contract.query_corpus_sha256
                == contract.query_corpus_sha256
            ):
                raise SourceTransportError(
                    "validation contract must bind an independent dev "
                    "candidate manifest and query corpus"
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
        self.warm_validation_run_contract = validation_contract
        # Online capabilities are process-local and are never serialized in a
        # training checkpoint. Reconfiguration invalidates every prior bind.
        self._warm_online_retriever = None
        self._warm_loaded_checkpoint_sha256: str | None = None
        # Formal trainer attestations bind the baseline bytes actually loaded,
        # not only the digest declared by the source-run contract.
        self._warm_loaded_base_checkpoint_sha256: str | None = None

    def load_checkpoint(self, path, optimizer=None):
        """Strictly restore a complete, contract-bound WARM checkpoint.

        Unlike the backward-compatible FastWAM loader, this public entry point
        deliberately exposes no switches that could weaken model strictness,
        proprio matching, legacy-checkpoint rejection, or metadata preflight.
        The separate :meth:`load_base_checkpoint` method is the only supported
        way to initialize WARM from a non-WARM baseline.
        """

        # Invalidate an earlier authorization *before* any filesystem or
        # checkpoint operation.  Every failure path must leave online use
        # disabled, including a missing file or a partially loaded state dict.
        self._warm_loaded_checkpoint_sha256 = None
        self._warm_online_retriever = None

        from fastwam.memory.manifest import sha256_file

        checkpoint_path = Path(path).expanduser().resolve()
        checkpoint_sha256 = sha256_file(checkpoint_path)
        payload = super().load_checkpoint(
            checkpoint_path,
            optimizer=optimizer,
            load_extra_state=True,
            strict_model_state=True,
            exact_proprio_state=True,
            allow_legacy_dit=False,
        )
        checkpoint_sha256_after = sha256_file(checkpoint_path)
        if checkpoint_sha256_after != checkpoint_sha256:
            self._warm_loaded_checkpoint_sha256 = None
            self._warm_online_retriever = None
            raise SourceTransportError(
                "WARM checkpoint changed while it was being loaded; online "
                "binding is forbidden"
            )
        # Set this only after metadata and state-dict preflight/load succeed.
        # A failed checkpoint attempt must not authorize an online contract.
        self._warm_loaded_checkpoint_sha256 = checkpoint_sha256_after
        self._warm_online_retriever = None
        return payload

    def load_base_checkpoint(self, path: str | Path) -> dict[str, Any]:
        """Load an immutable FastWAM baseline, never a prior WARM checkpoint."""

        self._require_warm_configuration()
        # Base initialization always revokes any prior WARM checkpoint
        # capability, even when validation/loading below fails.
        self._warm_loaded_checkpoint_sha256 = None
        self._warm_online_retriever = None
        self._warm_loaded_base_checkpoint_sha256 = None
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
        payload = super().load_checkpoint(
            checkpoint_path,
            load_extra_state=False,
            forbid_extra_keys=("warm_source",),
            strict_model_state=True,
            exact_proprio_state=True,
            allow_legacy_dit=False,
        )
        actual_sha256_after = sha256_file(checkpoint_path)
        if actual_sha256_after != actual_sha256:
            self._warm_loaded_checkpoint_sha256 = None
            self._warm_online_retriever = None
            raise SourceTransportError(
                "base checkpoint changed while it was being loaded; WARM "
                "initialization is not contract-bound"
            )
        self._warm_loaded_base_checkpoint_sha256 = actual_sha256_after
        return payload

    def training_attestation_metadata(self) -> dict[str, Any]:
        """Return closed-world source identities to the live trainer.

        This capability is intentionally unavailable until the immutable base
        checkpoint has been loaded and verified in-process.  The trainer uses
        the method's presence to require a clean-repository attestation for
        every formally published WARM weights checkpoint.
        """

        self._require_warm_configuration()
        contract = self.warm_run_contract
        if contract is None:
            raise SourceTransportError(
                "formal WARM training attestation requires a train source contract"
            )
        loaded_base = self._warm_loaded_base_checkpoint_sha256
        if loaded_base != contract.base_checkpoint_sha256:
            raise SourceTransportError(
                "formal WARM training attestation requires the contract-bound "
                "base checkpoint to be loaded first"
            )
        validation = self.warm_validation_run_contract
        if validation is None:
            raise SourceTransportError(
                "formal WARM training attestation requires an independent dev "
                "source contract, even when periodic validation is disabled"
            )
        return {
            "source_policy": self.warm_source_policy,
            "train_source_contract_sha256": contract.sha256,
            "dev_source_contract_sha256": validation.sha256,
            "base_checkpoint_sha256": loaded_base,
        }

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

        self._validate_dataset_against_contract(
            dataset,
            contract=contract,
            purpose="training",
        )

    def validate_validation_dataset(self, dataset: object) -> None:
        """Prove that held-out loss uses the independently bound dev cache."""

        self._require_warm_configuration()
        contract = self.warm_validation_run_contract
        if contract is None:
            raise SourceTransportError(
                "validation is disabled until warm_validation_run_contract "
                "binds an independent dev dataset/cache"
            )
        self._validate_dataset_against_contract(
            dataset,
            contract=contract,
            purpose="validation",
        )

    @staticmethod
    def _validate_dataset_against_contract(
        dataset: object,
        *,
        contract: WarmSourceRunContract,
        purpose: str,
    ) -> None:
        if purpose not in {"training", "validation"}:
            raise ValueError("purpose must be training or validation")

        from fastwam.memory.manifest import sha256_canonical_json
        from fastwam.memory.runtime_candidates import RuntimeCandidateResolver

        resolver = getattr(dataset, "resolver", None)
        if not isinstance(resolver, RuntimeCandidateResolver):
            raise SourceTransportError(
                f"contract-bound WARM {purpose} requires "
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
                f"{purpose} dataset does not match WarmSourceRunContract; "
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
            self.warm_validation_run_contract is not None
            and WARM_QUERY_SPLIT not in sample
        ):
            raise SourceTransportError(
                f"{WARM_QUERY_SPLIT} is required when independent train/dev "
                "candidate contracts are configured"
            )
        query_split = self._query_split_from_sample(sample, batch_size=batch_size)
        if query_split == "train":
            sample_contract = self.warm_run_contract
        else:
            sample_contract = self.warm_validation_run_contract
            if sample_contract is None:
                raise SourceTransportError(
                    "dev sample requires warm_validation_run_contract"
                )
        if sample_contract is not None and tuple(action.shape[1:]) != (
            sample_contract.action_horizon,
            sample_contract.action_dim,
        ):
            raise SourceTransportError(
                f"{query_split} action shape does not match its "
                "WarmSourceRunContract: "
                f"{tuple(action.shape[1:])} != "
                f"{(sample_contract.action_horizon, sample_contract.action_dim)}"
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

    @staticmethod
    def _query_split_from_sample(
        sample: Mapping[str, Any],
        *,
        batch_size: int,
    ) -> str:
        value = sample.get(WARM_QUERY_SPLIT, "train")
        if isinstance(value, str):
            values = (value,)
        elif isinstance(value, (tuple, list)):
            values = tuple(value)
            if len(values) != batch_size:
                raise SourceTransportError(
                    f"{WARM_QUERY_SPLIT} batch length must be {batch_size}"
                )
        else:
            raise TypeError(
                f"{WARM_QUERY_SPLIT} must be a split string or homogeneous "
                "batch of split strings"
            )
        if not values or any(item not in {"train", "dev"} for item in values):
            raise SourceTransportError(
                f"{WARM_QUERY_SPLIT} values must be 'train' or 'dev'"
            )
        if len(set(values)) != 1:
            raise SourceTransportError(
                "one WARM batch cannot mix train and dev candidate contracts"
            )
        return values[0]

    def training_loss(self, sample, tiled: bool = False):
        source_context = self._source_context_from_batch(sample)
        return self.training_loss_action_only(
            sample,
            action_source_context=source_context,
            memory_sigma=self.warm_memory_sigma,
            tiled=tiled,
        )

    def bind_online_retriever(self, retriever: object) -> None:
        """Bind one process-local capability to this loaded WARM checkpoint.

        Online candidates are not a tensor API.  They are accepted only from
        the exact retriever whose immutable rollout contract matches the train
        contract and the bytes of the WARM checkpoint loaded into this model.
        The retriever capability itself is deliberately not checkpoint state.
        """

        self._require_warm_configuration()
        # A failed rebind must never leave an older retriever authorized.
        self._warm_online_retriever = None
        if self.warm_source_policy == "oracle_action_top1":
            raise SourceTransportError(
                "oracle_action_top1 cannot bind an online retriever"
            )
        if self.warm_source_policy == "gaussian_null":
            raise SourceTransportError(
                "gaussian_null does not bind or access an online retriever"
            )
        contract = self.warm_run_contract
        if contract is None:
            raise SourceTransportError(
                "fixed online retrieval requires a train run contract"
            )
        validation_contract = self.warm_validation_run_contract
        if validation_contract is None:
            raise SourceTransportError(
                "fixed online retrieval requires an independently bound dev "
                "validation run contract"
            )
        if self._warm_loaded_checkpoint_sha256 is None:
            raise SourceTransportError(
                "bind_online_retriever requires a successfully loaded WARM "
                "checkpoint"
            )

        from fastwam.memory.online_retrieval import FrozenDinoOnlineRetriever
        from .online_contract import WarmOnlineRunContract

        if type(retriever) is not FrozenDinoOnlineRetriever:
            raise TypeError(
                "retriever must be an exact FrozenDinoOnlineRetriever instance"
            )
        if retriever.artifact_verified is not True:
            raise SourceTransportError(
                "online retriever has no closed-world artifact verification"
            )
        source_contract = getattr(retriever, "source_run_contract", None)
        online_contract = getattr(retriever, "online_run_contract", None)
        if not isinstance(source_contract, WarmSourceRunContract):
            raise SourceTransportError(
                "online retriever is missing its source run contract"
            )
        if not isinstance(online_contract, WarmOnlineRunContract):
            raise SourceTransportError(
                "online retriever is missing its online run contract"
            )
        if source_contract.sha256 != contract.sha256:
            raise SourceTransportError(
                "online retriever training run contract does not match model"
            )

        expected = {
            "training_run_contract_sha256": contract.sha256,
            "validation_run_contract_sha256": validation_contract.sha256,
            "warm_checkpoint_sha256": self._warm_loaded_checkpoint_sha256,
            "bank_manifest_sha256": contract.bank_manifest_sha256,
            "bank_content_sha256": contract.bank_content_sha256,
            "normalization_stats_sha256": contract.normalization_stats_sha256,
            "action_space_contract_sha256": (
                contract.action_space_contract_sha256
            ),
            "catalog_sha256": contract.catalog_sha256,
            "audit_sha256": contract.audit_sha256,
            "source_policy": self.warm_source_policy,
            "memory_sigma": self.warm_memory_sigma,
            "action_horizon": contract.action_horizon,
            "action_dim": contract.action_dim,
        }
        mismatches = {
            field: (value, getattr(online_contract, field, None))
            for field, value in expected.items()
            if getattr(online_contract, field, None) != value
        }
        if mismatches:
            details = ", ".join(
                f"{field}: model={model!r}, online={online!r}"
                for field, (model, online) in sorted(mismatches.items())
            )
            raise SourceTransportError(
                "online retriever contract does not match loaded model; " + details
            )
        self._warm_online_retriever = retriever

    def _validate_online_step_model_binding(self, online_step: Any) -> None:
        """Validate model-shape inputs before consuming the one-shot step."""

        model_input = online_step.model_input
        if tuple(model_input.shape[:2]) != (1, 3):
            raise SourceTransportError(
                "bound online model_input must have shape [1,3,H,W]"
            )
        if int(model_input.shape[2]) % 16 or int(model_input.shape[3]) % 16:
            raise SourceTransportError(
                "bound online model_input spatial dimensions must be multiples of 16"
            )

        proprio = online_step.proprio
        if self.proprio_dim is None:
            if proprio is not None:
                raise SourceTransportError(
                    "bound online step contains proprio but this model disables it"
                )
            return
        if proprio is None:
            raise SourceTransportError(
                "proprio-enabled WARM model requires bound online proprio"
            )
        if tuple(proprio.shape) != (1, int(self.proprio_dim)):
            raise SourceTransportError(
                "bound online proprio shape does not match model: "
                f"{tuple(proprio.shape)} != {(1, int(self.proprio_dim))}"
            )

    def _source_context_from_validated_online_step(
        self,
        online_step: Any,
    ) -> ActionSourceContext:
        """Convert one already capability-validated step into source tensors."""

        means = torch.tensor(
            online_step.candidate_means,
            dtype=self.torch_dtype,
            device=self.device,
        ).unsqueeze(0)
        valid = torch.tensor(
            online_step.candidate_valid_mask,
            dtype=torch.bool,
            device=self.device,
        ).unsqueeze(0)
        contract = self.warm_run_contract
        if contract is None or tuple(means.shape[2:]) != (
            contract.action_horizon,
            contract.action_dim,
        ):
            raise SourceTransportError(
                "bound online candidate action shape does not match model contract"
            )
        if valid.shape != means.shape[:2]:
            raise SourceTransportError(
                "bound online candidate means/mask ranks do not match"
            )
        components = select_source_components(
            valid,
            policy="fixed_context_top1",
            phase="infer",
        )
        return ActionSourceContext(
            component_indices=components,
            candidate_means=means,
            candidate_valid_mask=valid,
        )

    def build_inference_source_context(
        self,
        *,
        online_step: object | None = None,
        batch_size: int = 1,
    ) -> ActionSourceContext:
        """Build an inference source only from null or a bound online step."""

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
            # Do not inspect online_step or a retriever on the null branch.
            return ActionSourceContext(
                component_indices=torch.zeros(
                    (batch_size,), dtype=torch.long, device=self.device
                ),
                candidate_means=None,
                candidate_valid_mask=None,
            )
        if batch_size != 1:
            raise SourceTransportError(
                "online fixed retrieval currently supports one rollout step"
            )
        retriever = self._warm_online_retriever
        if retriever is None:
            raise SourceTransportError(
                "fixed_context_top1 requires bind_online_retriever before inference"
            )

        from fastwam.memory.online_retrieval import BoundOnlineStep

        if type(online_step) is not BoundOnlineStep:
            raise SourceTransportError(
                "fixed_context_top1 requires a retriever-produced BoundOnlineStep"
            )
        self._validate_online_step_model_binding(online_step)
        validated = retriever.assert_owned_bound_step(
            online_step,
            prompt=online_step.prompt,
            proprio=online_step.proprio,
            input_image=online_step.model_input,
        )
        if validated is not online_step:
            raise SourceTransportError(
                "online retriever must validate and return the identical capability"
            )
        return self._source_context_from_validated_online_step(online_step)

    @torch.no_grad()
    def infer_action(
        self,
        *args,
        online_step: object | None = None,
        **kwargs,
    ):
        """Run action-only inference with a bound online capability or null."""

        self._require_warm_configuration()
        forbidden = sorted(_FORBIDDEN_RAW_INFERENCE_FIELDS.intersection(kwargs))
        if forbidden:
            raise SourceTransportError(
                "raw/prebuilt WARM inference payloads are forbidden; "
                f"received {forbidden}. Use a BoundOnlineStep."
            )
        if self.warm_source_policy == "oracle_action_top1":
            raise SourceTransportError(
                "oracle_action_top1 checkpoints cannot run infer_action"
            )
        if self.warm_source_policy == "gaussian_null":
            # Deliberately do not inspect online_step or any retriever state.
            action_source_context = self.build_inference_source_context(batch_size=1)
            kwargs["action_source_context"] = action_source_context
            kwargs["memory_sigma"] = self.warm_memory_sigma
            output = super().infer_action(*args, **kwargs)
            source_seed = kwargs.get("seed")
            output["warm_online_telemetry"] = _json_safe(
                {
                    "retrieval": None,
                    "source": {
                        "policy": "gaussian_null",
                        "component": 0,
                        "selected_rank": None,
                        "selected_event_id": None,
                        "memory_selected": False,
                        "memory_sigma": float(self.warm_memory_sigma),
                        "derived_seed": (
                            None if source_seed is None else int(source_seed)
                        ),
                    },
                },
                field="WARM null online telemetry",
            )
            return output

        if args:
            raise SourceTransportError(
                "fixed_context_top1 accepts no positional model inputs; all "
                "prompt/proprio/image bindings come from BoundOnlineStep"
            )
        bound_input_fields = {
            "prompt",
            "input_image",
            "proprio",
            "context",
            "context_mask",
            "action_horizon",
            "seed",
        }
        externally_bound = sorted(bound_input_fields.intersection(kwargs))
        if externally_bound:
            raise SourceTransportError(
                "fixed_context_top1 model inputs are owned by BoundOnlineStep; "
                f"remove external fields {externally_bound}"
            )
        retriever = self._warm_online_retriever
        if retriever is None:
            raise SourceTransportError(
                "fixed_context_top1 requires bind_online_retriever before inference"
            )
        from fastwam.memory.online_retrieval import BoundOnlineStep

        if type(online_step) is not BoundOnlineStep:
            raise SourceTransportError(
                "fixed_context_top1 requires a retriever-produced BoundOnlineStep"
            )
        self._validate_online_step_model_binding(online_step)
        validated = retriever.validate_bound_step(
            online_step,
            prompt=online_step.prompt,
            proprio=online_step.proprio,
            input_image=online_step.model_input,
        )
        if validated is not online_step:
            raise SourceTransportError(
                "online retriever must validate and return the identical capability"
            )
        action_source_context = self._source_context_from_validated_online_step(
            online_step
        )
        kwargs.update(
            {
                "prompt": online_step.prompt,
                "input_image": torch.tensor(
                    online_step.model_input,
                    dtype=self.torch_dtype,
                    device=self.device,
                ),
                "action_horizon": int(self.warm_run_contract.action_horizon),
                "proprio": (
                    None
                    if online_step.proprio is None
                    else torch.tensor(
                        online_step.proprio,
                        dtype=self.torch_dtype,
                        device=self.device,
                    )
                ),
                "seed": int(online_step.derived_seed),
                "action_source_context": action_source_context,
                "memory_sigma": self.warm_memory_sigma,
            }
        )
        output = super().infer_action(**kwargs)
        component_value = output.get("source_component")
        if not isinstance(component_value, torch.Tensor) or component_value.numel() != 1:
            raise SourceTransportError(
                "FastWAM fixed-source output is missing one source component"
            )
        component = int(component_value.detach().cpu().reshape(-1)[0].item())
        if component not in {0, 1}:
            raise SourceTransportError(
                "fixed_context_top1 produced an invalid source component"
            )
        telemetry = {
            "retrieval": _fixed_retrieval_telemetry(online_step),
            "source": {
                "policy": "fixed_context_top1",
                "component": component,
                "selected_rank": 0 if component == 1 else None,
                "selected_event_id": (
                    _event_id_json(online_step.selected_event_id)
                    if component == 1
                    else None
                ),
                "memory_selected": component == 1,
                "memory_sigma": float(self.warm_memory_sigma),
                "derived_seed": int(online_step.derived_seed),
            },
        }
        output["warm_online_telemetry"] = _json_safe(
            telemetry, field="WARM online telemetry"
        )
        return output

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
                "fixed_context_top1 supports only infer_action with a "
                "BoundOnlineStep"
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
                "fixed_context_top1 supports only infer_action with a "
                "BoundOnlineStep"
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
        validation_contract = (
            None
            if self.warm_validation_run_contract is None
            else self.warm_validation_run_contract.to_dict()
        )
        validation_contract_sha256 = (
            None
            if self.warm_validation_run_contract is None
            else self.warm_validation_run_contract.sha256
        )
        return {
            "warm_source": {
                "schema": WARM_SOURCE_CHECKPOINT_SCHEMA,
                "version": WARM_SOURCE_CHECKPOINT_VERSION,
                "policy": self.warm_source_policy,
                "memory_sigma": self.warm_memory_sigma,
                "run_contract": contract,
                "run_contract_sha256": contract_sha256,
                "validation_run_contract": validation_contract,
                "validation_run_contract_sha256": validation_contract_sha256,
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
            "validation_run_contract",
            "validation_run_contract_sha256",
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
        validation_value = value["validation_run_contract"]
        loaded_validation = (
            None
            if validation_value is None
            else WarmSourceRunContract.from_dict(validation_value)
        )
        loaded_validation_hash = (
            None if loaded_validation is None else loaded_validation.sha256
        )
        if (
            value["validation_run_contract_sha256"]
            != loaded_validation_hash
        ):
            raise ValueError(
                "checkpoint validation_run_contract_sha256 is invalid"
            )
        configured_validation = self.warm_validation_run_contract
        configured_validation_hash = (
            None
            if configured_validation is None
            else configured_validation.sha256
        )
        if loaded_validation_hash != configured_validation_hash:
            raise ValueError(
                "checkpoint validation run contract does not match configured dev artifacts"
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
    "WARM_QUERY_SPLIT",
    "WARM_SOURCE_CHECKPOINT_SCHEMA",
    "WARM_SOURCE_CHECKPOINT_VERSION",
    "WarmSourceFastWAM",
]
