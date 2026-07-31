"""Compact M3 consequence-aligned retrieval and source-gating core.

This module intentionally stops at the boundary of the existing WARM source
transport.  It learns candidate utility, validates predicted consequences,
selects either a memory component or the explicit null component, and predicts
the scalar gate.  It does not contain another policy, a video model, or an
action tokenizer.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Literal, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .consequence_curriculum import CorruptionDecision, CorruptionMode


AUTO_SELECT = -2
FORCE_NULL = -1
_MAX_RERANKER_PARAMETERS = 5_000_000
Reduction = Literal["none", "mean", "sum"]


class ConsequenceAlignmentError(ValueError):
    """Raised when an M3 tensor or selection contract is invalid."""


@dataclass(frozen=True, slots=True)
class UtilityTargets:
    """Stop-gradient soft targets derived from GT action and world effect."""

    probabilities: torch.Tensor
    utilities: torch.Tensor
    action_distance: torch.Tensor
    effect_distance: torch.Tensor
    timing_distance: torch.Tensor
    valid_rows: torch.Tensor


@dataclass(frozen=True, slots=True)
class CorruptionControls:
    """Torch controls produced from deterministic curriculum decisions.

    ``forced_candidate_indices`` uses ``AUTO_SELECT`` for ordinary scoring,
    ``FORCE_NULL`` for the explicit Gaussian component, and a zero-based slot
    for a forced hard negative.
    """

    candidate_valid_mask: torch.Tensor
    forced_candidate_indices: torch.Tensor


@dataclass(frozen=True, slots=True)
class ConsequenceSelection:
    """Candidate/null selection and the features consumed by the gate."""

    candidate_indices: torch.Tensor
    component_indices: torch.Tensor
    memory_mask: torch.Tensor
    candidate_scores: torch.Tensor
    selected_score: torch.Tensor
    selection_margin: torch.Tensor
    selected_probability: torch.Tensor
    probability_margin: torch.Tensor
    normalized_entropy: torch.Tensor
    selected_consistency: torch.Tensor
    selected_support: torch.Tensor


@dataclass(frozen=True, slots=True)
class GateOutput:
    """Raw logits and null-safe source probabilities."""

    logits: torch.Tensor
    probability: torch.Tensor
    features: torch.Tensor
    memory_mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class SourceAcceptance:
    """Calibrated memory-source probability and explicit null decision."""

    effective_probability: torch.Tensor
    quality: torch.Tensor
    accepted_mask: torch.Tensor


def _require_positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConsequenceAlignmentError(f"{field} must be a positive integer")
    return value


def _finite_number(
    value: object,
    field: str,
    *,
    positive: bool = False,
    non_negative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ConsequenceAlignmentError(f"{field} must be finite")
    if positive and result <= 0.0:
        raise ConsequenceAlignmentError(f"{field} must be positive")
    if non_negative and result < 0.0:
        raise ConsequenceAlignmentError(f"{field} must be non-negative")
    return result


def _require_tensor(value: object, field: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{field} must be a torch.Tensor")
    return value


def _require_float_tensor(
    value: object,
    field: str,
    *,
    ndim: int,
) -> torch.Tensor:
    tensor = _require_tensor(value, field)
    if tensor.ndim != ndim:
        raise ConsequenceAlignmentError(
            f"{field} must be rank {ndim}, got shape {tuple(tensor.shape)}"
        )
    if not tensor.is_floating_point():
        raise TypeError(f"{field} must have a floating dtype, got {tensor.dtype}")
    if not bool(torch.isfinite(tensor).all().item()):
        raise ConsequenceAlignmentError(f"{field} must contain only finite values")
    return tensor


def _require_bool_matrix(
    value: object,
    field: str,
    *,
    shape: tuple[int, int] | None = None,
) -> torch.Tensor:
    tensor = _require_tensor(value, field)
    if tensor.ndim != 2:
        raise ConsequenceAlignmentError(
            f"{field} must have shape [B,K], got {tuple(tensor.shape)}"
        )
    if tensor.dtype != torch.bool:
        raise TypeError(f"{field} must have dtype bool, got {tensor.dtype}")
    if shape is not None and tensor.shape != shape:
        raise ConsequenceAlignmentError(
            f"{field} must have shape {shape}, got {tuple(tensor.shape)}"
        )
    return tensor


def _same_float_contract(
    reference: torch.Tensor,
    tensors: Sequence[tuple[str, torch.Tensor]],
) -> None:
    for field, tensor in tensors:
        if tensor.dtype != reference.dtype:
            raise TypeError(
                f"{field} must have dtype {reference.dtype}, got {tensor.dtype}"
            )
        if tensor.device != reference.device:
            raise ConsequenceAlignmentError(
                f"{field} must be on {reference.device}, got {tensor.device}"
            )


class ActionUtilityReranker(nn.Module):
    """A bounded two-layer MLP over context, action, effect, and timing.

    Scores remain finite even for invalid padding slots; callers must pass the
    same validity mask to the KL loss and consequence selector.  Zeroing
    invalid scores avoids manufacturing infinities that can leak into metrics.
    """

    def __init__(
        self,
        *,
        query_dim: int,
        candidate_context_dim: int,
        action_summary_dim: int,
        effect_dim: int,
        timing_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        dimensions = {
            "query_dim": query_dim,
            "candidate_context_dim": candidate_context_dim,
            "action_summary_dim": action_summary_dim,
            "effect_dim": effect_dim,
            "timing_dim": timing_dim,
            "hidden_dim": hidden_dim,
        }
        checked = {
            name: _require_positive_int(value, name)
            for name, value in dimensions.items()
        }
        self.query_dim = checked["query_dim"]
        self.candidate_context_dim = checked["candidate_context_dim"]
        self.action_summary_dim = checked["action_summary_dim"]
        self.effect_dim = checked["effect_dim"]
        self.timing_dim = checked["timing_dim"]
        input_dim = sum(
            (
                self.query_dim,
                self.candidate_context_dim,
                self.action_summary_dim,
                self.effect_dim,
                self.timing_dim,
            )
        )
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, checked["hidden_dim"]),
            nn.SiLU(),
            nn.Linear(checked["hidden_dim"], 1),
        )
        parameter_count = sum(parameter.numel() for parameter in self.parameters())
        if parameter_count >= _MAX_RERANKER_PARAMETERS:
            raise ConsequenceAlignmentError(
                "ActionUtilityReranker must remain below 5M parameters; "
                f"requested dimensions create {parameter_count:,}"
            )

    def forward(
        self,
        query_context: torch.Tensor,
        candidate_context: torch.Tensor,
        candidate_action_summary: torch.Tensor,
        candidate_effect: torch.Tensor,
        candidate_timing: torch.Tensor,
        candidate_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        query = _require_float_tensor(
            query_context, "query_context", ndim=2
        )
        context = _require_float_tensor(
            candidate_context, "candidate_context", ndim=3
        )
        action = _require_float_tensor(
            candidate_action_summary, "candidate_action_summary", ndim=3
        )
        effect = _require_float_tensor(
            candidate_effect, "candidate_effect", ndim=3
        )
        timing = _require_float_tensor(
            candidate_timing, "candidate_timing", ndim=3
        )
        batch_size, candidates = context.shape[:2]
        if batch_size <= 0:
            raise ConsequenceAlignmentError(
                "reranker inputs must have a non-empty batch dimension"
            )
        expected_shapes = {
            "query_context": (batch_size, self.query_dim),
            "candidate_context": (
                batch_size,
                candidates,
                self.candidate_context_dim,
            ),
            "candidate_action_summary": (
                batch_size,
                candidates,
                self.action_summary_dim,
            ),
            "candidate_effect": (batch_size, candidates, self.effect_dim),
            "candidate_timing": (batch_size, candidates, self.timing_dim),
        }
        observed = {
            "query_context": query,
            "candidate_context": context,
            "candidate_action_summary": action,
            "candidate_effect": effect,
            "candidate_timing": timing,
        }
        for field, expected in expected_shapes.items():
            if observed[field].shape != expected:
                raise ConsequenceAlignmentError(
                    f"{field} must have shape {expected}, got "
                    f"{tuple(observed[field].shape)}"
                )
        valid = _require_bool_matrix(
            candidate_valid_mask,
            "candidate_valid_mask",
            shape=(batch_size, candidates),
        )
        _same_float_contract(
            query,
            (
                ("candidate_context", context),
                ("candidate_action_summary", action),
                ("candidate_effect", effect),
                ("candidate_timing", timing),
            ),
        )
        if valid.device != query.device:
            raise ConsequenceAlignmentError(
                "candidate_valid_mask must share the feature device"
            )
        parameter = next(self.parameters())
        if parameter.device != query.device:
            raise ConsequenceAlignmentError(
                "reranker parameters and input features must share a device"
            )
        if candidates == 0:
            return query.new_empty((batch_size, 0))
        repeated_query = query.unsqueeze(1).expand(-1, candidates, -1)
        features = torch.cat(
            (repeated_query, context, action, effect, timing), dim=-1
        )
        scores = self.network(features).squeeze(-1)
        scores = scores.masked_fill(~valid, 0.0)
        if not bool(torch.isfinite(scores).all().item()):
            raise ConsequenceAlignmentError("reranker produced non-finite scores")
        return scores


def build_action_effect_utility_targets(
    candidate_actions: torch.Tensor,
    target_action: torch.Tensor,
    candidate_effects: torch.Tensor,
    target_effect: torch.Tensor,
    candidate_valid_mask: torch.Tensor,
    *,
    action_valid_mask: torch.Tensor | None = None,
    candidate_timing: torch.Tensor | None = None,
    target_timing: torch.Tensor | None = None,
    effect_weight: float = 1.0,
    timing_weight: float = 1.0,
    temperature: float = 1.0,
) -> UtilityTargets:
    """Build ``softmax(-(d_action + weights*d_effect)/temperature)`` targets."""

    candidates = _require_float_tensor(
        candidate_actions, "candidate_actions", ndim=4
    )
    target = _require_float_tensor(target_action, "target_action", ndim=3)
    effects = _require_float_tensor(
        candidate_effects, "candidate_effects", ndim=3
    )
    effect_target = _require_float_tensor(
        target_effect, "target_effect", ndim=2
    )
    batch_size, candidate_count, horizon, action_dim = candidates.shape
    if batch_size <= 0 or horizon <= 0 or action_dim <= 0:
        raise ConsequenceAlignmentError(
            "candidate_actions must have non-empty B, H, and D dimensions"
        )
    if target.shape != (batch_size, horizon, action_dim):
        raise ConsequenceAlignmentError(
            "target_action must have shape "
            f"{(batch_size, horizon, action_dim)}, got {tuple(target.shape)}"
        )
    if effects.shape[:2] != (batch_size, candidate_count) or effects.shape[2] <= 0:
        raise ConsequenceAlignmentError(
            "candidate_effects must have shape [B,K,E] with E > 0"
        )
    if effect_target.shape != (batch_size, effects.shape[2]):
        raise ConsequenceAlignmentError(
            f"target_effect must have shape {(batch_size, effects.shape[2])}, "
            f"got {tuple(effect_target.shape)}"
        )
    valid = _require_bool_matrix(
        candidate_valid_mask,
        "candidate_valid_mask",
        shape=(batch_size, candidate_count),
    )
    _same_float_contract(
        candidates,
        (
            ("target_action", target),
            ("candidate_effects", effects),
            ("target_effect", effect_target),
        ),
    )
    if valid.device != candidates.device:
        raise ConsequenceAlignmentError(
            "candidate_valid_mask must share the candidate feature device"
        )
    effect_lambda = _finite_number(
        effect_weight, "effect_weight", non_negative=True
    )
    timing_lambda = _finite_number(
        timing_weight, "timing_weight", non_negative=True
    )
    tau = _finite_number(temperature, "temperature", positive=True)

    action_error = (
        candidates.float() - target.unsqueeze(1).float()
    ).square().mean(dim=-1)
    if action_valid_mask is None:
        action_distance = action_error.mean(dim=-1)
    else:
        action_valid = _require_tensor(action_valid_mask, "action_valid_mask")
        if (
            action_valid.dtype != torch.bool
            or action_valid.shape != (batch_size, horizon)
            or action_valid.device != candidates.device
        ):
            raise ConsequenceAlignmentError(
                "action_valid_mask must be bool [B,H] on the action device"
            )
        action_weights = action_valid[:, None, :].to(dtype=action_error.dtype)
        action_distance = (action_error * action_weights).sum(dim=-1) / (
            action_weights.sum(dim=-1).clamp(min=1.0)
        )
    effect_distance = (
        effects.float() - effect_target.unsqueeze(1).float()
    ).square().mean(dim=-1)
    timing_distance = torch.zeros_like(action_distance)
    if (candidate_timing is None) != (target_timing is None):
        raise ConsequenceAlignmentError(
            "candidate_timing and target_timing must be provided together"
        )
    if candidate_timing is not None and target_timing is not None:
        timing = _require_float_tensor(
            candidate_timing, "candidate_timing", ndim=3
        )
        timing_target = _require_float_tensor(
            target_timing, "target_timing", ndim=2
        )
        if timing.shape[:2] != (batch_size, candidate_count) or timing.shape[2] <= 0:
            raise ConsequenceAlignmentError(
                "candidate_timing must have shape [B,K,T] with T > 0"
            )
        if timing_target.shape != (batch_size, timing.shape[2]):
            raise ConsequenceAlignmentError(
                f"target_timing must have shape {(batch_size, timing.shape[2])}"
            )
        _same_float_contract(
            candidates,
            (("candidate_timing", timing), ("target_timing", timing_target)),
        )
        timing_distance = (
            timing.float() - timing_target.unsqueeze(1).float()
        ).square().mean(dim=-1)

    utilities = -(
        action_distance
        + effect_lambda * effect_distance
        + timing_lambda * timing_distance
    )
    valid_rows = valid.any(dim=-1)
    probabilities = torch.zeros_like(utilities)
    if candidate_count > 0 and bool(valid_rows.any().item()):
        row_utilities = utilities[valid_rows] / tau
        row_mask = valid[valid_rows]
        row_utilities = row_utilities.masked_fill(~row_mask, -torch.inf)
        row_probabilities = torch.softmax(row_utilities, dim=-1)
        row_probabilities = row_probabilities.masked_fill(~row_mask, 0.0)
        probabilities[valid_rows] = row_probabilities
    return UtilityTargets(
        probabilities=probabilities.detach(),
        utilities=utilities.detach(),
        action_distance=action_distance.detach(),
        effect_distance=effect_distance.detach(),
        timing_distance=timing_distance.detach(),
        valid_rows=valid_rows,
    )


def utility_kl_divergence(
    predicted_scores: torch.Tensor,
    target_probabilities: torch.Tensor,
    candidate_valid_mask: torch.Tensor,
    *,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """Compute closed-world ``KL(target || softmax(scores))`` over valid rows."""

    scores = _require_float_tensor(
        predicted_scores, "predicted_scores", ndim=2
    )
    targets = _require_float_tensor(
        target_probabilities, "target_probabilities", ndim=2
    )
    if scores.shape != targets.shape:
        raise ConsequenceAlignmentError(
            "predicted_scores and target_probabilities must have identical [B,K] shapes"
        )
    if scores.shape[0] <= 0:
        raise ConsequenceAlignmentError(
            "predicted_scores must have a non-empty batch dimension"
        )
    valid = _require_bool_matrix(
        candidate_valid_mask,
        "candidate_valid_mask",
        shape=tuple(scores.shape),
    )
    if targets.device != scores.device or valid.device != scores.device:
        raise ConsequenceAlignmentError(
            "scores, target probabilities, and validity mask must share a device"
        )
    if reduction not in ("none", "mean", "sum"):
        raise ConsequenceAlignmentError(
            "reduction must be 'none', 'mean', or 'sum'"
        )
    tolerance = 1e-4
    if bool(torch.any(targets < -tolerance).item()) or bool(
        torch.any(targets > 1.0 + tolerance).item()
    ):
        raise ConsequenceAlignmentError(
            "target_probabilities must lie in [0,1]"
        )
    if bool(torch.any(targets.masked_select(~valid).abs() > tolerance).item()):
        raise ConsequenceAlignmentError(
            "invalid candidate slots must have zero target probability"
        )
    valid_rows = valid.any(dim=-1)
    targets_f = targets.float()
    scores_f = scores.float()
    row_sums = targets_f.sum(dim=-1)
    if bool(
        torch.any((row_sums[valid_rows] - 1.0).abs() > tolerance).item()
    ) or bool(torch.any(row_sums[~valid_rows].abs() > tolerance).item()):
        raise ConsequenceAlignmentError(
            "target probabilities must sum to one on valid rows and zero on null rows"
        )
    if not bool(valid_rows.any().item()):
        empty_loss = scores_f.sum() * 0.0
        if reduction == "none":
            return torch.zeros_like(row_sums) + empty_loss
        return empty_loss

    row_scores = scores_f[valid_rows]
    row_targets = targets_f[valid_rows]
    row_mask = valid[valid_rows]
    log_probabilities = torch.log_softmax(
        row_scores.masked_fill(~row_mask, -torch.inf), dim=-1
    )
    safe_target_log = row_targets.clamp_min(torch.finfo(row_targets.dtype).tiny).log()
    active = row_mask & (row_targets > 0.0)
    # Never evaluate 0 * (+inf) on padded candidates.  Although torch.where
    # masks the forward value, constructing that inactive NaN branch can
    # still poison gradients in fused/autocast kernels.
    active_log_probabilities = torch.where(
        active,
        log_probabilities,
        torch.zeros_like(log_probabilities),
    )
    terms = torch.where(
        active,
        row_targets * (safe_target_log - active_log_probabilities),
        torch.zeros_like(row_targets),
    )
    row_losses = terms.sum(dim=-1)
    if reduction == "mean":
        return row_losses.mean()
    if reduction == "sum":
        return row_losses.sum()
    result = torch.zeros_like(row_sums)
    result[valid_rows] = row_losses
    return result


def consequence_consistency(
    predicted_effect: torch.Tensor,
    transition_gist: torch.Tensor,
    *,
    magnitude_weight: float = 0.25,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Score direction agreement minus a log-magnitude mismatch penalty."""

    effects = _require_float_tensor(
        predicted_effect, "predicted_effect", ndim=3
    )
    gist = _require_float_tensor(transition_gist, "transition_gist", ndim=2)
    batch_size, candidate_count, effect_dim = effects.shape
    if batch_size <= 0 or effect_dim <= 0:
        raise ConsequenceAlignmentError(
            "predicted_effect must have non-empty B and E dimensions"
        )
    if gist.shape != (batch_size, effect_dim):
        raise ConsequenceAlignmentError(
            f"transition_gist must have shape {(batch_size, effect_dim)}, "
            f"got {tuple(gist.shape)}"
        )
    _same_float_contract(effects, (("transition_gist", gist),))
    weight = _finite_number(
        magnitude_weight, "magnitude_weight", non_negative=True
    )
    eps = _finite_number(epsilon, "epsilon", positive=True)
    if candidate_count == 0:
        return effects.new_empty((batch_size, 0))
    effects_f = effects.float()
    gist_f = gist.float().unsqueeze(1).expand(-1, candidate_count, -1)
    cosine = F.cosine_similarity(
        effects_f, gist_f, dim=-1, eps=eps
    ).clamp(min=-1.0, max=1.0)
    effect_norm = torch.linalg.vector_norm(effects_f, dim=-1)
    gist_norm = torch.linalg.vector_norm(gist_f, dim=-1)
    magnitude_penalty = (
        torch.log(effect_norm + eps) - torch.log(gist_norm + eps)
    ).abs()
    score = cosine - weight * magnitude_penalty
    if not bool(torch.isfinite(score).all().item()):
        raise ConsequenceAlignmentError(
            "consequence consistency produced non-finite scores"
        )
    return score.to(dtype=effects.dtype)


def build_corruption_controls(
    candidate_valid_mask: torch.Tensor,
    decisions: Sequence[CorruptionDecision | CorruptionMode],
    *,
    hard_negative_indices: torch.Tensor | None = None,
) -> CorruptionControls:
    """Translate deterministic decisions into validity and force controls."""

    valid = _require_bool_matrix(
        candidate_valid_mask, "candidate_valid_mask"
    )
    batch_size, candidate_count = valid.shape
    if batch_size <= 0:
        raise ConsequenceAlignmentError(
            "candidate_valid_mask must have a non-empty batch dimension"
        )
    if len(decisions) != batch_size:
        raise ConsequenceAlignmentError(
            f"decisions must contain {batch_size} rows, got {len(decisions)}"
        )
    modes: list[CorruptionMode] = []
    for decision in decisions:
        mode = decision.mode if isinstance(decision, CorruptionDecision) else decision
        if mode not in ("normal", "drop", "null", "hard_negative"):
            raise ConsequenceAlignmentError(
                f"unsupported corruption mode {mode!r}"
            )
        modes.append(mode)
    hard_rows = [index for index, mode in enumerate(modes) if mode == "hard_negative"]
    hard_indices: torch.Tensor | None = None
    if hard_rows:
        if hard_negative_indices is None:
            raise ConsequenceAlignmentError(
                "hard_negative decisions require hard_negative_indices"
            )
        hard_indices = _require_tensor(
            hard_negative_indices, "hard_negative_indices"
        )
        if hard_indices.dtype != torch.long:
            raise TypeError(
                "hard_negative_indices must have dtype int64"
            )
        if hard_indices.shape != (batch_size,) or hard_indices.device != valid.device:
            raise ConsequenceAlignmentError(
                "hard_negative_indices must have shape [B] on the validity-mask device"
            )

    effective_valid = valid.clone()
    forced = torch.full(
        (batch_size,), AUTO_SELECT, dtype=torch.long, device=valid.device
    )
    for row, mode in enumerate(modes):
        if mode == "normal":
            continue
        if mode == "drop":
            effective_valid[row] = False
            continue
        if mode == "null":
            forced[row] = FORCE_NULL
            continue
        assert hard_indices is not None
        slot = int(hard_indices[row].item())
        if not 0 <= slot < candidate_count:
            raise ConsequenceAlignmentError(
                f"hard-negative slot {slot} is outside [0,{candidate_count})"
            )
        if not bool(valid[row, slot].item()):
            raise ConsequenceAlignmentError(
                "hard-negative corruption selected an invalid candidate slot"
            )
        # Keep the original true-prefix candidate set intact.  Selection is
        # forced to the incompatible slot, while the utility reranker can
        # still learn the correct ordering and the gate learns rejection.
        # A sparse non-prefix mask would violate every downstream candidate
        # tensor contract when ``slot > 0``.
        forced[row] = slot
    return CorruptionControls(
        candidate_valid_mask=effective_valid,
        forced_candidate_indices=forced,
    )


def _null_scores(
    null_score: float | torch.Tensor,
    scores: torch.Tensor,
) -> torch.Tensor:
    batch_size = scores.shape[0]
    if isinstance(null_score, torch.Tensor):
        tensor = _require_float_tensor(null_score, "null_score", ndim=1)
        if tensor.shape != (batch_size,):
            raise ConsequenceAlignmentError(
                f"null_score must have shape {(batch_size,)}"
            )
        _same_float_contract(scores, (("null_score", tensor),))
        return tensor
    value = _finite_number(null_score, "null_score")
    return scores.new_full((batch_size,), value)


def select_consequence_candidate(
    reranker_scores: torch.Tensor,
    consistency_scores: torch.Tensor,
    support_count: torch.Tensor,
    candidate_valid_mask: torch.Tensor,
    *,
    consequence_weight: float = 1.0,
    support_weight: float = 0.0,
    candidate_prior: torch.Tensor | None = None,
    selection_temperature: float = 1.0,
    null_score: float | torch.Tensor = 0.0,
    automatic_null: bool = True,
    forced_candidate_indices: torch.Tensor | None = None,
) -> ConsequenceSelection:
    """Select a valid candidate or the explicit component-zero null source.

    With ``automatic_null=True``, selection uses a conservative strict
    comparison against the supplied calibrated null score.  With
    ``automatic_null=False``, the best valid candidate is selected and the
    separately supervised continuous source gate owns memory rejection.  This
    avoids comparing candidate-only softmax logits to an untrained constant.
    Curriculum overrides use -2 for automatic selection, -1 for forced null,
    or a zero-based candidate.
    """

    retrieval = _require_float_tensor(
        reranker_scores, "reranker_scores", ndim=2
    )
    consistency = _require_float_tensor(
        consistency_scores, "consistency_scores", ndim=2
    )
    if consistency.shape != retrieval.shape:
        raise ConsequenceAlignmentError(
            "consistency_scores must match reranker_scores [B,K]"
        )
    valid = _require_bool_matrix(
        candidate_valid_mask,
        "candidate_valid_mask",
        shape=tuple(retrieval.shape),
    )
    support = _require_tensor(support_count, "support_count")
    if support.ndim != 2 or support.shape != retrieval.shape:
        raise ConsequenceAlignmentError(
            "support_count must match reranker_scores [B,K]"
        )
    if support.device != retrieval.device or valid.device != retrieval.device:
        raise ConsequenceAlignmentError(
            "scores, support_count, and candidate_valid_mask must share a device"
        )
    if support.is_floating_point() and not bool(torch.isfinite(support).all().item()):
        raise ConsequenceAlignmentError("support_count must be finite")
    if support.dtype == torch.bool or support.is_complex():
        raise TypeError("support_count must have an integer or floating dtype")
    if bool(torch.any(support < 0).item()):
        raise ConsequenceAlignmentError("support_count must be non-negative")
    _same_float_contract(retrieval, (("consistency_scores", consistency),))
    consequence_lambda = _finite_number(
        consequence_weight, "consequence_weight", non_negative=True
    )
    support_lambda = _finite_number(
        support_weight, "support_weight", non_negative=True
    )
    tau = _finite_number(
        selection_temperature, "selection_temperature", positive=True
    )
    if not isinstance(automatic_null, bool):
        raise TypeError("automatic_null must be bool")
    null = _null_scores(null_score, retrieval)
    batch_size, candidate_count = retrieval.shape
    if batch_size <= 0:
        raise ConsequenceAlignmentError(
            "reranker_scores must have a non-empty batch dimension"
        )
    candidate_scores = (
        retrieval
        + consequence_lambda * consistency
        + support_lambda
        * torch.log1p(support.to(dtype=torch.float32)).to(dtype=retrieval.dtype)
    )
    if candidate_prior is not None:
        prior = _require_float_tensor(
            candidate_prior, "candidate_prior", ndim=2
        )
        if prior.shape != retrieval.shape:
            raise ConsequenceAlignmentError(
                "candidate_prior must match reranker_scores [B,K]"
            )
        _same_float_contract(retrieval, (("candidate_prior", prior),))
        candidate_scores = candidate_scores + prior
    if not bool(torch.isfinite(candidate_scores).all().item()):
        raise ConsequenceAlignmentError("candidate scores must be finite")

    forced = torch.full(
        (batch_size,), AUTO_SELECT, dtype=torch.long, device=retrieval.device
    )
    if forced_candidate_indices is not None:
        supplied = _require_tensor(
            forced_candidate_indices, "forced_candidate_indices"
        )
        if supplied.dtype != torch.long:
            raise TypeError("forced_candidate_indices must have dtype int64")
        if supplied.shape != (batch_size,) or supplied.device != retrieval.device:
            raise ConsequenceAlignmentError(
                "forced_candidate_indices must have shape [B] on the score device"
            )
        if bool(torch.any(supplied < AUTO_SELECT).item()) or bool(
            torch.any(supplied >= candidate_count).item()
        ):
            raise ConsequenceAlignmentError(
                f"forced_candidate_indices must lie in [{AUTO_SELECT},{candidate_count})"
            )
        forced = supplied
        force_memory = forced >= 0
        if bool(force_memory.any().item()):
            rows = torch.nonzero(force_memory, as_tuple=False).squeeze(1)
            if not bool(valid[rows, forced[rows]].all().item()):
                raise ConsequenceAlignmentError(
                    "forced_candidate_indices selects an invalid slot"
                )

    if candidate_count == 0:
        if bool(torch.any(forced >= 0).item()):
            raise ConsequenceAlignmentError(
                "cannot force a memory candidate when K is zero"
            )
        candidate_indices = torch.full(
            (batch_size,), FORCE_NULL, dtype=torch.long, device=retrieval.device
        )
        zeros = retrieval.new_zeros((batch_size,))
        return ConsequenceSelection(
            candidate_indices=candidate_indices,
            component_indices=torch.zeros_like(candidate_indices),
            memory_mask=torch.zeros_like(candidate_indices, dtype=torch.bool),
            candidate_scores=candidate_scores,
            selected_score=null,
            selection_margin=zeros,
            selected_probability=zeros,
            probability_margin=zeros,
            normalized_entropy=zeros,
            selected_consistency=zeros,
            selected_support=zeros,
        )

    masked_scores = candidate_scores.masked_fill(~valid, -torch.inf)
    has_candidate = valid.any(dim=-1)
    candidate_probabilities = torch.zeros_like(candidate_scores)
    normalized_entropy = retrieval.new_zeros((batch_size,))
    if bool(has_candidate.any().item()):
        rows = torch.nonzero(has_candidate, as_tuple=False).squeeze(1)
        row_mask = valid[rows]
        row_logits = (candidate_scores[rows].float() / tau).masked_fill(
            ~row_mask, -torch.inf
        )
        row_probabilities = torch.softmax(row_logits, dim=-1).masked_fill(
            ~row_mask, 0.0
        )
        candidate_probabilities[rows] = row_probabilities.to(
            dtype=candidate_scores.dtype
        )
        counts = row_mask.sum(dim=-1)
        entropy = -(
            row_probabilities
            * row_probabilities.clamp_min(torch.finfo(torch.float32).tiny).log()
        ).sum(dim=-1)
        denominator = counts.to(dtype=torch.float32).log()
        entropy = torch.where(
            counts > 1,
            entropy / denominator.clamp_min(torch.finfo(torch.float32).eps),
            torch.zeros_like(entropy),
        )
        normalized_entropy[rows] = entropy.to(dtype=retrieval.dtype)
    best_score, best_index = masked_scores.max(dim=-1)
    automatic_memory = (
        has_candidate & (best_score > null)
        if automatic_null
        else has_candidate
    )
    candidate_indices = torch.where(
        automatic_memory,
        best_index,
        torch.full_like(best_index, FORCE_NULL),
    )
    candidate_indices = torch.where(
        forced == FORCE_NULL,
        torch.full_like(candidate_indices, FORCE_NULL),
        candidate_indices,
    )
    candidate_indices = torch.where(
        forced >= 0, forced, candidate_indices
    )
    memory_mask = candidate_indices >= 0
    component_indices = torch.where(
        memory_mask, candidate_indices + 1, torch.zeros_like(candidate_indices)
    )

    selected_score = null.clone()
    selected_consistency = retrieval.new_zeros((batch_size,))
    selected_support = retrieval.new_zeros((batch_size,))
    selection_margin = retrieval.new_zeros((batch_size,))
    selected_probability = retrieval.new_zeros((batch_size,))
    probability_margin = retrieval.new_zeros((batch_size,))
    if bool(memory_mask.any().item()):
        rows = torch.nonzero(memory_mask, as_tuple=False).squeeze(1)
        slots = candidate_indices[rows]
        selected_score[rows] = candidate_scores[rows, slots]
        selected_consistency[rows] = consistency[rows, slots]
        selected_support[rows] = support[rows, slots].to(dtype=retrieval.dtype)
        selected_probability[rows] = candidate_probabilities[rows, slots]
        competitors = masked_scores.clone()
        competitors[rows, slots] = -torch.inf
        other_best = competitors[rows].max(dim=-1).values
        if automatic_null:
            alternative = torch.maximum(other_best, null[rows])
        else:
            # The gate consumes a top-1/top-2 margin.  A single-candidate row
            # has no evidence gap and therefore receives a neutral zero
            # margin rather than one measured against an uncalibrated null.
            alternative = torch.where(
                torch.isfinite(other_best), other_best, selected_score[rows]
            )
        selection_margin[rows] = selected_score[rows] - alternative
        probability_competitors = candidate_probabilities.clone()
        probability_competitors[rows, slots] = 0.0
        probability_margin[rows] = selected_probability[rows] - (
            probability_competitors[rows].max(dim=-1).values
        )

    null_rows = ~memory_mask
    null_with_candidates = null_rows & has_candidate
    selection_margin[null_with_candidates] = (
        null[null_with_candidates] - best_score[null_with_candidates]
    )
    return ConsequenceSelection(
        candidate_indices=candidate_indices,
        component_indices=component_indices,
        memory_mask=memory_mask,
        candidate_scores=candidate_scores,
        selected_score=selected_score,
        selection_margin=selection_margin,
        selected_probability=selected_probability,
        probability_margin=probability_margin,
        normalized_entropy=normalized_entropy,
        selected_consistency=selected_consistency,
        selected_support=selected_support,
    )


class SourceConfidenceGate(nn.Module):
    """Calibrated seven-feature source gate with a conservative initial bias.

    Candidate-only ranking logits have no meaningful absolute origin.  The
    gate therefore consumes probability/entropy statistics of the valid
    candidate distribution, plus consequence, support, deformation, and
    factual episode stagnation.  This makes the null decision invariant to an
    arbitrary constant shift of every reranker logit.
    """

    def __init__(self, hidden_dim: int = 16) -> None:
        super().__init__()
        width = _require_positive_int(hidden_dim, "hidden_dim")
        self.network = nn.Sequential(
            nn.Linear(7, width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )
        final = self.network[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.constant_(final.bias, -2.0)

    def forward(
        self,
        selection: ConsequenceSelection,
        action_deformation_norm: torch.Tensor,
        stagnation_score: torch.Tensor | None = None,
    ) -> GateOutput:
        if not isinstance(selection, ConsequenceSelection):
            raise TypeError("selection must be ConsequenceSelection")
        deformation = _require_float_tensor(
            action_deformation_norm, "action_deformation_norm", ndim=1
        )
        batch_size = selection.selected_score.shape[0]
        if batch_size <= 0:
            raise ConsequenceAlignmentError(
                "gate inputs must have a non-empty batch dimension"
            )
        fields = (
            ("selected_probability", selection.selected_probability),
            ("probability_margin", selection.probability_margin),
            ("normalized_entropy", selection.normalized_entropy),
            ("selected_consistency", selection.selected_consistency),
            ("selected_support", selection.selected_support),
        )
        for field, value in fields:
            checked = _require_float_tensor(value, field, ndim=1)
            if checked.shape != (batch_size,):
                raise ConsequenceAlignmentError(
                    f"{field} must have shape {(batch_size,)}"
                )
        if deformation.shape != (batch_size,):
            raise ConsequenceAlignmentError(
                f"action_deformation_norm must have shape {(batch_size,)}"
            )
        if bool(torch.any(deformation < 0).item()):
            raise ConsequenceAlignmentError(
                "action_deformation_norm must be non-negative"
            )
        if stagnation_score is None:
            stagnation = torch.zeros_like(deformation)
        else:
            stagnation = _require_float_tensor(
                stagnation_score, "stagnation_score", ndim=1
            )
            if stagnation.shape != (batch_size,):
                raise ConsequenceAlignmentError(
                    f"stagnation_score must have shape {(batch_size,)}"
                )
            if bool(torch.any(stagnation < 0).item()) or bool(
                torch.any(stagnation > 1).item()
            ):
                raise ConsequenceAlignmentError(
                    "stagnation_score must lie in [0,1]"
                )
        memory_mask = _require_tensor(selection.memory_mask, "memory_mask")
        if memory_mask.dtype != torch.bool or memory_mask.shape != (batch_size,):
            raise ConsequenceAlignmentError(
                "selection.memory_mask must be bool with shape [B]"
            )
        _same_float_contract(
            selection.selected_probability,
            (
                ("probability_margin", selection.probability_margin),
                ("normalized_entropy", selection.normalized_entropy),
                ("selected_consistency", selection.selected_consistency),
                ("selected_support", selection.selected_support),
                ("action_deformation_norm", deformation),
                ("stagnation_score", stagnation),
            ),
        )
        if memory_mask.device != selection.selected_probability.device:
            raise ConsequenceAlignmentError("memory_mask must share the gate device")
        if bool(torch.any(selection.selected_support < 0).item()):
            raise ConsequenceAlignmentError(
                "selection.selected_support must be non-negative"
            )
        parameter = next(self.parameters())
        if parameter.device != selection.selected_probability.device:
            raise ConsequenceAlignmentError(
                "gate parameters and features must share a device"
            )
        features = torch.stack(
            (
                selection.selected_probability,
                selection.probability_margin,
                1.0 - selection.normalized_entropy,
                selection.selected_consistency,
                torch.log1p(selection.selected_support),
                deformation,
                stagnation,
            ),
            dim=-1,
        )
        logits = self.network(features).squeeze(-1)
        probability = torch.sigmoid(logits)
        probability = torch.where(
            memory_mask, probability, torch.zeros_like(probability)
        )
        if not bool(torch.isfinite(probability).all().item()):
            raise ConsequenceAlignmentError("source gate produced non-finite values")
        return GateOutput(
            logits=logits,
            probability=probability,
            features=features,
            memory_mask=memory_mask,
        )


def calibrate_source_acceptance(
    gate: GateOutput,
    selection: ConsequenceSelection,
    stagnation_score: torch.Tensor,
    *,
    minimum_candidate_probability: float,
    maximum_candidate_entropy: float,
    stagnation_decay: float,
    stagnation_hard_threshold: float,
    inference_gate_threshold: float,
    hard_reject: bool,
) -> SourceAcceptance:
    """Combine learned usefulness with ambiguity and factual progress evidence.

    Candidate-only ranking logits have no calibrated absolute origin.  This
    layer consequently uses probability concentration and normalized entropy,
    then decays memory influence when the real closed-loop action history is
    repeating.  Inference additionally exposes an explicit Gaussian null
    decision so a weak memory cannot perturb every replan indefinitely.
    """

    if not isinstance(gate, GateOutput):
        raise TypeError("gate must be GateOutput")
    if not isinstance(selection, ConsequenceSelection):
        raise TypeError("selection must be ConsequenceSelection")
    stagnation = _require_float_tensor(
        stagnation_score, "stagnation_score", ndim=1
    )
    batch = int(gate.probability.shape[0])
    if stagnation.shape != (batch,):
        raise ConsequenceAlignmentError(
            f"stagnation_score must have shape {(batch,)}"
        )
    _same_float_contract(
        gate.probability,
        (
            ("selected_probability", selection.selected_probability),
            ("normalized_entropy", selection.normalized_entropy),
            ("stagnation_score", stagnation),
        ),
    )
    if gate.memory_mask.shape != (batch,) or selection.memory_mask.shape != (batch,):
        raise ConsequenceAlignmentError("source masks must have shape [B]")
    if not torch.equal(gate.memory_mask, selection.memory_mask):
        raise ConsequenceAlignmentError("gate and selection memory masks differ")
    if bool(((stagnation < 0) | (stagnation > 1)).any().item()):
        raise ConsequenceAlignmentError("stagnation_score must lie in [0,1]")

    minimum = _finite_number(
        minimum_candidate_probability,
        "minimum_candidate_probability",
        non_negative=True,
    )
    maximum_entropy = _finite_number(
        maximum_candidate_entropy,
        "maximum_candidate_entropy",
        positive=True,
    )
    decay = _finite_number(
        stagnation_decay, "stagnation_decay", non_negative=True
    )
    hard_stagnation = _finite_number(
        stagnation_hard_threshold,
        "stagnation_hard_threshold",
        non_negative=True,
    )
    gate_threshold = _finite_number(
        inference_gate_threshold,
        "inference_gate_threshold",
        non_negative=True,
    )
    if any(
        value > 1.0
        for value in (minimum, maximum_entropy, hard_stagnation, gate_threshold)
    ):
        raise ConsequenceAlignmentError(
            "source acceptance thresholds must lie in [0,1]"
        )
    if not isinstance(hard_reject, bool):
        raise TypeError("hard_reject must be bool")

    probability_quality = (
        (selection.selected_probability - minimum)
        / max(1.0 - minimum, 1e-6)
    ).clamp(0.0, 1.0)
    entropy_quality = (
        (maximum_entropy - selection.normalized_entropy) / maximum_entropy
    ).clamp(0.0, 1.0)
    progress_quality = torch.exp(-decay * stagnation.float()).to(
        dtype=gate.probability.dtype
    )
    quality = torch.sqrt(probability_quality * entropy_quality) * progress_quality
    effective = gate.probability * quality
    # Even during differentiable training, a candidate whose calibrated
    # quality is exactly zero is an explicit null source.  Keeping that row
    # marked as accepted would make source-usage telemetry disagree with the
    # tensor that actually enters flow matching.
    accepted = selection.memory_mask & (effective > 0)
    if hard_reject:
        accepted = (
            accepted
            & (stagnation < hard_stagnation)
            & (effective >= gate_threshold)
        )
        effective = torch.where(accepted, effective, torch.zeros_like(effective))
    else:
        effective = torch.where(
            selection.memory_mask, effective, torch.zeros_like(effective)
        )
    return SourceAcceptance(
        effective_probability=effective,
        quality=quality,
        accepted_mask=accepted,
    )


def utility_supervised_gate_target(
    selected_action: torch.Tensor,
    target_action: torch.Tensor,
    predicted_effect: torch.Tensor,
    target_effect: torch.Tensor,
    memory_mask: torch.Tensor,
    *,
    action_valid_mask: torch.Tensor | None = None,
    effect_weight: float = 1.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Build ``exp(-(d_action + lambda*d_effect)/temperature)`` targets."""

    selected = _require_float_tensor(
        selected_action, "selected_action", ndim=3
    )
    action_target = _require_float_tensor(
        target_action, "target_action", ndim=3
    )
    effect = _require_float_tensor(
        predicted_effect, "predicted_effect", ndim=2
    )
    effect_target = _require_float_tensor(
        target_effect, "target_effect", ndim=2
    )
    if selected.shape != action_target.shape:
        raise ConsequenceAlignmentError(
            "selected_action and target_action must have identical [B,H,D] shapes"
        )
    if any(int(size) <= 0 for size in selected.shape):
        raise ConsequenceAlignmentError(
            "selected_action must have non-empty B, H, and D dimensions"
        )
    if effect.shape != effect_target.shape or effect.shape[0] != selected.shape[0]:
        raise ConsequenceAlignmentError(
            "predicted_effect and target_effect must share [B,E] with the action batch"
        )
    if effect.shape[1] <= 0:
        raise ConsequenceAlignmentError(
            "predicted_effect must have a non-empty effect dimension"
        )
    mask = _require_tensor(memory_mask, "memory_mask")
    if mask.dtype != torch.bool or mask.shape != (selected.shape[0],):
        raise ConsequenceAlignmentError("memory_mask must be bool with shape [B]")
    _same_float_contract(
        selected,
        (
            ("target_action", action_target),
            ("predicted_effect", effect),
            ("target_effect", effect_target),
        ),
    )
    if mask.device != selected.device:
        raise ConsequenceAlignmentError("memory_mask must share the action device")
    effect_lambda = _finite_number(
        effect_weight, "effect_weight", non_negative=True
    )
    tau = _finite_number(temperature, "temperature", positive=True)
    action_error = (
        selected.float() - action_target.float()
    ).square().mean(dim=-1)
    if action_valid_mask is None:
        action_distance = action_error.mean(dim=-1)
    else:
        action_valid = _require_tensor(action_valid_mask, "action_valid_mask")
        if (
            action_valid.dtype != torch.bool
            or action_valid.shape != selected.shape[:2]
            or action_valid.device != selected.device
        ):
            raise ConsequenceAlignmentError(
                "action_valid_mask must be bool [B,H] on the action device"
            )
        weights = action_valid.to(dtype=action_error.dtype)
        action_distance = (action_error * weights).sum(dim=-1) / (
            weights.sum(dim=-1).clamp(min=1.0)
        )
    effect_distance = (
        effect.float() - effect_target.float()
    ).square().mean(dim=-1)
    target = torch.exp(-(action_distance + effect_lambda * effect_distance) / tau)
    target = torch.where(mask, target, torch.zeros_like(target))
    return target.to(dtype=selected.dtype).detach()


def utility_supervised_gate_bce(
    gate_output: GateOutput,
    target: torch.Tensor,
    *,
    sample_mask: torch.Tensor | None = None,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """BCE-with-logits over selected-memory rows; all-null batches are safe."""

    if not isinstance(gate_output, GateOutput):
        raise TypeError("gate_output must be GateOutput")
    logits = _require_float_tensor(gate_output.logits, "gate logits", ndim=1)
    targets = _require_float_tensor(target, "gate target", ndim=1)
    if targets.shape != logits.shape:
        raise ConsequenceAlignmentError("gate target must match logits shape [B]")
    _same_float_contract(logits, (("gate target", targets),))
    mask = _require_tensor(gate_output.memory_mask, "memory_mask")
    if mask.dtype != torch.bool or mask.shape != logits.shape or mask.device != logits.device:
        raise ConsequenceAlignmentError(
            "gate memory_mask must be bool [B] on the logits device"
        )
    if bool(torch.any(targets < 0).item()) or bool(torch.any(targets > 1).item()):
        raise ConsequenceAlignmentError("gate target must lie in [0,1]")
    if sample_mask is not None:
        supervised = _require_tensor(sample_mask, "sample_mask")
        if (
            supervised.dtype != torch.bool
            or supervised.shape != logits.shape
            or supervised.device != logits.device
        ):
            raise ConsequenceAlignmentError(
                "sample_mask must be bool [B] on the gate device"
            )
        mask = mask & supervised
    if reduction not in ("none", "mean", "sum"):
        raise ConsequenceAlignmentError(
            "reduction must be 'none', 'mean', or 'sum'"
        )
    if not bool(mask.any().item()):
        zero = logits.sum() * 0.0
        if reduction == "none":
            return torch.zeros_like(logits) + zero
        return zero
    row_loss = F.binary_cross_entropy_with_logits(
        logits[mask], targets[mask], reduction="none"
    )
    if reduction == "mean":
        return row_loss.mean()
    if reduction == "sum":
        return row_loss.sum()
    result = torch.zeros_like(logits)
    result[mask] = row_loss
    return result


__all__ = [
    "AUTO_SELECT",
    "FORCE_NULL",
    "ActionUtilityReranker",
    "ConsequenceAlignmentError",
    "ConsequenceSelection",
    "CorruptionControls",
    "GateOutput",
    "SourceAcceptance",
    "SourceConfidenceGate",
    "UtilityTargets",
    "build_action_effect_utility_targets",
    "build_corruption_controls",
    "calibrate_source_acceptance",
    "consequence_consistency",
    "select_consequence_candidate",
    "utility_kl_divergence",
    "utility_supervised_gate_bce",
    "utility_supervised_gate_target",
]
