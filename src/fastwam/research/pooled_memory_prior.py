"""A separately trained, same-record pooled-memory control for WARM.

All valid retrieved actions, visual preconditions, observed historical effects,
timing and phase records enter one four-query attention pool. The common causal
history gist is retained. This control has no learned action deformation, effect
predictor, utility reranker or confidence gate. It is not a named-method replica.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import torch
from torch import nn

from fastwam.research.simple_prior import (
    CONDITIONED_SCHEMA, ConditionedPriorFastWAM, DeterministicEvent, _create_prior,
)

POOLED_SCHEMA = "warm.research.pooled-event-prior-with-recollection.v1"
POOL_ARCHITECTURE = "all-events-linear-four-query-attention-v1"
SPATIAL_TOKENS = 4
PHASE_DIM = 4


class PooledMemoryEvent(DeterministicEvent):
    """Pool complete records before selecting the fixed coarse action prior."""

    def __init__(self, config, *, context_dim):
        super().__init__(config)
        self.context_dim = context_dim
        record_dim = (config.action_horizon * config.action_dim
                      + 2 * SPATIAL_TOKENS * config.event_dim
                      + context_dim + config.timing_dim + PHASE_DIM + SPATIAL_TOKENS)
        dim = config.model_dim
        self.record_projection = nn.Sequential(
            nn.Linear(record_dim, dim), nn.GELU(), nn.LayerNorm(dim),
        )
        self.history_projection = nn.Linear(config.gist_dim, dim)
        self.query_projection = nn.Linear(context_dim, dim)
        self.proprio_projection = nn.Linear(config.proprio_dim, dim)
        self.queries = nn.Parameter(torch.empty(config.context_query_count, dim))
        nn.init.normal_(self.queries, mean=0.0, std=0.02)
        self.pool = nn.MultiheadAttention(
            dim, config.num_heads, dropout=0.0, batch_first=True,
        )
        self.output_norm = nn.LayerNorm(dim)

    @staticmethod
    def _masked_field(value, mask, shape, name, *, reference):
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if not value.is_floating_point() or value.device != reference.device or value.dtype != reference.dtype:
            raise ValueError(f"{name} must share the action floating dtype and device")
        result = value.masked_fill(~mask, 0.0)
        if not bool(torch.isfinite(result).all().item()):
            raise ValueError(f"{name} has nonfinite active values")
        return result

    def forward(self, *, warped_actions, candidate_valid_mask, event_pre_tokens,
                event_delta_tokens, event_delta_mask, candidate_context,
                candidate_phase_features, gripper_timing, gist_tokens,
                query_context, proprio, **unused):
        cfg = self.config
        if warped_actions.ndim != 4:
            raise ValueError("warped_actions must be [B,K,H,D]")
        b, k, h, d = warped_actions.shape
        if b <= 0 or k <= 0 or (h, d) != (cfg.action_horizon, cfg.action_dim):
            raise ValueError("warped_actions must retain the configured complete action chunk")
        valid = candidate_valid_mask
        if (valid.shape != (b, k) or valid.dtype != torch.bool
                or valid.device != warped_actions.device):
            raise ValueError("candidate_valid_mask must be boolean [B,K] on the action device")
        mask = event_delta_mask
        if (mask.shape != (b, k, SPATIAL_TOKENS) or mask.dtype != torch.bool
                or mask.device != valid.device):
            raise ValueError("event_delta_mask must be boolean [B,K,4] on the action device")
        mask = mask & valid[:, :, None]
        def field(value, active, shape, name):
            return self._masked_field(value, active, shape, name, reference=warped_actions)
        actions = field(warped_actions, valid[:, :, None, None], (b,k,h,d), "actions")
        pre = field(event_pre_tokens, mask[..., None], (b,k,SPATIAL_TOKENS,cfg.event_dim), "event_pre")
        delta = field(event_delta_tokens, mask[..., None], pre.shape, "event_delta")
        context = field(candidate_context, valid[..., None], (b,k,self.context_dim), "candidate_context")
        timing = field(gripper_timing, valid[..., None], (b,k,cfg.timing_dim), "timing")
        phase = field(candidate_phase_features, valid[..., None], (b,k,PHASE_DIM), "phase")
        history = field(gist_tokens, torch.ones_like(gist_tokens, dtype=torch.bool),
                        (b,8,cfg.gist_dim), "gist_tokens")
        query = field(query_context, torch.ones_like(query_context, dtype=torch.bool),
                      (b,self.context_dim), "query_context")
        current_proprio = field(proprio, torch.ones_like(proprio, dtype=torch.bool),
                                (b,cfg.proprio_dim), "proprio")
        record = torch.cat((actions.flatten(-2), pre.flatten(-2), delta.flatten(-2),
                            context, timing, phase, mask.to(actions.dtype)), dim=-1)
        keys = self.record_projection(record).masked_fill(~valid[..., None], 0.0)
        any_valid = valid.any(dim=1)
        # A zero key is exposed only for empty rows, avoiding all-masked softmax.
        keys = torch.cat((keys, keys.new_zeros((b,1,cfg.model_dim))), dim=1)
        key_valid = torch.cat((valid, ~any_valid[:, None]), dim=1)
        queries = (self.queries[None] + self.history_projection(history.mean(dim=1))[:,None]
                   + self.query_projection(query)[:,None]
                   + self.proprio_projection(current_proprio)[:,None])
        attended, _ = self.pool(queries, keys, keys, key_padding_mask=~key_valid, need_weights=False)
        tokens = self.output_norm(queries + attended).masked_fill(~any_valid[:,None,None], 0.0)
        result = super().forward(warped_actions=actions, candidate_valid_mask=valid)
        return replace(result, action_context_tokens=(
            tokens[:,None].expand(-1,k,-1,-1).masked_fill(~valid[:,:,None,None],0.0)
        ))


class PooledMemoryPriorFastWAM(ConditionedPriorFastWAM):
    def configure_warm_retrospection(self, config):
        super().configure_warm_retrospection(config)
        # Extra control parameters cannot shift the common recipe's RNG stream.
        with torch.random.fork_rng(devices=[]):
            adapter = PooledMemoryEvent(self.retrospective_event_adapter.config,
                                        context_dim=self._require_retrospection().context_dim)
        self.retrospective_event_adapter = adapter.to(device=self.device, dtype=self.torch_dtype)

    def _source_only_event_inputs(self, inputs, *, context, payload, phase_features):
        return dict(inputs, event_pre_tokens=payload.effect_pre,
                    event_delta_tokens=payload.effect_delta,
                    candidate_context=context.candidate_context,
                    candidate_phase_features=phase_features,
                    query_context=context.query_context)

    def _retrospection_state(self):
        return dict(super()._retrospection_state(), schema=POOLED_SCHEMA,
                    pooled_architecture=POOL_ARCHITECTURE)

    def _validate_retrospection_state(self, value):
        if (not isinstance(value, Mapping) or value.get("schema") != POOLED_SCHEMA
                or value.get("version") != 1 or value.get("pooled_architecture") != POOL_ARCHITECTURE
                or set(value) != {"schema", "version", "config", "config_sha256", "pooled_architecture"}):
            raise ValueError("not a separately trained pooled-memory-prior checkpoint")
        translated = dict(value, schema=CONDITIONED_SCHEMA)
        del translated["pooled_architecture"]
        super()._validate_retrospection_state(translated)


def create_pooled_memory_prior(**kwargs):
    return _create_prior(PooledMemoryPriorFastWAM, **kwargs)
