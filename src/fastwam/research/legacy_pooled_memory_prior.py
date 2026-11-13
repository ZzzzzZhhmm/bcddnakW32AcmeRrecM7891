"""Explicit pooled-memory control for the audited original specialist source.

This is a new, untrained control. It neither loads the existing phase-aware
Pooled checkpoint nor changes the Full policy. Its factory binds the original
v3 implementation, without translating the shared-stage SourceAcceptance API.
"""
from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
import hashlib
import importlib
import inspect

import torch
from torch import nn
from torch.nn import functional as F

LEGACY_MODEL_SHA = "d89796cc5da67d431d911e5f9191c5d60a14599bd3763cc64198e1384dc0b67d"
LEGACY_CONSEQUENCE_SHA = "6430424190242669d23685fdded0aff75615197cac3a9534e3c52c0497a6b819"
SCHEMA = "warm.research.legacy-phase-free-pooled-control.v1"
ARCHITECTURE = "legacy-all-events-linear-four-query-attention-v1"
SOURCE_WEIGHT = 0.25
REMOVED_LOSSES = ("loss_retrieval", "loss_effect", "loss_gate", "loss_adaptation")


@lru_cache(maxsize=1)
def legacy_api():
    """Fail closed if a shared/current model was imported instead of v3."""
    model = importlib.import_module("fastwam.models.warm.retrospection_model")
    consequence = importlib.import_module("fastwam.models.warm.consequence")
    for module, expected in [(model, LEGACY_MODEL_SHA), (consequence, LEGACY_CONSEQUENCE_SHA)]:
        source = Path(inspect.getsourcefile(module))
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError("Audited legacy v3 source is required; no model API shim is permitted")
    if model.WARM_RETROSPECTION_CHECKPOINT_VERSION != 3:
        raise ValueError("The legacy pooled control requires retrospective checkpoint v3")
    return model, consequence


class LegacyPooledMemoryEvent(nn.Module):
    """Same phase-free all-record pool as the qualified small-module port."""

    def __init__(self, config, *, context_dim):
        super().__init__()
        if config.timing_dim != 4 or config.context_query_count != 4:
            raise ValueError("Legacy control requires original timing4 and four memory queries")
        self.config, self.context_dim = config, context_dim
        width = (config.action_horizon * config.action_dim + 8 * config.event_dim
                 + context_dim + config.timing_dim + 4)
        dim = config.model_dim
        self.record_projection = nn.Sequential(nn.Linear(width, dim), nn.GELU(), nn.LayerNorm(dim))
        self.history_projection = nn.Linear(config.gist_dim, dim)
        self.query_projection = nn.Linear(context_dim, dim)
        self.proprio_projection = nn.Linear(config.proprio_dim, dim)
        self.queries = nn.Parameter(torch.empty(4, dim))
        nn.init.normal_(self.queries, mean=0.0, std=0.02)
        self.pool = nn.MultiheadAttention(dim, config.num_heads, dropout=0.0, batch_first=True)
        self.output_norm = nn.LayerNorm(dim)

    @staticmethod
    def _field(value, mask, shape, name, reference):
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(shape):
            raise ValueError(f"{name} has an incompatible shape; expected {tuple(shape)}")
        if not value.is_floating_point() or value.dtype != reference.dtype or value.device != reference.device:
            raise ValueError(f"{name} must share the action dtype and device")
        result = value.masked_fill(~mask, 0.0)
        if not torch.isfinite(result).all():
            raise ValueError(f"{name} has nonfinite active values")
        return result

    def forward(self, *, warped_actions, candidate_valid_mask, event_pre_tokens,
                event_delta_tokens, event_delta_mask, candidate_context,
                gripper_timing, gist_tokens, query_context, proprio):
        cfg = self.config
        if warped_actions.ndim != 4:
            raise ValueError("warped_actions must have shape [B,K,H,D]")
        b, k, h, d = warped_actions.shape
        if b <= 0 or k <= 0 or (h, d) != (cfg.action_horizon, cfg.action_dim):
            raise ValueError("The configured full action chunk must be retained")
        valid = candidate_valid_mask
        if valid.shape != (b, k) or valid.dtype != torch.bool or valid.device != warped_actions.device:
            raise ValueError("candidate_valid_mask must be bool [B,K] on the action device")
        mask = event_delta_mask
        if mask.shape != (b, k, 4) or mask.dtype != torch.bool or mask.device != valid.device:
            raise ValueError("event_delta_mask must be bool [B,K,4] on the action device")
        mask = mask & valid[:, :, None]
        def field(value, active, shape, name):
            return self._field(value, active, shape, name, warped_actions)
        actions = field(warped_actions, valid[:, :, None, None], (b,k,h,d), "actions")
        pre = field(event_pre_tokens, mask[..., None], (b,k,4,cfg.event_dim), "event_pre")
        delta = field(event_delta_tokens, mask[..., None], pre.shape, "event_delta")
        context = field(candidate_context, valid[..., None], (b,k,self.context_dim), "candidate_context")
        timing = field(gripper_timing, valid[..., None], (b,k,4), "timing")
        history = field(gist_tokens, torch.ones_like(gist_tokens, dtype=torch.bool), (b,8,cfg.gist_dim), "gist")
        query = field(query_context, torch.ones_like(query_context, dtype=torch.bool), (b,self.context_dim), "query")
        state = field(proprio, torch.ones_like(proprio, dtype=torch.bool), (b,cfg.proprio_dim), "proprio")
        records = torch.cat((actions.flatten(-2), pre.flatten(-2), delta.flatten(-2),
                             context, timing, mask.to(actions.dtype)), dim=-1)
        keys = self.record_projection(records).masked_fill(~valid[..., None], 0.0)
        any_valid = valid.any(dim=1)
        keys = torch.cat((keys, keys.new_zeros((b,1,cfg.model_dim))), dim=1)
        key_valid = torch.cat((valid, ~any_valid[:, None]), dim=1)
        queries = (self.queries[None] + self.history_projection(history.mean(dim=1))[:,None]
                   + self.query_projection(query)[:,None] + self.proprio_projection(state)[:,None])
        attended, _ = self.pool(queries, keys, keys, key_padding_mask=~key_valid, need_weights=False)
        tokens = self.output_norm(queries + attended).masked_fill(~any_valid[:,None,None], 0.0)
        return actions, tokens


class LegacyCoarseRank(nn.Module):
    def forward(self, query, candidate_context, action_summary, effects, timing, valid):
        return -torch.arange(valid.shape[1], device=query.device, dtype=query.dtype)[None].expand(valid.shape)


class LegacyConstantGate(nn.Module):
    def forward(self, selection, deformation):
        _, consequence = legacy_api()
        probability = torch.full_like(deformation, SOURCE_WEIGHT) * selection.memory_mask
        return consequence.GateOutput(logits=torch.full_like(probability, -1.0986122886681098),
            probability=probability, features=probability[:,None], memory_mask=selection.memory_mask)


class _LegacyPooledMixin:
    def configure_warm_retrospection(self, config):
        model, _ = legacy_api()
        cfg = model._as_config(config)
        if cfg.timing_dim != 4 or any(getattr(cfg, name) != 0 for name in REMOVED_LOSSES):
            raise ValueError("Declare the original timing4 and zero losses for removed decision modules")
        super().configure_warm_retrospection(cfg)
        with torch.random.fork_rng(devices=[]):
            pool = LegacyPooledMemoryEvent(self.retrospective_event_adapter.config, context_dim=cfg.context_dim)
        self.retrospective_event_adapter = pool.to(device=self.device, dtype=self.torch_dtype)
        self.utility_reranker = LegacyCoarseRank()
        self.source_confidence_gate = LegacyConstantGate()

    def _resolve_action_source(self, *, base_gaussian, action_source_context, memory_sigma,
                              phase, final_video_tokens, world_token_streams=None,
                              text_context=None, text_context_mask=None, current_proprio=None,
                              current_video_latent=None):
        model, consequence = legacy_api()
        cfg = self._require_retrospection(); ctx = action_source_context
        if not isinstance(ctx, model.RetrospectiveSourceContext) or phase not in {"train", "infer"}:
            raise ValueError("A typed original source context and explicit train/infer phase are required")
        controls = self._online_experiment_controls()
        if phase == "infer":
            if controls.ablation_mode != "full" or controls.memory_corruption != "clean":
                raise ValueError("Legacy pooled deployment is explicitly Full/clean")
            self._warm_online_experiment_locked = True
        if world_token_streams is None or len(world_token_streams) != 2:
            raise ValueError("The original two current-frame Video DiT tap streams are required")
        if text_context is None or text_context_mask is None or current_proprio is None:
            raise ValueError("Actual text context, mask and current proprioception are required")
        early, late = world_token_streams
        b = base_gaussian.shape[0]
        if early.shape != late.shape or early.shape[0] != b or base_gaussian.shape != (b,cfg.action_horizon,cfg.action_dim):
            raise ValueError("The factual streams and Gaussian action chunk are not aligned")
        world_mask = torch.ones(early.shape[:2], device=early.device, dtype=torch.bool)
        bridge = self.semantic_bridge(early, late, world_mask)
        valid = ctx.candidate_valid_mask
        if ctx.candidate_effect_pre.shape[2] != 4 or ctx.candidate_effect_delta.shape != ctx.candidate_effect_pre.shape:
            raise ValueError("The original bank's four spatial pre/change cells are required")
        event_mask = valid[:,:,None].expand(-1,-1,4)
        if ctx.episode_action_summaries.shape[:2] != ctx.episode_action_mask.shape:
            raise ValueError("The original action-history summary/mask shapes differ")
        action_history = self.episode_action_projection(ctx.episode_action_summaries)
        action_history = action_history.masked_fill(~ctx.episode_action_mask[...,None], 0.0)
        episode = torch.cat((ctx.episode_tokens, action_history), dim=1)
        episode_mask = torch.cat((ctx.episode_mask, ctx.episode_action_mask), dim=1)
        # CUDA autocast keeps LayerNorm outputs in fp32, including the semantic
        # bridge residual. Bind the strict gist/pool interfaces to the native
        # action dtype; these casts are identities on the qualified non-AMP path.
        source_dtype = base_gaussian.dtype
        required = self.retrospective_gist(world_tokens=bridge.world_tokens.to(dtype=source_dtype), world_mask=bridge.token_mask,
            semantic_tokens=bridge.semantic_tokens.to(dtype=source_dtype), semantic_mask=torch.ones(bridge.semantic_tokens.shape[:2],device=early.device,dtype=torch.bool),
            episode_tokens=episode.to(dtype=source_dtype), episode_mask=episode_mask,
            event_pre_tokens=torch.zeros_like(ctx.candidate_effect_pre),
            event_delta_tokens=torch.zeros_like(ctx.candidate_effect_delta), event_token_mask=torch.zeros_like(event_mask),
            text_tokens=text_context.to(dtype=source_dtype), text_mask=text_context_mask)
        warped = model._warp_candidate_actions(ctx.candidate_actions,ctx.candidate_start_proprio,current_proprio,valid,cfg)
        actions, pooled = self.retrospective_event_adapter(warped_actions=warped,candidate_valid_mask=valid,
            event_pre_tokens=ctx.candidate_effect_pre,event_delta_tokens=ctx.candidate_effect_delta,
            event_delta_mask=event_mask,candidate_context=ctx.candidate_context,gripper_timing=ctx.candidate_timing,
            gist_tokens=required.gist_tokens.to(dtype=warped.dtype),query_context=ctx.query_context,proprio=current_proprio)
        scores = self.utility_reranker(ctx.query_context,ctx.candidate_context,None,None,None,valid)
        selection = consequence.select_consequence_candidate(scores,torch.zeros_like(scores),ctx.candidate_support,valid,
            consequence_weight=0.0,support_weight=0.0,automatic_null=False,forced_candidate_indices=ctx.forced_candidate_indices)
        selected = model._gather_candidate(actions,selection.candidate_indices)
        gate = self.source_confidence_gate(selection,base_gaussian.new_zeros((b,)))
        g = gate.probability.to(base_gaussian.dtype)
        source = g[:,None,None]*(selected+cfg.source_sigma_min*base_gaussian)+(1-g[:,None,None])*base_gaussian
        pooled = pooled.masked_fill(~selection.memory_mask[:,None,None],0.0)
        # This control presents all-record memory tokens directly. The fixed
        # source weight is not a learned acceptance mechanism or learned gate.
        conditioning = torch.cat((self.gist_to_text(required.gist_tokens),self.action_context_to_text(pooled)),dim=1)
        zero = base_gaussian.sum()*0.0;bridge_loss=gist_loss=zero
        if phase == "train":
            if any(getattr(ctx,k) is None for k in ['current_semantic_teacher','target_effect','target_action','target_action_valid_mask','future_valid_mask']):
                raise ValueError("Original stop-gradient training teachers are required")
            bridge_loss = self.semantic_bridge.alignment_loss(bridge,ctx.current_semantic_teacher)
            supervised = ctx.future_valid_mask & ctx.target_action_valid_mask.any(dim=1)
            if supervised.any():
                gist_loss = self.retrospective_gist.future_alignment_loss(required,model._mean_effect(ctx.target_effect),sample_mask=supervised,magnitude_weight=cfg.magnitude_weight)
        metrics = {"loss_warm_"+k:zero for k in ['retrieval','effect','gate','adaptation']}
        metrics.update(loss_warm_bridge=bridge_loss,loss_warm_gist=gist_loss,warm_gate_mean=g.mean(),
                       warm_selected_memory_rate=selection.memory_mask.float().mean(),warm_consequence_mean=zero)
        self._last_retrospection_diagnostics = dict(candidate_indices=selection.candidate_indices.detach(),gate=g.detach(),
            memory_relevance_gate=g.detach(),source_memory_mask=selection.memory_mask.detach(),
            consistency=torch.zeros_like(scores),predicted_consistency=torch.zeros_like(scores),reranker_scores=scores.detach(),
            required_transition=self.retrospective_gist.future_projection(required.pooled_gist).detach(),
            configured_ablation_mode=controls.ablation_mode,configured_memory_corruption=controls.memory_corruption,
            ablation_mode="full",memory_corruption="clean",experiment_id=controls.experiment_id,
            corruption_applied=torch.zeros_like(selection.memory_mask),corruption_fallback=torch.zeros_like(selection.memory_mask),
            pooled_control_architecture=ARCHITECTURE)
        if phase == "infer":
            if current_video_latent is None or current_video_latent.ndim != 5 or current_video_latent.shape[2] != 1:
                raise ValueError("The original factual current VAE latent is required")
            self._last_retrospection_diagnostics.update(factual_world_tokens=bridge.semantic_tokens.detach().to(device="cpu",dtype=torch.float32),
                factual_vae_latent=F.adaptive_avg_pool2d(current_video_latent[:,:,0],(4,8)).detach().to(device="cpu",dtype=torch.float32),
                factual_proprio=current_proprio.detach().to(device="cpu",dtype=torch.float32))
        return model.ActionSourceOutput(source=source,base_gaussian=base_gaussian,
            component_indices=selection.component_indices.detach(),memory_mask=selection.memory_mask.detach(),
            selected_means=selected.detach(),memory_sigma=float(cfg.source_sigma_min),conditioning_tokens=conditioning,
            auxiliary_loss=cfg.loss_bridge*bridge_loss+cfg.loss_gist*gist_loss,auxiliary_metrics=metrics,source_gate=g)

    def _retrospection_state(self):
        state = super()._retrospection_state()
        return dict(state,schema=SCHEMA,version=1,pooled_architecture=ARCHITECTURE,
                    source_weight=SOURCE_WEIGHT,original_model_sha256=LEGACY_MODEL_SHA)

    def _validate_retrospection_state(self, value):
        fields = {"schema","version","config","config_sha256","pooled_architecture","source_weight","original_model_sha256"}
        if (not isinstance(value,Mapping) or set(value)!=fields or value['schema']!=SCHEMA or value['version']!=1
                or value['pooled_architecture']!=ARCHITECTURE or value['source_weight']!=SOURCE_WEIGHT
                or value['original_model_sha256']!=LEGACY_MODEL_SHA):
            raise ValueError("A separately trained legacy phase-free pooled checkpoint is required")
        translated={k:value[k] for k in ['config','config_sha256']}
        translated.update(schema=legacy_api()[0].WARM_RETROSPECTION_CHECKPOINT_SCHEMA,version=3)
        super()._validate_retrospection_state(translated)


@lru_cache(maxsize=1)
def legacy_pooled_model_class():
    model, _ = legacy_api()
    cls=type("LegacyPooledMemoryPriorFastWAM",(_LegacyPooledMixin,model.WarmRetrospectionFastWAM),{"__module__":__name__})
    globals()[cls.__name__]=cls
    return cls


def create_legacy_pooled_memory_prior(*,retrospection,**kwargs):
    """Hydra/runtime factory, only inside a frozen original-v3 source assembly."""
    from omegaconf import DictConfig,OmegaConf
    model,_=legacy_api()
    runtime=importlib.import_module('fastwam.runtime')
    if isinstance(retrospection,DictConfig):retrospection=OmegaConf.to_container(retrospection,resolve=True)
    cfg=model._as_config(retrospection)
    if kwargs.pop('source_policy','fixed_context_top1')!='fixed_context_top1':
        raise ValueError('Legacy pooled control requires the original coarse retrieval family')
    return runtime.create_warm_source(source_policy='fixed_context_top1',_warm_model_class=legacy_pooled_model_class(),
        _warm_pretrained_extra={'warm_retrospection_config':cfg},**kwargs)
