"""Separately trained coarse-retrieval prior with the same factual recollection.

This experimental class has no learned event adaptation, consequence predictor,
utility reranker or confidence gate. A fixed .25 source weight is declared before
new outcomes. Its checkpoint schema prevents loading it as the submitted model.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import math
import torch
from torch import nn

from fastwam.models.warm.consequence import AUTO_SELECT,GateOutput,SourceAcceptance
from fastwam.models.warm.retrospective_event_adapter import RetrospectiveEventAdapter,RetrospectiveEventAdapterOutput
from fastwam.models.warm.retrospection_model import (
    WarmRetrospectionFastWAM,WARM_RETROSPECTION_CHECKPOINT_SCHEMA,WARM_RETROSPECTION_CHECKPOINT_VERSION)

SCHEMA='warm.research.coarse-prior-with-recollection.v1'
SOURCE_WEIGHT=.25
CONDITIONED_SCHEMA='warm.research.coarse-conditioned-prior-with-recollection.v1'


class CoarseRank(nn.Module):
    def forward(self,query,candidate_context,action_summary,effects,timing,valid):
        return -torch.arange(valid.shape[1],device=query.device,dtype=query.dtype)[None].expand(valid.shape)


class ConstantGate(nn.Module):
    def forward(self,selection,deformation,stagnation):
        probability=torch.full_like(deformation,SOURCE_WEIGHT)*selection.memory_mask
        logits=torch.full_like(probability,math.log(SOURCE_WEIGHT/(1-SOURCE_WEIGHT)))
        return GateOutput(logits=logits,probability=probability,
            features=torch.zeros((*probability.shape,1),device=probability.device,dtype=probability.dtype),
            memory_mask=selection.memory_mask)


class DeterministicEvent(nn.Module):
    """Only the common deterministic start-state warp, without learned residual."""
    def __init__(self,config):
        super().__init__();self.config=config

    def forward(self,*,warped_actions,candidate_valid_mask,**unused):
        actions=warped_actions.masked_fill(~candidate_valid_mask[:,:,None,None],0)
        b,k=actions.shape[:2];cfg=self.config
        return RetrospectiveEventAdapterOutput(adapted_action_mean=actions,
            action_residual=torch.zeros_like(actions),predicted_effect=actions.new_zeros((b,k,cfg.effect_dim)),
            action_context_tokens=actions.new_zeros((b,k,cfg.context_query_count,cfg.model_dim)),
            candidate_valid_mask=candidate_valid_mask)

    def predict_effects(self,*,actions,**unused):
        return actions.new_zeros((*actions.shape[:2],self.config.effect_dim))

    # Parent diagnostics remain defined but their coefficients must be zero.
    effect_alignment_loss=RetrospectiveEventAdapter.effect_alignment_loss


class SimplePriorFastWAM(WarmRetrospectionFastWAM):
    def configure_warm_retrospection(self,config):
        super().configure_warm_retrospection(config)
        cfg=self._require_retrospection()
        if any(getattr(cfg,k)!=0 for k in ('loss_retrieval','loss_effect','loss_gate','loss_adaptation')):
            raise ValueError('simple-prior recipe must disable losses for removed modules')
        self.retrospective_event_adapter=DeterministicEvent(self.retrospective_event_adapter.config)
        self.utility_reranker=CoarseRank();self.source_confidence_gate=ConstantGate()

    def _source_ablation_mode(self,phase,configured_mode):
        return 'source_only_no_consequence'

    def _source_candidate_indices(self,context):
        valid=context.candidate_valid_mask
        first=valid.long().argmax(dim=1)
        first=torch.where(valid.any(dim=1),first,torch.full_like(first,-1))
        supplied=context.forced_candidate_indices
        return first if supplied is None else torch.where(supplied==AUTO_SELECT,first,supplied)

    def _source_acceptance(self,acceptance,*,selection,thread_source_eligible,forced_rejection):
        eligible=selection.memory_mask & thread_source_eligible & ~forced_rejection
        probability=eligible.to(dtype=acceptance.effective_probability.dtype)*SOURCE_WEIGHT
        return SourceAcceptance(effective_probability=probability,quality=eligible.to(probability.dtype),accepted_mask=eligible)

    def _source_only_conditioning(self,required_gist,selected_action_context):
        history=self.gist_to_text(required_gist.gist_tokens)
        zeros=history.new_zeros((history.shape[0],selected_action_context.shape[1],history.shape[2]))
        return torch.cat((history,zeros),dim=1)

    def configure_trainable_modules(self):
        super().configure_trainable_modules()
        # These projections feed only the removed ranking/action-context paths.
        for name in ('episode_query_projection','candidate_phase_projection','action_context_to_text'):
            getattr(self,name).requires_grad_(False)
        return tuple(p for p in self.parameters() if p.requires_grad)

    def _retrospection_state(self):
        result=super()._retrospection_state();result.update(schema=SCHEMA,version=1)
        return result

    def _validate_retrospection_state(self,value):
        if not isinstance(value,Mapping) or value.get('schema')!=SCHEMA or value.get('version')!=1:
            raise ValueError('not a separately trained coarse-prior checkpoint')
        translated=dict(value,schema=WARM_RETROSPECTION_CHECKPOINT_SCHEMA,version=WARM_RETROSPECTION_CHECKPOINT_VERSION)
        super()._validate_retrospection_state(translated)


class DirectActionEvent(DeterministicEvent):
    """Lossless fixed tokenization of the same normalized, warped HxD action."""
    def forward(self,*,warped_actions,candidate_valid_mask,**unused):
        result=super().forward(warped_actions=warped_actions,candidate_valid_mask=candidate_valid_mask)
        b,k,h,d=warped_actions.shape;count=self.config.context_query_count
        width=(h*d+count-1)//count
        if width>self.config.model_dim:raise ValueError('direct action tokenization must retain every action value')
        flat=torch.nn.functional.pad(result.adapted_action_mean.flatten(-2),(0,count*width-h*d))
        tokens=flat.reshape(b,k,count,width)
        tokens=torch.nn.functional.pad(tokens,(0,self.config.model_dim-width))
        return replace(result,action_context_tokens=tokens)


class ConditionedPriorFastWAM(SimplePriorFastWAM):
    """A stronger simple control: coarse prior plus direct retrieved-action tokens.

    Keeps the same four retrieved-action token slots and eight history gist
    slots as Full. It still has no learned action adaptation, consequence model,
    reranker or confidence gate. This is a declared common-backbone control,
    not a reproduction of a named prior method.
    """
    def configure_warm_retrospection(self,config):
        super().configure_warm_retrospection(config)
        self.retrospective_event_adapter=DirectActionEvent(self.retrospective_event_adapter.config)
        nn.init.zeros_(self.action_context_to_text.weight)

    def _source_only_conditioning(self,required_gist,selected_action_context):
        return torch.cat((self.gist_to_text(required_gist.gist_tokens),
                          self.action_context_to_text(selected_action_context)),dim=1)

    def configure_trainable_modules(self):
        super().configure_trainable_modules()
        self.action_context_to_text.requires_grad_(True)
        return tuple(p for p in self.parameters() if p.requires_grad)

    def _retrospection_state(self):
        return dict(super()._retrospection_state(),schema=CONDITIONED_SCHEMA)

    def _validate_retrospection_state(self,value):
        if not isinstance(value,Mapping) or value.get('schema')!=CONDITIONED_SCHEMA or value.get('version')!=1:
            raise ValueError('not a separately trained coarse-conditioned-prior checkpoint')
        super()._validate_retrospection_state(dict(value,schema=SCHEMA))


def _create_prior(model_class,*,retrospection,**kwargs):
    from omegaconf import OmegaConf,DictConfig
    from fastwam.runtime import create_warm_source
    from fastwam.models.warm.retrospection_config import WarmRetrospectionConfig
    if isinstance(retrospection,DictConfig):retrospection=OmegaConf.to_container(retrospection,resolve=True)
    cfg=retrospection if isinstance(retrospection,WarmRetrospectionConfig) else WarmRetrospectionConfig.from_dict(retrospection)
    policy=kwargs.pop('source_policy','fixed_context_top1')
    if policy!='fixed_context_top1':raise ValueError('simple prior requires the same coarse retrieval family')
    return create_warm_source(source_policy=policy,_warm_model_class=model_class,
        _warm_pretrained_extra={'warm_retrospection_config':cfg},**kwargs)


def create_simple_prior(**kwargs):
    return _create_prior(SimplePriorFastWAM,**kwargs)


def create_conditioned_prior(**kwargs):
    return _create_prior(ConditionedPriorFastWAM,**kwargs)
