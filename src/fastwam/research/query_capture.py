"""Capture a genuine online replan for sealed, same-context branch generation.

Wrappers delegate exactly once and are removed before any simulator branch.
The counterfactual uses the production FastWAM sampler with the exact captured
factual context, independently configured null weights, and common Gaussian.
It does not replay a consumed BoundOnlineStep capability.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import fields,replace
from copy import deepcopy
import random
import numpy as np

from .policy_state import deployed_policy_fingerprint


@contextmanager
def isolated_generation_rng():
    """A diagnostic generation must not advance the factual rollout's RNG."""
    import torch
    python_state=random.getstate();numpy_state=np.random.get_state()
    devices=list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    with torch.random.fork_rng(devices=devices):
        try:yield
        finally:
            random.setstate(python_state);np.random.set_state(numpy_state)


def clone_inference_context(context):
    for name in ('target_action','target_effect','current_semantic_teacher','target_action_valid_mask','future_valid_mask'):
        if getattr(context,name,None) is not None:
            raise ValueError('future/teacher values are forbidden in an inference branch: '+name)
    values={}
    for field in fields(context):
        value=getattr(context,field.name)
        values[field.name]=value.detach().clone() if hasattr(value,'detach') else deepcopy(value)
    return replace(context,**values)


@contextmanager
def capture_online_replan(policy):
    model=policy.model
    original_infer=model.infer_action;original_source=model._resolve_action_source
    had_infer='infer_action' in vars(model);had_source='_resolve_action_source' in vars(model)
    record={'source_calls':0,'infer_calls':0}
    def source(*args,**kwargs):
        record['source_calls']+=1
        record['context']=clone_inference_context(kwargs['action_source_context'])
        return original_source(*args,**kwargs)
    def infer(*args,**kwargs):
        record['infer_calls']+=1
        record['online_step']=kwargs['online_step']
        record['sampler_options']={k:kwargs[k] for k in ('negative_prompt','text_cfg_scale','num_inference_steps',
            'sigma_shift','rand_device','tiled') if k in kwargs}
        output=original_infer(*args,**kwargs)
        action=output['action']
        record['action']=action.detach().float().cpu().numpy().copy() if hasattr(action,'detach') else np.array(action,copy=True)
        record['probe']=deepcopy(output['warm_research_probe'])
        return output
    model.infer_action=infer;model._resolve_action_source=source
    try:
        yield record
        if record['infer_calls']!=1 or record['source_calls']!=1:
            raise RuntimeError('capture must correspond to exactly one actual online replan')
    finally:
        if had_infer:model.infer_action=original_infer
        else:del model.infer_action
        if had_source:model._resolve_action_source=original_source
        else:del model._resolve_action_source


def generate_captured_null(full_policy,null_policy,capture):
    import torch
    from fastwam.models.wan22.fastwam import FastWAM

    if full_policy.checkpoint_sha256!=null_policy.checkpoint_sha256:
        raise ValueError('counterfactual weights differ')
    controls=getattr(null_policy.model,'_warm_research_probe',None)
    if controls is None or not controls.force_null or not controls.capture:
        raise ValueError('independent null model must be configured before first use')
    model=null_policy.model;step=capture['online_step']
    # This step is read for provenance, never passed through the consuming
    # online infer API again. Its factual tensors were already validated.
    full_policy.retriever.assert_owned_bound_step(step)
    before=deployed_policy_fingerprint(full_policy)
    with isolated_generation_rng(),torch.inference_mode():
        output=FastWAM.infer_action(model,prompt=step.prompt,
            input_image=torch.tensor(step.model_input,device=model.device,dtype=model.torch_dtype),
            action_horizon=32,proprio=torch.tensor(step.proprio,device=model.device,dtype=model.torch_dtype),
            action_source_context=capture['context'],seed=int(step.derived_seed),
            memory_sigma=model._require_retrospection().source_sigma_min,**capture['sampler_options'])
    probe=deepcopy(model._last_research_probe)
    if not torch.equal(capture['probe']['arrays']['base_gaussian'],probe['arrays']['base_gaussian']):
        raise RuntimeError('counterfactual Gaussian differs from the sealed Full query')
    if bool(probe['arrays']['g'].any()):raise RuntimeError('counterfactual did not withdraw retrieved influence')
    if deployed_policy_fingerprint(full_policy)!=before:raise RuntimeError('null generation mutated the Full policy state')
    action=output['action'].detach().float().cpu().numpy().copy()
    if action.shape!=(32,14) or not np.isfinite(action).all():raise RuntimeError('invalid null action chunk')
    return {'action':action,'probe':probe,'full_policy_state_sha256':before,'derived_seed':int(step.derived_seed),'factual_rng_isolated':True}
