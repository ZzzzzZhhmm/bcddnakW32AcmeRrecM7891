"""Inference-only coordinated gate withdrawal for the frozen v3 specialists.

This changes only the explicitly intervened gate probability. The source and
conditioning are checked against the original query-only gist and Gaussian.
It neither changes a checkpoint nor claims that the natural gate withdrew.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import fields, is_dataclass, replace
import hashlib
import inspect
from types import MethodType
from pathlib import Path

import torch

LEGACY_MODEL_SHA256 = 'd89796cc5da67d431d911e5f9191c5d60a14599bd3763cc64198e1384dc0b67d'


def _qualified_model(model):
    if torch.is_grad_enabled():
        raise ValueError('enter under inference_mode or no_grad before the intervention')
    if model.training or any(m.training for m in model.modules()):
        raise ValueError('an eval-only original model is required')
    state = model._retrospection_state()
    if state.get('version') != 3 or model._require_retrospection().timing_dim != 4:
        raise ValueError('only the original v3/timing4 model is supported')
    method = model._resolve_action_source
    path = inspect.getsourcefile(method)
    if path is None or hashlib.sha256(Path(path).read_bytes()).hexdigest() != LEGACY_MODEL_SHA256:
        raise ValueError('original source-resolution implementation differs')
    if '_resolve_action_source' in model.__dict__:
        raise ValueError('a preexisting source-resolution override is not allowed')
    controls = model._online_experiment_controls()
    if controls.ablation_mode != 'full' or controls.memory_corruption != 'clean':
        raise ValueError('the fixed Full/clean configuration is required')


@contextmanager
def legacy_candidate_withdrawal(model):
    """Temporarily intervene on one original-source evaluation, then restore it.

    Yield a witness containing observed natural/intervened gate tensors and
    checked source/conditioning tensors. Selection metadata stays unchanged;
    zero influence, rather than an empty retrieved bank, defines this control.
    Call exactly one inference while inside this scope. Exceptions remove all
    hooks without replacing the original exception or resetting online locks.
    """
    _qualified_model(model)
    modules = [model.source_confidence_gate, model.retrospective_gist,
               model.gist_to_text, model.action_context_to_text]
    if any(m._forward_hooks or m._forward_pre_hooks for m in modules):
        raise ValueError('preexisting hooks would make the intervention ambiguous')
    versions = [(name, p, int(p._version)) for name, p in
                list(model.named_parameters()) + list(model.named_buffers())]
    witness = dict(schema='warm.original-v3-explicit-candidate-withdrawal.v1',
                   natural_withdrawal_certified=False, gate_calls=0,
                   source_calls=0, source_implementation_sha256=LEGACY_MODEL_SHA256)
    cache = {'gist': [], 'gist_projection': [], 'action_projection': []}
    handles = []
    original = model._resolve_action_source

    def close_gate(_module, _args, output):
        if torch.is_grad_enabled() or not is_dataclass(output):
            raise ValueError('inference-only typed GateOutput required')
        if {f.name for f in fields(output)} != {'logits', 'probability', 'features', 'memory_mask'}:
            raise ValueError('original GateOutput layout differs')
        p = output.probability
        if (not isinstance(p, torch.Tensor) or p.ndim != 1 or not p.numel()
                or not p.is_floating_point() or not torch.isfinite(p).all()
                or torch.any(p < 0) or torch.any(p > 1)):
            raise ValueError('invalid natural gate probability must not be concealed')
        witness['gate_calls'] += 1
        if witness['gate_calls'] != 1:
            raise RuntimeError('only one fixed inference is authorized')
        witness['natural_probability'] = p.detach().cpu().clone()
        witness['natural_logits'] = output.logits.detach().cpu().clone()
        witness['selection_memory_mask'] = output.memory_mask.detach().cpu().clone()
        zero = torch.zeros_like(p)
        witness['intervened_probability'] = zero.detach().cpu().clone()
        return replace(output, probability=zero)

    def observe_gist(_module, _args, kwargs, output):
        if len(cache['gist']) >= 2:
            raise RuntimeError('original required/predictive gist call layout differs')
        if not cache['gist']:
            for k in ['event_pre_tokens', 'event_delta_tokens', 'event_token_mask']:
                if k not in kwargs or torch.count_nonzero(kwargs[k]).item() != 0:
                    raise RuntimeError('required gist received candidate information')
        cache['gist'].append(output.gist_tokens.detach().clone())

    def projection_observer(key):
        def hook(_module, args, output):
            cache[key].append((args[0].detach().clone(), output.detach().clone()))
        return hook

    def checked_source(_self, *args, **kwargs):
        if kwargs.get('phase') != 'infer':
            raise ValueError('only inference source resolution is authorized')
        witness['source_calls'] += 1
        if witness['source_calls'] != 1:
            raise RuntimeError('only one fixed inference is authorized')
        value = original(*args, **kwargs)
        if witness['gate_calls'] != 1:
            raise RuntimeError('source/gate call accounting differs')
        if len(cache['gist']) != 2 or any(len(cache[k]) != 1 for k in ['gist_projection','action_projection']):
            raise RuntimeError('original conditioning construction differs')
        required = cache['gist'][0]
        projected_input, projected_required = cache['gist_projection'][0]
        action_projection = cache['action_projection'][0][1]
        if not torch.equal(projected_input, required):
            raise RuntimeError('withdrawal did not preserve query-only gist')
        expected = torch.cat((projected_required, torch.zeros_like(action_projection)), dim=1)
        if (not torch.equal(value.source, value.base_gaussian)
                or torch.count_nonzero(value.source_gate).item() != 0
                or not torch.equal(value.conditioning_tokens, expected)):
            raise RuntimeError('candidate influence remains in source or conditioning')
        witness.update(source_exactly_base_gaussian=True,
                       conditioning_exactly_query_gist_and_zero_candidate_tokens=True,
                       factual_recollection_retained=True,
                       source=value.source.detach().cpu().clone(),
                       base_gaussian=value.base_gaussian.detach().cpu().clone(),
                       conditioning_tokens=value.conditioning_tokens.detach().cpu().clone(),
                       query_only_projected_gist=projected_required.detach().cpu().clone(),
                       zero_candidate_token_count=int(action_projection.shape[1]))
        return value

    try:
        handles.append(model.source_confidence_gate.register_forward_hook(close_gate))
        handles.append(model.retrospective_gist.register_forward_hook(observe_gist, with_kwargs=True))
        handles.append(model.gist_to_text.register_forward_hook(projection_observer('gist_projection')))
        handles.append(model.action_context_to_text.register_forward_hook(projection_observer('action_projection')))
        model._resolve_action_source = MethodType(checked_source, model)
        yield witness
        if witness['source_calls'] != 1 or witness['gate_calls'] != 1:
            raise RuntimeError('the authorized fixed inference was not completed')
        for name, p, version in versions:
            if int(p._version) != version:
                raise RuntimeError('intervention wrote a model parameter or buffer: ' + name)
    finally:
        for handle in handles:
            handle.remove()
        if '_resolve_action_source' in model.__dict__:
            del model._resolve_action_source
