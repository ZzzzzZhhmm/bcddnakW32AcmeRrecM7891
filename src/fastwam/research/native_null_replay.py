"""Fail-closed checks for replaying an authenticated native inference context.

This path reads sealed factual tensors and calls the production sampler. It
does not construct, consume, or pretend to own a historical BoundOnlineStep.
"""
from dataclasses import MISSING, fields
import numpy as np

FORBIDDEN = frozenset({'target_action', 'target_effect', 'current_semantic_teacher',
                       'target_action_valid_mask', 'future_valid_mask'})
ACTUAL = frozenset({'actual_base_gaussian', 'actual_source', 'actual_conditioning_tokens'})


def restore_factual_context(payload, context_type, *, device):
    import torch
    names = {field.name for field in fields(context_type)}
    if set(payload) - names - ACTUAL or FORBIDDEN.intersection(payload):
        raise ValueError('saved inference context contains unknown or future fields')
    required = {field.name for field in fields(context_type)
                if field.default is MISSING and field.default_factory is MISSING}
    if not required.issubset(payload) or not ACTUAL.issubset(payload):
        raise ValueError('saved factual context or actual source witness is incomplete')
    if not all(torch.is_tensor(value) for value in payload.values()):
        raise ValueError('saved context values must be authenticated tensors')
    values = {key: value.detach().clone().to(device=device)
              for key, value in payload.items() if key in names}
    if any(value.is_floating_point() and not bool(torch.isfinite(value).all())
           for value in values.values()):
        raise ValueError('nonfinite factual context')
    return context_type(**values)


def require_reproduced_full(action, original_action, probe, original_probe, payload):
    import torch
    same = lambda a, b: a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    left = np.asarray(action)
    right = np.asarray(original_action).reshape(32, 14)
    if (left.shape != (32, 14) or left.dtype != np.float32 or right.dtype != np.float32
            or not np.isfinite(left).all() or not np.array_equal(left, right)):
        raise ValueError('replayed Full H32 differs from its original sealed generation')
    if probe['force_null'] or probe['source_mode'] != 'full':
        raise ValueError('Full reproduction used an intervention')
    for name in ('base_gaussian', 'source', 'conditioning', 'g', 'alpha',
                 'selected_index', 'predicted_effect', 'historical_effect', 'required_effect'):
        if not same(probe['arrays'][name], original_probe['arrays'][name]):
            raise ValueError('replayed Full probe differs: ' + name)
    for name, witness in [('base_gaussian', 'actual_base_gaussian'),
                          ('source', 'actual_source'), ('conditioning', 'actual_conditioning_tokens')]:
        if not same(probe['arrays'][name], payload[witness]):
            raise ValueError('replayed Full differs from actual source witness: ' + witness)


def require_explicit_null_probe(probe, original_probe):
    import torch
    a = probe['arrays']
    if not probe['force_null'] or probe['source_mode'] != 'full':
        raise ValueError('independent null mode was not configured before inference')
    if not torch.equal(a['base_gaussian'], original_probe['arrays']['base_gaussian']):
        raise ValueError('null Gaussian differs from original Full')
    if bool(a['g'].any()) or bool(a['source_gate'].any()):
        raise ValueError('null effective source gates are not zero')
    if not torch.equal(a['source'], a['base_gaussian']):
        raise ValueError('null source is not exact Gaussian')
    tokens = a['conditioning']
    if (tuple(tokens.shape) != (1, 12, 4096) or not bool(torch.isfinite(tokens).all())
            or bool(tokens[:, 8:].any()) or not bool(tokens[:, :8].any())):
        raise ValueError('null must retain required gist and remove all four candidate slots')
    return dict(explicit_zero_influence_verified=True, natural_skill_absence_certified=False,
                same_factual_context_required=True, candidate_conditioning_slots=4)
