"""Complete, sealed candidate actions and within-query effect diagnostics.

Branch commands follow the processor's unchanged affine backward map. The
forward feature clamp is not applied to generated native commands. These
counterfactual branches never update the factual policy's action history.
"""
from __future__ import annotations

import numpy as np


def as_array(value):
    if hasattr(value,'detach'):
        value=value.detach().cpu()
        if value.is_floating_point():value=value.float()
        value=value.numpy()
    return np.asarray(value)


def decode_branch_actions(normalizer,actions):
    actions=np.asarray(actions,dtype=np.float32)
    if actions.shape!=(32,14) or not np.isfinite(actions).all():raise ValueError('finite H32x14 proposal required')
    native=normalizer.denormalize(actions)
    if native.shape!=actions.shape or not np.isfinite(native).all():raise ValueError('invalid original affine backward commands')
    return native


def prepare_candidate_pool(probe,normalizer):
    valid=as_array(probe['arrays']['valid']);actions=as_array(probe['arrays']['adapted_actions'])
    if valid.dtype!=np.bool_ or valid.shape!=(1,32) or actions.shape!=(1,32,32,14):
        raise ValueError('exact original K32 pool and H32x14 proposals required')
    if not valid.any():raise ValueError('an empty pool cannot qualify missing-skill coverage')
    native=np.stack([decode_branch_actions(normalizer,value) for value in actions[0]])
    commands={f'rank-{rank}':native[rank] for rank in range(32) if valid[0,rank]}
    evidence=dict(candidate_slots=32,valid_candidates=int(valid.sum()),executed_rank_ids=list(commands),
        original_actions_clipped=False,native_commands_clipped=False,
        generated_values_outside_feature_clip_by_rank=[int(np.count_nonzero(np.abs(row)>normalizer.clip)) for row in actions[0]],
        decoder='Original processor finite affine backward commands, without assuming global float32 invertibility.',
        history='Counterfactual native branches do not commit factual model-space action history.')
    return commands,native,evidence


def centered_effect_metrics(row):
    """Common endpoint support; no variation means no explained-variance claim."""
    candidates=[c for c in row['candidates'] if c['valid']]
    if len({c['candidate_id'] for c in candidates})!=len(candidates):raise ValueError('duplicate candidate IDs')
    if any(c['endpoint_status']!='complete_horizon' and c.get('observed_effect') is not None for c in candidates):
        raise ValueError('incomplete endpoints must not contain imputed effects')
    complete=[c for c in candidates if c['endpoint_status']=='complete_horizon']
    result=dict(valid_candidates=len(candidates),complete_endpoints=len(complete),
        support='Identical complete-H32 candidates for all references within this query.',
        references={},observed_centered_variance=None)
    if len(complete)<2:
        result['missing_reason']='Fewer than two complete candidate endpoints';return result
    for c in complete:
        if c['executed_steps']!=c['horizon'] or c['horizon']!=32:raise ValueError('exact H32 endpoints required')
    observed=np.asarray([c['observed_effect'] for c in complete],dtype=np.float64)
    predicted=np.asarray([c['predicted_effect'] for c in complete],dtype=np.float64)
    historical=np.asarray([c['historical_effect'] for c in complete],dtype=np.float64)
    if observed.ndim!=2 or observed.shape!=predicted.shape or observed.shape!=historical.shape or not observed.shape[1]:
        raise ValueError('matched finite candidate effect vectors required')
    if not all(np.isfinite(v).all() for v in (observed,predicted,historical)):raise ValueError('nonfinite effect')
    query=np.asarray(row.get('required_effect',np.zeros(observed.shape[1])),dtype=np.float64)
    if query.shape!=(observed.shape[1],) or not np.isfinite(query).all():raise ValueError('matched query effect required')
    centered=observed-observed.mean(axis=0,keepdims=True);variance=float(np.mean(centered**2))
    result['observed_centered_variance']=variance
    for name,value in [('predicted',predicted),('historical',historical),('constant_query',np.broadcast_to(query,observed.shape))]:
        error=float(np.mean(((value-value.mean(axis=0,keepdims=True))-centered)**2))
        result['references'][name]=dict(absolute_mse=float(np.mean((value-observed)**2)),
            centered_mse=error,centered_explained_variance=1-error/variance if variance>0 else None)
    result['missing_reason']=None if variance>0 else 'No observed within-query candidate effect variation'
    result['scope']='Candidate-specific variation only. Also report absolute effect error; not causal identification from factual training.'
    return result
