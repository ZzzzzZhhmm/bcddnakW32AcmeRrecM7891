"""Bind a same-Full-history null action seal to existing fresh-scene evidence."""
from __future__ import annotations

import numpy as np


def require_null_binding(seal, generation, *, checkpoint_sha256, plan_sha256,
                         core_source_sha256, runtime_source_sha256, generation_file_sha256,
                         model_actions_file_sha256, native_actions_file_sha256,
                         context_file_sha256, telemetry_file_sha256, task, seed, episode, query):
    expected=dict(checkpoint_sha256=checkpoint_sha256,plan_sha256=plan_sha256,
        core_source_sha256=core_source_sha256,task=task,seed=seed,episode_id=episode,
        query_ordinal=query,frame_index=query*4,phase='formal',horizon=32,execution_prefix=4,
        original_generation_sha256=generation_file_sha256,original_model_actions_sha256=model_actions_file_sha256,
        native_actions_sha256=native_actions_file_sha256,factual_context_sha256=context_file_sha256,
        original_telemetry_sha256=telemetry_file_sha256)
    if any(seal.get(key)!=value for key,value in expected.items()):
        raise ValueError('null seal differs from the fixed factual source/query/artifact')
    if seal['simulator_source']['sha256']!=runtime_source_sha256 or generation['array_sha256']!=model_actions_file_sha256:
        raise ValueError('null seal runtime or original generation file differs')
    flags=('exact_original_full_H32','exact_original_query_RGB_and_drive_target','same_context_and_Gaussian',
           'factual_rng_isolated','before_any_generated_null_execution')
    if any(seal.get(key) is not True for key in flags) or seal.get('counterfactual_branches_executed')!=0:
        raise ValueError('null actions require an exact pre-execution factual seal')
    proof=seal['binding_comparison']
    if (proof['comparison_bound_step_sha256']!=generation['bound_step_sha256'] or
        proof['original_bound_step_sha256']!=generation['bound_step_sha256'] or
        proof['actual_bound_step_sha256']!=seal['bound_step_sha256'] or not proof['actual_capability_unmodified']):
        raise ValueError('null seal binding proof differs')
    for key in ('candidate_payload_sha256','model_input_sha256','derived_seed'):
        if seal[key]!=generation[key]:raise ValueError('null seal scientific input differs: '+key)


def require_native_null_actions(actions, recorded_refined):
    if set(actions)!= {'refined','candidate_null'}:raise ValueError('exact refined/null action pair required')
    for value in (*actions.values(),recorded_refined):
        value=np.asarray(value)
        if value.dtype!=np.float32 or value.shape!=(32,14) or not np.isfinite(value).all():
            raise ValueError('finite native float32 H32x14 required')
    if not np.array_equal(actions['refined'],recorded_refined):
        raise ValueError('sealed Full H32 differs from original fresh-scene refined actions')
    # Equality between policies is a result, never a reason to replace a query.
    return {key:value.copy() for key,value in actions.items()}
