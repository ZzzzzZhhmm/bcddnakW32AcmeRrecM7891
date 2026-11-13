"""Prospective bindings for two original-PB q8 explicit withdrawal controls.

The existing namespace source/pool jobs remain immutable and own their original
outputs. This separate study must reproduce the recorded factual prefix before
constructing a new action intervention; it never supplies zero history.
"""
from pathlib import Path
import datetime
import hashlib
import importlib.util
import json

CORE = '4a1d05cb548f16bf6a9b239082b02d4697aa8e8c1f43c2f5d8ffbaec196462fd'
WEIGHT = 'a765d987b2df0d228782980592c39aaed0e837040c7f6f48c905e4641f79c285'
SIM = 'a97e308ca053c3aeb1bfec8e3c4c4f550aac7f6ae753e36bbce82e9522292379'
SCHEMA = 'warm.PB-namespace2-q8-explicit-null.v1'
PRIOR_RESERVED = 140.3054115725


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def reference(item):
    path = Path(item['path'])
    if sha(path) != item['sha256']:
        raise ValueError('Frozen original input changed: ' + str(path))
    return path


def namespace_module():
    spec = importlib.util.spec_from_file_location('PB_null_namespace_binding', Path(__file__).with_name('pb_namespace.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def deadline(first, latest):
    if first.tzinfo is None or latest.tzinfo is None:
        raise ValueError('Timezone-aware resource window required')
    result = min(first + datetime.timedelta(hours=3.5), latest)
    if (result-first).total_seconds() < 1800:
        raise ValueError('Insufficient unchanged writing window')
    return result


def case_deadline(first, stage_end):
    """Keep three minutes inside the same allocation for exit and raw sealing."""
    if first.tzinfo is None or stage_end.tzinfo is None:
        raise ValueError('Timezone-aware resource window required')
    result = min(first + datetime.timedelta(seconds=6000), stage_end-datetime.timedelta(seconds=180))
    if (result-first).total_seconds() < 1800:
        raise ValueError('Insufficient unchanged case window and archive margin')
    return result


def inference_context_payload(context):
    """Save complete inference tensors while rejecting every future target."""
    import dataclasses
    import torch
    if not dataclasses.is_dataclass(context):
        raise ValueError('Actual typed original source context required')
    values = {f.name:getattr(context,f.name) for f in dataclasses.fields(context)}
    forbidden = {'target_action','target_effect','current_semantic_teacher',
                 'target_action_valid_mask','future_valid_mask'}
    if any(values.get(key) is not None for key in forbidden):
        raise ValueError('Inference context contains a future target')
    required = {'query_context','candidate_context','candidate_actions','candidate_start_proprio',
                'candidate_effect_pre','candidate_effect_delta','candidate_timing','candidate_support',
                'candidate_valid_mask','episode_tokens','episode_mask',
                'episode_action_summaries','episode_action_mask'}
    if not required.issubset(values) or any(not torch.is_tensor(values[k]) for k in required):
        raise ValueError('Actual past history or candidate tensors are missing')
    if any(value is not None and not torch.is_tensor(value) for value in values.values()):
        raise ValueError('Unknown context field cannot be replaced by a type description')
    result = {key:value.detach().cpu().clone() for key,value in values.items() if torch.is_tensor(value)}
    if any(x.is_floating_point() and not bool(torch.isfinite(x).all()) for x in result.values()):
        raise ValueError('Nonfinite inference context')
    if tuple(result['candidate_actions'].shape)!=(1,32,32,14) or tuple(result['episode_tokens'].shape)!=(1,16,768):
        raise ValueError('Original v3 candidate/history tensor layout differs')
    return result


def validate_plan(plan, *, verify_inputs=True):
    expected = dict(schema=SCHEMA, case_ids=[0, 1], query=8, frame=32,
                    horizon=32, execution_prefix=4, checkpoint_step=12000,
                    original_source_sha256=CORE, checkpoint_sha256=WEIGHT,
                    simulator_source_sha256=SIM, automatic_restarts=0,
                    maximum_seconds=12600, maximum_case_seconds=6000,
                    additional_reserved_GPU_hours=3.5,
                    prior_reserved_GPU_hours=PRIOR_RESERVED,
                    cumulative_reserved_GPU_hours=143.8054115725,
                    lifetime_cap_GPU_hours=144, max_concurrent_GPUs=6,
                    original_windows_unchanged=True, policy_outcome_based_selection=False,
                    effect_descriptor='original_uniform_mean_DINO_delta_v1',
                    next_interaction_rubric='warm.rmbench.PB-counted-interaction.v1',
                    minimum_admission_free_bytes=4*1024**3,
                    runtime_free_space_guard_bytes=512*1024**2)
    if any(type(plan.get(k)) is not type(v) or plan.get(k) != v for k, v in expected.items()):
        raise ValueError('Fixed scope, original model, descriptor or reservation differs')
    if plan['latest_end_utc'] != '2026-11-13T11:00:00+00:00':
        raise ValueError('Original writing cutoff changed')
    if len(plan['cases']) != 2 or [x['case_id'] for x in plan['cases']] != [0, 1]:
        raise ValueError('Both fixed cases, in order, are required')
    root = Path(plan['root'])
    if not root.is_absolute() or root.name != 'PB_namespace2_null_20261113_v1':
        raise ValueError('Independent output root required')
    original_roots = set()
    for row in plan['cases']:
        if type(row['case_id']) is not int or type(row['physical_seed']) is not int:
            raise ValueError('Integer original case and physical seed required')
        d = root/f"case{row['case_id']:02d}"
        if Path(row['output']) != d/'results' or Path(row['runtime']) != d/'runtime' or Path(row['lease']) != d/'native.worker.lock':
            raise ValueError('Null case private output/runtime/lease differs')
        if verify_inputs:
            factual = read(reference(row['factual_plan']))
            _, entry = namespace_module().validate_execution_binding(factual, 'factual')
            if entry['case_id'] != row['case_id'] or entry['seed'] != row['physical_seed']:
                raise ValueError('Wrong fixed source case or physical seed')
            source = Path(factual['output'])
            if source == d/'results' or source.is_relative_to(root) or Path(row['source']) != source:
                raise ValueError('Original source is not separately protected')
            op = read(reference(row['factual_operator']))
            required = dict(status='complete_fixed_original_factual_episode',
                            namespace_case_id=row['case_id'], candidate_seed=row['physical_seed'],
                            source_sha256=CORE, source_sha256_after=CORE, checkpoint_sha256=WEIGHT,
                            plan_sha256=row['factual_plan']['sha256'], model_updates=0,
                            initial_observation_exact_to_qualification=True,
                            first_replan_exact_to_qualification=True,
                            original_namespace_initial_identity_exact=True)
            if any(op.get(k) != v for k, v in required.items()):
                raise ValueError('Incomplete or foreign factual source')
            if op['native_commands'] < 36 or (op['native_commands'] != 1500 and not op['success_flag']):
                raise ValueError('Fixed q8 is unreached or source is incomplete')
            if op['probe_status'].get('8') != 'sealed_before_execution':
                raise ValueError('Original q8 was not sealed before execution')
            original_roots.add(source)
            for item in row['inputs']:
                reference(item)
    if verify_inputs and len(original_roots) != 2:
        raise ValueError('Two different predeclared sources required')
    for item in plan.get('inputs', []):
        if verify_inputs:
            reference(item)
    return plan
