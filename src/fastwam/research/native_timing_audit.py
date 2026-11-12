"""Read logged action-ready wall times without a model or simulator.

These descriptive collector timings include evidence capture. They exclude
model loading and physical execution, and do not isolate a module's overhead.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

WEIGHTS = {'full': 'b0669047fbcb0f8f79fd25fea2de25f689cfd286d85ae899872c9b4435fff27b',
           'pooled_memory_prior': '9c9c9124bd4cb999b47b6bfd866b977c4c52357cc1af25c916bde68fa30b42fb'}
CORE = '87199cc6f815da5599fcb0ee9c0a9ffb67181f3750e34d16acc810f6218ffe80'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def finite_nonnegative(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('Invalid logged duration; do not filter or impute it')
    return float(value)


def quantile(values, probability):
    """Linear quantile, including both endpoints; no query-level interval."""
    ordered = sorted(values)
    if not ordered or not 0 <= probability <= 1:
        raise ValueError('Nonempty values and a probability are required')
    p = (len(ordered) - 1) * probability
    a = int(p); b = min(a + 1, len(ordered) - 1)
    return ordered[a] + (p - a) * (ordered[b] - ordered[a])


def summarize(values):
    if not values:
        return None
    return dict(n=len(values), minimum_s=min(values), median_s=quantile(values,.5),
                p95_s=quantile(values,.95), maximum_s=max(values), mean_s=sum(values)/len(values))


def audit_role(binding):
    for key in ('result','telemetry','exit'):
        ref = binding[key]
        if sha(ref['path']) != ref['sha256']:
            raise ValueError('Frozen raw input differs: ' + key)
    r = read(binding['result']['path']); exit_record = read(binding['exit']['path'])
    role = r['role']; task = r['task']; case = r['episode_id']
    if (r['status'] != 'complete_native' or exit_record['exit_code'] != 0
            or exit_record.get('bounded_termination',False)
            or role not in WEIGHTS or r['checkpoint_sha256'] != WEIGHTS[role]
            or r['evaluation_source_sha256'] != CORE
            or [task,case,role] != binding['identity']):
        raise ValueError('Only the frozen complete native roles are eligible')
    limit = {'battery_try':1000,'rearrange_blocks':700}[task]
    if (type(r['success']) is not int or r['success'] not in (0,1)
            or not 0 < r['executed_commands'] <= limit
            or (r['success']==0 and r['executed_commands']!=limit)):
        raise ValueError('Incomplete endpoint cannot supply a complete-role timing series')
    events = [json.loads(line) for line in Path(binding['telemetry']['path']).read_text(encoding='utf-8').splitlines()]
    headers = [v for v in events if v['kind']=='header']; replans = [v for v in events if v['kind']=='replan']
    if (len(headers)!=1 or headers[0]['checkpoint_sha256']!=WEIGHTS[role]
            or headers[0]['action_horizon']!=32 or headers[0]['replan_steps']!=4
            or len(replans)!=len(r['queries'])
            or [v['frame_index'] for v in replans]!=list(range(0,r['executed_commands'],4))):
        raise ValueError('Telemetry is missing, duplicated, reordered, or foreign')
    rows=[]
    for event, query in zip(replans,r['queries'],strict=True):
        retrieval = event['model']['retrieval']
        if (event['episode_index']!=case or event['frame_index']!=query['frame']
                or event['model_input_sha256']!=query['model_input_sha256']
                or event['candidate_payload_sha256']!=query['candidate_payload_sha256']
                or retrieval['bank_rows']!=query['bank_rows']
                or event['model']['source']['derived_seed']!=query['derived_seed']):
            raise ValueError('Timing record is not bound to its actual query')
        pipeline = finite_nonnegative(event['pipeline_s'])
        retrieve = finite_nonnegative(retrieval['latency_s']['total_s'])
        if retrieve > pipeline + 1e-6:
            raise ValueError('Nested retrieval duration exceeds its enclosing pipeline')
        rows.append(dict(task=task,case_id=case,role=role,frame=query['frame'],
                         first_replan=query['frame']==0,pipeline_s=pipeline,retrieval_s=retrieve,
                         remainder_with_capture_s=max(0.,pipeline-retrieve),scope=binding['scope']))
    return r,rows


def audit_plan(plan):
    if (plan.get('schema')!='warm.native-logged-timing-audit.v1'
            or type(plan.get('GPUs')) is not int or plan['GPUs']!=0
            or plan.get('new_policy_or_physical_actions')!=0
            or plan.get('read_only_scientific_inputs') is not True
            or len(plan.get('roles',[]))!=plan.get('expected_complete_roles')):
        raise ValueError('A frozen zero-GPU complete-role snapshot is required')
    for ref in plan['inputs']:
        if sha(ref['path'])!=ref['sha256']:
            raise ValueError('Source or snapshot proof changed')
    records={}; rows=[]; summaries=[]
    for binding in plan['roles']:
        r,raw=audit_role(binding);key=tuple(binding['identity'])
        if key in records:
            raise ValueError('Duplicate complete role; do not select an outcome')
        records[key]=r;rows.extend(raw)
        summaries.append(dict(identity=list(key),queries=len(raw),
                              all_pipeline=summarize([v['pipeline_s'] for v in raw]),
                              first_pipeline_s=raw[0]['pipeline_s'],
                              steady_pipeline=summarize([v['pipeline_s'] for v in raw[1:]])))
    pairs=[]
    for task,case in sorted({(t,c) for t,c,_ in records}):
        a=records.get((task,case,'full'));b=records.get((task,case,'pooled_memory_prior'))
        if a is None or b is None:continue
        if a['initial_identity']!=b['initial_identity'] or a['common_contract']!=b['common_contract']:
            raise ValueError('Initial state or common information differs')
        keys=['model_input_sha256','proprio_sha256','base_gaussian','bank_rows','candidate_payload_sha256']
        if any(a['queries'][0][k]!=b['queries'][0][k] for k in keys):
            raise ValueError('Initial query input, pool, or noise differs')
        pairs.append([task,case])
    if len(pairs)!=plan['expected_complete_pairs']:
        raise ValueError('Pair accounting differs from the frozen evidence snapshot')
    groups={}
    for task in ['battery_try','rearrange_blocks']:
        for role in WEIGHTS:
            selected=[v for v in rows if v['task']==task and v['role']==role]
            if not selected:continue
            key=task+'/'+role;groups[key]=dict(roles=sum(s['identity'][0]==task and s['identity'][2]==role for s in summaries))
            for name,subset in [('all',selected),('first',[v for v in selected if v['first_replan']]),('steady',[v for v in selected if not v['first_replan']])]:
                groups[key][name]={field:summarize([v[field] for v in subset]) for field in ['pipeline_s','retrieval_s','remainder_with_capture_s']}
    report=dict(schema=plan['schema'],GPUs=0,complete_roles=len(records),complete_pairs=len(pairs),
                logged_replans=len(rows),groups=groups,per_role=summaries,paired_cases=pairs,
                timing_definition=plan['timing_definition'],raw_input_bindings=plan['roles'],
                descriptive_only=True,production_or_isolated_module_latency_claimed=False,
                hardware_matched_comparison_claimed=False,query_level_confidence_interval=None,
                SR_or_missing_skill_inference=False,new_policy_or_physical_actions=0,release=False)
    return report,rows
