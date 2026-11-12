"""Predeclared, read-only PB counter outcomes for original-model interventions."""
from __future__ import annotations
RUBRIC='warm.rmbench.PB-counted-interaction.v1'
TRACE_PLAN_SHA256='56b3c310d52c0e454e2a588f76759b1af302c9c20d68da0443497b5fbe73c02a'
CORE_SHA256='4a1d05cb548f16bf6a9b239082b02d4697aa8e8c1f43c2f5d8ffbaec196462fd'
WEIGHT_SHA256='a765d987b2df0d228782980592c39aaed0e837040c7f6f48c905e4641f79c285'

def validate_plan(p):
 expected=dict(schema='warm.original-PB-first-state-fresh-pool.v1',task_name='press_button',root_seed=3407,physical_seed=0,checkpoint_step=12000,horizon=32,execution_prefix=4,checkpoint_sha256=WEIGHT_SHA256,original_source_sha256=CORE_SHA256,factual_source_plan_sha256=TRACE_PLAN_SHA256,next_interaction_rubric=RUBRIC,effect_descriptor='original_uniform_mean_DINO_delta_v1')
 if p.get('schema')=='warm.original-PB-namespace2-fresh-pool.v1':
  _,entry=namespace_module().validate_execution_binding(p,'pool');expected.update(schema=p['schema'],physical_seed=entry['seed'],factual_source_plan_sha256=p['namespace2_factual_plan']['sha256'])
 if any(type(p.get(k)) is not type(v) or p.get(k)!=v for k,v in expected.items()):raise ValueError('Pinned original PB source,model,seed,descriptor or rubric differs')
 if type(p.get('query_replan')) is not int or p['query_replan'] not in (8,48,96):raise ValueError('Only the three predeclared source queries are supported')
 if type(p.get('selected_rank')) is not int or not 0<=p['selected_rank']<32:raise ValueError('The source model must seal an original valid rank before intervention')
 return p

def namespace_module():
 import importlib.util
 from pathlib import Path
 spec=importlib.util.spec_from_file_location('PB_namespace_binding',Path(__file__).with_name('pb_namespace.py'))
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def validate_source(op,trace,query,plan=None):
 required=dict(status='complete_fixed_original_factual_episode',candidate_seed=0,source_sha256=CORE_SHA256,source_sha256_after=CORE_SHA256,checkpoint_sha256=WEIGHT_SHA256,initial_observation_exact_to_qualification=True,first_replan_exact_to_qualification=True,hidden_PB_counter_state_audited_only_not_policy_input=True,model_updates=0)
 if plan is not None and plan.get('schema')=='warm.original-PB-namespace2-fresh-pool.v1':
  validate_plan(plan);required.update(candidate_seed=plan['physical_seed'],plan_sha256=plan['factual_source_plan_sha256'],namespace_case_id=plan['namespace_case_id'],namespace_roster_sha256=plan['namespace2_roster']['sha256'],namespace_registration_sha256=plan['namespace2_registration']['sha256'],original_namespace_initial_identity_exact=True)
 if any(op.get(k)!=v for k,v in required.items()):raise ValueError('Original PB source is not a qualified complete factual episode')
 n=op['native_commands']
 if len(trace)!=n or not 0<n<=1500 or (n!=1500 and not op.get('success_flag')):raise ValueError('An incomplete source is not a new negative trial')
 if n<query*4+4:raise ValueError('Prespecified query prefix was not reached;no substitution')
 if any(r['index']!=i or r['take_action_cnt']!=i+1 or 'task_state_before' not in r or 'task_state_after' not in r for i,r in enumerate(trace)):raise ValueError('Native counters or read-only PB state records are incomplete')
 return n

def observe(env):
 names=('card_id_1','card_id_2','press_cnt_1','press_cnt_2','press_cnt_check_button','press_flag_1','press_flag_2','press_flag_check_button')
 native={n:getattr(env,n) for n in names};native={n:v.item() if hasattr(v,'item') else v for n,v in native.items()}
 targets=[native['card_id_1'],native['card_id_2']];counts=[native['press_cnt_1'],native['press_cnt_2']]
 if any(type(v) is not int or v<0 for v in targets+counts+[native['press_cnt_check_button']]):raise ValueError('Native PB integer counters required')
 return dict(rubric=RUBRIC,task='press_button',targets=targets,counts=counts,native_state=native,terminal_success=bool(env.eval_success))

def label(parent,endpoint,*,valid_branch):
 if parent['rubric']!=RUBRIC or endpoint['rubric']!=RUBRIC or parent['targets']!=endpoint['targets']:raise ValueError('Different task/count targets or rubric')
 r=dict(applicable=None,label_provenance=RUBRIC,scope='A completed counted interaction or official success;not generic motion quality or population SR',terminal_success=endpoint['terminal_success'])
 if not valid_branch:return {**r,'reason':'invalid physical branch;outcome remains unknown'}
 if parent['terminal_success'] or any(c>t for c,t in zip(parent['counts'],parent['targets'])):return {**r,'reason':'pre-existing terminal or irreversible over-counted parent;not blamed on proposal'}
 if any(b<a for a,b in zip(parent['counts'],endpoint['counts'])):return {**r,'reason':'unexpected native counter reset;retain unknown'}
 if any(c>t for c,t in zip(endpoint['counts'],endpoint['targets'])):return {**r,'applicable':0,'reason':'exceeded an official target count'}
 if endpoint['terminal_success']:
  if endpoint['counts']!=endpoint['targets']:return {**r,'reason':'official success and stored counters disagree'}
  return {**r,'applicable':1,'reason':'official native executor success'}
 if any(b>a for a,b in zip(parent['counts'],endpoint['counts'])):return {**r,'applicable':1,'reason':'completed an additional required press without over-counting'}
 return {**r,'applicable':0,'reason':'no newly completed required press or official success'}

def require_repeats(records,selected):
 keys=['anchor-a','anchor-aa',f'rank-{selected}','anchor-qual-final'];rows=[records[k] for k in keys];a=rows[0]
 if any(r['parent_hashes']!=a['parent_hashes'] for r in rows):raise ValueError('Fresh parents differ')
 for r in (rows[1],rows[3]):
  for name in ['endpoint_status','executed_steps','endpoint_hashes','observed_effect','independent_label','error','parent_progress','endpoint_progress','secondary_prefix4']:
   if r[name]!=a[name]:raise ValueError('Repeated Full terminal differs: '+name)
 return dict(identical_parent_and_repeated_Full_terminals=True,raw_distinct_physical_endpoint=rows[2]['endpoint_hashes']['physics']!=a['endpoint_hashes']['physics'],raw_endpoint_status=rows[2]['endpoint_status'],full_endpoint_status=a['endpoint_status'],full_H32_effect_available=a['endpoint_status']=='complete_horizon',no_imputation_for_censored_or_invalid_effects=True)

def repair(records,selected):
 a=records[f'rank-{selected}']['independent_label']['applicable'];b=records['anchor-a']['independent_label']['applicable'];known=a is not None and b is not None
 return dict(scope='One fixed source state;dependent interventions,not a population effect',selected_rank=selected,raw_label=a,refined_label=b,paired_known=known,repaired=(a==0 and b==1) if known else None,harmed=(a==1 and b==0) if known else None,rubric=RUBRIC,population_success_rate_estimated=False)


def immutable_json(path, value):
 """Verify an existing evidence file instead of replacing it on continuation."""
 import json
 from pathlib import Path
 path=Path(path)
 if path.exists():
  if json.loads(path.read_text())!=value:raise ValueError('Existing sealed evidence differs: '+str(path))
  return
 with path.open('x') as f:f.write(json.dumps(value,sort_keys=True,indent=2,allow_nan=False))
