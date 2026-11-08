"""Admit a fixed recovery source without selecting successful policy outcomes."""
from __future__ import annotations


def admit_formal_recovery(plan, source_identity, trials, *, task, seed,
                          episode_index, plan_sha256, core_source_sha256,
                          search_condition):
    if type(episode_index) is not int or not 0 <= episode_index < plan['local_recovery']['source_episodes_per_task']:
        raise ValueError('source episode outside the fixed recovery cohort')
    if task not in plan['tasks'] or not plan['closed_loop']['reset_seed_start'] <= seed < (
        plan['closed_loop']['reset_seed_start']+plan['closed_loop']['candidate_seed_limit']):
        raise ValueError('source task/seed outside the fixed cohort')
    expected=dict(task=task,phase='formal',checkpoint_sha256=plan['checkpoint_sha256'],
        plan_sha256=plan_sha256,core_source_sha256=core_source_sha256,search_condition=search_condition)
    if any(source_identity.get(key)!=value for key,value in expected.items()):
        raise ValueError('source model, protocol, task, condition or immutable core differs')
    matching=[row for row in trials if row['episode_id']==episode_index and row['seed']==seed]
    if len(matching)!=1:raise ValueError('exactly one source episode record required')
    row=matching[0]
    for key in ('task','phase','checkpoint_sha256','plan_sha256','search_condition'):
        if row.get(key)!=expected[key]:raise ValueError('source trial binding differs')
    full=row['roles'].get('full')
    if full is None or full.get('status')!='complete' or type(full.get('success')) is not int or full['success'] not in (0,1):
        raise ValueError('complete official-length Full source required, regardless of success')
    queries=list(plan['local_recovery']['query_replan_ordinals'])
    pool_queries=(list(plan['candidate_specific_effects']['query_replan_ordinals'])
        if episode_index<plan['candidate_specific_effects']['source_episodes_per_task'] else [])
    if len(queries)!=len(set(queries)) or any(type(q) is not int or q<0 for q in queries) or not set(pool_queries)<=set(queries):
        raise ValueError('invalid fixed recovery/pool query schedule')
    return dict(query_ordinals=queries,candidate_pool_query_ordinals=pool_queries,
        source_episode=episode_index,source_seed=seed,source_success=full['success'],
        selection='Fixed source episode index, independent of the policy success value.',
        unreached='Retain every planned query that is not reached; never replace based on outcomes.')
