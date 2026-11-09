from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pytest
from fastwam.memory.manifest import sha256_array
from fastwam.research.resumable_pairs import read_roles,unique_roles,paired_rows,next_role,summarize


def dump(p,value):p.write_text(json.dumps(value))


def fixture(tmp_path,role='full',success=0,count=700,index=0):
    root=tmp_path/role;root.mkdir()
    plan=dict(task='rearrange_blocks',checkpoint_sha256='c',evidence_plan_sha256='p',core_source_sha256='s',runtime_source_sha256='r',official_step_limit=700)
    identity=dict(task='rearrange_blocks',phase='formal',checkpoint_sha256='c',plan_sha256='p',search_condition='full_bank',core_source_sha256='s',collector_source_sha256='s')
    dump(root/'identity.json',identity);dump(root/'runtime_source_manifest.json',dict(sha256='r'))
    result=dict(status='complete',success=success,executed_commands=count,task='rearrange_blocks',phase='formal',episode_id=index,seed=100+index,role=role)
    row=dict(task='rearrange_blocks',phase='formal',checkpoint_sha256='c',plan_sha256='p',search_condition='full_bank',episode_id=index,seed=100+index,roles={role:result})
    (root/'role_trials.jsonl').write_text(json.dumps(row)+'\n')
    case=root/f'case-{index:04d}-seed-{100+index}'/role;case.mkdir(parents=True)
    dump(case/'result.json',result)
    obs=dict(native_qpos=np.zeros(14,dtype=np.float32),head=np.zeros((4,4,3),dtype=np.uint8))
    np.savez_compressed(case/'initial_observation.npz',**obs)
    dump(case/'initial_identity.json',dict(observation_arrays={k:sha256_array(v) for k,v in obs.items()},instruction='fixed',task_state={'stage':0}))
    np.savez_compressed(case/'qpos_trace.npz',**{k:np.zeros((count,14),dtype=np.float32) for k in ('commands','before','after')})
    return root,case,plan,{index:100+index},row


def test_complete_zero_is_preserved_but_missing_role_is_not_failure(tmp_path):
    root,case,p,seeds,row=fixture(tmp_path)
    records=read_roles(root,p,seeds);pairs=paired_rows(records,seeds,[0]);s=summarize(pairs)
    assert s['complete_pairs']==0 and s['pending_pairs']==1
    assert pairs[0]['roles']['candidate_null'] is None
    assert next_role(unique_roles(records),seeds,[0])==(0,'candidate_null',100)


def test_verified_pair_and_counterbalanced_next_role(tmp_path):
    a,_,p,seeds,_=fixture(tmp_path,'full');b,_,_,_,_=fixture(tmp_path,'candidate_null',success=1,count=4)
    records=read_roles(a,p,seeds)+read_roles(b,p,seeds);rows=paired_rows(records,seeds,[0])
    assert rows[0]['full_minus_null']==-1 and summarize(rows)['null_only']==1
    assert next_role(unique_roles(records),{0:100,1:101},[0,1])==(1,'candidate_null',101)


@pytest.mark.parametrize('kind',['truncated','nonfinite','bad_image','wrong_initial_qpos','unfinished','wrong_source','wrong_seed','wrong_simulator'])
def test_corrupted_or_differently_bound_evidence_is_rejected(tmp_path,kind):
    root,case,p,seeds,row=fixture(tmp_path)
    if kind in ('truncated','nonfinite'):
        arrays={k:np.zeros((700,14),dtype=np.float32) for k in ('commands','before','after')}
        if kind=='truncated':arrays['after']=arrays['after'][:-1]
        else:arrays['commands'][3,0]=np.nan
        np.savez_compressed(case/'qpos_trace.npz',**arrays)
    elif kind in ('bad_image','wrong_initial_qpos'):
        np.savez_compressed(case/'initial_observation.npz',native_qpos=np.ones(14) if kind=='wrong_initial_qpos' else np.zeros(14,dtype=np.float32),head=np.ones((4,4,3),dtype=np.uint8))
    elif kind=='unfinished':row['roles']['full']['executed_commands']=699;dump(case/'result.json',row['roles']['full']);(root/'role_trials.jsonl').write_text(json.dumps(row)+'\n')
    elif kind=='wrong_source':d=json.loads((root/'identity.json').read_text());d['collector_source_sha256']='other';dump(root/'identity.json',d)
    elif kind=='wrong_seed':seeds[0]=101
    else:dump(root/'runtime_source_manifest.json',dict(sha256='other'))
    with pytest.raises(ValueError):read_roles(root,p,seeds)


def test_duplicate_attempts_and_mismatched_reset_are_not_silently_chosen(tmp_path):
    a,_,p,seeds,_=fixture(tmp_path,'full');b,case,_,_,_=fixture(tmp_path,'candidate_null')
    x=read_roles(a,p,seeds);y=read_roles(b,p,seeds)
    with pytest.raises(ValueError,match='duplicate'):unique_roles(x+deepcopy(x))
    y[0]['initial_identity']['task_state']={'stage':1}
    with pytest.raises(ValueError,match='identities'):paired_rows(x+y,seeds,[0])


def test_interrupted_attempt_without_committed_trial_is_unknown(tmp_path):
    root,_,p,seeds,_=fixture(tmp_path)
    (root/'role_trials.jsonl').unlink()
    assert read_roles(root,p,seeds)==[]
    assert next_role({},seeds,[0])==(0,'full',100)


def test_interrupted_journal_tail_never_becomes_an_outcome(tmp_path):
    root,_,p,seeds,_=fixture(tmp_path)
    with (root/'role_trials.jsonl').open('a') as stream:stream.write('{"episode_id":1,"success":')
    assert len(read_roles(root,p,seeds))==1
