from pathlib import Path
import copy
import importlib.util
import json

import pytest
from fastwam.research.fresh_pool import read_verified_branch, seal_branch

p=Path(__file__).resolve().parents[1]/'src/fastwam/research/pb_continuation.py'
spec=importlib.util.spec_from_file_location('PB_branch_adoption',p)
M=importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
ORDER=['anchor-a','anchor-aa','rank-0','anchor-qual-final','rank-1','anchor-pool-final']


def inventory():
    return dict(schema='warm.original-PB-remaining-branches.v1',
        new_work='remaining_original_branches_only',prior_clock_reset=False,
        completed={n:{} for n in ORDER[:4]},
        unknown={'rank-1':dict(directory='old/attempt-001',
            complete_command_records=7,reason='known terminal interruption')},
        unstarted=['anchor-pool-final'])


def test_unknown_is_accounted_without_becoming_a_new_request():
    m=M.require_inventory(inventory(),ORDER)
    assert m['unstarted']==['anchor-pool-final']
    r=M.unknown_record(m,'rank-1','fixed-query')
    assert r['observed_effect'] is None and r['independent_label']['applicable'] is None
    assert r['endpoint_hashes'] is None and r['parent_hashes'] is None
    assert not r['parent_recreated'] and r['executed_steps']==7


@pytest.mark.parametrize('change', ['overlap','missing','substitution','unqualified','clock'])
def test_no_replacement_duplicate_unknown_retry_or_clock_reset(change):
    m=inventory()
    if change=='overlap':m['unstarted'].insert(0,'rank-1')
    if change=='missing':m['unknown']={}
    if change=='substitution':m['unstarted']=['another-rank']
    if change=='unqualified':m['unknown']['rank-0']=m['completed'].pop('rank-0')
    if change=='clock':m['prior_clock_reset']=True
    with pytest.raises(ValueError):M.require_inventory(m,ORDER)


def test_remaining_order_is_fixed_before_outcomes():
    m=inventory();m['unknown']={};m['unstarted']=['anchor-pool-final','rank-1']
    with pytest.raises(ValueError,match='order'):M.require_inventory(m,ORDER)


def test_resource_plan_cannot_change_candidates_source_or_horizon(tmp_path):
    base=dict(output=str(tmp_path/'old'),horizon=32,selected_rank=0,
        inputs=[dict(path='frozen',sha256='unchanged')],source='original')
    new=dict(base,output=str(tmp_path/'new'),collector_commit='executor-only')
    assert M.require_resource_only_plan(base,new)=={'output','collector_commit'}
    for key,value in [('horizon',4),('selected_rank',1),('source','other'),('inputs',[])]:
        bad=dict(new,**{key:value})
        with pytest.raises(ValueError):M.require_resource_only_plan(base,bad)
    with pytest.raises(ValueError):
        M.require_resource_only_plan(base,dict(base,output=str(tmp_path/'old/nested')))


def prepared(tmp_path):
    base=tmp_path/'original';base.mkdir();out=tmp_path/'continuation'
    plan=tmp_path/'base-plan.json';plan.write_text(json.dumps(dict(output=str(base),horizon=32)))
    m=inventory();m.update(base_plan=str(plan),base_plan_sha256=M.sha(plan),
        sealed_query_sha256='original-sealed-pool',old_stage_operator=str(tmp_path/'terminal.json'))
    terminal=Path(m['old_stage_operator']);terminal.write_text(json.dumps(dict(status='terminal_fixed_PB_stage_all_existing_output_archived')))
    m['bound_files']=[dict(path=str(terminal),sha256=M.sha(terminal))]
    for name in ORDER[:4]:
        d=base/'branches'/name/'attempt-001';d.mkdir(parents=True)
        row=dict(query_id='q',branch_id=name,parent_recreated=True,
            endpoint_status='complete_horizon',executed_steps=32,observed_effect=[1.0],
            independent_label=dict(applicable=0),sealed_query_sha256=m['sealed_query_sha256'])
        seal_branch(d,row)
        m['completed'][name]=dict(directory=str(d),complete_sha256=M.sha(d/'complete.json'))
    return m,dict(output=str(out),horizon=32),out


def test_adoption_preserves_all_original_bytes_and_never_creates_unknown_attempt(tmp_path):
    m,new,out=prepared(tmp_path)
    path=tmp_path/'manifest.json';path.write_text(json.dumps(m))
    M.bind_manifest(path,M.sha(path),new,ORDER,m['sealed_query_sha256'])
    M.adopt_completed(m,out,query_id='q',verifier=read_verified_branch)
    M.adopt_completed(m,out,query_id='q',verifier=read_verified_branch)
    assert not (out/'branches/rank-1').exists()
    for name,item in m['completed'].items():
        assert M.sha(out/'branches'/name/'attempt-001/complete.json')==item['complete_sha256']
        assert Path(item['directory']).is_dir()


def test_active_predecessor_and_changed_terminal_evidence_are_rejected(tmp_path):
    m,new,_=prepared(tmp_path);p=tmp_path/'manifest.json';p.write_text(json.dumps(m))
    Path(m['old_stage_operator']).write_text(json.dumps(dict(status='running')))
    with pytest.raises(ValueError,match='changed'):
        M.bind_manifest(p,M.sha(p),new,ORDER,m['sealed_query_sha256'])
    m['bound_files'][0]['sha256']=M.sha(m['old_stage_operator']);p.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='not terminal'):
        M.bind_manifest(p,M.sha(p),new,ORDER,m['sealed_query_sha256'])


def test_changed_adopted_result_cannot_be_silently_overwritten(tmp_path):
    m,_,out=prepared(tmp_path);M.adopt_completed(m,out,query_id='q',verifier=read_verified_branch)
    changed=out/'branches/anchor-a/attempt-001/result.json';changed.write_text('{}')
    with pytest.raises(ValueError):M.adopt_completed(m,out,query_id='q',verifier=read_verified_branch)


def test_foreign_branch_directory_is_rejected(tmp_path):
    m,_,out=prepared(tmp_path);m['completed']['anchor-a']['directory']=str(tmp_path)
    with pytest.raises(ValueError,match='Foreign'):M.adopt_completed(m,out,query_id='q',verifier=read_verified_branch)
