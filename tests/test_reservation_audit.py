from copy import deepcopy
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import pytest
from fastwam.research.reservation_audit import reconcile


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.fixture
def plan(tmp_path):
    d=tmp_path;state=save(d/'state.json',dict(authoritative_active_execution=dict(reserved_upper_GPU_hours=12)))
    refs=[]
    for i,(first,end,post) in enumerate([(10,13,10),(13,15,12)]):
        refs.append(save(d/f'stage{i}/registration.json',dict(stage_first_admitted_utc=f'2026-11-12T{first:02d}:00:00+00:00',stage_absolute_deadline_utc=f'2026-11-12T{end:02d}:00:00+00:00',new_resource_GPU_hours_upper_bound=end-first,combined_reserved_GPU_hours_upper_bound=post)))
    live=save(d/'live.json',dict(utc='2026-11-12T16:00:00+00:00',inspected_old_stage_paths=[str(d/'stage0'),str(d/'stage1')],active_old_stage_paths=[]));closed=[]
    for i,hour in [(0,12),(1,14)]:
        closed.append(dict(stage=str(d/f'stage{i}'),registration=refs[i],operator=save(d/f'stage{i}/operator.json',dict(status='terminal_fixed_roles_invocation_complete',updated_utc=f'2026-11-12T{hour}:00:00+00:00')),terminal=save(d/f'stage{i}/exit.json',dict(status='terminal_fixed_roles_invocation_complete',ended_utc=f'2026-11-12T{hour}:00:00+00:00'))))
    return dict(schema='warm.closed-reservation-reconciliation.v1',GPUs=0,state=state,expected_upper_before=12,lineage=[dict(registration=r) for r in refs],process_attestation=live,closed=closed)


def run(p):return reconcile(p,now=datetime(2026,11,12,16,0,10,tzinfo=timezone.utc))


def test_only_unused_closed_tails_are_released(plan):
    r=run(plan);assert r['released_unused_tails']==2 and r['upper_after']==10
    assert not r['scientific_outcome_files_read'] and not r['original_windows_modified'] and not r['actual_billed_GPU_hours_estimated']


@pytest.mark.parametrize('changed',[False,True])
def test_closed_failure_retains_its_raw_evidence_before_any_credit(plan,changed):
    p=deepcopy(plan);r=p['closed'][0];r['operator']=save(Path(r['operator']['path']),dict(status='failed_or_interrupted_preserve_all',ended_utc='2026-11-12T12:00:00+00:00'))
    r['terminal']=save(Path(r['terminal']['path']),dict(exit_code=1,end_utc='2026-11-12T12:00:00+00:00'))
    raw=Path(r['operator']['path']).parent/'retained_unknown.json';r['retained_raw_archive']=save(raw,dict(success=None))
    if changed:
        raw.write_text('changed')
        with pytest.raises(ValueError,match='preserved'):run(p)
    else:assert run(p)['released_unused_tails']==2 and json.loads(raw.read_text())['success'] is None


@pytest.mark.parametrize('bad',['changed_input','active','stale','duplicate','already_credited','lineage_gap','not_charged','nonterminal','unknown_exit','future_end','missing_inspection','foreign_stage','nan_lineage','boolean_GPU'])
def test_credits_do_not_bypass_unknowns_or_double_count(plan,bad):
    p=deepcopy(plan);r=p['closed'][0]
    if bad=='changed_input':Path(r['operator']['path']).write_text('{}')
    elif bad in ('active','stale','missing_inspection'):
        path=Path(p['process_attestation']['path']);v=json.loads(path.read_text())
        if bad=='active':v['active_old_stage_paths']=['stage0']
        elif bad=='stale':v['utc']='2026-11-12T15:00:00+00:00'
        else:v['inspected_old_stage_paths']=[]
        p['process_attestation']=save(path,v)
    elif bad=='duplicate':p['closed'].append(r)
    elif bad=='already_credited':
        path=Path(p['state']['path']);v=json.loads(path.read_text());v['credited_closed_reservation_ids']=[r['registration']['path']];p['state']=save(path,v)
    elif bad=='lineage_gap':p['lineage'].reverse()
    elif bad=='not_charged':r['registration']=dict(path='unrelated',sha256='x')
    elif bad=='nonterminal':r['operator']=save(Path(r['operator']['path']),dict(status='running_0010-full'))
    elif bad=='unknown_exit':r['terminal']=save(Path(r['terminal']['path']),dict(exit_code=None,ended_utc='2026-11-12T12:00:00+00:00'))
    elif bad=='foreign_stage':r['stage']='unrelated'
    elif bad=='boolean_GPU':p['GPUs']=False
    elif bad=='nan_lineage':
        path=Path(p['lineage'][0]['registration']['path']);v=json.loads(path.read_text());v['combined_reserved_GPU_hours_upper_bound']=float('nan');p['lineage'][0]['registration']=save(path,v)
    else:
        path=Path(p['lineage'][0]['registration']['path']);v=json.loads(path.read_text());v['stage_absolute_deadline_utc']='2026-11-13T13:00:00+00:00';v['new_resource_GPU_hours_upper_bound']=27;p['lineage'][0]['registration']=save(path,v)
    with pytest.raises((ValueError,FileNotFoundError)):run(p)
