"""Parser fixtures are engineering checks, not robot measurements."""
import json
import pytest
from fastwam.research.native_timing_audit import CORE,WEIGHTS,audit_role,finite_nonnegative,quantile,sha,summarize


@pytest.mark.parametrize('bad',[True,None,-1,float('nan'),float('inf'),'1'])
def test_invalid_durations_are_not_silently_filtered(bad):
    with pytest.raises(ValueError):finite_nonnegative(bad)


def test_quantile_includes_both_endpoints():
    assert quantile([100,0,10],.95)==pytest.approx(91)
    assert quantile([5],.5)==5
    assert summarize([]) is None


@pytest.fixture
def raw(tmp_path):
    q=dict(frame=0,model_input_sha256='m',candidate_payload_sha256='c',bank_rows=[1],derived_seed=7)
    result=dict(role='full',task='battery_try',episode_id=1,status='complete_native',checkpoint_sha256=WEIGHTS['full'],evaluation_source_sha256=CORE,success=1,executed_commands=4,queries=[q])
    event=dict(kind='replan',episode_index=1,frame_index=0,model_input_sha256='m',candidate_payload_sha256='c',pipeline_s=1.,model=dict(retrieval=dict(bank_rows=[1],latency_s=dict(total_s=.25)),source=dict(derived_seed=7)))
    rows=[dict(kind='header',checkpoint_sha256=WEIGHTS['full'],action_horizon=32,replan_steps=4),event]
    for name,obj in [('result',result),('exit',dict(exit_code=0))]: (tmp_path/(name+'.json')).write_text(json.dumps(obj))
    (tmp_path/'telemetry.jsonl').write_text('\n'.join(json.dumps(x) for x in rows))
    bind={k:dict(path=str(tmp_path/f),sha256=sha(tmp_path/f)) for k,f in [('result','result.json'),('exit','exit.json'),('telemetry','telemetry.jsonl')]}
    bind.update(identity=['battery_try',1,'full'],scope='test_fixture');return bind,tmp_path,event,rows


def test_logged_record_is_bound_to_actual_query(raw):
    bind,_,_,_=raw;_,rows=audit_role(bind)
    assert rows[0]['pipeline_s']==1 and rows[0]['remainder_with_capture_s']==.75


@pytest.mark.parametrize('bad',['duplicate','foreign_query','negative_nested','changed_hash','unknown_endpoint'])
def test_evidence_corruption_is_rejected(raw,bad):
    bind,root,event,rows=raw
    if bad=='duplicate':rows.append(event)
    elif bad=='foreign_query':event['model_input_sha256']='other'
    elif bad=='negative_nested':event['model']['retrieval']['latency_s']['total_s']=2.
    elif bad=='unknown_endpoint':
        p=root/'result.json';r=json.loads(p.read_text());r.update(status='loading_typed_policy',success=None);p.write_text(json.dumps(r));bind['result']['sha256']=sha(p)
    p=root/'telemetry.jsonl';p.write_text('\n'.join(json.dumps(x) for x in rows))
    if bad!='changed_hash':bind['telemetry']['sha256']=sha(p)
    else:p.write_text(p.read_text()+'\n{}')
    with pytest.raises(ValueError):audit_role(bind)
