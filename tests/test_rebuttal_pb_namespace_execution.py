from pathlib import Path
import datetime,importlib.util
import pytest

ROOT=Path(__file__).resolve().parents[1]
def load(name,leaf):
    spec=importlib.util.spec_from_file_location(name,ROOT/leaf);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
Q=load('PB_ns_qualification_test','scripts/qualify_rebuttal_original_PB.py')
T=load('PB_ns_trace_test','scripts/collect_rebuttal_original_specialist_trace.py')
M=load('PB_ns_mechanism_test','src/fastwam/research/pb_mechanism.py')
S=load('PB_ns_stage_test','scripts/run_rebuttal_pb_namespace_source.py')


def test_case1_uses_its_admitted_seed_without_relaxing_original_model(monkeypatch):
    for module,kind in [(Q,'qualification'),(T,'factual')]:
        monkeypatch.setattr(module.N,'validate_execution_binding',lambda p,k:({},dict(case_id=1,seed=340800007)))
    p=dict(Q.PB_PROTOCOL,schema=Q.N.SCHEMAS['qualification'],physical_seed=340800007,original_source_sha256=Q.CORE_SHA,original_source_commit=Q.COMMIT)
    assert Q.validate_qualification_plan(p)['physical_seed']==340800007
    p['checkpoint_step']=3000
    with pytest.raises(ValueError):Q.validate_qualification_plan(p)
    trace=dict(schema=T.N.SCHEMAS['factual'],physical_seed=340800007,max_native_commands=1500,capture_replans=[8,48,96],qualified_original_run='press_button-s3407-v2',checkpoint_step=12000,task_name='press_button',root_seed=3407,original_source_sha256=Q.CORE_SHA,original_source_commit=Q.COMMIT,checkpoint_sha256=Q.PB_PROTOCOL['checkpoint_sha256'])
    assert T.validate_trace_plan(trace)=='press_button'
    trace['physical_seed']=340800000
    with pytest.raises(ValueError):T.validate_trace_plan(trace)


def test_new_source_cannot_reuse_physical0_or_foreign_case_operator(monkeypatch):
    class Binding:
        @staticmethod
        def validate_execution_binding(p,k):return {},dict(case_id=1,seed=340800007)
    monkeypatch.setattr(M,'namespace_module',lambda:Binding)
    plan=dict(schema='warm.original-PB-namespace2-fresh-pool.v1',physical_seed=340800007,namespace_case_id=1,namespace2_factual_plan=dict(sha256='newtrace'),factual_source_plan_sha256='newtrace',namespace2_registration=dict(sha256='registration'),namespace2_roster=dict(sha256='roster'),task_name='press_button',root_seed=3407,checkpoint_step=12000,horizon=32,execution_prefix=4,checkpoint_sha256=M.WEIGHT_SHA256,original_source_sha256=M.CORE_SHA256,next_interaction_rubric=M.RUBRIC,effect_descriptor='original_uniform_mean_DINO_delta_v1',query_replan=8,selected_rank=0)
    op=dict(status='complete_fixed_original_factual_episode',candidate_seed=340800007,source_sha256=M.CORE_SHA256,source_sha256_after=M.CORE_SHA256,checkpoint_sha256=M.WEIGHT_SHA256,initial_observation_exact_to_qualification=True,first_replan_exact_to_qualification=True,hidden_PB_counter_state_audited_only_not_policy_input=True,model_updates=0,native_commands=1500,success_flag=False,plan_sha256='newtrace',namespace_case_id=1,namespace_roster_sha256='roster',namespace_registration_sha256='registration',original_namespace_initial_identity_exact=True)
    trace=[dict(index=i,take_action_cnt=i+1,task_state_before={},task_state_after={}) for i in range(1500)]
    assert M.validate_source(op,trace,8,plan)==1500
    for key,value in [('candidate_seed',0),('namespace_case_id',0),('plan_sha256',M.TRACE_PLAN_SHA256),('namespace_roster_sha256','other'),('original_namespace_initial_identity_exact',False)]:
        changed=dict(op);changed[key]=value
        with pytest.raises(ValueError):M.validate_source(changed,trace,8,plan)


def test_namespace_deadline_is_clipped_to_writing_cutoff_and_never_reset():
    utc=datetime.timezone.utc;first=datetime.datetime(2026,11,13,0,tzinfo=utc);latest=datetime.datetime(2026,11,13,11,tzinfo=utc)
    assert S.fixed_deadline(first,latest)==first+datetime.timedelta(hours=10)
    assert S.fixed_deadline(first+datetime.timedelta(hours=5),latest)==latest
    with pytest.raises(ValueError):S.fixed_deadline(latest-datetime.timedelta(minutes=10),latest)
    with pytest.raises(ValueError):S.fixed_deadline(first.replace(tzinfo=None),latest)
