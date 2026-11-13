from pathlib import Path
import copy,importlib.util,pytest

s=importlib.util.spec_from_file_location('pb_sealed_guard',Path(__file__).resolve().parents[1]/'src/fastwam/research/pb_null_sealed_physics.py')
g=importlib.util.module_from_spec(s);s.loader.exec_module(g)

def valid():
    return dict(status='failed_or_interrupted_q8_null_preserve_all',case_id=0,physical_seed=340800000,query=8,frame=32,
                live_reproduction_queries=9,extra_policy_queries=3,reproduced_native_commands=36,new_branch_commands=0,
                model_updates=0,branches={},error="OutOfMemoryError('known allocation')",
                traceback='FrozenFactualEffectReader.from_source_runtime')

def test_known_postseal_failure_admitted():
    assert g.validate_known_failure(valid(),dict(status='failed',exit_code=1),0)

@pytest.mark.parametrize('key,value',[
    ('new_branch_commands',1),('branches',{'Full-pre':{}}),('extra_policy_queries',2),
    ('live_reproduction_queries',8),('reproduced_native_commands',35),('model_updates',1),
    ('case_id',1),('physical_seed',340800001),('query',48),('frame',192),
    ('error',"ImportError('other')"),('traceback','other allocation'),('status','running')])
def test_no_unknown_or_partial_action_retry(key,value):
    op=valid();op[key]=value
    with pytest.raises(ValueError):g.validate_known_failure(op,dict(status='failed',exit_code=1),0)

@pytest.mark.parametrize('runner',[dict(status='running',exit_code=None),dict(status='complete',exit_code=0),dict(status='failed',exit_code=-15)])
def test_terminal_exit_is_required(runner):
    with pytest.raises(ValueError):g.validate_known_failure(valid(),runner,0)

def test_no_top_level_fastwam_package_import():
    import subprocess,sys
    script="import runpy,sys;runpy.run_path(sys.argv[1]);assert 'fastwam' not in sys.modules"
    subprocess.run([sys.executable,'-c',script,str(Path(g.__file__))],check=True)


def test_output_parent_is_checked_before_inference(tmp_path):
    out=tmp_path/'case00'/'results'
    with pytest.raises(ValueError):g.check_output_readiness(tmp_path,0,out)
    out.parent.mkdir();g.check_output_readiness(tmp_path,0,out)
    out.mkdir()
    with pytest.raises(ValueError):g.check_output_readiness(tmp_path,0,out)

def test_materialized_seal_bytes_and_no_operator_copy(tmp_path):
    import hashlib
    source=tmp_path/'source';source.mkdir();out=tmp_path/'out';out.mkdir();files={}
    for i,name in enumerate(g.SEALED_INPUT_NAMES):
        p=source/name;p.write_bytes(('sealed-'+str(i)).encode());files[name]=dict(path=str(p),sha256=g.sha(p))
    files['operator.json']=dict(path='must-not-read',sha256='bad')
    result=g.materialize_sealed_inputs(dict(files=files),out)
    assert set(result)==set(g.SEALED_INPUT_NAMES) and not (out/'operator.json').exists()
    for name in g.SEALED_INPUT_NAMES:assert (out/name).read_bytes()==(source/name).read_bytes()
    with pytest.raises(ValueError):g.materialize_sealed_inputs(dict(files=files),out)

def test_changed_sealed_input_rejected_before_copy(tmp_path):
    p=tmp_path/'input';p.write_bytes(b'changed');out=tmp_path/'out';out.mkdir()
    files={name:dict(path=str(p),sha256='0'*64) for name in g.SEALED_INPUT_NAMES}
    with pytest.raises(ValueError):g.materialize_sealed_inputs(dict(files=files),out)
    assert not list(out.iterdir())


def test_unknown_full_pre_cannot_be_requested():
    assert g.physical_roles(dict(physical_roles=['candidate-null','Full-post']))==['candidate-null','Full-post']
    for roles in [['Full-pre'],['Full-post','candidate-null'],['Full-pre','Full-post']]:
        with pytest.raises(ValueError):g.physical_roles(dict(physical_roles=roles))

def test_single_post_full_does_not_claim_two_sided_repeat():
    row=dict(endpoint_status='complete_horizon')
    full,two_sided=g.full_control_summary({'candidate-null':{},'Full-post':row})
    assert full is row and two_sided is False

def test_missing_unstarted_role_is_not_complete():
    with pytest.raises(ValueError):g.full_control_summary({'Full-post':{}})

def test_only_known_writer_failure_qualifies():
    op=dict(status='failed_or_interrupted_q8_null_preserve_all',case_id=0,mode='sealed_action_physics_only_continuation',new_policy_loaded=False,live_reproduction_queries=0,extra_policy_queries=0,model_updates=0,active_branch='Full-pre',branches={},new_branch_commands=12,error="FileNotFoundError(2,'missing')",traceback='action_seal.json')
    g.validate_unstarted_failure(op,dict(status='failed',exit_code=1),0)
    for key,value in [('active_branch','candidate-null'),('new_branch_commands',0),('extra_policy_queries',1),('branches',{'candidate-null':{}}),('traceback','other missing file')]:
        changed=dict(op);changed[key]=value
        with pytest.raises(ValueError):g.validate_unstarted_failure(changed,dict(status='failed',exit_code=1),0)


@pytest.mark.parametrize('loaded',[True,False])
def test_reader_handoff_keeps_room_above_observed_live_allocations(loaded):
    # Actual failing H100 run:24.19GiB live,79.18GiB total;the old .12 cap
    # allowed9.50GiB and failed despite54.98GiB physically free.
    live=24.19 if loaded else 0
    allowed=g.reader_memory_fraction(loaded)*79.18
    assert allowed>live+2.0 and allowed<=.48*79.18

@pytest.mark.parametrize('state',[None,0,1])
def test_reader_handoff_rejects_ambiguous_lifetime(state):
    with pytest.raises(ValueError):g.reader_memory_fraction(state)
