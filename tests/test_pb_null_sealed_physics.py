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
