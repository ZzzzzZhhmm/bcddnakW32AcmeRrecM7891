from pathlib import Path
import ast,copy,importlib.util,json,os
import pytest

ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj
M=load('reset_test_helpers',ROOT/'src/fastwam/research/matched_native.py')
Q=load('reset_qualifier',ROOT/'scripts/qualify_rebuttal_matched_resets.py')

def isolation(tmp_path,name):
    root=tmp_path/name;runtime=root/'runtime';runtime.mkdir(parents=True);writable=root/'private';writable.mkdir()
    env={key:str(writable/key) for key in M.NATIVE_PRIVATE_ENVIRONMENTS}
    for value in env.values():Path(value).mkdir()
    item=dict(worker_id=name,runtime_root=str(runtime),writable_root=str(writable),environment=env,ipc_kind='anonymous_multiprocessing_Pipe',read_only_asset_root='/fixed/assets')
    (runtime/'MATCHED_WORKER_ISOLATION.json').write_text(json.dumps(item))
    return item

def test_disjoint_real_directories(tmp_path):
    assert M.validate_native_isolations([isolation(tmp_path,'a'),isolation(tmp_path,'b')])

def test_identical_roots_rejected(tmp_path):
    a=isolation(tmp_path,'a');b=copy.deepcopy(a);b['worker_id']='b'
    with pytest.raises(ValueError,match='overlap'):M.validate_native_isolations([a,b])

def test_shared_cache_rejected(tmp_path):
    a=isolation(tmp_path,'a');b=isolation(tmp_path,'b');b['environment']['TMPDIR']=a['environment']['TMPDIR']
    with pytest.raises(ValueError,match='escapes'):M.validate_native_isolations([a,b])

def test_incomplete_cache_environment_rejected(tmp_path):
    a=isolation(tmp_path,'a');del a['environment']['CUDA_CACHE_PATH']
    with pytest.raises(ValueError,match='Incomplete'):M.validate_native_isolations([a])

def test_private_symlink_rejected(tmp_path):
    a=isolation(tmp_path,'a');p=Path(a['environment']['TEMP']);p.rmdir();p.symlink_to(tmp_path,target_is_directory=True)
    with pytest.raises(ValueError,match='escapes'):M.validate_native_isolations([a])

def test_worker_local_lease_excludes_duplicate(tmp_path,monkeypatch):
    import sys
    monkeypatch.delitem(sys.modules,'torch',raising=False)
    a=isolation(tmp_path,'a');plan=dict(native_runtime=a['runtime_root'],runtime_isolation=a)
    old={k:os.environ.get(k) for k in a['environment']};fd=M.activate_native_isolation(plan)
    try:
        assert os.environ['TMPDIR']==a['environment']['TMPDIR']
        with pytest.raises(BlockingIOError):M.activate_native_isolation(plan)
    finally:
        os.close(fd)
        for k,v in old.items():
            if v is None:os.environ.pop(k,None)
            else:os.environ[k]=v

def test_legacy_prefix_cannot_change_or_skip():
    old=[dict(episode_id=i,seed=340800000+i) for i in range(10)]
    Q.validate_prefix(old,old);Q.validate_prefix(old[:3],old)
    with pytest.raises(ValueError,match='prefix changed'):Q.validate_prefix(old[:2]+[dict(episode_id=2,seed=340800005)],old)

def test_task_state_reader_matches_native_worker_exactly():
    def body(path):return ast.dump(next(x for x in ast.parse(path.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='task_values'),include_attributes=False)
    assert body(ROOT/'scripts/qualify_rebuttal_matched_resets.py')==body(ROOT/'scripts/run_rebuttal_matched_native.py')

def test_roster_failure_has_no_outcome_fallback():
    tree=ast.parse((ROOT/'scripts/qualify_rebuttal_matched_resets.py').read_text())
    roster=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='roster')
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('load_checkpoint','infer_action','_replan') for n in ast.walk(roster))
    assert 'reset does not reproduce' in ast.unparse(roster)
