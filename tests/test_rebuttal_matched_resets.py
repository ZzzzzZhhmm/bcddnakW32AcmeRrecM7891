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
    item=dict(worker_id=name,runtime_root=str(runtime),writable_root=str(writable),environment=env,ipc_kind='anonymous_multiprocessing_Pipe',read_only_asset_root='/fixed/assets',persistent_lease_path=str(root/'native-worker.lock'))
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


def test_nvrtc_long_temp_root_is_rejected(tmp_path):
    a=isolation(tmp_path,'a');a['environment']['TMPDIR']=str(Path(a['writable_root'])/('x'*110))
    with pytest.raises(ValueError,match='NVRTC'):M.validate_native_isolations([a])

def test_native_lease_cannot_move_to_ephemeral_cache(tmp_path):
    a=isolation(tmp_path,'a');a['persistent_lease_path']=str(Path(a['writable_root'])/'lock')
    with pytest.raises(ValueError,match='persistent'):M.validate_native_isolations([a])


def test_official_auxiliary_front_camera_is_preserved(tmp_path,monkeypatch):
    import numpy as np,sys,types
    generator=types.ModuleType('generate_episode_instructions')
    generator.generate_episode_descriptions=lambda *_:[{'unseen':['fixed prompt']}]
    monkeypatch.setitem(sys.modules,'generate_episode_instructions',generator)
    monkeypatch.setattr(Q,'task_values',lambda _:dict(fixed=1))
    cameras={k:dict(rgb=np.full((2,3,3),i,dtype=np.uint8)) for i,k in enumerate(('head_camera','left_camera','right_camera','front_camera'))}
    env=types.SimpleNamespace(cameras=types.SimpleNamespace(static_camera_name=['head_camera','front_camera']),set_instruction=lambda **_:None,get_obs=lambda:dict(observation=cameras,joint_action=dict(vector=np.zeros(14,dtype=np.float32))))
    result=Q.observe_initial(env,'rearrange_blocks',dict(expert_info={'info':{}}),tmp_path/'initial.npz')
    assert set(result['observation_arrays'])==set(cameras)|{'native_qpos'}
    with np.load(tmp_path/'initial.npz',allow_pickle=False) as raw:assert np.array_equal(raw['front_camera'],cameras['front_camera']['rgb'])

def test_missing_official_front_camera_is_not_silently_dropped(tmp_path,monkeypatch):
    import numpy as np,sys,types
    generator=types.ModuleType('generate_episode_instructions');generator.generate_episode_descriptions=lambda *_:[{'unseen':['fixed']}]
    monkeypatch.setitem(sys.modules,'generate_episode_instructions',generator)
    cameras={k:dict(rgb=np.zeros((2,3,3),dtype=np.uint8)) for k in ('head_camera','left_camera','right_camera')}
    env=types.SimpleNamespace(cameras=types.SimpleNamespace(static_camera_name=['head_camera','front_camera']),set_instruction=lambda **_:None,get_obs=lambda:dict(observation=cameras,joint_action=dict(vector=np.zeros(14,dtype=np.float32))))
    with pytest.raises(ValueError,match='inventory differs'):Q.observe_initial(env,'rearrange_blocks',dict(expert_info={'info':{}}),tmp_path/'none.npz')


def test_only_pinned_geometric_expert_exception_is_a_rejection():
    detail=dict(stage='play_once',error_type='AssertionError',error_message='target_pose cannot be None for move action.',frames=[dict(name='grasp_actor',filename='/fixed/envs/_base_task.py'),dict(name='__init__',filename='/fixed/envs/utils/action.py')])
    assert Q.known_expert_geometry_rejection(detail)
    for key,value in [('stage','setup'),('error_type','RuntimeError'),('error_message','CUDA out of memory')]:
        bad=copy.deepcopy(detail);bad[key]=value;assert not Q.known_expert_geometry_rejection(bad)
    bad=copy.deepcopy(detail);bad['frames'][-1]['filename']='/model/action.py';assert not Q.known_expert_geometry_rejection(bad)

def test_unknown_uninspected_reset_exit_cannot_resume(tmp_path):
    source=tmp_path/'old';source.mkdir();out=tmp_path/'new';out.mkdir()
    proof=dict(schema='warm.rebuttal.reset-resume-reconciliation.v1',source_root=str(source),task='battery_try',native_source_sha256='sim',known_exit=None,original_window_reset=False,policy_outcomes_used=False)
    path=tmp_path/'proof.json';path.write_text(json.dumps(proof))
    plan=dict(task='battery_try',runtime_source_sha256='sim',resume_proof=dict(path=str(path),sha256=M.file_sha(path)))
    with pytest.raises(ValueError,match='uninspected'):Q.prepare_resume(plan,out)

def test_resume_requires_raw_prefix_hashes(tmp_path):
    source=tmp_path/'old';(source/'candidates').mkdir(parents=True);(source/'candidates/a').write_text('original')
    out=tmp_path/'new';out.mkdir()
    proof=dict(schema='warm.rebuttal.reset-resume-reconciliation.v1',source_root=str(source),task='battery_try',native_source_sha256='sim',known_exit=1,original_window_reset=False,policy_outcomes_used=False,source_manifest={'candidates/a':dict(sha256='changed')})
    path=tmp_path/'proof.json';path.write_text(json.dumps(proof))
    plan=dict(task='battery_try',runtime_source_sha256='sim',resume_proof=dict(path=str(path),sha256=M.file_sha(path)))
    with pytest.raises(ValueError,match='prefix changed'):Q.prepare_resume(plan,out)
