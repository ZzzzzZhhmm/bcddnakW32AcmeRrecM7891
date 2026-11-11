from pathlib import Path
import datetime,importlib.util
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('generation_admission',ROOT/'scripts/generate_rebuttal_native_query_null.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
NOW=datetime.datetime(2026,11,11,21,tzinfo=datetime.timezone.utc)

def admission(tmp_path, qualification=True, seconds=1800):
    return dict(schema='warm.native-DEV-q8-full-qualification.v1' if qualification else 'warm.native-DEV-q8-null-admission.v1',
        first_admitted_utc=(NOW-datetime.timedelta(seconds=1)).isoformat(),
        absolute_deadline_utc=(NOW+datetime.timedelta(seconds=seconds-1)).isoformat(),
        sealed_query_sha256='fixed-source',generation_output=str(tmp_path/'output'),
        study_kind='full_reproduction_engineering_only',null_generation_authorized=False,
        maximum_full_replays=1,maximum_physical_branches=0,previous_null_window_reopened=False)

def check(a,tmp_path,qualification=True):
    return M.require_generation_admission(a,full_qualification_only=qualification,
        seal_sha='fixed-source',output=tmp_path/'output',now=NOW)

def test_original_null_and_new_engineering_domains_remain_separate(tmp_path):
    check(admission(tmp_path),tmp_path)
    check(admission(tmp_path,False,3600),tmp_path,False)
    with pytest.raises(ValueError):check(admission(tmp_path,False),tmp_path)
    with pytest.raises(ValueError):check(admission(tmp_path),tmp_path,False)

def test_new_engineering_window_cannot_extend_or_reopen_science(tmp_path):
    with pytest.raises(ValueError):check(admission(tmp_path,seconds=1801),tmp_path)
    a=admission(tmp_path);a['absolute_deadline_utc']=NOW.isoformat()
    with pytest.raises(ValueError):check(a,tmp_path)
    a=admission(tmp_path);a['previous_null_window_reopened']=True
    with pytest.raises(ValueError):check(a,tmp_path)

def test_engineering_replay_cannot_authorize_null_physics_or_other_source(tmp_path):
    for key,value in [('null_generation_authorized',True),('maximum_physical_branches',1),('maximum_full_replays',2),('sealed_query_sha256','other-source')]:
        a=admission(tmp_path);a[key]=value
        with pytest.raises(ValueError):check(a,tmp_path)
