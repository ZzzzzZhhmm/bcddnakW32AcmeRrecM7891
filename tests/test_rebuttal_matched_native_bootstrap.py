from pathlib import Path
import importlib.util
import pytest

p=Path(__file__).resolve().parents[1]/'src/fastwam/research/matched_native.py'
spec=importlib.util.spec_from_file_location('matched_worker_bootstrap',p)
M=importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def test_direct_worker_binds_the_same_assets_without_its_parent(tmp_path):
    env={}
    M.bind_local_model_root({'model_root':str(tmp_path)},env)
    assert env['DIFFSYNTH_MODEL_BASE_PATH']==str(tmp_path)
    assert list(env)==['DIFFSYNTH_MODEL_BASE_PATH']


def test_wrong_launch_assets_are_rejected_without_replacing_them(tmp_path):
    env={'DIFFSYNTH_MODEL_BASE_PATH':str(tmp_path/'other')}
    with pytest.raises(ValueError,match='differs'):
        M.bind_local_model_root({'model_root':str(tmp_path)},env)
    assert env['DIFFSYNTH_MODEL_BASE_PATH']==str(tmp_path/'other')


def test_missing_or_relative_local_root_cannot_fall_back_to_a_download(tmp_path):
    for root in [str(tmp_path/'missing'),'relative-checkpoints']:
        env={}
        with pytest.raises(ValueError,match='missing or relative'):
            M.bind_local_model_root({'model_root':root},env)
        assert env=={}
