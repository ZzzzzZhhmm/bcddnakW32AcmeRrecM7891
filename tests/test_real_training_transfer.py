"""CPU handoff checks using synthetic fixtures; not robot or GPU qualification."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from fastwam.preprocessing.config import load_config
from fastwam.preprocessing.contracts import PreparationError, read_json, write_json, file_sha256
from fastwam.preprocessing.prepare import prepare
from fastwam.preprocessing.features import precompute
from fastwam.preprocessing.memory import build_memory
from fastwam.real.preprocessing.contract import PIPER_JOINT_CONTROL_MODE
from fastwam.real.preprocessing.piper import PiperTeleopAdapter
from fastwam.real.episodes import ContractError
from tests.test_unified_preprocessing import config, manifest, real_episode, FixtureEncoder

REPO = Path(__file__).resolve().parents[1]


def joint_fixture(cfg, name, *, offset=1, split="train"):
    entry = real_episode(cfg, name, offset=offset, split=split)
    p = Path(cfg["source"]["root"]) / name / "commands.jsonl"
    rows = [json.loads(s) for s in p.read_text().splitlines()]
    for i,row in enumerate(rows):
        row["joint_target_rad"] = (np.arange(6)*.1 + offset*.01 + i*.001).tolist()
    p.write_text("\n".join(json.dumps(r) for r in rows)+"\n", encoding="utf-8")
    return entry


def test_joint_prepare_and_optional_memory_use_accepted_targets(tmp_path):
    cfg = config(tmp_path)
    cfg["profile"]["control_mode"] = PIPER_JOINT_CONTROL_MODE
    entries = [joint_fixture(cfg,"a",offset=1),joint_fixture(cfg,"b",offset=20),joint_fixture(cfg,"dev",offset=80,split="dev")]
    manifest(cfg,entries)
    cfgfile=tmp_path/'config.json';write_json(cfgfile,cfg)
    cfg=load_config(cfgfile)
    raw=PiperTeleopAdapter(cfg['source'],cfg['profile']).read(entries[0])
    np.testing.assert_allclose(raw.actions[0,:6],np.arange(6)*.1+.01)
    assert not np.array_equal(raw.actions[0,:6],raw.states[0,:6])
    # Exact TCP state remains unchanged relative to the legacy adapter.
    tcp_profile={**cfg['profile'],'control_mode':'base_delta_tcp_rotvec_plus_absolute_gripper_width_m'}
    tcp=PiperTeleopAdapter(cfg['source'],tcp_profile).read(entries[0])
    np.testing.assert_array_equal(raw.states,tcp.states)
    p=prepare(cfg);stats=read_json(p/'dataset_stats.json')
    assert stats['num_episodes']==2 and stats['num_transition']==16
    assert stats['action']['default']['global_max'][0]<.3  # dev .8 excluded
    import pyarrow.parquet as pq
    table=next((p/'dataset/data').rglob('episode_000000.parquet'))
    np.testing.assert_allclose(pq.read_table(table)['action'].to_pylist(),raw.actions,atol=1e-7)
    precompute(cfg,_test_encoder=FixtureEncoder())
    m=build_memory(cfg)
    assert read_json(m/'COMPLETE.json')['source_episodes']==2
    # TCP proprio and joint actions must not be subtracted from one another.
    assert read_json(m/'training_bindings.json')['recollection_action_mode']=='delta'


@pytest.mark.parametrize('bad',[None,[1]*5,[1,2,3,4,5,float('nan')]])
def test_joint_mode_rejects_missing_malformed_or_nonfinite_targets(tmp_path,bad):
    cfg=config(tmp_path);cfg['profile']['control_mode']=PIPER_JOINT_CONTROL_MODE
    entry=joint_fixture(cfg,'a');p=Path(cfg['source']['root'])/'a/commands.jsonl'
    rows=[json.loads(s) for s in p.read_text().splitlines()];rows[0]['joint_target_rad']=bad
    p.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
    with pytest.raises((PreparationError, ContractError),match='joint_target_rad|non-finite'):
        PiperTeleopAdapter(cfg['source'],cfg['profile']).read(entry)


def test_unknown_action_mode_still_fails(tmp_path):
    cfg=config(tmp_path);cfg['profile']['control_mode']='joint_velocity_mislabeled'
    p=tmp_path/'config.json';write_json(p,cfg)
    with pytest.raises(PreparationError,match='control_mode'):load_config(p)


def load_script(name,monkeypatch):
    folder=REPO/'scripts/real';monkeypatch.syspath_prepend(str(folder))
    spec=importlib.util.spec_from_file_location(name,folder/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_plain_configuration_has_no_memory_dependency_and_detects_tamper(tmp_path,monkeypatch):
    from omegaconf import OmegaConf
    from fastwam.preprocessing.prepare import training_data_config
    from types import SimpleNamespace
    # Explicit metadata-only unit fixture, never a real dataset release.
    proc=tmp_path/'processed';p=proc/'prepared';p.mkdir(parents=True)
    template=read_json(REPO/'configs/real/transfer/preprocess_joint.template.json')
    info={'features':{f'observation.images.{k}':{'shape':[480,640,3]} for k in ['external','wrist']}}
    write_json(p/'dataset/meta/info.json',info)
    data=training_data_config(template['profile'],[str(p/'dataset')],p,SimpleNamespace(episodes=[SimpleNamespace(split='train'),SimpleNamespace(split='dev')]))
    (p/'data_config.yaml').write_text(OmegaConf.to_yaml(OmegaConf.create(data)))
    marker={'synthetic':False,'profile':template['profile'],'files':{'data_config.yaml':file_sha256(p/'data_config.yaml')}}
    write_json(p/'COMPLETE.json',marker)
    ckpt=tmp_path/'init.pt';ckpt.write_bytes(b'unit fixture; not model weights')
    mod=load_script('configure_transfer_training',monkeypatch)
    kwargs=dict(processed=proc,run_dir=tmp_path/'run',text_cache=tmp_path/'text',model_config=REPO/'configs/model/fastwam.yaml',checkpoint=ckpt)
    cfg=mod.compose(**kwargs)
    assert cfg.model.proprio_dim==7 and cfg.model.action_dit_config.action_dim==7
    assert cfg.data.train.num_frames==33 and 'warm_candidates' not in cfg.data
    assert cfg.resume==str(ckpt.resolve()) and cfg.run_steps is None
    assert not (proc/'memory').exists()
    # Exercise the documented Hydra config-name route without loading a GPU model.
    from hydra import initialize_config_dir, compose
    generated=tmp_path/'training_configs';generated.mkdir()
    (generated/'train_new_tasks.local.yaml').write_text(OmegaConf.to_yaml(cfg))
    with initialize_config_dir(config_dir=str(generated),version_base='1.3'):
        rebuilt=compose(config_name='train_new_tasks.local',overrides=['run_steps=2'])
        assert rebuilt.run_steps==2 and rebuilt.model.proprio_dim==7
    (p/'data_config.yaml').write_text('tampered')
    with pytest.raises(ValueError,match='changed'):mod.compose(**kwargs)


def test_bundle_excludes_runtime_assets(monkeypatch):
    mod=load_script('package_training_transfer',monkeypatch)
    for n in ['src/fastwam/model.pt','configs/real/piper.local.yaml','scripts/tmp/key.json','real/piper/raw/data.json','checkpoints/token.json','.env']:
        assert not mod.eligible(n)
    assert mod.eligible('src/fastwam/real/episodes.py')
    assert mod.eligible('docs/real/WAM_TRANSFER_ZH.md')
