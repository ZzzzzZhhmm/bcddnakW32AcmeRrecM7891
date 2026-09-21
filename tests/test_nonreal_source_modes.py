import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('source_modes', Path(__file__).resolve().parents[1]/'scripts/run_nonreal_source_modes.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


@pytest.mark.parametrize('confounded',[False,True])
def test_merge_requires_equal_conditioning_across_independent_processes(tmp_path,confounded):
    prefix = dict(task='test',episode_index=1,frame_index=20,dataset_index=0,dataset_sample_index=0)
    identity = {key:'same' for key in ('checkpoint_sha256','train_config_sha256','bank_sha256','query_corpus_sha256',
        'normalizer_sha256','prefix_manifest_sha256','torch_version','gpu','dtype','nfe')}
    for mode in driver.MODES:
        root = tmp_path/('mode-'+mode)
        probe = root/'probes'/('prefix-0000-'+mode)
        probe.mkdir(parents=True)
        (root/'manifest.json').write_text(json.dumps(identity))
        (root/'frozen_prefixes.json').write_text(json.dumps([prefix]))
        (root/'summary.json').write_text(json.dumps(dict(status='complete')))
        arrays = {key:np.zeros(1) for key in ('base_gaussian','conditioning','g','alpha','selected_index','adapted_actions','valid','source')}
        if confounded and mode=='gaussian':
            arrays['conditioning'] += 1
        np.savez(probe/'proposal.npz',**arrays)
        (probe/'proposal.json').write_text('{}')
        hashes = {key:hashlib.sha256((probe/filename).read_bytes()).hexdigest() for key,filename in
                  (('array_sha256','proposal.npz'),('metadata_sha256','proposal.json'))}
        row = dict(prefix,query_id='prefix-0000',mode=mode,probe=hashes,nonzero_gate=False,
                   mean_g=0,mean_noise_scale_squared=1,tolerance=1e-6 if mode=='full' else None)
        (root/'source_results.jsonl').write_text(json.dumps(row)+'\n')
        actions = {mode:np.zeros((32,14))}
        if mode=='full':
            actions['repeat']=actions[mode]
        np.savez(root/'prefix-0000-source-actions.npz',**actions)
    if confounded:
        with pytest.raises(ValueError,match='conditioning'):
            driver.merge(tmp_path)
        assert not (tmp_path/'summary.json').exists()
    else:
        result=driver.merge(tmp_path)
        assert result['queries']==1 and result['nonzero_gate_queries']==0


def test_driver_binds_one_mode_per_process(tmp_path):
    commands=driver.commands(tmp_path,['--seed','3407'])
    assert len(commands)==3
    for mode,command in zip(driver.MODES,commands):
        assert command[command.index('--source-mode')+1]==mode
        assert command[command.index('--output')+1]==str(tmp_path/('mode-'+mode))
