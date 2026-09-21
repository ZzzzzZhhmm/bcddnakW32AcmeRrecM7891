#!/usr/bin/env python3
"""Independently check fixed-prefix source interventions from saved arrays."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def verify(root):
    if not __debug__:
        raise RuntimeError('do not disable verification assertions with -O')
    data = root / 'dev50'
    summary = json.loads((data / 'summary.json').read_text())
    prefixes = json.loads((data / 'frozen_prefixes.json').read_text())
    rows = [json.loads(line) for line in (data / 'source_results.jsonl').read_text().splitlines()]
    runs = [json.loads(path.read_text()) for path in (root / 'job_logs').glob('*/run_manifest.json')]
    runs = [run for run in runs if any(arg.endswith('/dev50') for arg in run['argv'])]
    assert len(runs) == 1 and runs[0]['status'] == 'complete' and runs[0]['exit_code'] == 0
    assert runs[0]['source_sha256'] == runs[0]['source_sha256_after']
    assert summary['queries'] == len(prefixes) and len(rows) == 3 * len(prefixes)
    assert len({(r['query_id'],r['mode']) for r in rows}) == len(rows)
    nonzero = 0
    reconstructed = {mode: [] for mode in ('full','scale_only','gaussian')}
    for number, prefix in enumerate(prefixes):
        query = f'prefix-{number:04d}'
        modes = {row['mode']: row for row in rows if row['query_id'] == query}
        assert set(modes) == {'full','scale_only','gaussian'}
        probes = {}
        actions = np.load(data / (query+'-source-actions.npz'), allow_pickle=False)
        assert all(np.isfinite(actions[key]).all() and actions[key].shape == (32,14) for key in actions.files)
        tolerance = max(1e-6, float(np.max(np.abs(actions['full']-actions['repeat']))))
        for mode,row in modes.items():
            assert all(row[key] == value for key,value in prefix.items())
            directory = data / 'probes' / (query+'-'+mode)
            for filename,key in (('proposal.npz','array_sha256'),('proposal.json','metadata_sha256')):
                assert hashlib.sha256((directory/filename).read_bytes()).hexdigest() == row['probe'][key]
            probes[mode] = np.load(directory/'proposal.npz', allow_pickle=False)
            assert row['tolerance'] == tolerance
        for mode,row in modes.items():
            probe, reference = probes[mode], probes['full']
            for key in ('g','alpha','conditioning','selected_index','valid','adapted_actions','base_gaussian'):
                assert np.array_equal(probe[key],reference[key])
            delta = actions[mode]-actions['full']
            metrics = dict(mean_g=float(probe['g'].mean()),
                           mean_noise_scale_squared=float(np.mean(probe['source_noise_scale']**2)),
                           source_rms_delta=float(np.sqrt(np.mean((probe['source']-reference['source'])**2))),
                           action_rms_delta=float(np.sqrt(np.mean(delta**2))))
            for key,value in metrics.items():
                assert np.isclose(value,row[key],atol=1e-6,rtol=1e-5)
            assert row['nonzero_gate'] == bool(probe['g'].any())
            reconstructed[mode].append(metrics)
        assert np.array_equal(probes['gaussian']['source'],probes['gaussian']['base_gaussian'])
        nonzero += int(bool(probes['full']['g'].any()))
    assert summary['nonzero_gate_queries'] == nonzero
    for mode,values in reconstructed.items():
        for key in values[0]:
            assert np.isclose(np.mean([v[key] for v in values]),summary['modes'][mode][key],atol=1e-6,rtol=1e-5)
    return dict(status='verified_from_arrays',queries=len(prefixes),nonzero_gate_queries=nonzero,
                source_sha256=runs[0]['source_sha256'],modes=summary['modes'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.root)))
