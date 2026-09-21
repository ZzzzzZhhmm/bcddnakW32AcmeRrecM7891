#!/usr/bin/env python3
"""Recompute a completed full-null check from archived arrays, not its summary."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def verify(root):
    if not __debug__:
        raise RuntimeError('verification requires Python assertions enabled; do not use -O')
    data = root / "dev50"
    summary = json.loads((data / "summary.json").read_text())
    identity = json.loads((data / "manifest.json").read_text())
    prefixes = json.loads((data / "frozen_prefixes.json").read_text())
    rows = [json.loads(line) for line in (data / "query_results.jsonl").read_text().splitlines()]
    runs = [json.loads(path.read_text()) for path in (root / "job_logs").glob("*/run_manifest.json")]
    runs = [run for run in runs if any(arg.endswith("/dev50") for arg in run['argv'])]
    assert len(runs) == 1 and runs[0]['status'] == 'complete' and runs[0]['exit_code'] == 0
    assert runs[0]['source_sha256'] == runs[0]['source_sha256_after']
    assert hashlib.sha256((data / 'frozen_prefixes.json').read_bytes()).hexdigest() == identity['prefix_manifest_sha256']
    assert len(rows) == len(prefixes) == summary['queries']
    assert len({row['query_id'] for row in rows}) == len(rows)
    maximum = 0.0
    for row, prefix in zip(rows, prefixes):
        assert all(row[key] == value for key,value in prefix.items())
        arrays = np.load(data / (row['query_id'] + '-actions.npz'), allow_pickle=False)
        assert all(np.isfinite(arrays[key]).all() and arrays[key].shape == (32,14) for key in ('base','repeat','changed'))
        identical = float(np.max(np.abs(arrays['base']-arrays['repeat'])))
        error = float(np.max(np.abs(arrays['base']-arrays['changed'])))
        assert identical == row['identical_max_error'] and error == row['changed_max_error']
        assert row['tolerance'] == max(1e-6, identical) and error <= row['tolerance']
        probe_arrays = []
        for kind in ('base', 'changed'):
            directory = data / 'probes' / (row['query_id'] + '-' + kind)
            metadata = directory / 'proposal.json'
            payload = directory / 'proposal.npz'
            assert hashlib.sha256(metadata.read_bytes()).hexdigest() == row[kind+'_probe']['metadata_sha256']
            assert hashlib.sha256(payload.read_bytes()).hexdigest() == row[kind+'_probe']['array_sha256']
            proposal = np.load(payload, allow_pickle=False)
            assert np.array_equal(proposal['source'], proposal['base_gaussian']) and not proposal['g'].any()
            probe_arrays.append(proposal)
        assert np.array_equal(probe_arrays[0]['conditioning'], probe_arrays[1]['conditioning'])
        assert row['passed'] is True and row['null_paths'] is True
        maximum = max(maximum,error)
    assert summary['status'] == 'passed' and maximum == summary['max_error']
    return dict(status='verified_from_arrays', queries=len(rows), episodes=len({(r['task'],r['episode_index']) for r in rows}),
                max_error=maximum, source_sha256=runs[0]['source_sha256'], checkpoint_sha256=identity['checkpoint_sha256'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.root)))
