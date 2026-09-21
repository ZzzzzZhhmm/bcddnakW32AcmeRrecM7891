#!/usr/bin/env python3
"""Run three immutable inference processes, then validate and pair their output."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

MODES = ('full', 'scale_only', 'gaussian')


def commands(output, forwarded):
    script = Path(__file__).with_name('probe_nonreal_full_null.py')
    return [[sys.executable, str(script), *forwarded, '--experiment', 'source', '--source-mode', mode,
             '--output', str(output / ('mode-'+mode))] for mode in MODES]


def merge(output):
    def read(path):
        return json.loads(path.read_text())
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    roots = {mode: output / ('mode-'+mode) for mode in MODES}
    manifests = {mode: read(root/'manifest.json') for mode,root in roots.items()}
    prefixes = read(roots['full']/'frozen_prefixes.json')
    identities = ('checkpoint_sha256','train_config_sha256','bank_sha256','query_corpus_sha256',
                  'normalizer_sha256','prefix_manifest_sha256','torch_version','gpu','dtype','nfe')
    for mode,root in roots.items():
        require(read(root/'summary.json')['status'] == 'complete', 'mode did not finish')
        require(read(root/'frozen_prefixes.json') == prefixes, 'mode changed frozen prefixes')
        require(all(manifests[mode][key] == manifests['full'][key] for key in identities), 'mode changed scientific identity')
    records = {mode: [json.loads(line) for line in (root/'source_results.jsonl').read_text().splitlines()]
               for mode,root in roots.items()}
    for mode in MODES:
        require(len(records[mode]) == len(prefixes), 'incomplete mode records')
    paired = []
    for number,prefix in enumerate(prefixes):
        query = f'prefix-{number:04d}'
        probes, actions = {}, {}
        for mode,root in roots.items():
            row = records[mode][number]
            require(row['query_id'] == query and row['mode'] == mode and all(row[key] == value for key,value in prefix.items()), 'query identity mismatch')
            source = root/'probes'/(query+'-'+mode)
            for filename,key in (('proposal.npz','array_sha256'),('proposal.json','metadata_sha256')):
                require(hashlib.sha256((source/filename).read_bytes()).hexdigest() == row['probe'][key], 'probe hash changed')
            probes[mode] = np.load(source/'proposal.npz',allow_pickle=False)
            stored = np.load(root/(query+'-source-actions.npz'),allow_pickle=False)
            actions[mode] = stored[mode]
            if mode == 'full':
                actions['repeat'] = stored['repeat']
        tolerance = max(1e-6,float(np.max(np.abs(actions['full']-actions['repeat']))))
        require(tolerance == records['full'][number]['tolerance'], 'tolerance changed')
        for mode in MODES:
            a,b = probes['full'],probes[mode]
            for key in ('base_gaussian','conditioning','g','alpha','selected_index','adapted_actions','valid'):
                require(np.array_equal(a[key],b[key]), f'{query}/{mode}: fixed field changed: {key}')
            delta = actions[mode]-actions['full']
            row = dict(records[mode][number], tolerance=tolerance,
                       action_rms_delta=float(np.sqrt(np.mean(delta**2))), action_max_delta=float(np.max(np.abs(delta))),
                       source_rms_delta=float(np.sqrt(np.mean((a['source']-b['source'])**2))))
            target = output/'probes'/(query+'-'+mode)
            shutil.copytree(roots[mode]/'probes'/target.name,target)
            row['probe'] = dict(row['probe'],path=str(target))
            paired.append(row)
        np.savez_compressed(output/(query+'-source-actions.npz'),**actions)
    shutil.copy2(roots['full']/'frozen_prefixes.json',output/'frozen_prefixes.json')
    (output/'manifest.json').write_text(json.dumps(dict(manifests=manifests),indent=2))
    (output/'source_results.jsonl').write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in paired))
    summary = dict(status='complete',evidence_type='engineering',queries=len(prefixes),
                   episodes=len({(p['task'],p['episode_index']) for p in prefixes}),
                   nonzero_gate_queries=sum(row['nonzero_gate'] for row in paired if row['mode']=='full'),
                   modes={mode:{key:float(np.mean([r[key] for r in paired if r['mode']==mode]))
                                for key in ('mean_g','mean_noise_scale_squared','source_rms_delta','action_rms_delta')}
                          for mode in MODES},
                   limitation='smoke checkpoint source intervention, not independent training, online parity or task SR')
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args,forwarded = parser.parse_known_args()
    if any(x in forwarded for x in ('--source-mode','--experiment')):
        raise ValueError('driver owns immutable experiment and mode arguments')
    args.output.mkdir(parents=True,exist_ok=False)
    started = time.monotonic()
    for mode,command in zip(MODES,commands(args.output,forwarded)):
        print(f'stage=source_mode_start mode={mode}',flush=True)
        code = subprocess.call(command)
        with (args.output/'mode_processes.jsonl').open('a') as handle:
            handle.write(json.dumps(dict(mode=mode,argv=command,exit_code=code,elapsed_seconds=time.monotonic()-started))+'\n')
        if code:
            return code
    print(json.dumps(merge(args.output)),flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
