#!/usr/bin/env python3
"""Describe source activation coverage from verified saved probes, without a GPU.

This is not false/true acceptance: no independent applicability labels exist.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from verify_nonreal_source_archive import verify


def analyze(root):
    verification = verify(root)
    rows = []
    for prefix in json.loads((root/'dev50/frozen_prefixes.json').read_text()):
        number = len(rows)
        directory = root/'dev50/probes'/f'prefix-{number:04d}-full'
        metadata = json.loads((directory/'proposal.json').read_text())
        if metadata['force_null'] or metadata['source_mode'] != 'full':
            raise ValueError('requires natural full-source probes')
        arrays = np.load(directory/'proposal.npz',allow_pickle=False)
        alpha, gate, eligible, zeta = [arrays[key].reshape(-1) for key in ('alpha','g','v_det','zeta')]
        if any(len(value) != 1 for value in (alpha,gate,eligible,zeta)):
            raise ValueError('this report requires batch=1')
        row = dict(prefix,query_id=f'prefix-{number:04d}',raw_alpha=float(alpha[0]),effective_g=float(gate[0]),
                   v_det=bool(eligible[0]),stagnation=float(zeta[0]),valid_candidates=int(arrays['valid'].sum()),
                   nominal_gate_threshold=float(metadata['gate_threshold']))
        # Keep predicates separate: veto and low alpha may overlap.
        row['alpha_below_nominal_threshold'] = row['raw_alpha'] < row['nominal_gate_threshold']
        rows.append(row)
    def aggregate(group):
        return dict(queries=len(group),episodes=len({(r['task'],r['episode_index']) for r in group}),
                    nonzero_g=sum(r['effective_g'] != 0 for r in group),
                    empty_candidates=sum(r['valid_candidates'] == 0 for r in group),
                    v_det_false=sum(not r['v_det'] for r in group),
                    eligible_alpha_below_nominal_threshold=sum(r['v_det'] and r['alpha_below_nominal_threshold'] for r in group),
                    alpha_min=float(min(r['raw_alpha'] for r in group)),alpha_max=float(max(r['raw_alpha'] for r in group)),
                    alpha_mean=float(np.mean([r['raw_alpha'] for r in group])),
                    stagnation_min=min(r['stagnation'] for r in group),stagnation_max=max(r['stagnation'] for r in group))
    return dict(schema='warm.gate-activation-diagnostic.v1',evidence_type='engineering',verification=verification,
                overall=aggregate(rows),tasks={task:aggregate([r for r in rows if r['task']==task]) for task in sorted({r['task'] for r in rows})},
                rows=rows,limitation='No applicability labels, no FA/TA or task SR. Alpha comparisons use nominal float threshold; saved effective g is authoritative.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=analyze(args.run_root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as handle:
        json.dump(report,handle,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({key:report[key] for key in ('overall','tasks','limitation')}))
