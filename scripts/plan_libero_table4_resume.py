#!/usr/bin/env python3
"""Validate completed artifacts and partition only missing initial states."""
import argparse
import json
from pathlib import Path

from merge_libero_table4 import load_shards
from assemble_nonreal_candidates import assemble
from fastwam.research.libero_table4 import make_plan
from fastwam.research.evidence import canonical


def plan_resume(reuse, workers):
    if workers not in range(1, 5):
        raise ValueError('request one to four workers')
    finished = set()
    checkpoint = None
    if reuse:
        index, branches, labels, plans, _, _ = load_shards(reuse)
        expected = make_plan()
        protocol = ('seed','horizon','top_k','replan_steps','outcome','policy','effect_head',
                    'effect_projection','candidate_metrics','bootstrap','metrics_seed')
        for p in plans:
            if p['cohort'] != 'libero10-table4-s3407-v1' or any(p[k] != expected[k] for k in protocol):
                raise ValueError('reused shard disagrees with the frozen collection protocol')
            finished.update(p['episodes'])
        rows = assemble(index, branches, labels)
        identities = {r['checkpoint_sha256'] for r in rows}
        if len(identities) != 1:
            raise ValueError('reused shards must use one checkpoint')
        checkpoint = next(iter(identities))
        for directory in reuse:
            receipt = json.loads((directory/'execution.json').read_text())
            if receipt['checkpoint_sha256'] != checkpoint:
                raise ValueError('completion receipt and proposal checkpoint disagree')
            q = [json.loads(line) for line in (directory/'qualification.jsonl').read_text().splitlines()]
            passed = {int(r['query_id'].split('task')[1].split('-')[0]) for r in q if r.get('status') == 'passed'}
            repeat = {int(r['query_id'].split('task')[1].split('-')[0]) for r in q if r.get('dino_repeat_max_abs', float('inf')) <= 1e-6}
            if passed != set(range(10)) or repeat != set(range(10)):
                raise ValueError('reused shard lacks per-task restoration/encoding qualification')
        for row in rows:
            if not 1 <= len(row['candidates']) <= 32 or any(
                    c['endpoint_status'] not in ('complete_horizon','incomplete_horizon')
                    or c['applicable'] not in (0,1) for c in row['candidates']):
                raise ValueError('reused shard has unresolved candidates or labels')
    missing = sorted(set(range(10)) - finished)
    count = min(workers, len(missing))
    return dict(schema='warm.libero.table4.resume.v1',
                reuse_shards=[str(p.resolve()) for p in reuse],
                completed_initial_states=sorted(finished), missing_initial_states=missing,
                episode_groups=[missing[rank::count] for rank in range(count)],
                checkpoint_sha256=checkpoint, planned_source_episodes=100,
                new_source_episodes=10*len(missing))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse-shards', nargs='*', type=Path, default=[])
    parser.add_argument('--workers', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = plan_resume(args.reuse_shards, args.workers)
    with args.output.open('xb') as stream:
        stream.write(canonical(result))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
