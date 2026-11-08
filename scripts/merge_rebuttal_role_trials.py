#!/usr/bin/env python3
"""Pair isolated role workers only after matching source and sealed reset data."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fastwam.research.evidence import canonical


def merge_records(records):
    by_case={};identities=set()
    for record in records:
        row=record['row'];identity=record['identity'];role,=row['roles']
        if role not in ('full','candidate_null'):raise ValueError('unknown role')
        bindings=tuple(identity[k] for k in ('core_source_sha256','collector_source_sha256',
            'checkpoint_sha256','plan_sha256','search_condition','task','phase'))
        identities.add(bindings)
        if any(row[k]!=identity[k] for k in ('checkpoint_sha256','plan_sha256','search_condition','task','phase')):
            raise ValueError('worker row differs from its bound identity')
        key=(row['task'],row['episode_id']);case=by_case.setdefault(key,{})
        if role in case:raise ValueError('duplicate role attempt: adjudicate retries explicitly')
        case[role]=record
    if len(identities)>1:raise ValueError('source/model/protocol/intervention mismatch')
    output=[]
    for (task,episode_id),case in sorted(by_case.items()):
        present=list(case.values());first=present[0]['row']
        if any(r['row']['seed']!=first['seed'] for r in present):raise ValueError('paired reset seeds differ')
        if any(r['initial_identity']!=present[0]['initial_identity'] for r in present):
            raise ValueError('sealed initial camera/qpos/task/instruction identities differ')
        roles={role:r['row']['roles'][role] for role,r in case.items()}
        valid=set(roles)=={'full','candidate_null'} and all(r['status']=='complete' for r in roles.values())
        output.append(dict(schema='warm.rebuttal.paired-episode.v1',task=task,episode_id=episode_id,
            seed=first['seed'],phase=first['phase'],checkpoint_sha256=first['checkpoint_sha256'],
            plan_sha256=first['plan_sha256'],search_condition=first['search_condition'],
            status='complete' if valid else 'incomplete_pair',roles=roles,
            full_minus_null=roles['full']['success']-roles['candidate_null']['success'] if valid else None,
            exact_initial_identity_checked=len(present)==2,
            worker_evidence=[r['worker_evidence'] for r in present]))
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();records=[]
    for directory in a.inputs:
        identity=json.loads((directory/'identity.json').read_text())
        for line in (directory/'role_trials.jsonl').read_text().splitlines():
            row=json.loads(line);role,=row['roles']
            case=directory/f"case-{row['episode_id']:04d}-seed-{row['seed']}"/role
            records.append(dict(row=row,identity=identity,worker_evidence=str(case),
                initial_identity=json.loads((case/'initial_identity.json').read_text())))
    rows=merge_records(records)
    if a.output.resolve().is_relative_to(ROOT):raise ValueError('runtime output must be outside source')
    with a.output.open('xb') as f:
        for row in rows:f.write(canonical(row)+b'\n')
    print(json.dumps(dict(pairs=len(rows),complete=sum(r['status']=='complete' for r in rows),output=str(a.output))))


if __name__=='__main__':main()
