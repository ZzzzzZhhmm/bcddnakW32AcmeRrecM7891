#!/usr/bin/env python3
"""Recompute recorded-replay costs from durable timing and policy logs."""
import argparse
import json
from pathlib import Path

import numpy as np


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def verify(root, profile):
    if not __debug__:
        raise RuntimeError('verification requires assertions; do not run with -O')
    data=root/profile
    runs=[read_json(p) for p in root.glob('job_logs*/**/run_manifest.json')]
    runs=[r for r in runs if any(str(a).endswith('/'+profile) for a in r['argv'])]
    assert len(runs)==1, 'ambiguous or missing job identity'
    run=runs[0]
    assert run['status']=='complete' and run['exit_code']==0
    assert run['source_sha256']==run['source_sha256_after']
    argv=run['argv']
    count=int(argv[argv.index('--queries-per-episode')+1])
    warmup=int(argv[argv.index('--warmup')+1])
    episodes=read_json(data/'frozen_episodes.json')
    assert len(episodes)==int(argv[argv.index('--episodes')+1])
    runtime=read_json(data/'runtime_args.json')
    manifest=read_json(data/'manifest.json')
    stride=int(runtime['replan_steps'])
    assert stride==manifest['replan_steps']
    assert runtime['num_inference_steps']==manifest['nfe']
    expected=[]
    for is_warmup,episode,n in [(True,episodes[0],warmup)]+[(False,e,count) for e in episodes]:
        expected.extend((is_warmup,episode['dataset_index'],episode['recorded_episode'],q*stride) for q in range(n))
    rows=read_rows(data/'timings.jsonl')
    assert [(r['warmup'],r['dataset_index'],r['recorded_episode'],r['recorded_frame']) for r in rows]==expected
    telemetry=read_rows(data/'policy_telemetry.jsonl')
    headers=[r for r in telemetry if r.get('kind')=='header']
    assert len(headers)==1
    header=headers[0]
    assert header['checkpoint_sha256']==manifest['checkpoint_sha256']
    assert header['online_run_contract_sha256']==manifest['online_contract_sha256']
    assert header['root_seed']==runtime['seed']
    assert header['task_name']==manifest['task']
    replans=[r for r in telemetry if r.get('kind')=='replan']
    assert len(replans)==len(rows)
    assert [r['frame_index'] for r in replans]==[r['recorded_frame'] for r in rows]
    begins=[r for r in telemetry if r.get('kind')=='episode_begin']
    ends=[r for r in telemetry if r.get('kind')=='episode_end']
    assert len(begins)==len(ends)==len(episodes)+1
    for i,(begin,end) in enumerate(zip(begins,ends,strict=True)):
        assert begin['episode_index']==end['episode_index']
        assert end['reason']=='recorded_replay_end_no_outcome'
        n=warmup if i==0 else count
        episode_queries=[r for r in replans if r['episode_index']==begin['episode_index']]
        assert [r['frame_index'] for r in episode_queries]==list(range(0,n*stride,stride))
        assert end['executed_policy_actions']==n*stride
    qualified=read_json(data/'data_qualification.json')
    assert qualified['status']=='qualified'
    assert qualified['recorded_prefixes']==len(episodes)*count+max(0,warmup-count)
    measured=[r for r in rows if not r['warmup']]
    values=np.asarray([r['end_to_end_s'] for r in measured])
    assert np.isfinite(values).all() and (values>0).all()
    result=dict(queries=len(measured),episodes=len(episodes),warmup_queries=warmup,
                median_s=float(np.median(values)),p95_s=float(np.quantile(values,.95)),
                peak_allocated_gib=max(r['peak_allocated_gib'] for r in measured),
                peak_reserved_gib=max(r['peak_reserved_gib'] for r in measured))
    summary=read_json(data/'summary.json')
    assert summary['status']=='complete'
    for key,value in result.items():
        assert np.isclose(value,summary[key],rtol=1e-12,atol=1e-12), key
    measured_queries=[q for q,t in zip(replans,rows,strict=True) if not t['warmup']]
    result['nonzero_gate_queries']=sum(q['model']['source']['gate']!=0 for q in measured_queries)
    for label,get_value in {
        'learned_gate':lambda q:q['model']['source']['learned_gate'],
        'stagnation_score':lambda q:q['model']['source']['stagnation_score'],
        'candidate_count':lambda q:q['candidate_count'],
        'history_token_count':lambda q:(q['history_before_replan'] or {}).get('token_count',0),
        'history_event_count':lambda q:(q['history_before_replan'] or {}).get('event_count',0),
        'action_summary_count':lambda q:(q['history_before_replan'] or {}).get('action_summary_count',0),
    }.items():
        observed=[get_value(q) for q in measured_queries]
        assert np.isfinite(observed).all()
        result[label+'_range']=[min(observed),max(observed)]
    result.update(status='verified_from_timing_and_policy_logs',source_sha256=run['source_sha256'],
                  wrapper_elapsed_s=run['elapsed_s'],scope=manifest['scope'])
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--profile',default='profile200_r2')
    args=parser.parse_args()
    print(json.dumps(verify(args.root,args.profile),indent=2))
