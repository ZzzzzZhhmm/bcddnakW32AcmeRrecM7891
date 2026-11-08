#!/usr/bin/env python3
"""Exploratory leave-source-out effect probes on excluded qualification data.

This trains small CPU diagnostics, never a policy or a replacement paper head.
Fixed features and ridge strength are not selected using held-out truth.
"""
from pathlib import Path
import argparse,hashlib,io,json,sys,tarfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fastwam.research.effect_probe import ridge_predict,centered_error


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    if out.is_relative_to(ROOT):raise ValueError('diagnostic evidence must be outside the source tree')
    out.mkdir(parents=True,exist_ok=False)
    sources=[]
    for entry in json.loads(args.index.read_text()):
        archive=Path(entry['archive']);digest=hashlib.sha256(archive.read_bytes()).hexdigest()
        if digest!=entry['sha256']:raise ValueError('source archive identity differs')
        with tarfile.open(archive) as tar:
            assembled=json.loads(tar.extractfile(entry['base']+'candidate_effects.jsonl').read())
            if not assembled['cohort'].startswith('excluded_'):raise ValueError('formal TEST truth cannot train an exploratory probe')
            probe=entry['base']+entry['query_directory']+'candidate-pool-probe/'+assembled['query_id']+'/proposal.npz'
            with np.load(io.BytesIO(tar.extractfile(probe).read()),allow_pickle=False) as data:
                features={name:data[name][0].reshape(32,-1).astype(np.float64) for name in ('adapted_actions','historical_effect','predicted_effect')}
        complete=[candidate for candidate in assembled['candidates'] if candidate['endpoint_status']=='complete_horizon']
        ranks=np.asarray([int(candidate['candidate_id'].split('-')[1]) for candidate in complete])
        truth=np.asarray([candidate['observed_effect'] for candidate in complete],dtype=np.float64)
        if len(complete)<2:raise ValueError('insufficient complete candidate effects')
        sources.append(dict(group=(assembled['task'],str(assembled['episode_id'])),query_id=assembled['query_id'],
            archive_sha256=digest,features=features,ranks=ranks,truth=truth,task=assembled['task']))
    groups=sorted(set(row['group'] for row in sources))
    if len({row['query_id'] for row in sources})!=len(sources):raise ValueError('duplicate source query')
    if len(groups)<2:raise ValueError('at least two independent source groups required')
    report=dict(schema='warm.research.excluded-effect-probe.v1',relative_regularization=.1,
        scope='Exploratory excluded-source diagnostic, not original-head performance, a policy checkpoint or formal rebuttal evidence.',
        generalization_limit='Only the available held-out source groups; do not report candidate samples as independent source episodes.',folds=[])
    for group in groups:
        train=[row for row in sources if row['group']!=group];test=[row for row in sources if row['group']==group]
        truth=np.concatenate([row['truth']-row['truth'].mean(0) for row in train])
        for feature in ('adapted_actions','historical_effect','predicted_effect'):
            train_x=np.concatenate([row['features'][feature][row['ranks']]-row['features'][feature][row['ranks']].mean(0) for row in train])
            for row in test:
                # All32 sealed test features are available before outcomes.
                all_test=row['features'][feature]-row['features'][feature].mean(0)
                prediction,fit=ridge_predict(train_x,truth,all_test,relative_regularization=.1)
                name=f'{len(report["folds"]):04d}-{feature}-prediction.npy';np.save(out/name,prediction,allow_pickle=False)
                report['folds'].append(dict(test_group=list(group),train_groups=[list(g) for g in groups if g!=group],
                    query_id=row['query_id'],feature=feature,fit=fit,original_head=centered_error(row['features']['predicted_effect'][row['ranks']],row['truth']),
                    diagnostic=centered_error(prediction[row['ranks']],row['truth']),prediction_sha256=hashlib.sha256((out/name).read_bytes()).hexdigest(),
                    source_archive_sha256=row['archive_sha256'],candidate_slots=32,complete_endpoints=len(row['ranks'])))
    (out/'report.json').write_text(json.dumps(report,indent=2));(out/'input_index.json').write_bytes(args.index.read_bytes())
    print(json.dumps(report))


if __name__=='__main__':main()
