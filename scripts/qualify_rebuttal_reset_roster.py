"""Freeze the original expert-admitted reset order independently of policy outcomes."""
from __future__ import annotations
import argparse,hashlib,importlib,json,os,random,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--core-code',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--task',choices=('rearrange_blocks','battery_try'),required=True)
    p.add_argument('--count',type=int,default=10);p.add_argument('--seed-start',type=int,default=340800000);p.add_argument('--candidate-limit',type=int,default=2000)
    a=p.parse_args();assert 1<=a.count<=50 and a.seed_start==340800000 and a.candidate_limit==2000
    core=a.core_code.resolve(strict=True);out=a.output.resolve();assert not out.is_relative_to(core) and not out.is_relative_to(ROOT)
    out.mkdir(parents=True,exist_ok=False);sys.path[:0]=[str(core),str(core/'src')]
    from scripts.qualify_rebuttal_rmbench_task import resolved_config,source_identity as runtime_identity
    from scripts.nonreal_job import source_identity
    from fastwam.research.evidence import canonical
    source=source_identity(core);assert source['sha256']=='5d7748e77a648511c5ddcfc0c65c3fac9933bbeb0e2bff20c61a7fd7e6356638'
    runtime=a.runtime.resolve(strict=True);sim=runtime_identity(runtime)
    identity=dict(schema='warm.rebuttal.reset-roster.v1',task=a.task,core_source_sha256=source['sha256'],runtime_source_sha256=sim['sha256'],
        collector_source_sha256=source_identity(ROOT)['sha256'],
        seed_start=a.seed_start,candidate_limit=a.candidate_limit,requested_cases=a.count,
        scope='Original official expert admission only; no Full/null outcomes used for reset selection.')
    (out/'identity.json').write_bytes(canonical(identity));os.chdir(runtime);sys.path[:0]=[str(runtime),str(runtime/'description/utils')]
    from envs.utils.create_actor import UnStableError
    env=getattr(importlib.import_module('envs.'+a.task),a.task)();accepted=[]
    try:
        for offset in range(a.candidate_limit):
            seed=a.seed_start+offset;config=resolved_config(runtime,a.task,seed);config.update(now_ep_num=len(accepted),eval_mode=True,is_test=True)
            row=dict(seed=seed,candidate_offset=offset,episode_id_if_admitted=len(accepted))
            try:
                random.seed(seed);env.setup_demo(**config);info=env.play_once();env.close_env()
                ok=bool(env.plan_success and env.check_success());row.update(status='accepted' if ok else 'expert_rejected',plan_success=bool(env.plan_success))
            except UnStableError as exc:
                row.update(status='unstable_seed',error=str(exc));ok=False;env.close_env()
            except Exception as exc:
                row.update(status='invalid_runtime',error=repr(exc))
                with (out/'admission.jsonl').open('ab') as stream:stream.write(canonical(row)+b'\n');stream.flush();os.fsync(stream.fileno())
                raise
            with (out/'admission.jsonl').open('ab') as stream:
                stream.write(canonical(row)+b'\n');stream.flush();os.fsync(stream.fileno())
            if ok:accepted.append(dict(episode_id=len(accepted),seed=seed))
            print(json.dumps(dict(stage='expert_admission',**row,accepted_count=len(accepted))),flush=True)
            if len(accepted)==a.count:break
        if len(accepted)!=a.count:raise RuntimeError('original candidate seed budget exhausted')
        final=dict(identity,status='qualified_reset_roster_only',entries=accepted,core_source_sha256_after=source_identity(core)['sha256'],
            runtime_source_sha256_after=runtime_identity(runtime)['sha256'])
        assert final['core_source_sha256_after']==identity['core_source_sha256'] and final['runtime_source_sha256_after']==identity['runtime_source_sha256']
        (out/'roster.json').write_bytes(canonical(final));print(json.dumps(final),flush=True)
    finally:
        if hasattr(env,'scene'):env.close_env(clear_cache=True)


if __name__=='__main__':main()
