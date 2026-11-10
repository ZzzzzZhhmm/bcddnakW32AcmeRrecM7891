#!/usr/bin/env python3
"""Complete an environment measurement without admitting a training resume.

The strict probe keeps its original exit-2 mismatch behavior. This measurement
operator preserves that exit and the mismatch data; success only means the
static descriptors were collected, never that WARM training may resume.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe-code',type=Path,required=True)
    parser.add_argument('--expected-probe-source-sha256',required=True)
    parser.add_argument('--parent-attestation',type=Path,required=True)
    parser.add_argument('--expected-parent-attestation-sha256',required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    args=parser.parse_args()
    code=args.probe_code.resolve(strict=True)
    sys.path[:0]=[str(code),str(code/'src')]
    from scripts.nonreal_job import source_identity
    if source_identity(code)['sha256']!=args.expected_probe_source_sha256:
        raise ValueError('Frozen probe implementation changed')
    if args.output_root.resolve().is_relative_to(code):
        raise ValueError('Runtime measurements must remain outside source')
    args.output_root.mkdir(parents=True,exist_ok=False)
    output=args.output_root/'environment.json'
    argv=[sys.executable,str(code/'scripts/probe_rebuttal_training_environment.py'),
          '--parent-attestation',str(args.parent_attestation),
          '--expected-parent-attestation-sha256',args.expected_parent_attestation_sha256,
          '--output',str(output)]
    environment=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
    started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    completed=subprocess.run(argv,cwd=code,env=environment,capture_output=True,text=True,timeout=300)
    (args.output_root/'probe.stdout.log').write_text(completed.stdout)
    (args.output_root/'probe.stderr.log').write_text(completed.stderr)
    if completed.returncode not in (0,2) or not output.is_file():
        raise RuntimeError('Probe failed to collect a complete measurement; retain logs')
    observed=json.loads(output.read_text())
    if (observed.get('phase')!='pre_model_environment_only'
            or observed.get('optimizer_updates')!=0
            or observed.get('formal_resume_validation_bypassed') is not False
            or observed.get('full_numerical_runtime_verified') is not False):
        raise ValueError('Measurement output claims an unsupported resume admission')
    differs=bool(observed['differences'])
    if completed.returncode!=(2 if differs else 0):
        raise ValueError('Probe exit and recorded differences disagree')
    if source_identity(code)['sha256']!=args.expected_probe_source_sha256:
        raise ValueError('Probe source changed during measurement')
    receipt=dict(schema='warm.rebuttal.environment-measurement.v1',
        status='measurement_completed_not_resume_admitted',started_utc=started,
        ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        argv=argv,probe_exit_code=completed.returncode,parent_static_environment_matches=not differs,
        differences=observed['differences'],optimizer_updates=0,model_loaded=False,
        resume_admitted=False,formal_resume_validation_bypassed=False,
        scope='Static environment measurement only; an ACP success status means data collection completed, not numerical/training equivalence.')
    (args.output_root/'measurement_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
