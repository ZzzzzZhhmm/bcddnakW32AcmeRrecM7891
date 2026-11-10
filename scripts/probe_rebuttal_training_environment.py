#!/usr/bin/env python3
"""Diagnose an ACP resume environment before expensive model/dataset loading."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from fastwam.research.training_environment import (
    capture_static_training_environment, compare_static_training_environment,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent-attestation', type=Path, required=True)
    parser.add_argument('--expected-parent-attestation-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve the existing diagnostic; use its receipt before any retry')
    raw = args.parent_attestation.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != args.expected_parent_attestation_sha256:
        raise ValueError('Frozen parent attestation changed')
    parent = json.loads(raw)
    if parent.get('schema') != 'warm.training-attestation' or parent.get('actual_global_step') != 200:
        raise ValueError('This diagnostic is bound to the original step200 Full parent')
    if parent.get('source_policy') != 'fixed_context_top1':
        raise ValueError('Unexpected training role')
    observed = capture_static_training_environment()
    result = compare_static_training_environment(parent['training_runtime'], observed)
    result.update(schema='warm.rebuttal.training-environment-probe.v1',
                  checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  parent_attestation_sha256=digest, parent_checkpoint_sha256=parent['checkpoint_sha256'],
                  parent_step=parent['actual_global_step'], observed_static_environment=observed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive output prevents an unknown diagnostic submission from being overwritten.
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    return 2 if result['differences'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
