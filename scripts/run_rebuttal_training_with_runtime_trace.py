#!/usr/bin/env python3
"""Observe the original trainer's numerical runtime without changing its result.

Accelerate launches this observer in place of the Python entrypoint only. The
original script, arguments, configuration and strict resume validator still run.
Returned runtime objects are passed through unchanged, including mismatches.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parents[1]
_MISSING = object()


def source_digest(root):
    # Match the logged runner's filesystem digest without importing its fastwam
    # package before the bound original training package has been selected.
    records = {}
    for directory in ('src', 'scripts', 'configs', 'experiments', 'tests'):
        for path in sorted((Path(root) / directory).rglob('*')):
            if (path.is_file() and '__pycache__' not in path.parts
                    and path.suffix in {'.py', '.sh', '.json', '.yaml', '.yml'}):
                records[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    canonical = json.dumps(records, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(canonical).hexdigest()


def differences(expected, actual, path=''):
    """Retain every differing field and its type; never filter runtime facts."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        rows = []
        for key in sorted(expected.keys() | actual.keys()):
            rows.extend(differences(expected.get(key, _MISSING),
                                    actual.get(key, _MISSING),
                                    path + '.' + key if path else key))
        return rows
    if type(expected) is type(actual) and expected == actual:
        return []
    row = dict(field=path, expected_present=expected is not _MISSING,
               observed_present=actual is not _MISSING)
    if expected is not _MISSING:
        row['expected'] = expected
    if actual is not _MISSING:
        row['observed'] = actual
    return [row]


def write_exclusive(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def serialized_parent_runtime(parent):
    if (parent.get('schema') != 'warm.training-attestation'
            or parent.get('actual_global_step') != 200
            or parent.get('source_policy') != 'fixed_context_top1'):
        raise ValueError('Observer requires the original Full update200 attestation')
    runtime = parent['training_runtime']
    if not isinstance(runtime, dict):
        raise TypeError('Serialized training_runtime must be an object')
    return runtime


class RuntimeObserver:
    """A diagnostic tee, never a runtime-normalization or admission layer."""

    def __init__(self, capture, expected_runtime, output):
        self.capture = capture
        self.expected_runtime = expected_runtime
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.calls = 0

    def __call__(self, *args, **kwargs):
        observed = self.capture(*args, **kwargs)
        self.calls += 1
        write_exclusive(self.output / f'runtime-{self.calls:04d}.json', dict(
            schema='warm.rebuttal.post-prepare-runtime-observation.v1',
            utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            phase='original_trainer_runtime_capture_return',
            observed_runtime=observed, expected_parent_runtime=self.expected_runtime,
            differences=differences(self.expected_runtime, observed),
            capture_call=self.calls, returned_object_modified=False,
            formal_resume_validation_bypassed=False,
            resume_admission_inferred=False,
            scope='Observed original capture return only; the original validator must compare it. No training-update or resume-success inference.'))
        return observed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-code', type=Path, required=True)
    parser.add_argument('--expected-training-source-sha256', required=True)
    parser.add_argument('--expected-observer-source-sha256', required=True)
    parser.add_argument('--parent-attestation', type=Path, required=True)
    parser.add_argument('--expected-parent-attestation-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('training_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.training_args or args.training_args[0] != '--':
        raise ValueError('Forward original training arguments after an explicit --')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('This observation is bound to the original single-rank segment')
    code = args.training_code.resolve(strict=True)
    entrypoint = code / 'scripts/train.py'
    if source_digest(ROOT) != args.expected_observer_source_sha256:
        raise ValueError('Frozen observer source changed')
    if source_digest(code) != args.expected_training_source_sha256:
        raise ValueError('Frozen original training source changed')
    parent_bytes = args.parent_attestation.read_bytes()
    if hashlib.sha256(parent_bytes).hexdigest() != args.expected_parent_attestation_sha256:
        raise ValueError('Original parent attestation changed')
    parent = json.loads(parent_bytes)
    expected_runtime = serialized_parent_runtime(parent)
    output = args.output.resolve()
    if output.is_relative_to(ROOT) or output.is_relative_to(code):
        raise ValueError('Runtime observations must remain outside both source snapshots')
    sys.path[:0] = [str(code / 'src'), str(code)]
    from fastwam.models.warm import training_attestation
    if not Path(training_attestation.__file__).resolve().is_relative_to(code):
        raise ValueError('A different attestation implementation was imported')
    original = training_attestation.capture_training_runtime
    observer = RuntimeObserver(original, expected_runtime, output)
    binding = dict(schema='warm.rebuttal.training-runtime-observer-binding.v1',
                   original_training_source_sha256=args.expected_training_source_sha256,
                   observer_source_sha256=args.expected_observer_source_sha256,
                   parent_attestation_sha256=args.expected_parent_attestation_sha256,
                   original_entrypoint=str(entrypoint),
                   original_entrypoint_sha256=hashlib.sha256(entrypoint.read_bytes()).hexdigest(),
                   forwarded_argv=[str(entrypoint), *args.training_args[1:]],
                   parent_step=parent['actual_global_step'],
                   formal_resume_validation_bypassed=False)
    write_exclusive(output / 'binding.json', binding)
    training_attestation.capture_training_runtime = observer
    previous_argv = sys.argv
    status, error = 'original_entrypoint_returned', None
    try:
        sys.argv = binding['forwarded_argv']
        runpy.run_path(str(entrypoint), run_name='__main__')
    except BaseException as exc:
        status, error = 'original_entrypoint_raised', repr(exc)
        raise
    finally:
        sys.argv = previous_argv
        training_attestation.capture_training_runtime = original
        write_exclusive(output / 'entrypoint_exit.json', dict(
            utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            status=status, error=error, observed_runtime_calls=observer.calls,
            training_updates_inferred=False, resume_admission_inferred=False,
            formal_resume_validation_bypassed=False))


if __name__ == '__main__':
    main()
