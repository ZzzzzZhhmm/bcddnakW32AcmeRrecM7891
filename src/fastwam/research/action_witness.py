"""Seal generated actions before consumer validation and any simulator command.

Preserves the original output and exceptions. This witness distinguishes model
range failures from normalization round-off without clipping or changing guards.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import sys
import numpy as np

from .evidence import canonical


def _write(path,data):
    with path.open('xb') as stream:
        stream.write(data);stream.flush();os.fsync(stream.fileno())


@contextmanager
def seal_generated_action(model,directory,identity):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    original=model.infer_action;had_method='infer_action' in vars(model)
    record=dict(calls=0,output_sealed=False)

    def infer(*args,**kwargs):
        record['calls']+=1
        if record['calls']!=1:raise RuntimeError('one generated action per witness required')
        output=original(*args,**kwargs);value=output['action']
        array=value.detach().float().cpu().numpy().copy() if hasattr(value,'detach') else np.array(value,copy=True)
        path=directory/'model_actions.npy'
        with path.open('xb') as stream:
            np.save(stream,array,allow_pickle=False);stream.flush();os.fsync(stream.fileno())
        step=kwargs.get('online_step');finite=np.isfinite(array);finite_values=array[finite]
        metadata=dict(identity,shape=list(array.shape),dtype=str(array.dtype),
            array_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            all_finite=bool(finite.all()),finite_values=int(finite.sum()),
            minimum=float(finite_values.min()) if finite_values.size else None,
            maximum=float(finite_values.max()) if finite_values.size else None,
            before_consumer_validation=True,before_any_execution_of_this_chunk=True,
            bound_step_sha256=getattr(step,'step_sha256',None),
            model_input_sha256=getattr(step,'model_input_sha256',None),
            candidate_payload_sha256=getattr(step,'candidate_payload_sha256',None),
            derived_seed=int(step.derived_seed) if step is not None else None)
        _write(directory/'generation.json',canonical(metadata));record['output_sealed']=True
        return output

    model.infer_action=infer
    try:
        yield record
        if record['calls']!=1 or not record['output_sealed']:raise RuntimeError('missing generated action witness')
    finally:
        error=sys.exc_info()[1]
        if had_method:model.infer_action=original
        else:del model.infer_action
        _write(directory/'consumer_status.json',canonical(dict(record,
            consumer_completed=error is None,error=None if error is None else repr(error))))
