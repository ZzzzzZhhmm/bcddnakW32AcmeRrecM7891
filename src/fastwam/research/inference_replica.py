"""Inference-only module replicas sharing parameter storage and separate state.

The replicas must run serially under inference_mode. Their weights and buffers
are shared read-only; mutable module attributes, hooks and diagnostic caches are
deep-copied. This is an execution optimization, not a new null-policy definition.
"""
from __future__ import annotations

from copy import deepcopy


def shared_weight_inference_replica(model):
    """Clone a cold eval model without allocating another copy of its weights.

    Reject an already used online model so that no consumed capability, factual
    history or inference lock needs to be reset or bypassed. The replica gets its
    own legitimate research controls before first inference.
    """
    if model.training or any(module.training for module in model.modules()):
        raise ValueError('an eval-only model is required')
    if getattr(model,'_warm_online_experiment_locked',False):
        raise ValueError('create the replica before first online inference')
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise ValueError('training gradients cannot be shared')
    parameters=dict(model.named_parameters(remove_duplicate=False))
    buffers=dict(model.named_buffers(remove_duplicate=False))
    if not parameters:raise ValueError('a parameterized model is required')
    memo={id(value):value for value in (*parameters.values(),*buffers.values())}
    # A deployed model owns a retriever capability containing immutable bank
    # views and mutable issued/consumed cursors. A diagnostic clone must own
    # none of it. It receives already validated factual tensors through the
    # production FastWAM kernel; its consuming online API remains unbound.
    retriever=getattr(model,'_warm_online_retriever',None)
    if retriever is not None:memo[id(retriever)]=None
    replica=deepcopy(model,memo)
    if replica is model:raise RuntimeError('model was not independently cloned')
    if getattr(replica,'_warm_online_retriever',None) is not None:
        raise RuntimeError('diagnostic replica retained an online capability')
    original_modules=dict(model.named_modules(remove_duplicate=False))
    copied_modules=dict(replica.named_modules(remove_duplicate=False))
    if set(original_modules)!=set(copied_modules):raise RuntimeError('module topology differs')
    if any(original_modules[name] is copied_modules[name] for name in original_modules):
        raise RuntimeError('mutable module object was shared')
    for collection,original in ((replica.named_parameters,parameters),(replica.named_buffers,buffers)):
        copied=dict(collection(remove_duplicate=False))
        if set(copied)!=set(original) or any(copied[name] is not value for name,value in original.items()):
            raise RuntimeError('parameter or buffer identity differs')
    # _version changes on ordinary in-place tensor writes. Validate this receipt
    # around every diagnostic generation, in addition to factual policy state.
    receipt=tuple((name,value,int(value._version)) for name,value in
        [('parameter:'+name,value) for name,value in parameters.items()]+
        [('buffer:'+name,value) for name,value in buffers.items()])
    return replica,receipt


def assert_shared_weights_unchanged(receipt):
    for name,value,version in receipt:
        if int(value._version)!=version:raise RuntimeError('inference wrote shared weights/buffers: '+name)
