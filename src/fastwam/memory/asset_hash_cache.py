"""Opt-in, process-local digest reuse for immutable large Linux assets.

Every asset is hashed in full on its first use. Subsequent reuse opens the
actual file and checks device/inode/size/mtime/ctime before and after the lookup.
No persistent cache or caller-supplied digest is trusted. Default remains off.
"""
from __future__ import annotations
from collections import OrderedDict
import hashlib
import os
from pathlib import Path
import stat
import sys
from threading import RLock

_CACHE=OrderedDict()
_LOCK=RLock()
_COUNTERS=dict(hits=0,misses=0,hashed_bytes=0)


def _after_fork():
    global _CACHE,_LOCK,_COUNTERS
    _CACHE=OrderedDict();_LOCK=RLock();_COUNTERS=dict(hits=0,misses=0,hashed_bytes=0)


if hasattr(os,'register_at_fork'):os.register_at_fork(after_in_child=_after_fork)


def enabled():
    return sys.platform=='linux' and os.environ.get('WARM_IMMUTABLE_ASSET_HASH_CACHE')=='1'


def _identity(info):
    return (info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns,info.st_mode)


def sha256_asset_file(path):
    path=Path(path)
    with path.open('rb') as handle:
        before=os.fstat(handle.fileno());identity=_identity(before)
        eligible=enabled() and stat.S_ISREG(before.st_mode) and before.st_size>=1024*1024
        key=(os.getpid(),str(path.resolve()),*identity)
        if eligible:
            with _LOCK:
                cached=_CACHE.get(key)
                if cached is not None:
                    if _identity(os.fstat(handle.fileno()))!=identity:
                        raise RuntimeError('asset metadata changed during digest lookup: '+str(path))
                    _CACHE.move_to_end(key);_COUNTERS['hits']+=1
                    return cached
        digest=hashlib.sha256();count=0
        for block in iter(lambda:handle.read(1024*1024),b''):
            digest.update(block);count+=len(block)
        if _identity(os.fstat(handle.fileno()))!=identity or count!=before.st_size:
            raise RuntimeError('asset changed while hashing: '+str(path))
    result=digest.hexdigest()
    with _LOCK:
        _COUNTERS['misses']+=1;_COUNTERS['hashed_bytes']+=count
        if eligible:
            _CACHE[key]=result;_CACHE.move_to_end(key)
            while len(_CACHE)>256:_CACHE.popitem(last=False)
    return result


def cache_statistics():
    with _LOCK:return dict(_COUNTERS,enabled=enabled(),entries=len(_CACHE),scope='Per-process metadata-guarded large-file reuse; first access always hashes complete bytes.')
