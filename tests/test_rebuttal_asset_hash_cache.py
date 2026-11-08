"""Digest reuse never substitutes a trusted string for the first byte hash."""
import hashlib
import os
import sys
import time
import pytest
from fastwam.memory.manifest import sha256_file
from fastwam.memory.asset_hash_cache import cache_statistics


def test_default_hash_path_still_reads_changed_bytes(tmp_path,monkeypatch):
    monkeypatch.delenv('WARM_IMMUTABLE_ASSET_HASH_CACHE',raising=False)
    path=tmp_path/'asset';path.write_bytes(b'old')
    assert sha256_file(path)==hashlib.sha256(b'old').hexdigest()
    path.write_bytes(b'new')
    assert sha256_file(path)==hashlib.sha256(b'new').hexdigest()


@pytest.mark.skipif(sys.platform!='linux',reason='Metadata cache is Linux-only')
def test_mutation_with_restored_size_and_mtime_invalidates_reuse(tmp_path,monkeypatch):
    monkeypatch.setenv('WARM_IMMUTABLE_ASSET_HASH_CACHE','1')
    path=tmp_path/'asset';path.write_bytes(b'a'*(1024*1024))
    before=cache_statistics();original=sha256_file(path);metadata=path.stat()
    assert sha256_file(path)==original and cache_statistics()['hits']==before['hits']+1
    time.sleep(.002);path.write_bytes(b'b'*(1024*1024));os.utime(path,ns=(metadata.st_atime_ns,metadata.st_mtime_ns))
    changed=sha256_file(path)
    assert changed==hashlib.sha256(b'b'*(1024*1024)).hexdigest() and changed!=original
    assert cache_statistics()['misses']==before['misses']+2


@pytest.mark.skipif(sys.platform!='linux',reason='Metadata cache is Linux-only')
def test_atomic_replacement_with_preserved_times_is_rehashed(tmp_path,monkeypatch):
    monkeypatch.setenv('WARM_IMMUTABLE_ASSET_HASH_CACHE','1')
    path=tmp_path/'asset';path.write_bytes(b'a'*(1024*1024));old=path.stat();sha256_file(path)
    replacement=tmp_path/'replacement';replacement.write_bytes(b'c'*(1024*1024))
    os.utime(replacement,ns=(old.st_atime_ns,old.st_mtime_ns));replacement.replace(path)
    assert sha256_file(path)==hashlib.sha256(b'c'*(1024*1024)).hexdigest()
