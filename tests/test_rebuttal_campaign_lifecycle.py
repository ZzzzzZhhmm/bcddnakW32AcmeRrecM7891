import importlib.util,json,subprocess,sys,time
from pathlib import Path
import pytest


def test_campaign_deadline_does_not_reset_with_the_next_container():
    spec=importlib.util.spec_from_file_location('campaign',Path(__file__).resolve().parents[1]/'scripts/run_rebuttal_resumable_campaign.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    identity=dict(plan_sha256='fixed',first_started_utc='2026-11-09T00:00:00+00:00')
    start=m.datetime.fromisoformat(identity['first_started_utc']).timestamp()
    assert m.campaign_deadline(identity,'fixed',1000,start+900)==start+1000
    with pytest.raises(TimeoutError):m.campaign_deadline(identity,'fixed',1000,start+1001)
    with pytest.raises(ValueError):m.campaign_deadline(identity,'changed',1000,start+900)
    with pytest.raises(ValueError):m.campaign_deadline(identity,'fixed',1000,start-61)


@pytest.mark.skipif(sys.platform=='win32',reason='native POSIX lease qualification')
def test_worker_inherits_campaign_lease_until_it_exits(tmp_path):
    import fcntl
    lock=(tmp_path/'lease').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(.35)'],pass_fds=(lock.fileno(),))
    lock.close()
    other=(tmp_path/'lease').open('a')
    with pytest.raises(BlockingIOError):fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)
    child.wait(timeout=3);fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB);other.close()


def test_verified_role_archive_is_complete_and_cannot_be_replaced(tmp_path):
    from fastwam.research.resumable_pairs import archive_verified_role
    worker=tmp_path/'worker';worker.mkdir();(worker/'raw.json').write_text('{"success":0}')
    receipt=archive_verified_role(worker)
    assert receipt['files']==1 and archive_verified_role(worker)==receipt
    (tmp_path/'verified_worker_evidence.tar.gz').write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='differs'):archive_verified_role(worker)


def test_archive_receipt_can_be_recovered_after_atomic_archive_commit(tmp_path):
    from fastwam.research.resumable_pairs import archive_verified_role
    worker=tmp_path/'worker';worker.mkdir();(worker/'raw.json').write_text('{"success":0}')
    receipt=archive_verified_role(worker);(tmp_path/'archive_receipt.json').unlink()
    assert archive_verified_role(worker)==receipt
