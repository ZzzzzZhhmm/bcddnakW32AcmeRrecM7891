from __future__ import annotations

import json
import multiprocessing
import os
from pathlib import Path

import pytest

from fastwam.utils.artifact_claim import (
    ArtifactAlreadyClaimedError,
    ArtifactClaimError,
    ArtifactClaimOwnershipError,
    artifact_claim,
)


def _attempt_claim_in_child(lock_path: str, result_queue: multiprocessing.Queue) -> None:
    try:
        with artifact_claim(lock_path, purpose="child publisher"):
            result_queue.put(("acquired", ""))
    except ArtifactAlreadyClaimedError as error:
        result_queue.put(("blocked", str(error)))


def test_artifact_claim_persists_metadata_and_cleans_up(tmp_path: Path) -> None:
    lock_path = tmp_path / ".bank.warm-build.lock"

    with artifact_claim(lock_path, purpose="publish event bank") as claim:
        assert claim.path == lock_path.absolute()
        assert lock_path.is_file()

        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        assert payload["schema"] == "warm.artifact-claim"
        assert payload["version"] == 1
        assert payload["token"] == claim.record.token
        assert payload["pid"] == os.getpid()
        assert payload["purpose"] == "publish event bank"
        assert payload["created_at_utc"].endswith("+00:00")
        assert payload["hostname"]

    assert not lock_path.exists()


def test_artifact_claim_cleans_up_when_body_raises(tmp_path: Path) -> None:
    lock_path = tmp_path / ".candidate.lock"

    with pytest.raises(RuntimeError, match="body failed"):
        with artifact_claim(lock_path, purpose="publish candidates"):
            raise RuntimeError("body failed")

    assert not lock_path.exists()


def test_existing_claim_reports_owner_and_is_not_modified(tmp_path: Path) -> None:
    lock_path = tmp_path / ".oracle.lock"
    existing = {
        "schema": "warm.artifact-claim",
        "version": 1,
        "token": "existing-token",
        "pid": 4321,
        "purpose": "first oracle writer",
        "created_at_utc": "2026-07-13T00:00:00+00:00",
        "hostname": "worker-a",
    }
    original = json.dumps(existing, sort_keys=True)
    lock_path.write_text(original, encoding="utf-8")

    with pytest.raises(ArtifactAlreadyClaimedError) as error_info:
        artifact_claim(lock_path, purpose="second oracle writer")

    message = str(error_info.value)
    assert str(lock_path.absolute()) in message
    assert "pid=4321" in message
    assert "first oracle writer" in message
    assert "stale claim" in message
    assert lock_path.read_text(encoding="utf-8") == original


def test_claim_blocks_a_separate_process(tmp_path: Path) -> None:
    lock_path = tmp_path / ".cross-process.lock"
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()

    with artifact_claim(lock_path, purpose="parent publisher"):
        process = context.Process(
            target=_attempt_claim_in_child,
            args=(str(lock_path), result_queue),
        )
        process.start()
        process.join(timeout=15)

        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            pytest.fail("child process did not finish while testing artifact claim")

        assert process.exitcode == 0
        status, message = result_queue.get(timeout=5)
        assert status == "blocked"
        assert "parent publisher" in message

    assert not lock_path.exists()


def test_release_refuses_to_delete_replaced_claim(tmp_path: Path) -> None:
    lock_path = tmp_path / ".ownership.lock"
    claim = artifact_claim(lock_path, purpose="original writer")

    with pytest.raises(ArtifactClaimOwnershipError, match="ownership token changed"):
        with claim:
            replacement = json.loads(lock_path.read_text(encoding="utf-8"))
            replacement["token"] = "replacement-owner"
            replacement["pid"] = 9999
            lock_path.write_text(json.dumps(replacement), encoding="utf-8")

    assert lock_path.exists()
    assert json.loads(lock_path.read_text(encoding="utf-8"))["token"] == "replacement-owner"


def test_claim_requires_nonempty_purpose_and_existing_parent(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        artifact_claim(tmp_path / ".empty.lock", purpose="  ")

    with pytest.raises(ArtifactClaimError, match="parent directory does not exist"):
        artifact_claim(tmp_path / "missing" / ".claim.lock", purpose="publish")
