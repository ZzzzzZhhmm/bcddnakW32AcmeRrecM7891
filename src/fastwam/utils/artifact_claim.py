"""Cross-process exclusive claims for publishing immutable artifacts.

The claim is deliberately implemented as a small lock file rather than an
in-process mutex.  Creating the file with ``O_CREAT | O_EXCL`` is the atomic
operation that elects a single writer across independent processes.

Claims are not automatically treated as stale.  If a process crashes, an
operator must inspect and remove the lock explicitly; silently stealing a
claim could allow two writers to publish the same artifact.
"""

from __future__ import annotations

import json
import os
import socket
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any
from uuid import uuid4


_CLAIM_SCHEMA = "warm.artifact-claim"
_CLAIM_VERSION = 1
_MAX_EXISTING_CLAIM_BYTES = 64 * 1024


class ArtifactClaimError(RuntimeError):
    """Base error for artifact-claim acquisition and release failures."""


class ArtifactAlreadyClaimedError(ArtifactClaimError):
    """Raised when another process already owns the requested claim."""


class ArtifactClaimOwnershipError(ArtifactClaimError):
    """Raised when a claim can no longer be proven to belong to its owner."""


@dataclass(frozen=True)
class ArtifactClaimRecord:
    """Human-readable metadata persisted in an artifact claim file."""

    schema: str
    version: int
    token: str
    pid: int
    purpose: str
    created_at_utc: str
    hostname: str


def _claim_flags() -> int:
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    flags |= getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    return flags


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        if written <= 0:  # pragma: no cover - defensive guard around os.write
            raise OSError("os.write made no progress while writing artifact claim")
        view = view[written:]


def _read_claim_mapping(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        raw = handle.read(_MAX_EXISTING_CLAIM_BYTES + 1)
    if len(raw) > _MAX_EXISTING_CLAIM_BYTES:
        raise ValueError("claim file exceeds the 64 KiB safety limit")
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("claim file does not contain a JSON object")
    return value


def _existing_claim_description(path: Path) -> str:
    try:
        value = _read_claim_mapping(path)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        return f"metadata could not be read ({type(error).__name__}: {error})"

    pid = value.get("pid", "unknown")
    purpose = value.get("purpose", "unknown")
    created = value.get("created_at_utc", "unknown")
    hostname = value.get("hostname", "unknown")
    return (
        f"pid={pid!r}, purpose={purpose!r}, hostname={hostname!r}, "
        f"created_at_utc={created!r}"
    )


class ArtifactClaim:
    """Context manager holding one exclusive artifact-publishing claim.

    Construct this class through :func:`artifact_claim`.  The lock file is
    acquired during construction so callers cannot accidentally create an
    unacquired claim object.
    """

    def __init__(self, path: Path, record: ArtifactClaimRecord) -> None:
        self.path = path
        self.record = record
        self._released = False

    def __enter__(self) -> ArtifactClaim:
        return self

    def _release(self) -> None:
        if self._released:
            return

        try:
            value = _read_claim_mapping(self.path)
        except FileNotFoundError:
            self._released = True
            return
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
            raise ArtifactClaimOwnershipError(
                f"Refusing to remove artifact claim {self.path}: its ownership "
                f"metadata cannot be verified ({type(error).__name__}: {error})."
            ) from error

        if value.get("token") != self.record.token:
            raise ArtifactClaimOwnershipError(
                f"Refusing to remove artifact claim {self.path}: the ownership "
                "token changed while the claim was held. The current lock was "
                "left in place."
            )

        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        self._released = True

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del exc_type, traceback
        try:
            self._release()
        except ArtifactClaimError as release_error:
            if exc_value is not None:
                if hasattr(exc_value, "add_note"):
                    exc_value.add_note(f"Artifact claim cleanup also failed: {release_error}")
                return False
            raise
        return False


def artifact_claim(
    lock_path: str | os.PathLike[str],
    *,
    purpose: str,
) -> ArtifactClaim:
    """Acquire an exclusive cross-process claim and return its context manager.

    Args:
        lock_path: Dedicated lock-file path.  Its parent directory must already
            exist; this helper never creates artifact directories implicitly.
        purpose: Short human-readable description of the publication operation.

    Raises:
        ArtifactAlreadyClaimedError: If the claim path already exists.
        ArtifactClaimError: If the claim cannot be created or durably written.
        ValueError: If ``purpose`` is empty.
    """

    if not isinstance(purpose, str) or not purpose.strip():
        raise ValueError("artifact claim purpose must be a non-empty string")

    path = Path(lock_path).expanduser().absolute()
    if not path.parent.is_dir():
        raise ArtifactClaimError(
            f"Cannot create artifact claim {path}: parent directory does not exist."
        )

    record = ArtifactClaimRecord(
        schema=_CLAIM_SCHEMA,
        version=_CLAIM_VERSION,
        token=uuid4().hex,
        pid=os.getpid(),
        purpose=purpose.strip(),
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        hostname=socket.gethostname(),
    )
    encoded = (
        json.dumps(asdict(record), sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")

    try:
        fd = os.open(path, _claim_flags(), 0o600)
    except FileExistsError as error:
        description = _existing_claim_description(path)
        raise ArtifactAlreadyClaimedError(
            f"Artifact claim already exists at {path}. Another writer may be "
            f"publishing this artifact; existing claim: {description}. Inspect "
            "and remove a stale claim only after verifying that its owner is no "
            "longer running."
        ) from error
    except OSError as error:
        raise ArtifactClaimError(f"Failed to create artifact claim {path}: {error}") from error

    try:
        _write_all(fd, encoded)
        os.fsync(fd)
    except OSError as error:
        try:
            os.close(fd)
        finally:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise ArtifactClaimError(
            f"Failed to durably write artifact claim {path}: {error}"
        ) from error
    else:
        os.close(fd)

    return ArtifactClaim(path=path, record=record)


__all__ = [
    "ArtifactAlreadyClaimedError",
    "ArtifactClaim",
    "ArtifactClaimError",
    "ArtifactClaimOwnershipError",
    "ArtifactClaimRecord",
    "artifact_claim",
]
