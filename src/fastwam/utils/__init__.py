from .artifact_claim import (
    ArtifactAlreadyClaimedError,
    ArtifactClaim,
    ArtifactClaimError,
    ArtifactClaimOwnershipError,
    ArtifactClaimRecord,
    artifact_claim,
)
from .fs import ensure_dir


def __getattr__(name: str):
    """Keep optional video dependencies out of lightweight utility imports."""

    if name == "save_mp4":
        from .video_io import save_mp4

        return save_mp4
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ArtifactAlreadyClaimedError",
    "ArtifactClaim",
    "ArtifactClaimError",
    "ArtifactClaimOwnershipError",
    "ArtifactClaimRecord",
    "artifact_claim",
    "ensure_dir",
    "save_mp4",
]
