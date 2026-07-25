"""Import guard for the pinned RGB-only RMBench evaluation protocol.

The official RMBench checkout imports :mod:`open3d` unconditionally from its
camera and file-saving modules, even when both depth and point-cloud outputs
are disabled.  Downloading the 400 MB Open3D wheel only to satisfy those two
imports makes ephemeral ACP setup unnecessarily slow.

The WARM runtime preflight permits this guard only for the pinned checkout and
verifies that depth, pointcloud, and segmentation are disabled in
``task_config/demo_clean.yml``.  Any accidental Open3D use therefore fails
immediately instead of silently changing evaluation behavior.
"""

from __future__ import annotations


__warm_rgb_only_shim__ = True
__version__ = "0.18.0-warm-rgb-only-guard"
__all__: tuple[str, ...] = ()


def __getattr__(name: str) -> object:
    raise RuntimeError(
        "Open3D was accessed during WARM's pinned RGB-only RMBench evaluation "
        f"(attribute={name!r}). The formal protocol must keep depth, "
        "pointcloud, mesh_segmentation, and actor_segmentation disabled. "
        "Install the real open3d==0.18.0 runtime before enabling those outputs."
    )
