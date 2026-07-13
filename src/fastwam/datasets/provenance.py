"""Dataset provenance helpers used by WARM leakage controls.

The same episode number can occur in more than one dataset inside a
``MultiLeRobotDataset``.  Consequently, an episode is identified by the pair
``(dataset_index, episode_index)`` inside a configured dataset collection.
"""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping


PROVENANCE_KEYS: tuple[str, ...] = (
    "dataset_index",
    "episode_index",
    "frame_index",
    "timestamp",
    "task_index",
)


def copy_provenance(
    source: Mapping[str, Any],
    destination: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    """Copy available provenance fields without changing their tensor types."""

    for key in PROVENANCE_KEYS:
        if key in source:
            destination[key] = source[key]
    return destination


def _scalar_int(value: Any, *, field: str) -> int:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool):
        raise TypeError(f"{field} must be an integer scalar, got bool")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field} must be an integer scalar") from exc
    if result < 0:
        raise ValueError(f"{field} must be non-negative, got {result}")
    return result


def global_episode_id(sample: Mapping[str, Any]) -> tuple[int, int]:
    """Return the collision-free episode identity for a processed sample."""

    missing = [key for key in ("dataset_index", "episode_index") if key not in sample]
    if missing:
        raise KeyError(f"Missing episode provenance fields: {', '.join(missing)}")
    return (
        _scalar_int(sample["dataset_index"], field="dataset_index"),
        _scalar_int(sample["episode_index"], field="episode_index"),
    )
