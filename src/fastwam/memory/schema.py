"""Stable identities and schema constants for WARM event banks.

The event id deliberately contains the dataset namespace as well as the
dataset/episode indices.  An episode exclusion therefore cannot accidentally
mix two concatenated datasets which happen to reuse a local episode index.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Mapping, TypeAlias


EVENT_BANK_SCHEMA = "warm.event-bank"
EVENT_BANK_SCHEMA_VERSION = 2

EpisodeKey: TypeAlias = tuple[str, int, int]


def _non_negative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field} must be a non-negative integer, got {type(value).__name__}")
    result = int(value)
    if result < 0:
        raise ValueError(f"{field} must be non-negative, got {result}")
    return result


@dataclass(frozen=True, order=True, slots=True)
class EventId:
    """Globally stable identity of a fixed-horizon event chunk.

    ``start_frame`` identifies an event within an episode, while
    :meth:`episode_key` intentionally omits it so callers can exclude the
    complete source episode during retrieval.
    """

    dataset_id: str
    dataset_index: int
    episode_index: int
    start_frame: int

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_id, str):
            raise TypeError("dataset_id must be a string")
        if not self.dataset_id or not self.dataset_id.strip():
            raise ValueError("dataset_id must not be empty")
        if self.dataset_id != self.dataset_id.strip():
            raise ValueError("dataset_id must not contain leading or trailing whitespace")
        if "\x00" in self.dataset_id:
            raise ValueError("dataset_id must not contain NUL characters")

        object.__setattr__(
            self, "dataset_index", _non_negative_int(self.dataset_index, "dataset_index")
        )
        object.__setattr__(
            self, "episode_index", _non_negative_int(self.episode_index, "episode_index")
        )
        object.__setattr__(self, "start_frame", _non_negative_int(self.start_frame, "start_frame"))

    @property
    def episode_key(self) -> EpisodeKey:
        """Return the identity used for leave-entire-episode-out retrieval."""

        return (self.dataset_id, self.dataset_index, self.episode_index)

    def to_dict(self) -> dict[str, str | int]:
        return {
            "dataset_id": self.dataset_id,
            "dataset_index": self.dataset_index,
            "episode_index": self.episode_index,
            "start_frame": self.start_frame,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EventId":
        if not isinstance(value, Mapping):
            raise TypeError("event id must be a mapping")
        expected = {"dataset_id", "dataset_index", "episode_index", "start_frame"}
        actual = set(value)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(f"invalid event id fields; missing={missing}, extra={extra}")
        return cls(
            dataset_id=value["dataset_id"],
            dataset_index=value["dataset_index"],
            episode_index=value["episode_index"],
            start_frame=value["start_frame"],
        )


def coerce_episode_key(value: EventId | EpisodeKey) -> EpisodeKey:
    """Validate and normalize an episode-exclusion key."""

    if isinstance(value, EventId):
        return value.episode_key
    if not isinstance(value, tuple) or len(value) != 3:
        raise TypeError("episode key must be EventId or (dataset_id, dataset_index, episode_index)")
    dataset_id, dataset_index, episode_index = value
    # Reuse EventId's strict validation without inventing a second identity contract.
    return EventId(dataset_id, dataset_index, episode_index, 0).episode_key
