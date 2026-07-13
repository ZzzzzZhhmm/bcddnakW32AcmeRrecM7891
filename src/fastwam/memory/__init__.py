"""NumPy reference infrastructure for WARM retrospective event memory."""

from .event_bank import (
    MANIFEST_FILENAME,
    PAYLOAD_FILENAME,
    EventBank,
    EventBankError,
    IntegrityError,
    SearchResult,
)
from .bank_builder import EpisodeFeatures, build_event_bank
from .manifest import ArraySpec, EventBankManifest, ManifestError
from .event_mining import (
    EventMiningConfig,
    EventMiningResult,
    mine_episode_events,
    mine_event_candidates,
    select_window_starts,
)
from .schema import (
    EVENT_BANK_SCHEMA,
    EVENT_BANK_SCHEMA_VERSION,
    EpisodeKey,
    EventId,
)

__all__ = [
    "ArraySpec",
    "EVENT_BANK_SCHEMA",
    "EVENT_BANK_SCHEMA_VERSION",
    "EpisodeKey",
    "EventBank",
    "EventBankError",
    "EventBankManifest",
    "EventId",
    "EpisodeFeatures",
    "EventMiningConfig",
    "EventMiningResult",
    "IntegrityError",
    "MANIFEST_FILENAME",
    "ManifestError",
    "PAYLOAD_FILENAME",
    "SearchResult",
    "build_event_bank",
    "mine_episode_events",
    "mine_event_candidates",
    "select_window_starts",
]
