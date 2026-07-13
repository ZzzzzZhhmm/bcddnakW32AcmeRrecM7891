"""FastWAM dataset interfaces.

WARM's runtime adapter is imported lazily so the lightweight catalog/audit
tools can import ``fastwam.datasets`` without eagerly importing Torch and the
full memory package.
"""

from __future__ import annotations


_WARM_EXPORTS = frozenset(
    {
        "RuntimeCandidateDatasetAdapter",
        "RuntimeCandidateDatasetContractError",
        "RuntimeCandidateDatasetError",
        "RuntimeCandidateMissingRowError",
        "WARM_CANDIDATE_EVENT_INDEX",
        "WARM_CANDIDATE_FIELDS",
        "WARM_CANDIDATE_MASK",
        "WARM_CANDIDATE_MU",
        "WARM_CANDIDATE_SCORE",
        "WARM_ORACLE_CANDIDATE_INDEX",
    }
)


def __getattr__(name: str):
    if name in _WARM_EXPORTS:
        from . import warm_candidates

        return getattr(warm_candidates, name)
    raise AttributeError(name)


__all__ = sorted(_WARM_EXPORTS)
