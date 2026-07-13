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
        "WARM_QUERY_SPLIT",
    }
)

_WARM_RETROSPECTIVE_EXPORTS = frozenset(
    {
        "RetrospectiveFeatureStore",
        "RetrospectiveFeatureStoreError",
        "RuntimeRetrospectiveDatasetAdapter",
        "WARM_CANDIDATE_CONTEXT",
        "WARM_CANDIDATE_EFFECT_DELTA",
        "WARM_CANDIDATE_EFFECT_POST",
        "WARM_CANDIDATE_EFFECT_PRE",
        "WARM_CANDIDATE_GRIPPER",
        "WARM_CANDIDATE_START_PROPRIO",
        "WARM_CANDIDATE_SUPPORT",
        "WARM_CANDIDATE_TIMING",
        "WARM_CURRENT_CONTEXT",
        "WARM_CURRENT_SEMANTIC",
        "WARM_EPISODE_ACTION_MASK",
        "WARM_EPISODE_ACTION_SUMMARIES",
        "WARM_EPISODE_MASK",
        "WARM_EPISODE_TOKENS",
        "WARM_FUTURE_SEMANTIC",
        "WARM_FUTURE_VALID",
        "WARM_RETROSPECTIVE_FIELDS",
        "WARM_TARGET_EFFECT",
        "collect_feature_payloads",
        "collect_feature_payloads_from_list",
    }
)


def __getattr__(name: str):
    if name in _WARM_EXPORTS:
        from . import warm_candidates

        return getattr(warm_candidates, name)
    if name in _WARM_RETROSPECTIVE_EXPORTS:
        from . import warm_retrospective

        return getattr(warm_retrospective, name)
    raise AttributeError(name)


__all__ = sorted(_WARM_EXPORTS | _WARM_RETROSPECTIVE_EXPORTS)
