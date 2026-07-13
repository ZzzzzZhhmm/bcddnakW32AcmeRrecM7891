"""WARM mathematical reference components."""

from .candidate_selection import (
    NULL_COMPONENT,
    CandidateMixture,
    EffectScoreBreakdown,
    MemorySource,
    build_candidate_mixture,
    sample_component,
    sample_memory_source,
    sample_null_gaussian_source,
    sample_source_for_component,
    sample_source_from_mixture,
    score_observed_effects,
    softmax_probabilities,
)
from .source_contract import (
    SOURCE_RUN_SCHEMA,
    SOURCE_RUN_SCHEMA_VERSION,
    SourceRunContractError,
    WarmSourceRunContract,
)
__all__ = [
    "NULL_COMPONENT",
    "CandidateMixture",
    "EffectScoreBreakdown",
    "MemorySource",
    "build_candidate_mixture",
    "sample_component",
    "sample_memory_source",
    "sample_null_gaussian_source",
    "sample_source_for_component",
    "sample_source_from_mixture",
    "score_observed_effects",
    "softmax_probabilities",
    "SOURCE_RUN_SCHEMA",
    "SOURCE_RUN_SCHEMA_VERSION",
    "SourceRunContractError",
    "WarmSourceRunContract",
]

# The Windows source-development environment intentionally need not install the
# multi-gigabyte CUDA/PyTorch stack.  Keep the NumPy reference contract
# importable there, while exposing the Torch runtime whenever torch is present.
try:
    from .source_transport import (
        ActionFlowPair,
        ActionSourceContext,
        ActionSourceOutput,
        SourceGeometry,
        SourceTransportError,
        build_action_flow_pair,
        measure_source_geometry,
        resolve_action_source,
        select_source_components,
    )
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
else:
    __all__.extend(
        [
            "ActionFlowPair",
            "ActionSourceContext",
            "ActionSourceOutput",
            "SourceGeometry",
            "SourceTransportError",
            "build_action_flow_pair",
            "measure_source_geometry",
            "resolve_action_source",
            "select_source_components",
        ]
    )
