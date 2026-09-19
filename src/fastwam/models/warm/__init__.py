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
from .online_contract import (
    ONLINE_RETRIEVAL_IMPLEMENTATION,
    ONLINE_RUN_SCHEMA,
    ONLINE_RUN_SCHEMA_VERSION,
    OnlineRunContractError,
    WarmOnlineRunContract,
)
from .online_pair_contract import (
    ALLOWED_CONFIG_DIFFERENCE_PATHS,
    ONLINE_PAIR_KIND,
    ONLINE_PAIR_SCHEMA,
    ONLINE_PAIR_SCHEMA_VERSION,
    OnlinePairContractError,
    WarmOnlinePairContract,
)
from .training_attestation import (
    SHARED_RECIPE_IGNORED_PATHS,
    V1_SHARED_RECIPE_IGNORED_PATHS,
    TRAINING_ATTESTATION_SCHEMA,
    TRAINING_ATTESTATION_VERSION,
    TrainingAttestationError,
    WarmTrainingAttestation,
    WarmTrainingRunContext,
    capture_actual_optimizer_facts,
    capture_actual_scheduler_chain,
    capture_training_runtime,
    clean_git_commit,
    load_training_attestation,
    publish_training_attestation,
    prepare_formal_resume_lineage,
    sha256_training_state_tree,
    training_attestation_path,
    training_config_hashes,
    verify_training_attestation,
)
from .retrospection_config import (
    WarmRetrospectionConfig,
    WarmRetrospectionConfigError,
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
    "ONLINE_RETRIEVAL_IMPLEMENTATION",
    "ONLINE_RUN_SCHEMA",
    "ONLINE_RUN_SCHEMA_VERSION",
    "OnlineRunContractError",
    "WarmOnlineRunContract",
    "ALLOWED_CONFIG_DIFFERENCE_PATHS",
    "ONLINE_PAIR_KIND",
    "ONLINE_PAIR_SCHEMA",
    "ONLINE_PAIR_SCHEMA_VERSION",
    "OnlinePairContractError",
    "WarmOnlinePairContract",
    "SHARED_RECIPE_IGNORED_PATHS",
    "V1_SHARED_RECIPE_IGNORED_PATHS",
    "TRAINING_ATTESTATION_SCHEMA",
    "TRAINING_ATTESTATION_VERSION",
    "TrainingAttestationError",
    "WarmTrainingAttestation",
    "WarmTrainingRunContext",
    "capture_actual_optimizer_facts",
    "capture_actual_scheduler_chain",
    "capture_training_runtime",
    "clean_git_commit",
    "load_training_attestation",
    "publish_training_attestation",
    "prepare_formal_resume_lineage",
    "sha256_training_state_tree",
    "training_attestation_path",
    "training_config_hashes",
    "verify_training_attestation",
    "WarmRetrospectionConfig",
    "WarmRetrospectionConfigError",
]

# The Windows source-development environment intentionally need not install the
# multi-gigabyte CUDA/PyTorch stack.  Keep the NumPy reference contract
# importable there, while exposing the Torch runtime whenever torch is present.
# Retrospection is imported lazily: FastWAM imports source_transport during
# module import, and an eager retrospection_model import would cycle back.
_RETROSPECTION_EXPORTS = {
    "OnlineExperimentControls",
    "RetrospectiveSourceContext",
    "WARM_ONLINE_ABLATION_MODES",
    "WARM_ONLINE_MEMORY_CORRUPTIONS",
    "WarmRetrospectionError",
    "WarmRetrospectionFastWAM",
}
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
            *_RETROSPECTION_EXPORTS,
        ]
    )


def __getattr__(name: str):
    if name in _RETROSPECTION_EXPORTS:
        from . import retrospection_model

        return getattr(retrospection_model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
