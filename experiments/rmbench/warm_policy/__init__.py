"""Official RMBench deployment adapter for the full WARM policy.

RMBench imports the configured policy *package* rather than its deploy module,
so these re-exports are the public evaluator interface.
"""

from .deploy_policy import eval, get_model, reset_model

__all__ = ["eval", "get_model", "reset_model"]
