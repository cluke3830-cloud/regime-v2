"""HMM kernels — Numba-JIT optimised forward filter.

Brief 5.4 of the regime upgrade plan. The audit (§5.5.2, §8.5.3)
named the legacy dashboard's ``_hmm_forward_probs`` as the best math
kernel in the file (score 95/100) and prescribed a Numba port for
production deployment.
"""

from src.hmm.forward_filter_optimised import (
    forward_filter_log_space,
    forward_filter_naive,
)

__all__ = ["forward_filter_log_space", "forward_filter_naive"]
