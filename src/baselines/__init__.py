"""Phase 3 econometric baselines — TVTP-MSAR, HSMM, MS-GARCH.

These three classical econometric regime models are the "tier-1 baselines"
the audit (§4.2) names as the literature standard that every regime
classifier must beat. Each becomes a strategy_fn variant for the CPCV
harness, increasing ensemble diversity and reducing PBO toward < 50%.
"""

from src.baselines.hsmm import (
    DEFAULT_K3_POSITIONS,
    DurationAwareHMM,
    make_hsmm_strategy,
)
from src.baselines.ms_garch import (
    GARCHVolatilityModel,
    evaluate_forecast_rmse_vs_rolling,
    make_ms_garch_strategy,
)
from src.baselines.tvtp_msar import (
    MarkovSwitchingAR,
    make_tvtp_msar_strategy,
)

__all__ = [
    "MarkovSwitchingAR",
    "make_tvtp_msar_strategy",
    "DurationAwareHMM",
    "DEFAULT_K3_POSITIONS",
    "make_hsmm_strategy",
    "GARCHVolatilityModel",
    "evaluate_forecast_rmse_vs_rolling",
    "make_ms_garch_strategy",
]
