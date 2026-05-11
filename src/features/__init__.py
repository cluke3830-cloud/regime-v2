"""Feature engineering — causally-clean price features for the regime models.

Phase 2.1.1: Tier-1 expansion. Replaces the 2-feature ``default_feature_fn``
with a 14-feature set covering the audit's §5.2 prescribed feature layer
that comes from price/return data alone (no external macro or cross-asset
data — those land in 2.1.2).

Causal convention enforced HERE so callers don't have to worry about peek:
every feature at row ``t`` depends only on data through row ``t-1``. The
strategy makes its decision *at the close of t-1*, holds through *the
close of t*, earning ``log_ret[t]``.
"""

from src.features.aux_data import (
    fetch_aux_data_bundle,
    fetch_fred_series,
    fetch_vix,
    fetch_vix3m,
    fetch_yf_close,
)
from src.features.price_features import (
    FEATURE_COLUMNS_V1,
    FEATURE_COLUMNS_V2,
    FEATURE_COLUMNS_V2_ADD,
    NON_FEATURE_COLUMNS,
    compute_features_v1,
    compute_features_v2,
)

__all__ = [
    "compute_features_v1",
    "compute_features_v2",
    "FEATURE_COLUMNS_V1",
    "FEATURE_COLUMNS_V2",
    "FEATURE_COLUMNS_V2_ADD",
    "NON_FEATURE_COLUMNS",
    "fetch_aux_data_bundle",
    "fetch_fred_series",
    "fetch_vix",
    "fetch_vix3m",
    "fetch_yf_close",
]