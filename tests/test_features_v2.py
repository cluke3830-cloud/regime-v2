"""Tests for compute_features_v2 + aux_data (Brief 2.1.2).

Critical tests:
  - Causal hygiene on v2 (perturb future aux data → features at earlier
    indices unchanged).
  - All 23 columns (v1 + 9 Tier-2) present when all aux data provided.
  - Graceful degradation when aux series are None — feature column
    becomes constant 0, frame survives.
  - Aux-data alignment: VIX/FRED series with a different (sparser)
    calendar reindex + ffill correctly.
  - All hermetic — no network calls.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.price_features import (  # noqa: E402
    FEATURE_COLUMNS_V1,
    FEATURE_COLUMNS_V2,
    FEATURE_COLUMNS_V2_ADD,
    NON_FEATURE_COLUMNS,
    compute_features_v1,
    compute_features_v2,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _gbm(
    n: int = 600, drift: float = 0.0003, vol: float = 0.012, seed: int = 0
) -> pd.Series:
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n)
    log_ret = drift - 0.5 * vol ** 2 + vol * eps
    return pd.Series(
        np.exp(np.log(100.0) + np.cumsum(log_ret)),
        index=pd.date_range("2018-01-01", periods=n, freq="D"),
        name="close",
    )


def _make_aux_bundle(close_index: pd.DatetimeIndex, seed: int = 1) -> dict:
    """Synthetic aux data aligned to ``close_index``."""
    rng = np.random.default_rng(seed)
    n = len(close_index)
    bundle: dict = {}
    # VIX-like: mean-reverting positive series, fat tails
    bundle["vix"] = pd.Series(
        np.exp(np.log(18) + 0.1 * np.cumsum(rng.standard_normal(n) * 0.1)),
        index=close_index, name="vix",
    )
    bundle["vix3m"] = pd.Series(
        np.exp(np.log(19) + 0.05 * np.cumsum(rng.standard_normal(n) * 0.08)),
        index=close_index, name="vix3m",
    )
    bundle["tlt"] = _gbm(n=n, drift=-0.0001, vol=0.008, seed=seed + 1)
    bundle["tlt"].index = close_index
    bundle["gld"] = _gbm(n=n, drift=0.0002, vol=0.009, seed=seed + 2)
    bundle["gld"].index = close_index
    bundle["term_spread"] = pd.Series(
        np.cumsum(rng.standard_normal(n) * 0.02),
        index=close_index, name="term_spread",
    )
    bundle["credit_spread"] = pd.Series(
        2.0 + np.cumsum(rng.standard_normal(n) * 0.01),
        index=close_index, name="credit_spread",
    )
    return bundle


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_v2_full_bundle_has_all_23_features():
    """v2 schema: 14 Tier-1 + 9 Tier-2 (5 VIX/macro + 2 cross-asset corrs +
    yang_zhang_vol + vix_slope) = 23.
    """
    close = _gbm(n=600)
    aux = _make_aux_bundle(close.index)
    f = compute_features_v2(close, **aux)
    for col in NON_FEATURE_COLUMNS + FEATURE_COLUMNS_V2:
        assert col in f.columns, f"missing column: {col}"
    assert len(FEATURE_COLUMNS_V2) == 23


def test_v2_v1_subset_matches_v1():
    """The v1 sub-columns of v2 must equal compute_features_v1 exactly.

    Verifies we didn't accidentally change the v1 semantics while
    rebuilding them inside v2.
    """
    close = _gbm(n=800, seed=4)
    aux = _make_aux_bundle(close.index)
    v1 = compute_features_v1(close)
    v2 = compute_features_v2(close, **aux)
    common = v1.index.intersection(v2.index)
    for col in FEATURE_COLUMNS_V1:
        np.testing.assert_allclose(
            v1.loc[common, col].to_numpy(),
            v2.loc[common, col].to_numpy(),
            rtol=1e-9, err_msg=f"{col} diverged between v1 and v2",
        )


# ---------------------------------------------------------------------------
# Causal hygiene
# ---------------------------------------------------------------------------


def test_v2_causal_no_lookahead_on_close():
    """Same perturb-future test as v1, applied to the close series.

    Aux data here is constant across the run, so any change in features
    at an earlier index can only come from a look-ahead in the close
    handling.
    """
    n = 500
    close = _gbm(n=n, seed=88)
    aux = _make_aux_bundle(close.index)
    baseline = compute_features_v2(close, **aux)
    if len(baseline) < 20:
        pytest.skip("not enough surviving rows")

    rng = np.random.default_rng(0)
    sample = rng.choice(baseline.index[:-50], size=10, replace=False)
    for ts in sorted(sample):
        pos = close.index.get_loc(ts)
        mutated = close.copy()
        mutated.iloc[pos + 1:] = close.iloc[pos] * rng.uniform(
            0.5, 1.5, size=n - pos - 1
        )
        recomputed = compute_features_v2(mutated, **aux)
        for col in FEATURE_COLUMNS_V2:
            assert np.isclose(
                baseline.loc[ts, col], recomputed.loc[ts, col],
                rtol=1e-9, equal_nan=True,
            ), f"feature `{col}` at {ts} changed under future-close mutation"


def test_v2_causal_no_lookahead_on_aux():
    """Mutate future values of VIX/TLT/GLD/FRED, recompute, earlier
    rows must be identical.
    """
    n = 400
    close = _gbm(n=n, seed=21)
    aux = _make_aux_bundle(close.index)
    baseline = compute_features_v2(close, **aux)

    rng = np.random.default_rng(0)
    sample = rng.choice(baseline.index[:-50], size=5, replace=False)
    for ts in sorted(sample):
        pos = close.index.get_loc(ts)
        mutated_aux = {
            k: (v.copy() if isinstance(v, pd.Series) else v)
            for k, v in aux.items()
        }
        for k, series in mutated_aux.items():
            if isinstance(series, pd.Series):
                series.iloc[pos + 1:] = rng.standard_normal(n - pos - 1)
        recomputed = compute_features_v2(close, **mutated_aux)
        for col in FEATURE_COLUMNS_V2:
            assert np.isclose(
                baseline.loc[ts, col], recomputed.loc[ts, col],
                rtol=1e-9, equal_nan=True,
            ), f"feature `{col}` at {ts} changed under future-aux mutation"


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------


def test_v2_missing_aux_columns_become_zero():
    """If every aux series is None, the seven Tier-2 columns must NOT
    nuke the frame — they degrade to 0.0 and the v1 features still
    populate. (XGBoost treats a constant column as zero information.)
    """
    close = _gbm(n=600)
    f = compute_features_v2(close)  # all aux None
    assert len(f) > 0
    for col in FEATURE_COLUMNS_V2_ADD:
        assert (f[col] == 0.0).all(), (
            f"{col} expected all-zero on missing aux, got {f[col].head()}"
        )
    # v1 columns must still have real, non-constant values
    assert f["vol_ewma"].std() > 0
    assert f["mom_20"].std() > 0


def test_v2_partial_aux_only_macro_missing():
    """VIX + cross-asset present, FRED missing → vix/corr columns are
    real, term/credit spreads are zero.
    """
    close = _gbm(n=500)
    aux = _make_aux_bundle(close.index)
    aux["term_spread"] = None
    aux["credit_spread"] = None
    f = compute_features_v2(close, **aux)
    assert (f["term_spread"] == 0.0).all()
    assert (f["credit_spread"] == 0.0).all()
    # VIX and corr columns should have non-trivial variation
    assert f["vix_log"].std() > 0
    assert f["corr_tlt_63"].std() > 0


def test_v2_aux_with_different_calendar_is_aligned():
    """Pass an aux series with a sparser/weekly index — the function must
    forward-fill it onto the close calendar without raising.
    """
    close = _gbm(n=400)
    aux = _make_aux_bundle(close.index)
    # Pretend FRED gives us weekly data on every 5th day
    fred_idx = close.index[::5]
    aux["term_spread"] = aux["term_spread"].loc[fred_idx]
    f = compute_features_v2(close, **aux)
    # Should fully populate term_spread (no all-NaN → no zero-fill kick-in)
    assert f["term_spread"].std() > 0


def test_v2_no_nans_in_output():
    """As with v1, the returned frame must be NaN-free."""
    close = _gbm(n=800)
    aux = _make_aux_bundle(close.index)
    f = compute_features_v2(close, **aux)
    bad = f.isna().sum()
    nonzero = bad[bad > 0]
    assert nonzero.empty, f"NaN found in: {nonzero.to_dict()}"


# ---------------------------------------------------------------------------
# aux_data fetcher input validation (no network — mock-style)
# ---------------------------------------------------------------------------


def test_fetch_fred_series_missing_key_raises(monkeypatch):
    from src.features import aux_data
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="API key"):
        aux_data.fetch_fred_series("T10Y2Y", "2020-01-01", "2020-02-01")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))