"""Tests for src.features.price_features.compute_features_v1 (Brief 2.1.1).

Critical tests:
  - Causal hygiene: perturb future close values, recompute features at an
    earlier index, must be identical. THIS IS THE BLOCKING TEST — if it
    fails, every downstream metric is contaminated.
  - All FEATURE_COLUMNS_V1 present in output.
  - No NaNs in returned frame.
  - Sanity: features have reasonable ranges (vol > 0, momenta finite,
    drawdown ≤ 0).
  - Vol ratios are positive and finite.
  - Burn-in: roughly the first 252 bars (longest rolling window) are
    dropped.
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
    NON_FEATURE_COLUMNS,
    compute_features_v1,
)


def _gbm(n: int = 1500, drift: float = 0.0003, vol: float = 0.012, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n)
    log_ret = drift - 0.5 * vol ** 2 + vol * eps
    return pd.Series(
        np.exp(np.log(100.0) + np.cumsum(log_ret)),
        index=pd.date_range("2015-01-01", periods=n, freq="D"),
        name="close",
    )


# ---------------------------------------------------------------------------
# Causal hygiene — THE blocking test
# ---------------------------------------------------------------------------


def test_causal_no_lookahead():
    """For any index t in the output, every feature column must be
    invariant under mutation of close[>t]. Procedure: compute baseline
    features, mutate close after a chosen index k, recompute, assert
    features[k] == baseline_features[k] for every feature column.

    Tests a sample of 20 indices to keep wall-clock reasonable.
    """
    n = 600
    close = _gbm(n=n, seed=77)
    baseline = compute_features_v1(close)
    if len(baseline) < 30:
        pytest.skip("not enough surviving rows for causal test")

    rng = np.random.default_rng(0)
    sample_indices = rng.choice(baseline.index[:-100], size=20, replace=False)
    for ts in sorted(sample_indices):
        pos = close.index.get_loc(ts)
        mutated = close.copy()
        # Scramble everything strictly after this position
        scramble = rng.uniform(0.5, 1.5, size=n - pos - 1)
        mutated.iloc[pos + 1:] = close.iloc[pos] * scramble

        recomputed = compute_features_v1(mutated)
        # ts must still be in the recomputed frame (its features depend on
        # data through pos, which is unchanged)
        assert ts in recomputed.index, (
            f"row {ts} (pos {pos}) disappeared after future perturbation"
        )
        for col in FEATURE_COLUMNS_V1:
            assert np.isclose(
                baseline.loc[ts, col], recomputed.loc[ts, col],
                rtol=1e-9, equal_nan=True,
            ), (
                f"feature `{col}` at {ts} (pos {pos}) changed when future "
                f"prices were mutated → look-ahead detected"
            )


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_all_columns_present():
    close = _gbm(n=1000)
    f = compute_features_v1(close)
    for col in FEATURE_COLUMNS_V1 + NON_FEATURE_COLUMNS:
        assert col in f.columns, f"missing column: {col}"


def test_no_nans_in_output():
    """Burn-in NaNs must be dropped — any NaN slipping through breaks
    XGBoost training silently in subtle ways."""
    close = _gbm(n=1500)
    f = compute_features_v1(close)
    nan_counts = f.isna().sum()
    bad = nan_counts[nan_counts > 0]
    assert bad.empty, f"NaN found in columns: {bad.to_dict()}"


def test_burnin_dropped_roughly_252():
    """With the longest rolling window at 252 and shift(1), the first
    ~252 bars become NaN and get dropped. Output should start around
    bar 252 ± a few.
    """
    close = _gbm(n=1000)
    f = compute_features_v1(close)
    first_kept = close.index.get_loc(f.index[0])
    # Tighter: must drop at least 252 (longest mom_252 window) but not
    # more than 260 (some slack for shift and rolling.std min_periods).
    assert 252 <= first_kept <= 260, (
        f"burn-in dropped {first_kept} rows; expected ~252"
    )


# ---------------------------------------------------------------------------
# Sanity ranges
# ---------------------------------------------------------------------------


def test_volatilities_positive():
    close = _gbm(n=1000)
    f = compute_features_v1(close)
    for col in ("vol_short", "vol_ewma", "vol_long", "vol_yearly"):
        assert (f[col] > 0).all(), f"{col} has non-positive values"


def test_vol_ratios_positive_and_finite():
    close = _gbm(n=1000)
    f = compute_features_v1(close)
    assert (f["vol_ratio_sl"] > 0).all()
    assert np.isfinite(f["vol_ratio_sl"]).all()
    assert (f["vol_ratio_ly"] > 0).all()
    assert np.isfinite(f["vol_ratio_ly"]).all()


def test_drawdown_non_positive():
    close = _gbm(n=1000)
    f = compute_features_v1(close)
    # By definition, close / rolling_max - 1 ∈ (-1, 0]. Strict equality
    # at peaks; negative below peaks.
    assert (f["drawdown_252"] <= 1e-9).all()
    assert (f["drawdown_252"] >= -1).all()


def test_autocorr_in_bounds():
    close = _gbm(n=1000)
    f = compute_features_v1(close)
    assert (f["autocorr_63"].between(-1, 1)).all()


def test_uptrend_makes_momentum_positive_on_avg():
    """Strong-drift GBM → average mom_20 should be solidly positive."""
    close = _gbm(n=1000, drift=0.002, vol=0.005, seed=11)
    f = compute_features_v1(close)
    assert f["mom_20"].mean() > 0.01  # > 1% over 20-bar average


def test_input_validation():
    with pytest.raises(TypeError, match="pd.Series"):
        compute_features_v1(np.zeros(100))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))