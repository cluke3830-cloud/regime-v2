"""Tests for src.features.intraday_rv (Brief 5.2)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.intraday_rv import (  # noqa: E402
    compute_bipower_variation,
    compute_realised_semivariance,
    compute_realised_skewness,
    compute_realised_variance,
    compute_yang_zhang_vol,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _synthetic_intraday(
    n_days: int = 10, bars_per_day: int = 78, daily_vol: float = 0.012,
    seed: int = 0,
) -> pd.DataFrame:
    """Synthetic 5-minute intraday bars for `n_days` business days."""
    rng = np.random.default_rng(seed)
    # 5-min vol = daily_vol / sqrt(bars_per_day)
    bar_vol = daily_vol / np.sqrt(bars_per_day)
    total_bars = n_days * bars_per_day
    log_returns = rng.standard_normal(total_bars) * bar_vol
    close = np.exp(np.log(100.0) + np.cumsum(log_returns))
    # 78 bars / day × 5 min/bar = 6.5 hours of trading
    timestamps = []
    for d in range(n_days):
        base = pd.Timestamp("2024-01-01") + pd.Timedelta(days=d)
        for b in range(bars_per_day):
            timestamps.append(base + pd.Timedelta(minutes=b * 5))
    return pd.DataFrame({"close": close}, index=pd.DatetimeIndex(timestamps))


def _synthetic_ohlc(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    log_ret = rng.standard_normal(n) * 0.012
    close = np.exp(np.log(100.0) + np.cumsum(log_ret))
    prev_close = np.concatenate([[close[0]], close[:-1]])
    # Open near previous close + small overnight gap
    open_ = prev_close * np.exp(rng.standard_normal(n) * 0.003)
    # High/low straddle close with intraday range
    intraday_range = np.abs(rng.standard_normal(n)) * 0.008
    high = np.maximum(open_, close) * np.exp(intraday_range)
    low = np.minimum(open_, close) * np.exp(-intraday_range)
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
    }, index=pd.date_range("2024-01-01", periods=n, freq="D"))


# ---------------------------------------------------------------------------
# Realised variance
# ---------------------------------------------------------------------------


def test_rv_shape_and_positive():
    df = _synthetic_intraday(n_days=10)
    rv = compute_realised_variance(df)
    assert len(rv) == 10
    assert (rv > 0).all()


def test_rv_matches_population_vol():
    """For 10 days of synthetic data at daily_vol=0.012, daily RV
    should be approximately daily_vol² = 0.000144.
    """
    df = _synthetic_intraday(n_days=200, bars_per_day=78, daily_vol=0.012, seed=11)
    rv = compute_realised_variance(df)
    expected = 0.012 ** 2
    assert 0.5 * expected < rv.mean() < 2.0 * expected, (
        f"mean RV {rv.mean():.6f} too far from expected {expected:.6f}"
    )


def test_rv_requires_close_column():
    df = pd.DataFrame({"open": [100.0]}, index=pd.date_range("2024-01-01", periods=1))
    with pytest.raises(KeyError, match="close"):
        compute_realised_variance(df)


# ---------------------------------------------------------------------------
# Realised semivariance
# ---------------------------------------------------------------------------


def test_semivariance_sums_to_rv():
    """RSV+ + RSV- ≈ RV (exact when all bars are non-zero; tiny floor
    when some zero returns get dropped by the strict-positive/negative
    indicators).
    """
    df = _synthetic_intraday(n_days=20)
    rv = compute_realised_variance(df)
    rsv_pos, rsv_neg = compute_realised_semivariance(df)
    np.testing.assert_allclose(
        (rsv_pos + rsv_neg).to_numpy(), rv.to_numpy(), rtol=1e-9, atol=1e-12
    )


def test_semivariance_balanced_on_symmetric_returns():
    """Symmetric return distribution → RSV+ ≈ RSV-."""
    df = _synthetic_intraday(n_days=100, seed=22)
    rsv_pos, rsv_neg = compute_realised_semivariance(df)
    ratio = (rsv_pos.mean() / rsv_neg.mean())
    assert 0.7 < ratio < 1.3, f"symmetric expected ~1.0 ratio, got {ratio:.3f}"


# ---------------------------------------------------------------------------
# Realised skewness
# ---------------------------------------------------------------------------


def test_realised_skewness_finite():
    df = _synthetic_intraday(n_days=20)
    rs = compute_realised_skewness(df)
    assert np.isfinite(rs).all()


def test_realised_skewness_symmetric_returns_near_zero():
    df = _synthetic_intraday(n_days=200, seed=33)
    rs = compute_realised_skewness(df)
    # Average daily RS on symmetric returns should be near 0
    assert abs(rs.mean()) < 0.5


# ---------------------------------------------------------------------------
# Bipower variation
# ---------------------------------------------------------------------------


def test_bipower_variation_positive_and_similar_to_rv():
    """In the absence of jumps, BV ≈ RV. Synthetic data has no jumps,
    so BV/RV should be near 1.
    """
    df = _synthetic_intraday(n_days=100)
    rv = compute_realised_variance(df)
    bv = compute_bipower_variation(df)
    ratio = (bv.mean() / rv.mean())
    assert 0.6 < ratio < 1.4, f"BV/RV ratio {ratio:.3f} too far from 1"


# ---------------------------------------------------------------------------
# Yang-Zhang OHLC vol
# ---------------------------------------------------------------------------


def test_yang_zhang_shape_and_positive():
    df = _synthetic_ohlc(n=100)
    yz = compute_yang_zhang_vol(df, window=20)
    assert yz.shape == (100,)
    # First 20 will be NaN; the rest should be positive
    assert (yz.iloc[20:] > 0).all()


def test_yang_zhang_rejects_missing_columns():
    df = pd.DataFrame({"close": [100.0]}, index=pd.date_range("2024-01-01", periods=1))
    with pytest.raises(KeyError, match="open"):
        compute_yang_zhang_vol(df)


def test_yang_zhang_recovers_synthetic_vol():
    """For synthetic OHLC with daily_vol=0.012, YZ should give annualised
    vol ≈ 0.012 * sqrt(252) ≈ 0.19.
    """
    df = _synthetic_ohlc(n=500, seed=99)
    yz = compute_yang_zhang_vol(df, window=50)
    expected_annual = 0.012 * np.sqrt(252)
    # YZ is a noisy estimate; allow ±2x slack
    assert 0.5 * expected_annual < yz.iloc[100:].mean() < 2.0 * expected_annual


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
