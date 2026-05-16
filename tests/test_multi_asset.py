"""Acceptance tests for src.validation.multi_asset (Brief 1.5).

Audit-prescribed acceptance test:
    Cross-asset table renders in validation_report.md. The strategy passes
    a soft gate if mean Sharpe across the asset universe is positive AND
    at least 70% of assets show positive Sharpe.

All tests here are HERMETIC — they exercise the harness via
``evaluate_close`` on synthetic GBM closes. No yfinance network calls.
The full-network evaluate_multi_asset path is covered by an integration
test gated on ``--run-network``.
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

from src.strategies.benchmarks import buy_and_hold, flat, momentum_20d  # noqa: E402
from src.validation.multi_asset import (  # noqa: E402
    DEFAULT_UNIVERSE,
    default_feature_fn,
    evaluate_close,
    evaluate_one_asset,
    multi_asset_summary,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _gbm_close(n: int, drift: float, vol: float, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n)
    log_ret = drift - 0.5 * vol ** 2 + vol * eps
    return pd.Series(
        np.exp(np.log(100.0) + np.cumsum(log_ret)),
        index=pd.date_range("2015-01-01", periods=n, freq="D"),
        name="close",
    )


# ---------------------------------------------------------------------------
# default_feature_fn
# ---------------------------------------------------------------------------


def test_default_feature_fn_shapes_and_causal():
    """Brief 2.1.1: default_feature_fn now delegates to the Tier-1
    feature pipeline (14 features + close). Verify the schema and the
    strict-causal property of mom_20 — at the first surviving row,
    mom_20 must equal the sum of the prior 20 log returns, NOT including
    the current bar.

    Needs ≥ 300 bars because the longest rolling window (mom_252) plus
    shift(1) eats ~252 bars of burn-in.
    """
    close = _gbm_close(400, drift=0.0005, vol=0.01, seed=3)
    features, log_ret = default_feature_fn(close)
    assert len(features) == len(log_ret)
    assert features.index.equals(log_ret.index)
    # The new pipeline ships 14 features + close
    expected_cols = {
        "close",
        "mom_5", "mom_20", "mom_63", "mom_252",
        "vol_short", "vol_ewma", "vol_long", "vol_yearly",
        "vol_ratio_sl", "vol_ratio_ly",
        "shock_z", "drawdown_252", "autocorr_63", "trend_dir",
    }
    assert set(features.columns) == expected_cols

    full_log_ret = np.log(close).diff()
    first_valid = features.index[0]
    pos = close.index.get_loc(first_valid)
    expected_mom = full_log_ret.iloc[pos - 20: pos].sum()
    actual_mom = features.loc[first_valid, "mom_20"]
    assert actual_mom == pytest.approx(expected_mom, rel=1e-9)


# ---------------------------------------------------------------------------
# evaluate_close — hermetic
# ---------------------------------------------------------------------------


def test_evaluate_close_buy_and_hold_uptrend():
    """Buy-and-hold on a strongly upward-drifting GBM should produce
    consistently positive Sharpe across CPCV paths.
    """
    close = _gbm_close(1500, drift=0.0008, vol=0.01, seed=42)
    metrics = evaluate_close(
        close, buy_and_hold,
        n_splits=10, n_test_groups=2, n_trials=1,
    )
    assert metrics["n_paths"] == 45
    assert metrics["sharpe_p50"] > 0
    assert metrics["sharpe_mean"] > 0
    assert metrics["dsr_p_value"] > 0.5  # significant edge (no multi-trial deflation)


def test_evaluate_close_flat_strategy_zero_sharpe():
    """`flat` always returns zero positions → strategy returns are zero →
    Sharpe across paths is exactly zero (not NaN — we deliberately handle
    sigma=0 inside _compute_path_metrics).
    """
    close = _gbm_close(1000, drift=0.0003, vol=0.012, seed=11)
    metrics = evaluate_close(
        close, flat,
        n_splits=8, n_test_groups=2, n_trials=5,
    )
    assert metrics["sharpe_p50"] == 0.0
    assert metrics["sharpe_p05"] == 0.0
    assert metrics["sharpe_p95"] == 0.0


def test_evaluate_close_reproducibility():
    """Same close + same params → identical metrics dict."""
    close = _gbm_close(800, drift=0.0005, vol=0.012, seed=7)
    m1 = evaluate_close(close, buy_and_hold, n_splits=8, n_test_groups=2, n_trials=10)
    m2 = evaluate_close(close, buy_and_hold, n_splits=8, n_test_groups=2, n_trials=10)
    for key in ("sharpe_p05", "sharpe_p50", "sharpe_p95", "dsr_p_value"):
        assert m1[key] == m2[key], f"{key} not reproducible: {m1[key]} vs {m2[key]}"


def test_evaluate_close_returns_dsr_for_real_strategy():
    """A real strategy with non-zero returns must produce a finite DSR."""
    close = _gbm_close(1200, drift=0.0006, vol=0.012, seed=21)
    metrics = evaluate_close(
        close, momentum_20d,
        n_splits=10, n_test_groups=2, n_trials=10,
    )
    assert np.isfinite(metrics["dsr_p_value"])
    assert 0.0 <= metrics["dsr_p_value"] <= 1.0
    assert np.isfinite(metrics["dsr_observed_sharpe"])


# ---------------------------------------------------------------------------
# multi_asset_summary — soft gate logic
# ---------------------------------------------------------------------------


def _mk_result(ticker: str, sharpe_p50: float, error: str | None = None) -> dict:
    """Helper to build a fake per-asset metrics dict."""
    nan = float("nan")
    if error is not None:
        return {
            "ticker": ticker, "error": error,
            "sharpe_p50": nan, "sharpe_p05": nan, "sharpe_p95": nan,
            "sharpe_mean": nan, "max_dd_p05": nan, "max_dd_p50": nan,
            "max_dd_p95": nan, "dsr_p_value": nan, "dsr_observed_sharpe": nan,
            "n_bars": 0, "n_paths": 0,
        }
    return {
        "ticker": ticker, "error": None,
        "sharpe_p50": sharpe_p50, "sharpe_p05": sharpe_p50 - 0.5,
        "sharpe_p95": sharpe_p50 + 0.5, "sharpe_mean": sharpe_p50,
        "max_dd_p05": -0.30, "max_dd_p50": -0.15, "max_dd_p95": -0.05,
        "dsr_p_value": 0.9 if sharpe_p50 > 0 else 0.3,
        "dsr_observed_sharpe": sharpe_p50,
        "n_bars": 2500, "n_paths": 45,
    }


def test_summary_all_positive_passes_gate():
    """10 assets, all positive Sharpe → fraction = 1.0, gate passes."""
    results = {
        f"ASSET{i}": _mk_result(f"ASSET{i}", 0.5)
        for i in range(10)
    }
    summary = multi_asset_summary(results)
    assert summary["n_assets"] == 10
    assert summary["n_evaluated"] == 10
    assert summary["n_failed"] == 0
    assert summary["n_positive_sharpe"] == 10
    assert summary["fraction_positive_sharpe"] == 1.0
    assert summary["mean_sharpe_p50"] == pytest.approx(0.5)
    assert summary["passes_gate"] is True


def test_summary_exactly_70_percent_passes_gate():
    """7/10 positive → exactly at the threshold, mean still positive → pass."""
    results = {}
    for i in range(7):
        results[f"GOOD{i}"] = _mk_result(f"GOOD{i}", 0.6)
    for i in range(3):
        results[f"BAD{i}"] = _mk_result(f"BAD{i}", -0.3)
    summary = multi_asset_summary(results)
    assert summary["fraction_positive_sharpe"] == pytest.approx(0.7)
    assert summary["mean_sharpe_p50"] > 0  # 7 * 0.6 - 3 * 0.3 = 3.3 → mean 0.33
    assert summary["passes_gate"] is True


def test_summary_69_percent_fails_gate():
    """Strictly below 70% → fails the gate even if mean is positive."""
    results = {}
    for i in range(6):
        results[f"GOOD{i}"] = _mk_result(f"GOOD{i}", 0.8)
    for i in range(4):
        results[f"BAD{i}"] = _mk_result(f"BAD{i}", -0.1)
    summary = multi_asset_summary(results)
    assert summary["fraction_positive_sharpe"] == pytest.approx(0.6)
    assert summary["passes_gate"] is False


def test_summary_positive_fraction_but_negative_mean_fails():
    """8/10 positive but their mean is dragged below zero by tail-heavy
    losers → both AND-clauses required, gate fails.
    """
    results = {}
    for i in range(8):
        results[f"GOOD{i}"] = _mk_result(f"GOOD{i}", 0.1)
    for i in range(2):
        results[f"BAD{i}"] = _mk_result(f"BAD{i}", -5.0)  # huge negative
    summary = multi_asset_summary(results)
    assert summary["fraction_positive_sharpe"] == pytest.approx(0.8)
    assert summary["mean_sharpe_p50"] < 0  # 8·0.1 - 2·5 = -9.2 → -0.92
    assert summary["passes_gate"] is False


def test_summary_handles_failed_assets():
    """Failed downloads must not count against the positive-Sharpe fraction."""
    results = {
        "GOOD1": _mk_result("GOOD1", 0.4),
        "GOOD2": _mk_result("GOOD2", 0.6),
        "GOOD3": _mk_result("GOOD3", 0.3),
        "FAILED": _mk_result("FAILED", float("nan"), error="network timeout"),
    }
    summary = multi_asset_summary(results)
    assert summary["n_assets"] == 4
    assert summary["n_evaluated"] == 3
    assert summary["n_failed"] == 1
    assert summary["fraction_positive_sharpe"] == pytest.approx(1.0)


def test_summary_all_failed():
    """If every download failed, summary degrades gracefully."""
    results = {
        f"FAIL{i}": _mk_result(f"FAIL{i}", float("nan"), error="x")
        for i in range(3)
    }
    summary = multi_asset_summary(results)
    assert summary["n_evaluated"] == 0
    assert summary["passes_gate"] is False
    assert summary["fraction_positive_sharpe"] == 0.0
    assert np.isnan(summary["mean_sharpe_p50"])


def test_summary_custom_threshold():
    """The gate threshold is configurable."""
    results = {
        f"GOOD{i}": _mk_result(f"GOOD{i}", 0.5) for i in range(5)
    }
    for i in range(5):
        results[f"BAD{i}"] = _mk_result(f"BAD{i}", -0.5)
    summary = multi_asset_summary(results, soft_gate_fraction=0.40)
    assert summary["fraction_positive_sharpe"] == pytest.approx(0.5)
    # mean Sharpe is exactly 0 → fails the strict-greater-than-0 mean clause
    assert summary["mean_sharpe_p50"] == pytest.approx(0.0)
    assert summary["passes_gate"] is False  # mean must be > 0, not >= 0


# ---------------------------------------------------------------------------
# evaluate_one_asset — failure handling (no network)
# ---------------------------------------------------------------------------


def test_evaluate_one_asset_handles_download_failure(monkeypatch):
    """If the loader raises, evaluate_one_asset returns an error dict
    instead of propagating the exception.
    """
    from src.validation import multi_asset as ma_mod

    def fake_load(*args, **kwargs):
        raise RuntimeError("simulated yfinance failure")

    monkeypatch.setattr(ma_mod, "load_close", fake_load)
    result = ma_mod.evaluate_one_asset(
        "SOMETHING", buy_and_hold,
        start="2015-01-01", end="2025-01-01",
    )
    assert result["error"] is not None
    assert "simulated yfinance failure" in result["error"]
    assert np.isnan(result["sharpe_p50"])
    assert result["ticker"] == "SOMETHING"


def test_evaluate_one_asset_handles_too_few_bars(monkeypatch):
    """If the asset has fewer bars than n_splits * 50, return error
    rather than letting CPCV explode.
    """
    from src.validation import multi_asset as ma_mod

    def tiny_load(*args, **kwargs):
        return _gbm_close(100, drift=0.0, vol=0.01, seed=0)

    monkeypatch.setattr(ma_mod, "load_close", tiny_load)
    result = ma_mod.evaluate_one_asset(
        "TINY", buy_and_hold,
        start="2024-01-01", end="2024-12-31",
        n_splits=10,
    )
    assert result["error"] is not None
    assert "too few bars" in result["error"]


# ---------------------------------------------------------------------------
# DEFAULT_UNIVERSE sanity
# ---------------------------------------------------------------------------


def test_default_universe_has_13_tickers():
    assert len(DEFAULT_UNIVERSE) == 13  # 10 original + 3 Forex (Phase 7)
    # spans equity, fixed income, gold, crypto, FX
    assert "SPY" in DEFAULT_UNIVERSE
    assert "TLT" in DEFAULT_UNIVERSE
    assert "GLD" in DEFAULT_UNIVERSE
    assert "EURUSD=X" in DEFAULT_UNIVERSE
    assert "GBPUSD=X" in DEFAULT_UNIVERSE
    assert "AUDUSD=X" in DEFAULT_UNIVERSE
    assert "BTC-USD" in DEFAULT_UNIVERSE


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))