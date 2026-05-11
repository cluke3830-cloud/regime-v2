"""Tests for src.baselines.ms_garch (Brief 3.3)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines.ms_garch import (  # noqa: E402
    GARCHVolatilityModel,
    evaluate_forecast_rmse_vs_rolling,
    make_ms_garch_strategy,
)
from src.validation.cpcv_runner import run_cpcv_validation  # noqa: E402


def _vol_clustered_returns(n: int = 500, seed: int = 0) -> pd.Series:
    """GBM with vol clusters (alternating low/high vol regimes)."""
    rng = np.random.default_rng(seed)
    base_vol = 0.005
    high_vol = 0.025
    state = 0
    rets = np.zeros(n)
    for t in range(n):
        if rng.uniform() < 0.02:
            state = 1 - state
        vol = base_vol if state == 0 else high_vol
        rets[t] = rng.standard_normal() * vol
    return pd.Series(
        rets,
        index=pd.date_range("2018-01-01", periods=n, freq="D"),
        name="returns",
    )


def _synthetic_close_features(n: int = 500, seed: int = 0):
    rets = _vol_clustered_returns(n=n, seed=seed)
    close = pd.Series(
        np.exp(np.cumsum(rets.to_numpy())),
        index=rets.index, name="close",
    )
    features = pd.DataFrame({"close": close})
    log_ret = np.log(close).diff().fillna(0.0)
    return features, log_ret


# ---------------------------------------------------------------------------
# GARCH fit / predict
# ---------------------------------------------------------------------------


def test_garch_fit_recovers_positive_params():
    rets = _vol_clustered_returns(n=600)
    model = GARCHVolatilityModel().fit(rets)
    assert model.omega is not None and model.omega > 0
    assert model.alpha is not None and model.alpha >= 0
    assert model.beta is not None and model.beta >= 0
    # Stationarity boundary (alpha + beta must not be EXPLOSIVE).
    # GARCH on highly persistent vol clusters can converge right at the
    # boundary (sum ≈ 1.0); we require sum ≤ 1.0 + tiny epsilon.
    assert (model.alpha + model.beta) <= 1.0 + 1e-6
    # unconditional_var is set EITHER from the closed form (when
    # strictly stationary) OR from the last train cond-vol (when at
    # boundary). Both yield a finite positive value.
    assert model.unconditional_var is not None and model.unconditional_var > 0


def test_garch_fit_falls_back_on_short_data():
    short = pd.Series(
        np.random.default_rng(0).standard_normal(20) * 0.01,
        index=pd.date_range("2020-01-01", periods=20, freq="D"),
    )
    model = GARCHVolatilityModel().fit(short)
    assert model.omega is None
    # predict_volatility falls back to rolling 21-day
    out = model.predict_volatility(short)
    assert out.shape == short.shape


def test_garch_predict_volatility_shape_and_positive():
    rets = _vol_clustered_returns(n=500)
    model = GARCHVolatilityModel().fit(rets)
    vol = model.predict_volatility(rets)
    assert vol.shape == rets.shape
    assert (vol > 0).all()


def test_garch_recursion_is_causal():
    """At bar t the GARCH conditional vol should depend only on
    returns[<t]. Mutate returns[t:] and verify cond_vol[<t] unchanged.
    """
    rets = _vol_clustered_returns(n=400, seed=7)
    model = GARCHVolatilityModel().fit(rets.iloc[:300])
    baseline_vol = model.predict_volatility(rets)
    rng = np.random.default_rng(0)
    sample = rng.choice(np.arange(50, 350), size=10, replace=False)
    for t_idx in sorted(sample):
        mutated = rets.copy()
        mutated.iloc[t_idx + 1:] *= rng.uniform(0.5, 2.0, size=len(mutated) - t_idx - 1)
        mut_vol = model.predict_volatility(mutated)
        # Bars up to and including t_idx must match baseline
        assert np.isclose(
            mut_vol.iloc[t_idx], baseline_vol.iloc[t_idx], rtol=1e-9
        ), f"GARCH cond vol at bar {t_idx} changed under future mutation"


# ---------------------------------------------------------------------------
# Acceptance gate
# ---------------------------------------------------------------------------


def test_evaluate_forecast_rmse_returns_dict():
    rets = _vol_clustered_returns(n=500)
    model = GARCHVolatilityModel().fit(rets)
    vol = model.predict_volatility(rets)
    out = evaluate_forecast_rmse_vs_rolling(rets, vol)
    assert "garch_rmse" in out and "rolling_rmse" in out and "passes_gate" in out
    if np.isfinite(out["garch_rmse"]):
        assert out["garch_rmse"] >= 0
        assert out["rolling_rmse"] >= 0


def test_evaluate_forecast_rmse_degenerate():
    """With < 30 valid rows, returns NaN + False."""
    short = pd.Series(np.zeros(10), index=pd.date_range("2020-01-01", periods=10))
    vol = pd.Series(np.full(10, 0.01), index=short.index)
    out = evaluate_forecast_rmse_vs_rolling(short, vol)
    assert out["passes_gate"] is False


# ---------------------------------------------------------------------------
# Strategy_fn
# ---------------------------------------------------------------------------


def test_strategy_runs_in_cpcv():
    features, log_returns = _synthetic_close_features(n=800, seed=21)
    strategy = make_ms_garch_strategy()
    report = run_cpcv_validation(
        strategy, features, log_returns,
        strategy_name="ms_garch",
        n_splits=6, n_test_groups=2, n_trials=5,
    )
    assert report.n_paths == 15
    assert np.isfinite(report.sharpe_p50)


def test_strategy_positions_in_range():
    features, _ = _synthetic_close_features(n=500)
    strategy = make_ms_garch_strategy(
        target_ann_vol=0.14, max_position=1.0, min_position=0.0,
    )
    f_train = features.iloc[:350]
    f_test = features.iloc[350:]
    out = strategy(f_train, f_test)
    assert (out >= 0.0 - 1e-9).all()
    assert (out <= 1.0 + 1e-9).all()


def test_strategy_requires_close():
    rng = np.random.default_rng(0)
    f_train = pd.DataFrame({"mom_20": rng.standard_normal(50)})
    f_test = pd.DataFrame({"mom_20": rng.standard_normal(20)})
    strategy = make_ms_garch_strategy()
    with pytest.raises(KeyError, match="close"):
        strategy(f_train, f_test)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
