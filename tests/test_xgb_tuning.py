"""Tests for src.regime.xgb_tuning (Brief 2.1.3).

Key invariants under test:
  - tune_xgb_hparams picks every key in the param_grid.
  - Reproducibility: same (X, y, seed) → identical best_params and scores.
  - On a problem with KNOWN best capacity, the tuner picks something
    sensible (not the most extreme overfitting corner).
  - On a problem where one combo is obviously dominant, the tuner picks
    that combo.
  - The tuned strategy_fn runs cleanly inside CPCV and beats the trivial
    flat baseline on signal-rich synthetic data.
  - Degenerate input (single-class y) doesn't crash — falls back to
    first grid entry.

All hermetic — no network.
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

from src.regime.xgb_tuning import (  # noqa: E402
    DEFAULT_PARAM_GRID_FULL,
    DEFAULT_PARAM_GRID_SMALL,
    make_tuned_regime_xgboost_strategy,
    tune_xgb_hparams,
)
from src.strategies.benchmarks import flat  # noqa: E402
from src.validation.cpcv_runner import run_cpcv_validation  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_classification_data(n: int = 600, seed: int = 0):
    """Strong signal in feature[0]: deterministic 3-class label."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 4))
    y = np.zeros(n, dtype=np.int64)
    y[X[:, 0] > 0.5] = 1
    y[X[:, 0] < -0.5] = -1
    return X, y


# ---------------------------------------------------------------------------
# tune_xgb_hparams — core behaviour
# ---------------------------------------------------------------------------


def test_tuner_returns_grid_keys():
    X, y = _make_classification_data(n=400)
    grid = {"max_depth": [3, 5], "n_estimators": [50, 100]}
    best, scores = tune_xgb_hparams(
        X, y,
        param_grid=grid,
        inner_n_splits=4, inner_n_test_groups=1,
    )
    assert set(best.keys()) == set(grid.keys())
    # All grid combos must appear in scores
    assert len(scores) == 2 * 2
    for combo in scores:
        assert isinstance(combo, tuple)
        assert len(combo) == 2


def test_tuner_is_reproducible():
    X, y = _make_classification_data(n=500, seed=11)
    grid = {"max_depth": [3, 5], "n_estimators": [50, 100]}
    best1, scores1 = tune_xgb_hparams(
        X, y, param_grid=grid, inner_n_splits=4, seed=42,
    )
    best2, scores2 = tune_xgb_hparams(
        X, y, param_grid=grid, inner_n_splits=4, seed=42,
    )
    assert best1 == best2
    assert scores1 == scores2


def test_tuner_picks_dominant_combo():
    """When the grid spans an obvious capacity gap, the tuner must pick
    a combo with sufficient capacity. Construction: under-fit corners
    (n_estimators=10, regardless of depth) score ~0.64 log-loss; full
    corners (n_estimators=200) score ~0.01. We assert: (i) the worst
    combo is one of the n=10 corners, and (ii) the best combo has
    n=200.
    """
    X, y = _make_classification_data(n=800, seed=3)
    grid = {
        "max_depth":    [2, 5],
        "n_estimators": [10, 200],
    }
    best, scores = tune_xgb_hparams(
        X, y, param_grid=grid,
        inner_n_splits=4, inner_n_test_groups=1, seed=42,
    )
    worst = max(scores, key=lambda k: scores[k])
    # worst is whichever of the under-fit corners
    assert worst[1] == 10, (
        f"expected worst to have n_estimators=10, got {worst}. "
        f"Scores: {scores}"
    )
    # best must have enough capacity (n_estimators=200) to fit the signal
    assert best["n_estimators"] == 200, (
        f"expected best to have n_estimators=200, got {best}. "
        f"Scores: {scores}"
    )


def test_tuner_single_class_y_does_not_crash():
    """Edge case: y is all the same class. RegimeXGBoost falls back to
    predicting the prior; tuner shouldn't propagate that as an exception.
    """
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 3))
    y = np.zeros(100, dtype=np.int64)
    grid = {"max_depth": [3, 5]}
    best, scores = tune_xgb_hparams(
        X, y, param_grid=grid,
        inner_n_splits=4, inner_n_test_groups=1,
    )
    # All scores should be finite (we always return SOMETHING)
    assert "max_depth" in best
    assert len(scores) == 2


def test_tuner_with_sample_weights():
    """Tuner accepts sample weights without crash."""
    X, y = _make_classification_data(n=400)
    weights = np.linspace(0.5, 1.5, len(X))  # mild time-decay-like
    grid = {"max_depth": [3, 5], "n_estimators": [50]}
    best, _ = tune_xgb_hparams(
        X, y,
        sample_weight=weights,
        param_grid=grid,
        inner_n_splits=4,
    )
    assert "max_depth" in best


def test_default_param_grids_well_formed():
    """Smoke test the canned grids."""
    for grid in (DEFAULT_PARAM_GRID_SMALL, DEFAULT_PARAM_GRID_FULL):
        assert "max_depth" in grid
        assert "n_estimators" in grid
        for k, vs in grid.items():
            assert len(vs) >= 1
    # SMALL must be a proper subset count of FULL
    small_n = (
        len(DEFAULT_PARAM_GRID_SMALL["max_depth"]) *
        len(DEFAULT_PARAM_GRID_SMALL["eta"]) *
        len(DEFAULT_PARAM_GRID_SMALL["n_estimators"])
    )
    full_n = (
        len(DEFAULT_PARAM_GRID_FULL["max_depth"]) *
        len(DEFAULT_PARAM_GRID_FULL["eta"]) *
        len(DEFAULT_PARAM_GRID_FULL["n_estimators"])
    )
    assert small_n == 4
    assert full_n == 36


# ---------------------------------------------------------------------------
# make_tuned_regime_xgboost_strategy — full CPCV integration
# ---------------------------------------------------------------------------


def _make_synthetic_close_features(n: int = 1200, seed: int = 42):
    """Regime-dependent drift series — proven learnable in Brief 2.1 tests."""
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n)
    log_ret = np.zeros(n)
    for t in range(20, n):
        recent_mom = log_ret[t - 20: t].sum()
        drift = 0.001 if recent_mom > 0 else -0.0005
        log_ret[t] = drift + 0.01 * eps[t]
    close = pd.Series(
        np.exp(np.log(100.0) + np.cumsum(log_ret)),
        index=pd.date_range("2015-01-01", periods=n, freq="D"),
        name="close",
    )
    log_returns = np.log(close).diff()
    mom_20 = log_returns.rolling(20).sum().shift(1)
    vol_ewma = log_returns.ewm(alpha=0.06, min_periods=20).std()
    features = pd.DataFrame({
        "close": close, "mom_20": mom_20, "vol_ewma": vol_ewma
    }).dropna()
    return features, log_returns.loc[features.index]


def test_tuned_strategy_runs_in_cpcv():
    """Smoke test — tuned strategy executes through CPCV without crashing."""
    features, log_returns = _make_synthetic_close_features(n=600)
    grid = {"max_depth": [3, 5], "n_estimators": [50]}
    strategy = make_tuned_regime_xgboost_strategy(
        param_grid=grid,
        inner_n_splits=3, inner_n_test_groups=1,
    )
    report = run_cpcv_validation(
        strategy, features, log_returns,
        strategy_name="xgb_tuned",
        n_splits=6, n_test_groups=2, n_trials=10,
    )
    assert report.n_paths == 15  # C(6, 2)
    assert np.isfinite(report.sharpe_p50)


def test_tuned_strategy_beats_flat_on_signal():
    """On the regime-dependent-drift dataset, the tuned strategy must
    produce a materially positive OOS Sharpe — and beat the flat
    baseline (which is exactly 0).
    """
    features, log_returns = _make_synthetic_close_features(n=1200, seed=7)
    grid = {"max_depth": [3, 5], "n_estimators": [50]}
    tuned = make_tuned_regime_xgboost_strategy(
        param_grid=grid,
        inner_n_splits=3, inner_n_test_groups=1,
    )
    tuned_report = run_cpcv_validation(
        tuned, features, log_returns,
        strategy_name="xgb_tuned", n_splits=8, n_test_groups=2, n_trials=5,
    )
    flat_report = run_cpcv_validation(
        flat, features, log_returns,
        strategy_name="flat", n_splits=8, n_test_groups=2, n_trials=5,
    )
    assert tuned_report.sharpe_p50 > flat_report.sharpe_p50
    assert tuned_report.sharpe_p50 > 0.3


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))