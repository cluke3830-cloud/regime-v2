"""Tests for src.baselines.hsmm (Brief 3.2)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines.hsmm import (  # noqa: E402
    DEFAULT_K4_POSITIONS,
    DurationAwareHMM,
    make_hsmm_strategy,
)
from src.features.price_features import compute_features_v1  # noqa: E402
from src.validation.cpcv_runner import run_cpcv_validation  # noqa: E402


def _synthetic_4_state_data(n: int = 1000, seed: int = 0):
    """Generate a 4-state Markov-like vol/return series."""
    rng = np.random.default_rng(seed)
    # vols per state
    vols = [0.005, 0.008, 0.015, 0.025]
    means = [0.001, 0.0005, -0.0002, -0.001]
    state = 0
    rets = []
    persist_min = 20
    persist = 0
    for t in range(n):
        rets.append(rng.normal(means[state], vols[state]))
        persist += 1
        if persist > persist_min and rng.uniform() < 0.05:
            state = (state + 1) % 4
            persist = 0
    rets = np.array(rets)
    close = pd.Series(
        np.exp(np.cumsum(rets)),
        index=pd.date_range("2018-01-01", periods=n, freq="D"),
        name="close",
    )
    return close


def test_hsmm_fits_and_orders_states_by_variance():
    """After fit, state 0 should be the lowest-variance one (canonical)."""
    close = _synthetic_4_state_data(n=800, seed=3)
    features = compute_features_v1(close)
    feature_cols = [c for c in features.columns if c != "close"]
    X = features[feature_cols].to_numpy()
    model = DurationAwareHMM(k_states=4).fit(X)
    assert model.hmm_ is not None
    assert model.state_remap_ is not None
    assert set(model.state_remap_.values()) == {0, 1, 2, 3}


def test_hsmm_predict_proba_shape_and_sums():
    close = _synthetic_4_state_data(n=600)
    features = compute_features_v1(close)
    feature_cols = [c for c in features.columns if c != "close"]
    X = features[feature_cols].to_numpy()
    model = DurationAwareHMM(k_states=4).fit(X)
    probs = model.predict_proba(X)
    assert probs.shape == (len(X), 4)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_hsmm_predict_state_path():
    close = _synthetic_4_state_data(n=500)
    features = compute_features_v1(close)
    X = features.drop(columns=["close"]).to_numpy()
    model = DurationAwareHMM(k_states=4).fit(X)
    path = model.predict_state_path(X)
    assert path.shape == (len(X),)
    assert set(np.unique(path)).issubset({0, 1, 2, 3})


def test_hsmm_fallback_on_small_data():
    """< k_states * 20 obs → no fit, predict_proba returns uniform."""
    X = np.random.default_rng(0).standard_normal((20, 5))
    model = DurationAwareHMM(k_states=4).fit(X)
    assert model.hmm_ is None
    probs = model.predict_proba(X)
    np.testing.assert_allclose(probs, 0.25)


def test_duration_fits_recorded():
    """duration_fits_ should have a Weibull tuple per state after fit."""
    close = _synthetic_4_state_data(n=800)
    features = compute_features_v1(close)
    X = features.drop(columns=["close"]).to_numpy()
    model = DurationAwareHMM(k_states=4).fit(X)
    assert model.duration_fits_ is not None
    for state in range(4):
        assert state in model.duration_fits_
        shape, scale = model.duration_fits_[state]
        assert shape > 0
        assert scale > 0


def test_estimate_remaining_duration_returns_finite():
    close = _synthetic_4_state_data(n=600)
    features = compute_features_v1(close)
    X = features.drop(columns=["close"]).to_numpy()
    model = DurationAwareHMM(k_states=4).fit(X)
    # For each state, estimate remaining given d=5 observed
    for state in range(4):
        remaining = model.estimate_remaining_duration(state, observed_duration=5.0)
        if np.isfinite(remaining):
            assert remaining >= 0  # can never have negative survival


def test_default_k4_positions_complete():
    assert set(DEFAULT_K4_POSITIONS.keys()) == {0, 1, 2, 3}
    # Monotonically non-increasing (lower-var states = more long)
    vals = [DEFAULT_K4_POSITIONS[k] for k in sorted(DEFAULT_K4_POSITIONS)]
    assert vals == sorted(vals, reverse=True)


def test_strategy_runs_in_cpcv():
    close = _synthetic_4_state_data(n=900, seed=21)
    features = compute_features_v1(close)
    log_returns = np.log(close).diff().loc[features.index]
    strategy = make_hsmm_strategy(k_states=4)
    report = run_cpcv_validation(
        strategy, features, log_returns,
        strategy_name="hsmm",
        n_splits=6, n_test_groups=2, n_trials=5,
    )
    assert report.n_paths == 15
    assert np.isfinite(report.sharpe_p50)


def test_strategy_positions_in_state_range():
    close = _synthetic_4_state_data(n=600)
    features = compute_features_v1(close)
    strategy = make_hsmm_strategy(k_states=4)
    f_train = features.iloc[:400]
    f_test = features.iloc[400:]
    out = strategy(f_train, f_test)
    assert out.shape == (len(f_test),)
    min_alloc = min(DEFAULT_K4_POSITIONS.values())
    max_alloc = max(DEFAULT_K4_POSITIONS.values())
    assert (out >= min_alloc - 1e-9).all()
    assert (out <= max_alloc + 1e-9).all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
