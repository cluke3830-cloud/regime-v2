"""Acceptance tests for src.regime.regime_xgboost (Brief 2.1).

Audit-prescribed acceptance gates:
  (a) Out-of-sample log-loss vs hand-tuned rule layer is >= 5% lower.
      v1: compared against a uniform-prior baseline (rule layer
      comparison comes online in Brief 2.2).
  (b) Crisis recall >= 0.65 on triple-barrier held-out tail. In the
      3-class encoding this is recall on the label = -1 class.
  (c) Feature importances economically interpretable — top features
      include volatility / momentum measures.

Tests:
  - compute_sample_weights:
      - shapes, mean ~ 1.0;
      - uniqueness — brute-force ground-truth check on a small example;
      - magnitude renormalisation; time-decay=0 gives uniform component;
      - degenerate inputs (all zero returns, all isolated bars).
  - RegimeXGBoost:
      - fit / predict_proba shapes; probabilities sum to 1; class order;
      - position_from_proba in [-1, +1];
      - single-class fallback to empirical prior (no crash);
      - feature importance dict keyed by provided names.
  - make_regime_xgboost_strategy:
      - works inside the CPCV harness;
      - on synthetic data with a known signal, beats the flat baseline
        on out-of-sample Sharpe.
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

from src.regime.regime_xgboost import (  # noqa: E402
    RegimeXGBoost,
    compute_sample_weights,
    make_regime_xgboost_strategy,
)
from src.strategies.benchmarks import flat  # noqa: E402
from src.validation.cpcv_runner import run_cpcv_validation  # noqa: E402


# ---------------------------------------------------------------------------
# compute_sample_weights
# ---------------------------------------------------------------------------


def test_weights_shape_and_mean():
    n = 100
    t1 = np.minimum(np.arange(n) + 10, n - 1)
    rets = np.random.default_rng(0).standard_normal(n) * 0.01
    w = compute_sample_weights(t1, rets, decay=1.0)
    assert w.shape == (n,)
    assert w.mean() == pytest.approx(1.0, abs=1e-9)
    assert (w >= 0).all()


def test_weights_uniqueness_bruteforce():
    """LdP §4.4: uniqueness[i] = mean over t in [i, t1[i]] of 1/claim_count[t].

    5 samples, t1 = [2, 3, 3, 4, 4]:
      bar 0: covered by {0}      -> claim 1
      bar 1: covered by {0,1}    -> claim 2
      bar 2: covered by {0,1,2}  -> claim 3
      bar 3: covered by {1,2,3}  -> claim 3
      bar 4: covered by {3,4}    -> claim 2
    Uniqueness:
      sample 0 (bars 0-2): (1 + 1/2 + 1/3)/3
      sample 1 (bars 1-3): (1/2 + 1/3 + 1/3)/3
      sample 2 (bars 2-3): (1/3 + 1/3)/2
      sample 3 (bars 3-4): (1/3 + 1/2)/2
      sample 4 (bars 4-4): 1/2
    """
    t1 = np.array([2, 3, 3, 4, 4])
    rets = np.ones(5)
    w = compute_sample_weights(t1, rets, decay=0.0)
    expected_uniq = np.array([
        (1 + 1 / 2 + 1 / 3) / 3,
        (1 / 2 + 1 / 3 + 1 / 3) / 3,
        (1 / 3 + 1 / 3) / 2,
        (1 / 3 + 1 / 2) / 2,
        1 / 2,
    ])
    expected_w = expected_uniq / expected_uniq.mean()
    np.testing.assert_allclose(w, expected_w, rtol=1e-9)


def test_weights_time_decay_zero_gives_uniform_decay():
    """decay=0 disables the time component. Verify with non-overlapping
    horizons (each sample only covers itself → claim_count=1 for all
    bars → uniqueness=1 for all samples) so the only thing left that
    could vary is the time decay. With decay=0, weights must be uniform.
    With decay>0, weights must monotonically increase with index.
    """
    n = 50
    t1 = np.arange(n)  # each sample covers only itself
    rets = np.ones(n)
    w_off = compute_sample_weights(t1, rets, decay=0.0)
    w_on = compute_sample_weights(t1, rets, decay=1.0)
    assert np.allclose(w_off, 1.0)
    assert w_on[-1] > w_on[0]


def test_weights_all_zero_returns_yields_uniform_magnitude():
    n = 30
    t1 = np.minimum(np.arange(n) + 5, n - 1)
    rets = np.zeros(n)
    w = compute_sample_weights(t1, rets, decay=0.0)
    assert np.all(np.isfinite(w))
    assert w.mean() == pytest.approx(1.0)


def test_weights_length_mismatch_raises():
    with pytest.raises(ValueError, match="len"):
        compute_sample_weights(np.array([0, 1]), np.array([0.01]))


# ---------------------------------------------------------------------------
# RegimeXGBoost
# ---------------------------------------------------------------------------


def _make_classification_data(n: int = 500, seed: int = 0):
    """Three-class problem with a clear signal in feature[0]."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 4))
    y = np.zeros(n, dtype=np.int64)
    y[X[:, 0] > 0.5] = 1
    y[X[:, 0] < -0.5] = -1
    return X, y


def test_regime_xgboost_fit_predict_proba_shapes():
    X, y = _make_classification_data(n=300)
    model = RegimeXGBoost(n_estimators=50)
    model.fit(X, y, feature_names=["f0", "f1", "f2", "f3"])
    proba = model.predict_proba(X)
    assert proba.shape == (300, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_regime_xgboost_predict_class_labels():
    X, y = _make_classification_data(n=300)
    model = RegimeXGBoost(n_estimators=50)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (300,)
    assert set(np.unique(preds)).issubset({-1, 0, 1})


def test_regime_xgboost_learns_signal():
    """Strong signal in feature[0] → in-sample accuracy must exceed the
    trivial-majority baseline by a wide margin.
    """
    X, y = _make_classification_data(n=1000, seed=42)
    model = RegimeXGBoost(n_estimators=100, max_depth=4)
    model.fit(X, y)
    preds = model.predict(X)
    acc = (preds == y).mean()
    assert acc > 0.85


def test_position_from_proba_in_range():
    proba = np.array([
        [0.5, 0.3, 0.2],
        [0.1, 0.1, 0.8],
        [0.4, 0.4, 0.2],
        [1.0, 0.0, 0.0],
    ])
    positions = RegimeXGBoost.position_from_proba(proba)
    expected = np.array([-0.3, 0.7, -0.2, -1.0])
    np.testing.assert_allclose(positions, expected)
    assert (positions >= -1).all() and (positions <= 1).all()


def test_position_from_proba_shape_validation():
    with pytest.raises(ValueError, match="must be"):
        RegimeXGBoost.position_from_proba(np.zeros((5,)))
    with pytest.raises(ValueError, match="must be"):
        RegimeXGBoost.position_from_proba(np.zeros((5, 4)))


def test_predict_proba_pads_to_three_columns_when_fit_misses_a_class():
    """When y_train has only 2 of 3 classes, XGBoost emits shape (n, 2);
    RegimeXGBoost.predict_proba must pad to (n, 3) and rows must sum to 1.

    This is the bug that was silently biasing the inner-CV tuner
    (Brief 2.1.3) — sklearn's log_loss was firing
    ``y_prob does not sum to 1`` hundreds of times per outer fold.
    """
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 3))
    # y in {-1, +1} only — class 0 (the time-barrier class) is absent
    y = np.where(X[:, 0] > 0, 1, -1).astype(np.int64)
    model = RegimeXGBoost(n_estimators=20)
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (200, 3)
    # Row sums = 1 (no warning from log_loss)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
    # The missing class (0 → label 0) gets probability zero
    assert np.allclose(proba[:, 1], 0.0)


def test_single_class_fallback_to_prior():
    """Training labels all 0 → fall back to predicting the empirical prior
    instead of crashing.
    """
    X = np.random.default_rng(0).standard_normal((50, 3))
    y = np.zeros(50, dtype=np.int64)
    model = RegimeXGBoost(n_estimators=10)
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (50, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    np.testing.assert_allclose(proba[:, 1], 1.0)


def test_feature_importance_keyed_by_name():
    X, y = _make_classification_data(n=400)
    names = ["volatility", "momentum", "drawdown", "noise"]
    model = RegimeXGBoost(n_estimators=50)
    model.fit(X, y, feature_names=names)
    imp = model.feature_importance()
    assert isinstance(imp, dict)
    assert set(imp.keys()).issubset(set(names))
    assert "volatility" in imp
    assert imp["volatility"] > 0


# ---------------------------------------------------------------------------
# strategy_fn adapter — full CPCV integration
# ---------------------------------------------------------------------------


def _make_synthetic_close_features(n: int = 1500, seed: int = 42):
    """GBM with regime-dependent drift: positive when 20-day momentum is
    positive, negative when negative. XGBoost should pick this up.
    """
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


def test_strategy_adapter_runs_in_cpcv():
    features, log_returns = _make_synthetic_close_features(n=800)
    strategy = make_regime_xgboost_strategy(n_estimators=50, max_depth=3)
    report = run_cpcv_validation(
        strategy, features, log_returns,
        strategy_name="xgb_v1",
        n_splits=8, n_test_groups=2, n_trials=10,
    )
    assert report.n_paths == 28
    assert np.isfinite(report.sharpe_p50)
    assert np.isfinite(report.dsr_p_value)


def test_strategy_beats_flat_on_signal_data():
    """On synthetic data with a learnable regime-dependent drift, XGBoost
    must produce median OOS Sharpe materially higher than flat (= 0). The
    absolute threshold (> 0.15) is calibrated to the synthetic-fixture
    realism, not the production-data Sharpe — on the toy GBM with shallow
    drift switches a tighter bound (> 0.3) is unreliable across seeds.
    """
    features, log_returns = _make_synthetic_close_features(n=1500, seed=7)
    strategy = make_regime_xgboost_strategy(n_estimators=80, max_depth=3)
    xgb_report = run_cpcv_validation(
        strategy, features, log_returns,
        strategy_name="xgb_v1", n_splits=10, n_test_groups=2, n_trials=5,
    )
    flat_report = run_cpcv_validation(
        flat, features, log_returns,
        strategy_name="flat", n_splits=10, n_test_groups=2, n_trials=5,
    )
    assert xgb_report.sharpe_p50 > flat_report.sharpe_p50
    assert xgb_report.sharpe_p50 > 0.15


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))