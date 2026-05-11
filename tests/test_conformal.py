"""Tests for src.regime.conformal (Brief 4.2)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.regime.conformal import (  # noqa: E402
    AdaptiveConformal,
    make_conformal_calibrated_strategy,
    regime_xgboost_proba_fn,
)


# ---------------------------------------------------------------------------
# AdaptiveConformal core
# ---------------------------------------------------------------------------


def test_constructor_validation():
    with pytest.raises(ValueError, match="alpha"):
        AdaptiveConformal(alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        AdaptiveConformal(alpha=1.0)
    with pytest.raises(ValueError, match="gamma"):
        AdaptiveConformal(gamma=0.0)
    with pytest.raises(ValueError, match="window"):
        AdaptiveConformal(window=10)


def test_predict_set_shape():
    cal = AdaptiveConformal(alpha=0.1)
    p_hat = np.array([0.2, 0.5, 0.3])
    pred_set, p_cal = cal.update_and_predict(p_hat, y_true=1)
    assert pred_set.shape == (3,)
    assert p_cal.shape == (3,)
    assert np.isclose(p_cal.sum(), 1.0)


def test_warmup_includes_all_classes():
    """In warm-up (< 30 scores), prediction set includes EVERY class."""
    cal = AdaptiveConformal(alpha=0.1)
    p_hat = np.array([0.05, 0.05, 0.90])
    pred_set, _ = cal.update_and_predict(p_hat, y_true=2)
    # All-True during warm-up
    assert pred_set.all()


def test_alpha_t_drift_under_distribution_shift():
    """Update rule: ``alpha_t += gamma * (alpha - I(miss))``.

    When labels are EASY (we cover often, miss < alpha), the update is
    POSITIVE → alpha_t drifts UP (toward 1 — wider acceptance OK).
    When labels are ADVERSARIAL (we miss often, miss > alpha), the
    update is NEGATIVE → alpha_t drifts DOWN (toward 0 — calibrator
    forces a higher quantile cutoff, widening the prediction SET).
    """
    rng = np.random.default_rng(0)
    cal = AdaptiveConformal(alpha=0.1, gamma=0.05)
    for _ in range(200):
        p_hat = rng.dirichlet([1.0, 1.0, 1.0])
        cal.update_and_predict(p_hat, y_true=int(np.argmax(p_hat)))
    alpha_after_easy = cal.alpha_t

    for _ in range(200):
        p_hat = rng.dirichlet([1.0, 1.0, 1.0])
        cal.update_and_predict(p_hat, y_true=int(np.argmin(p_hat)))
    alpha_after_hard = cal.alpha_t

    # Adversarial regime → alpha_t DROPS materially below the easy-regime
    # value. The DIRECTION of drift matters, not just monotone.
    assert alpha_after_hard < alpha_after_easy, (
        f"alpha_t should drop under adversarial labels: "
        f"easy={alpha_after_easy:.3f}, hard={alpha_after_hard:.3f}"
    )


def test_long_run_coverage_near_target():
    """Over a long run on well-calibrated probabilities, the empirical
    coverage should be near 1 - alpha.
    """
    rng = np.random.default_rng(5)
    alpha = 0.10
    cal = AdaptiveConformal(alpha=alpha, gamma=0.01)
    n_runs = 1500
    n_covered = 0
    for _ in range(n_runs):
        p_hat = rng.dirichlet([2.0, 2.0, 2.0])
        # Sample y_true ~ Categorical(p_hat) — perfectly calibrated source
        y_true = int(rng.choice(3, p=p_hat))
        pred_set, _ = cal.update_and_predict(p_hat, y_true=y_true)
        if pred_set[y_true]:
            n_covered += 1
    empirical = n_covered / n_runs
    # Target coverage is 0.9; allow ±5% slack on this sample size
    assert 0.85 <= empirical <= 0.97, (
        f"empirical coverage {empirical:.3f} too far from target 0.9"
    )


def test_calibrated_probs_renorm_to_one():
    cal = AdaptiveConformal(alpha=0.1)
    # Force out of warm-up by injecting fake scores
    cal.scores_ = [0.5] * 40
    p_hat = np.array([0.6, 0.3, 0.1])
    _, p_cal = cal.update_and_predict(p_hat, y_true=0)
    assert np.isclose(p_cal.sum(), 1.0, atol=1e-9) or p_cal.sum() == 0.0


def test_predict_only_no_state_update():
    cal = AdaptiveConformal(alpha=0.1)
    initial_scores = list(cal.scores_)
    initial_alpha = cal.alpha_t
    p_hat = np.array([0.3, 0.4, 0.3])
    cal.predict_only(p_hat)
    assert cal.scores_ == initial_scores
    assert cal.alpha_t == initial_alpha


# ---------------------------------------------------------------------------
# regime_xgboost_proba_fn + conformal wrapper — CPCV smoke
# ---------------------------------------------------------------------------


def _make_synthetic_features(n: int = 500, seed: int = 0):
    rng = np.random.default_rng(seed)
    log_ret = 0.0003 + 0.012 * rng.standard_normal(n)
    close = pd.Series(
        np.exp(np.log(100.0) + np.cumsum(log_ret)),
        index=pd.date_range("2018-01-01", periods=n, freq="D"),
        name="close",
    )
    log_returns = np.log(close).diff()
    mom_20 = log_returns.rolling(20).sum().shift(1)
    vol_ewma = log_returns.ewm(alpha=0.06, min_periods=20).std()
    features = pd.DataFrame({
        "close": close, "mom_20": mom_20, "vol_ewma": vol_ewma,
    }).dropna()
    return features, log_returns.loc[features.index]


def test_conformal_strategy_runs_in_cpcv():
    from src.validation.cpcv_runner import run_cpcv_validation

    features, log_returns = _make_synthetic_features(n=600)
    base_proba = regime_xgboost_proba_fn(n_estimators=30, max_depth=3)
    strategy = make_conformal_calibrated_strategy(
        base_proba_fn=base_proba, alpha=0.10, gamma=0.01,
    )
    report = run_cpcv_validation(
        strategy, features, log_returns,
        strategy_name="conformal_xgb",
        n_splits=5, n_test_groups=2, n_trials=5,
    )
    assert report.n_paths == 10  # C(5, 2)
    assert np.isfinite(report.sharpe_p50)


def test_conformal_positions_in_range():
    features, _ = _make_synthetic_features(n=400)
    base_proba = regime_xgboost_proba_fn(n_estimators=20, max_depth=3)
    strategy = make_conformal_calibrated_strategy(base_proba_fn=base_proba)
    f_train = features.iloc[:280]
    f_test = features.iloc[280:]
    out = strategy(f_train, f_test)
    assert out.shape == (len(f_test),)
    assert (out >= -1.0 - 1e-9).all()
    assert (out <= 1.0 + 1e-9).all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
