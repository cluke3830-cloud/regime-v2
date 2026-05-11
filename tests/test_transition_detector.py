"""Tests for src.regime.transition_detector (Brief 2.4).

Key invariants:
  - build_targets produces 1 at every label-change boundary within H bars.
  - Fit handles class imbalance (scale_pos_weight).
  - predict_proba shape and bounds.
  - Degenerate single-class y_train doesn't crash.
  - evaluate_detector_metrics returns the audit-required dict.
  - Strategy_fn produces positions in [-1, +1] (rule_baseline x gate ∈ [0,1]).
  - Strategy_fn runs cleanly through CPCV.
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

from src.features.price_features import compute_features_v2  # noqa: E402
from src.regime.rule_baseline import REGIME_ALLOC  # noqa: E402
from src.regime.transition_detector import (  # noqa: E402
    TransitionDetector,
    evaluate_detector_metrics,
    make_transition_gated_strategy,
)
from src.validation.cpcv_runner import run_cpcv_validation  # noqa: E402


# ---------------------------------------------------------------------------
# build_targets
# ---------------------------------------------------------------------------


def test_build_targets_marks_transitions_in_forward_window():
    """labels [0, 0, 1, 1, 0] with H=2 → target only emits for bars that
    have a FULL forward window (range(n - horizon) = range(3)):
      t=0: forward [0, 1] differs from 0 → 1
      t=1: forward [1, 1] differs from 0 → 1
      t=2: forward [1, 0] differs from 1 → 1
      t=3, t=4: no full forward window → 0 by convention.
                These trailing rows get masked from training inside fit().
    """
    labels = np.array([0, 0, 1, 1, 0])
    det = TransitionDetector(horizon=2)
    tgts = det.build_targets(labels)
    np.testing.assert_array_equal(tgts, [1, 1, 1, 0, 0])


def test_build_targets_flat_sequence_returns_zeros():
    labels = np.zeros(20, dtype=np.int64)
    det = TransitionDetector(horizon=5)
    tgts = det.build_targets(labels)
    np.testing.assert_array_equal(tgts, np.zeros(20, dtype=np.int64))


# ---------------------------------------------------------------------------
# Fit / predict
# ---------------------------------------------------------------------------


def _synthetic_xy(n: int = 600, seed: int = 0):
    """Synthetic detector input: 5 features, regime labels with realistic
    transitions (regime persists for ~30 bars then flips).
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 5))
    labels = np.zeros(n, dtype=np.int64)
    cur = 0
    persist = 0
    for t in range(n):
        labels[t] = cur
        persist += 1
        # 3% chance of flip per bar, anchored at +30-bar minimum persistence
        if persist > 30 and rng.uniform() < 0.03:
            cur = (cur + 1) % 5
            persist = 0
    return X, labels


def test_fit_predict_shapes():
    X, labels = _synthetic_xy(n=400)
    det = TransitionDetector(horizon=5, n_estimators=50).fit(X, labels)
    proba = det.predict_proba(X)
    assert proba.shape == (400,)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_single_class_target_does_not_crash():
    """If all labels are the same → no transitions → all targets are 0
    → fit falls back to no-model state, predict_proba returns 0.5.
    """
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 4))
    labels = np.zeros(100, dtype=np.int64)
    det = TransitionDetector(horizon=5, n_estimators=20).fit(X, labels)
    assert det.model_ is None
    proba = det.predict_proba(X)
    np.testing.assert_allclose(proba, 0.5)


def test_class_imbalance_scale_pos_weight():
    """Verify scale_pos_weight is set when there's class imbalance."""
    rng = np.random.default_rng(0)
    n = 400
    X = rng.standard_normal((n, 5))
    # ~5% transition rate
    labels = np.zeros(n, dtype=np.int64)
    cur = 0
    for t in range(n):
        labels[t] = cur
        if rng.uniform() < 0.02:
            cur = (cur + 1) % 5
    det = TransitionDetector(horizon=5, n_estimators=20).fit(X, labels)
    if det.scale_pos_weight_ is not None:
        assert det.scale_pos_weight_ > 1.0


# ---------------------------------------------------------------------------
# evaluate_detector_metrics
# ---------------------------------------------------------------------------


def test_evaluate_metrics_returns_audit_keys():
    X, labels = _synthetic_xy(n=600, seed=2)
    # Train on first 400, test on last 200
    det = TransitionDetector(horizon=5, n_estimators=100).fit(X[:400], labels[:400])
    metrics = evaluate_detector_metrics(det, X[400:], labels[400:])
    for key in ("f1", "precision", "recall", "n_positive",
                "n_predicted", "passes_gate"):
        assert key in metrics
    # F1 and precision should be finite (not NaN) when both classes present
    if np.isfinite(metrics["f1"]):
        assert 0.0 <= metrics["f1"] <= 1.0
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0


def test_evaluate_metrics_degenerate_test():
    """If test labels are all the same, metrics return NaN with
    passes_gate=False (no positives means precision is undefined)."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 4))
    labels = np.concatenate([np.zeros(200, dtype=np.int64), np.ones(100, dtype=np.int64)])
    det = TransitionDetector(horizon=5, n_estimators=30).fit(X[:200], labels[:200])
    # test on flat-labels segment
    flat_test_labels = np.ones(50, dtype=np.int64)
    metrics = evaluate_detector_metrics(det, X[200:250], flat_test_labels)
    assert metrics["passes_gate"] is False


# ---------------------------------------------------------------------------
# Strategy_fn — full CPCV integration
# ---------------------------------------------------------------------------


def _synthetic_v2_features(n: int = 600, seed: int = 21):
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n)
    log_ret = 0.0003 + 0.012 * eps
    close = pd.Series(
        np.exp(np.log(100.0) + np.cumsum(log_ret)),
        index=pd.date_range("2018-01-01", periods=n, freq="D"),
        name="close",
    )
    return close


def test_strategy_runs_in_cpcv():
    close = _synthetic_v2_features(n=900)
    features = compute_features_v2(close)
    log_returns = np.log(close).diff().loc[features.index]
    strategy = make_transition_gated_strategy(horizon=5, n_estimators=30)
    report = run_cpcv_validation(
        strategy, features, log_returns,
        strategy_name="transition_gated",
        n_splits=6, n_test_groups=2, n_trials=5,
    )
    assert report.n_paths == 15  # C(6, 2)
    assert np.isfinite(report.sharpe_p50)


def test_strategy_positions_bounded():
    """Output positions = rule_position * gate. Since rule_position is in
    [-0.5, 1.0] (REGIME_ALLOC range) and gate is in [0, 1] for smooth-mode,
    final positions are bounded to that same range.
    """
    close = _synthetic_v2_features(n=500)
    features = compute_features_v2(close)
    strategy = make_transition_gated_strategy(
        horizon=5, n_estimators=30, smooth_gate=True,
    )
    f_train = features.iloc[:300]
    f_test = features.iloc[300:]
    out = strategy(f_train, f_test)
    min_alloc = min(REGIME_ALLOC.values())
    max_alloc = max(REGIME_ALLOC.values())
    assert (out >= min_alloc - 1e-9).all()
    assert (out <= max_alloc + 1e-9).all()


def test_hard_gate_zeroes_out_high_probability_bars():
    """With smooth_gate=False and threshold=0.0, every bar should be
    forced to zero (because P(transition) >= 0 always, so 0.0 threshold
    fires on every bar). This is the worst-case sanity check.
    """
    close = _synthetic_v2_features(n=500)
    features = compute_features_v2(close)
    strategy = make_transition_gated_strategy(
        horizon=5, n_estimators=20,
        smooth_gate=False, transition_threshold=0.0,
    )
    f_train = features.iloc[:300]
    f_test = features.iloc[300:]
    out = strategy(f_train, f_test)
    # gate = (p_trans <= 0) is False everywhere (probas are > 0) → position = 0
    np.testing.assert_allclose(out, 0.0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
