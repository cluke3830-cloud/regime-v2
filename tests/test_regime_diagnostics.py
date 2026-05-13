"""Tests for src.validation.regime_diagnostics + src.regime.gmm_hmm.select_hmm_k."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.validation.regime_diagnostics import (  # noqa: E402
    nber_alignment,
    reliability_diagram,
    regime_stability,
    cross_model_concordance,
    _cohen_kappa,
)


# ---------------------------------------------------------------------------
# regime_stability
# ---------------------------------------------------------------------------


def test_stability_constant_label():
    """All Bull → 1 episode, mean duration = n_bars, flip rate = 0."""
    idx = pd.date_range("2020-01-01", periods=100, freq="B")
    labels = pd.Series(np.zeros(100, dtype=int), index=idx)
    stab = regime_stability(labels)
    assert stab["flip_rate"] == 0.0
    assert stab["per_regime"][0]["n_episodes"] == 1
    assert stab["per_regime"][0]["mean_duration_bars"] == 100.0
    assert stab["per_regime"][0]["pct_time"] == 1.0
    assert stab["dominant_regime"] == 0


def test_stability_alternating_labels():
    """Alternating 0/1 → flip rate near 1, mean duration = 1."""
    idx = pd.date_range("2020-01-01", periods=100, freq="B")
    labels = pd.Series(np.arange(100) % 2, index=idx)
    stab = regime_stability(labels)
    # 99 transitions / 99 bar-gaps = 1.0
    assert stab["flip_rate"] == pytest.approx(1.0)
    assert stab["per_regime"][0]["mean_duration_bars"] == 1.0
    assert stab["per_regime"][1]["mean_duration_bars"] == 1.0


def test_stability_known_runs():
    """[0]*30 + [1]*20 + [2]*50 → 3 episodes, distinct durations."""
    idx = pd.date_range("2020-01-01", periods=100, freq="B")
    labels = pd.Series([0]*30 + [1]*20 + [2]*50, index=idx)
    stab = regime_stability(labels)
    assert stab["per_regime"][0]["mean_duration_bars"] == 30.0
    assert stab["per_regime"][1]["mean_duration_bars"] == 20.0
    assert stab["per_regime"][2]["mean_duration_bars"] == 50.0
    # 2 transitions over 99 gaps
    assert stab["flip_rate"] == pytest.approx(2 / 99, abs=1e-4)


# ---------------------------------------------------------------------------
# reliability_diagram
# ---------------------------------------------------------------------------


def test_calibration_perfect():
    """Indicator probs matching the actual labels → ECE = 0."""
    n = 300
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3, size=n)
    proba = np.zeros((n, 3))
    for i, lbl in enumerate(labels):
        proba[i, lbl] = 1.0
    cal = reliability_diagram(proba, labels, n_bins=10)
    assert cal["mean_ece"] == pytest.approx(0.0, abs=1e-9)


def test_calibration_uniform_random():
    """Uniform random probs → ECE bounded but non-zero."""
    n = 500
    rng = np.random.default_rng(42)
    labels = rng.integers(0, 3, size=n)
    proba = rng.dirichlet(np.ones(3), size=n)
    cal = reliability_diagram(proba, labels, n_bins=10)
    assert 0.0 <= cal["mean_ece"] <= 0.5
    assert len(cal["ece_per_class"]) == 3


def test_calibration_returns_bin_data():
    """Bin data is returned for plotting."""
    n = 200
    rng = np.random.default_rng(1)
    proba = rng.dirichlet(np.ones(3), size=n)
    labels = rng.integers(0, 3, size=n)
    cal = reliability_diagram(proba, labels, n_bins=5)
    assert len(cal["bins"]) == 3
    for class_bins in cal["bins"]:
        assert len(class_bins["bin_centers"]) == 5
        assert len(class_bins["bin_accuracy"]) == 5


# ---------------------------------------------------------------------------
# cross_model_concordance
# ---------------------------------------------------------------------------


def test_concordance_perfect_agreement():
    """When all three models agree, consensus = 1.0."""
    idx = pd.date_range("2020-01-01", periods=100, freq="B")
    rule = pd.Series([0]*40 + [1]*30 + [2]*30, index=idx)
    gmm = rule.copy()
    # TVTP must map back to rule's labels: low_vol → 0/1, high_vol → 2
    p_high = pd.Series([0.0]*40 + [0.0]*30 + [1.0]*30, index=idx)
    p_low = 1 - p_high
    tvtp = pd.DataFrame({"p_low_vol": p_low, "p_high_vol": p_high})
    conc = cross_model_concordance(rule, gmm, tvtp)
    assert conc["rule_gmm_agreement"] == 1.0
    assert conc["consensus_score"] == 1.0
    assert conc["cohen_kappa_rule_gmm"] == pytest.approx(1.0, abs=1e-9)


def test_concordance_complete_disagreement():
    """When rule says 0 and GMM says 2 everywhere, agreement = 0."""
    idx = pd.date_range("2020-01-01", periods=100, freq="B")
    rule = pd.Series([0]*100, index=idx)
    gmm = pd.Series([2]*100, index=idx)
    tvtp = pd.DataFrame({
        "p_low_vol": [0.5]*100,
        "p_high_vol": [0.5]*100,
    }, index=idx)
    conc = cross_model_concordance(rule, gmm, tvtp)
    assert conc["rule_gmm_agreement"] == 0.0
    # Cohen's κ for constant disagreement (both single-valued, different): special case
    # p_o = 0, p_e = 0 (since prob(rule==2)=0), so result is 0 or undefined
    assert conc["cohen_kappa_rule_gmm"] <= 0.0


def test_concordance_confusion_matrix_shape():
    """Confusion is 3×3."""
    idx = pd.date_range("2020-01-01", periods=50, freq="B")
    rule = pd.Series([0, 1, 2] * 16 + [0, 1], index=idx)
    gmm = pd.Series([1, 0, 2] * 16 + [0, 1], index=idx)
    tvtp = pd.DataFrame({"p_low_vol": [0.7]*50, "p_high_vol": [0.3]*50}, index=idx)
    conc = cross_model_concordance(rule, gmm, tvtp)
    assert len(conc["confusion_rule_gmm"]) == 3
    assert all(len(row) == 3 for row in conc["confusion_rule_gmm"])
    # Counts sum to n_aligned
    total = sum(sum(row) for row in conc["confusion_rule_gmm"])
    assert total == conc["n_aligned"]


def test_cohen_kappa_chance_agreement():
    """Cohen's κ ≈ 0 when agreement is at chance level."""
    rng = np.random.default_rng(0)
    a = rng.integers(0, 3, size=1000)
    b = rng.integers(0, 3, size=1000)
    k = _cohen_kappa(a, b, n_classes=3)
    assert abs(k) < 0.1  # Random labels — chance agreement


# ---------------------------------------------------------------------------
# nber_alignment — only tests the graceful-failure path (no FRED in CI).
# ---------------------------------------------------------------------------


def test_nber_alignment_missing_key():
    """Without a FRED key, function returns usrec_available=False, no crash."""
    idx = pd.date_range("2020-01-01", periods=50, freq="B")
    labels = pd.Series([0]*30 + [2]*20, index=idx)
    # Pass an obviously-bad key to force a graceful fail
    result = nber_alignment(labels, fred_api_key="not_a_real_key_xxxxxxxx")
    # Either it errored gracefully OR (rarely) FRED accepted unknown key as anon
    assert "usrec_available" in result


# ---------------------------------------------------------------------------
# select_hmm_k
# ---------------------------------------------------------------------------


def test_select_hmm_k_returns_table():
    """K-sweep returns valid entries for at least K=2 with correct parameter count.

    Higher K (4, 5) may fail to fit on a short synthetic series due to
    singular covariance — the function should skip those Ks gracefully
    rather than crash. That's a legitimate finding the validation report
    will surface.
    """
    from src.regime.gmm_hmm import select_hmm_k
    rng = np.random.default_rng(7)
    n = 1000
    # 3-regime synthetic: low/mid/high vol
    rets = np.concatenate([
        rng.normal(0.001, 0.005, size=n // 3),
        rng.normal(0.0, 0.012, size=n // 3),
        rng.normal(-0.002, 0.025, size=n - 2 * (n // 3)),
    ])
    close = pd.Series(np.exp(rets).cumprod(),
                      index=pd.date_range("2010-01-01", periods=n, freq="B"))
    results = select_hmm_k(close, k_range=(2, 5), n_iter=100)
    # K=2 must always succeed on a 1000-bar series
    assert 2 in results, f"K=2 must fit, got {list(results.keys())}"
    for k, r in results.items():
        assert {"aic", "bic", "log_likelihood", "n_params", "n_obs"} <= r.keys()
        # Formula (full cov, d=2): (K-1) init + K(K-1) trans + 2K means + 3K cov
        # = K² + 5K - 1
        assert r["n_params"] == k * k + 5 * k - 1
        assert r["n_obs"] > 0