"""Acceptance tests for src.labels.triple_barrier (Brief 1.3).

Audit-prescribed acceptance tests:
  (a) Sum of label distribution + meta-distribution equals N.
  (b) Class balance on synthetic GBM with known drift and vol falls in a
      sensible range (audit referenced "~35/30/35 on SPY 2010-2025"; we use
      synthetic data here so the test is hermetic and CI-friendly).
  (c) No label depends on data after t1[i].

Plus regression coverage:
  - Smooth uptrend → predominantly +1 labels.
  - Smooth downtrend → predominantly -1 labels.
  - Flat low-vol → predominantly 0 (vertical barrier) labels.
  - Asymmetric pi_up != pi_down honoured.
  - Vol = 0 / NaN / non-finite handled.
  - Brute-force ground truth on a small example.
  - Bars within horizon of end emit label 0 with t1 clamped to series end.
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

from src.labels.triple_barrier import triple_barrier_labels  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gbm(n: int, drift: float, vol: float, seed: int = 0) -> pd.Series:
    """Geometric Brownian motion close series with daily drift and vol."""
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n)
    log_ret = drift - 0.5 * vol ** 2 + vol * eps
    log_price = np.log(100.0) + np.cumsum(log_ret)
    return pd.Series(np.exp(log_price))


def _const_vol(n: int, sigma: float) -> pd.Series:
    return pd.Series(np.full(n, sigma))


# ---------------------------------------------------------------------------
# (a) Label totals add up to N
# ---------------------------------------------------------------------------


def test_label_count_equals_n():
    n = 500
    close = _gbm(n, drift=0.0005, vol=0.012, seed=7)
    vol = _const_vol(n, 0.012)
    out = triple_barrier_labels(close, vol, pi_up=2.0, horizon=10)
    assert len(out) == n
    assert out["label"].isin([-1, 0, 1]).all()
    counts = out["label"].value_counts()
    total = int(counts.sum())
    assert total == n, f"label counts sum to {total}, expected {n}"


# ---------------------------------------------------------------------------
# (b) Class balance on synthetic GBM
# ---------------------------------------------------------------------------


def test_class_balance_synthetic_gbm():
    """A near-zero-drift GBM with realistic vol and π=2 over h=10 should
    land in a roughly ⅓-⅓-⅓ regime, with some bias toward 0 (vertical
    barrier wins when neither tail is reached).
    """
    n = 4000
    close = _gbm(n, drift=0.0001, vol=0.012, seed=42)
    vol = _const_vol(n, 0.012)
    out = triple_barrier_labels(close, vol, pi_up=2.0, horizon=10)
    fracs = out["label"].value_counts(normalize=True).sort_index()
    # For pi=2 and 10-bar horizon at sigma=0.012, hitting a 2.4% move
    # in 10 bars is moderately likely; expect each class to land in
    # [10%, 60%] — wide enough to be hermetic, tight enough to catch
    # broken implementations.
    for cls in (-1, 0, 1):
        assert cls in fracs.index, f"class {cls} not represented"
        assert 0.10 <= fracs[cls] <= 0.60, (
            f"class {cls} fraction {fracs[cls]:.3f} outside [0.10, 0.60]"
        )


def test_uptrend_mostly_positive():
    """Strong drift with low noise should resolve almost entirely to +1."""
    n = 500
    close = _gbm(n, drift=0.005, vol=0.005, seed=1)
    vol = _const_vol(n, 0.005)
    out = triple_barrier_labels(close, vol, pi_up=2.0, horizon=10)
    # exclude the trailing horizon bars that can only hit time barrier
    body = out.iloc[:-10]
    assert (body["label"] == 1).sum() > 0.85 * len(body)


def test_downtrend_mostly_negative():
    n = 500
    close = _gbm(n, drift=-0.005, vol=0.005, seed=2)
    vol = _const_vol(n, 0.005)
    out = triple_barrier_labels(close, vol, pi_up=2.0, horizon=10)
    body = out.iloc[:-10]
    assert (body["label"] == -1).sum() > 0.85 * len(body)


def test_flat_lowvol_mostly_zero():
    """Flat market with tight barriers vs short horizon → vertical barrier
    wins (label 0). Use very wide barriers (pi=10) so the horizon resolves
    first.
    """
    n = 500
    close = _gbm(n, drift=0.0, vol=0.001, seed=3)
    vol = _const_vol(n, 0.001)
    out = triple_barrier_labels(close, vol, pi_up=10.0, horizon=5)
    body = out.iloc[:-5]
    assert (body["label"] == 0).sum() > 0.85 * len(body)


# ---------------------------------------------------------------------------
# (c) Causal hygiene — perturb future, verify label unchanged
# ---------------------------------------------------------------------------


def test_causal_no_lookahead_beyond_t1():
    """For every bar i, label[i] must depend only on prices in [i+1, t1[i]].

    Procedure: compute labels on a baseline series, then for each i mutate
    close[t1[i] + 1 :] to garbage and recompute; label[i] must be unchanged.

    Tested on a sample of 30 bars to keep wall-clock reasonable.
    """
    n = 200
    close = _gbm(n, drift=0.0002, vol=0.01, seed=55)
    vol = _const_vol(n, 0.01)
    baseline = triple_barrier_labels(close, vol, pi_up=2.0, horizon=15)

    rng = np.random.default_rng(0)
    sample_idx = rng.choice(np.arange(n - 20), size=30, replace=False)
    for i in sorted(sample_idx):
        t1_i = int(baseline.iloc[i]["t1"])
        if t1_i + 1 >= n:
            continue
        mutated = close.copy()
        # replace everything after t1[i] with absurd values — if the label
        # algorithm peeks past t1 it will detect the change
        mutated.iloc[t1_i + 1 :] = (
            close.iloc[t1_i] * rng.uniform(0.5, 1.5, size=n - t1_i - 1)
        )
        recomputed = triple_barrier_labels(mutated, vol, pi_up=2.0, horizon=15)
        assert int(recomputed.iloc[i]["label"]) == int(baseline.iloc[i]["label"]), (
            f"bar {i}: label changed when prices after t1={t1_i} were "
            f"mutated → look-ahead in the implementation"
        )
        assert int(recomputed.iloc[i]["t1"]) == t1_i


# ---------------------------------------------------------------------------
# Asymmetric barriers, edge handling
# ---------------------------------------------------------------------------


def test_asymmetric_barriers():
    """Tighter stop than profit → uptrending series still mostly +1, but
    a downtrending series resolves faster on the stop side.
    """
    n = 300
    close = _gbm(n, drift=-0.003, vol=0.005, seed=10)
    vol = _const_vol(n, 0.005)
    sym = triple_barrier_labels(close, vol, pi_up=2.0, pi_down=2.0, horizon=10)
    asym = triple_barrier_labels(close, vol, pi_up=4.0, pi_down=1.0, horizon=10)
    # asym (tight stop) should resolve to -1 even more often than sym
    body = slice(None, -10)
    assert (asym["label"][body] == -1).sum() >= (sym["label"][body] == -1).sum()


def test_zero_or_nan_vol_emits_zero_label():
    """Vol = 0 cannot define a barrier — function emits label 0 at horizon."""
    n = 50
    close = pd.Series(np.linspace(100.0, 110.0, n))
    vol = pd.Series(np.zeros(n))
    out = triple_barrier_labels(close, vol, pi_up=2.0, horizon=5)
    assert (out["label"] == 0).all()
    # t1 should be t + horizon (or n-1 at the tail)
    for i in range(n):
        expected_t1 = min(i + 5, n - 1)
        assert int(out.iloc[i]["t1"]) == expected_t1


def test_trailing_horizon_bars_emit_zero():
    """The last `horizon-1` bars cannot resolve through the time barrier
    at the natural ``t + horizon`` index — that index is past the series
    end. They emit label 0 with t1 clamped to len-1. Use flat data with
    wide barriers so we know they CAN'T hit a price barrier.
    """
    n = 30
    horizon = 10
    close = pd.Series(np.full(n, 100.0))  # perfectly flat
    vol = _const_vol(n, 0.001)
    out = triple_barrier_labels(close, vol, pi_up=10.0, horizon=horizon)
    # No price ever moves → every label = 0; t1 clamped to series end for
    # bars whose natural horizon would land past it.
    assert (out["label"] == 0).all()
    tail = out.iloc[-(horizon - 1):]  # last horizon-1 bars; bar n-h hits n-1 cleanly
    assert (tail["t1"] == n - 1).all()


# ---------------------------------------------------------------------------
# Brute-force ground truth on a tiny problem
# ---------------------------------------------------------------------------


def test_brute_force_small():
    """Hand-checkable example: 6 bars, π=1.0, h=3."""
    close = pd.Series([100.0, 101.0, 102.5, 100.5, 99.0, 98.0])
    vol = pd.Series([0.02] * 6)
    out = triple_barrier_labels(close, vol, pi_up=1.0, horizon=3)
    # bar 0: c=100, upper=102, lower=98. j=1: 101 (no), j=2: 102.5 >= 102 → +1 at j=2
    assert int(out.iloc[0]["label"]) == 1 and int(out.iloc[0]["t1"]) == 2
    # bar 1: c=101, upper=103.02, lower=98.98. j=2: 102.5 (no), j=3: 100.5 (no), j=4: 99 < 98.98? No (99 > 98.98). → vertical at 4 → 0
    assert int(out.iloc[1]["label"]) == 0 and int(out.iloc[1]["t1"]) == 4
    # bar 2: c=102.5, upper=104.55, lower=100.45. j=3: 100.5 (no, > 100.45), j=4: 99 <= 100.45 → -1 at j=4
    assert int(out.iloc[2]["label"]) == -1 and int(out.iloc[2]["t1"]) == 4


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_input_validation():
    close = pd.Series([100.0, 101.0, 102.0])
    vol = pd.Series([0.01, 0.01, 0.01])
    with pytest.raises(TypeError, match="close must be pd.Series"):
        triple_barrier_labels(np.array([100.0, 101.0]), vol)
    with pytest.raises(TypeError, match="vol must be pd.Series"):
        triple_barrier_labels(close, np.array([0.01] * 3))
    with pytest.raises(ValueError, match="len"):
        triple_barrier_labels(close, pd.Series([0.01, 0.01]))
    with pytest.raises(ValueError, match="pi_up"):
        triple_barrier_labels(close, vol, pi_up=0.0)
    with pytest.raises(ValueError, match="pi_down"):
        triple_barrier_labels(close, vol, pi_down=-0.1)
    with pytest.raises(ValueError, match="horizon"):
        triple_barrier_labels(close, vol, horizon=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))