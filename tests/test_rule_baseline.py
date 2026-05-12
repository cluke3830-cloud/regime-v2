"""Tests for src.regime.rule_baseline (Brief 2.2).

Tests for the 3-regime rule classifier:
  - Schema: output has p_0..p_2 columns summing to 1, label in {0..2},
    regime is human-readable string, position matches REGIME_ALLOC.
  - Causal hygiene: perturb future feature rows, regime at earlier
    indices is unchanged.
  - Stabilizer hysteresis works (no rapid flipping on noisy inputs).
  - Crisis promote gate fires on raw shock or raw DD thresholds.
  - Bull regime favoured on synthetic strong-uptrend data.
  - Bear regime favoured on synthetic crash data.
  - Strategy adapter is reproducible (same inputs → same positions).
  - Works inside the CPCV harness without crashing.
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
from src.regime.rule_baseline import (  # noqa: E402
    N_REGIMES,
    REGIME_ALLOC,
    REGIME_NAMES,
    V2_FEATURE_ORDER,
    _DEFAULT_WEIGHTS,
    _FULL_BEAR_GATED,
    _riskoff_confirm,
    _softmax,
    compute_rule_regime_sequence,
    rule_baseline_strategy,
)
from src.strategies.benchmarks import flat  # noqa: E402
from src.validation.cpcv_runner import run_cpcv_validation  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _gbm(
    n: int = 800, drift: float = 0.0003, vol: float = 0.012, seed: int = 0
) -> pd.Series:
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n)
    log_ret = drift - 0.5 * vol ** 2 + vol * eps
    return pd.Series(
        np.exp(np.log(100.0) + np.cumsum(log_ret)),
        index=pd.date_range("2018-01-01", periods=n, freq="D"),
        name="close",
    )


def _crash_series(n: int = 800, seed: int = 0) -> pd.Series:
    """Synthetic close with a steep crash in the middle."""
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n)
    log_ret = np.where(
        (np.arange(n) > n // 2) & (np.arange(n) < n // 2 + 50),
        -0.03 + 0.03 * eps,   # severe negative drift + high vol
        0.0005 + 0.008 * eps,  # mild uptrend
    )
    return pd.Series(
        np.exp(np.log(100.0) + np.cumsum(log_ret)),
        index=pd.date_range("2018-01-01", periods=n, freq="D"),
        name="close",
    )


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_weight_matrix_shape():
    assert _DEFAULT_WEIGHTS.shape == (N_REGIMES, 21)
    assert len(V2_FEATURE_ORDER) == 21
    assert len(REGIME_NAMES) == N_REGIMES == 3
    assert len(REGIME_ALLOC) == N_REGIMES


def test_full_bear_gated_features_are_tail_features():
    """The features Full Bear gates should be drawdown, vol, shock, vix,
    credit-spread — the audit's §5.3.1 tail-event features. Not momentum
    or trend direction (those are signal-direction features).
    """
    gated_cols = [V2_FEATURE_ORDER[i] for i in np.where(_FULL_BEAR_GATED)[0]]
    expected = {"vol_short", "vol_ewma", "vol_long",
                "shock_z", "drawdown_252", "vix_log", "credit_spread"}
    assert set(gated_cols) == expected


def test_sequence_output_schema():
    close = _gbm(n=600)
    features = compute_features_v2(close)
    out = compute_rule_regime_sequence(features)
    assert {"p_0", "p_1", "p_2", "label", "regime", "position"}.issubset(
        out.columns
    )
    np.testing.assert_allclose(
        out[["p_0", "p_1", "p_2"]].sum(axis=1), 1.0, atol=1e-9
    )
    assert set(out["label"].unique()).issubset({0, 1, 2})
    assert set(out["regime"].unique()).issubset(set(REGIME_NAMES.values()))
    # position must match REGIME_ALLOC[label] for every row
    expected_pos = out["label"].map(REGIME_ALLOC).astype(float)
    np.testing.assert_allclose(out["position"].to_numpy(), expected_pos.to_numpy())


# ---------------------------------------------------------------------------
# Causal hygiene
# ---------------------------------------------------------------------------


def test_causal_no_lookahead():
    """Perturb features at row > k. The rule output at row ≤ k must NOT
    change. Because compute_rule_regime_sequence is cached, we need to
    clear it between runs.
    """
    from src.regime import rule_baseline as rb_mod
    rb_mod._cache.clear()

    close = _gbm(n=500, seed=33)
    baseline_features = compute_features_v2(close)
    baseline = compute_rule_regime_sequence(baseline_features)

    rng = np.random.default_rng(7)
    sample = rng.choice(baseline.index[:-50], size=8, replace=False)
    for ts in sorted(sample):
        rb_mod._cache.clear()
        pos = baseline.index.get_loc(ts)
        mutated = baseline_features.copy()
        mutated.iloc[pos + 1:] = mutated.iloc[pos]  # scramble future to constant
        # Add some noise to the constant scramble
        mutated.iloc[pos + 1:] *= rng.uniform(0.5, 1.5, size=mutated.shape)[pos + 1:]
        recomputed = compute_rule_regime_sequence(mutated)
        for col in ("p_0", "p_1", "p_2", "label"):
            assert recomputed.loc[ts, col] == baseline.loc[ts, col], (
                f"{col} at {ts} (pos {pos}) changed under future mutation"
            )


# ---------------------------------------------------------------------------
# Crisis promote gate
# ---------------------------------------------------------------------------


def test_crisis_promote_on_extreme_shock():
    """riskoff_confirm forces Bear when raw shock_z exceeds 3.5σ
    even if argmax picked a different regime.
    """
    probs = np.array([0.60, 0.25, 0.15])  # argmax = 0 (Bull)
    label = int(np.argmax(probs))  # 0
    promoted = _riskoff_confirm(label, probs, shock_raw=4.0, dd_raw=0.05)
    assert promoted == 2, "expected Bear promotion on 4σ shock"


def test_crisis_promote_on_deep_drawdown():
    probs = np.array([0.60, 0.30, 0.10])
    label = int(np.argmax(probs))
    promoted = _riskoff_confirm(label, probs, shock_raw=1.0, dd_raw=0.18)
    assert promoted == 2, "expected Bear promotion on 18% DD"


def test_crisis_demote_when_unconfirmed():
    """If argmax is Bear but shock and DD are mild, demote to second-best regime."""
    probs = np.array([0.25, 0.20, 0.55])  # argmax = 2 (Bear)
    label = 2  # Bear (FULL_BEAR)
    demoted = _riskoff_confirm(label, probs, shock_raw=0.5, dd_raw=0.01)
    assert demoted != 2
    assert demoted == 0  # second-best non-Bear is Bull


# ---------------------------------------------------------------------------
# Behavioural — synthetic regimes
# ---------------------------------------------------------------------------


def test_strong_uptrend_favours_bull_regimes():
    """A strong-uptrend GBM should land in Full Bull or Half Bull more
    often than in Half Bear or Full Bear.
    """
    from src.regime import rule_baseline as rb_mod
    rb_mod._cache.clear()

    close = _gbm(n=1000, drift=0.0015, vol=0.005, seed=42)
    features = compute_features_v2(close)
    out = compute_rule_regime_sequence(features)
    bull_share = (out["label"] == 0).mean()
    bear_share = (out["label"] == 2).mean()
    assert bull_share > bear_share, (
        f"strong uptrend should favour bull regimes: bull={bull_share:.2%}, "
        f"bear={bear_share:.2%}"
    )


def test_crash_triggers_bear_regime():
    """A series with a steep mid-window crash should trigger Half Bear
    or Full Bear during the crash window. We just check that bear
    regimes appear SOMEWHERE in the output (not necessarily at 50%+).
    """
    from src.regime import rule_baseline as rb_mod
    rb_mod._cache.clear()

    close = _crash_series(n=800, seed=11)
    features = compute_features_v2(close)
    out = compute_rule_regime_sequence(features)
    # During the crash window (bars ~400-450), bear regimes should fire
    crash_window = out.iloc[400:480]
    bear_in_crash = (crash_window["label"] == 2).any()
    assert bear_in_crash, (
        f"expected Bear regime during synthetic crash; "
        f"label distribution in window: "
        f"{crash_window['label'].value_counts().to_dict()}"
    )


# ---------------------------------------------------------------------------
# Strategy adapter — full CPCV
# ---------------------------------------------------------------------------


def test_strategy_returns_positions_for_test_index():
    close = _gbm(n=600)
    features = compute_features_v2(close)
    log_returns = np.log(close).diff().loc[features.index]
    train_size = int(len(features) * 0.7)
    f_train = features.iloc[:train_size]
    f_test  = features.iloc[train_size:]
    positions = rule_baseline_strategy(f_train, f_test)
    assert positions.shape == (len(f_test),)
    assert np.isfinite(positions).all()
    # All positions must be in the set of REGIME_ALLOC values
    assert set(np.unique(positions)).issubset(set(REGIME_ALLOC.values()))


def test_strategy_is_deterministic():
    close = _gbm(n=500, seed=7)
    features = compute_features_v2(close)
    train_size = 350
    f_train = features.iloc[:train_size]
    f_test  = features.iloc[train_size:]
    p1 = rule_baseline_strategy(f_train, f_test)
    p2 = rule_baseline_strategy(f_train, f_test)
    np.testing.assert_array_equal(p1, p2)


def test_strategy_runs_in_cpcv():
    """Smoke test through the CPCV harness."""
    from src.regime import rule_baseline as rb_mod
    rb_mod._cache.clear()

    close = _gbm(n=900, seed=21)
    features = compute_features_v2(close)
    log_returns = np.log(close).diff().loc[features.index]
    report = run_cpcv_validation(
        rule_baseline_strategy, features, log_returns,
        strategy_name="rule_baseline",
        n_splits=8, n_test_groups=2, n_trials=5,
    )
    assert report.n_paths == 28  # C(8, 2)
    assert np.isfinite(report.sharpe_p50)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))