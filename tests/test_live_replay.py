"""Tests for src.validation.live_replay (Brief 5.3) AND the
adaptedness invariant applied across every strategy in the project.

This is the audit's §10.12 "Murphy's Law checklist Item 1" — the
single most important test in the entire codebase. If any strategy
fails it, every downstream metric is contaminated.
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
from src.regime.rule_baseline import rule_baseline_strategy  # noqa: E402
from src.strategies.benchmarks import buy_and_hold, flat, momentum_20d  # noqa: E402
from src.validation.live_replay import (  # noqa: E402
    replay_strategy_bar_by_bar,
    verify_no_lookahead,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _synthetic_close(n: int = 500, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    log_ret = 0.0003 + 0.012 * rng.standard_normal(n)
    return pd.Series(
        np.exp(np.log(100.0) + np.cumsum(log_ret)),
        index=pd.date_range("2018-01-01", periods=n, freq="D"),
        name="close",
    )


# ---------------------------------------------------------------------------
# verify_no_lookahead — sanity on simple strategies
# ---------------------------------------------------------------------------


def test_verify_no_lookahead_on_buy_and_hold():
    """buy_and_hold returns ones; perturbations can't affect it."""
    close = _synthetic_close(n=400)
    features = compute_features_v2(close)
    f_train = features.iloc[:300]
    f_test = features.iloc[300:]
    out = verify_no_lookahead(buy_and_hold, f_train, f_test, n_samples=10)
    assert out["passed"] is True
    assert out["max_delta"] == 0.0


def test_verify_no_lookahead_on_flat():
    close = _synthetic_close(n=400)
    features = compute_features_v2(close)
    out = verify_no_lookahead(flat, features.iloc[:300], features.iloc[300:],
                              n_samples=10)
    assert out["passed"] is True
    assert out["max_delta"] == 0.0


def test_verify_no_lookahead_on_momentum_20d():
    """momentum_20d reads mom_20 which is in features_test. Perturbing
    future bars doesn't change the values at earlier bars → must pass.
    """
    close = _synthetic_close(n=500)
    features = compute_features_v2(close)
    out = verify_no_lookahead(
        momentum_20d, features.iloc[:350], features.iloc[350:],
        n_samples=10,
    )
    assert out["passed"] is True, f"violations: {out['violations']}"


def test_verify_no_lookahead_on_rule_baseline():
    """rule_baseline concats train+test internally; verify that mutating
    test[t+1:] doesn't change the position at bar t.
    """
    close = _synthetic_close(n=500)
    features = compute_features_v2(close)
    out = verify_no_lookahead(
        rule_baseline_strategy, features.iloc[:350], features.iloc[350:],
        n_samples=8,
    )
    assert out["passed"] is True, f"violations: {out['violations']}"


# ---------------------------------------------------------------------------
# Negative test — a strategy that DOES peek at the future must FAIL
# ---------------------------------------------------------------------------


def test_verify_no_lookahead_catches_explicit_peek():
    """Construct a deliberately leaky strategy that uses the LAST bar's
    feature value for every prediction. Verify the test catches it.

    Needs ≥ 500 bars of close so compute_features_v2's 252-bar burn-in
    leaves enough rows for a meaningful train/test split.
    """
    close = _synthetic_close(n=700)
    features = compute_features_v2(close)
    n_features = len(features)

    def leaky_strategy(features_train, features_test):
        last_mom = float(features_test["mom_20"].iloc[-1])
        return np.full(len(features_test), last_mom, dtype=float)

    f_train = features.iloc[: int(n_features * 0.7)]
    f_test = features.iloc[int(n_features * 0.7):]
    out = verify_no_lookahead(leaky_strategy, f_train, f_test, n_samples=10)
    assert not out["passed"], "expected verify_no_lookahead to CATCH the leak"
    assert len(out["violations"]) > 0


# ---------------------------------------------------------------------------
# replay_strategy_bar_by_bar — exact equivalence
# ---------------------------------------------------------------------------


def test_replay_matches_batch_for_buy_and_hold():
    close = _synthetic_close(n=700)  # need > 252 + train + test
    features = compute_features_v2(close)
    n_features = len(features)
    f_train = features.iloc[: int(n_features * 0.7)]
    f_test = features.iloc[int(n_features * 0.7):]
    assert len(f_test) > 20, f"f_test too short: {len(f_test)}"
    batch, replayed = replay_strategy_bar_by_bar(
        buy_and_hold, f_train, f_test, max_bars=20,
    )
    # buy_and_hold always returns +1
    assert (batch == 1.0).all()
    valid = ~np.isnan(replayed)
    assert (replayed[valid] == batch[valid]).all()


def test_replay_matches_batch_for_momentum():
    close = _synthetic_close(n=700)
    features = compute_features_v2(close)
    n_features = len(features)
    f_train = features.iloc[: int(n_features * 0.7)]
    f_test = features.iloc[int(n_features * 0.7):]
    batch, replayed = replay_strategy_bar_by_bar(
        momentum_20d, f_train, f_test, max_bars=15,
    )
    valid = ~np.isnan(replayed)
    np.testing.assert_array_equal(replayed[valid], batch[valid])


# ---------------------------------------------------------------------------
# The "every strategy passes" gate
# ---------------------------------------------------------------------------


def test_all_deterministic_strategies_pass_no_lookahead():
    """Run verify_no_lookahead on every deterministic strategy in the
    repo simultaneously. Audit §10.12: this is the gate that flips
    deployment from 'maybe' to 'safe'.
    """
    close = _synthetic_close(n=500, seed=11)
    features = compute_features_v2(close)
    f_train = features.iloc[:350]
    f_test = features.iloc[350:]

    strategies = {
        "buy_and_hold": buy_and_hold,
        "flat": flat,
        "momentum_20d": momentum_20d,
        "rule_baseline": rule_baseline_strategy,
    }
    failures = []
    for name, fn in strategies.items():
        out = verify_no_lookahead(fn, f_train, f_test, n_samples=6)
        if not out["passed"]:
            failures.append(f"{name}: {len(out['violations'])} violations")
    assert not failures, (
        f"strategies failing the no-lookahead gate: {failures}"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
