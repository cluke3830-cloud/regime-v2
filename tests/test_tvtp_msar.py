"""Tests for src.baselines.tvtp_msar (Brief 3.1).

Tests:
  - MarkovSwitchingAR fits a synthetic two-regime series.
  - State remap puts low-variance regime at index 0 deterministically.
  - predict_proba returns shape (n, 2) summing to 1.
  - Failed fit (too few obs) falls back to 50/50.
  - Strategy_fn runs through CPCV without crashing.
  - Strategy emits positions in [-0.3, +1.0] (the state_positions range).
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

from src.baselines.tvtp_msar import MarkovSwitchingAR, make_tvtp_msar_strategy  # noqa: E402
from src.validation.cpcv_runner import run_cpcv_validation  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _two_regime_returns(
    n: int = 600, low_vol: float = 0.005, high_vol: float = 0.020,
    p_stay: float = 0.95, seed: int = 0,
) -> pd.Series:
    """Synthetic 2-regime return series. State 0 = low_vol, state 1 = high_vol.

    Markov chain with stay-probability p_stay; each bar's return ~ N(0, vol_state^2).
    """
    rng = np.random.default_rng(seed)
    state = np.zeros(n, dtype=np.int64)
    for t in range(1, n):
        if rng.uniform() < p_stay:
            state[t] = state[t - 1]
        else:
            state[t] = 1 - state[t - 1]
    vols = np.where(state == 0, low_vol, high_vol)
    rets = rng.standard_normal(n) * vols
    return pd.Series(
        rets,
        index=pd.date_range("2018-01-01", periods=n, freq="D"),
        name="returns",
    )


def _synthetic_close_features(n: int = 600, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rets = _two_regime_returns(n=n, seed=seed)
    close = pd.Series(
        np.exp(np.cumsum(rets.to_numpy())),
        index=rets.index, name="close",
    )
    features = pd.DataFrame({"close": close})
    log_ret = np.log(close).diff().fillna(0.0)
    return features, log_ret


# ---------------------------------------------------------------------------
# Core fit / predict
# ---------------------------------------------------------------------------


def test_msar_fits_two_regime_series():
    """The fit should recognise the two distinct variance regimes."""
    rets = _two_regime_returns(n=500, seed=7)
    model = MarkovSwitchingAR(k_regimes=2, order=1).fit(rets)
    assert model.params_ is not None, "expected successful fit on 500-bar 2-regime series"
    assert model.state_remap_ is not None
    assert set(model.state_remap_.values()) == {0, 1}


def test_msar_predict_proba_shape_and_sums():
    rets = _two_regime_returns(n=400, seed=1)
    model = MarkovSwitchingAR().fit(rets)
    probs = model.predict_proba(rets)
    assert isinstance(probs, pd.DataFrame)
    assert set(probs.columns) == {"p_low_vol", "p_high_vol"}
    assert len(probs) == len(rets)
    np.testing.assert_allclose(probs.sum(axis=1).to_numpy(), 1.0, atol=1e-9)
    assert (probs >= 0).all().all() and (probs <= 1).all().all()


def test_msar_falls_back_to_uniform_on_short_input():
    """< 50 obs → no fit attempted, predict_proba returns 0.5/0.5."""
    short = pd.Series(
        np.random.default_rng(0).standard_normal(20) * 0.01,
        index=pd.date_range("2020-01-01", periods=20, freq="D"),
    )
    model = MarkovSwitchingAR().fit(short)
    assert model.params_ is None
    probs = model.predict_proba(short)
    np.testing.assert_allclose(probs["p_low_vol"].to_numpy(), 0.5)
    np.testing.assert_allclose(probs["p_high_vol"].to_numpy(), 0.5)


def test_msar_state0_lower_variance_after_remap():
    """state 0 (low_vol) should dominate during the low-variance segments."""
    rets = _two_regime_returns(n=600, low_vol=0.003, high_vol=0.025, seed=4)
    model = MarkovSwitchingAR().fit(rets)
    probs = model.predict_proba(rets)
    # On bars where the realised vol is small, p_low_vol should be > 0.5 on average
    abs_r = rets.abs()
    low_realised = abs_r < abs_r.median()
    assert probs.loc[low_realised, "p_low_vol"].mean() > 0.5, (
        f"on low-realised-vol bars, p_low_vol should be > 0.5 on average; "
        f"got {probs.loc[low_realised, 'p_low_vol'].mean():.3f}"
    )


# ---------------------------------------------------------------------------
# Strategy_fn
# ---------------------------------------------------------------------------


def test_strategy_runs_in_cpcv():
    features, log_returns = _synthetic_close_features(n=900, seed=21)
    strategy = make_tvtp_msar_strategy()
    report = run_cpcv_validation(
        strategy, features, log_returns,
        strategy_name="tvtp_msar",
        n_splits=6, n_test_groups=2, n_trials=5,
    )
    assert report.n_paths == 15  # C(6, 2)
    assert np.isfinite(report.sharpe_p50)


def test_strategy_positions_in_state_range():
    features, _ = _synthetic_close_features(n=500, seed=11)
    strategy = make_tvtp_msar_strategy(state_positions={0: 1.0, 1: -0.3})
    f_train = features.iloc[:350]
    f_test = features.iloc[350:]
    out = strategy(f_train, f_test)
    # Output is a probability-weighted mean of {1.0, -0.3}, so bounded
    assert (out >= -0.3 - 1e-9).all(), f"out below state 1 allocation: min={out.min()}"
    assert (out <= 1.0 + 1e-9).all(), f"out above state 0 allocation: max={out.max()}"


def test_strategy_requires_close_column():
    rng = np.random.default_rng(0)
    f_train = pd.DataFrame({"vol_ewma": rng.standard_normal(100)})
    f_test = pd.DataFrame({"vol_ewma": rng.standard_normal(50)})
    strategy = make_tvtp_msar_strategy()
    with pytest.raises(KeyError, match="close"):
        strategy(f_train, f_test)


def test_strategy_custom_state_positions():
    """Custom state_positions should be honoured (e.g., short-only defense)."""
    features, _ = _synthetic_close_features(n=400, seed=2)
    strategy = make_tvtp_msar_strategy(state_positions={0: 0.5, 1: -1.0})
    f_train = features.iloc[:280]
    f_test = features.iloc[280:]
    out = strategy(f_train, f_test)
    assert (out >= -1.0 - 1e-9).all()
    assert (out <= 0.5 + 1e-9).all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
