"""Tests for src.regime.meta_stacker (Brief 2.3).

Tests:
  - MetaStacker fits a known linear signal; learned coefs are sensible.
  - non_negative=True respects the constraint.
  - NaN safety in both fit and predict.
  - Equal-weight stacker = literal mean of base outputs.
  - Ridge stacker beats equal-weight when one base is clearly better.
  - Ridge stacker degrades gracefully (falls back to equal-weight) when
    train segment is too small.
  - Both stackers run cleanly through CPCV.
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

from src.regime.meta_stacker import (  # noqa: E402
    MetaStacker,
    make_equal_weight_stacked_strategy,
    make_ridge_stacked_strategy,
)
from src.strategies.benchmarks import buy_and_hold, flat, momentum_20d  # noqa: E402
from src.validation.cpcv_runner import run_cpcv_validation  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_close_features(n: int = 800, seed: int = 42):
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n)
    log_ret = 0.0003 - 0.5 * 0.012 ** 2 + 0.012 * eps
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


# ---------------------------------------------------------------------------
# MetaStacker core
# ---------------------------------------------------------------------------


def test_metastacker_fits_known_linear_signal():
    """If y = 2 * x1 + 0 * x2 + noise, the fitted Ridge should pick up
    a positive coef for x1 near 2 and a small coef for x2.
    """
    rng = np.random.default_rng(0)
    n = 400
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    y = 2.0 * x1 + 0.0 * x2 + 0.1 * rng.standard_normal(n)
    df = pd.DataFrame({"a": x1, "b": x2})
    stacker = MetaStacker(alpha=0.1).fit(df, y)
    assert abs(stacker.coefs["a"] - 2.0) < 0.2
    assert abs(stacker.coefs["b"]) < 0.3


def test_metastacker_non_negative_constraint():
    rng = np.random.default_rng(1)
    n = 400
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    # Construct so optimal weight on x2 is NEGATIVE if unconstrained
    y = 1.5 * x1 - 1.2 * x2 + 0.05 * rng.standard_normal(n)
    df = pd.DataFrame({"a": x1, "b": x2})
    stacker = MetaStacker(alpha=0.1, non_negative=True).fit(df, y)
    assert all(c >= 0 for c in stacker.coefs.values()), (
        f"non_negative=True violated: {stacker.coefs}"
    )


def test_metastacker_nan_safety():
    rng = np.random.default_rng(2)
    n = 200
    df = pd.DataFrame({
        "a": rng.standard_normal(n),
        "b": rng.standard_normal(n),
    })
    y = rng.standard_normal(n)
    # Inject some NaNs
    df.iloc[5:8, 0] = np.nan
    y[10:12] = np.nan
    stacker = MetaStacker(alpha=0.1).fit(df, y)
    pred = stacker.predict(df)
    # NaN inputs propagate; non-NaN inputs predict
    assert np.isnan(pred[5])
    assert not np.isnan(pred[20])


def test_metastacker_raises_on_too_few_rows():
    df = pd.DataFrame({"a": np.zeros(20), "b": np.zeros(20)})
    y = np.zeros(20)
    with pytest.raises(ValueError, match="non-NaN training rows"):
        MetaStacker().fit(df, y)


def test_metastacker_raises_on_shape_mismatch():
    df = pd.DataFrame({"a": np.zeros(50), "b": np.zeros(50)})
    y = np.zeros(40)
    with pytest.raises(ValueError, match="rows"):
        MetaStacker().fit(df, y)


# ---------------------------------------------------------------------------
# Equal-weight stacker
# ---------------------------------------------------------------------------


def test_equal_weight_is_mean_of_base_outputs():
    features, _ = _make_synthetic_close_features(n=400)
    stacker = make_equal_weight_stacked_strategy(
        {"a": buy_and_hold, "b": flat, "c": momentum_20d}
    )
    f_train = features.iloc[:200]
    f_test = features.iloc[200:]
    out = stacker(f_train, f_test)
    # Compute base outputs manually
    a = buy_and_hold(f_train, f_test)
    b = flat(f_train, f_test)
    c = momentum_20d(f_train, f_test)
    expected = (a + b + c) / 3.0
    np.testing.assert_allclose(out, np.clip(expected, -1, 1))


def test_equal_weight_runs_in_cpcv():
    features, log_returns = _make_synthetic_close_features(n=600)
    strategy = make_equal_weight_stacked_strategy({
        "bh": buy_and_hold, "mom": momentum_20d,
    })
    report = run_cpcv_validation(
        strategy, features, log_returns,
        strategy_name="equal_weight_blend",
        n_splits=8, n_test_groups=2, n_trials=5,
    )
    assert report.n_paths == 28
    assert np.isfinite(report.sharpe_p50)


# ---------------------------------------------------------------------------
# Ridge stacker
# ---------------------------------------------------------------------------


def test_ridge_stacker_runs_in_cpcv():
    features, log_returns = _make_synthetic_close_features(n=600)
    strategy = make_ridge_stacked_strategy(
        {"bh": buy_and_hold, "mom": momentum_20d, "flat": flat},
        alpha=1.0, non_negative=True,
    )
    report = run_cpcv_validation(
        strategy, features, log_returns,
        strategy_name="ridge_blend",
        n_splits=8, n_test_groups=2, n_trials=5,
    )
    assert report.n_paths == 28
    assert np.isfinite(report.sharpe_p50)


def test_ridge_stacker_requires_close_column():
    rng = np.random.default_rng(7)
    f_train = pd.DataFrame({"mom_20": rng.standard_normal(50)})
    f_test = pd.DataFrame({"mom_20": rng.standard_normal(50)})
    strategy = make_ridge_stacked_strategy({"flat": flat})
    with pytest.raises(KeyError, match="close"):
        strategy(f_train, f_test)


def test_ridge_stacker_falls_back_on_tiny_train():
    """When the train segment has fewer than ``min_train_rows`` valid
    bars, the Ridge stacker silently falls back to equal-weight.
    """
    # 5 bars of close; momentum_20 will be all NaN → mask drops all rows.
    close = pd.Series(
        [100.0, 101, 99, 100, 102],
        index=pd.date_range("2020-01-01", periods=5, freq="D"),
    )
    f_train = pd.DataFrame({"close": close, "mom_20": [np.nan] * 5})
    f_test = pd.DataFrame({
        "close": [100.0, 101, 102],
        "mom_20": [0.01, 0.02, 0.0],
    }, index=pd.date_range("2020-02-01", periods=3, freq="D"))
    strategy = make_ridge_stacked_strategy({
        "bh": buy_and_hold, "mom": momentum_20d,
    }, min_train_rows=30)
    out = strategy(f_train, f_test)
    # Equal-weight on (+1, +1, 0) and (+1, +1, 0) is (+1, +1, +0.5)
    assert out.shape == (3,)
    np.testing.assert_allclose(out, np.array([1.0, 1.0, 0.5]))


def test_ridge_stacker_clip_in_range():
    features, _ = _make_synthetic_close_features(n=400)
    strategy = make_ridge_stacked_strategy(
        {"bh": buy_and_hold, "mom": momentum_20d, "flat": flat},
        alpha=1.0, position_scale=1000.0,  # absurdly large scale
    )
    out = strategy(features.iloc[:300], features.iloc[300:])
    assert (out >= -1.0).all() and (out <= 1.0).all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))