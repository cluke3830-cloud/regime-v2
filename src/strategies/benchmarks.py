"""Benchmark strategies — the sanity baselines every regime model must beat.

Each function obeys the :class:`StrategyFn` protocol:

    fn(features_train, features_test) -> positions_test (1-D array)

where ``positions_test`` has length ``len(features_test)`` and contains
positions in {-1, 0, +1} (or fractional values for risk-scaled positions).

Strategies here have no learned state — train data is unused. They exist so
the CPCV harness has something concrete to validate against and so PBO
becomes well-defined with ≥ 2 strategy variants.

Causal hygiene — each position at test bar t uses only features available
through bar t. The harness multiplies positions × bar-returns where
bar-return at t is the close-to-close log return from t-1 to t, so the
position is making a one-bar-ahead bet (which is exactly the buy-on-close-
hold-to-next-close convention).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def buy_and_hold(features_train: pd.DataFrame, features_test: pd.DataFrame) -> np.ndarray:
    """Always long. Returns +1 for every test bar.

    The simplest possible strategy. If your regime model can't beat this
    on a structurally up-trending asset (SPY since 1993), something is
    very wrong.
    """
    return np.ones(len(features_test), dtype=float)


def flat(features_train: pd.DataFrame, features_test: pd.DataFrame) -> np.ndarray:
    """Always flat. Zero risk, zero return. Useful as a Sharpe-= 0 reference."""
    return np.zeros(len(features_test), dtype=float)


def momentum_20d(
    features_train: pd.DataFrame,
    features_test: pd.DataFrame,
    *,
    lookback: int = 20,
    momentum_col: str = "mom_20",
) -> np.ndarray:
    """Long when 20-day momentum is positive, flat otherwise.

    Expects ``features_test`` to contain a ``momentum_col`` column with
    the trailing-20-bar log-return at each bar (causally computed). At
    test bar t the position is ``+1`` if ``mom_20[t] > 0``, else ``0``.

    Falls back to using the train-side last value if the column is
    missing or NaN at test bar — defensive but should not fire in
    practice when features are computed correctly.
    """
    if momentum_col not in features_test.columns:
        raise KeyError(
            f"momentum_20d requires a '{momentum_col}' column in features_test"
        )
    mom = features_test[momentum_col].to_numpy(dtype=float)
    positions = np.where(mom > 0, 1.0, 0.0)
    positions[np.isnan(mom)] = 0.0
    return positions


__all__ = ["buy_and_hold", "momentum_20d", "flat"]