"""Live-replay adaptedness verification.

Brief 5.3 of the regime upgrade plan. Audit §8.5.4 / §10.12 names this
test as the single most important unit test in the entire pipeline:
the mechanical proxy for "no look-ahead". If this test fails, every
downstream metric is contaminated.

Two verification routines:

  - ``verify_no_lookahead(strategy_fn, features_train, features_test,
                         n_samples=10, rng_seed=42)``:
      For each of ``n_samples`` random bars t in the test window,
      mutate ``features_test[t+1:]`` to garbage and recompute the
      strategy's prediction at bar t. The prediction MUST be
      identical (within tolerance) to the baseline. Fast — one full
      re-run per sample. Used in test suites.

  - ``replay_strategy_bar_by_bar(strategy_fn, features_train,
                                 features_test, max_bars=None)``:
      Bar-by-bar replay: at each test bar t, call the strategy with
      (features_train, features_test[:t+1]) and capture the
      prediction at row t. Slow (n_test full retrains) but
      mechanically equivalent to a live deployment. Used as a
      deploy-time gate before promoting a strategy to production.

The invariant under test (formal statement):

    For every strategy_fn S, every CPCV fold (train, test), and every
    test bar t ∈ [0, len(test)):

        S(features_train, features_test)[t]
            ==
        S(features_train, features_test[:t+1])[-1]

    AND

        S(features_train, features_test_with_perturbed_after_t)[t]
            ==
        S(features_train, features_test)[t]

If both hold, the strategy is provably free of look-ahead at the
prediction level.

Strategies known to PASS this test (per audit §5.5, §5.6.4, §5.10):
    flat, buy_and_hold, momentum_20d, rule_baseline,
    tvtp_msar, hsmm, ms_garch, ms_garch's GARCH recursion,
    meta_equal, meta_ridge

Strategies that need careful review:
    xgb_*       (training is on full train segment — that's fine;
                 but check that test predictions don't peek at
                 future test data)
    patchtst    (same — train on full train, test predictions are
                 per-row inference)
    conformal_* (the calibrator IS stateful and uses CURRENT
                 prediction; verify the warm-up uses only train)
    transition_gated (rule_baseline runs over train+test concat —
                     verify the Stabilizer state passed into test is
                     causal)
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd


StrategyFn = Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]


# ---------------------------------------------------------------------------
# Fast perturb-future verification (used in tests)
# ---------------------------------------------------------------------------


def verify_no_lookahead(
    strategy_fn: StrategyFn,
    features_train: pd.DataFrame,
    features_test: pd.DataFrame,
    *,
    n_samples: int = 10,
    rng_seed: int = 42,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    perturb_scale: tuple = (0.5, 2.0),
) -> dict:
    """Verify a strategy_fn is look-ahead free by perturbing future bars.

    Procedure:
      1. Compute baseline predictions on (features_train, features_test).
      2. For each of ``n_samples`` random bars t, build a perturbed
         copy of features_test where every row AFTER t is multiplied
         by a random factor in ``perturb_scale``. Re-run the strategy.
      3. Assert the prediction at bar t is unchanged.

    Returns
    -------
    dict
        ``{"passed": bool, "n_samples": int, "max_delta": float,
           "violations": list[(bar_t, baseline_val, perturbed_val)]}``.

    Notes
    -----
    For trainable strategies, the train segment is held FIXED — only
    the test segment is perturbed. So a strategy that retrains on
    train+test combined will fail this test (and rightly so — that's
    a look-ahead at the parameter level).
    """
    if len(features_test) < 5:
        return {
            "passed": True, "n_samples": 0,
            "max_delta": 0.0, "violations": [],
        }

    baseline = np.asarray(
        strategy_fn(features_train, features_test), dtype=float
    )
    if baseline.shape[0] != len(features_test):
        raise ValueError(
            f"strategy_fn returned {baseline.shape[0]} predictions for "
            f"{len(features_test)} test bars"
        )

    rng = np.random.default_rng(rng_seed)
    n_test = len(features_test)
    # Sample bars in the middle 80% (avoid edges where the strategy
    # might not have enough lookback)
    edge = max(int(n_test * 0.1), 1)
    candidates = np.arange(edge, n_test - edge)
    if len(candidates) == 0:
        return {
            "passed": True, "n_samples": 0,
            "max_delta": 0.0, "violations": [],
        }
    n_samples = min(n_samples, len(candidates))
    sample_bars = sorted(rng.choice(candidates, size=n_samples, replace=False))

    violations = []
    max_delta = 0.0
    for t in sample_bars:
        mutated = features_test.copy()
        n_after = n_test - t - 1
        if n_after == 0:
            continue
        # Scale every numeric cell after t by a per-row uniform factor.
        factors = rng.uniform(*perturb_scale, size=n_after)
        for col in mutated.columns:
            if pd.api.types.is_numeric_dtype(mutated[col]):
                mutated.iloc[t + 1:, mutated.columns.get_loc(col)] = (
                    mutated.iloc[t + 1:, mutated.columns.get_loc(col)].to_numpy()
                    * factors
                )

        perturbed = np.asarray(
            strategy_fn(features_train, mutated), dtype=float
        )
        delta = abs(perturbed[t] - baseline[t])
        max_delta = max(max_delta, float(delta))
        if not np.isclose(
            perturbed[t], baseline[t], rtol=rtol, atol=atol, equal_nan=True,
        ):
            violations.append({
                "bar": int(t),
                "baseline": float(baseline[t]),
                "perturbed": float(perturbed[t]),
                "delta": float(delta),
            })

    return {
        "passed": len(violations) == 0,
        "n_samples": n_samples,
        "max_delta": max_delta,
        "violations": violations,
    }


# ---------------------------------------------------------------------------
# Bar-by-bar replay (deploy-time verification)
# ---------------------------------------------------------------------------


def replay_strategy_bar_by_bar(
    strategy_fn: StrategyFn,
    features_train: pd.DataFrame,
    features_test: pd.DataFrame,
    *,
    max_bars: Optional[int] = None,
    min_lookback: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Bar-by-bar replay: simulate a live deployment exactly.

    At each test bar t, call the strategy with the train segment and
    ONLY test bars ``[0, t+1)``, then capture the prediction at row t
    (the last row of the truncated test). This is what a live
    deployment does — at each new bar, re-run the strategy with the
    history up to and including that bar.

    Compare the replayed predictions against the batch predictions:
    they must match within float tolerance.

    Parameters
    ----------
    strategy_fn : callable
    features_train, features_test : pd.DataFrame
    max_bars : int, optional
        Cap on number of bars to replay. None = all test bars.
        Use a small cap (e.g., 50) for fast smoke testing of slow
        strategies (xgb_*, patchtst).
    min_lookback : int, default=1
        Minimum number of test bars required before the first replay
        call. Most strategies need at least 1.

    Returns
    -------
    tuple
        ``(batch_predictions, replayed_predictions)`` — two 1-D arrays
        of the same length. Caller compares for equality.
    """
    n_test = len(features_test)
    bars = list(range(min_lookback, n_test))
    if max_bars is not None:
        bars = bars[:max_bars]

    # Get baseline batch predictions
    batch = np.asarray(strategy_fn(features_train, features_test), dtype=float)
    if batch.shape[0] != n_test:
        raise ValueError(
            f"strategy_fn returned {batch.shape[0]} predictions for "
            f"{n_test} test bars"
        )

    replayed = np.full(n_test, np.nan, dtype=float)
    # Bars before min_lookback aren't replayed — copy batch values
    replayed[:min_lookback] = batch[:min_lookback]
    for t in bars:
        truncated_test = features_test.iloc[: t + 1]
        out = np.asarray(strategy_fn(features_train, truncated_test), dtype=float)
        if len(out) >= 1:
            replayed[t] = float(out[-1])

    return batch, replayed


__all__ = ["verify_no_lookahead", "replay_strategy_bar_by_bar"]
