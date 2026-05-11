"""Stacked meta-learner — combine base strategies into a ridge-weighted blend.

Brief 2.3 of the regime upgrade plan. Audit prescription (§8.2.3 + Brief
2.3 spec): replace the dashboard's hand-tuned linear blend (rule 0.45 +
HMM 0.35 + LSTM 0.20) with a *learned* meta-classifier whose weights come
from cross-validated training-set fit.

Architecture v1 — pragmatic scope:

  Stacker base strategies = the DETERMINISTIC ones only:
    - buy_and_hold       (always +1)
    - momentum_20d       (+1 if 20-day mom > 0, else 0)
    - rule_baseline      (5-regime rule classifier, deterministic given features)

  Why deterministic-only: a stacker calling a base strategy's
  ``strategy_fn(features_train, features_train)`` to get IN-SAMPLE
  train positions is valid IFF the base strategy doesn't actually fit
  anything on train. For the XGBoost variants, the in-sample-on-train
  trick gives BIASED predictions (because xgb_v1.fit(train).predict(train)
  is leakage). Handling that requires nested CV inside the stacker —
  defer to Brief 2.3.1.

  Two stacker modes:
    1. EQUAL_WEIGHT — w_i = 1/k for k base strategies. Robust baseline,
       no parameters, can't overfit. The audit's §5.8.1 fallback when
       blend weights become "stale" (which is exactly the diagnosis the
       dashboard's v9.2/v9.3 changelog confirms).
    2. RIDGE — ``w = argmin || y_train - X_train @ w ||^2 + alpha*||w||^2``
       where X_train is the (n, k) matrix of base positions on the
       training segment and y_train is the realised log-return vector.
       ``positive=True`` constraint optional (rejects short signals from
       any individual base learner).

Output post-blend: clip to [-1, +1] so the harness never sees a
position outside its expected range.

Note on the audit acceptance gate ("Brier strictly < either single
component; calibration plot bins within 5pp of diagonal") — that's a
PROBABILITY-CALIBRATION gate, which assumes the stacker emits
class probabilities. v1 is a *regression* stacker on positions, not a
classifier. The equivalent gate for our setup is "Sharpe strictly >
the best single base" + "max-DD not materially worse". We use Sharpe
as the proxy and report it explicitly in the validation report.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


StrategyFn = Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]


# ---------------------------------------------------------------------------
# MetaStacker — wraps a Ridge regressor on (positions → forward returns)
# ---------------------------------------------------------------------------


class MetaStacker:
    """Ridge regression on (base_positions → realised returns).

    A fitted ``MetaStacker`` returns predicted-return values for new
    base-position vectors. The wrapper handles NaN masking, name-
    keeping, and exposes the learned ``coefs`` dict for diagnostics.

    Parameters
    ----------
    alpha : float
        L2 regularisation strength. Larger ``alpha`` → weights pulled
        toward zero. Defaults to 1.0 (sklearn default).
    non_negative : bool
        If True, requires ``coef_ >= 0``. Useful when you don't want
        the meta-learner to *short* a base strategy's signal.

    Attributes
    ----------
    strategy_names_ : list[str]
        Names of the base strategies in column order.
    coefs : dict[str, float]
        Learned coefficient per base strategy (sign + magnitude).
    """

    def __init__(self, alpha: float = 1.0, non_negative: bool = False):
        self.alpha = alpha
        self.non_negative = non_negative
        self._model = Ridge(alpha=alpha, positive=non_negative, fit_intercept=True)
        self.strategy_names_: Optional[List[str]] = None

    def fit(self, positions: pd.DataFrame, returns: np.ndarray) -> "MetaStacker":
        """Fit on (n, k) positions matrix and (n,) returns vector."""
        if positions.shape[0] != len(returns):
            raise ValueError(
                f"positions has {positions.shape[0]} rows but returns has {len(returns)}"
            )
        self.strategy_names_ = list(positions.columns)
        X = positions.to_numpy(dtype=float)
        y = np.asarray(returns, dtype=float)
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        if mask.sum() < 30:
            raise ValueError(
                f"need >= 30 non-NaN training rows, got {mask.sum()}"
            )
        self._model.fit(X[mask], y[mask])
        return self

    def predict(self, positions: pd.DataFrame) -> np.ndarray:
        """Predict expected return per row. NaN-safe at the row level."""
        X = positions.to_numpy(dtype=float)
        out = np.full(len(positions), np.nan, dtype=float)
        mask = ~np.isnan(X).any(axis=1)
        out[mask] = self._model.predict(X[mask])
        return out

    @property
    def coefs(self) -> Dict[str, float]:
        if self.strategy_names_ is None:
            return {}
        return dict(zip(self.strategy_names_, self._model.coef_))

    @property
    def intercept(self) -> float:
        return float(self._model.intercept_)


# ---------------------------------------------------------------------------
# Strategy factories
# ---------------------------------------------------------------------------


def _gather_positions(
    base_strategies: Dict[str, StrategyFn],
    features_for_train: pd.DataFrame,
    features_for_test: pd.DataFrame,
) -> pd.DataFrame:
    """Run each base on (features_for_train, features_for_test) and
    stack the resulting positions into a (n_test, k) DataFrame indexed
    by ``features_for_test.index``.

    For deterministic base strategies, ``features_for_train`` is used
    only as the "context window" some bases need (e.g. the rule
    classifier's Stabilizer state). The base's positions on
    ``features_for_test`` are returned.
    """
    cols = {}
    for name, fn in base_strategies.items():
        pos = fn(features_for_train, features_for_test)
        cols[name] = np.asarray(pos, dtype=float)
    return pd.DataFrame(cols, index=features_for_test.index)


def make_equal_weight_stacked_strategy(
    base_strategies: Dict[str, StrategyFn],
    *,
    clip: bool = True,
) -> StrategyFn:
    """Equal-weight average of base strategies. Cannot overfit.

    Reference: audit §5.8.1 names equal-weighting as the fallback when
    blend weights "become stale" — this is the parameter-free version.
    """

    def strategy_fn(
        features_train: pd.DataFrame, features_test: pd.DataFrame
    ) -> np.ndarray:
        test_pos = _gather_positions(
            base_strategies, features_train, features_test
        )
        blend = test_pos.mean(axis=1).to_numpy(dtype=float)
        return np.clip(blend, -1.0, 1.0) if clip else blend

    return strategy_fn


def make_ridge_stacked_strategy(
    base_strategies: Dict[str, StrategyFn],
    *,
    alpha: float = 1.0,
    non_negative: bool = True,
    clip: bool = True,
    position_scale: float = 100.0,
    close_col: str = "close",
    min_train_rows: int = 30,
) -> StrategyFn:
    """Ridge-weighted blend. Weights learned per CPCV fold.

    Per outer fold:
      1. Compute each base's positions on ``features_train`` (in-sample-
         on-train — VALID for deterministic bases only).
      2. Compute realised log-returns on ``features_train`` from
         ``features_train[close_col]``.
      3. Fit ``MetaStacker(alpha)`` mapping ``positions_train ->
         returns_train``.
      4. Apply ``stacker.predict`` to ``positions_test`` and rescale
         by ``position_scale`` so the output is in roughly [-1, +1].
      5. Clip to [-1, +1] if ``clip``.

    The ``position_scale`` accounts for the fact that the Ridge target
    is daily log-return (~1e-2 magnitude), but we want positions in
    [-1, +1]. The default 100.0 maps a predicted 1% return to a
    position of 1.0 — aggressive but consistent with the existing
    strategies' position scale.

    Falls back to equal-weight when training data is degenerate
    (< ``min_train_rows`` valid bars).
    """

    def strategy_fn(
        features_train: pd.DataFrame, features_test: pd.DataFrame
    ) -> np.ndarray:
        if close_col not in features_train.columns:
            raise KeyError(
                f"ridge_stacker requires '{close_col}' in features_train"
            )

        # Step 1 — base positions on train (in-sample-on-train pattern;
        # valid only for deterministic bases that don't fit on train).
        train_pos = _gather_positions(
            base_strategies, features_train, features_train
        )
        # Step 2 — realised log returns on train (use close-to-close).
        log_ret = np.log(features_train[close_col]).diff().to_numpy(dtype=float)

        # Step 3 — fit
        mask = ~np.isnan(log_ret) & ~train_pos.isna().any(axis=1).to_numpy()
        if mask.sum() < min_train_rows:
            # Degenerate train segment → fall back to equal weight
            test_pos_fallback = _gather_positions(
                base_strategies, features_train, features_test
            )
            blend = test_pos_fallback.mean(axis=1).to_numpy(dtype=float)
            return np.clip(blend, -1.0, 1.0) if clip else blend

        stacker = MetaStacker(alpha=alpha, non_negative=non_negative)
        # Index-align the masked rows
        train_pos_masked = train_pos.iloc[mask]
        stacker.fit(train_pos_masked, log_ret[mask])

        # Step 4 — predict on test
        test_pos = _gather_positions(
            base_strategies, features_train, features_test
        )
        pred_returns = stacker.predict(test_pos)

        # Step 5 — scale predicted return → position. NaN-safe.
        scaled = pred_returns * position_scale
        scaled = np.where(np.isnan(scaled), 0.0, scaled)
        return np.clip(scaled, -1.0, 1.0) if clip else scaled

    return strategy_fn


__all__ = [
    "MetaStacker",
    "make_equal_weight_stacked_strategy",
    "make_ridge_stacked_strategy",
]