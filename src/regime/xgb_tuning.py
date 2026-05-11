"""Nested CPCV grid search for RegimeXGBoost hyperparameters.

Brief 2.1.3 of the regime upgrade plan. Implements audit §8.2.1's
prescribed nested cross-validation: an inner CPCV loop tunes
``(max_depth, eta, n_estimators)`` per outer fold, replacing the
hand-tuned hparams that bottleneck xgb_v1/v2 in Brief 2.1.2.

Why nested CV (not a global grid search):
    A single global grid search picks the hparams that look best on the
    pooled CPCV paths — which leaks the test set into hparam selection.
    Nested CV picks hparams INSIDE each outer fold's training segment
    only, then evaluates the refit model on that fold's untouched test.
    This is the López de Prado §11.6 best practice.

Public surface:

    tune_xgb_hparams(X, y, sample_weight, *, param_grid, ...) -> (best, scores)
        Pure function — runs inner CPCV grid search on a single
        (X, y, sample_weight) triple. Returns the best params (lowest
        mean OOS log-loss across inner paths) and the full score dict.

    make_tuned_regime_xgboost_strategy(*, param_grid, **fixed_kwargs) -> StrategyFn
        Strategy factory matching the CPCV harness contract. On each
        outer fold call: triple-barrier label → sample weights → inner
        CPCV grid search → refit on full outer-train → predict outer-test.

Grid budget:
    DEFAULT_PARAM_GRID_SMALL — 2×1×2 = 4 combos. ~1-2 min wall-clock per
    outer fold; ~10-20 min for the full 45-path outer CPCV.
    DEFAULT_PARAM_GRID_FULL  — audit §8.2.1's 4×3×3 = 36 combos. ~10×
    slower; use sparingly. Bump n_trials to ~150 to honestly account
    for the search.

References
----------
López de Prado, M. (2018). *Advances in Financial Machine Learning*.
    Wiley. §11.6 (nested CV best practice).
"""

from __future__ import annotations

import itertools
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from src.labels.triple_barrier import triple_barrier_labels
from src.regime.regime_xgboost import (
    RegimeXGBoost,
    _LABEL_TO_IDX,
    compute_sample_weights,
)
from src.validation.cv_purged import CombinatorialPurgedKFold


# ---------------------------------------------------------------------------
# Pre-canned grids
# ---------------------------------------------------------------------------


DEFAULT_PARAM_GRID_SMALL: Dict[str, List[Any]] = {
    "max_depth":    [3, 5],
    "eta":          [0.05],
    "n_estimators": [100, 200],
}
"""4-combo grid for fast iteration. Inner-CV-validated picks one of:
    (depth=3, n=100): shallowest, fastest, max regularisation
    (depth=3, n=200): shallow but more rounds
    (depth=5, n=100): deeper, fewer rounds
    (depth=5, n=200): more capacity (the "let it overfit" corner)
"""

DEFAULT_PARAM_GRID_FULL: Dict[str, List[Any]] = {
    "max_depth":    [3, 4, 5, 6],
    "eta":          [0.03, 0.05, 0.10],
    "n_estimators": [100, 200, 400],
}
"""Audit §8.2.1's prescribed 36-combo grid. Wall-clock ~10× DEFAULT_SMALL.
   Bump DSR n_trials to ~150 when shipping this in the validation report.
"""


# ---------------------------------------------------------------------------
# Core tuner
# ---------------------------------------------------------------------------


def tune_xgb_hparams(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    *,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    inner_n_splits: int = 5,
    inner_n_test_groups: int = 1,
    inner_embargo_pct: float = 0.01,
    label_horizons: Optional[np.ndarray] = None,
    fixed_xgb_kwargs: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    verbose: bool = False,
) -> Tuple[Dict[str, Any], Dict[Tuple, float]]:
    """Inner CPCV grid search over ``param_grid``.

    Iterates every Cartesian combination of ``param_grid`` values, fits a
    :class:`RegimeXGBoost` on each inner-CV training segment, scores by
    mean OOS log-loss across the inner paths, and returns the combo with
    the lowest mean log-loss.

    Parameters
    ----------
    X, y : np.ndarray
        Training data. ``y`` in ``{-1, 0, +1}``.
    sample_weight : np.ndarray, optional
        Per-sample weights from :func:`compute_sample_weights`.
    param_grid :
        Dict of ``{name: [values]}``. Defaults to ``DEFAULT_PARAM_GRID_SMALL``.
        Names must match :class:`RegimeXGBoost` constructor kwargs
        (``max_depth``, ``eta``, ``n_estimators``, ``subsample``,
        ``colsample_bytree``, ``reg_lambda``, ``reg_alpha``).
    inner_n_splits, inner_n_test_groups, inner_embargo_pct :
        Inner CPCV configuration. Default 5 splits × 1 test group → 5
        inner OOS paths per combo.
    label_horizons :
        For the inner CPCV's purge step. Pass ``t1 - arange(n)`` when y
        comes from triple-barrier labels.
    fixed_xgb_kwargs : dict, optional
        Hyperparameters held FIXED across the grid (e.g., ``subsample``,
        ``seed``). Merged with each grid combo before instantiation;
        grid values win on key conflicts.
    seed :
        Determinism for both the inner CPCV index assignment and the
        XGBoost models. Same seed → same picks.
    verbose :
        If True, prints best/worst combo and timing.

    Returns
    -------
    best_params : dict
        The grid combo with the lowest mean inner-OOS log-loss.
    scores : dict
        Maps each grid combo tuple → mean log-loss. Useful for diagnostics.
    """
    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRID_SMALL
    fixed_xgb_kwargs = dict(fixed_xgb_kwargs or {})

    keys = list(param_grid.keys())
    grid = list(itertools.product(*(param_grid[k] for k in keys)))

    inner_cv = CombinatorialPurgedKFold(
        n_splits=inner_n_splits,
        n_test_groups=inner_n_test_groups,
        embargo_pct=inner_embargo_pct,
        label_horizons=label_horizons,
    )
    inner_paths = list(inner_cv.split(X))

    scores: Dict[Tuple, float] = {}
    start = time.time()
    for combo_idx, combo in enumerate(grid):
        params = dict(zip(keys, combo))
        full_params = {**fixed_xgb_kwargs, "seed": seed, **params}
        path_losses: List[float] = []

        for train_idx, test_idx in inner_paths:
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            sw_tr = (
                sample_weight[train_idx]
                if sample_weight is not None else None
            )

            model = RegimeXGBoost(**full_params)
            model.fit(X_tr, y_tr, sample_weight=sw_tr)
            proba = model.predict_proba(X_te)

            y_te_idx = np.array(
                [_LABEL_TO_IDX[int(v)] for v in y_te], dtype=np.int64
            )
            try:
                ll = log_loss(y_te_idx, proba, labels=[0, 1, 2])
                path_losses.append(float(ll))
            except ValueError:
                continue

        scores[combo] = (
            float(np.mean(path_losses)) if path_losses else float("inf")
        )

    if not scores or all(np.isinf(v) for v in scores.values()):
        # Degenerate — every fit failed. Fall back to the first combo.
        best_combo = grid[0]
    else:
        best_combo = min(scores, key=lambda k: scores[k])
    best_params = dict(zip(keys, best_combo))

    if verbose:
        elapsed = time.time() - start
        worst_combo = max(scores, key=lambda k: scores[k])
        print(
            f"[tune] {len(grid)} combos × {len(inner_paths)} inner paths "
            f"in {elapsed:.1f}s; best={dict(zip(keys, best_combo))} "
            f"({scores[best_combo]:.4f}), worst={dict(zip(keys, worst_combo))} "
            f"({scores[worst_combo]:.4f})"
        )

    return best_params, scores


# ---------------------------------------------------------------------------
# Tuned strategy factory
# ---------------------------------------------------------------------------


def make_tuned_regime_xgboost_strategy(
    *,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    inner_n_splits: int = 5,
    inner_n_test_groups: int = 1,
    inner_embargo_pct: float = 0.01,
    pi_up: float = 2.0,
    pi_down: Optional[float] = None,
    horizon: int = 10,
    decay: float = 1.0,
    feature_cols: Optional[List[str]] = None,
    close_col: str = "close",
    vol_col: str = "vol_ewma",
    verbose: bool = False,
    **fixed_xgb_kwargs,
) -> Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]:
    """Build a strategy_fn that tunes XGBoost hparams per outer CPCV fold.

    Per-outer-fold pipeline:
      1. Triple-barrier labels on ``features_train``.
      2. Sample weights (uniqueness × magnitude × time-decay).
      3. INNER CPCV grid search over ``param_grid``.
      4. Refit RegimeXGBoost on full outer-train with the winning hparams.
      5. Predict probabilities on ``features_test``.
      6. Return positions = ``p(+1) - p(-1)``.

    Parameters
    ----------
    param_grid :
        Dict mapping ``RegimeXGBoost`` constructor kwarg names → lists
        of candidate values. Defaults to :data:`DEFAULT_PARAM_GRID_SMALL`.
    inner_n_splits, inner_n_test_groups, inner_embargo_pct :
        Forwarded to the inner CPCV.
    pi_up, pi_down, horizon, decay :
        Triple-barrier label parameters.
    feature_cols :
        Columns from the input DataFrame to use as model features.
        Defaults to "every column except ``close_col``".
    close_col, vol_col :
        Column names for the underlying close and per-bar vol (used by
        triple-barrier label computation).
    verbose :
        If True, every fold prints its tuner result.
    **fixed_xgb_kwargs :
        Hyperparameters HELD FIXED across the grid (e.g. ``subsample``,
        ``colsample_bytree``, ``reg_lambda``). Forwarded to both the
        inner-CV models and the final refit.

    Returns
    -------
    StrategyFn
        Drop-in replacement for ``make_regime_xgboost_strategy`` —
        same signature, same harness contract.
    """
    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRID_SMALL

    def strategy_fn(
        features_train: pd.DataFrame, features_test: pd.DataFrame
    ) -> np.ndarray:
        labels = triple_barrier_labels(
            close=features_train[close_col],
            vol=features_train[vol_col],
            pi_up=pi_up,
            pi_down=pi_down,
            horizon=horizon,
        )
        t1 = labels["t1"].to_numpy(dtype=np.int64)
        rets = labels["ret"].to_numpy(dtype=float)
        y = labels["label"].to_numpy(dtype=np.int64)
        weights = compute_sample_weights(t1, rets, decay=decay)

        cols = feature_cols
        if cols is None:
            cols = [c for c in features_train.columns if c != close_col]
        X_train = features_train[cols].to_numpy(dtype=float)
        X_test = features_test[cols].to_numpy(dtype=float)
        label_horizons = (t1 - np.arange(len(t1))).astype(np.int64)

        best_params, _ = tune_xgb_hparams(
            X_train, y,
            sample_weight=weights,
            param_grid=param_grid,
            inner_n_splits=inner_n_splits,
            inner_n_test_groups=inner_n_test_groups,
            inner_embargo_pct=inner_embargo_pct,
            label_horizons=label_horizons,
            fixed_xgb_kwargs=fixed_xgb_kwargs,
            seed=fixed_xgb_kwargs.get("seed", 42),
            verbose=verbose,
        )

        model = RegimeXGBoost(**{**fixed_xgb_kwargs, **best_params})
        model.fit(X_train, y, sample_weight=weights, feature_names=cols)
        proba = model.predict_proba(X_test)
        return model.position_from_proba(proba)

    return strategy_fn


__all__ = [
    "DEFAULT_PARAM_GRID_SMALL",
    "DEFAULT_PARAM_GRID_FULL",
    "tune_xgb_hparams",
    "make_tuned_regime_xgboost_strategy",
]
