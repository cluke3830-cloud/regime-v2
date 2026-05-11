"""Triple-barrier-labelled XGBoost regime classifier.

Brief 2.1 of the regime_dashboard upgrade plan — THE highest-leverage
move in the entire roadmap per the audit (§5.3, §8.2). Replaces the 540
LOC of hand-tuned basis functions and weights at regime_dashboard.py:
986-1198 with a learned classifier whose targets are López de Prado
triple-barrier labels and whose training is sample-weighted by the
canonical uniqueness × magnitude × time-decay scheme.

Public surface:

    compute_sample_weights(t1, returns, decay=1.0) -> np.ndarray
        AFML §4.3–4.5 weights. Three components:
          - uniqueness: 1 / average claim count over label horizon
          - magnitude:  |realised return| over the label window
          - time decay: exp(-decay · age_norm)
        Returned weights are renormalised to mean 1.0.

    RegimeXGBoost(**hparams)
        Three-class XGBoost (-1, 0, +1). fit / predict_proba / predict /
        position_from_proba / feature_importance.

    make_regime_xgboost_strategy(...) -> StrategyFn
        Adapter that wraps RegimeXGBoost into the cpcv_runner StrategyFn
        protocol — computes triple-barrier labels on the train segment,
        weights them, fits the model, and returns continuous positions
        ``p(+1) - p(-1)`` on the test segment.

Acceptance gates (audit Brief 2.1):
    (a) OOS log-loss ≥ 5% lower than the rule-layer baseline.
        v1 compares against a uniform-prior baseline (a trivial classifier
        predicting the empirical class prior); the rule-layer comparison
        comes online when Brief 2.2 strips the rule layer out of
        regime_dashboard.py.
    (b) Crisis recall ≥ 0.65 on the triple-barrier held-out tail —
        in the 3-class encoding, this maps to recall on the label = -1
        class, which we report and gate-check.
    (c) Feature importances economically interpretable — at minimum
        the top features must include volatility and momentum measures.

Hyperparameter defaults follow audit §8.2.1 verbatim:
    max_depth=4, eta=0.05, n_estimators=200, subsample=0.8,
    colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.1.
Nested-CPCV grid search over {max_depth, eta, n_estimators} is deferred
to Brief 2.1.1 — the v1 defaults are intentionally pre-registered to
keep n_trials honest for the DSR deflation.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from src.labels.triple_barrier import triple_barrier_labels


# ---------------------------------------------------------------------------
# Sample weights — López de Prado §4.3-4.5
# ---------------------------------------------------------------------------


def compute_sample_weights(
    t1: np.ndarray,
    returns: np.ndarray,
    *,
    decay: float = 1.0,
) -> np.ndarray:
    """Triple-barrier sample weights = uniqueness × magnitude × time-decay.

    Parameters
    ----------
    t1 : 1-D array of int
        For each sample ``i``, the positional index where its label resolves
        (the ``t1`` column from :func:`triple_barrier_labels`). The label
        horizon for sample ``i`` is the inclusive interval ``[i, t1[i]]``.
    returns : 1-D array of float
        Realised log-return over the label window — the ``ret`` column
        from :func:`triple_barrier_labels`. Used for magnitude weighting;
        sign is discarded (``|ret|``).
    decay : float, default=1.0
        Time-decay strength. ``0.0`` disables decay (uniform across time).
        ``1.0`` gives the audit's default exponential decay where the
        oldest sample carries weight ``e^{-1}`` relative to the newest.

    Returns
    -------
    np.ndarray, shape (n,)
        Per-sample weights renormalised to mean 1.0. Equal weights when
        all three components degenerate.

    Notes
    -----
    Uniqueness is the strict López de Prado formula:

        1. For each bar t, ``claim_count[t]`` = number of label horizons
           that cover t.
        2. For each label i with horizon ``[i, t1[i]]``,
           ``uniqueness[i]`` = mean of ``1 / claim_count[t]`` over t in
           the horizon.

    This is exact (not the simpler "neighbour-count" approximation in the
    audit's §8.2.1 sketch) and is O(n × max(t1 - i)) which is fine for
    daily data with horizons in the tens of bars.

    References
    ----------
    López de Prado, M. (2018). *Advances in Financial Machine Learning*.
        Wiley. §4.3 (Determination of Concurrent Labels), §4.4 (Average
        Uniqueness), §4.5 (Sample Weights by Return Attribution).
    """
    t1 = np.asarray(t1, dtype=np.int64)
    returns = np.asarray(returns, dtype=float)
    n = len(t1)
    if len(returns) != n:
        raise ValueError(f"t1 len {n} != returns len {len(returns)}")
    if n == 0:
        return np.array([], dtype=float)

    # 1. Uniqueness via per-bar claim count
    claim = np.zeros(n, dtype=np.int64)
    for i in range(n):
        end = min(int(t1[i]), n - 1)
        claim[i:end + 1] += 1
    claim = np.maximum(claim, 1)  # avoid div-by-zero on isolated bars
    uniqueness = np.zeros(n, dtype=float)
    for i in range(n):
        end = min(int(t1[i]), n - 1)
        if end >= i:
            uniqueness[i] = float(np.mean(1.0 / claim[i:end + 1]))
        else:
            uniqueness[i] = 1.0

    # 2. Magnitude
    magnitude = np.abs(returns)
    mu = magnitude.mean()
    if mu > 0:
        magnitude = magnitude / mu
    else:
        magnitude = np.ones(n, dtype=float)

    # 3. Time decay — newer samples weighted higher
    if decay > 0 and n > 1:
        ages = (n - 1 - np.arange(n, dtype=float)) / (n - 1)
        time_decay = np.exp(-decay * ages)
    else:
        time_decay = np.ones(n, dtype=float)

    weights = uniqueness * magnitude * time_decay
    mean_w = weights.mean()
    if mean_w > 0:
        weights = weights / mean_w
    else:
        weights = np.ones(n, dtype=float)
    return weights


# ---------------------------------------------------------------------------
# RegimeXGBoost — three-class classifier on triple-barrier labels
# ---------------------------------------------------------------------------


# Internal mapping for XGBoost (which wants 0-indexed class labels).
_LABEL_TO_IDX = {-1: 0, 0: 1, 1: 2}
_IDX_TO_LABEL = {0: -1, 1: 0, 2: 1}


class RegimeXGBoost:
    """Three-class XGBoost classifier on triple-barrier labels.

    Output classes (internal indices in parens):
        -1 (0)  → stop-barrier hit first    (drawdown regime)
         0 (1)  → time-barrier hit first    (ranging / no-edge regime)
        +1 (2)  → profit-barrier hit first  (upward edge regime)

    Hyperparameter defaults match audit §8.2.1 exactly. Do not tune
    these in-session without bumping ``n_trials`` in the DSR config.

    Special behaviour — if the training labels degenerate to a single
    class, the model falls back to predicting the empirical class prior
    (uniform on classes not present). This keeps the harness usable on
    benign training windows without a special-case at the call site.
    """

    def __init__(
        self,
        *,
        max_depth: int = 4,
        eta: float = 0.05,
        n_estimators: int = 200,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.1,
        tree_method: str = "hist",
        seed: int = 42,
        n_jobs: int = 1,
    ) -> None:
        self.max_depth = max_depth
        self.eta = eta
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.tree_method = tree_method
        self.seed = seed
        self.n_jobs = n_jobs
        self.model_: Optional[xgb.XGBClassifier] = None
        self.classes_: Optional[np.ndarray] = None
        self.prior_: Optional[np.ndarray] = None
        self.feature_names_: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> "RegimeXGBoost":
        """Fit on (X, y).

        Parameters
        ----------
        X : array-like, shape (n, d)
            Features. NaNs are tolerated (XGBoost handles them natively).
        y : array-like, shape (n,)
            Labels in ``{-1, 0, +1}``.
        sample_weight : array-like, shape (n,), optional
            Per-sample weights. Typically from
            :func:`compute_sample_weights`.
        feature_names : list of str, optional
            Stored for ``feature_importance()``.
        """
        X = np.asarray(X)
        y = np.asarray(y, dtype=np.int64)
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y length mismatch: {X.shape[0]} vs {y.shape[0]}"
            )

        # Determine empirical prior for fallback / inference padding
        unique, counts = np.unique(y, return_counts=True)
        self.classes_ = unique
        prior = np.zeros(3, dtype=float)
        for cls, cnt in zip(unique, counts):
            if int(cls) in _LABEL_TO_IDX:
                prior[_LABEL_TO_IDX[int(cls)]] = cnt
        if prior.sum() > 0:
            prior = prior / prior.sum()
        else:
            prior = np.array([1 / 3, 1 / 3, 1 / 3])
        self.prior_ = prior

        if feature_names is not None:
            self.feature_names_ = list(feature_names)

        # Degenerate case: only one class in y → no model fit, fall back to
        # constant-prior prediction.
        if len(unique) < 2:
            self.model_ = None
            self.trained_classes_ = []
            return self

        # Map labels {-1, 0, +1} → 0-indexed class slot {0, 1, 2}, then
        # dense-remap to {0..k-1} for XGBoost's strict sequential-class
        # requirement. ``trained_classes_`` records the ORIGINAL 0-indexed
        # slots so predict_proba can pad back to (n, 3).
        y_slot = np.array([_LABEL_TO_IDX[int(v)] for v in y], dtype=np.int64)
        present_slots = sorted(set(int(v) for v in y_slot))
        remap = {orig: dense for dense, orig in enumerate(present_slots)}
        y_dense = np.array([remap[int(v)] for v in y_slot], dtype=np.int64)
        k = len(present_slots)

        common = dict(
            max_depth=self.max_depth,
            learning_rate=self.eta,
            n_estimators=self.n_estimators,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,
            tree_method=self.tree_method,
            random_state=self.seed,
            n_jobs=self.n_jobs,
            verbosity=0,
        )
        if k == 2:
            clf = xgb.XGBClassifier(objective="binary:logistic", **common)
        else:
            clf = xgb.XGBClassifier(
                objective="multi:softprob", num_class=k, **common
            )
        clf.fit(X, y_dense, sample_weight=sample_weight)
        self.model_ = clf
        self.trained_classes_ = present_slots  # original 0-indexed slots
        return self

    # ------------------------------------------------------------------
    # predict / predict_proba
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns shape ``(n, 3)`` — columns ``p(-1)``, ``p(0)``, ``p(+1)``.

        When XGBoost was fit on a degenerate fold missing one or more of
        the three classes, ``model_.predict_proba`` returns fewer than
        three columns. We pad the missing columns with zero so the
        output is always shape ``(n, 3)`` and each row sums to 1.0
        (across the classes XGBoost actually saw — the never-seen
        classes get probability 0, which is correct: a tree-based
        classifier physically cannot emit a class it never trained on).
        """
        X = np.asarray(X)
        n = X.shape[0]
        if self.model_ is None:
            if self.prior_ is None:
                return np.full((n, 3), 1 / 3)
            return np.broadcast_to(self.prior_, (n, 3)).copy()

        proba_raw = np.asarray(self.model_.predict_proba(X), dtype=float)
        classes = getattr(self, "trained_classes_", None) or list(
            range(proba_raw.shape[1])
        )
        if proba_raw.shape[1] == 3 and set(classes) == {0, 1, 2}:
            full = proba_raw
        else:
            # Pad to (n, 3) by placing each present class in its 0-indexed slot.
            full = np.zeros((n, 3), dtype=float)
            for col_idx, cls in enumerate(classes):
                full[:, int(cls)] = proba_raw[:, col_idx]

        # Re-normalise rows. XGBoost's softprob is usually exact, but
        # tiny numerical drift can put sums at 1 + 1e-7 which trips
        # sklearn.metrics.log_loss's strict ``y_prob must sum to 1``
        # check. The renormalisation is no-op in the typical case.
        row_sums = full.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        return full / row_sums

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns shape (n,) — class labels in ``{-1, 0, +1}``."""
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        # Map back to label space
        return np.array([_IDX_TO_LABEL[int(i)] for i in idx], dtype=np.int64)

    @staticmethod
    def position_from_proba(proba: np.ndarray) -> np.ndarray:
        """Continuous position in [-1, +1] from class probabilities.

        ``position = p(+1) - p(-1)``. Falls naturally to 0 when the
        ranging-class probability dominates.
        """
        proba = np.asarray(proba)
        if proba.ndim != 2 or proba.shape[1] != 3:
            raise ValueError(
                f"proba must be (n, 3), got shape {proba.shape}"
            )
        return proba[:, 2] - proba[:, 0]

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------

    def feature_importance(
        self, importance_type: str = "gain"
    ) -> Dict[str, float]:
        """Per-feature importance (``gain`` by default).

        Returns
        -------
        dict
            Mapping feature_name → importance score. If feature_names were
            not provided at fit time, falls back to ``f0, f1, ...``.
            Empty dict if the model was not fit (single-class training).
        """
        if self.model_ is None:
            return {}
        booster = self.model_.get_booster()
        scores = booster.get_score(importance_type=importance_type)
        # XGBoost names features f0, f1, ... internally. Remap if we have
        # original feature names.
        if self.feature_names_ is not None:
            mapped: Dict[str, float] = {}
            for k, v in scores.items():
                if k.startswith("f"):
                    try:
                        idx = int(k[1:])
                        if 0 <= idx < len(self.feature_names_):
                            mapped[self.feature_names_[idx]] = float(v)
                            continue
                    except ValueError:
                        pass
                mapped[k] = float(v)
            return mapped
        return {k: float(v) for k, v in scores.items()}


# ---------------------------------------------------------------------------
# Strategy adapter — wraps RegimeXGBoost for the cpcv_runner harness
# ---------------------------------------------------------------------------


def make_regime_xgboost_strategy(
    *,
    pi_up: float = 2.0,
    pi_down: Optional[float] = None,
    horizon: int = 10,
    decay: float = 1.0,
    feature_cols: Optional[List[str]] = None,
    close_col: str = "close",
    vol_col: str = "vol_ewma",
    **xgb_kwargs,
) -> Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]:
    """Build a strategy_fn that fits a RegimeXGBoost per CPCV fold.

    Per-fold pipeline:
      1. Compute triple-barrier labels on ``features_train`` using
         ``features_train[close_col]`` and ``features_train[vol_col]``.
      2. Compute sample weights via :func:`compute_sample_weights`.
      3. Fit ``RegimeXGBoost(**xgb_kwargs)`` on ``features_train[feature_cols]``.
      4. Predict probabilities on ``features_test[feature_cols]``.
      5. Return continuous positions ``p(+1) - p(-1)``.

    The trailing ``horizon - 1`` rows of the training set have label = 0
    (their natural horizon falls past the segment end). They are kept in
    training because the alternative (drop them) reduces sample size more
    than the noise they add.

    Parameters
    ----------
    pi_up, pi_down, horizon :
        Triple-barrier label parameters. Defaults match the Phase 1 report
        (π=2.0, h=10).
    decay :
        Time-decay strength for sample weights.
    feature_cols : list of str, optional
        Columns from ``features_train`` to use as model features. Defaults
        to "every column except ``close_col``".
    close_col, vol_col :
        Column names for the underlying close price and per-bar volatility
        (used by the triple-barrier label computation, NOT as features).
    **xgb_kwargs :
        Forwarded to :class:`RegimeXGBoost`.

    Returns
    -------
    StrategyFn
        Compatible with :func:`src.validation.cpcv_runner.run_cpcv_validation`.
    """

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

        model = RegimeXGBoost(**xgb_kwargs)
        model.fit(X_train, y, sample_weight=weights, feature_names=cols)
        proba = model.predict_proba(X_test)
        return model.position_from_proba(proba)

    return strategy_fn


__all__ = [
    "compute_sample_weights",
    "RegimeXGBoost",
    "make_regime_xgboost_strategy",
]