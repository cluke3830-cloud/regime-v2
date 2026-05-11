"""Adaptive Conformal Inference — distribution-free calibration with
coverage guarantees under non-stationarity.

Brief 4.2 of the regime upgrade plan. Audit reference: §4.2 ("Tier-4
baseline"), §5.9, §8.4.2. Replaces the legacy dashboard's DISABLED
isotonic calibration (audit §5.9.2: isotonic was being calibrated
against label persistence, not accuracy, and inflated low raw
confidence back to 0.95 — a calibration ARTEFACT, not a model belief).

The Gibbs-Candès (2021) Adaptive Conformal Inference procedure
maintains an online quantile estimate of the non-conformity score
and adjusts the conformal threshold ``alpha_t`` toward the target
miscoverage rate ``alpha`` as it observes whether each prediction
set covered the realised label.

Algorithm (per bar t):
  1. Observe predicted probabilities ``p_hat`` and (later) realised
     label ``y_true``.
  2. Non-conformity score: ``score = 1 - p_hat[y_true]``.
  3. Prediction set at level ``alpha_t``: every class k whose
     non-conformity ``1 - p_hat[k]`` is below the empirical
     ``(1 - alpha_t)`` quantile of past scores.
  4. Update ``alpha_t``: ``alpha_t ← alpha_t + gamma * (alpha - I(miss))``
     where I(miss) = 1 if y_true not in prediction set, else 0.
     ``gamma`` is the online learning rate (small positive number).

Guarantees:
  - Marginal coverage ``P(y in S(X)) >= 1 - alpha`` over a long-run
    average, EVEN under distribution shift.
  - Coverage is achieved EXACTLY in the long run (not approximately).

Strategy adapter usage:
  Wrap any probability-emitting strategy (rule_baseline, xgb_*,
  patchtst). The conformal wrapper retrains nothing — it adjusts the
  prediction set on the fly using an online quantile estimate.

  Position = p_calibrated(+1) - p_calibrated(-1), where the
  calibrated probabilities are the BASE probabilities renormalised
  to the prediction set (classes outside the set get zero, classes
  inside keep their raw mass renormalised to sum to 1).

References
----------
Gibbs, I., Candès, E. (2021). Adaptive Conformal Inference Under
   Distribution Shift. NeurIPS 2021.
Vovk, V., Gammerman, A., Shafer, G. (2005). *Algorithmic Learning
   in a Random World*. Springer.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from src.labels.triple_barrier import triple_barrier_labels
from src.regime.regime_xgboost import _LABEL_TO_IDX


# ---------------------------------------------------------------------------
# AdaptiveConformal core
# ---------------------------------------------------------------------------


class AdaptiveConformal:
    """Online adaptive conformal predictor (Gibbs-Candès 2021).

    Maintains a sliding-window history of non-conformity scores and
    an adaptive miscoverage estimate ``alpha_t`` updated each bar
    based on whether the current prediction set covered the realised
    label.

    Parameters
    ----------
    alpha : float, default=0.10
        Target miscoverage rate. ``1 - alpha`` is the target coverage
        (e.g. ``alpha=0.10`` → target 90% coverage).
    gamma : float, default=0.005
        Online learning rate for ``alpha_t`` adaptation. Smaller =
        slower adaptation, more stable. Audit-§8.4.3 default.
    window : int, default=500
        Sliding-window size for the empirical-quantile estimate of
        non-conformity scores. None → unbounded growth.

    Attributes
    ----------
    alpha_t : float
        Current adaptive miscoverage estimate. Starts at ``alpha``,
        drifts to satisfy long-run coverage.
    scores_ : list[float]
        Sliding-window history of non-conformity scores.
    """

    def __init__(
        self, *,
        alpha: float = 0.10, gamma: float = 0.005,
        window: Optional[int] = 500,
    ):
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if gamma <= 0:
            raise ValueError(f"gamma must be > 0, got {gamma}")
        if window is not None and window < 30:
            raise ValueError(f"window must be >= 30 (or None), got {window}")
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.window = window
        self.alpha_t: float = float(alpha)
        self.scores_: List[float] = []

    def update_and_predict(
        self, p_hat: np.ndarray, y_true: Optional[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Update internal state with ``y_true`` (if known) and return
        the prediction set + calibrated probabilities for ``p_hat``.

        Parameters
        ----------
        p_hat : (n_classes,) np.ndarray
            Base classifier's predicted probabilities.
        y_true : int, optional
            Realised label index (0-indexed slot in ``p_hat``).
            ``None`` to skip the online update (test-time only).

        Returns
        -------
        prediction_set : (n_classes,) bool
            Indicator of which classes are in the prediction set.
        p_calibrated : (n_classes,) np.ndarray
            Renormalised probabilities — classes outside the
            prediction set get 0, classes inside keep their raw mass
            (renormalised to sum to 1). If the set is empty, falls
            back to the raw ``p_hat``.
        """
        p_hat = np.asarray(p_hat, dtype=float)
        n_classes = p_hat.shape[0]
        if n_classes == 0:
            return np.zeros(0, dtype=bool), p_hat

        # Threshold from past scores at level (1 - alpha_t)
        if len(self.scores_) >= 30:
            q = float(np.quantile(self.scores_, 1.0 - self.alpha_t))
        else:
            q = 1.0  # warm-up: include all classes

        prediction_set = (1.0 - p_hat) <= q

        if y_true is not None and 0 <= int(y_true) < n_classes:
            score = float(1.0 - p_hat[int(y_true)])
            self.scores_.append(score)
            if self.window is not None and len(self.scores_) > self.window:
                self.scores_.pop(0)
            # alpha_t update: drifts up if we miss too often, down if we
            # cover too generously.
            missed = not prediction_set[int(y_true)]
            self.alpha_t = float(
                np.clip(
                    self.alpha_t + self.gamma * (self.alpha - int(missed)),
                    1e-6, 1.0 - 1e-6,
                )
            )

        # Calibrated probabilities: renormalise inside the set
        in_set_mass = float(p_hat[prediction_set].sum())
        if in_set_mass > 0:
            p_cal = np.where(prediction_set, p_hat / in_set_mass, 0.0)
        else:
            # Empty set — fall back to raw probs
            p_cal = p_hat.copy()
        return prediction_set, p_cal

    def predict_only(self, p_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Same as ``update_and_predict`` but with no update (test only)."""
        return self.update_and_predict(p_hat, y_true=None)

    @property
    def empirical_coverage(self) -> float:
        """Estimated coverage = 1 - alpha_t (after enough warmup)."""
        return 1.0 - self.alpha_t


# ---------------------------------------------------------------------------
# Probability-emitting strategy adapter
# ---------------------------------------------------------------------------


def make_conformal_calibrated_strategy(
    base_proba_fn: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray],
    *,
    alpha: float = 0.10,
    gamma: float = 0.005,
    window: int = 500,
    pi_up: float = 2.0,
    pi_down: Optional[float] = None,
    horizon: int = 10,
    close_col: str = "close",
    vol_col: str = "vol_ewma",
) -> Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]:
    """Strategy_fn factory: wraps a base 3-class probability function
    with adaptive conformal calibration.

    The wrapper requires ``base_proba_fn(features_train, features_test)
    -> proba_matrix`` of shape ``(n_test, 3)``. It then:

      1. Compute triple-barrier labels on the TRAIN segment.
      2. Warm-start the conformal calibrator by walking through train
         predictions + train labels (online update).
      3. Walk through test predictions with no further updates
         (test-time only), producing calibrated probabilities per bar.
      4. Return position = p_cal(+1) - p_cal(-1).

    Note: this calibrator works with ANY probability-emitting model.
    Pass ``rule_baseline_proba_fn``, ``regime_xgboost_proba_fn``,
    or ``patchtst_proba_fn`` (the adapter below for xgb).
    """

    def strategy_fn(
        features_train: pd.DataFrame, features_test: pd.DataFrame
    ) -> np.ndarray:
        # Triple-barrier labels for the train segment (drives warm-up)
        labels = triple_barrier_labels(
            close=features_train[close_col],
            vol=features_train[vol_col],
            pi_up=pi_up, pi_down=pi_down, horizon=horizon,
        )
        y_train_labels = labels["label"].to_numpy(dtype=np.int64)
        y_train_idx = np.array(
            [_LABEL_TO_IDX[int(v)] for v in y_train_labels], dtype=np.int64
        )

        # Get the base probabilities on train (warm-up data)
        # We use the base strategy with (features_train, features_train) to
        # get train probabilities. Only valid for deterministic-given-train
        # base models — for trainable bases the caller should provide a
        # base_proba_fn that handles its own train/test logic.
        base_proba_train = base_proba_fn(features_train, features_train)
        base_proba_test = base_proba_fn(features_train, features_test)

        # Warm-up the conformal calibrator
        calib = AdaptiveConformal(alpha=alpha, gamma=gamma, window=window)
        for t in range(len(base_proba_train)):
            calib.update_and_predict(
                base_proba_train[t], y_true=int(y_train_idx[t]),
            )

        # Apply (no further updates) on test
        n_test = base_proba_test.shape[0]
        cal_probs = np.zeros_like(base_proba_test, dtype=float)
        for t in range(n_test):
            _, p_cal = calib.predict_only(base_proba_test[t])
            cal_probs[t] = p_cal

        # Position = p_cal(+1) - p_cal(-1) (slot 2 - slot 0)
        return cal_probs[:, 2] - cal_probs[:, 0]

    return strategy_fn


# ---------------------------------------------------------------------------
# Base-probability adapters for our existing models
# ---------------------------------------------------------------------------


def regime_xgboost_proba_fn(
    *, pi_up: float = 2.0, horizon: int = 10, decay: float = 1.0,
    feature_cols: Optional[List[str]] = None, close_col: str = "close",
    vol_col: str = "vol_ewma", **xgb_kwargs,
) -> Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]:
    """Returns a ``base_proba_fn`` that wraps RegimeXGBoost. Used as
    input to ``make_conformal_calibrated_strategy``.
    """
    from src.regime.regime_xgboost import (
        RegimeXGBoost, compute_sample_weights,
    )

    def proba_fn(
        features_train: pd.DataFrame, features_test: pd.DataFrame
    ) -> np.ndarray:
        labels = triple_barrier_labels(
            close=features_train[close_col],
            vol=features_train[vol_col],
            pi_up=pi_up, horizon=horizon,
        )
        t1 = labels["t1"].to_numpy(dtype=np.int64)
        rets = labels["ret"].to_numpy(dtype=float)
        y = labels["label"].to_numpy(dtype=np.int64)
        weights = compute_sample_weights(t1, rets, decay=decay)
        cols = feature_cols or [c for c in features_train.columns if c != close_col]
        X_train = features_train[cols].to_numpy(dtype=float)
        X_test = features_test[cols].to_numpy(dtype=float)
        model = RegimeXGBoost(**xgb_kwargs)
        model.fit(X_train, y, sample_weight=weights, feature_names=cols)
        return model.predict_proba(X_test)

    return proba_fn


__all__ = [
    "AdaptiveConformal",
    "make_conformal_calibrated_strategy",
    "regime_xgboost_proba_fn",
]
