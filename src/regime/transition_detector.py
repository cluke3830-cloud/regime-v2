"""Transition detector — binary classifier for "regime change in next H bars".

Brief 2.4 of the regime upgrade plan. The audit (§5.7) flagged the legacy
dashboard's transition detector as broken (F1 = 0.24, precision = 15%,
recall = 69%) — disabled in production via ``TRANS_ENABLED=False``. This
module rebuilds the detector with:

  - **Better target**: binary "any regime change in the next H bars"
    (vs the legacy multinomial which was harder to fit).
  - **Class-balanced loss**: XGBoost ``scale_pos_weight`` set to
    ``n_negative / n_positive`` so the rare-positive transition class
    gets adequate gradient.
  - **Richer features**: v2 features (21 cols) + rule_baseline's 5-class
    regime probabilities (26 cols total) — gives the detector both the
    raw market-state inputs AND the rule classifier's own uncertainty
    signal.
  - **Causal hygiene**: forward-target labels are built from rule_baseline
    sequence over the SAME training segment. The last H rows of training
    data are dropped (their forward window peeks past segment end).

Acceptance gate (audit Brief 2.4):
    F1 ≥ 0.40, precision ≥ 0.40 on an OOS held-out fold.

Strategy wrapper:
    ``make_transition_gated_strategy`` returns a strategy_fn that runs
    rule_baseline but de-risks (scales position toward zero) when the
    detector's ``P(transition) > threshold``. The idea: don't trust the
    regime classification near transition boundaries — go flat through
    the uncertainty.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from src.regime.rule_baseline import (
    V2_FEATURE_ORDER,
    compute_rule_regime_sequence,
)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class TransitionDetector:
    """XGBoost binary classifier: P(regime change in next ``horizon`` bars).

    Parameters
    ----------
    horizon : int, default=5
        Forward window over which transitions are considered.
        ``target[t] = 1`` iff any of ``label[t+1..t+horizon]`` differs
        from ``label[t]``.
    max_depth, eta, n_estimators, reg_lambda, reg_alpha :
        Standard XGBoost hyperparameters. Defaults are conservative
        (depth=4, n_est=200, L2=1.0) — same shape as the audit's rule
        layer XGBoost.
    seed : int
        Determinism.
    """

    def __init__(
        self,
        *,
        horizon: int = 5,
        max_depth: int = 4,
        eta: float = 0.05,
        n_estimators: int = 200,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.1,
        seed: int = 42,
    ):
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        self.horizon = horizon
        self.max_depth = max_depth
        self.eta = eta
        self.n_estimators = n_estimators
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.seed = seed
        self.model_: Optional[xgb.XGBClassifier] = None
        self.feature_names_: Optional[List[str]] = None
        self.scale_pos_weight_: Optional[float] = None

    def build_targets(self, labels: np.ndarray) -> np.ndarray:
        """Construct binary ``target[t] = 1`` if any of the next ``horizon``
        labels differs from ``label[t]``. The last ``horizon`` rows have
        no full forward window — they get target=0 by convention but are
        masked from training inside ``fit``.
        """
        labels = np.asarray(labels, dtype=np.int64)
        n = len(labels)
        out = np.zeros(n, dtype=np.int64)
        for t in range(n - self.horizon):
            forward = labels[t + 1: t + 1 + self.horizon]
            if (forward != labels[t]).any():
                out[t] = 1
        return out

    def fit(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        *,
        feature_names: Optional[List[str]] = None,
    ) -> "TransitionDetector":
        """Train on (X, labels). Builds the binary target internally.

        Masks the trailing ``horizon`` rows from training so the
        forward-window labels don't leak past the segment end.
        """
        X = np.asarray(X, dtype=float)
        if X.shape[0] != len(labels):
            raise ValueError(
                f"X has {X.shape[0]} rows but labels has {len(labels)}"
            )
        y = self.build_targets(labels)
        valid_end = max(X.shape[0] - self.horizon, 0)
        X_train, y_train = X[:valid_end], y[:valid_end]

        if len(y_train) < 30 or len(np.unique(y_train)) < 2:
            # Degenerate — no model. predict_proba returns 0.5 baseline.
            self.model_ = None
            return self

        pos = int(y_train.sum())
        neg = int(len(y_train) - pos)
        # Cap scale_pos_weight: extreme imbalance can destabilise XGBoost
        spw = float(np.clip(neg / max(pos, 1), 1.0, 50.0))
        self.scale_pos_weight_ = spw

        self.model_ = xgb.XGBClassifier(
            objective="binary:logistic",
            max_depth=self.max_depth,
            learning_rate=self.eta,
            n_estimators=self.n_estimators,
            scale_pos_weight=spw,
            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,
            random_state=self.seed,
            tree_method="hist",
            verbosity=0,
        )
        self.model_.fit(X_train, y_train)
        if feature_names is not None:
            self.feature_names_ = list(feature_names)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns P(transition in next ``horizon`` bars), shape (n,)."""
        X = np.asarray(X, dtype=float)
        if self.model_ is None:
            return np.full(len(X), 0.5, dtype=float)
        return self.model_.predict_proba(X)[:, 1].astype(float)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(np.int64)


# ---------------------------------------------------------------------------
# Evaluation helper — returns F1 / precision / recall for the audit gate
# ---------------------------------------------------------------------------


def evaluate_detector_metrics(
    detector: TransitionDetector,
    X_test: np.ndarray,
    test_labels: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute F1 / precision / recall on a held-out test segment.

    Builds the binary forward-window target from ``test_labels`` and
    compares the detector's prediction at threshold.

    Returns
    -------
    dict
        ``{"f1", "precision", "recall", "n_positive", "n_predicted", "passes_gate"}``.
        ``passes_gate`` is True iff F1 >= 0.40 AND precision >= 0.40
        (audit Brief 2.4 gate).
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    y_true = detector.build_targets(test_labels)
    valid_end = max(len(X_test) - detector.horizon, 0)
    if valid_end == 0:
        return {
            "f1": float("nan"), "precision": float("nan"), "recall": float("nan"),
            "n_positive": 0, "n_predicted": 0, "passes_gate": False,
        }
    y_true_valid = y_true[:valid_end]
    y_pred = detector.predict(X_test[:valid_end], threshold=threshold)

    if len(np.unique(y_true_valid)) < 2:
        # Degenerate — can't compute meaningful metrics
        return {
            "f1": float("nan"), "precision": float("nan"), "recall": float("nan"),
            "n_positive": int(y_true_valid.sum()),
            "n_predicted": int(y_pred.sum()),
            "passes_gate": False,
        }

    f1 = float(f1_score(y_true_valid, y_pred, zero_division=0))
    prec = float(precision_score(y_true_valid, y_pred, zero_division=0))
    rec = float(recall_score(y_true_valid, y_pred, zero_division=0))
    return {
        "f1": f1, "precision": prec, "recall": rec,
        "n_positive": int(y_true_valid.sum()),
        "n_predicted": int(y_pred.sum()),
        "passes_gate": (f1 >= 0.40 and prec >= 0.40),
    }


# ---------------------------------------------------------------------------
# Strategy wrapper — rule_baseline gated by the transition detector
# ---------------------------------------------------------------------------


def make_transition_gated_strategy(
    *,
    transition_threshold: float = 0.7,
    horizon: int = 5,
    smooth_gate: bool = True,
    **detector_kwargs,
) -> Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]:
    """Build a strategy_fn that runs rule_baseline gated by a fresh
    transition detector trained on each CPCV outer fold.

    Per outer fold:
      1. Compute rule_baseline labels + probs on the train segment.
      2. Train a ``TransitionDetector`` on (v2_features + rule_probs,
         rule_labels) over the train segment.
      3. Compute rule_baseline labels + probs on the (train ∪ test)
         concatenation (Stabilizer needs train state).
      4. Predict P(transition) on the test segment via the detector.
      5. Position = ``rule_position * gate(P)`` where ``gate`` is
         either a hard threshold (``smooth_gate=False``) or a smooth
         ``(1 - P)`` damping (``smooth_gate=True``, default).

    Hard threshold: position → 0 when ``P > transition_threshold``,
    unchanged otherwise.
    Smooth: position *= ``max(0, 1 - P)`` so higher P → more damping.
    """

    def strategy_fn(
        features_train: pd.DataFrame, features_test: pd.DataFrame
    ) -> np.ndarray:
        # Step 1: rule sequence on train
        train_seq = compute_rule_regime_sequence(features_train)
        train_labels = train_seq["label"].to_numpy(dtype=np.int64)
        # Detector input features = v2 cols + 5 regime probs
        feat_cols = [c for c in V2_FEATURE_ORDER if c in features_train.columns]
        prob_cols = [f"p_{r}" for r in range(5)]
        X_train = np.column_stack([
            features_train[feat_cols].to_numpy(dtype=float),
            train_seq[prob_cols].to_numpy(dtype=float),
        ])

        # Step 2: fit detector
        detector = TransitionDetector(horizon=horizon, **detector_kwargs)
        detector.fit(X_train, train_labels,
                     feature_names=feat_cols + prob_cols)

        # Step 3: rule sequence on (train ∪ test) — Stabilizer continuity
        combined = pd.concat([features_train, features_test]).sort_index()
        combined = combined[~combined.index.duplicated(keep="first")]
        combined_seq = compute_rule_regime_sequence(combined)
        test_seq = combined_seq.loc[features_test.index]

        # Step 4: predict on test
        X_test = np.column_stack([
            features_test[feat_cols].to_numpy(dtype=float),
            test_seq[prob_cols].to_numpy(dtype=float),
        ])
        p_trans = detector.predict_proba(X_test)

        # Step 5: gate the rule_baseline position
        base_pos = test_seq["position"].to_numpy(dtype=float)
        if smooth_gate:
            gate = np.clip(1.0 - p_trans, 0.0, 1.0)
        else:
            gate = (p_trans <= transition_threshold).astype(float)
        return base_pos * gate

    return strategy_fn


__all__ = [
    "TransitionDetector",
    "evaluate_detector_metrics",
    "make_transition_gated_strategy",
]
