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

import math
from typing import Any, Callable, Dict, List, Optional

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
        # Detector input features = v2 cols + 3 regime probs
        feat_cols = [c for c in V2_FEATURE_ORDER if c in features_train.columns]
        prob_cols = [f"p_{r}" for r in range(3)]
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
    # Phase 3 — heuristic transition-risk signal for the website payload.
    # The trained XGBoost above is the strategy-layer signal (gates positions
    # inside CPCV). The heuristics below are the website-layer signal —
    # interpretable, model-free, fast enough to compute on every payload
    # build — used to populate the `transition_risk` field.
    "compute_margin_compression",
    "compute_regime_persistence",
    "compute_second_prob_acceleration",
    "compute_transition_risk",
]


# ===========================================================================
# Phase 3 — heuristic transition-risk signal for the website payload
# ===========================================================================
#
# Why heuristics instead of just using the trained XGBoost detector above?
#   - Interpretability: users see *why* risk is high ("Bull margin compressed
#     from 0.65 to 0.20 in 5 days"), not a black-box score.
#   - No per-asset model training on every payload build (the existing
#     detector is trained per CPCV outer fold inside the strategy_fn; for
#     live inference we'd need to persist a per-asset global model).
#   - Compounded signal mirrors how a discretionary trader actually reads
#     a regime board: dominant prob shrinking + regime overstaying its
#     welcome + a challenger rising = transition imminent.


def compute_margin_compression(
    prob_history: np.ndarray,
    lookback_bars: int = 5,
) -> Dict[str, float]:
    """How much the (top - second) probability margin has shrunk over the
    last ``lookback_bars`` bars.

    Parameters
    ----------
    prob_history : np.ndarray, shape (T, K)
        Per-bar probability vectors. Last row is the current bar.
    lookback_bars : int
        Comparison window. Returns NaN when the history is shorter than
        ``lookback_bars + 1``.

    Returns
    -------
    {
      "margin_now":         float,  # current top - second
      "margin_lookback":    float,  # margin lookback_bars ago
      "compression_pct":    float,  # 1 - (margin_now / margin_lookback)
                                    # Positive = margin shrinking (more risk)
                                    # Negative = margin expanding (less risk)
      "score":              float [0, 1],  # 0=no compression, 1=fully compressed
    }
    """
    p = np.asarray(prob_history, dtype=float)
    if p.ndim != 2:
        raise ValueError(f"prob_history must be 2-D, got shape {p.shape}")
    n_rows = p.shape[0]
    if n_rows <= lookback_bars:
        return {
            "margin_now": float("nan"),
            "margin_lookback": float("nan"),
            "compression_pct": float("nan"),
            "score": 0.0,
        }

    def _margin(row: np.ndarray) -> float:
        row = row[np.isfinite(row)]
        if row.size < 2:
            return float("nan")
        sorted_desc = np.sort(row)[::-1]
        return float(sorted_desc[0] - sorted_desc[1])

    margin_now = _margin(p[-1])
    margin_lookback = _margin(p[-(lookback_bars + 1)])

    if not (math.isfinite(margin_now) and math.isfinite(margin_lookback)):
        return {
            "margin_now": margin_now,
            "margin_lookback": margin_lookback,
            "compression_pct": float("nan"),
            "score": 0.0,
        }
    if margin_lookback <= 1e-9:
        # Margins were already collapsed lookback bars ago — no NEW info
        compression_pct = 0.0
    else:
        compression_pct = float(1.0 - (margin_now / margin_lookback))
    # Score: 0 when margin grew or stayed flat, ramps to 1 as it collapses.
    score = float(np.clip(compression_pct, 0.0, 1.0))
    return {
        "margin_now": float(margin_now),
        "margin_lookback": float(margin_lookback),
        "compression_pct": float(compression_pct),
        "score": score,
    }


def compute_regime_persistence(
    label_history: np.ndarray,
) -> Dict[str, float]:
    """How long the CURRENT regime has been active, relative to the
    historical distribution of regime episode durations.

    Uses the empirical 75th percentile of episode lengths for the same
    regime as the "typical mean-revert horizon". When current streak is
    above p75, regimes statistically tend to flip soon.

    Parameters
    ----------
    label_history : np.ndarray, shape (T,)
        Integer regime labels. Last element is the current regime.

    Returns
    -------
    {
      "current_streak":          int,   # bars in current regime
      "current_regime":          int,
      "typical_p75_duration":    float,
      "median_duration":         float,
      "persistence_percentile":  float [0, 1],  # streak's percentile in same-regime durations
      "expected_remaining_days": int,   # max(0, median - streak)
      "score":                   float [0, 1],  # 0 below median, ramps to 1 at 2x p75
    }
    """
    labels = np.asarray(label_history, dtype=np.int64)
    if labels.size == 0:
        return {
            "current_streak": 0, "current_regime": -1,
            "typical_p75_duration": float("nan"),
            "median_duration": float("nan"),
            "persistence_percentile": 0.0,
            "expected_remaining_days": 0,
            "score": 0.0,
        }

    current = int(labels[-1])
    streak = 1
    for i in range(len(labels) - 2, -1, -1):
        if int(labels[i]) != current:
            break
        streak += 1

    # Collect all PAST episodes for the same regime (exclude current ongoing one)
    durations: List[int] = []
    if labels.size >= 2:
        run_len = 1
        for i in range(1, labels.size):
            if int(labels[i]) == int(labels[i - 1]):
                run_len += 1
            else:
                if int(labels[i - 1]) == current:
                    durations.append(run_len)
                run_len = 1
        # Don't include the current trailing run (incomplete)

    if not durations:
        # No prior history for this regime — can't judge persistence
        return {
            "current_streak": streak, "current_regime": current,
            "typical_p75_duration": float("nan"),
            "median_duration": float("nan"),
            "persistence_percentile": 0.0,
            "expected_remaining_days": 0,
            "score": 0.0,
        }

    arr = np.array(durations, dtype=float)
    p75 = float(np.percentile(arr, 75))
    median = float(np.percentile(arr, 50))
    # Percentile of the streak in same-regime durations
    pct = float((arr <= streak).mean())

    # Expected remaining: how much longer until median, floored at 0
    expected_remaining = int(max(0, round(median - streak)))

    # Score: 0 when below median, ramps linearly to 1 between p75 and 2*p75
    if streak <= median:
        score = 0.0
    elif streak <= p75:
        score = float(0.3 * (streak - median) / max(p75 - median, 1.0))
    else:
        denom = max(p75, 1.0)
        score = float(min(1.0, 0.3 + 0.7 * (streak - p75) / denom))

    return {
        "current_streak": int(streak),
        "current_regime": current,
        "typical_p75_duration": p75,
        "median_duration": median,
        "persistence_percentile": pct,
        "expected_remaining_days": expected_remaining,
        "score": float(np.clip(score, 0.0, 1.0)),
    }


def compute_second_prob_acceleration(
    prob_history: np.ndarray,
    window: int = 5,
) -> Dict[str, float]:
    """How fast the SECOND-best regime's probability is accelerating
    upward. A rising challenger is the leading indicator of a regime flip.

    Parameters
    ----------
    prob_history : np.ndarray, shape (T, K)
        Last row is current bar.
    window : int
        Window for computing the trend in the second-best prob.

    Returns
    -------
    {
      "second_regime_now":        int,   # argsort()[-2] of current row
      "second_prob_now":          float,
      "second_prob_window_ago":   float,
      "delta":                    float, # now - window_ago
      "score":                    float [0, 1],
    }
    """
    p = np.asarray(prob_history, dtype=float)
    if p.ndim != 2 or p.shape[0] <= window:
        return {
            "second_regime_now": -1,
            "second_prob_now": float("nan"),
            "second_prob_window_ago": float("nan"),
            "delta": float("nan"),
            "score": 0.0,
        }
    current = p[-1]
    finite_mask = np.isfinite(current)
    if finite_mask.sum() < 2:
        return {
            "second_regime_now": -1,
            "second_prob_now": float("nan"),
            "second_prob_window_ago": float("nan"),
            "delta": float("nan"),
            "score": 0.0,
        }
    order = np.argsort(current)[::-1]
    second_idx = int(order[1])
    second_now = float(current[second_idx])
    past_row = p[-(window + 1)]
    if not np.isfinite(past_row[second_idx]):
        return {
            "second_regime_now": second_idx,
            "second_prob_now": second_now,
            "second_prob_window_ago": float("nan"),
            "delta": float("nan"),
            "score": 0.0,
        }
    second_past = float(past_row[second_idx])
    delta = second_now - second_past
    # Score: 0 when delta <= 0 (not rising), ramps to 1 at delta=0.20
    # (a 20 pp jump in second prob over `window` bars is very fast).
    score = float(np.clip(delta / 0.20, 0.0, 1.0))
    return {
        "second_regime_now": second_idx,
        "second_prob_now": second_now,
        "second_prob_window_ago": second_past,
        "delta": delta,
        "score": score,
    }


def compute_transition_risk(
    prob_history: np.ndarray,
    label_history: np.ndarray,
    regime_names: Optional[Dict[int, str]] = None,
    lookback_bars: int = 5,
) -> Dict[str, Any]:
    """Compose the three heuristic signals into a single transition-risk
    diagnostic for the API payload.

    Combined score is a weighted average:
        score = 0.4 * margin_compression + 0.3 * persistence + 0.3 * acceleration

    Level rules (combined-only — DO NOT short-circuit on any single sub-score):
        high   : combined >= 0.50
        medium : combined >= 0.30
        low    : otherwise

    Thresholds calibrated against the causal hit-rate probe on SPY
    2000-2025 (3264 bars, fusion source). The bar-level score
    distribution is highly bimodal: p75 = 0.300 (persistence-only firing,
    a low-signal cohort with ~9% hit rate near the unconditional 5%
    baseline) and p99 = 0.619 (real confluence). The cliff between
    "noise" and "signal" sits sharply at combined ≈ 0.40 in the hit-rate
    sweep; 0.50 is the audit-conservative pick.

    Empirical hit rates at the chosen thresholds:
        combined >= 0.50  →  2.0% of bars, 78% hit rate, 15.4× lift
        combined >= 0.30  →  30.4% of bars, 9.4% hit rate, 1.8× lift
        unconditional baseline: 5.1% (P(regime change in next 5 days))

    Earlier draft used "any sub-score >= 0.75" as a shortcut to "high",
    but the persistence sub-score hits 1.0 on ~25% of bars by construction
    (any bar above the historical p75 of its regime fires it), which made
    "high" fire on 35% of bars with only 7.7% hit rate. The combined-only
    rule fixes this.

    Parameters
    ----------
    prob_history : np.ndarray, shape (T, K)
        Bar-by-bar regime probability vectors (e.g., fusion posterior).
    label_history : np.ndarray, shape (T,)
        Bar-by-bar argmax labels.
    regime_names : dict, optional
        {0: "Bull", 1: "Neutral", 2: "Bear"} for the reason field.
    lookback_bars : int
        Window used by margin compression + second-prob acceleration.
    """
    names = regime_names or {0: "Bull", 1: "Neutral", 2: "Bear"}

    margin = compute_margin_compression(prob_history, lookback_bars=lookback_bars)
    persist = compute_regime_persistence(label_history)
    accel = compute_second_prob_acceleration(prob_history, window=lookback_bars)

    combined = float(
        0.4 * margin["score"] + 0.3 * persist["score"] + 0.3 * accel["score"]
    )
    combined = float(np.clip(combined, 0.0, 1.0))

    if combined >= 0.50:
        level = "high"
    elif combined >= 0.30:
        level = "medium"
    else:
        level = "low"

    top_alt_idx = accel.get("second_regime_now", -1)
    top_alt_name = names.get(int(top_alt_idx), "—") if top_alt_idx >= 0 else "—"
    current_regime_name = names.get(int(persist["current_regime"]), "—")

    reasons: List[str] = []
    if margin["score"] >= 0.4 and math.isfinite(margin.get("compression_pct", float("nan"))):
        reasons.append(
            f"top-second margin shrunk {margin['compression_pct'] * 100:.0f}% "
            f"over last {lookback_bars} bars "
            f"({margin['margin_lookback']:.2f} → {margin['margin_now']:.2f})"
        )
    if persist["score"] >= 0.4:
        reasons.append(
            f"{current_regime_name} regime held for {persist['current_streak']} bars "
            f"(typical p75 = {persist['typical_p75_duration']:.0f}, "
            f"percentile = {persist['persistence_percentile'] * 100:.0f}%)"
        )
    if accel["score"] >= 0.4 and math.isfinite(accel.get("delta", float("nan"))):
        reasons.append(
            f"P({top_alt_name}) rising {accel['delta'] * 100:+.0f} pp "
            f"over last {lookback_bars} bars"
        )
    if not reasons:
        reasons.append("all sub-signals below alert thresholds")

    return {
        "level": level,
        "score": combined,
        "current_regime": current_regime_name,
        "days_in_regime": int(persist["current_streak"]),
        "typical_p75_duration": float(persist["typical_p75_duration"])
            if math.isfinite(persist["typical_p75_duration"]) else None,
        "expected_remaining_days": int(persist["expected_remaining_days"]),
        "top_alternative_regime": top_alt_name,
        "components": {
            "margin_compression": {
                "score": margin["score"],
                "compression_pct": margin["compression_pct"]
                    if math.isfinite(margin.get("compression_pct", float("nan"))) else None,
                "margin_now": margin["margin_now"]
                    if math.isfinite(margin.get("margin_now", float("nan"))) else None,
                "margin_lookback": margin["margin_lookback"]
                    if math.isfinite(margin.get("margin_lookback", float("nan"))) else None,
            },
            "regime_persistence": {
                "score": persist["score"],
                "current_streak": int(persist["current_streak"]),
                "typical_p75": float(persist["typical_p75_duration"])
                    if math.isfinite(persist["typical_p75_duration"]) else None,
                "persistence_percentile": float(persist["persistence_percentile"]),
            },
            "second_prob_acceleration": {
                "score": accel["score"],
                "second_regime": top_alt_name,
                "delta": float(accel["delta"]) if math.isfinite(accel.get("delta", float("nan"))) else None,
                "second_prob_now": float(accel["second_prob_now"])
                    if math.isfinite(accel.get("second_prob_now", float("nan"))) else None,
            },
        },
        "reasons": reasons,
    }
