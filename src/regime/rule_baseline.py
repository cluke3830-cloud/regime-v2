"""5-regime rule classifier — Regime_v2's own hand-designed baseline.

Brief 2.2 of the regime upgrade plan. This is the comparison strategy the
XGBoost variants need to beat (audit acceptance gate (a)). Unlike the
legacy dashboard's 6-regime taxonomy (Calm Trend / Volatile Trend /
Low-Vol Range / High-Vol Churn / Correction / Crisis), Regime_v2 ships a
cleaner purely-directional 5-regime view:

    0  Full Bull   — strong uptrend, low vol, shallow drawdown
    1  Half Bull   — moderate uptrend or slow grind, contained vol
    2  Chop        — sideways, no directional edge, mean-reverting
    3  Half Bear   — moderate downtrend, elevated vol, deeper DD
    4  Full Bear   — severe stress (the audit's "Crisis" — extreme DD +
                     shock + credit/VIX spike)

Each regime has a deterministic position allocation:

    Full Bull   : +1.00   (full long)
    Half Bull   : +0.70   (long but de-risked)
    Chop        : +0.20   (light long; chop favours mean-reversion not
                           directional bets)
    Half Bear   : -0.20   (light defense)
    Full Bear   : -0.50   (short, flight to safety)

Scoring architecture (5 × 21 signed weights):

    For each regime r at bar t:
      score_r(t) = sum_i  W[r, i] * b_i(t)

    where b_i(t) is the i-th basis function applied to the i-th v2
    feature at bar t. Most basis functions are linear in the [0, 1]
    normalised feature; the Full Bear regime adds non-linear ``g(x) =
    max(0, min(1, (x-0.5)*2))`` gates on its tail-event features so it
    only scores when DD / vol / shock are genuinely in the top half of
    their rolling-252 range. This mirrors the audit §5.3.1 fix that
    the dashboard added in v9.4 for its Crisis regime.

    Weights are hand-tuned via financial intuition (NOT data-fitted —
    learning the weights would defeat the point of having a rule
    baseline). Each weight is the importance of that feature for that
    regime, signed to indicate direction (positive = "high feature
    value favours this regime", negative = "low feature value favours
    this regime").

Causal hygiene: every input is already ``.shift(1)``-clean from
``compute_features_v2``. The rolling-252 min-max renormalisation here
is past-only by construction. The Stabilizer is feedback-clean
(majority vote on raw labels, not stabilised history — same trick the
legacy dashboard used to avoid sticky-trap pathology).

Architectural choices that diverge from legacy dashboard:
  - 5 regimes, not 6 (cleaner directional taxonomy, fewer params)
  - Signed linear weights (not unsigned + (1-x) tricks)
  - Single gating non-linearity, scoped to Full Bear only
  - Weights expressed as 5x21 numpy matrix rather than a dict of basis
    closures (easier to inspect, hand-tune, and unit-test)

Same as legacy:
  - SOFTMAX_TEMP = 2.8 (audit §5.3.2)
  - Stabilizer hyperparameters (HYSTERESIS_THRESH=0.04, MIN_PERSIST=1,
    MAJORITY_WIN=2 — audit §5.4)
  - Crisis promote thresholds (shock_z > 3.5 OR raw_dd > 15% per
    audit §5.3.4)
  - risk_condition over-extension penalty pattern

References
----------
``etc/regime_dashboard.py:986-1198`` — the 6-regime classifier we're
   replacing as Regime_v2's rule baseline.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------

N_REGIMES = 5
REGIME_NAMES = {
    0: "Full Bull",
    1: "Half Bull",
    2: "Chop",
    3: "Half Bear",
    4: "Full Bear",
}
REGIME_ALLOC = {
    0:  1.00,
    1:  0.70,
    2:  0.20,
    3: -0.20,
    4: -0.50,
}
FULL_BEAR = 4  # used by the riskoff_confirm gate


# ---------------------------------------------------------------------------
# Hyperparameters — same as the legacy dashboard (audit §5.3–5.4)
# ---------------------------------------------------------------------------

SOFTMAX_TEMP        = 2.8
HYSTERESIS_THRESH   = 0.04
MIN_PERSIST         = 1
MAJORITY_WIN        = 2

# Tail-event gates (operate on RAW shock_z and raw drawdown, not normed)
RISKOFF_SHOCK_Z      = 2.0
RISKOFF_DD_THRESH    = 0.035
CRISIS_PROMOTE_SHOCK = 3.5
CRISIS_PROMOTE_DD    = 0.15

# Anti-stagnation
OVEREXT_BARS    = 60
OVEREXT_PENALTY = 0.20

NORM_WINDOW = 252


# ---------------------------------------------------------------------------
# v2 feature column order — fixes the index of every weight matrix below
# ---------------------------------------------------------------------------

V2_FEATURE_ORDER = [
    # 0-3 multi-horizon momentum
    "mom_5", "mom_20", "mom_63", "mom_252",
    # 4-7 EWMA vol pyramid
    "vol_short", "vol_ewma", "vol_long", "vol_yearly",
    # 8-9 vol ratios
    "vol_ratio_sl", "vol_ratio_ly",
    # 10 shock
    "shock_z",
    # 11 drawdown
    "drawdown_252",
    # 12 autocorrelation
    "autocorr_63",
    # 13 trend direction
    "trend_dir",
    # 14-16 VIX
    "vix_log", "vix_change", "vix_term",
    # 17-18 cross-asset
    "corr_tlt_63", "corr_gld_63",
    # 19-20 macro
    "term_spread", "credit_spread",
]
assert len(V2_FEATURE_ORDER) == 21


# ---------------------------------------------------------------------------
# Weight matrix — 5 regimes × 21 features
# ---------------------------------------------------------------------------
#
# Sign convention: positive = "feature ↑ favours this regime",
#                  negative = "feature ↑ disfavours this regime".
# Magnitudes are hand-tuned via financial intuition (see module docstring).
# Each row's absolute values do NOT need to sum to 1 — the softmax handles
# normalisation. But for legibility the magnitudes are chosen so each row's
# sum-of-|w| is roughly 1.0 (loosely interpretable as a probability mass).
#
# Columns correspond to V2_FEATURE_ORDER.

_W_FULL_BULL = np.array([
    # mom_5, mom_20, mom_63, mom_252
    +0.04, +0.08, +0.08, +0.04,
    # vol_short, vol_ewma, vol_long, vol_yearly
    -0.06, -0.10, -0.06, -0.03,
    # vol_ratio_sl, vol_ratio_ly
    -0.04, -0.02,
    # shock_z
    -0.04,
    # drawdown_252  (drawdown is in [-1, 0]; normed will be in [0, 1] where
    # 0 = deepest DD, 1 = at the high. So we WANT high normed value.)
    +0.10,
    # autocorr_63
    0.0,
    # trend_dir
    +0.12,
    # vix_log, vix_change, vix_term
    -0.05, -0.02, -0.03,
    # corr_tlt_63, corr_gld_63 — risk-on: stocks/bonds move OPPOSITELY
    -0.02, 0.0,
    # term_spread (steep curve = healthy economy)
    +0.03,
    # credit_spread (tight = bullish)
    -0.04,
])

_W_HALF_BULL = np.array([
    +0.03, +0.05, +0.05, +0.03,
    -0.04, -0.06, -0.04, -0.02,
    -0.02, -0.01,
    -0.02,
    +0.07,
    +0.02,
    +0.06,
    -0.03, -0.01, -0.02,
    -0.01, 0.0,
    +0.02,
    -0.02,
])

_W_CHOP = np.array([
    -0.02, -0.03, -0.02, 0.0,
    +0.03, +0.04, +0.03, 0.0,
    +0.02, +0.02,
    +0.02,
    +0.02,
    # Chop's signature is mean-reversion → high (rolling) autocorr
    +0.10,
    # No directional preference
    -0.02,  # mildly disfavours strong directional trend_dir
    +0.02, 0.0, +0.02,
    0.0, +0.01,
    0.0,
    +0.02,
])

_W_HALF_BEAR = np.array([
    -0.04, -0.06, -0.05, -0.03,
    +0.04, +0.06, +0.04, +0.02,
    +0.03, +0.02,
    +0.05,
    -0.07,  # deeper DD (low normed value) favours this regime
    +0.01,
    -0.07,  # negative trend_dir
    +0.05, +0.03, +0.04,
    +0.02, +0.02,
    -0.02,
    +0.04,
])

# Full Bear differs: applies a tail-gate ``g(x) = max(0, min(1, (x-0.5)*2))``
# on its biggest signals (drawdown, vol, shock, vix, credit spread) so it
# only fires when those features are in the TOP HALF of their rolling-252
# range. This is the audit-§5.3.1 fix the legacy dashboard added in v9.4.
# The flag column says: which features to gate (g(x)) vs use raw normed.
_W_FULL_BEAR = np.array([
    -0.04, -0.05, -0.04, -0.02,
    +0.05, +0.07, +0.06, +0.03,   # gated below
    +0.04, +0.02,
    +0.08,                          # gated
    -0.10,                          # GATED on deep DD direction
    0.0,
    -0.06,                          # strong negative trend_dir
    +0.08, +0.05, +0.06,            # vix_log gated
    +0.03, +0.03,
    -0.04,
    +0.06,                          # gated
])

# Which features get gated for Full Bear (True = apply g(x) instead of raw).
_FULL_BEAR_GATED = np.zeros(21, dtype=bool)
_FULL_BEAR_GATED[[4, 5, 6, 10, 11, 14, 20]] = True
#  vol_short(4), vol_ewma(5), vol_long(6), shock_z(10),
#  drawdown_252(11), vix_log(14), credit_spread(20)

_DEFAULT_WEIGHTS = np.vstack([
    _W_FULL_BULL,
    _W_HALF_BULL,
    _W_CHOP,
    _W_HALF_BEAR,
    _W_FULL_BEAR,
])
assert _DEFAULT_WEIGHTS.shape == (N_REGIMES, 21)


# ---------------------------------------------------------------------------
# Rolling min-max normalisation
# ---------------------------------------------------------------------------


def _rolling_minmax_norm(s: pd.Series, window: int = NORM_WINDOW) -> pd.Series:
    """Map each value to [0, 1] using a rolling-window min-max. Causal —
    uses only past ``window`` bars. Returns 0.5 (neutral) when the window
    is degenerate (min == max) or undefined."""
    roll_min = s.rolling(window, min_periods=20).min()
    roll_max = s.rolling(window, min_periods=20).max()
    span = (roll_max - roll_min).replace(0, np.nan)
    return ((s - roll_min) / span).clip(0.0, 1.0).fillna(0.5)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _softmax(x: np.ndarray, temp: float = SOFTMAX_TEMP) -> np.ndarray:
    z = x * temp
    e = np.exp(z - z.max())
    return e / e.sum()


def _g(x: float) -> float:
    """Crisis tail-gate: zero below the 50th pctl of normed feature, ramps
    linearly to 1.0 at the top. From audit §5.3.1 / dashboard v9.4."""
    return max(0.0, min(1.0, (x - 0.5) * 2.0))


def _row_score(features_normed: np.ndarray) -> np.ndarray:
    """Compute the 5-regime raw score vector for one bar.

    ``features_normed`` is shape (21,), each value in [0, 1].
    Returns a length-5 score vector (NOT softmaxed yet).
    """
    scores = np.zeros(N_REGIMES, dtype=float)
    # Regimes 0-3: linear dot product
    for r in range(N_REGIMES - 1):
        scores[r] = float(np.dot(_DEFAULT_WEIGHTS[r], features_normed))
    # Regime 4 (Full Bear): apply g() on gated features before dot product
    bear_basis = features_normed.copy()
    for i in np.where(_FULL_BEAR_GATED)[0]:
        bear_basis[i] = _g(features_normed[i])
    scores[FULL_BEAR] = float(np.dot(_DEFAULT_WEIGHTS[FULL_BEAR], bear_basis))
    return scores


# ---------------------------------------------------------------------------
# Stabilizer (verbatim from dashboard line 1125-1163)
# ---------------------------------------------------------------------------


class Stabilizer:
    """Majority vote on raw labels + hysteresis + min-persistence.

    The majority vote uses raw (pre-stabilisation) inputs so the
    feedback loop cannot create a "sticky trap" that prevents
    transitions even when probabilities strongly favour a new regime.
    """

    def __init__(self):
        self.current: Optional[int] = None
        self.persist: int = 0
        self.history: list[int] = []
        self._raw: list[int] = []

    def step(self, label: int, probs: np.ndarray) -> int:
        self._raw.append(label)
        tail = self._raw[-MAJORITY_WIN:]
        if len(tail) >= MAJORITY_WIN:
            label = Counter(tail).most_common(1)[0][0]
        if self.current is not None and label != self.current:
            if probs[label] - probs[self.current] <= HYSTERESIS_THRESH:
                label = self.current
        if self.current is not None and label != self.current:
            if self.persist < MIN_PERSIST:
                label = self.current
        if label != self.current:
            self.current = label
            self.persist = 1
        else:
            self.persist += 1
        self.history.append(label)
        return label


def _risk_condition(
    sc: np.ndarray, current: Optional[int], persist: int,
    vol_n: float, vol_prev: Optional[float],
) -> np.ndarray:
    """Over-extension penalty on the current regime + Chop-escalation."""
    sc = sc.copy()
    if current is not None and persist > OVEREXT_BARS:
        sc[current] *= (1.0 - OVEREXT_PENALTY)
    # Chop (2) suppression when vol keeps rising: escalate to Half Bear (3)
    # or Half Bull (1) depending on which scored higher.
    if current == 2 and persist > 30 and vol_prev is not None and vol_n > vol_prev:
        best = 3 if sc[3] >= sc[1] else 1
        sc[2] = min(sc[2], sc[best] - 0.01)
    return sc


def _riskoff_confirm(
    label: int, probs: np.ndarray, shock_raw: float, dd_raw: float,
) -> int:
    """Promote/demote Full Bear. Mirrors audit §5.3.4 logic.

    PROMOTE: shock_raw > 3.5σ OR raw drawdown > 15% → force Full Bear.
    DEMOTE : if argmax is Full Bear but shock_raw is not >2σ AND dd not
             >3.5%, fall back to the second-best non-Full-Bear regime.
    """
    if label != FULL_BEAR and (shock_raw > CRISIS_PROMOTE_SHOCK
                                or dd_raw > CRISIS_PROMOTE_DD):
        return FULL_BEAR
    if label == FULL_BEAR and not (shock_raw > RISKOFF_SHOCK_Z
                                   and dd_raw > RISKOFF_DD_THRESH):
        tmp = probs.copy()
        tmp[FULL_BEAR] = 0.0
        label = int(np.argmax(tmp))
    return label


# ---------------------------------------------------------------------------
# Full sequence over a v2 feature dataframe
# ---------------------------------------------------------------------------


# The CPCV harness calls our strategy_fn 45 times (once per outer fold)
# with different (train, test) slices. We DON'T cache here: an earlier
# implementation cached by (shape, first_index, last_index) and collided
# across CPCV folds that happened to share those keys but had different
# row subsets — produced silent wrong-fold answers and KeyError on
# strategy_fn slicing. Per-fold recompute is ~1s on 2000-bar v2 data,
# so 45s amortised cost across one full report — acceptable.
_cache: dict = {}  # retained for the test that clears it (no-op now)


def compute_rule_regime_sequence(features_v2: pd.DataFrame) -> pd.DataFrame:
    """Run the 5-regime rule classifier over an entire feature frame.

    Returns
    -------
    pd.DataFrame
        Indexed identically to ``features_v2``, with columns:
          ``p_0..p_4``  : softmaxed regime probabilities
          ``label``     : stabilised regime label in {0..4}
          ``regime``    : the human-readable regime name
          ``position``  : the per-regime allocation from REGIME_ALLOC
    """
    # Re-normalise each v2 feature to [0, 1] via rolling-252 min-max.
    # The renorm here is INDEPENDENT of any normalisation already applied
    # by compute_features_v2 — those produce scale-mixed outputs (some
    # already in [0, 1], some not). The rolling min-max homogenises.
    normed = pd.DataFrame(index=features_v2.index)
    for col in V2_FEATURE_ORDER:
        if col not in features_v2.columns:
            normed[col] = 0.5
        else:
            normed[col] = _rolling_minmax_norm(features_v2[col])

    # Special handling: drawdown is already in [-1, 0]. After min-max norm,
    # 0 in normed-space = deepest DD, 1 in normed-space = at the high.
    # That direction matches all the weight signs we picked.

    # Raw shock and raw DD for the riskoff_confirm gate (NOT normed).
    shock_raw = features_v2["shock_z"].to_numpy(dtype=float)
    dd_raw    = (-features_v2["drawdown_252"]).clip(lower=0).to_numpy(dtype=float)

    inputs = normed[V2_FEATURE_ORDER].to_numpy(dtype=float)
    vol_n  = normed["vol_ewma"].to_numpy(dtype=float)
    n = len(features_v2)

    probs_out = np.zeros((n, N_REGIMES), dtype=float)
    labels_out = np.zeros(n, dtype=np.int64)
    prev_vol: Optional[float] = None
    stab = Stabilizer()

    for t in range(n):
        sc = _row_score(inputs[t])
        sc = _risk_condition(sc, stab.current, stab.persist, vol_n[t], prev_vol)
        probs = _softmax(sc)
        raw_label = int(np.argmax(probs))
        confirmed = _riskoff_confirm(
            raw_label, probs,
            shock_raw=float(shock_raw[t]) if np.isfinite(shock_raw[t]) else 0.0,
            dd_raw=float(dd_raw[t])    if np.isfinite(dd_raw[t])    else 0.0,
        )
        stable = stab.step(confirmed, probs)
        probs_out[t] = probs
        labels_out[t] = stable
        prev_vol = vol_n[t]

    out = pd.DataFrame(
        probs_out, index=features_v2.index,
        columns=[f"p_{r}" for r in range(N_REGIMES)],
    )
    out["label"]    = labels_out
    out["regime"]   = out["label"].map(REGIME_NAMES)
    out["position"] = out["label"].map(REGIME_ALLOC).astype(float)
    return out


# ---------------------------------------------------------------------------
# Strategy adapter
# ---------------------------------------------------------------------------


def rule_baseline_strategy(
    features_train: pd.DataFrame, features_test: pd.DataFrame,
) -> np.ndarray:
    """Strategy_fn — slots into ``cpcv_runner.run_cpcv_multi_strategy``.

    Concatenates train + test, runs the rule classifier in time order
    (so the Stabilizer carries state into the test region — mirrors the
    dashboard's continuous-from-history-start convention), then slices
    out the test-region positions.
    """
    combined = pd.concat([features_train, features_test]).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    rule_out = compute_rule_regime_sequence(combined)
    return rule_out.loc[features_test.index, "position"].to_numpy(dtype=float)


__all__ = [
    "N_REGIMES",
    "REGIME_NAMES",
    "REGIME_ALLOC",
    "V2_FEATURE_ORDER",
    "compute_rule_regime_sequence",
    "rule_baseline_strategy",
    "Stabilizer",
    "_softmax",
    "_row_score",
    "_risk_condition",
    "_riskoff_confirm",
    "_DEFAULT_WEIGHTS",
    "_FULL_BEAR_GATED",
]