"""Snapshot generator for the Regime_v2 web dashboard.

Reads cached parquet for the 10-asset DEFAULT_UNIVERSE, runs the
rule-baseline 5-regime classifier and TVTP-MSAR champion on each asset,
and emits per-asset JSON snapshots that the Vercel-hosted Next.js
front-end ships at build time.

Output layout (all paths relative to repo root):
    dashboard/public/data/assets.json          # universe metadata
    dashboard/public/data/regimes/{ticker}.json  # one per asset
    dashboard/public/data/summary.json         # all-asset latest-bar grid

Why static JSON instead of a live API?
    TVTP-MSAR fits statsmodels MarkovAutoregression on every refresh —
    ~3-5s per asset, plus yfinance/FRED IO. Serverless functions on
    Vercel cap at 10s and don't carry statsmodels. The dashboard is
    "live as of the last build"; a daily GitHub Actions cron (or manual
    `make dashboard-data`) rebuilds the JSON and triggers a redeploy.

Acceptance:
    `python scripts/build_dashboard_data.py` produces 12 JSON files
    (10 assets + assets.json + summary.json) in dashboard/public/data/,
    each file < 500KB, no NaN/Infinity, valid JSON.
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines.tvtp_msar import MarkovSwitchingAR  # noqa: E402
from src.features.aux_data import fetch_aux_data_bundle  # noqa: E402
from src.features.price_features import compute_features_v2  # noqa: E402
from src.regime.gmm_hmm import (  # noqa: E402
    STATE_NAMES as GMM_STATE_NAMES,
    compute_gmm_hmm_sequence,
)
from src.regime.rule_baseline import (  # noqa: E402
    REGIME_ALLOC,
    REGIME_NAMES,
    compute_rule_regime_sequence,
)
from src.validation.multi_asset import DEFAULT_UNIVERSE, load_close  # noqa: E402


# Pretty names for the asset cards
ASSET_NAMES: Dict[str, str] = {
    "SPY":     "S&P 500",
    "QQQ":     "Nasdaq 100",
    "DIA":     "Dow Jones",
    "IWM":     "Russell 2000",
    "EFA":     "Developed ex-US",
    "EEM":     "Emerging Markets",
    "GLD":     "Gold",
    "TLT":     "20Y Treasuries",
    "BTC-USD": "Bitcoin",
    "JPY=X":   "USD / JPY",
}

# 3-regime color palette (Bloomberg dark)
REGIME_COLORS = {
    0: "#22c55e",  # Bull    — green
    1: "#a3a3a3",  # Neutral — grey
    2: "#ef4444",  # Bear    — red
}

# Validation report metrics — multi-asset robustness of the champion
# (ms_garch) net of 2 bps × |Δposition| transaction cost. Pulled from
# validation_report.md. Bump whenever `make validate` reshuffles the
# champion or shifts the cost model.
ASSET_BACKTEST_STATS: Dict[str, Dict[str, float]] = {
    "SPY":     {"sharpe_p05":  0.083, "sharpe_p50": 1.025, "sharpe_p95": 2.098, "max_dd_p50": -0.113, "dsr_sharpe": 1.025},
    "QQQ":     {"sharpe_p05":  0.019, "sharpe_p50": 0.958, "sharpe_p95": 1.987, "max_dd_p50": -0.128, "dsr_sharpe": 0.958},
    "DIA":     {"sharpe_p05": -0.069, "sharpe_p50": 0.937, "sharpe_p95": 2.066, "max_dd_p50": -0.123, "dsr_sharpe": 0.937},
    "IWM":     {"sharpe_p05": -0.664, "sharpe_p50": 0.284, "sharpe_p95": 1.342, "max_dd_p50": -0.157, "dsr_sharpe": 0.284},
    "EFA":     {"sharpe_p05": -0.485, "sharpe_p50": 0.259, "sharpe_p95": 1.388, "max_dd_p50": -0.150, "dsr_sharpe": 0.259},
    "EEM":     {"sharpe_p05": -0.902, "sharpe_p50": 0.153, "sharpe_p95": 1.293, "max_dd_p50": -0.175, "dsr_sharpe": 0.153},
    "GLD":     {"sharpe_p05": -0.046, "sharpe_p50": 0.650, "sharpe_p95": 1.496, "max_dd_p50": -0.153, "dsr_sharpe": 0.650},
    "TLT":     {"sharpe_p05": -0.964, "sharpe_p50": -0.082, "sharpe_p95": 1.017, "max_dd_p50": -0.205, "dsr_sharpe": 0.000},
    "BTC-USD": {"sharpe_p05": -0.306, "sharpe_p50": 0.930, "sharpe_p95": 2.044, "max_dd_p50": -0.230, "dsr_sharpe": 0.930},
    "JPY=X":   {"sharpe_p05": -0.726, "sharpe_p50": 0.328, "sharpe_p95": 1.282, "max_dd_p50": -0.129, "dsr_sharpe": 0.328},
}


HISTORY_BARS = 504  # ~2 years of trading days kept in the per-asset JSON.


def _finite(x: Any) -> Optional[float]:
    """Coerce numpy/float to plain Python float, mapping NaN/Inf → None.
    JSON spec doesn't allow Infinity/NaN; the front-end's chart libs
    misbehave on them too."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _compute_tvtp_probs(
    close: pd.Series, train_frac: float = 0.7
) -> Optional[pd.DataFrame]:
    """Fit MS-AR on early train portion, filter on full series.

    Returns DataFrame indexed by `close.index[1:]` with columns
    {p_low_vol, p_high_vol, position}.

    Returns None if the fit fails.
    """
    returns = np.log(close).diff().dropna()
    if len(returns) < 200:
        return None
    cut = int(len(returns) * train_frac)
    train_ret = returns.iloc[:cut]

    model = MarkovSwitchingAR(k_regimes=2, order=1, switching_variance=True)
    model.fit(train_ret)
    if model.params_ is None:
        return None
    try:
        probs = model.predict_proba(returns)
    except Exception:
        return None
    # state 0 = low-vol bull (+1.00), state 1 = high-vol stress (-0.30)
    pos = probs["p_low_vol"] * 1.00 + probs["p_high_vol"] * (-0.30)
    probs = probs.copy()
    probs["position"] = pos
    return probs


def _log_opinion_pool(
    gmm_proba: np.ndarray,
    tvtp_proba: np.ndarray,
    mapping: np.ndarray | None = None,
    eps: float = 1e-9,
) -> np.ndarray:
    """Combine GMM-HMM (3-state) + TVTP-MSAR (2→3-mapped) into a single posterior.

    The TVTP→3-class mapping is learned from rule_baseline label frequencies
    on the same series (see ``src.strategies.fusion.empirical_tvtp_3class_mapping``),
    replacing the old hardcoded 70/30 prior. If no mapping is passed, the
    prior is used as a fallback.

    Log-opinion-pool: p_fused ∝ exp(log p_gmm + log p_tvtp_3class). Agreement
    sharpens the posterior; disagreement flattens it. The Shannon entropy
    of the fused distribution serves as a natural uncertainty signal.
    """
    from src.strategies.fusion import apply_log_opinion_pool, _PRIOR_TVTP_MAPPING
    if mapping is None:
        mapping = _PRIOR_TVTP_MAPPING
    return apply_log_opinion_pool(gmm_proba, tvtp_proba, mapping, eps=eps)


# ---------------------------------------------------------------------------
# Phase 2 — Probability API: categorical confidence + cross-model consensus
# ---------------------------------------------------------------------------
#
# These promote two pieces of logic from the dashboard layer to first-class
# API fields:
#   - current_confidence : a categorical "how sure is the model" signal
#     derived from the fusion posterior. The Shannon entropy was already
#     in the payload; what was missing was the level/reason/margin a
#     consumer can branch on without re-implementing thresholds.
#   - model_consensus    : explicit cross-model agreement (rule + GMM +
#     TVTP vs the fusion label). Mirrors the 2-of-3 confirmation logic
#     in dashboard/lib/types.ts::confirmedActiveLabel but exposes it for
#     non-dashboard clients (alerts, third-party integrations).


def _classify_confidence(
    probs: List[Optional[float]],
    regime_names: Dict[int, str],
    entropy_normalised: Optional[float],
) -> Dict[str, Any]:
    """Categorize regime certainty from a probability vector.

    Levels follow the ULTRAPLAN spec:
      high   : top_prob > 0.65 AND second_prob < 0.25
      medium : top_prob > 0.50
      low    : otherwise

    score is 1 - normalized_entropy when entropy is available (closer to 1
    means a more peaked posterior); falls back to top_prob when entropy is
    None (e.g., fusion failed and we're staring at a rule-baseline argmax).
    """
    clean = [(i, float(p)) for i, p in enumerate(probs) if p is not None]
    if not clean:
        return {
            "level": "low", "score": 0.0,
            "top_regime": None, "top_prob": None,
            "second_regime": None, "second_prob": None,
            "margin": None,
            "reason": "no probability vector available",
        }
    clean.sort(key=lambda x: x[1], reverse=True)
    top_i, top_p = clean[0]
    second_i, second_p = clean[1] if len(clean) > 1 else (None, 0.0)
    margin = top_p - second_p

    if top_p > 0.65 and second_p < 0.25:
        level = "high"
    elif top_p > 0.50:
        level = "medium"
    else:
        level = "low"

    if entropy_normalised is None or not math.isfinite(entropy_normalised):
        score = float(top_p)
    else:
        score = float(max(0.0, min(1.0, 1.0 - entropy_normalised)))

    top_name = regime_names.get(top_i, str(top_i))
    second_name = regime_names.get(second_i, str(second_i)) if second_i is not None else None
    reason = (
        f"P({top_name})={top_p:.2f} leads P({second_name})={second_p:.2f} "
        f"by {margin:.2f}"
    )
    if entropy_normalised is not None and math.isfinite(entropy_normalised):
        reason += f"; entropy={entropy_normalised:.2f}"

    return {
        "level": level,
        "score": score,
        "top_regime": top_name,
        "top_prob": float(top_p),
        "second_regime": second_name,
        "second_prob": float(second_p),
        "margin": float(margin),
        "reason": reason,
    }


def _tvtp_to_3class_label(
    p_low_vol: Optional[float],
    p_high_vol: Optional[float],
) -> Optional[int]:
    """Map TVTP's 2-state output to the 3-class {Bull=0, Neutral=1, Bear=2}.

    TVTP doesn't model a Neutral state directly. We call a regime Neutral
    when neither state dominates (max < 0.60), otherwise pick the argmax:
    p_low_vol → Bull, p_high_vol → Bear. Returns None when both inputs
    are None.
    """
    if p_low_vol is None and p_high_vol is None:
        return None
    pl = float(p_low_vol or 0.0)
    ph = float(p_high_vol or 0.0)
    top = max(pl, ph)
    if top < 0.60:
        return 1   # Neutral
    return 0 if pl >= ph else 2


def _compute_model_consensus(
    rule_label: int,
    gmm_label: int,
    tvtp_low_vol: Optional[float],
    tvtp_high_vol: Optional[float],
    fusion_label: int,
    regime_names: Dict[int, str],
) -> Dict[str, Any]:
    """Cross-model agreement diagnostic. Counts how many of the three
    independent models (rule, GMM, TVTP) match the fused label.

    Levels:
      unanimous : all 3 agree with fusion
      strong    : 2 of 3 agree
      split     : 1 of 3 agrees
      divided   : 0 of 3 agree
    """
    tvtp_label = _tvtp_to_3class_label(tvtp_low_vol, tvtp_high_vol)
    voters = {
        "rule":   int(rule_label),
        "gmm":    int(gmm_label),
        "tvtp":   tvtp_label if tvtp_label is None else int(tvtp_label),
        "fusion": int(fusion_label),
    }
    other_models = ("rule", "gmm", "tvtp")
    agreeing = [m for m in other_models if voters[m] == voters["fusion"]]
    dissenting = [m for m in other_models if voters[m] is not None and voters[m] != voters["fusion"]]
    n_valid = sum(1 for m in other_models if voters[m] is not None)
    agreement_count = len(agreeing)
    agreement_pct = agreement_count / n_valid if n_valid > 0 else 0.0

    if agreement_count == 3:
        level = "unanimous"
    elif agreement_count == 2:
        level = "strong"
    elif agreement_count == 1:
        level = "split"
    else:
        level = "divided"

    return {
        "models": {
            "rule":   voters["rule"],
            "gmm":    voters["gmm"],
            "tvtp":   voters["tvtp"],
            "fusion": voters["fusion"],
        },
        "model_names": {
            "rule":   regime_names.get(voters["rule"], str(voters["rule"])),
            "gmm":    regime_names.get(voters["gmm"], str(voters["gmm"])),
            "tvtp":   regime_names.get(voters["tvtp"], "—") if voters["tvtp"] is not None else "—",
            "fusion": regime_names.get(voters["fusion"], str(voters["fusion"])),
        },
        "agreement_count": agreement_count,
        "agreement_pct":   float(agreement_pct),
        "dissenters":      dissenting,
        "level":           level,
    }


def _build_asset_payload(
    ticker: str, close: pd.Series, aux_bundle: dict
) -> Dict[str, Any]:
    """Build the per-asset JSON-ready dict."""

    # Features (causally-clean per compute_features_v2 docstring)
    features = compute_features_v2(
        close,
        vix=aux_bundle.get("vix"),
        vix3m=aux_bundle.get("vix3m"),
        vix6m=aux_bundle.get("vix6m"),
        vix9d=aux_bundle.get("vix9d"),
        skew=aux_bundle.get("skew"),
        vvix=aux_bundle.get("vvix"),
        tlt=aux_bundle.get("tlt"),
        gld=aux_bundle.get("gld"),
        term_spread=aux_bundle.get("term_spread"),
        credit_spread=aux_bundle.get("credit_spread"),
    )

    # Rule-baseline regime sequence
    rule_seq = compute_rule_regime_sequence(features)

    # TVTP-MSAR (the champion)
    tvtp = _compute_tvtp_probs(close)

    # GMM+HMM — unsupervised regime detector (complement to rule baseline)
    gmm = compute_gmm_hmm_sequence(close)

    # Align everything to the rule_seq index
    aligned = rule_seq.copy()
    aligned["close"] = close.reindex(aligned.index)
    if tvtp is not None:
        aligned["tvtp_low_vol"] = tvtp["p_low_vol"].reindex(aligned.index)
        aligned["tvtp_high_vol"] = tvtp["p_high_vol"].reindex(aligned.index)
        aligned["tvtp_position"] = tvtp["position"].reindex(aligned.index)
    else:
        aligned["tvtp_low_vol"] = np.nan
        aligned["tvtp_high_vol"] = np.nan
        aligned["tvtp_position"] = 0.0
    if gmm is not None:
        aligned["gmm_p0"] = gmm["p_0"].reindex(aligned.index)
        aligned["gmm_p1"] = gmm["p_1"].reindex(aligned.index)
        aligned["gmm_p2"] = gmm["p_2"].reindex(aligned.index)
        aligned["gmm_label"] = gmm["label"].reindex(aligned.index).fillna(1).astype(int)
    else:
        aligned["gmm_p0"] = np.nan
        aligned["gmm_p1"] = np.nan
        aligned["gmm_p2"] = np.nan
        aligned["gmm_label"] = 1

    # ------------------------------------------------------------------
    # Multi-model fusion (log-opinion-pool of GMM + TVTP) with empirical
    # mapping learned from rule_baseline label frequencies on this series.
    # ------------------------------------------------------------------
    if tvtp is not None and gmm is not None:
        from src.strategies.fusion import empirical_tvtp_3class_mapping
        mapping = empirical_tvtp_3class_mapping(
            tvtp[["p_low_vol", "p_high_vol"]],
            rule_seq["label"],
        )
        gmm_arr = aligned[["gmm_p0", "gmm_p1", "gmm_p2"]].fillna(1.0 / 3).to_numpy()
        tvtp_arr = aligned[["tvtp_low_vol", "tvtp_high_vol"]].fillna(0.5).to_numpy()
        fused = _log_opinion_pool(gmm_arr, tvtp_arr, mapping=mapping)
        aligned["fusion_p0"] = fused[:, 0]
        aligned["fusion_p1"] = fused[:, 1]
        aligned["fusion_p2"] = fused[:, 2]
        aligned["fusion_label"] = fused.argmax(axis=1).astype(int)
        # Shannon entropy normalised by log(K=3)
        eps = 1e-12
        h = -(fused * np.log(fused + eps)).sum(axis=1)
        aligned["fusion_entropy"] = h / np.log(3)
    else:
        aligned["fusion_p0"] = np.nan
        aligned["fusion_p1"] = np.nan
        aligned["fusion_p2"] = np.nan
        aligned["fusion_label"] = 1
        aligned["fusion_entropy"] = np.nan

    # ------------------------------------------------------------------
    # Walk-forward equity curves (each at unit notional, t-1 signal → t return)
    # ------------------------------------------------------------------
    log_ret = np.log(aligned["close"]).diff().fillna(0.0)

    # 1) TVTP-MSAR champion (log-space cumsum, position in [-0.3, +1.0])
    tvtp_pos_lag = aligned["tvtp_position"].shift(1).fillna(0.0)
    tvtp_returns = tvtp_pos_lag * log_ret
    tvtp_equity = np.exp(tvtp_returns.cumsum())

    # 2) Buy-and-hold (log-space cumsum)
    bh_equity = np.exp(log_ret.cumsum())

    # 3) Rule-baseline (log-space cumsum, position in [-0.3, +1.0])
    rule_pos_lag = aligned["position"].shift(1).fillna(0.0)
    rule_returns = rule_pos_lag * log_ret
    rule_equity = np.exp(rule_returns.cumsum())

    # Truncate to last HISTORY_BARS bars for the chart
    tail = aligned.tail(HISTORY_BARS)
    tail_tvtp_eq = tvtp_equity.tail(HISTORY_BARS)
    tail_rule_eq = rule_equity.tail(HISTORY_BARS)
    tail_bh_eq = bh_equity.tail(HISTORY_BARS)
    # Re-base equity curves to 1.0 at the start of the window
    if len(tail_tvtp_eq) > 0:
        tail_tvtp_eq = tail_tvtp_eq / tail_tvtp_eq.iloc[0]
        tail_rule_eq = tail_rule_eq / tail_rule_eq.iloc[0]
        tail_bh_eq = tail_bh_eq / tail_bh_eq.iloc[0]

    history = []
    for ts, row in tail.iterrows():
        date_str = ts.strftime("%Y-%m-%d")
        history.append({
            "date":  date_str,
            "close": _finite(row["close"]),
            "label": int(row["label"]),
            "regime": REGIME_NAMES[int(row["label"])],
            "alloc": _finite(row["position"]),
            "p0": _finite(row["p_0"]),
            "p1": _finite(row["p_1"]),
            "p2": _finite(row["p_2"]),
            "gmm_label": int(row["gmm_label"]),
            "gmm_p0": _finite(row["gmm_p0"]),
            "gmm_p1": _finite(row["gmm_p1"]),
            "gmm_p2": _finite(row["gmm_p2"]),
            "tvtp_low":  _finite(row["tvtp_low_vol"]),
            "tvtp_high": _finite(row["tvtp_high_vol"]),
            "tvtp_pos":  _finite(row["tvtp_position"]),
            "fusion_label": int(row["fusion_label"]),
            "fusion_p0": _finite(row["fusion_p0"]),
            "fusion_p1": _finite(row["fusion_p1"]),
            "fusion_p2": _finite(row["fusion_p2"]),
            "fusion_entropy": _finite(row["fusion_entropy"]),
            "eq_tvtp":  _finite(tail_tvtp_eq.get(ts, np.nan)),
            "eq_rule":  _finite(tail_rule_eq.get(ts, np.nan)),
            "eq_bh":    _finite(tail_bh_eq.get(ts, np.nan)),
        })

    # Latest-bar snapshot for the asset card
    last = aligned.iloc[-1]
    last_label = int(last["label"])
    last_probs = [
        _finite(last["p_0"]), _finite(last["p_1"]), _finite(last["p_2"]),
    ]
    last_tvtp_pos = _finite(last.get("tvtp_position", 0.0)) or 0.0
    last_tvtp_low = _finite(last.get("tvtp_low_vol", 0.0)) or 0.0
    last_tvtp_high = _finite(last.get("tvtp_high_vol", 0.0)) or 0.0

    last_gmm_label = int(last.get("gmm_label", 1))
    last_gmm_probs = [
        _finite(last.get("gmm_p0", 0.0)),
        _finite(last.get("gmm_p1", 0.0)),
        _finite(last.get("gmm_p2", 0.0)),
    ]

    # Transition matrix on the rule-baseline label sequence (3y)
    trans_window = aligned["label"].tail(min(len(aligned), 252 * 3)).astype(int)
    transition_matrix = _empirical_transition_matrix(trans_window.to_numpy(), n_states=3)

    return {
        "ticker": ticker,
        "name":   ASSET_NAMES.get(ticker, ticker),
        "as_of":  aligned.index[-1].strftime("%Y-%m-%d"),
        "current_close": _finite(last["close"]),
        "current_regime": {
            "label": last_label,
            "name":  REGIME_NAMES[last_label],
            "alloc": REGIME_ALLOC[last_label],
            "color": REGIME_COLORS[last_label],
            "probs": last_probs,
        },
        "current_tvtp": {
            "p_low_vol":  last_tvtp_low,
            "p_high_vol": last_tvtp_high,
            "position":   last_tvtp_pos,
            "state":      "Low-Vol Bull" if last_tvtp_pos > 0 else "High-Vol Defense",
        },
        "current_gmm": {
            "label": last_gmm_label,
            "name":  GMM_STATE_NAMES.get(last_gmm_label, "—"),
            "probs": last_gmm_probs,
        },
        "current_fusion": {
            "label": int(last.get("fusion_label", 1)),
            "name":  REGIME_NAMES[int(last.get("fusion_label", 1))],
            "probs": [
                _finite(last.get("fusion_p0")),
                _finite(last.get("fusion_p1")),
                _finite(last.get("fusion_p2")),
            ],
            "entropy": _finite(last.get("fusion_entropy")),
        },
        # Phase 2 — categorical confidence + cross-model consensus.
        # Confidence is derived from the fusion posterior (the multi-model
        # log-opinion-pool) because that's our best probabilistic estimate;
        # falls back to rule-baseline probs when fusion is unavailable.
        "current_confidence": _classify_confidence(
            probs=(
                [
                    _finite(last.get("fusion_p0")),
                    _finite(last.get("fusion_p1")),
                    _finite(last.get("fusion_p2")),
                ]
                if _finite(last.get("fusion_p0")) is not None
                else last_probs
            ),
            regime_names=REGIME_NAMES,
            entropy_normalised=_finite(last.get("fusion_entropy")),
        ),
        "model_consensus": _compute_model_consensus(
            rule_label=last_label,
            gmm_label=last_gmm_label,
            tvtp_low_vol=last_tvtp_low,
            tvtp_high_vol=last_tvtp_high,
            fusion_label=int(last.get("fusion_label", 1)),
            regime_names=REGIME_NAMES,
        ),
        # Phase 3 — heuristic transition-risk signal (model-free; uses the
        # fusion posterior history for margin compression + second-prob
        # acceleration, and fusion-label streaks for persistence-vs-typical).
        # The trained XGBoost detector in src.regime.transition_detector
        # remains the strategy-layer signal (it gates `transition_gated`
        # inside CPCV); these heuristics are the website-layer signal and
        # surface *why* the risk is elevated rather than a black-box score.
        "transition_risk": _build_transition_risk(aligned, REGIME_NAMES),
        "stats": ASSET_BACKTEST_STATS.get(ticker, {}),
        "transition_matrix": transition_matrix,
        "history": history,
    }


def _build_transition_risk(
    aligned: pd.DataFrame,
    regime_names: Dict[int, str],
) -> Dict[str, Any]:
    """Pick the best available probability+label series and hand them to
    ``compute_transition_risk``. Prefers fusion (multi-model posterior),
    falls back to rule-baseline when fusion didn't converge."""
    from src.regime.transition_detector import compute_transition_risk

    fusion_cols = ["fusion_p0", "fusion_p1", "fusion_p2"]
    if (all(c in aligned.columns for c in fusion_cols)
            and not aligned[fusion_cols].iloc[-1].isna().all()):
        prob_hist = aligned[fusion_cols].to_numpy(dtype=float)
        label_hist = aligned["fusion_label"].to_numpy(dtype=np.int64)
        source = "fusion"
    else:
        prob_hist = aligned[["p_0", "p_1", "p_2"]].to_numpy(dtype=float)
        label_hist = aligned["label"].to_numpy(dtype=np.int64)
        source = "rule_baseline"

    risk = compute_transition_risk(prob_hist, label_hist, regime_names=regime_names)
    risk["source_model"] = source
    return risk


def _empirical_transition_matrix(labels: np.ndarray, n_states: int) -> List[List[float]]:
    """Empirical p(s' | s) over the label sequence. Rows sum to 1."""
    counts = np.zeros((n_states, n_states), dtype=float)
    for i in range(1, len(labels)):
        a, b = int(labels[i - 1]), int(labels[i])
        if 0 <= a < n_states and 0 <= b < n_states:
            counts[a, b] += 1.0
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    probs = counts / row_sums
    return [[round(float(v), 4) for v in row] for row in probs]


def main(out_dir: Optional[Path] = None, backend: str = "yfinance") -> int:
    """Build dashboard JSON files.

    Parameters
    ----------
    out_dir : Path, optional
        Output directory (default: ``dashboard/public/data``).
    backend : str
        ``"yfinance"`` (default) — use cached yfinance data.
        ``"ibkr"``    — fetch live daily bars from IB Gateway (port 4004).
                        Requires ib_insync + a running IBKR Gateway.
    """
    import argparse as _ap
    _parser = _ap.ArgumentParser(add_help=False)
    _parser.add_argument("--backend", choices=["yfinance", "ibkr"], default=backend)
    _parser.add_argument("--out-dir", default=None)
    _args, _ = _parser.parse_known_args()
    backend = _args.backend
    if _args.out_dir is not None:
        out_dir = Path(_args.out_dir)

    if out_dir is None:
        out_dir = ROOT / "dashboard" / "public" / "data"
    regimes_dir = out_dir / "regimes"
    regimes_dir.mkdir(parents=True, exist_ok=True)

    if backend == "ibkr":
        from src.features.ibkr_daily import (  # noqa: PLC0415
            fetch_ibkr_aux_bundle,
            fetch_ibkr_daily,
        )
        _n_bars = 504  # ~2 years of daily bars from IBKR
        print(f"[snapshot] backend=ibkr  n_bars={_n_bars}")
        print("[snapshot] fetching aux bundle from IBKR + FRED...")
        aux_bundle = fetch_ibkr_aux_bundle(
            n_bars=_n_bars,
            fred_api_key=os.environ.get("FRED_API_KEY"),
        )
    else:
        cache_dir = ROOT / "data" / "cache"
        start = "2015-01-01"
        # yfinance treats `end` as exclusive — push it 1 day past today so we
        # capture today's bar once it settles after market close.
        end = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")
        # GHA runs in a fresh container with no cache; locally each day creates
        # a new parquet under `data/cache/`. Set REGIME_V2_NO_CACHE=1 to force
        # a re-fetch instead of relying on yesterday's cached file.
        if os.environ.get("REGIME_V2_NO_CACHE") == "1":
            cache_dir = None
        print(f"[snapshot] backend=yfinance  window: {start} → {end}  "
              f"cache={'on' if cache_dir else 'off'}")
        print("[snapshot] fetching aux bundle (VIX, VIX3M, TLT, GLD, FRED)...")
        aux_bundle = fetch_aux_data_bundle(
            start, end,
            cache_dir=cache_dir,
            fred_api_key=os.environ.get("FRED_API_KEY"),
        )

    summary_assets: List[Dict[str, Any]] = []
    for i, ticker in enumerate(DEFAULT_UNIVERSE, 1):
        print(f"[snapshot] [{i:>2}/{len(DEFAULT_UNIVERSE)}] {ticker}...", flush=True)
        try:
            if backend == "ibkr":
                close = fetch_ibkr_daily(ticker, n_bars=_n_bars)  # type: ignore[possibly-undefined]
            else:
                close = load_close(ticker, start, end, cache_dir=cache_dir)  # type: ignore[possibly-undefined]
        except Exception as exc:
            print(f"  ! load failed: {exc}")
            continue

        try:
            payload = _build_asset_payload(ticker, close, aux_bundle)
        except Exception as exc:
            print(f"  ! payload build failed: {exc}")
            import traceback
            traceback.print_exc()
            continue

        # Write per-asset
        safe = ticker.replace("=", "_").replace("/", "_")
        asset_path = regimes_dir / f"{safe}.json"
        with open(asset_path, "w") as fh:
            json.dump(payload, fh, separators=(",", ":"))
        print(f"  ok  {asset_path.relative_to(ROOT)}  "
              f"({asset_path.stat().st_size / 1024:.1f} KB)  "
              f"regime={payload['current_regime']['name']}  "
              f"tvtp_pos={payload['current_tvtp']['position']:+.2f}")

        # Summary entry
        summary_assets.append({
            "ticker": ticker,
            "safe":   safe,
            "name":   payload["name"],
            "as_of":  payload["as_of"],
            "close":  payload["current_close"],
            "regime": payload["current_regime"],
            "tvtp":   payload["current_tvtp"],
            "sharpe_p50": payload["stats"].get("sharpe_p50"),
            "max_dd_p50": payload["stats"].get("max_dd_p50"),
        })

    # assets.json — front-end uses this to render the universe selector
    assets_index = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "universe": [
            {"ticker": t, "safe": t.replace("=", "_").replace("/", "_"),
             "name": ASSET_NAMES.get(t, t)}
            for t in DEFAULT_UNIVERSE
        ],
    }
    with open(out_dir / "assets.json", "w") as fh:
        json.dump(assets_index, fh, indent=2)
    print(f"[snapshot] wrote {out_dir / 'assets.json'}")

    # Sanity gate: refuse to ship a near-empty summary. yfinance occasionally
    # throttles or fails all 10 tickers in one go, and on that day the daily
    # GHA used to commit an empty summary.json — wiping the dashboard's grid
    # until the next manual rebuild. Fail loudly so the GHA step fails and
    # the previous good summary stays on `main`.
    n_required = max(1, int(len(DEFAULT_UNIVERSE) * 0.7))  # ≥70% coverage
    if len(summary_assets) < n_required:
        print(
            f"[snapshot] ERROR: only {len(summary_assets)}/{len(DEFAULT_UNIVERSE)} "
            f"assets succeeded; need ≥ {n_required}. NOT writing summary.json "
            f"so the previous good snapshot stays on main.",
            file=sys.stderr,
        )
        return 1

    # summary.json — multi-asset grid + headline stats
    summary = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_assets": len(summary_assets),
        "regime_names": REGIME_NAMES,
        "regime_alloc": REGIME_ALLOC,
        "regime_colors": REGIME_COLORS,
        "assets": summary_assets,
    }
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[snapshot] wrote {out_dir / 'summary.json'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
