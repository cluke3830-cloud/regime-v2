"""Cross-asset market consensus aggregator.

Phase 4 of the regime-detection-website roadmap. The per-asset
``current_fusion`` field tells you what one asset's models think; this
module tells you what *the market* thinks by aggregating the fusion
labels across the entire universe.

Why this matters for the website
--------------------------------
Per-asset regime calls (SPY says Bull, TLT says Bear) are easy to
generate but lonely — a user looking at SPY=Bull doesn't know if that's
a market-wide signal or one asset on its own. Cross-asset consensus
exposes that distinction directly:

  "8 of 10 assets agree on Bull (strong consensus); dissenters: TLT, EEM"

For traders this is a sanity check on single-asset calls. For retail
users it's a piece of information they can't derive by skimming a grid.
For the website's positioning, it's the structural differentiator.

Composition with later phases
-----------------------------
Adding new assets (Phase 7 Forex, Phase 8 Crypto) automatically makes
consensus richer — the voter pool grows from 10 → 14 → 17 without any
change to this module.

Inputs
------
A list of per-asset payload dicts (each shaped like the ``_build_asset_payload``
output). The aggregator reads only a small subset of fields:
  - ticker
  - current_fusion.label, current_fusion.probs (preferred), OR
    current_regime.label as fallback when fusion is unavailable
  - current_confidence.score (Phase 2; used for weighted-confidence metric)

Level rules
-----------
  strong   : agreement_pct >= 0.80
  moderate : agreement_pct >= 0.60
  split    : agreement_pct >= 0.40
  divided  : agreement_pct <  0.40
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

DEFAULT_REGIME_NAMES: Dict[int, str] = {0: "Bull", 1: "Neutral", 2: "Bear"}


def _extract_asset_label(payload: Dict[str, Any]) -> Optional[int]:
    """Get the most authoritative regime label from a payload.

    Prefers ``current_fusion.label`` (multi-model log-opinion-pool), falls
    back to ``current_regime.label`` (rule baseline) when fusion is missing.
    Returns None when neither is available.
    """
    fusion = payload.get("current_fusion") or {}
    if fusion.get("label") is not None:
        return int(fusion["label"])
    regime = payload.get("current_regime") or {}
    if regime.get("label") is not None:
        return int(regime["label"])
    return None


def _extract_asset_probs(payload: Dict[str, Any]) -> Optional[List[float]]:
    """Get the most authoritative probability vector from a payload.

    Same precedence as ``_extract_asset_label``. Returns None when neither
    fusion nor rule probs are available, or when all entries are null.
    """
    fusion = payload.get("current_fusion") or {}
    fprobs = fusion.get("probs")
    if fprobs and any(p is not None for p in fprobs):
        return [float(p) if p is not None else 0.0 for p in fprobs]
    regime = payload.get("current_regime") or {}
    rprobs = regime.get("probs")
    if rprobs and any(p is not None for p in rprobs):
        return [float(p) if p is not None else 0.0 for p in rprobs]
    return None


def _categorize_consensus_level(agreement_pct: float) -> str:
    if agreement_pct >= 0.80:
        return "strong"
    if agreement_pct >= 0.60:
        return "moderate"
    if agreement_pct >= 0.40:
        return "split"
    return "divided"


def compute_market_consensus(
    asset_payloads: List[Dict[str, Any]],
    regime_names: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    """Aggregate per-asset regime labels into a market-wide consensus.

    Parameters
    ----------
    asset_payloads : list of dict
        Per-asset payload dicts (the same shape ``_build_asset_payload``
        produces). Assets whose label can't be extracted are skipped from
        the vote tally but listed in ``failed_extractions``.
    regime_names : dict, optional
        Override the 0/1/2 → name mapping. Defaults to
        ``{0: "Bull", 1: "Neutral", 2: "Bear"}``.

    Returns
    -------
    dict with keys:
        - regime               : str (winning regime name)
        - regime_label         : int (winning regime label)
        - level                : "strong" | "moderate" | "split" | "divided"
        - agreement_count      : int (# voting for winner)
        - agreement_pct        : float [0, 1]
        - n_assets             : int (total voters)
        - n_failed             : int (assets skipped from vote)
        - mean_confidence      : float | None (mean Phase-2 score among agreeing voters)
        - regime_counts        : { name: count } (full distribution)
        - voters               : [ { ticker, regime, regime_label,
                                     top_prob, agrees } ]
        - dissenters           : [ { ticker, regime } ] (voters disagreeing with winner)
        - failed_extractions   : [ ticker ] (assets with no usable label)
    """
    names = regime_names or DEFAULT_REGIME_NAMES

    voters: List[Dict[str, Any]] = []
    failed: List[str] = []

    for payload in asset_payloads:
        ticker = payload.get("ticker", "?")
        label = _extract_asset_label(payload)
        if label is None:
            failed.append(ticker)
            continue
        probs = _extract_asset_probs(payload)
        top_prob = max(probs) if probs else None
        voters.append({
            "ticker": ticker,
            "regime_label": int(label),
            "regime": names.get(int(label), str(label)),
            "top_prob": float(top_prob) if top_prob is not None else None,
        })

    if not voters:
        return {
            "regime": None,
            "regime_label": None,
            "level": "divided",
            "agreement_count": 0,
            "agreement_pct": 0.0,
            "n_assets": 0,
            "n_failed": len(failed),
            "mean_confidence": None,
            "regime_counts": {},
            "voters": [],
            "dissenters": [],
            "failed_extractions": failed,
        }

    label_counts = Counter(v["regime_label"] for v in voters)
    # Tie-break: when two regimes tie, pick the one with higher cumulative
    # top-prob — i.e., the regime whose voters were most certain.
    top_count = max(label_counts.values())
    contenders = [lbl for lbl, n in label_counts.items() if n == top_count]
    if len(contenders) == 1:
        winner_label = contenders[0]
    else:
        sums = {
            lbl: sum(v["top_prob"] or 0.0
                     for v in voters if v["regime_label"] == lbl)
            for lbl in contenders
        }
        winner_label = max(contenders, key=lambda lbl: sums[lbl])

    agreement_count = label_counts[winner_label]
    agreement_pct = agreement_count / len(voters)
    level = _categorize_consensus_level(agreement_pct)

    # Mean confidence (Phase 2 current_confidence.score) among AGREEING voters.
    agreeing_scores: List[float] = []
    for payload, voter in zip(asset_payloads, voters):
        # NOTE: zip pairs by position; works because we never reordered
        # asset_payloads vs voters and failed extractions are skipped from
        # voters but stay in asset_payloads. We need to re-map carefully.
        pass

    # Rebuild the mean-confidence calculation cleanly: only over voters
    # (which already exclude failed extractions) and only over the winning
    # regime.
    payload_by_ticker = {p.get("ticker", "?"): p for p in asset_payloads}
    for voter in voters:
        if voter["regime_label"] != winner_label:
            continue
        p = payload_by_ticker.get(voter["ticker"], {})
        conf = (p.get("current_confidence") or {}).get("score")
        if conf is not None:
            agreeing_scores.append(float(conf))
    mean_confidence = (
        float(sum(agreeing_scores) / len(agreeing_scores))
        if agreeing_scores else None
    )

    # Stamp each voter with whether they agree
    for v in voters:
        v["agrees"] = (v["regime_label"] == winner_label)

    dissenters = [
        {"ticker": v["ticker"], "regime": v["regime"]}
        for v in voters if not v["agrees"]
    ]

    regime_counts = {
        names.get(lbl, str(lbl)): int(n) for lbl, n in label_counts.items()
    }
    # Ensure all regime names are present (0-count makes UI tables stable)
    for lbl, name in names.items():
        regime_counts.setdefault(name, 0)

    return {
        "regime": names.get(int(winner_label), str(winner_label)),
        "regime_label": int(winner_label),
        "level": level,
        "agreement_count": int(agreement_count),
        "agreement_pct": float(agreement_pct),
        "n_assets": int(len(voters)),
        "n_failed": int(len(failed)),
        "mean_confidence": mean_confidence,
        "regime_counts": regime_counts,
        "voters": voters,
        "dissenters": dissenters,
        "failed_extractions": failed,
    }


__all__ = [
    "compute_market_consensus",
    "DEFAULT_REGIME_NAMES",
]