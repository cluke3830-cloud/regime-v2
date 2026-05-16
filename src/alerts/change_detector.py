"""Detect regime changes between two summary.json snapshots (Phase 6).

Pure functions — no I/O, no side effects. Easy to test and to run in
GitHub Actions after each dashboard build.

Inputs
------
Two dicts shaped like ``summary.json``::

    {
      "generated_at": "2026-05-16T22:00:00Z",
      "assets": [
        {
          "ticker": "SPY",
          "name": "S&P 500",
          "as_of": "2026-05-16",
          "regime": {"label": 0, "name": "Bull", ...},
          ...
        }, ...
      ],
      "consensus": {
        "regime": "Bull",
        "regime_label": 0,
        "level": "strong",
        "agreement_count": 8,
        "n_assets": 10,
        ...
      }
    }

Output
------
A ``ChangeReport`` dict::

    {
      "generated_at": str,        # curr snapshot's generated_at
      "prev_date": str | None,    # prev snapshot's generated_at
      "curr_date": str | None,    # curr snapshot's generated_at
      "has_changes": bool,
      "asset_changes": [          # only assets that changed
        {
          "ticker": str,
          "name": str,
          "from_regime": str,
          "from_label": int,
          "to_regime": str,
          "to_label": int,
          "transition_risk": str | None,   # "high"/"medium"/"low"/None
        }
      ],
      "consensus_change": {       # None if consensus unchanged
        "from_regime": str | None,
        "from_level": str,
        "to_regime": str | None,
        "to_level": str,
      } | None,
      "no_change_tickers": [str], # assets that stayed the same
    }
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _regime_label(asset: Dict[str, Any]) -> Optional[int]:
    regime = asset.get("regime") or {}
    lbl = regime.get("label")
    return int(lbl) if lbl is not None else None


def _regime_name(asset: Dict[str, Any]) -> str:
    regime = asset.get("regime") or {}
    return str(regime.get("name", "Unknown"))


def _transition_risk_level(asset: Dict[str, Any]) -> Optional[str]:
    """Surface transition_risk.level if present in the full asset payload."""
    tr = asset.get("transition_risk") or {}
    lvl = tr.get("level")
    return str(lvl) if lvl else None


def detect_regime_changes(
    prev_summary: Dict[str, Any],
    curr_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare two summary.json snapshots and return a ChangeReport.

    Parameters
    ----------
    prev_summary : dict
        The summary.json from the previous build.
    curr_summary : dict
        The summary.json from the current build.

    Returns
    -------
    dict
        ChangeReport (see module docstring for shape).
    """
    prev_assets: List[Dict[str, Any]] = prev_summary.get("assets") or []
    curr_assets: List[Dict[str, Any]] = curr_summary.get("assets") or []

    prev_by_ticker = {a["ticker"]: a for a in prev_assets if "ticker" in a}
    curr_by_ticker = {a["ticker"]: a for a in curr_assets if "ticker" in a}

    asset_changes: List[Dict[str, Any]] = []
    no_change_tickers: List[str] = []

    for ticker, curr_asset in curr_by_ticker.items():
        prev_asset = prev_by_ticker.get(ticker)
        if prev_asset is None:
            # New asset — not a change event, just skip
            continue

        prev_label = _regime_label(prev_asset)
        curr_label = _regime_label(curr_asset)

        if prev_label is None or curr_label is None:
            continue

        if prev_label == curr_label:
            no_change_tickers.append(ticker)
            continue

        asset_changes.append({
            "ticker": ticker,
            "name": curr_asset.get("name", ticker),
            "from_regime": _regime_name(prev_asset),
            "from_label": int(prev_label),
            "to_regime": _regime_name(curr_asset),
            "to_label": int(curr_label),
            "transition_risk": _transition_risk_level(curr_asset),
        })

    # Consensus-level change detection
    prev_cons = prev_summary.get("consensus") or {}
    curr_cons = curr_summary.get("consensus") or {}

    prev_regime = prev_cons.get("regime")
    curr_regime = curr_cons.get("regime")
    prev_level = str(prev_cons.get("level", ""))
    curr_level = str(curr_cons.get("level", ""))

    consensus_changed = (prev_regime != curr_regime) or (prev_level != curr_level)
    consensus_change: Optional[Dict[str, Any]] = None
    if consensus_changed and (prev_cons or curr_cons):
        consensus_change = {
            "from_regime": prev_regime,
            "from_level": prev_level,
            "to_regime": curr_regime,
            "to_level": curr_level,
        }

    has_changes = bool(asset_changes) or consensus_change is not None

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "prev_date": prev_summary.get("generated_at"),
        "curr_date": curr_summary.get("generated_at"),
        "has_changes": has_changes,
        "asset_changes": asset_changes,
        "consensus_change": consensus_change,
        "no_change_tickers": sorted(no_change_tickers),
    }


__all__ = ["detect_regime_changes"]