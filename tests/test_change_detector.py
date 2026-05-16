"""Tests for src/alerts/change_detector.py (Phase 6)."""
from __future__ import annotations
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.alerts.change_detector import detect_regime_changes  # noqa: E402


def _regime(label: int) -> dict:
    names = {0: "Bull", 1: "Neutral", 2: "Bear"}
    return {"label": label, "name": names[label]}


def _asset(ticker: str, label: int, transition_risk: str | None = None) -> dict:
    a: dict = {"ticker": ticker, "name": f"Name-{ticker}", "regime": _regime(label)}
    if transition_risk:
        a["transition_risk"] = {"level": transition_risk}
    return a


def _consensus(regime: str | None, level: str) -> dict:
    return {"regime": regime, "level": level}


def _summary(assets: list, regime: str = "Bull", level: str = "moderate") -> dict:
    return {
        "generated_at": "2026-05-16T22:00:00Z",
        "assets": assets,
        "consensus": _consensus(regime, level),
    }


# ---------------------------------------------------------------------------
# No-change cases
# ---------------------------------------------------------------------------


class TestNoChanges:
    def test_identical_snapshots_no_asset_changes(self):
        assets = [_asset("SPY", 0), _asset("TLT", 2)]
        s = _summary(assets)
        r = detect_regime_changes(s, s)
        assert r["asset_changes"] == []
        assert r["consensus_change"] is None
        assert r["has_changes"] is False

    def test_no_change_tickers_populated(self):
        assets = [_asset("SPY", 0), _asset("QQQ", 0)]
        s = _summary(assets)
        r = detect_regime_changes(s, s)
        assert set(r["no_change_tickers"]) == {"SPY", "QQQ"}

    def test_has_changes_false_when_same(self):
        s = _summary([_asset("GLD", 1)])
        r = detect_regime_changes(s, s)
        assert r["has_changes"] is False


# ---------------------------------------------------------------------------
# Asset regime changes
# ---------------------------------------------------------------------------


class TestAssetChanges:
    def test_single_asset_change_detected(self):
        prev = _summary([_asset("SPY", 0)])  # Bull
        curr = _summary([_asset("SPY", 2)])  # Bear
        r = detect_regime_changes(prev, curr)
        assert len(r["asset_changes"]) == 1
        c = r["asset_changes"][0]
        assert c["ticker"] == "SPY"
        assert c["from_regime"] == "Bull"
        assert c["to_regime"] == "Bear"
        assert c["from_label"] == 0
        assert c["to_label"] == 2

    def test_multiple_changes_all_captured(self):
        prev = _summary([_asset("SPY", 0), _asset("TLT", 2), _asset("GLD", 1)])
        curr = _summary([_asset("SPY", 1), _asset("TLT", 1), _asset("GLD", 1)])
        r = detect_regime_changes(prev, curr)
        tickers = {c["ticker"] for c in r["asset_changes"]}
        assert tickers == {"SPY", "TLT"}
        assert r["no_change_tickers"] == ["GLD"]

    def test_transition_risk_surfaced_in_change(self):
        prev = _summary([_asset("SPY", 0)])
        curr = _summary([_asset("SPY", 1, transition_risk="high")])
        r = detect_regime_changes(prev, curr)
        assert r["asset_changes"][0]["transition_risk"] == "high"

    def test_no_transition_risk_is_none(self):
        prev = _summary([_asset("SPY", 0)])
        curr = _summary([_asset("SPY", 2)])
        r = detect_regime_changes(prev, curr)
        assert r["asset_changes"][0]["transition_risk"] is None

    def test_has_changes_true_on_asset_change(self):
        prev = _summary([_asset("SPY", 0)])
        curr = _summary([_asset("SPY", 2)])
        assert detect_regime_changes(prev, curr)["has_changes"] is True


# ---------------------------------------------------------------------------
# Consensus changes
# ---------------------------------------------------------------------------


class TestConsensusChanges:
    def test_consensus_regime_change_detected(self):
        prev = _summary([], regime="Bull", level="strong")
        curr = _summary([], regime="Bear", level="split")
        r = detect_regime_changes(prev, curr)
        assert r["consensus_change"] is not None
        assert r["consensus_change"]["from_regime"] == "Bull"
        assert r["consensus_change"]["to_regime"] == "Bear"

    def test_consensus_level_change_only(self):
        """Regime name same but level changes → consensus_change fires."""
        prev = _summary([], regime="Bull", level="strong")
        curr = _summary([], regime="Bull", level="moderate")
        r = detect_regime_changes(prev, curr)
        assert r["consensus_change"] is not None
        assert r["consensus_change"]["from_level"] == "strong"
        assert r["consensus_change"]["to_level"] == "moderate"

    def test_consensus_unchanged_is_none(self):
        prev = _summary([], regime="Neutral", level="split")
        curr = _summary([], regime="Neutral", level="split")
        r = detect_regime_changes(prev, curr)
        assert r["consensus_change"] is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_new_ticker_in_curr_not_in_prev_skipped(self):
        """New asset with no prev baseline → not reported as a change."""
        prev = _summary([_asset("SPY", 0)])
        curr = _summary([_asset("SPY", 0), _asset("NEW", 2)])
        r = detect_regime_changes(prev, curr)
        assert r["asset_changes"] == []

    def test_ticker_missing_from_curr_graceful(self):
        """Ticker in prev but not curr → no crash, just not in output."""
        prev = _summary([_asset("SPY", 0), _asset("TLT", 1)])
        curr = _summary([_asset("SPY", 0)])
        r = detect_regime_changes(prev, curr)
        assert r["asset_changes"] == []

    def test_empty_both_summaries(self):
        r = detect_regime_changes({"assets": [], "consensus": {}}, {"assets": [], "consensus": {}})
        assert r["has_changes"] is False
        assert r["asset_changes"] == []

    def test_output_keys_present(self):
        s = _summary([_asset("SPY", 0)])
        r = detect_regime_changes(s, s)
        required = {
            "generated_at", "prev_date", "curr_date",
            "has_changes", "asset_changes", "consensus_change", "no_change_tickers",
        }
        assert required.issubset(set(r.keys()))

    def test_prev_curr_dates_recorded(self):
        prev = {"generated_at": "2026-05-15T22:00:00Z", "assets": [], "consensus": {}}
        curr = {"generated_at": "2026-05-16T22:00:00Z", "assets": [], "consensus": {}}
        r = detect_regime_changes(prev, curr)
        assert r["prev_date"] == "2026-05-15T22:00:00Z"
        assert r["curr_date"] == "2026-05-16T22:00:00Z"
