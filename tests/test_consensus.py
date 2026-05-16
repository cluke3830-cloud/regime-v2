"""Tests for src/regime/consensus.py (Phase 4).

Pure-function tests with synthetic per-asset payload dicts. Real
end-to-end correctness is covered by the dashboard snapshot pipeline
(which runs the same function on real fusion outputs).
"""
from __future__ import annotations
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.regime.consensus import compute_market_consensus  # noqa: E402


def _mk(ticker: str, label: int, top_prob: float = 0.7,
        confidence: float | None = None,
        use_rule_fallback: bool = False) -> dict:
    """Build a minimal asset payload for consensus testing."""
    payload: dict = {"ticker": ticker}
    if use_rule_fallback:
        # No current_fusion → fall back to current_regime
        probs = [0.0, 0.0, 0.0]
        probs[label] = top_prob
        # distribute the rest
        remainder = (1.0 - top_prob) / 2
        for i in range(3):
            if i != label:
                probs[i] = remainder
        payload["current_regime"] = {"label": label, "probs": probs}
    else:
        probs = [0.0, 0.0, 0.0]
        probs[label] = top_prob
        remainder = (1.0 - top_prob) / 2
        for i in range(3):
            if i != label:
                probs[i] = remainder
        payload["current_fusion"] = {"label": label, "probs": probs}
    if confidence is not None:
        payload["current_confidence"] = {"score": confidence}
    return payload


# ---------------------------------------------------------------------------
# Level rules
# ---------------------------------------------------------------------------


class TestConsensusLevels:
    def test_strong_at_80pct(self):
        """8 of 10 agreeing → strong consensus."""
        payloads = [_mk(f"T{i}", 0) for i in range(8)]
        payloads += [_mk("X", 1), _mk("Y", 2)]
        c = compute_market_consensus(payloads)
        assert c["level"] == "strong"
        assert c["agreement_count"] == 8
        assert c["agreement_pct"] == pytest.approx(0.80)
        assert c["regime"] == "Bull"

    def test_moderate_at_70pct(self):
        """7 of 10 agreeing → moderate."""
        payloads = [_mk(f"T{i}", 1) for i in range(7)]
        payloads += [_mk(f"D{i}", 0) for i in range(3)]
        c = compute_market_consensus(payloads)
        assert c["level"] == "moderate"
        assert c["regime"] == "Neutral"

    def test_split_at_50pct(self):
        """5 of 10 agreeing → split."""
        payloads = [_mk(f"T{i}", 2) for i in range(5)]
        payloads += [_mk(f"X{i}", 0) for i in range(3)]
        payloads += [_mk(f"Y{i}", 1) for i in range(2)]
        c = compute_market_consensus(payloads)
        assert c["level"] == "split"
        assert c["regime"] == "Bear"
        assert c["agreement_count"] == 5

    def test_divided_below_40pct(self):
        """3 of 10 → divided."""
        # 3 Bull, 3 Neutral, 3 Bear, 1 Bull → Bull wins with 4/10 = 40% (split)
        # Need stricter divided case: 3 Bull, 4 Neutral, 3 Bear → Neutral 40% (split)
        # True divided: 3 / 4 / 3 / 3 (impossible with 3 regimes — try N=12)
        payloads = [_mk(f"A{i}", 0) for i in range(3)]
        payloads += [_mk(f"B{i}", 1) for i in range(3)]
        payloads += [_mk(f"C{i}", 2) for i in range(3)]
        # winner: Bull (3/9 = 33%) — tie-break by sum_of_top_probs
        c = compute_market_consensus(payloads)
        assert c["level"] == "divided"
        assert c["agreement_pct"] == pytest.approx(3/9)


# ---------------------------------------------------------------------------
# Tie-breaking
# ---------------------------------------------------------------------------


class TestTieBreaking:
    def test_tie_broken_by_top_prob_sum(self):
        """When two regimes tie on count, pick the one with higher cumulative
        top_prob (i.e., voters were more certain)."""
        # 3 Bull voters with prob=0.55 each (sum=1.65)
        # 3 Bear voters with prob=0.85 each (sum=2.55) → Bear wins tie
        payloads = [_mk(f"B{i}", 0, top_prob=0.55) for i in range(3)]
        payloads += [_mk(f"X{i}", 2, top_prob=0.85) for i in range(3)]
        c = compute_market_consensus(payloads)
        assert c["regime"] == "Bear"
        assert c["agreement_count"] == 3


# ---------------------------------------------------------------------------
# Failed extractions
# ---------------------------------------------------------------------------


class TestFailedExtractions:
    def test_payload_without_fusion_falls_back_to_rule(self):
        """No current_fusion → use current_regime."""
        payloads = [
            _mk("A", 0),                              # uses fusion
            _mk("B", 0, use_rule_fallback=True),      # uses rule fallback
        ]
        c = compute_market_consensus(payloads)
        assert c["n_assets"] == 2
        assert c["agreement_count"] == 2
        assert c["n_failed"] == 0

    def test_completely_unparseable_payload_marked_as_failed(self):
        """No current_fusion AND no current_regime → skip + record."""
        payloads = [
            _mk("OK", 0),
            {"ticker": "BAD"},                        # no labels anywhere
        ]
        c = compute_market_consensus(payloads)
        assert c["n_assets"] == 1
        assert c["n_failed"] == 1
        assert "BAD" in c["failed_extractions"]

    def test_all_failed_returns_empty_consensus(self):
        payloads = [{"ticker": "A"}, {"ticker": "B"}]
        c = compute_market_consensus(payloads)
        assert c["n_assets"] == 0
        assert c["n_failed"] == 2
        assert c["regime"] is None
        assert c["level"] == "divided"


# ---------------------------------------------------------------------------
# Mean confidence
# ---------------------------------------------------------------------------


class TestMeanConfidence:
    def test_mean_confidence_only_over_agreeing_voters(self):
        """mean_confidence averages current_confidence.score over voters
        agreeing with the winning regime — dissenters' confidence is
        excluded."""
        # 3 Bull voters with confs 0.8, 0.6, 0.7 → mean = 0.70
        # 1 Bear voter (dissenter) with conf 0.95 — IGNORED
        payloads = [
            _mk("A", 0, confidence=0.8),
            _mk("B", 0, confidence=0.6),
            _mk("C", 0, confidence=0.7),
            _mk("D", 2, confidence=0.95),
        ]
        c = compute_market_consensus(payloads)
        assert c["regime"] == "Bull"
        assert c["mean_confidence"] == pytest.approx(0.70)

    def test_mean_confidence_none_when_no_voters_have_score(self):
        """If no agreeing voter exposed current_confidence, mean is None."""
        payloads = [_mk("A", 0), _mk("B", 0), _mk("C", 0)]
        c = compute_market_consensus(payloads)
        assert c["mean_confidence"] is None


# ---------------------------------------------------------------------------
# Shape / contract
# ---------------------------------------------------------------------------


class TestOutputContract:
    def test_required_keys_present(self):
        payloads = [_mk("A", 0), _mk("B", 1)]
        c = compute_market_consensus(payloads)
        required = {
            "regime", "regime_label", "level",
            "agreement_count", "agreement_pct",
            "n_assets", "n_failed", "mean_confidence",
            "regime_counts", "voters", "dissenters", "failed_extractions",
        }
        assert required.issubset(set(c.keys()))

    def test_regime_counts_includes_all_regimes_with_zero(self):
        """All three regime names should appear in regime_counts, even
        when no voter selected that regime (stable UI tables)."""
        payloads = [_mk(f"T{i}", 0) for i in range(5)]
        c = compute_market_consensus(payloads)
        assert set(c["regime_counts"].keys()) == {"Bull", "Neutral", "Bear"}
        assert c["regime_counts"]["Bull"] == 5
        assert c["regime_counts"]["Neutral"] == 0
        assert c["regime_counts"]["Bear"] == 0

    def test_voters_stamped_with_agrees_flag(self):
        payloads = [_mk("BULL1", 0), _mk("BULL2", 0), _mk("BEAR", 2)]
        c = compute_market_consensus(payloads)
        voters_by_t = {v["ticker"]: v for v in c["voters"]}
        assert voters_by_t["BULL1"]["agrees"] is True
        assert voters_by_t["BULL2"]["agrees"] is True
        assert voters_by_t["BEAR"]["agrees"] is False

    def test_dissenters_list_matches_voters(self):
        payloads = [_mk("BULL1", 0), _mk("BEAR", 2), _mk("NEU", 1)]
        c = compute_market_consensus(payloads)
        assert {d["ticker"] for d in c["dissenters"]} == {"BEAR", "NEU"}