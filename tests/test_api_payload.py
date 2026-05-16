"""Tests for the Phase 2 Probability API additions to build_dashboard_data.py.

Covers the three module-level helpers that produce the new
``current_confidence`` and ``model_consensus`` payload fields:
  - ``_classify_confidence``
  - ``_compute_model_consensus``
  - ``_tvtp_to_3class_label``

These are independently unit-testable (pure functions, no IO, no model fits)
so we exercise them with explicit probability vectors instead of running
the full CPCV pipeline. The full-payload smoke is covered by
``scripts/build_dashboard_data.py`` runs.
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_dashboard_data import (  # noqa: E402
    _classify_confidence,
    _compute_model_consensus,
    _tvtp_to_3class_label,
)

NAMES = {0: "Bull", 1: "Neutral", 2: "Bear"}


# ---------------------------------------------------------------------------
# _classify_confidence
# ---------------------------------------------------------------------------


class TestClassifyConfidence:
    def test_peaked_bull_is_high(self):
        """Top prob > 0.65 + second < 0.25 → high."""
        out = _classify_confidence([0.78, 0.15, 0.07], NAMES, entropy_normalised=0.35)
        assert out["level"] == "high"
        assert out["top_regime"] == "Bull"
        assert out["top_prob"] == pytest.approx(0.78)
        assert out["second_regime"] == "Neutral"
        assert out["second_prob"] == pytest.approx(0.15)
        assert out["margin"] == pytest.approx(0.63)
        # score = 1 - entropy
        assert out["score"] == pytest.approx(0.65, abs=1e-6)

    def test_clear_leader_with_high_second_is_medium(self):
        """Top prob > 0.50 but second prob ≥ 0.25 → medium (not high)."""
        out = _classify_confidence([0.55, 0.30, 0.15], NAMES, entropy_normalised=0.81)
        assert out["level"] == "medium"
        assert out["top_regime"] == "Bull"
        assert out["margin"] == pytest.approx(0.25)

    def test_just_over_threshold_with_low_second_is_high(self):
        """Boundary: top=0.66, second=0.20 → high (both clauses met)."""
        out = _classify_confidence([0.66, 0.20, 0.14], NAMES, entropy_normalised=0.55)
        assert out["level"] == "high"

    def test_ambiguous_three_way_is_low(self):
        """All three regimes near-equal → low."""
        out = _classify_confidence([0.40, 0.35, 0.25], NAMES, entropy_normalised=0.96)
        assert out["level"] == "low"
        assert out["margin"] == pytest.approx(0.05)
        # Entropy near 1 → score near 0
        assert out["score"] == pytest.approx(0.04, abs=1e-6)

    def test_exactly_uniform_is_low(self):
        """Uniform (1/3, 1/3, 1/3) is the maximum-uncertainty case."""
        out = _classify_confidence([1/3, 1/3, 1/3], NAMES, entropy_normalised=1.0)
        assert out["level"] == "low"
        assert out["score"] == pytest.approx(0.0, abs=1e-6)

    def test_bear_dominant_is_high(self):
        """Regime classification is symmetric across the three classes."""
        out = _classify_confidence([0.05, 0.10, 0.85], NAMES, entropy_normalised=0.30)
        assert out["level"] == "high"
        assert out["top_regime"] == "Bear"
        assert out["second_regime"] == "Neutral"

    def test_none_probs_falls_back_safely(self):
        """All-None inputs → low confidence with explanatory reason."""
        out = _classify_confidence([None, None, None], NAMES, entropy_normalised=None)
        assert out["level"] == "low"
        assert out["top_prob"] is None
        assert "no probability" in out["reason"].lower()

    def test_partial_none_uses_remaining_values(self):
        """Missing one prob → ignore None, classify from what remains."""
        out = _classify_confidence([0.70, None, 0.10], NAMES, entropy_normalised=0.40)
        assert out["top_regime"] == "Bull"
        assert out["second_regime"] == "Bear"
        # Note: with second_prob = 0.10 < 0.25 AND top > 0.65 → high
        assert out["level"] == "high"

    def test_score_falls_back_to_top_prob_when_entropy_missing(self):
        """When entropy is None, score := top_prob (least-bad proxy)."""
        out = _classify_confidence([0.62, 0.28, 0.10], NAMES, entropy_normalised=None)
        assert out["score"] == pytest.approx(0.62)

    def test_score_clamped_to_unit_interval(self):
        """Score must be in [0, 1] even if entropy is malformed."""
        out_neg = _classify_confidence(
            [0.5, 0.3, 0.2], NAMES, entropy_normalised=1.5
        )  # entropy > 1 → score should clamp to 0
        out_pos = _classify_confidence(
            [0.5, 0.3, 0.2], NAMES, entropy_normalised=-0.5
        )  # entropy < 0 → score should clamp to 1
        assert 0.0 <= out_neg["score"] <= 1.0
        assert 0.0 <= out_pos["score"] <= 1.0

    def test_reason_includes_named_regimes(self):
        """Human-readable reason mentions both regime names + margin."""
        out = _classify_confidence([0.70, 0.20, 0.10], NAMES, entropy_normalised=0.50)
        assert "Bull" in out["reason"]
        assert "Neutral" in out["reason"]
        assert "0.50" in out["reason"]   # margin
        assert "entropy=0.50" in out["reason"]


# ---------------------------------------------------------------------------
# _tvtp_to_3class_label
# ---------------------------------------------------------------------------


class TestTvtpTo3ClassLabel:
    @pytest.mark.parametrize("p_low,p_high,expected", [
        # Threshold semantics: impl uses strict `top < 0.60` for Neutral,
        # so exactly 0.60 still labels as the argmax regime.
        (0.9, 0.1, 0),     # dominant low-vol → Bull
        (0.7, 0.3, 0),     # > 0.60 → Bull
        (0.6, 0.4, 0),     # exactly at threshold → Bull (strict `<` semantics)
        (0.59, 0.41, 1),   # just below threshold → Neutral
        (0.5, 0.5, 1),     # split → Neutral
        (0.41, 0.59, 1),   # symmetric just below threshold → Neutral
        (0.4, 0.6, 2),     # exactly at threshold → Bear (argmax wins at =)
        (0.3, 0.7, 2),     # > 0.60 → Bear
        (0.05, 0.95, 2),   # dominant high-vol → Bear
    ])
    def test_mapping(self, p_low, p_high, expected):
        assert _tvtp_to_3class_label(p_low, p_high) == expected

    def test_none_inputs_returns_none(self):
        assert _tvtp_to_3class_label(None, None) is None

    def test_one_none_treats_other_as_dominant(self):
        # If p_high is None and p_low=0.9, return Bull
        assert _tvtp_to_3class_label(0.9, None) == 0
        # If p_low is None and p_high=0.9, return Bear
        assert _tvtp_to_3class_label(None, 0.9) == 2


# ---------------------------------------------------------------------------
# _compute_model_consensus
# ---------------------------------------------------------------------------


class TestModelConsensus:
    def test_unanimous_when_all_agree(self):
        """rule=0 gmm=0 tvtp=0 fusion=0 → unanimous."""
        out = _compute_model_consensus(
            rule_label=0, gmm_label=0,
            tvtp_low_vol=0.9, tvtp_high_vol=0.1,   # maps to 0
            fusion_label=0, regime_names=NAMES,
        )
        assert out["level"] == "unanimous"
        assert out["agreement_count"] == 3
        assert out["agreement_pct"] == pytest.approx(1.0)
        assert out["dissenters"] == []
        assert out["models"]["fusion"] == 0
        assert out["model_names"]["fusion"] == "Bull"

    def test_strong_when_two_of_three_agree(self):
        """rule=Bull gmm=Bull tvtp=Bear fusion=Bull → strong (2 of 3)."""
        out = _compute_model_consensus(
            rule_label=0, gmm_label=0,
            tvtp_low_vol=0.1, tvtp_high_vol=0.9,   # maps to 2 Bear
            fusion_label=0, regime_names=NAMES,
        )
        assert out["level"] == "strong"
        assert out["agreement_count"] == 2
        assert out["agreement_pct"] == pytest.approx(2/3)
        assert out["dissenters"] == ["tvtp"]

    def test_split_when_one_of_three_agrees(self):
        """rule=Bull gmm=Neutral tvtp=Bear fusion=Bull → split."""
        out = _compute_model_consensus(
            rule_label=0, gmm_label=1,
            tvtp_low_vol=0.1, tvtp_high_vol=0.9,   # maps to 2
            fusion_label=0, regime_names=NAMES,
        )
        assert out["level"] == "split"
        assert out["agreement_count"] == 1
        assert sorted(out["dissenters"]) == ["gmm", "tvtp"]

    def test_divided_when_zero_agree(self):
        """rule=Bull gmm=Neutral tvtp=Bear fusion=Bear (anti-fusion) → 1/3.

        Constructing a true 0/3: fusion=Neutral, rule=Bull, gmm=Bear,
        tvtp=Bull → 0 agree."""
        out = _compute_model_consensus(
            rule_label=0, gmm_label=2,
            tvtp_low_vol=0.9, tvtp_high_vol=0.1,   # maps to 0 (Bull)
            fusion_label=1, regime_names=NAMES,
        )
        assert out["level"] == "divided"
        assert out["agreement_count"] == 0
        assert sorted(out["dissenters"]) == ["gmm", "rule", "tvtp"]

    def test_tvtp_neutral_when_split(self):
        """TVTP p_low ≈ p_high → maps to Neutral.

        If fusion is Neutral too, TVTP agrees → 1 vote toward consensus."""
        out = _compute_model_consensus(
            rule_label=1, gmm_label=1,
            tvtp_low_vol=0.5, tvtp_high_vol=0.5,   # maps to 1 Neutral
            fusion_label=1, regime_names=NAMES,
        )
        assert out["level"] == "unanimous"
        assert out["models"]["tvtp"] == 1

    def test_tvtp_none_when_both_inputs_none(self):
        """When TVTP probs are missing, tvtp model entry is None and
        doesn't count as either agreeing or dissenting."""
        out = _compute_model_consensus(
            rule_label=0, gmm_label=0,
            tvtp_low_vol=None, tvtp_high_vol=None,
            fusion_label=0, regime_names=NAMES,
        )
        # rule + gmm agree with fusion; tvtp is None (skip)
        assert out["models"]["tvtp"] is None
        assert out["agreement_count"] == 2   # rule + gmm
        # Denominator excludes tvtp when None
        assert out["agreement_pct"] == pytest.approx(1.0)
        # Dissenters list also excludes None entries
        assert out["dissenters"] == []

    def test_model_names_field_human_readable(self):
        out = _compute_model_consensus(
            rule_label=2, gmm_label=2,
            tvtp_low_vol=0.1, tvtp_high_vol=0.9,
            fusion_label=2, regime_names=NAMES,
        )
        assert out["model_names"] == {
            "rule": "Bear", "gmm": "Bear",
            "tvtp": "Bear", "fusion": "Bear",
        }


# ---------------------------------------------------------------------------
# Sanity: the helpers feed into _build_asset_payload contract
# ---------------------------------------------------------------------------


def test_confidence_contract_keys():
    """Whoever consumes current_confidence depends on this exact shape."""
    out = _classify_confidence([0.7, 0.2, 0.1], NAMES, entropy_normalised=0.5)
    assert set(out.keys()) == {
        "level", "score",
        "top_regime", "top_prob",
        "second_regime", "second_prob",
        "margin", "reason",
    }
    assert out["level"] in ("high", "medium", "low")


def test_consensus_contract_keys():
    """Whoever consumes model_consensus depends on this exact shape."""
    out = _compute_model_consensus(
        rule_label=0, gmm_label=0,
        tvtp_low_vol=0.9, tvtp_high_vol=0.1,
        fusion_label=0, regime_names=NAMES,
    )
    assert set(out.keys()) == {
        "models", "model_names",
        "agreement_count", "agreement_pct",
        "dissenters", "level",
    }
    assert set(out["models"].keys()) == {"rule", "gmm", "tvtp", "fusion"}
    assert set(out["model_names"].keys()) == {"rule", "gmm", "tvtp", "fusion"}
    assert out["level"] in ("unanimous", "strong", "split", "divided")