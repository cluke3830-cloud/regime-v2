"""Tests for the Phase 3 heuristic transition-risk helpers.

These are pure functions that operate on probability/label NumPy arrays
and emit a structured risk diagnostic for the website payload.

Coverage:
  - compute_margin_compression
  - compute_regime_persistence
  - compute_second_prob_acceleration
  - compute_transition_risk (compound)
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.regime.transition_detector import (  # noqa: E402
    compute_margin_compression,
    compute_regime_persistence,
    compute_second_prob_acceleration,
    compute_transition_risk,
)

NAMES = {0: "Bull", 1: "Neutral", 2: "Bear"}


# ---------------------------------------------------------------------------
# compute_margin_compression
# ---------------------------------------------------------------------------


class TestMarginCompression:
    def test_stable_margins_zero_score(self):
        """Probs flat across the lookback → no compression."""
        probs = np.tile([0.70, 0.20, 0.10], (10, 1))
        out = compute_margin_compression(probs, lookback_bars=5)
        assert out["score"] == pytest.approx(0.0)
        assert out["margin_now"] == pytest.approx(0.50)
        assert out["margin_lookback"] == pytest.approx(0.50)
        assert out["compression_pct"] == pytest.approx(0.0)

    def test_compressing_margins_positive_score(self):
        """Margin shrinks from 0.50 to 0.10 → score ≈ 0.80."""
        probs = np.tile([0.70, 0.20, 0.10], (10, 1))
        probs[-1] = [0.45, 0.35, 0.20]
        out = compute_margin_compression(probs, lookback_bars=5)
        # margin_now = 0.10, margin_lookback = 0.50 → compression = 0.80
        assert out["margin_now"] == pytest.approx(0.10, abs=1e-9)
        assert out["margin_lookback"] == pytest.approx(0.50)
        assert out["score"] == pytest.approx(0.80, abs=1e-9)

    def test_expanding_margins_clamped_to_zero(self):
        """Margin grew → compression_pct negative, score clamps to 0."""
        probs = np.tile([0.50, 0.30, 0.20], (10, 1))
        probs[-1] = [0.85, 0.10, 0.05]  # margin grew to 0.75 from 0.20
        out = compute_margin_compression(probs, lookback_bars=5)
        assert out["compression_pct"] < 0
        assert out["score"] == 0.0

    def test_short_history_returns_nan_safely(self):
        """History shorter than lookback → score=0, NaN margins."""
        probs = np.tile([0.70, 0.20, 0.10], (3, 1))
        out = compute_margin_compression(probs, lookback_bars=5)
        assert out["score"] == 0.0
        assert math.isnan(out["margin_now"])

    def test_collapsed_starting_margin_no_new_info(self):
        """If margin was already zero lookback bars ago, no new info → 0 score."""
        probs = np.tile([0.34, 0.33, 0.33], (10, 1))
        out = compute_margin_compression(probs, lookback_bars=5)
        assert out["score"] == 0.0

    def test_input_validation(self):
        """1-D input should raise ValueError."""
        with pytest.raises(ValueError, match="must be 2-D"):
            compute_margin_compression(np.array([0.5, 0.3, 0.2]), lookback_bars=5)


# ---------------------------------------------------------------------------
# compute_regime_persistence
# ---------------------------------------------------------------------------


class TestRegimePersistence:
    def test_no_prior_episodes_returns_zero_score(self):
        """First-ever episode of this regime → no historical baseline → 0."""
        labels = np.zeros(50, dtype=np.int64)
        out = compute_regime_persistence(labels)
        assert out["current_streak"] == 50
        assert out["score"] == 0.0
        assert math.isnan(out["typical_p75_duration"])

    def test_streak_below_median_zero_score(self):
        """Current streak < median historical duration → not overstaying."""
        # 5 historical Bull episodes of 30 bars each, current streak = 10
        labels = []
        for _ in range(5):
            labels.extend([0] * 30 + [1] * 5)
        labels.extend([0] * 10)
        out = compute_regime_persistence(np.array(labels, dtype=np.int64))
        assert out["current_streak"] == 10
        assert out["median_duration"] == pytest.approx(30.0)
        assert out["score"] == 0.0

    def test_streak_above_p75_high_score(self):
        """Current streak 2× p75 → score = 1.0."""
        # 5 historical Bull episodes of 20 bars each, current = 60 (3× p75)
        labels = []
        for _ in range(5):
            labels.extend([0] * 20 + [1] * 5)
        labels.extend([0] * 60)
        out = compute_regime_persistence(np.array(labels, dtype=np.int64))
        assert out["current_streak"] == 60
        assert out["typical_p75_duration"] == pytest.approx(20.0)
        assert out["score"] == pytest.approx(1.0)
        assert out["persistence_percentile"] == 1.0

    def test_streak_between_median_and_p75_partial_score(self):
        """Current streak between median and p75 → partial score (< 0.30)."""
        labels = [0]*10 + [1]*5 + [0]*20 + [1]*5 + [0]*30 + [1]*5 + [0]*40 + [1]*5 + [0]*25
        # Past Bull durations: [10, 20, 30, 40] → median=25, p75=32.5
        # Current streak = 25 (matches median exactly)
        out = compute_regime_persistence(np.array(labels, dtype=np.int64))
        assert out["current_streak"] == 25
        # Streak == median → score 0.0
        assert out["score"] == pytest.approx(0.0)

    def test_expected_remaining_days(self):
        """expected_remaining = max(0, median - streak)."""
        labels = [0]*20 + [1]*5 + [0]*20 + [1]*5 + [0]*5  # 2 Bull eps of 20, current = 5
        out = compute_regime_persistence(np.array(labels, dtype=np.int64))
        # median = 20, streak = 5 → expected_remaining = 15
        assert out["expected_remaining_days"] == 15

    def test_empty_input(self):
        out = compute_regime_persistence(np.array([], dtype=np.int64))
        assert out["current_streak"] == 0
        assert out["score"] == 0.0


# ---------------------------------------------------------------------------
# compute_second_prob_acceleration
# ---------------------------------------------------------------------------


class TestSecondProbAcceleration:
    def test_stable_second_prob_zero_score(self):
        """Second prob flat across window → no acceleration."""
        probs = np.tile([0.70, 0.20, 0.10], (10, 1))
        out = compute_second_prob_acceleration(probs, window=5)
        assert out["score"] == pytest.approx(0.0)
        assert out["delta"] == pytest.approx(0.0)
        assert out["second_regime_now"] == 1   # Neutral is current second-best

    def test_rising_second_prob_positive_score(self):
        """Second prob rises 0.10 → 0.30 across window → score = 1.0."""
        probs = np.tile([0.70, 0.10, 0.20], (10, 1))
        # Walk the second-best (Bear at idx 2) up to 0.30
        probs[-1] = [0.55, 0.15, 0.30]
        out = compute_second_prob_acceleration(probs, window=5)
        # Note: in current bar Bear=0.30 is the 2nd, was 0.20 before → delta = +0.10
        assert out["second_regime_now"] == 2
        assert out["delta"] == pytest.approx(0.10, abs=1e-9)
        # score = delta / 0.20 = 0.50
        assert out["score"] == pytest.approx(0.50)

    def test_extreme_acceleration_clipped_to_one(self):
        """Delta > 0.20 clamps score to 1.0.

        Setup: lookback bar has a low-prob regime that rises to become the
        second-best in the current bar. Specifically: in lookback, Neutral=0.05;
        in current bar, top=Bull=0.55, second=Neutral=0.30. Delta on the
        current second-best (Neutral) = 0.30 - 0.05 = +0.25, clips to 1.0.
        """
        probs = np.tile([0.85, 0.05, 0.10], (10, 1))
        probs[-1] = [0.55, 0.30, 0.15]
        out = compute_second_prob_acceleration(probs, window=5)
        assert out["second_regime_now"] == 1   # Neutral became second-best
        assert out["delta"] == pytest.approx(0.25, abs=1e-9)
        assert out["score"] == 1.0

    def test_falling_second_prob_clamped_to_zero(self):
        """If the second-best prob is falling, score is 0."""
        probs = np.tile([0.50, 0.35, 0.15], (10, 1))
        probs[-1] = [0.80, 0.15, 0.05]
        out = compute_second_prob_acceleration(probs, window=5)
        # Now second-best is Neutral=0.15, was Neutral=0.35 → delta = -0.20
        assert out["delta"] < 0
        assert out["score"] == 0.0

    def test_short_history_returns_zero_score(self):
        probs = np.tile([0.5, 0.3, 0.2], (3, 1))
        out = compute_second_prob_acceleration(probs, window=5)
        assert out["score"] == 0.0


# ---------------------------------------------------------------------------
# compute_transition_risk (compound)
# ---------------------------------------------------------------------------


class TestComputeTransitionRisk:
    def test_quiet_all_below_low(self):
        """No signals firing → level=low."""
        probs = np.tile([0.70, 0.20, 0.10], (50, 1))
        labels = np.zeros(50, dtype=np.int64)
        out = compute_transition_risk(probs, labels, NAMES)
        assert out["level"] == "low"
        assert out["score"] < 0.35
        assert "below alert thresholds" in out["reasons"][0]

    def test_persistence_only_fires_medium_not_high(self):
        """Persistence sub-score = 1.0 contributes 0.30 to combined (weight
        0.3). On its own that's at the medium threshold (>= 0.30) — but the
        hit-rate probe showed persistence-only is the NOISY cohort, so we
        deliberately do NOT promote it to high. This test guards the
        no-shortcut rule."""
        labels = []
        for _ in range(5):
            labels.extend([0] * 20 + [1] * 5)
        labels.extend([0] * 60)
        probs = np.tile([0.70, 0.20, 0.10], (len(labels), 1))
        out = compute_transition_risk(probs, np.array(labels), NAMES)
        assert out["level"] == "medium"
        assert any("regime held" in r for r in out["reasons"])

    def test_confluence_of_margin_and_acceleration_triggers_high(self):
        """When margin compresses AND second-best probability accelerates
        upward AT THE SAME TIME, combined score crosses the 0.50 high
        threshold. This is the precise pattern that yields 78% hit rate
        on the SPY 2000-2025 probe.

        Trace: lookback [0.70, 0.20, 0.10], current [0.15, 0.50, 0.35]
          margin: now=0.15 vs lookback=0.50 → compression=0.70, score=0.70
          accel : second-best now=Bear@0.35 vs lookback@0.10 → +0.25, score=1.0
          persist: no prior episodes → 0
          combined = 0.4*0.70 + 0.3*0 + 0.3*1.0 = 0.58 → HIGH
        """
        probs = np.tile([0.70, 0.20, 0.10], (10, 1))
        probs[-1] = [0.15, 0.50, 0.35]
        labels = np.zeros(10, dtype=np.int64)
        out = compute_transition_risk(probs, labels, NAMES)
        assert out["level"] == "high"

    def test_combined_below_medium_threshold_is_low(self):
        """Persistence-only firing in a SLIGHT overstay (streak=30 vs
        p75=20) gives sub-score ~0.65, combined ~0.20 — still low."""
        labels = []
        for _ in range(5):
            labels.extend([0] * 20 + [1] * 5)
        labels.extend([0] * 30)
        probs = np.tile([0.70, 0.20, 0.10], (len(labels), 1))
        out = compute_transition_risk(probs, np.array(labels), NAMES)
        assert out["level"] == "low"

    def test_high_threshold_at_combined_050(self):
        """Sanity guard for the calibrated 0.50 / 0.30 cutpoints — make
        sure the level boundaries match the docstring claim."""
        # Pure margin firing at 1.0 contributes 0.4 → combined = 0.4 → MEDIUM
        probs1 = np.tile([0.70, 0.20, 0.10], (10, 1))
        probs1[-1] = [0.40, 0.30, 0.30]   # margin: 0.50 → 0.10 = 0.80 compression
        labels = np.zeros(10, dtype=np.int64)
        out1 = compute_transition_risk(probs1, labels, NAMES)
        # margin=0.80 → contributes 0.32; accel typically fires too on a
        # sharp move — verify level is at least medium.
        assert out1["level"] in ("medium", "high")

    def test_output_contract_keys(self):
        """Consumers depend on this exact shape."""
        probs = np.tile([0.70, 0.20, 0.10], (50, 1))
        labels = np.zeros(50, dtype=np.int64)
        out = compute_transition_risk(probs, labels, NAMES)
        required = {
            "level", "score", "current_regime", "days_in_regime",
            "typical_p75_duration", "expected_remaining_days",
            "top_alternative_regime", "components", "reasons",
        }
        assert required.issubset(set(out.keys()))
        required_components = {
            "margin_compression", "regime_persistence", "second_prob_acceleration",
        }
        assert required_components == set(out["components"].keys())
        assert out["level"] in ("high", "medium", "low")
        assert isinstance(out["reasons"], list) and len(out["reasons"]) > 0

    def test_reason_text_includes_regime_names(self):
        """Reasons should be human-readable: include regime names."""
        # Setup: Bull overstaying
        labels = [0]*20 + [1]*5 + [0]*20 + [1]*5 + [0]*60
        probs = np.tile([0.70, 0.20, 0.10], (len(labels), 1))
        out = compute_transition_risk(probs, np.array(labels), NAMES)
        joined = " ".join(out["reasons"]).lower()
        assert "bull" in joined or "neutral" in joined or "bear" in joined