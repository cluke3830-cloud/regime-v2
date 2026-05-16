"""Tests for src.validation.cost_model."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.validation.cost_model import ASSET_COST_BPS, CostModel  # noqa: E402
from src.validation.multi_asset import DEFAULT_UNIVERSE  # noqa: E402


# ---------------------------------------------------------------------------
# Asset table coverage
# ---------------------------------------------------------------------------


def test_all_universe_tickers_in_table():
    """Every ticker in the 10-asset universe must have an explicit cost entry."""
    missing = [t for t in DEFAULT_UNIVERSE if t not in ASSET_COST_BPS]
    assert missing == [], f"Missing cost entries: {missing}"


def test_default_key_present():
    assert "_default" in ASSET_COST_BPS
    assert ASSET_COST_BPS["_default"] > 0


def test_all_costs_positive():
    for ticker, bps in ASSET_COST_BPS.items():
        assert bps > 0, f"{ticker} cost must be positive"


def test_spy_cheaper_than_eem():
    assert ASSET_COST_BPS["SPY"] < ASSET_COST_BPS["EEM"]


def test_crypto_most_expensive():
    """Crypto assets should be the most expensive in the universe (widest spreads)."""
    crypto = {"BTC-USD", "ETH-USD", "SOL-USD"}
    non_crypto = [v for k, v in ASSET_COST_BPS.items() if k not in crypto | {"_default"}]
    assert max(ASSET_COST_BPS[t] for t in crypto) > max(non_crypto)


# ---------------------------------------------------------------------------
# CostModel.base_bps
# ---------------------------------------------------------------------------


def test_base_bps_known_ticker():
    assert CostModel("SPY").base_bps() == ASSET_COST_BPS["SPY"]


def test_base_bps_unknown_falls_back_to_default():
    assert CostModel("UNKNOWN_TICKER").base_bps() == ASSET_COST_BPS["_default"]


# ---------------------------------------------------------------------------
# CostModel.compute_tc — flat (no volume)
# ---------------------------------------------------------------------------


def test_flat_cost_entry_position():
    """Opening a +1 position from flat: cost = base_bps / 1e4."""
    m = CostModel("SPY")
    pos = np.array([0.0, 1.0, 1.0, 1.0])
    tc = m.compute_tc(pos)
    # bar 0: Δpos = |1 - 0| = 1 → cost = 0.5 bps / 1e4 = 5e-6
    # bar 1: Δpos = 0 → 0
    np.testing.assert_allclose(tc[0], 0.0)   # prepend=0, so bar 0 is 0→0
    np.testing.assert_allclose(tc[1], m.base_bps() / 1e4)  # 0→1 flip
    np.testing.assert_allclose(tc[2], 0.0)   # no change
    np.testing.assert_allclose(tc[3], 0.0)


def test_flat_cost_full_flip():
    """A +1 → -1 flip costs 2 × base_bps."""
    m = CostModel("SPY")
    pos = np.array([1.0, -1.0])
    tc = m.compute_tc(pos)
    # bar 0: |1 - 0| = 1 → base_bps / 1e4
    # bar 1: |-1 - 1| = 2 → 2 × base_bps / 1e4
    np.testing.assert_allclose(tc[0], m.base_bps() / 1e4)
    np.testing.assert_allclose(tc[1], 2 * m.base_bps() / 1e4)


def test_flat_cost_matches_old_formula():
    """With no volume, compute_tc must equal (bps/1e4) × |Δpos|."""
    bps = ASSET_COST_BPS["QQQ"]
    m = CostModel("QQQ")
    rng = np.random.default_rng(0)
    pos = rng.uniform(-1, 1, size=50)
    tc = m.compute_tc(pos)
    expected = (bps / 1e4) * np.abs(np.diff(pos, prepend=0.0))
    np.testing.assert_allclose(tc, expected, rtol=1e-9)


# ---------------------------------------------------------------------------
# CostModel.compute_tc — with volume (Amihud)
# ---------------------------------------------------------------------------


def test_amihud_low_volume_inflates_cost():
    """A day with 10% of average volume → cost ≈ 3× base (clipped)."""
    m = CostModel("SPY")
    pos = np.array([0.0, 1.0])
    avg_vol = 1_000_000
    # volume[1] = 10% of avg → ratio = 10 → clipped to 3
    volume = np.array([avg_vol, avg_vol // 10])
    tc = m.compute_tc(pos, volume=volume)
    expected_inflated = 3.0 * m.base_bps() / 1e4
    np.testing.assert_allclose(tc[1], expected_inflated, rtol=1e-6)


def test_amihud_high_volume_compresses_cost():
    """A day with 10× average volume → effective cost ≈ 0.5× base (clipped).

    Build a series where many bars are at avg_vol (so median ≈ avg_vol)
    and the LAST bar is at 10× avg_vol WITH a position change, so the
    Amihud ratio = 0.1 → clips to 0.5.
    """
    m = CostModel("SPY")
    avg_vol = 1_000_000
    n = 51  # odd length so median is exactly avg_vol when 50 bars are at avg_vol
    # Position: flat until last bar, then open a +1 position
    pos = np.zeros(n)
    pos[-1] = 1.0
    # Volume: all bars at avg_vol except the last which is 10×
    volume = np.full(n, avg_vol, dtype=float)
    volume[-1] = avg_vol * 10
    tc = m.compute_tc(pos, volume=volume)
    # median(volume) = avg_vol (50 out of 51 bars)
    # ratio for last bar = avg_vol / (10 * avg_vol) = 0.1 → clipped to 0.5
    expected_compressed = 0.5 * m.base_bps() / 1e4   # Δpos=1 at last bar
    np.testing.assert_allclose(tc[-1], expected_compressed, rtol=1e-6)


def test_amihud_normal_volume_close_to_base():
    """All bars at average volume → effective_bps ≈ base_bps."""
    m = CostModel("EEM")
    n = 30
    pos = np.ones(n)
    pos[0] = 0.0  # first bar opens from flat
    avg_vol = 5_000_000
    volume = np.full(n, avg_vol, dtype=float)
    tc = m.compute_tc(pos, volume=volume)
    # bar 1: Δpos=1, ratio=1, effective=base
    np.testing.assert_allclose(tc[1], m.base_bps() / 1e4, rtol=1e-6)


def test_all_zero_volume_falls_back_to_flat():
    m = CostModel("SPY")
    pos = np.array([0.0, 1.0, 0.0])
    tc_flat = m.compute_tc(pos)
    tc_zero_vol = m.compute_tc(pos, volume=np.zeros(3))
    np.testing.assert_allclose(tc_zero_vol, tc_flat)


def test_none_volume_falls_back_to_flat():
    m = CostModel("TLT")
    pos = np.array([0.0, 0.5, -0.5])
    tc_flat = m.compute_tc(pos, volume=None)
    tc_no_arg = m.compute_tc(pos)
    np.testing.assert_allclose(tc_flat, tc_no_arg)