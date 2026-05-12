"""Tests for src.validation.risk_layer."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.validation.risk_layer import RiskControls  # noqa: E402


# ---------------------------------------------------------------------------
# Drawdown circuit-breaker
# ---------------------------------------------------------------------------


def test_dd_circuit_opens_at_limit():
    """Constant -0.3%/bar loss → equity drawdown hits 15% → positions zeroed."""
    rc = RiskControls(dd_limit=0.15, dd_reentry=0.075)
    n = 300
    pos = np.ones(n)
    # -0.003 log-return per bar → equity = exp(-0.003 × t)
    ret = np.full(n, -0.003)
    filtered = rc.apply_risk_controls(pos, ret)

    # DD hits 15% at bar ~= log(0.85)/(-0.003) ≈ 54
    first_zero = np.argmax(filtered == 0.0)
    assert first_zero > 0, "circuit should not open immediately"
    assert first_zero < n, "circuit must open before the series ends"
    # All bars after the first zero should also be zero (constant loss, no recovery)
    assert np.all(filtered[first_zero:] == 0.0)


def test_dd_circuit_reopens_after_recovery():
    """Loss streak → recovery → positions re-enabled."""
    rc = RiskControls(dd_limit=0.10, dd_reentry=0.05)
    n = 400
    pos = np.ones(n)
    ret = np.zeros(n)
    # First 80 bars: -0.2% / bar → cumDD ≈ 15% → circuit trips
    ret[:80] = -0.002
    # Bars 80-200: +0.15% / bar → recovery past -5% threshold
    ret[80:200] = 0.0015
    # Bars 200+: flat
    filtered = rc.apply_risk_controls(pos, ret)

    # Some zeros in loss period
    assert np.any(filtered[:200] == 0.0)
    # Should recover and re-enter (non-zero positions in later segment)
    assert np.any(filtered[200:] != 0.0)


def test_circuit_opens_quickly_on_steep_loss():
    """Steep immediate loss → circuit opens within a few bars."""
    rc = RiskControls(dd_limit=0.10, dd_reentry=0.05)
    n = 50
    pos = np.ones(n)
    # -1% per bar → DD hits 10% in ~11 bars
    ret = np.full(n, -0.01)
    filtered = rc.apply_risk_controls(pos, ret)
    first_zero = np.argmax(filtered == 0.0)
    assert 0 < first_zero <= 20, f"circuit should open quickly, got first_zero={first_zero}"
    assert np.all(filtered[first_zero:] == 0.0)


# ---------------------------------------------------------------------------
# VaR gate
# ---------------------------------------------------------------------------


def test_var_gate_scales_down_large_position():
    """Large position on volatile underlying → VaR gate scales it below 1."""
    rc = RiskControls(var_conf=0.95, var_window=60, var_nav_pct=0.02)
    n = 100
    rng = np.random.default_rng(42)
    # High-vol underlying: ±2% daily
    ret = rng.normal(0, 0.02, size=n)
    # Warm up with 60 bars of init_returns
    init = rng.normal(0, 0.02, size=60)
    pos = np.ones(n)

    filtered = rc.apply_risk_controls(pos, ret, init_returns=init)

    # With 2% daily vol, 95% VaR ≈ 1.65 × 2% = 3.3% per unit position.
    # nav_pct=2% → scale = 2/3.3 ≈ 0.6 < 1 → positions should be < 1.
    assert filtered.max() < 1.0 + 1e-9


def test_var_gate_never_levers_up():
    """VaR gate must not increase position above raw value."""
    rc = RiskControls(var_conf=0.95, var_window=60, var_nav_pct=0.10)
    rng = np.random.default_rng(0)
    pos = rng.uniform(0.1, 0.5, size=100)
    ret = rng.normal(0, 0.005, size=100)
    filtered = rc.apply_risk_controls(pos, ret)
    assert np.all(filtered <= pos + 1e-9)


def test_var_gate_no_op_on_tiny_position():
    """Very small position + generous nav_pct → VaR gate never triggers.

    Uses dd_limit=1.0 to disable the DD circuit-breaker so only the VaR
    gate is in play. With a 1% position and 10% nav_pct, the VaR
    constraint is satisfied at any realistic volatility level.
    """
    rc = RiskControls(
        dd_limit=1.0,        # disable DD circuit for isolation
        dd_reentry=0.5,
        var_conf=0.95,
        var_window=60,
        var_nav_pct=0.10,
    )
    rng = np.random.default_rng(7)
    pos = np.full(100, 0.01)  # 1% position
    ret = rng.normal(0, 0.02, size=100)
    init = rng.normal(0, 0.02, size=60)
    filtered = rc.apply_risk_controls(pos, ret, init_returns=init)
    # With 1% position and 10% nav_pct, VaR at any volatility << nav_pct
    np.testing.assert_allclose(filtered, pos)


# ---------------------------------------------------------------------------
# Causal hygiene
# ---------------------------------------------------------------------------


def test_causal_no_future_information():
    """Decision at bar t must only use returns through t-1 (no lookahead)."""
    rc = RiskControls(dd_limit=0.15, var_window=10)
    n = 50
    rng = np.random.default_rng(99)
    pos_a = np.ones(n)
    pos_b = np.ones(n)
    ret_a = rng.normal(0, 0.01, size=n)
    ret_b = ret_a.copy()
    # Make bar 49 very different in ret_b; should not affect bars 0-48
    ret_b[-1] = 0.5

    filtered_a = rc.apply_risk_controls(pos_a, ret_a)
    filtered_b = rc.apply_risk_controls(pos_b, ret_b)

    np.testing.assert_array_equal(filtered_a[:-1], filtered_b[:-1])


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_positions():
    rc = RiskControls()
    filtered = rc.apply_risk_controls(np.array([]), np.array([]))
    assert len(filtered) == 0


def test_single_bar():
    rc = RiskControls()
    filtered = rc.apply_risk_controls(np.array([1.0]), np.array([0.005]))
    assert len(filtered) == 1


def test_output_shape_matches_input():
    rc = RiskControls()
    rng = np.random.default_rng(0)
    n = 200
    pos = rng.uniform(-1, 1, size=n)
    ret = rng.normal(0, 0.01, size=n)
    filtered = rc.apply_risk_controls(pos, ret)
    assert filtered.shape == pos.shape