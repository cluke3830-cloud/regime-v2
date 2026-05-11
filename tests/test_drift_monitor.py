"""Tests for src.monitoring.drift_monitor (Brief 5.1)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.monitoring.drift_monitor import (  # noqa: E402
    DriftMonitor,
    calibration_drift,
    population_stability_index,
    rolling_mmd,
)


# ---------------------------------------------------------------------------
# PSI
# ---------------------------------------------------------------------------


def test_psi_zero_on_identical_distributions():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1000)
    psi = population_stability_index(x, x.copy())
    assert psi >= 0
    assert psi < 0.01


def test_psi_positive_on_shifted_distribution():
    rng = np.random.default_rng(0)
    ref = rng.standard_normal(1000)
    shifted = rng.standard_normal(1000) + 2.0  # mean shift
    psi = population_stability_index(ref, shifted)
    assert psi > 0.25, f"shifted distribution should trigger PSI > 0.25, got {psi}"


def test_psi_nan_on_degenerate_input():
    psi = population_stability_index(np.zeros(5), np.zeros(5))
    assert np.isnan(psi) or psi == 0.0


# ---------------------------------------------------------------------------
# MMD
# ---------------------------------------------------------------------------


def test_mmd_near_zero_on_identical_distributions():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((300, 5))
    mmd = rolling_mmd(x, x.copy())
    assert mmd >= 0
    assert mmd < 0.05


def test_mmd_positive_on_distribution_shift():
    rng = np.random.default_rng(0)
    ref = rng.standard_normal((300, 5))
    shifted = rng.standard_normal((300, 5)) + 1.5
    mmd = rolling_mmd(ref, shifted)
    assert mmd > 0.05, f"shifted distribution should trigger MMD > 0.05, got {mmd}"


def test_mmd_nan_on_tiny_input():
    mmd = rolling_mmd(np.array([[0.0]]), np.array([[1.0]]))
    assert np.isnan(mmd)


# ---------------------------------------------------------------------------
# Calibration drift
# ---------------------------------------------------------------------------


def test_calibration_drift_zero_when_unchanged():
    d = calibration_drift(0.90, 0.90, target_coverage=0.90)
    assert d == 0.0


def test_calibration_drift_positive_when_worsened():
    # Reference was at target; current drifted away
    d = calibration_drift(0.90, 0.75, target_coverage=0.90)
    assert d > 0


def test_calibration_drift_negative_when_improved():
    # Reference was off-target; current is at target
    d = calibration_drift(0.75, 0.90, target_coverage=0.90)
    assert d < 0


# ---------------------------------------------------------------------------
# DriftMonitor end-to-end
# ---------------------------------------------------------------------------


def test_drift_monitor_no_trigger_on_iid_data():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame(rng.standard_normal((500, 4)), columns=list("abcd"))
    cur = pd.DataFrame(rng.standard_normal((500, 4)), columns=list("abcd"))
    monitor = DriftMonitor(ref)
    out = monitor.check_drift(cur)
    assert out["trigger"] is False
    assert out["psi_max"] < 0.25
    assert "triggers" in out


def test_drift_monitor_triggers_on_shifted_data():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame(rng.standard_normal((500, 4)), columns=list("abcd"))
    cur = pd.DataFrame(rng.standard_normal((500, 4)) + 2.5, columns=list("abcd"))
    monitor = DriftMonitor(ref)
    out = monitor.check_drift(cur)
    assert out["trigger"] is True
    assert len(out["triggers"]) >= 1


def test_drift_monitor_calibration_signal():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame(rng.standard_normal((400, 3)), columns=list("xyz"))
    monitor = DriftMonitor(
        ref, reference_coverage=0.90, target_coverage=0.90,
        cal_drift_threshold=0.05,
    )
    # Feed same-distribution features but degraded coverage
    out = monitor.check_drift(ref.copy(), current_coverage=0.75)
    assert "cal_drift" in out and out["cal_drift"] is not None
    assert out["cal_drift"] > 0
    assert out["trigger"] is True
    assert any("cal_drift" in t for t in out["triggers"])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
