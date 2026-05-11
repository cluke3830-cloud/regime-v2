"""Tests for src.hmm.forward_filter_optimised (Brief 5.4).

Verifies the Numba-JIT'd kernel matches the numpy reference to
machine precision, and benchmarks the speedup.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hmm.forward_filter_optimised import (  # noqa: E402
    forward_filter_log_space,
    forward_filter_naive,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _hmm_log_inputs(T: int = 500, K: int = 5, seed: int = 0):
    """Synthesize log_emis, log_trans, log_start for an HMM."""
    rng = np.random.default_rng(seed)
    # Random emissions ∼ standard normal in log-space (i.e. arbitrary log-likelihoods)
    log_emis = rng.standard_normal((T, K))
    # Random row-stochastic transition matrix
    raw = rng.uniform(0.1, 1.0, size=(K, K))
    trans = raw / raw.sum(axis=1, keepdims=True)
    log_trans = np.log(trans)
    # Uniform start distribution
    log_start = np.log(np.full(K, 1.0 / K))
    return log_emis, log_trans, log_start


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------


def test_naive_vs_jit_match_to_machine_precision():
    le, lt, ls = _hmm_log_inputs(T=200, K=4, seed=1)
    _, probs_naive = forward_filter_naive(le, lt, ls)
    _, probs_jit = forward_filter_log_space(le, lt, ls)
    np.testing.assert_allclose(probs_naive, probs_jit, atol=1e-10)


def test_probabilities_sum_to_one_per_bar():
    le, lt, ls = _hmm_log_inputs(T=300, K=3)
    _, probs = forward_filter_log_space(le, lt, ls)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-9)


def test_probabilities_in_unit_interval():
    le, lt, ls = _hmm_log_inputs(T=300, K=6)
    _, probs = forward_filter_log_space(le, lt, ls)
    assert (probs >= 0).all() and (probs <= 1.0 + 1e-9).all()


def test_two_state_single_bar():
    """Hand-checkable: 1 bar, 2 states, uniform start, known log-emis."""
    log_emis = np.array([[np.log(0.6), np.log(0.4)]])
    log_trans = np.log(np.array([[0.5, 0.5], [0.5, 0.5]]))
    log_start = np.log(np.array([0.5, 0.5]))
    _, probs = forward_filter_log_space(log_emis, log_trans, log_start)
    # alpha[0] = start * emis = [0.5*0.6, 0.5*0.4] = [0.3, 0.2]
    # normalised: [0.6, 0.4]
    np.testing.assert_allclose(probs[0], [0.6, 0.4], atol=1e-9)


def test_handles_neg_inf_emissions():
    """When an emission is -inf, that state should get probability 0."""
    T, K = 50, 3
    le, lt, ls = _hmm_log_inputs(T=T, K=K, seed=5)
    # Zero-out emission for state 0 at every bar
    le[:, 0] = -np.inf
    _, probs = forward_filter_log_space(le, lt, ls)
    # State 0 must have ~zero probability everywhere
    assert (probs[:, 0] < 1e-9).all()


# ---------------------------------------------------------------------------
# Performance (smoke benchmark — not a hard gate)
# ---------------------------------------------------------------------------


def test_jit_faster_than_naive_on_long_series():
    """The JIT kernel should be measurably faster than the naive numpy
    reference on a realistic-sized input. Allow generous tolerance
    (≥ 3× speedup) — on cold JIT cache the JIT can be slower on the
    first call due to compilation overhead.
    """
    le, lt, ls = _hmm_log_inputs(T=3000, K=5, seed=11)

    # Warm up the JIT cache (first call compiles)
    forward_filter_log_space(le, lt, ls)

    # Benchmark naive
    n_iters = 3
    t0 = time.perf_counter()
    for _ in range(n_iters):
        forward_filter_naive(le, lt, ls)
    naive_time = (time.perf_counter() - t0) / n_iters

    # Benchmark JIT (already compiled)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        forward_filter_log_space(le, lt, ls)
    jit_time = (time.perf_counter() - t0) / n_iters

    speedup = naive_time / max(jit_time, 1e-9)
    # We aim for ≥ 50× per audit; allow as low as 3× to avoid flakiness
    # on CI machines.
    assert speedup >= 3.0, (
        f"expected ≥ 3× speedup, got {speedup:.1f}× "
        f"(naive {naive_time:.3f}s, jit {jit_time:.3f}s)"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
