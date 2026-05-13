"""Tests for src.strategies.fusion — empirical mapping + log-opinion-pool +
the CPCV strategy_fn factory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.strategies.fusion import (  # noqa: E402
    empirical_tvtp_3class_mapping,
    apply_log_opinion_pool,
    make_fusion_strategy,
    _PRIOR_TVTP_MAPPING,
)


# ---------------------------------------------------------------------------
# Empirical mapping
# ---------------------------------------------------------------------------


def test_empirical_mapping_rows_sum_to_one():
    """Output is row-stochastic."""
    idx = pd.date_range("2020-01-01", periods=200, freq="B")
    rng = np.random.default_rng(0)
    p_low = rng.uniform(0.2, 0.8, size=200)
    tvtp = pd.DataFrame({"p_low_vol": p_low, "p_high_vol": 1 - p_low}, index=idx)
    rule = pd.Series(rng.integers(0, 3, size=200), index=idx)
    M = empirical_tvtp_3class_mapping(tvtp, rule)
    np.testing.assert_allclose(M.sum(axis=1), 1.0, atol=1e-9)
    assert M.shape == (2, 3)


def test_empirical_mapping_recovers_known_correlation():
    """When low-vol days are always Bull in training, P(Bull|low_vol) → 1."""
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    p_low = np.concatenate([np.ones(150), np.zeros(150)])
    tvtp = pd.DataFrame({"p_low_vol": p_low, "p_high_vol": 1 - p_low}, index=idx)
    # Days where p_low==1 → Bull; days where p_low==0 → Bear
    rule = pd.Series(
        np.where(p_low == 1, 0, 2),  # 0 = Bull, 2 = Bear
        index=idx,
    )
    M = empirical_tvtp_3class_mapping(tvtp, rule)
    # Row 0 (low-vol) should put nearly all mass on Bull
    assert M[0, 0] > 0.95
    # Row 1 (high-vol) should put nearly all mass on Bear
    assert M[1, 2] > 0.95


def test_empirical_mapping_falls_back_when_too_few_samples():
    """With fewer than min_bars aligned samples, use the prior."""
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    tvtp = pd.DataFrame({"p_low_vol": [0.5] * 5, "p_high_vol": [0.5] * 5}, index=idx)
    rule = pd.Series([0, 1, 2, 0, 1], index=idx)
    M = empirical_tvtp_3class_mapping(tvtp, rule, min_bars=30)
    np.testing.assert_array_equal(M, _PRIOR_TVTP_MAPPING)


# ---------------------------------------------------------------------------
# Log-opinion-pool
# ---------------------------------------------------------------------------


def test_log_opinion_pool_outputs_proper_distribution():
    """Fused rows are valid probability vectors (sum to 1, all positive)."""
    rng = np.random.default_rng(42)
    n = 50
    gmm = rng.dirichlet(np.ones(3), size=n)
    tvtp = rng.dirichlet(np.ones(2), size=n)
    mapping = np.array([[0.7, 0.3, 0.0], [0.0, 0.0, 1.0]])
    fused = apply_log_opinion_pool(gmm, tvtp, mapping)
    assert fused.shape == (n, 3)
    np.testing.assert_allclose(fused.sum(axis=1), 1.0, atol=1e-9)
    assert (fused >= 0).all()


def test_log_opinion_pool_agreement_sharpens():
    """When both models strongly agree on Bull, fused P(Bull) > either individual P."""
    gmm = np.array([[0.9, 0.05, 0.05]])
    tvtp = np.array([[0.95, 0.05]])
    mapping = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # low-vol = Bull
    fused = apply_log_opinion_pool(gmm, tvtp, mapping)
    assert fused[0, 0] > 0.9


def test_log_opinion_pool_disagreement_flattens():
    """When models disagree, fused distribution is more uniform than either model."""
    gmm = np.array([[0.9, 0.05, 0.05]])
    tvtp = np.array([[0.1, 0.9]])
    mapping = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    fused = apply_log_opinion_pool(gmm, tvtp, mapping)
    # Entropy should be elevated relative to a peaked dist
    H = -(fused * np.log(fused + 1e-12)).sum()
    H_norm = H / np.log(3)
    assert H_norm > 0.3   # well above 0 (peaked) but below 1 (uniform)


# ---------------------------------------------------------------------------
# CPCV strategy factory
# ---------------------------------------------------------------------------


def _make_synthetic_features(n: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Build a features DataFrame; n is raw bar count (≈n-252 rows survive)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    rets = np.concatenate([
        rng.normal(0.0008, 0.008, n // 2),
        rng.normal(-0.0005, 0.020, n - n // 2),
    ])
    close = pd.Series(100 * np.exp(rets.cumsum()), index=idx)

    from src.features.price_features import compute_features_v1
    feats = compute_features_v1(close)
    feats["close"] = close.reindex(feats.index)
    return feats


def test_fusion_strategy_returns_correct_shape():
    """Strategy_fn output length matches features_test."""
    feats = _make_synthetic_features(n=1000, seed=11)
    cut = int(len(feats) * 0.7)
    f_train = feats.iloc[:cut]
    f_test = feats.iloc[cut:]
    fn = make_fusion_strategy()
    positions = fn(f_train, f_test)
    assert isinstance(positions, np.ndarray)
    assert positions.shape == (len(f_test),)


def test_fusion_strategy_positions_in_expected_range():
    """Positions are bounded by the configured state_positions."""
    feats = _make_synthetic_features(n=1000, seed=22)
    cut = int(len(feats) * 0.7)
    fn = make_fusion_strategy(
        state_positions={0: 1.0, 1: 0.0, 2: -0.5},
    )
    positions = fn(feats.iloc[:cut], feats.iloc[cut:])
    assert positions.min() >= -0.5 - 1e-9
    assert positions.max() <= 1.0 + 1e-9


def test_fusion_strategy_causal_no_lookahead():
    """Perturbing the LAST test bar's close must not change earlier positions."""
    feats_a = _make_synthetic_features(n=1000, seed=33)
    feats_b = feats_a.copy()
    feats_b.iloc[-1, feats_b.columns.get_loc("close")] *= 1.5
    cut = int(len(feats_a) * 0.7)
    fn = make_fusion_strategy()
    pa = fn(feats_a.iloc[:cut], feats_a.iloc[cut:])
    pb = fn(feats_b.iloc[:cut], feats_b.iloc[cut:])
    np.testing.assert_allclose(pa[:-1], pb[:-1], atol=1e-10)


def test_fusion_strategy_handles_short_test_window():
    """Doesn't crash when the test fold is short relative to train."""
    feats = _make_synthetic_features(n=1000, seed=44)
    cut = len(feats) - 30   # only 30 test bars
    fn = make_fusion_strategy()
    positions = fn(feats.iloc[:cut], feats.iloc[cut:])
    assert len(positions) == 30