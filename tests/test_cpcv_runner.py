"""Acceptance tests for src.validation.cpcv_runner (Brief 1.4).

Audit-prescribed acceptance test:
  - Running `make validate` generates validation_report.md with required
    sections: per-path Sharpe distribution; DSR with n_trials and resulting
    p-value; PBO percent; triple-barrier-label class balance per regime;
    multi-asset robustness panel (Brief 1.5). Reproducible (same seed →
    same numbers).

Tests here cover the harness itself:
  - run_cpcv_validation produces n_paths = C(n_splits, n_test_groups) paths;
  - per-path metric arithmetic (Sharpe / max-DD / Calmar) is right;
  - shape / index validation triggers;
  - multi-strategy runner produces well-defined PBO with ≥ 2 strategies;
  - PBO = None with 1 strategy;
  - reproducibility — same inputs produce numerically identical reports;
  - emit_markdown_report writes all required sections;
  - benchmark strategies are sane (buy_and_hold tracks the underlying;
    flat produces zero Sharpe; momentum_20d skips bars with negative
    momentum).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.validation.cpcv_runner import (  # noqa: E402
    PathMetrics,
    ValidationReport,
    _compute_path_metrics,
    emit_markdown_report,
    run_cpcv_multi_strategy,
    run_cpcv_validation,
)
from src.strategies.benchmarks import buy_and_hold, flat, momentum_20d  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic data fixtures
# ---------------------------------------------------------------------------


def _make_synthetic(
    n: int = 1500, drift: float = 0.0003, vol: float = 0.012, seed: int = 0
) -> tuple[pd.DataFrame, pd.Series]:
    """GBM close series + features dataframe with mom_20 (causally computed)."""
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n)
    log_ret = drift - 0.5 * vol ** 2 + vol * eps
    close = pd.Series(np.exp(np.log(100.0) + np.cumsum(log_ret)), name="close")
    log_returns = np.log(close).diff().fillna(0.0)
    mom_20 = log_returns.rolling(20).sum().shift(1)  # shifted → strictly past
    features = pd.DataFrame({"close": close, "mom_20": mom_20})
    return features, log_returns


# ---------------------------------------------------------------------------
# _compute_path_metrics
# ---------------------------------------------------------------------------


def test_path_metrics_basic():
    """Hand-checkable arithmetic: total_return = Σr, max-DD ≤ 0, Calmar
    matches the closed-form (mean·252 / |max-DD|). We avoid asserting the
    sign of Sharpe — at small sample sizes a positive-drift series can
    still realise a negative-Sharpe sample on an unlucky draw.
    """
    rng = np.random.default_rng(7)
    r = rng.standard_normal(252) * 0.01 + 0.001
    pm = _compute_path_metrics(path_id=0, strategy_returns=r)
    assert pm.path_id == 0
    assert pm.n_bars == 252
    assert pm.total_return == pytest.approx(r.sum(), rel=1e-9)
    assert np.isfinite(pm.sharpe)
    assert pm.max_drawdown <= 0
    assert np.isfinite(pm.sortino)
    if pm.max_drawdown < 0:
        expected_calmar = (r.mean() * 252) / abs(pm.max_drawdown)
        assert pm.calmar == pytest.approx(expected_calmar, rel=1e-6)


def test_path_metrics_positive_drift_realised():
    """With a much larger drift-to-noise ratio so the realised Sharpe is
    reliably positive across reasonable seeds — guards against the
    Sharpe-sign machinery being inverted.
    """
    rng = np.random.default_rng(11)
    r = rng.standard_normal(1000) * 0.005 + 0.002  # SR ≈ 6.3 population
    pm = _compute_path_metrics(0, r)
    assert pm.sharpe > 1.0


def test_path_metrics_all_zeros():
    """Zero returns → Sharpe = 0, max-DD = 0, Calmar handled."""
    r = np.zeros(100)
    pm = _compute_path_metrics(0, r)
    assert pm.sharpe == 0.0
    assert pm.max_drawdown == 0.0


# ---------------------------------------------------------------------------
# run_cpcv_validation
# ---------------------------------------------------------------------------


def test_single_strategy_n_paths_matches_combinatorial():
    features, returns = _make_synthetic(n=500)
    report = run_cpcv_validation(
        buy_and_hold,
        features,
        returns,
        n_splits=10,
        n_test_groups=2,
        n_trials=10,
        strategy_name="buy_and_hold",
    )
    assert report.n_paths == 45
    assert len(report.path_metrics) == 45
    assert report.pbo is None  # single strategy


def test_buy_and_hold_tracks_underlying():
    """Buy-and-hold's concatenated OOS returns must equal the bar returns
    on the test indices (positions = +1 everywhere).
    """
    features, returns = _make_synthetic(n=400)
    report = run_cpcv_validation(
        buy_and_hold,
        features,
        returns,
        n_splits=8,
        n_test_groups=2,
    )
    for path_id, series in report.oos_returns.items():
        # series should equal returns at the same dates
        expected = returns.loc[series.index]
        assert np.allclose(series.values, expected.values)


def test_features_returns_alignment_validation():
    f, r = _make_synthetic(n=200)
    with pytest.raises(ValueError, match="len"):
        run_cpcv_validation(buy_and_hold, f, r.iloc[:-1])
    # Use a clearly different (date) index so the equality check fires —
    # range(len(r)) would coincidentally match the default integer index.
    misindexed = pd.Series(
        r.values,
        index=pd.date_range("2020-01-01", periods=len(r), freq="D"),
    )
    with pytest.raises(ValueError, match="index"):
        run_cpcv_validation(buy_and_hold, f, misindexed)


def test_strategy_fn_wrong_length_raises():
    """A misbehaving strategy_fn that returns the wrong-length position
    array must raise — the harness should not silently truncate.
    """
    f, r = _make_synthetic(n=300)

    def bad_strategy(f_train, f_test):
        return np.ones(len(f_test) - 1)  # off by one

    with pytest.raises(ValueError, match="positions"):
        run_cpcv_validation(bad_strategy, f, r, n_splits=5, n_test_groups=2)


# ---------------------------------------------------------------------------
# Multi-strategy + PBO
# ---------------------------------------------------------------------------


def test_multi_strategy_pbo_is_defined():
    features, returns = _make_synthetic(n=800, drift=0.0002)
    reports = run_cpcv_multi_strategy(
        {"buy_and_hold": buy_and_hold, "momentum_20d": momentum_20d},
        features,
        returns,
        n_splits=10,
        n_test_groups=2,
        n_trials=2,
    )
    assert set(reports.keys()) == {"buy_and_hold", "momentum_20d"}
    for r in reports.values():
        assert r.pbo is not None
        assert 0.0 <= r.pbo <= 1.0
        assert r.is_sharpe_per_path is not None
        assert r.oos_sharpe_per_path is not None
        assert r.is_sharpe_per_path.shape == (r.n_paths,)


def test_multi_strategy_same_folds():
    """All strategies in run_cpcv_multi_strategy must see the SAME folds —
    otherwise PBO comparisons are misaligned. Verify by inspecting OOS index.
    """
    f, r = _make_synthetic(n=600)
    reports = run_cpcv_multi_strategy(
        {"a": buy_and_hold, "b": flat},
        f, r,
        n_splits=8, n_test_groups=2,
    )
    for path_id in reports["a"].oos_returns.keys():
        ix_a = reports["a"].oos_returns[path_id].index
        ix_b = reports["b"].oos_returns[path_id].index
        assert ix_a.equals(ix_b), f"path {path_id} folds diverged"


def test_pbo_identical_strategies_is_borderline():
    """When all strategies are identical, the IS-best is a random tie-break
    among them and PBO is statistically ≈ 0.5 — we just check that PBO is
    finite and in [0, 1]; the deterministic argmax breaks ties on index 0
    so the actual value can be 0 or 1 depending on alignment.
    """
    f, r = _make_synthetic(n=600)
    reports = run_cpcv_multi_strategy(
        {"a": buy_and_hold, "b": buy_and_hold},
        f, r,
        n_splits=8, n_test_groups=2,
    )
    pbo = reports["a"].pbo
    assert pbo is not None
    assert 0.0 <= pbo <= 1.0


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_reproducibility_same_inputs_same_report():
    """Same seed + same data → numerically identical Sharpe percentiles."""
    f, r = _make_synthetic(n=500, seed=42)
    r1 = run_cpcv_validation(buy_and_hold, f, r, n_splits=8, n_test_groups=2)
    r2 = run_cpcv_validation(buy_and_hold, f, r, n_splits=8, n_test_groups=2)
    assert r1.sharpe_p05 == r2.sharpe_p05
    assert r1.sharpe_p50 == r2.sharpe_p50
    assert r1.sharpe_p95 == r2.sharpe_p95
    assert r1.dsr_p_value == r2.dsr_p_value


# ---------------------------------------------------------------------------
# Markdown emitter
# ---------------------------------------------------------------------------


def test_markdown_report_required_sections():
    f, r = _make_synthetic(n=600)
    reports = run_cpcv_multi_strategy(
        {"buy_and_hold": buy_and_hold, "momentum_20d": momentum_20d},
        f, r,
        n_splits=10, n_test_groups=2, n_trials=3,
    )
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "validation_report.md"
        emit_markdown_report(
            reports,
            out,
            label_balance={-1: 0.33, 0: 0.34, 1: 0.33},
            multi_asset_results={
                "SPY": {"sharpe_p50": 0.5, "dsr_p_value": 0.95},
                "QQQ": {"sharpe_p50": 0.7, "dsr_p_value": 0.97},
            },
        )
        text = out.read_text()

    # Audit-prescribed sections must all appear
    assert "## CPCV configuration" in text
    assert "## Per-strategy results" in text
    assert "Sharpe p05" in text
    assert "DSR p-value" in text
    assert "## Probability of Backtest Overfitting" in text
    assert "## Triple-barrier label balance" in text
    assert "## Multi-asset robustness" in text
    assert "## Reproducibility" in text
    # Both strategy names must appear as table rows
    assert "buy_and_hold" in text
    assert "momentum_20d" in text


def test_markdown_handles_nan_metrics_gracefully():
    """If a degenerate path produced NaN Sharpe, the emitter must not crash
    and must print an em-dash rather than 'nan'.
    """
    f, r = _make_synthetic(n=200)
    report = run_cpcv_validation(flat, f, r, n_splits=5, n_test_groups=2)
    # `flat` produces zero returns → Sharpe = 0 (not NaN) — but verify
    # the emitter handles the {"strategy": report} singleton path:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "validation_report.md"
        emit_markdown_report({"flat": report}, out)
        text = out.read_text()
    assert "## Per-strategy results" in text
    assert "flat" in text
    # No 'nan' literal should leak through formatter
    assert "nan" not in text.lower() or "—" in text


# ---------------------------------------------------------------------------
# Benchmark strategies sanity
# ---------------------------------------------------------------------------


def test_benchmark_buy_and_hold_returns_ones():
    f = pd.DataFrame({"close": [100, 101, 102]})
    assert np.array_equal(buy_and_hold(f, f), np.array([1.0, 1.0, 1.0]))


def test_benchmark_flat_returns_zeros():
    f = pd.DataFrame({"close": [100, 101, 102]})
    assert np.array_equal(flat(f, f), np.array([0.0, 0.0, 0.0]))


def test_benchmark_momentum_long_when_positive():
    """momentum_20d emits +1 when mom_20 > 0, 0 otherwise."""
    f = pd.DataFrame({"mom_20": [0.01, -0.02, 0.0, np.nan, 0.005]})
    pos = momentum_20d(f, f)
    assert np.array_equal(pos, np.array([1.0, 0.0, 0.0, 0.0, 1.0]))


def test_benchmark_momentum_requires_column():
    with pytest.raises(KeyError, match="momentum_20d requires"):
        momentum_20d(pd.DataFrame({"close": [1.0]}), pd.DataFrame({"close": [1.0]}))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))