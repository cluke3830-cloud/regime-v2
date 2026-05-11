"""Tests for src.regime.patchtst (Brief 4.1)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.regime.patchtst import (  # noqa: E402
    DeepEnsembleTransformer,
    TransformerRegimeClassifier,
    build_sequences,
    make_patchtst_strategy,
)
from src.validation.cpcv_runner import run_cpcv_validation  # noqa: E402


# ---------------------------------------------------------------------------
# build_sequences
# ---------------------------------------------------------------------------


def test_build_sequences_shape_and_indices():
    X = np.arange(100 * 5, dtype=np.float32).reshape(100, 5)
    X_seq, idx = build_sequences(X, seq_len=20)
    assert X_seq.shape == (81, 20, 5)
    assert idx[0] == 19 and idx[-1] == 99
    # First window should match X[0:20]
    np.testing.assert_array_equal(X_seq[0], X[0:20])
    # Last window should match X[80:100]
    np.testing.assert_array_equal(X_seq[-1], X[80:100])


def test_build_sequences_short_input_returns_empty():
    X = np.zeros((10, 5), dtype=np.float32)
    X_seq, idx = build_sequences(X, seq_len=20)
    assert X_seq.shape == (0, 20, 5)
    assert len(idx) == 0


# ---------------------------------------------------------------------------
# Model forward shape
# ---------------------------------------------------------------------------


def test_model_forward_returns_logits():
    import torch
    model = TransformerRegimeClassifier(n_features=21, seq_len=30)
    x = torch.randn(8, 30, 21)
    out = model(x)
    assert out.shape == (8, 3)


# ---------------------------------------------------------------------------
# Ensemble fit / predict
# ---------------------------------------------------------------------------


def _make_classification_data(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 21)).astype(np.float32)
    y = np.zeros(n, dtype=np.int64)
    # Signal: when feature 0 > 0.5, label=+1; < -0.5, label=-1
    y[X[:, 0] > 0.5] = 1
    y[X[:, 0] < -0.5] = -1
    return X, y


def test_ensemble_fit_predict_shapes():
    X, y = _make_classification_data(n=400)
    model = DeepEnsembleTransformer(
        n_seeds=2, seq_len=20, epochs=3, batch_size=32,
    )
    model.fit(X, y)
    assert len(model.models_) <= 2
    proba = model.predict_proba(X)
    assert proba.shape == (400, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_ensemble_too_few_rows_no_models():
    """< seq_len + 50 → no models, predict_proba returns uniform."""
    X = np.zeros((30, 21), dtype=np.float32)
    y = np.zeros(30, dtype=np.int64)
    model = DeepEnsembleTransformer(n_seeds=2, seq_len=20, epochs=2)
    model.fit(X, y)
    assert len(model.models_) == 0
    proba = model.predict_proba(X)
    np.testing.assert_allclose(proba, 1.0 / 3.0)


def test_ensemble_n_seeds_validation():
    with pytest.raises(ValueError, match="n_seeds"):
        DeepEnsembleTransformer(n_seeds=0)


# ---------------------------------------------------------------------------
# Strategy_fn — full CPCV
# ---------------------------------------------------------------------------


def _make_synthetic_close_features(n: int = 700, seed: int = 0):
    rng = np.random.default_rng(seed)
    log_ret = 0.0003 + 0.012 * rng.standard_normal(n)
    close = pd.Series(
        np.exp(np.log(100.0) + np.cumsum(log_ret)),
        index=pd.date_range("2018-01-01", periods=n, freq="D"),
        name="close",
    )
    log_returns = np.log(close).diff()
    mom_20 = log_returns.rolling(20).sum().shift(1)
    vol_ewma = log_returns.ewm(alpha=0.06, min_periods=20).std()
    features = pd.DataFrame({
        "close": close, "mom_20": mom_20, "vol_ewma": vol_ewma,
    }).dropna()
    return features, log_returns.loc[features.index]


def test_patchtst_strategy_runs_in_cpcv():
    """Smoke test — runs through CPCV without crashing.

    Uses small seq_len + few epochs + 1 seed to keep wall-clock under
    ~30 sec.
    """
    features, log_returns = _make_synthetic_close_features(n=400)
    strategy = make_patchtst_strategy(
        seq_len=15, n_seeds=1, epochs=3, batch_size=32,
    )
    report = run_cpcv_validation(
        strategy, features, log_returns,
        strategy_name="patchtst",
        n_splits=5, n_test_groups=2, n_trials=5,
    )
    assert report.n_paths == 10  # C(5, 2)
    assert np.isfinite(report.sharpe_p50)


def test_patchtst_positions_in_range():
    """Position = p(+1) - p(-1) ∈ [-1, +1] since probs sum to 1."""
    features, _ = _make_synthetic_close_features(n=300)
    strategy = make_patchtst_strategy(
        seq_len=15, n_seeds=1, epochs=2, batch_size=32,
    )
    f_train = features.iloc[:200]
    f_test = features.iloc[200:]
    out = strategy(f_train, f_test)
    assert out.shape == (len(f_test),)
    assert (out >= -1.0 - 1e-6).all()
    assert (out <= 1.0 + 1e-6).all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
