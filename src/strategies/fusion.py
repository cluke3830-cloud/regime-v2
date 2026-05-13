"""Multi-model fusion strategy — log-opinion-pool of GMM-HMM + TVTP-MSAR.

The dashboard already displays a fused regime posterior, but the CPCV harness
backtests rule_baseline labels. Without a CPCV trial against the fused signal,
fusion is a display artifact, not a tested strategy. This module fixes that.

Two design choices worth flagging:

1. **Causal fit, causal mapping.** Both base models are fit on the training
   slice only. The TVTP→3-class mapping is *learned* from the training rule
   labels (no fudge factors like the dashboard's 70/30 prior).

2. **Empirical TVTP→3-class.** Each TVTP bar contributes ``p_low_vol`` mass
   to the "low-vol" row and ``p_high_vol`` mass to the "high-vol" row,
   distributed across the rule baseline's {Bull, Neutral, Bear}. The
   resulting 2×3 row-stochastic matrix is what the dashboard's hardcoded
   priors should always have been — calibrated to actual market history.
"""

from __future__ import annotations

import warnings
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.baselines.tvtp_msar import MarkovSwitchingAR
from src.regime.gmm_hmm import (
    N_STATES, _build_features, _forward_filter,
)
from src.regime.rule_baseline import compute_rule_regime_sequence


_DEFAULT_FUSION_POSITIONS: Dict[int, float] = {0: 1.00, 1: 0.00, 2: -0.50}
# Fallback mapping for cold-start train slices where the empirical fit fails.
# Calibrated against the legacy dashboard prior (70% Bull, 30% Neutral for
# low-vol days; 100% Bear for high-vol). Survives missing classes gracefully.
_PRIOR_TVTP_MAPPING = np.array([
    [0.70, 0.30, 0.00],  # low-vol  → P(Bull) = 0.7, P(Neutral) = 0.3
    [0.00, 0.00, 1.00],  # high-vol → P(Bear) = 1.0
])


# ---------------------------------------------------------------------------
# Empirical TVTP → 3-class mapping (Improvement #2)
# ---------------------------------------------------------------------------


def empirical_tvtp_3class_mapping(
    tvtp_probs: pd.DataFrame,
    rule_labels: pd.Series,
    min_bars: int = 30,
) -> np.ndarray:
    """Learn P(rule_label=c | tvtp_state=s) from training data.

    Each bar contributes its TVTP probability mass to the appropriate row;
    each row is then row-normalised so it sums to 1. The output is a
    row-stochastic 2×3 matrix.

    Falls back to ``_PRIOR_TVTP_MAPPING`` if fewer than *min_bars* aligned
    samples are available (training fold too short for an empirical fit).

    Parameters
    ----------
    tvtp_probs : pd.DataFrame
        Columns ``p_low_vol`` and ``p_high_vol``.
    rule_labels : pd.Series
        Integer rule_baseline labels (0/1/2).
    min_bars : int
        Minimum aligned bars required to trust the empirical fit.

    Returns
    -------
    np.ndarray, shape (2, 3)
        Row-stochastic mapping. Row 0 = low-vol, row 1 = high-vol.
    """
    common = tvtp_probs.index.intersection(rule_labels.index)
    if len(common) < min_bars:
        return _PRIOR_TVTP_MAPPING.copy()

    tp = tvtp_probs.loc[common]
    rl = rule_labels.loc[common].astype(int).to_numpy()
    p_low = tp["p_low_vol"].to_numpy()
    p_high = tp["p_high_vol"].to_numpy()

    mapping = np.zeros((2, 3), dtype=float)
    counts = np.zeros(2, dtype=float)
    for c in range(3):
        mask = (rl == c)
        mapping[0, c] = p_low[mask].sum()
        mapping[1, c] = p_high[mask].sum()
    counts[0] = p_low.sum()
    counts[1] = p_high.sum()

    for s in range(2):
        if counts[s] > 1e-6:
            mapping[s] /= counts[s]
        else:
            mapping[s] = _PRIOR_TVTP_MAPPING[s]

    return mapping


# ---------------------------------------------------------------------------
# Log-opinion-pool with learned mapping (parametrised version of the
# dashboard helper; used in CPCV strategy and the dashboard alike)
# ---------------------------------------------------------------------------


def apply_log_opinion_pool(
    gmm_proba: np.ndarray,
    tvtp_proba: np.ndarray,
    mapping: np.ndarray,
    eps: float = 1e-9,
) -> np.ndarray:
    """Log-opinion-pool of GMM-HMM (3-class) and TVTP (2-class lifted to 3).

    Parameters
    ----------
    gmm_proba : np.ndarray, shape (n, 3)
        GMM-HMM posteriors.
    tvtp_proba : np.ndarray, shape (n, 2)
        TVTP-MSAR posteriors over (low_vol, high_vol).
    mapping : np.ndarray, shape (2, 3)
        Row-stochastic learned mapping from TVTP states to 3-class labels.
    eps : float
        Floor added before logarithm for numerical stability.

    Returns
    -------
    np.ndarray, shape (n, 3)
        Fused posteriors, row-normalised.
    """
    tvtp_3 = tvtp_proba @ mapping   # (n, 2) @ (2, 3) → (n, 3)
    log_pool = np.log(gmm_proba + eps) + np.log(tvtp_3 + eps)
    log_pool -= log_pool.max(axis=1, keepdims=True)
    pool = np.exp(log_pool)
    return pool / pool.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# CPCV strategy factory (Improvement #1)
# ---------------------------------------------------------------------------


def make_fusion_strategy(
    *,
    state_positions: Optional[Dict[int, float]] = None,
    close_col: str = "close",
    hmm_n_iter: int = 200,
    hmm_seed: int = 42,
) -> Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]:
    """Strategy factory for the fused multi-model regime classifier.

    Per outer CPCV fold:

      1. Compute combined-time-order log returns from
         ``features_train[close_col] ⊕ features_test[close_col]``.
      2. Fit MS-AR(1) (TVTP) on train returns; forward-filter on combined.
      3. Fit GaussianHMM (full cov, K=3) on the train rows of
         ``(return, vol)`` features; forward-filter on combined.
      4. Compute rule-baseline labels on the training features and learn
         the empirical TVTP→3-class mapping from them.
      5. Apply the log-opinion-pool with that mapping → fused posterior.
      6. Map fused posteriors to positions via *state_positions*
         (default: Bull +1.0, Neutral 0.0, Bear -0.5).
      7. Shift positions by 1 bar (causal) and reindex to
         ``features_test.index``.

    Returns
    -------
    Callable
        Standard CPCV strategy_fn taking ``(features_train, features_test)``
        and returning an np.ndarray of positions.
    """
    if state_positions is None:
        state_positions = dict(_DEFAULT_FUSION_POSITIONS)

    def strategy_fn(
        features_train: pd.DataFrame, features_test: pd.DataFrame
    ) -> np.ndarray:
        if close_col not in features_train.columns:
            raise KeyError(f"fusion strategy requires '{close_col}' in features_train")

        # ---- 1. Combined close + returns
        combined_close = pd.concat([
            features_train[close_col], features_test[close_col],
        ]).sort_index()
        combined_close = combined_close[~combined_close.index.duplicated(keep="first")]
        combined_returns = np.log(combined_close).diff().dropna()
        train_returns = combined_returns.loc[
            combined_returns.index.isin(features_train.index)
        ]

        # ---- 2. TVTP-MSAR: fit on train, filter on combined
        msar = MarkovSwitchingAR(k_regimes=2, order=1, switching_variance=True)
        try:
            msar.fit(train_returns)
            tvtp_probs = msar.predict_proba(combined_returns)
        except Exception:
            tvtp_probs = None

        # ---- 3. GMM-HMM: fit on train rows of (return, vol), filter on combined
        gmm_probs = _fit_filter_gmm(combined_close, features_train.index,
                                    n_iter=hmm_n_iter, seed=hmm_seed)

        if tvtp_probs is None or gmm_probs is None:
            # Both base models are required for fusion; if either fails,
            # fall back to a flat-zero position for this fold rather than
            # raise — the CPCV harness will record this fold's Sharpe as
            # 0 and the rest of the runs continue.
            return np.zeros(len(features_test), dtype=float)

        # ---- 4. Rule labels on train + empirical mapping
        try:
            rule_seq_train = compute_rule_regime_sequence(features_train)
            rule_labels_train = rule_seq_train["label"]
        except Exception:
            rule_labels_train = pd.Series(
                np.ones(len(features_train), dtype=int),
                index=features_train.index,
            )
        tvtp_train = tvtp_probs.loc[
            tvtp_probs.index.isin(features_train.index)
        ]
        mapping = empirical_tvtp_3class_mapping(tvtp_train, rule_labels_train)

        # ---- 5. Log-opinion-pool on aligned index
        common = tvtp_probs.index.intersection(gmm_probs.index)
        if len(common) == 0:
            return np.zeros(len(features_test), dtype=float)
        tvtp_arr = tvtp_probs.loc[common, ["p_low_vol", "p_high_vol"]].to_numpy()
        gmm_arr = gmm_probs.loc[common, ["p_0", "p_1", "p_2"]].to_numpy()
        fused = apply_log_opinion_pool(gmm_arr, tvtp_arr, mapping)

        # ---- 6. Posterior-weighted positions
        position_series = pd.Series(
            fused[:, 0] * state_positions[0]
            + fused[:, 1] * state_positions[1]
            + fused[:, 2] * state_positions[2],
            index=common,
        )

        # ---- 7. Causal shift + reindex to test
        position_series = position_series.shift(1).fillna(0.0)
        aligned = position_series.reindex(features_test.index).fillna(0.0)
        return aligned.to_numpy(dtype=float)

    return strategy_fn


# ---------------------------------------------------------------------------
# Internal helper — train/test split for GMM-HMM
# ---------------------------------------------------------------------------


def _fit_filter_gmm(
    combined_close: pd.Series,
    train_index: pd.Index,
    *,
    n_iter: int = 200,
    seed: int = 42,
) -> Optional[pd.DataFrame]:
    """Causal fit-on-train, filter-on-combined GMM-HMM."""
    from hmmlearn.hmm import GaussianHMM

    combined_feats = _build_features(combined_close)
    if len(combined_feats) < 100:
        return None

    train_mask = combined_feats.index.isin(train_index)
    if train_mask.sum() < 50:
        return None

    X_combined = combined_feats.to_numpy(dtype=float)
    X_train = X_combined[train_mask]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hmm = GaussianHMM(
                n_components=N_STATES,
                covariance_type="full",
                n_iter=n_iter,
                random_state=seed,
                tol=1e-3,
            )
            hmm.fit(X_train)

            # Variance ranking (col 1 = vol) → canonical low/mid/high order
            if hmm.covars_.ndim == 3:
                vol_var = np.array([c[1, 1] for c in hmm.covars_])
            else:
                vol_var = hmm.covars_[:, 1]
            order = np.argsort(vol_var)
            remap = {int(src): int(dst) for dst, src in enumerate(order)}

            # Forward filter — guarded against ill-conditioned covariances
            # that can fail on out-of-distribution OOS bars.
            probs_raw = _forward_filter(hmm, X_combined)
    except Exception:
        return None

    col_order = [None] * N_STATES
    for raw, canon in remap.items():
        col_order[canon] = raw
    probs = probs_raw[:, col_order]

    return pd.DataFrame(
        probs, index=combined_feats.index,
        columns=[f"p_{i}" for i in range(N_STATES)],
    )


__all__ = [
    "empirical_tvtp_3class_mapping",
    "apply_log_opinion_pool",
    "make_fusion_strategy",
]