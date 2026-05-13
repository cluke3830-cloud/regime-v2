"""GMM + HMM regime classifier — lightweight 3-state unsupervised baseline.

A complementary regime detector to the hand-tuned rule baseline. Where the
rule baseline encodes financial intuition via signed weights on 21 features,
this model lets the data speak: it fits a 3-state Gaussian HMM on a small
returns + vol feature set, then maps states to Bull/Neutral/Bear by
ascending variance (low-var → Bull, mid-var → Neutral, high-var → Bear).

Why have both?
- Rule baseline is interpretable but biased toward the designer's priors.
- GMM+HMM is unsupervised; surprises in the data shift the regime
  boundaries automatically.
- Disagreement between the two flags ambiguity; agreement = high confidence.

Architecture:
- Features: log return + EWMA realised vol (lag-1 to be causal).
- Model: hmmlearn GaussianHMM, full covariance, k_states=3.
- Filter: Hamilton α-pass (forward-only — no future leakage).
- Output: posterior P(Bull|t), P(Neutral|t), P(Bear|t) + Viterbi label.

This is intentionally minimal — a 2-feature model is easier to reason about
and less prone to overfitting than the 21-feature HSMM baseline.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning, module="hmmlearn")


N_STATES = 3
STATE_NAMES = {0: "Bull", 1: "Neutral", 2: "Bear"}
STATE_COLORS = {0: "#22c55e", 1: "#a3a3a3", 2: "#ef4444"}


def _build_features(close: pd.Series, vol_halflife: int = 20) -> pd.DataFrame:
    """Build a tiny 2-feature frame: log return + lag-1 EWMA vol.

    Vol is explicitly lagged by one bar (``.shift(1)``) so it depends only
    on returns through ``t-1`` — matching the rule-baseline convention in
    ``compute_features_v2`` (Brief 2.2) and removing any same-bar overlap
    between the return feature and the vol feature.

    The log return at bar t remains the contemporaneous return; callers
    must still shift output positions by one bar before trading to avoid
    lookahead.
    """
    r = np.log(close).diff()
    sq = r.pow(2)
    vol = sq.ewm(halflife=vol_halflife, adjust=False).mean().pow(0.5).shift(1)
    out = pd.DataFrame({"r": r, "vol": vol}).dropna()
    return out


def _forward_filter(hmm, X: np.ndarray) -> np.ndarray:
    """Hamilton (1989) α-pass — causal, no backward smoothing."""
    n = X.shape[0]
    k = hmm.n_components
    log_emit = hmm._compute_log_likelihood(X)
    log_trans = np.log(np.maximum(hmm.transmat_, 1e-300))
    log_start = np.log(np.maximum(hmm.startprob_, 1e-300))
    alphas = np.zeros((n, k))
    log_alpha = log_start + log_emit[0]
    log_alpha -= np.logaddexp.reduce(log_alpha)
    alphas[0] = np.exp(log_alpha)
    for t in range(1, n):
        log_alpha_prev = np.log(np.maximum(alphas[t - 1], 1e-300))
        log_pred = np.logaddexp.reduce(log_alpha_prev[:, None] + log_trans, axis=0)
        log_alpha = log_pred + log_emit[t]
        log_alpha -= np.logaddexp.reduce(log_alpha)
        alphas[t] = np.exp(log_alpha)
    return alphas


def compute_gmm_hmm_sequence(
    close: pd.Series,
    *,
    seed: int = 42,
    n_iter: int = 200,
) -> Optional[pd.DataFrame]:
    """Fit GMM+HMM on (log-return, EWMA vol) and return per-bar regime probs.

    Returns
    -------
    DataFrame indexed identically to ``close[1:]`` (or None if the fit
    fails) with columns:
      ``p_0, p_1, p_2`` — posterior probabilities (Bull / Neutral / Bear)
      ``label``         — Viterbi-style argmax remapped by variance
      ``regime``        — human-readable name
    """
    from hmmlearn.hmm import GaussianHMM

    feats = _build_features(close)
    if len(feats) < 100:
        return None
    X = feats.to_numpy(dtype=float)

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
            hmm.fit(X)
    except Exception:
        return None

    # Variance ranking on the vol-feature column (col 1) → canonical order.
    try:
        if hmm.covars_.ndim == 3:
            vol_var = np.array([c[1, 1] for c in hmm.covars_])
        else:
            vol_var = hmm.covars_[:, 1]
        order = np.argsort(vol_var)  # ascending: low → Bull, high → Bear
        remap = {int(src): int(dst) for dst, src in enumerate(order)}
    except Exception:
        remap = {k: k for k in range(N_STATES)}

    probs_raw = _forward_filter(hmm, X)
    # Reorder columns to canonical
    col_order = [None] * N_STATES
    for raw, canon in remap.items():
        col_order[canon] = raw
    probs = probs_raw[:, col_order]

    labels = probs.argmax(axis=1)
    out = pd.DataFrame(
        probs, index=feats.index,
        columns=[f"p_{i}" for i in range(N_STATES)],
    )
    out["label"] = labels
    out["regime"] = out["label"].map(STATE_NAMES)
    return out


def select_hmm_k(
    close: pd.Series,
    k_range: tuple[int, int] = (2, 5),
    n_iter: int = 200,
    seed: int = 42,
) -> dict[int, dict]:
    """Fit GaussianHMM for K = k_range[0] .. k_range[1] and return BIC/AIC per K.

    Justifies (or refutes) the K=3 assumption by letting the information
    criterion pick the optimal number of states from the data.

    Parameters
    ----------
    close : pd.Series
        Asset close prices.
    k_range : (int, int)
        Inclusive range of K values to evaluate.
    n_iter, seed : int
        Passed to GaussianHMM.

    Returns
    -------
    dict mapping K → {"aic": float, "bic": float, "log_likelihood": float,
                      "n_params": int, "n_obs": int}

    Parameter count formula (full covariance, d=2 features):
        n_params = (K-1) initial + K*(K-1) transition + K*d means
                 + K*(d*(d+1)//2) cov   →   K² + 5K - 1   for d=2
    """
    from hmmlearn.hmm import GaussianHMM

    feats = _build_features(close)
    if len(feats) < 100:
        return {}
    X = feats.to_numpy(dtype=float)
    n, d = X.shape

    results: dict[int, dict] = {}
    for k in range(k_range[0], k_range[1] + 1):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hmm = GaussianHMM(
                    n_components=k,
                    covariance_type="full",
                    n_iter=n_iter,
                    random_state=seed,
                    tol=1e-3,
                )
                hmm.fit(X)
                ll = hmm.score(X)
        except Exception:
            continue

        # (K-1) initial probs + K*(K-1) transition + K*d means + K*(d*(d+1)//2) full cov
        n_params = (k - 1) + k * (k - 1) + k * d + k * (d * (d + 1) // 2)
        bic = -2 * ll + n_params * np.log(n)
        aic = -2 * ll + 2 * n_params
        results[k] = {
            "aic": round(aic, 2),
            "bic": round(bic, 2),
            "log_likelihood": round(ll, 2),
            "n_params": n_params,
            "n_obs": n,
        }

    return results


__all__ = [
    "N_STATES",
    "STATE_NAMES",
    "STATE_COLORS",
    "compute_gmm_hmm_sequence",
    "select_hmm_k",
]
