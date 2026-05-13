"""Regime diagnostic utilities — NBER alignment, calibration, stability, concordance.

Four standalone functions that can be called from make_validation_report.py or
interactively. All are pure-Python + numpy/pandas; no model fitting required.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. NBER recession alignment
# ---------------------------------------------------------------------------


def nber_alignment(
    labels: pd.Series,
    fred_api_key: Optional[str] = None,
) -> dict:
    """Measure how well Bear labels (label==2) align with NBER recessions.

    FRED series USREC: 1 = NBER recession, 0 = expansion (monthly).
    Fetched, resampled to business-day frequency, and aligned to *labels*.

    Parameters
    ----------
    labels : pd.Series
        DatetimeIndex'd regime labels (0=Bull, 1=Neutral, 2=Bear).
    fred_api_key : str, optional
        FRED API key. If None, the function tries the FRED_API_KEY env var.

    Returns
    -------
    dict with keys:
        precision   — fraction of Bear bars that fall in an NBER recession
        recall      — fraction of NBER recession bars labeled Bear
        f1          — harmonic mean of precision + recall
        median_days_to_bear — median lag (calendar days) from recession start
                              to first Bear label within that recession period
        n_recession_bars    — daily bars classified as recession
        n_bear_bars         — daily bars labeled Bear
        n_overlap           — bars that are both Bear and recession
        usrec_available     — whether FRED fetch succeeded
    """
    import os

    key = fred_api_key or os.environ.get("FRED_API_KEY")
    usrec = None
    try:
        import fredapi
        fred = fredapi.Fred(api_key=key)
        start = labels.index.min().strftime("%Y-%m-%d")
        end = labels.index.max().strftime("%Y-%m-%d")
        usrec_monthly = fred.get_series("USREC", start, end)
        # Upsample monthly → business daily, forward-fill
        usrec = (
            usrec_monthly
            .resample("B")
            .ffill()
            .reindex(labels.index)
            .ffill()
            .fillna(0)
        )
    except Exception as exc:
        return {
            "usrec_available": False,
            "error": str(exc)[:120],
            "precision": None, "recall": None, "f1": None,
            "median_days_to_bear": None,
            "n_recession_bars": None, "n_bear_bars": None, "n_overlap": None,
        }

    bear = (labels == 2).astype(int)
    recession = (usrec > 0).astype(int)

    n_bear = bear.sum()
    n_rec = recession.sum()
    n_overlap = (bear & recession).sum()

    precision = float(n_overlap / n_bear) if n_bear > 0 else 0.0
    recall = float(n_overlap / n_rec) if n_rec > 0 else 0.0
    denom = precision + recall
    f1 = 2 * precision * recall / denom if denom > 0 else 0.0

    # Median days-to-first-Bear for each NBER recession episode
    lags = []
    in_rec = False
    rec_start = None
    for date, r in usrec.items():
        if r == 1 and not in_rec:
            in_rec = True
            rec_start = date
        elif r == 0 and in_rec:
            in_rec = False
            # Find first Bear bar in [rec_start, date)
            window = bear.loc[rec_start:date]
            bear_dates = window[window == 1].index
            if len(bear_dates) > 0:
                lags.append((bear_dates[0] - rec_start).days)

    return {
        "usrec_available": True,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "median_days_to_bear": int(np.median(lags)) if lags else None,
        "n_recession_bars": int(n_rec),
        "n_bear_bars": int(n_bear),
        "n_overlap": int(n_overlap),
    }


# ---------------------------------------------------------------------------
# 2. Reliability diagram + ECE
# ---------------------------------------------------------------------------


def reliability_diagram(
    proba: np.ndarray,
    reference_labels: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Compute reliability diagram data and Expected Calibration Error (ECE).

    For each class c, treats the problem as binary (P(label=c) vs actual==c)
    and bins predicted probabilities into *n_bins* equal-width buckets.

    Parameters
    ----------
    proba : np.ndarray, shape (n, K)
        Model soft probabilities (rows sum to 1).
    reference_labels : np.ndarray, shape (n,)
        Hard labels 0..K-1 treated as ground truth.
    n_bins : int
        Number of equal-width probability bins in [0, 1].

    Returns
    -------
    dict:
        ece_per_class  — list of per-class ECE values
        mean_ece       — average ECE across classes
        bins           — list of per-class bin data dicts (for plotting)
    """
    proba = np.asarray(proba, dtype=float)
    reference_labels = np.asarray(reference_labels, dtype=int)
    n, K = proba.shape
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    ece_per_class = []
    bins_per_class = []

    for c in range(K):
        p_c = proba[:, c]
        y_c = (reference_labels == c).astype(float)

        bin_acc = np.zeros(n_bins)
        bin_conf = np.zeros(n_bins)
        bin_count = np.zeros(n_bins, dtype=int)

        for b in range(n_bins):
            lo, hi = bin_edges[b], bin_edges[b + 1]
            mask = (p_c >= lo) & (p_c < hi) if b < n_bins - 1 else (p_c >= lo) & (p_c <= hi)
            if mask.sum() > 0:
                bin_acc[b] = y_c[mask].mean()
                bin_conf[b] = p_c[mask].mean()
                bin_count[b] = mask.sum()

        ece_c = float(np.sum(bin_count / n * np.abs(bin_conf - bin_acc)))
        ece_per_class.append(round(ece_c, 4))
        bins_per_class.append({
            "class": c,
            "bin_centers": bin_centers.tolist(),
            "bin_accuracy": bin_acc.tolist(),
            "bin_confidence": bin_conf.tolist(),
            "bin_count": bin_count.tolist(),
            "ece": round(ece_c, 4),
        })

    return {
        "ece_per_class": ece_per_class,
        "mean_ece": round(float(np.mean(ece_per_class)), 4),
        "bins": bins_per_class,
    }


# ---------------------------------------------------------------------------
# 3. Regime stability index
# ---------------------------------------------------------------------------


def regime_stability(labels: pd.Series) -> dict:
    """Run-length statistics per regime and global flip rate.

    Parameters
    ----------
    labels : pd.Series
        DatetimeIndex'd integer regime labels (0/1/2).

    Returns
    -------
    dict:
        per_regime — dict keyed by label int: mean/median duration, n_episodes, pct_time
        flip_rate  — n_transitions / (n_bars - 1)
        dominant_regime — label with longest mean run
    """
    arr = labels.to_numpy(dtype=int)
    n = len(arr)
    if n == 0:
        return {"per_regime": {}, "flip_rate": None, "dominant_regime": None}

    # Run-length encoding
    runs: list[tuple[int, int]] = []
    cur_label = arr[0]
    cur_len = 1
    for i in range(1, n):
        if arr[i] == cur_label:
            cur_len += 1
        else:
            runs.append((cur_label, cur_len))
            cur_label = arr[i]
            cur_len = 1
    runs.append((cur_label, cur_len))

    n_transitions = len(runs) - 1
    flip_rate = n_transitions / (n - 1) if n > 1 else 0.0

    by_regime: dict[int, list[int]] = defaultdict(list)
    for lbl, dur in runs:
        by_regime[lbl].append(dur)

    per_regime = {}
    for lbl, durs in sorted(by_regime.items()):
        per_regime[int(lbl)] = {
            "mean_duration_bars": round(float(np.mean(durs)), 2),
            "median_duration_bars": float(np.median(durs)),
            "n_episodes": len(durs),
            "pct_time": round(float(sum(durs) / n), 4),
        }

    dominant = max(per_regime.items(), key=lambda kv: kv[1]["mean_duration_bars"])[0]

    return {
        "per_regime": per_regime,
        "flip_rate": round(flip_rate, 4),
        "dominant_regime": dominant,
    }


# ---------------------------------------------------------------------------
# 4. Cross-model concordance
# ---------------------------------------------------------------------------


def cross_model_concordance(
    rule_labels: pd.Series,
    gmm_labels: pd.Series,
    tvtp_proba: pd.DataFrame,
) -> dict:
    """Pairwise agreement rates and consensus score across three models.

    Parameters
    ----------
    rule_labels : pd.Series
        Rule-baseline labels (0/1/2).
    gmm_labels : pd.Series
        GMM-HMM labels (0/1/2).
    tvtp_proba : pd.DataFrame
        Must contain columns ``p_low_vol`` and ``p_high_vol``.

    Returns
    -------
    dict:
        rule_gmm_agreement    — fraction of bars both agree (3-class)
        rule_tvtp_agreement   — fraction of bars both agree (Bear vs non-Bear)
        gmm_tvtp_agreement    — fraction of bars both agree (Bear vs non-Bear)
        consensus_score       — fraction of bars where ≥2 of 3 models agree
        cohen_kappa_rule_gmm  — Cohen's κ between rule and GMM labels
        confusion_rule_gmm    — 3×3 confusion matrix (row=rule, col=gmm)
        n_aligned             — number of bars in common index
    """
    # Align to common index
    common = rule_labels.index.intersection(gmm_labels.index).intersection(tvtp_proba.index)
    if len(common) == 0:
        return {"n_aligned": 0, "error": "No common index bars"}

    rl = rule_labels.reindex(common).fillna(1).astype(int).to_numpy()
    gl = gmm_labels.reindex(common).fillna(1).astype(int).to_numpy()
    p_high = tvtp_proba["p_high_vol"].reindex(common).fillna(0.5).to_numpy()
    # Map TVTP → 3-class: high-vol (>0.5) → Bear (2), else Bull (0)
    tl = (p_high > 0.5).astype(int) * 2

    n = len(rl)

    # Pairwise agreement
    rule_gmm_agree = float((rl == gl).mean())
    rule_tvtp_agree = float(((rl == 2) == (tl == 2)).mean())
    gmm_tvtp_agree = float(((gl == 2) == (tl == 2)).mean())

    # Consensus: ≥2 of 3 agree on the same label
    agreement_rg = rl == gl
    agreement_rt = rl == tl
    agreement_gt = gl == tl
    consensus = float((agreement_rg | agreement_rt | agreement_gt).mean())

    # Cohen's κ (rule vs GMM)
    kappa = _cohen_kappa(rl, gl, n_classes=3)

    # Confusion matrix (rule=row, gmm=col)
    conf = np.zeros((3, 3), dtype=int)
    for r, g in zip(rl, gl):
        if 0 <= r < 3 and 0 <= g < 3:
            conf[r, g] += 1
    confusion = conf.tolist()

    return {
        "rule_gmm_agreement": round(rule_gmm_agree, 4),
        "rule_tvtp_agreement": round(rule_tvtp_agree, 4),
        "gmm_tvtp_agreement": round(gmm_tvtp_agree, 4),
        "consensus_score": round(consensus, 4),
        "cohen_kappa_rule_gmm": round(kappa, 4),
        "confusion_rule_gmm": confusion,
        "n_aligned": n,
    }


def _cohen_kappa(a: np.ndarray, b: np.ndarray, n_classes: int) -> float:
    """Cohen's κ for multi-class labels."""
    n = len(a)
    if n == 0:
        return 0.0
    p_o = float((a == b).mean())  # observed agreement
    # Expected agreement under independence
    p_e = sum(
        float((a == c).mean()) * float((b == c).mean())
        for c in range(n_classes)
    )
    if abs(1 - p_e) < 1e-12:
        return 1.0 if p_o >= 1.0 else 0.0
    return (p_o - p_e) / (1.0 - p_e)


__all__ = [
    "nber_alignment",
    "reliability_diagram",
    "regime_stability",
    "cross_model_concordance",
]
