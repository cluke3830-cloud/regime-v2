"""Drift monitor — Population Stability Index, MMD, calibration drift.

Brief 5.1 of the regime upgrade plan. Audit §7 enumerates four
failure-mode scenarios that will eventually hit a live regime detector;
all four share a common antidote: a continuous monitoring layer that
watches inputs, parameters, and outputs for distribution shift.

Three drift signals shipped here:

  - **PSI** (Population Stability Index): bin-aligned distributional
    distance between a reference window and a current window. Standard
    risk-management metric, threshold ≥ 0.25 = "significant" shift,
    ≥ 0.50 = "major" shift.

  - **MMD** (Maximum Mean Discrepancy with Gaussian kernel): joint-
    distribution distance across all features simultaneously. Catches
    multivariate shifts that PSI's per-feature view misses (e.g., the
    individual feature distributions are unchanged but their CORRELATION
    structure flipped).

  - **Calibration drift**: empirical coverage gap of conformal sets vs
    target. If a conformal calibrator targets 90% coverage but the
    realised coverage drops to 75%, the model's miscalibration has
    drifted.

A single ``DriftMonitor`` class plumbs all three together: it accepts
a reference data snapshot at construction, then exposes ``check_drift``
which returns a dict of all three signals + a boolean ``trigger``
field (True iff any signal exceeds its threshold).

Usage pattern in production:
    monitor = DriftMonitor(reference_features, reference_proba)
    for new_batch in live_stream:
        result = monitor.check_drift(new_batch)
        if result["trigger"]:
            alert(result["reason"])
            schedule_retrain()
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# PSI
# ---------------------------------------------------------------------------


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-4,
) -> float:
    """Population Stability Index between two 1-D distributions.

    Bins are derived from ``reference`` quantiles (10 bins by default).
    Each bin's reference and current fractions are compared via
    ``(p_curr - p_ref) * log(p_curr / p_ref)`` and summed. Both
    fractions are floored at ``eps`` to avoid log(0).

    Thresholds (industry-standard):
      < 0.10  → no significant shift
      0.10-0.25 → moderate shift; investigate
      ≥ 0.25  → significant shift; retrain candidate

    Parameters
    ----------
    reference, current : 1-D arrays
        Samples from the two distributions. NaNs are dropped.
    n_bins : int
        Number of quantile bins.
    eps : float
        Lower clip on per-bin fractions.

    Returns
    -------
    float
        PSI value (non-negative).
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if len(ref) < n_bins or len(cur) < n_bins:
        return float("nan")

    # Quantile bin edges from reference (np.unique to handle ties)
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(ref, quantiles))
    if len(edges) < 3:
        return 0.0  # reference is degenerate (near-constant)
    # Extend outer edges to ±inf so all samples bin
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_frac = np.maximum(ref_counts / max(ref_counts.sum(), 1), eps)
    cur_frac = np.maximum(cur_counts / max(cur_counts.sum(), 1), eps)
    psi = float(np.sum((cur_frac - ref_frac) * np.log(cur_frac / ref_frac)))
    return psi


# ---------------------------------------------------------------------------
# MMD with Gaussian (RBF) kernel
# ---------------------------------------------------------------------------


def rolling_mmd(
    reference: np.ndarray,
    current: np.ndarray,
    sigma: Optional[float] = None,
    max_samples: int = 500,
    rng_seed: int = 42,
) -> float:
    """Maximum Mean Discrepancy² with a Gaussian kernel.

    Computes the unbiased U-statistic estimate of the squared MMD
    between two multivariate distributions. ``sigma`` defaults to the
    median pairwise Euclidean distance across the pooled data
    ("median heuristic").

    For computational tractability, subsamples both distributions to
    at most ``max_samples`` points each (with seeded RNG for
    reproducibility).

    Parameters
    ----------
    reference, current : (n_ref, d), (n_cur, d) arrays
        Two multivariate samples.
    sigma : float, optional
        Gaussian kernel bandwidth. None → median-heuristic.
    max_samples : int
        Upper cap on samples per distribution.

    Returns
    -------
    float
        MMD² estimate. Larger = more distributional difference.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    if ref.ndim == 1:
        ref = ref.reshape(-1, 1)
    if cur.ndim == 1:
        cur = cur.reshape(-1, 1)
    # Drop NaN rows
    ref = ref[~np.isnan(ref).any(axis=1)]
    cur = cur[~np.isnan(cur).any(axis=1)]
    if len(ref) < 2 or len(cur) < 2:
        return float("nan")

    rng = np.random.default_rng(rng_seed)
    if len(ref) > max_samples:
        idx = rng.choice(len(ref), size=max_samples, replace=False)
        ref = ref[idx]
    if len(cur) > max_samples:
        idx = rng.choice(len(cur), size=max_samples, replace=False)
        cur = cur[idx]

    # Median heuristic for sigma
    if sigma is None:
        pooled = np.vstack([ref, cur])
        n_pool = pooled.shape[0]
        sub = pooled[rng.choice(n_pool, size=min(n_pool, 300), replace=False)]
        diffs = sub[:, None, :] - sub[None, :, :]
        dists2 = np.sum(diffs ** 2, axis=-1)
        med_sq = np.median(dists2[dists2 > 0]) if (dists2 > 0).any() else 1.0
        sigma = float(np.sqrt(med_sq / 2.0)) or 1.0

    inv_2s2 = 1.0 / (2.0 * sigma ** 2)

    def _kernel(a, b):
        diffs = a[:, None, :] - b[None, :, :]
        d2 = np.sum(diffs ** 2, axis=-1)
        return np.exp(-d2 * inv_2s2)

    Kxx = _kernel(ref, ref)
    Kyy = _kernel(cur, cur)
    Kxy = _kernel(ref, cur)
    n, m = len(ref), len(cur)
    # Unbiased estimator: subtract diagonal
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    mmd2 = (
        Kxx.sum() / (n * (n - 1))
        + Kyy.sum() / (m * (m - 1))
        - 2.0 * Kxy.sum() / (n * m)
    )
    return float(max(mmd2, 0.0))


# ---------------------------------------------------------------------------
# Calibration drift
# ---------------------------------------------------------------------------


def calibration_drift(
    reference_coverage: float,
    current_coverage: float,
    target_coverage: float,
) -> float:
    """Calibration drift = |current - target| - |reference - target|.

    Positive → calibration has WORSENED vs reference. Negative →
    calibration has IMPROVED.
    """
    ref_gap = abs(reference_coverage - target_coverage)
    cur_gap = abs(current_coverage - target_coverage)
    return float(cur_gap - ref_gap)


# ---------------------------------------------------------------------------
# All-in-one monitor
# ---------------------------------------------------------------------------


class DriftMonitor:
    """End-to-end drift monitor for production deployments.

    Constructs from a reference snapshot. Each ``check_drift(new_data)``
    call returns a dict of PSI / MMD / calibration metrics + a single
    boolean trigger.

    Parameters
    ----------
    reference_features : pd.DataFrame
        Reference feature snapshot. PSI runs per column; MMD on the
        joint distribution.
    reference_coverage : float, optional
        Conformal-coverage rate at reference. Used by
        ``calibration_drift``. Skip if not using a conformal-calibrated
        strategy.
    target_coverage : float, default=0.90
        Target coverage for ``calibration_drift``.
    psi_threshold : float, default=0.25
    mmd_threshold : float, default=0.10
    cal_drift_threshold : float, default=0.05
    """

    def __init__(
        self, reference_features: pd.DataFrame,
        *,
        reference_coverage: Optional[float] = None,
        target_coverage: float = 0.90,
        psi_threshold: float = 0.25,
        mmd_threshold: float = 0.10,
        cal_drift_threshold: float = 0.05,
    ):
        self.reference_features = reference_features
        self.reference_coverage = reference_coverage
        self.target_coverage = target_coverage
        self.psi_threshold = psi_threshold
        self.mmd_threshold = mmd_threshold
        self.cal_drift_threshold = cal_drift_threshold

    def check_drift(
        self,
        current_features: pd.DataFrame,
        current_coverage: Optional[float] = None,
    ) -> Dict[str, object]:
        """Compute all drift signals on a new feature batch.

        Returns
        -------
        dict
          {
            "psi_per_col": {col: float, ...},
            "psi_max": float,
            "mmd2": float,
            "cal_drift": float or None,
            "triggers": list[str],  # names of breached metrics
            "trigger": bool,
          }
        """
        psi_per_col: Dict[str, float] = {}
        for col in self.reference_features.columns:
            if col not in current_features.columns:
                continue
            psi_per_col[col] = population_stability_index(
                self.reference_features[col].to_numpy(),
                current_features[col].to_numpy(),
            )
        psi_max = max(
            (v for v in psi_per_col.values() if np.isfinite(v)),
            default=float("nan"),
        )

        # MMD over the common-columns joint
        common = [c for c in self.reference_features.columns
                  if c in current_features.columns]
        mmd2 = rolling_mmd(
            self.reference_features[common].to_numpy(),
            current_features[common].to_numpy(),
        )

        cal_d: Optional[float] = None
        if self.reference_coverage is not None and current_coverage is not None:
            cal_d = calibration_drift(
                reference_coverage=self.reference_coverage,
                current_coverage=current_coverage,
                target_coverage=self.target_coverage,
            )

        triggers = []
        if np.isfinite(psi_max) and psi_max >= self.psi_threshold:
            triggers.append(f"psi_max={psi_max:.3f}>={self.psi_threshold}")
        if np.isfinite(mmd2) and mmd2 >= self.mmd_threshold:
            triggers.append(f"mmd2={mmd2:.3f}>={self.mmd_threshold}")
        if cal_d is not None and cal_d >= self.cal_drift_threshold:
            triggers.append(f"cal_drift={cal_d:.3f}>={self.cal_drift_threshold}")

        return {
            "psi_per_col": psi_per_col,
            "psi_max": psi_max,
            "mmd2": mmd2,
            "cal_drift": cal_d,
            "triggers": triggers,
            "trigger": bool(triggers),
        }


__all__ = [
    "population_stability_index",
    "rolling_mmd",
    "calibration_drift",
    "DriftMonitor",
]
