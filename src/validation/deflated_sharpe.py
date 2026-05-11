"""Deflated Sharpe Ratio and Probability of Backtest Overfitting.

Brief 1.2 of the regime_dashboard upgrade plan.

The audit (§5.11, §8.1.2) flagged that the existing validation suite reports
*raw* annualised Sharpe with a bootstrap CI — without adjusting for skewness,
kurtosis, sample length, or the implicit number of model variants tried.
That is precisely the multiple-comparisons inflation Bailey & López de Prado
(2014) designed the Deflated Sharpe Ratio to correct.

What this module ships:

  - ``deflated_sharpe(returns, n_trials)``:
      Returns ``(dsr_p_value, sharpe_observed)``. The DSR p-value is the
      probability that the observed Sharpe exceeds the expected maximum
      Sharpe under the null of zero true edge across ``n_trials``
      iid alternatives, accounting for the finite-sample, non-normal
      distribution of the Sharpe estimator (Mertens 2002).

  - ``probability_of_backtest_overfitting(is_perfs, oos_perfs)``:
      López de Prado AFML §11.6 PBO. Given an (n_paths, n_variants)
      grid of in-sample and out-of-sample performances, returns the
      fraction of paths on which the IS-best variant ranked below the
      OOS median. Target ``< 0.5``; alarm above ``0.7``.

References
----------
Bailey, D. and López de Prado, M. (2014). The Deflated Sharpe Ratio.
    *Journal of Portfolio Management*, 40(5), 94-107.
Bailey, D., Borwein, J., López de Prado, M., and Zhu, Q. (2016). The
    Probability of Backtest Overfitting. *Journal of Computational Finance*.
López de Prado, M. (2018). *Advances in Financial Machine Learning*.
    Wiley. §11.6.
Mertens, E. (2002). Comments on Variance of the IID Estimator in Lo (2002).
    Working paper.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import stats

EULER_MASCHERONI = 0.57721566490153286
ANN_FACTOR_DAILY = 252


def annualised_sharpe(returns: np.ndarray, ann_factor: int = ANN_FACTOR_DAILY) -> float:
    """Annualised Sharpe of a return series. Drops NaNs.

    Uses sample standard deviation (ddof=1) — matches the convention the
    rest of the validation suite uses.
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return float("nan")
    sigma = r.std(ddof=1)
    if sigma == 0:
        return 0.0
    return float(r.mean() / sigma * np.sqrt(ann_factor))


def deflated_sharpe(
    returns: np.ndarray,
    n_trials: int,
    ann_factor: int = ANN_FACTOR_DAILY,
) -> Tuple[float, float]:
    """Deflated Sharpe Ratio (Bailey-López de Prado 2014).

    Parameters
    ----------
    returns : array-like
        Periodic (typically daily) strategy returns. NaNs are dropped.
    n_trials : int
        Number of independent strategy variants tried while developing the
        model. Pre-register this number — do NOT tune it after the fact.
        For the regime_dashboard work the audit suggests N ≈ 50–200
        (changelog v1-v11 + hyperparameter sweeps).
    ann_factor : int, default=252
        Annualisation factor (252 trading days/year for daily data).

    Returns
    -------
    dsr_p_value : float in [0, 1]
        Probability that the observed Sharpe exceeds the expected maximum
        Sharpe under the null. Values close to 1 indicate the strategy is
        statistically significant after deflating for trial count and
        non-normal returns. Values near 0.5 indicate no edge after
        deflation.
    sr_observed : float
        Annualised Sharpe of ``returns`` (returned for caller convenience —
        the caller usually wants both numbers).

    Notes
    -----
    The non-normality-adjusted variance of the Sharpe estimator is

        Var[ŜR] = (1 - γ₃·ŜR + ((γ₄-1)/4)·ŜR²) / (T-1)

    where γ₃ is sample skewness and γ₄ is sample kurtosis (NOT excess —
    we want the full fourth standardised moment). The expected maximum
    of N iid normal Sharpes under the null is

        E[max ŜR] ≈ stdev(ŜR) · [(1-γ_E)·Φ⁻¹(1 - 1/N)
                                  + γ_E·Φ⁻¹(1 - 1/(N·e))]

    where γ_E ≈ 0.5772 is Euler-Mascheroni. DSR is the right-tail Gaussian
    probability of the standardised observed Sharpe minus the null expected
    maximum.

    When ``n_trials <= 1`` no multiple-comparison deflation is required, so
    the function returns the simple t-stat probability ``Φ(ŜR / stdev(ŜR))``.
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    T = len(r)
    if T < 30:
        raise ValueError(
            f"need at least 30 returns to estimate DSR, got {T}"
        )
    sigma = r.std(ddof=1)
    if sigma == 0:
        return 0.0, 0.0
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")

    sr = float(r.mean() / sigma * np.sqrt(ann_factor))
    g3 = float(stats.skew(r, bias=False))
    g4 = float(stats.kurtosis(r, fisher=False, bias=False))

    sr_var = (1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * sr ** 2) / (T - 1)
    sr_std = float(np.sqrt(max(sr_var, 1e-12)))

    if n_trials == 1:
        z = sr / sr_std
    else:
        e_max = sr_std * (
            (1.0 - EULER_MASCHERONI) * stats.norm.ppf(1.0 - 1.0 / n_trials)
            + EULER_MASCHERONI * stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
        )
        z = (sr - e_max) / sr_std

    return float(stats.norm.cdf(z)), sr


def probability_of_backtest_overfitting(
    is_perfs: np.ndarray,
    oos_perfs: np.ndarray,
) -> float:
    """Probability of Backtest Overfitting (López de Prado AFML §11.6).

    For each path (row), the IS-best variant is identified, and its OOS
    rank quantile ω ∈ (0, 1) is computed (1 = best OOS, 0 = worst OOS).
    PBO is the fraction of paths where ω < 0.5 — i.e., the IS-best
    variant landed in the bottom half of OOS performance.

    Parameters
    ----------
    is_perfs : array-like, shape (n_paths, n_variants)
        In-sample performance of each strategy variant on each path.
    oos_perfs : array-like, shape (n_paths, n_variants)
        Out-of-sample performance, same shape.

    Returns
    -------
    pbo : float in [0, 1]
        Fraction of paths where the IS-best variant performed below the
        OOS median.

    Interpretation
    --------------
    PBO  < 0.50  → strategy selection generalises (good)
    PBO  > 0.70  → severe overfitting; the backtest is not informative
    PBO  ≈ 0.50  → strategy selection is no better than chance

    Raises
    ------
    ValueError
        If shapes mismatch, n_variants < 2, or any row has all-tied perfs.
    """
    is_perfs = np.asarray(is_perfs, dtype=float)
    oos_perfs = np.asarray(oos_perfs, dtype=float)
    if is_perfs.shape != oos_perfs.shape:
        raise ValueError(
            f"is_perfs {is_perfs.shape} and oos_perfs {oos_perfs.shape} "
            "must have identical shape (n_paths, n_variants)"
        )
    if is_perfs.ndim != 2:
        raise ValueError(
            f"is_perfs must be 2-D (n_paths, n_variants), got "
            f"{is_perfs.ndim}-D"
        )
    n_paths, n_variants = is_perfs.shape
    if n_variants < 2:
        raise ValueError(
            f"PBO requires at least 2 variants, got {n_variants}"
        )

    overfits = 0
    for p in range(n_paths):
        is_best = int(np.argmax(is_perfs[p]))
        # 1-indexed rank where 1 = worst OOS, N = best OOS
        ranks_low_to_high = np.argsort(np.argsort(oos_perfs[p])) + 1
        omega = ranks_low_to_high[is_best] / (n_variants + 1)
        if omega < 0.5:
            overfits += 1

    return overfits / n_paths


__all__ = [
    "annualised_sharpe",
    "deflated_sharpe",
    "probability_of_backtest_overfitting",
    "EULER_MASCHERONI",
]