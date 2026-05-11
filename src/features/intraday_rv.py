"""Intraday realised variance, semivariance, and Yang-Zhang fallback.

Brief 5.2 of the regime upgrade plan. Audit reference: §2.5, §5.2.13,
§8.5.2. The audit's headline claim: replacing squared-daily-return-
derived realised vol with proper intraday realised variance improves
signal-to-noise by 1-2 orders of magnitude.

What ships here:

  - ``compute_realised_variance(intraday_bars)``:
      Σ over intraday return² aggregated to daily resolution.
      Standard RV per Andersen-Bollerslev (1998).

  - ``compute_realised_semivariance(intraday_bars)``:
      Splits RV into upside (RSV+) and downside (RSV-) components
      per Barndorff-Nielsen et al. (2008). Upside-vs-downside ratio
      is a robust asymmetric-risk signal.

  - ``compute_realised_skewness(intraday_bars)``:
      Third-moment estimate from intraday returns. Daily-roll-up.

  - ``compute_yang_zhang_vol(ohlc, window)``:
      OHLC-only realised vol estimator (Yang & Zhang 2000). Used as
      the audit's "fallback when intraday unavailable" — much better
      than squared daily returns even without minute bars.

  - ``compute_bipower_variation(intraday_bars)``:
      Robust-to-jumps alternative to RV per Barndorff-Nielsen &
      Shephard (2004). Useful for separating "continuous" volatility
      from jump contributions.

Production plumbing:
  yfinance only serves ~60 days of 5-minute bars and ~7 days of 1-min
  bars — not enough for a 10-year backtest. To deploy these features
  in production, plumb to IBKR via the existing ``live_deployment/
  bar_fetcher.py`` (audit §8.5.2 deferred). The functions here accept
  ANY intraday OHLC DataFrame with a DatetimeIndex — they don't care
  where the data came from.

References
----------
Andersen, T. G., Bollerslev, T. (1998). Answering the Skeptics:
   Yes, Standard Volatility Models Do Provide Accurate Forecasts.
   *International Economic Review*, 39.
Barndorff-Nielsen, O. E. et al. (2008). Measuring Downside Risk:
   Realised Semivariance. *Festschrift in Honour of Robert F. Engle*.
Yang, D., Zhang, Q. (2000). Drift-Independent Volatility Estimation
   Based on High, Low, Open, and Close Prices. *J. Business*, 73.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Realised Variance / Semivariance / Skewness from intraday bars
# ---------------------------------------------------------------------------


def _intraday_log_returns(intraday_bars: pd.DataFrame) -> pd.Series:
    """Compute log returns from an intraday close series (any frequency).

    Input: DataFrame with DatetimeIndex and a 'close' column.
    Output: Series of log returns indexed by the same DatetimeIndex
    (first bar dropped).
    """
    if "close" not in intraday_bars.columns:
        raise KeyError("intraday_bars must have a 'close' column")
    return np.log(intraday_bars["close"]).diff().dropna()


def compute_realised_variance(
    intraday_bars: pd.DataFrame,
    daily_index: str = "D",
) -> pd.Series:
    """Daily realised variance from intraday returns.

    RV_t = Σ_i r_{t,i}² over all intraday returns r_{t,i} on day t.
    Output is the SUMMED variance (NOT the volatility — take sqrt
    if you want vol).

    Parameters
    ----------
    intraday_bars : pd.DataFrame
        Intraday bars with DatetimeIndex + 'close' column.
    daily_index : str, default 'D'
        pandas resample rule for the daily roll-up.

    Returns
    -------
    pd.Series
        Daily realised variance indexed by the date.
    """
    intraday_ret = _intraday_log_returns(intraday_bars)
    return (intraday_ret ** 2).resample(daily_index).sum()


def compute_realised_semivariance(
    intraday_bars: pd.DataFrame,
    daily_index: str = "D",
) -> Tuple[pd.Series, pd.Series]:
    """Daily upside / downside semivariance (Barndorff-Nielsen 2008).

    RSV+_t = Σ_i r_{t,i}² · I(r_{t,i} > 0)
    RSV-_t = Σ_i r_{t,i}² · I(r_{t,i} < 0)

    Returns ``(rsv_pos, rsv_neg)`` aligned to daily index.
    """
    intraday_ret = _intraday_log_returns(intraday_bars)
    pos = (intraday_ret.clip(lower=0) ** 2).resample(daily_index).sum()
    neg = (intraday_ret.clip(upper=0) ** 2).resample(daily_index).sum()
    return pos, neg


def compute_realised_skewness(
    intraday_bars: pd.DataFrame,
    daily_index: str = "D",
) -> pd.Series:
    """Daily realised skewness from intraday returns (Amaya et al. 2015).

    RS_t = √n · Σ r³ / (Σ r²)^{3/2}, where n is the number of intraday
    returns on day t.
    """
    intraday_ret = _intraday_log_returns(intraday_bars)
    n_per_day = intraday_ret.resample(daily_index).count()
    r2_sum = (intraday_ret ** 2).resample(daily_index).sum()
    r3_sum = (intraday_ret ** 3).resample(daily_index).sum()
    # Guard against zero denominators
    denom = (r2_sum ** 1.5).replace(0, np.nan)
    rs = np.sqrt(n_per_day) * r3_sum / denom
    return rs.fillna(0.0)


def compute_bipower_variation(
    intraday_bars: pd.DataFrame,
    daily_index: str = "D",
) -> pd.Series:
    """Bipower variation — jump-robust realised variance estimator.

    BV_t = (π/2) · Σ_{i>0} |r_{t,i}| · |r_{t,i-1}|

    The product of adjacent absolute returns kills the jump-component
    while preserving the continuous-vol component. Comparing RV vs BV
    on the same day flags jumps: large RV - BV → jump occurred.
    """
    intraday_ret = _intraday_log_returns(intraday_bars).abs()
    paired = intraday_ret * intraday_ret.shift(1)
    bv = (np.pi / 2.0) * paired.resample(daily_index).sum()
    return bv


# ---------------------------------------------------------------------------
# Yang-Zhang OHLC realised vol (audit fallback)
# ---------------------------------------------------------------------------


def compute_yang_zhang_vol(
    ohlc: pd.DataFrame, window: int = 20,
) -> pd.Series:
    """Yang-Zhang OHLC realised vol — best estimator from OHLC alone.

    Combines three components:
      - Overnight (close-to-open) return variance
      - Open-to-close drift variance
      - Rogers-Satchell intraday vol estimator

    The weighting parameter k is set per Yang & Zhang (2000) to
    minimise estimator variance.

    Parameters
    ----------
    ohlc : pd.DataFrame
        Daily OHLC with columns ``open``, ``high``, ``low``, ``close``.
    window : int, default=20
        Rolling window (in days) for the rolling variance estimates.

    Returns
    -------
    pd.Series
        Daily annualised volatility forecast.
    """
    for col in ("open", "high", "low", "close"):
        if col not in ohlc.columns:
            raise KeyError(f"ohlc must have '{col}' column")
    o = ohlc["open"]
    h = ohlc["high"]
    l = ohlc["low"]
    c = ohlc["close"]
    c_prev = c.shift(1)

    # Overnight return: log(open / prev close)
    log_oc = np.log(o / c_prev)
    # Open-to-close return: log(close / open)
    log_co = np.log(c / o)
    # Rogers-Satchell: log(h/c)·log(h/o) + log(l/c)·log(l/o)
    log_ho = np.log(h / o)
    log_hc = np.log(h / c)
    log_lo = np.log(l / o)
    log_lc = np.log(l / c)
    rs = log_hc * log_ho + log_lc * log_lo

    # Rolling variances
    sigma_oc_2 = log_oc.rolling(window).var()
    sigma_co_2 = log_co.rolling(window).var()
    sigma_rs   = rs.rolling(window).mean()

    # Yang-Zhang weighting parameter
    n = window
    k = 0.34 / (1.34 + (n + 1) / (n - 1))

    sigma2 = sigma_oc_2 + k * sigma_co_2 + (1 - k) * sigma_rs
    return np.sqrt(sigma2.clip(lower=0)) * np.sqrt(252)


__all__ = [
    "compute_realised_variance",
    "compute_realised_semivariance",
    "compute_realised_skewness",
    "compute_bipower_variation",
    "compute_yang_zhang_vol",
]
