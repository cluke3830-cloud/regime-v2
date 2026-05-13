"""Tier-1 price features for the regime classifier.

Brief 2.1.1 of the regime upgrade plan. Built to fix the empirical failure
in Brief 2.1: xgb_v1 had only 2 features (``mom_20``, ``vol_ewma``) and
lost to a literal 20-day moving average on SPY. The audit (§5.2) prescribes
~21 features for the rule layer; this module ships the 14 of those that
need *only price and return data*. The remaining 7 (cross-asset
correlations, FRED macro, VIX term structure, volume anomaly) need
external data and ship in Brief 2.1.2.

Causal hygiene — STRICT:
    Every feature at row ``t`` depends only on ``log_ret[<t]``.
    Implemented by shifting every rolling/EWMA series by 1 before storing.

    Decision convention: at the close of bar ``t-1``, observe features
    at index ``t``, decide ``position[t]``, hold through close of bar ``t``,
    earn ``log_ret[t]``. The CPCV harness ( :func:`run_cpcv_validation` )
    consumes positions and bar-returns under exactly this contract.

Feature inventory (audit §5.2 cross-reference in [brackets]):

    Multi-horizon momentum (audit §5.2.9):
        mom_5    — trailing 5-bar log-return sum
        mom_20   — trailing 20-bar log-return sum
        mom_63   — trailing 63-bar log-return sum (~1 quarter)
        mom_252  — trailing 252-bar log-return sum (~1 year)

    EWMA volatility pyramid (audit §5.2.1, §5.2.9):
        vol_short   — λ=0.85,  half-life ≈ 4 bars   (fast vol)
        vol_ewma    — λ=0.94,  half-life ≈ 11 bars  (RiskMetrics canonical)
        vol_long    — λ=0.98,  half-life ≈ 34 bars  (medium vol)
        vol_yearly  — λ=0.995, half-life ≈ 138 bars (slow vol)

    Vol structure (derived):
        vol_ratio_sl  — vol_short / vol_long  (expansion/contraction signal)
        vol_ratio_ly  — vol_long / vol_yearly (term-structure proxy)

    Shock and drawdown (audit §5.2.3, §5.2.5):
        shock_z       — EWMA z-score of |r_t|, lookback ≈ 50 bars
        drawdown_252  — close / rolling-252-max - 1

    Trend and autocorrelation (audit §5.2.4, §5.2.7):
        autocorr_63  — 63-bar rolling Pearson autocorrelation of log returns
        trend_dir    — 21-bar MA / 63-bar MA - 1

That's 14 features. ``close`` is also retained in the output (the
strategy adapter pulls it for triple-barrier label computation; XGBoost
itself receives only the 14 feature columns via :data:`FEATURE_COLUMNS_V1`).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


FEATURE_COLUMNS_V1: list[str] = [
    "mom_5", "mom_20", "mom_63", "mom_252",
    "vol_short", "vol_ewma", "vol_long", "vol_yearly",
    "vol_ratio_sl", "vol_ratio_ly",
    "shock_z",
    "drawdown_252",
    "autocorr_63",
    "trend_dir",
]

# Brief 2.1.2 — Tier-2 cross-asset & macro additions.
FEATURE_COLUMNS_V2_ADD: list[str] = [
    "vix_log",        # log(VIX) — fear gauge level
    "vix_change",     # 5-day change in VIX (acceleration signal)
    "vix_term",       # VIX / VIX3M — backwardation > 1 = stress
    "vix_slope",      # VIX3M - VIX — contango>0 (calm), backwardation<0 (stress)
    "corr_tlt_63",    # 63-bar rolling Pearson corr(asset, TLT)
    "corr_gld_63",    # 63-bar rolling Pearson corr(asset, GLD)
    "term_spread",    # T10Y2Y from FRED — yield curve
    "credit_spread",  # BAA10Y from FRED — credit risk premium
    "yang_zhang_vol", # Yang-Zhang (2000) vol from daily OHLC (20-bar window)
]

FEATURE_COLUMNS_V2: list[str] = FEATURE_COLUMNS_V1 + FEATURE_COLUMNS_V2_ADD

# Columns kept in the DataFrame but NOT fed to the classifier (needed for
# downstream computations like triple-barrier labels).
NON_FEATURE_COLUMNS: list[str] = ["close"]


def compute_features_v1(close: pd.Series) -> pd.DataFrame:
    """Causally-clean Tier-1 features from a close series.

    Parameters
    ----------
    close : pd.Series
        Monotone-time-indexed close-to-close prices.

    Returns
    -------
    pd.DataFrame
        Indexed by the same date index, with columns
        ``NON_FEATURE_COLUMNS + FEATURE_COLUMNS_V1``. Rows with any NaN
        (burn-in) are dropped — the leading ~252 bars typically.

    Causal invariant
    ----------------
    For every column ``c`` in :data:`FEATURE_COLUMNS_V1` and every row
    index ``t`` in the output, ``c[t]`` depends only on values of
    ``close`` at strictly earlier indices. Verified by the test suite via
    a "perturb future" check.

    Notes
    -----
    The four EWMA-vol horizons (λ ∈ {0.85, 0.94, 0.98, 0.995}) follow the
    audit §5.2.9 multi-scale pyramid. Together with ``vol_ratio_sl`` and
    ``vol_ratio_ly`` they give XGBoost a clean way to express
    "vol-of-vol" without a hand-engineered ratio in the basis function.

    ``min_periods=20`` keeps the EWMA series from spitting out values
    from bar 1 (where the estimate is dominated by the seed). Burn-in
    NaNs are dropped at the end.
    """
    if not isinstance(close, pd.Series):
        raise TypeError(
            f"close must be pd.Series, got {type(close).__name__}"
        )

    log_ret = np.log(close).diff()

    f = pd.DataFrame(index=close.index)
    f["close"] = close

    # ---- Multi-horizon momentum (audit §5.2.9) — shift(1) → strictly past
    for h in (5, 20, 63, 252):
        f[f"mom_{h}"] = log_ret.rolling(h).sum().shift(1)

    # ---- EWMA vol pyramid (audit §5.2.1, §5.2.9)
    # λ = 1 - α; we pass α directly. Names match audit nomenclature.
    vol_alphas = {
        "vol_short":  0.15,   # λ=0.85,  half-life ~4
        "vol_ewma":   0.06,   # λ=0.94,  half-life ~11 (RiskMetrics)
        "vol_long":   0.02,   # λ=0.98,  half-life ~34
        "vol_yearly": 0.005,  # λ=0.995, half-life ~138
    }
    for name, alpha in vol_alphas.items():
        # .ewm().std() at row t naturally includes log_ret[t] (a peek);
        # .shift(1) strips that out → uses only data through t-1.
        f[name] = log_ret.ewm(alpha=alpha, min_periods=20).std().shift(1)

    # ---- Vol ratios — guard against div-by-zero with replace+ffill
    f["vol_ratio_sl"] = f["vol_short"] / f["vol_long"].replace(0, np.nan)
    f["vol_ratio_ly"] = f["vol_long"]  / f["vol_yearly"].replace(0, np.nan)

    # ---- Shock-z (audit §5.2.5) — EWMA z-score of |r|
    abs_r = log_ret.abs()
    abs_mean = abs_r.ewm(alpha=0.02, min_periods=20).mean()
    abs_std  = abs_r.ewm(alpha=0.02, min_periods=20).std()
    f["shock_z"] = ((abs_r - abs_mean) / abs_std.replace(0, np.nan)).shift(1)

    # ---- Drawdown (audit §5.2.3)
    rolling_max = close.rolling(252, min_periods=20).max()
    f["drawdown_252"] = (close / rolling_max - 1).shift(1)

    # ---- Return autocorrelation (audit §5.2.4)
    f["autocorr_63"] = (
        log_ret.rolling(63).corr(log_ret.shift(1)).shift(1)
    )

    # ---- Trend direction (audit §5.2.7)
    ma_short = close.rolling(21).mean()
    ma_long  = close.rolling(63).mean()
    f["trend_dir"] = (ma_short / ma_long - 1).shift(1)

    f = f.dropna()
    return f


def compute_features_v2(
    close: pd.Series,
    *,
    ohlc: Optional["pd.DataFrame"] = None,
    vix: Optional[pd.Series] = None,
    vix3m: Optional[pd.Series] = None,
    tlt: Optional[pd.Series] = None,
    gld: Optional[pd.Series] = None,
    term_spread: Optional[pd.Series] = None,
    credit_spread: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Tier-2 features = Tier-1 + cross-asset + macro (Brief 2.1.2).

    Adds 7 columns on top of :func:`compute_features_v1`. Any auxiliary
    series passed as ``None`` results in that column being filled with
    NaN — XGBoost handles NaN natively, so a partial macro feed degrades
    gracefully rather than crashing.

    Causal hygiene — same contract as v1. Every Tier-2 column is
    ``.shift(1)``-ed before storage so ``features[t]`` uses only
    auxiliary data through ``t-1``.

    Parameters
    ----------
    close : pd.Series
        Underlying asset's close-to-close prices.
    vix, vix3m : pd.Series, optional
        CBOE VIX and 3-month VIX close series. Need both for ``vix_term``;
        ``vix`` alone is enough for ``vix_log`` and ``vix_change``.
    tlt, gld : pd.Series, optional
        Cross-asset close series for rolling-correlation features. If
        either is None, the corresponding correlation column is NaN.
    term_spread, credit_spread : pd.Series, optional
        FRED macro series (T10Y2Y, BAA10Y). Forward-filled to the
        asset's trading calendar.

    Returns
    -------
    pd.DataFrame
        Same index/dropna contract as v1. Columns:
        ``NON_FEATURE_COLUMNS + FEATURE_COLUMNS_V2``.
    """
    base = compute_features_v1(close)
    # We rebuild v1 from scratch but DON'T dropna yet so the Tier-2
    # columns get the full index for alignment, then dropna at the end.
    log_ret = np.log(close).diff()

    # Recompute v1 columns on the full close index (no dropna yet):
    f = pd.DataFrame(index=close.index)
    f["close"] = close
    for h in (5, 20, 63, 252):
        f[f"mom_{h}"] = log_ret.rolling(h).sum().shift(1)
    vol_alphas = {
        "vol_short": 0.15, "vol_ewma": 0.06,
        "vol_long":  0.02, "vol_yearly": 0.005,
    }
    for name, alpha in vol_alphas.items():
        f[name] = log_ret.ewm(alpha=alpha, min_periods=20).std().shift(1)
    f["vol_ratio_sl"] = f["vol_short"] / f["vol_long"].replace(0, np.nan)
    f["vol_ratio_ly"] = f["vol_long"]  / f["vol_yearly"].replace(0, np.nan)
    abs_r = log_ret.abs()
    abs_mean = abs_r.ewm(alpha=0.02, min_periods=20).mean()
    abs_std  = abs_r.ewm(alpha=0.02, min_periods=20).std()
    f["shock_z"] = ((abs_r - abs_mean) / abs_std.replace(0, np.nan)).shift(1)
    rolling_max = close.rolling(252, min_periods=20).max()
    f["drawdown_252"] = (close / rolling_max - 1).shift(1)
    f["autocorr_63"] = (
        log_ret.rolling(63).corr(log_ret.shift(1)).shift(1)
    )
    ma_short = close.rolling(21).mean()
    ma_long  = close.rolling(63).mean()
    f["trend_dir"] = (ma_short / ma_long - 1).shift(1)

    # ---- Tier-2: Yang-Zhang volatility (from daily OHLC, more accurate than EWMA)
    if ohlc is not None:
        from src.features.intraday_rv import compute_yang_zhang_vol
        yz = compute_yang_zhang_vol(ohlc, window=20).reindex(close.index).ffill()
        f["yang_zhang_vol"] = yz.shift(1)
    else:
        f["yang_zhang_vol"] = np.nan

    # ---- Tier-2: VIX features
    if vix is not None:
        vix_aligned = vix.reindex(close.index).ffill()
        # Guard against non-positive VIX (synthetic test data) — real VIX
        # is always > 0 but we don't want runtime warnings to clutter
        # downstream log noise. Clip then log.
        f["vix_log"]    = np.log(vix_aligned.clip(lower=1e-9)).shift(1)
        f["vix_change"] = (vix_aligned - vix_aligned.shift(5)).shift(1)
    else:
        f["vix_log"]    = np.nan
        f["vix_change"] = np.nan

    if vix is not None and vix3m is not None:
        vix_aligned   = vix.reindex(close.index).ffill()
        vix3m_aligned = vix3m.reindex(close.index).ffill()
        # Backwardation (vix_term > 1) is the canonical stress signal.
        f["vix_term"] = (vix_aligned / vix3m_aligned.replace(0, np.nan)).shift(1)
        # Additive slope: positive = contango (calm), negative = backwardation (stress).
        f["vix_slope"] = (vix3m_aligned - vix_aligned).shift(1)
    else:
        f["vix_term"] = np.nan
        f["vix_slope"] = np.nan

    # ---- Tier-2: Cross-asset rolling correlations
    asset_ret = log_ret
    for name, cross in (("corr_tlt_63", tlt), ("corr_gld_63", gld)):
        if cross is None:
            f[name] = np.nan
            continue
        cross_ret = np.log(cross.reindex(close.index).ffill()).diff()
        f[name] = asset_ret.rolling(63).corr(cross_ret).shift(1)

    # ---- Tier-2: FRED macro spreads
    for name, series in (
        ("term_spread", term_spread),
        ("credit_spread", credit_spread),
    ):
        if series is None:
            f[name] = np.nan
            continue
        # FRED weekday calendar may not match yfinance — reindex + ffill.
        aligned = series.reindex(close.index).ffill()
        f[name] = aligned.shift(1)

    # Final dropna — same behaviour as v1: rows with ANY remaining NaN
    # in REQUIRED columns get dropped. Aux columns that are entirely
    # NaN (failed fetch) get filled with the column median (or 0 if
    # all-NaN) so the dropna doesn't wipe the whole frame.
    f = _fill_dead_aux_columns(f, FEATURE_COLUMNS_V2_ADD)
    f = f.dropna()
    return f


def _fill_dead_aux_columns(
    df: pd.DataFrame, aux_cols: list[str]
) -> pd.DataFrame:
    """Replace entirely-NaN aux columns with the column median (or 0).

    Lets the v2 frame survive a missing macro feed: the dead column
    becomes a constant (zero information) instead of nuking every row
    via dropna. XGBoost will still ignore a constant column.
    """
    for col in aux_cols:
        if col in df.columns and df[col].isna().all():
            df[col] = 0.0
    return df


__all__ = [
    "compute_features_v1",
    "compute_features_v2",
    "FEATURE_COLUMNS_V1",
    "FEATURE_COLUMNS_V2",
    "FEATURE_COLUMNS_V2_ADD",
    "NON_FEATURE_COLUMNS",
]