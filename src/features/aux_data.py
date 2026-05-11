"""Auxiliary data fetchers — VIX, cross-asset closes, FRED macro series.

Brief 2.1.2 of the regime upgrade plan. Powers the cross-asset and macro
feature columns added in :func:`compute_features_v2`.

Design contract:
    Every fetcher returns a ``pd.Series`` (or ``pd.DataFrame``) indexed
    by date, OR raises ``RuntimeError`` on hard failure (no data /
    network down / missing API key for required-key endpoints). The
    high-level :func:`compute_features_v2` catches these and substitutes
    NaN columns so a missing macro endpoint never crashes the harness.

Caching:
    All fetchers accept an optional ``cache_dir`` and Parquet-cache the
    raw response. The cache key is ``{name}_{start}_{end}.parquet``.
    Re-runs on the same date range are byte-stable.

FRED API:
    Direct HTTP to ``https://api.stlouisfed.org/fred/series/observations``.
    Series codes used in this module:
      - ``T10Y2Y``  : 10-year Treasury minus 2-year Treasury (term spread)
      - ``BAA10Y``  : Moody's BAA corporate bond yield - 10y Treasury
                      (credit spread proxy)
    FRED API key passed via ``FRED_API_KEY`` env var or ``api_key`` kw.
    Free tier: 120 requests/min — far more than we need.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests


FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"


# ---------------------------------------------------------------------------
# yfinance loaders for VIX and cross-asset closes
# ---------------------------------------------------------------------------


def _safe_ticker(ticker: str) -> str:
    """Replace characters yfinance puts in tickers but Parquet doesn't like."""
    return ticker.replace("=", "_").replace("^", "_").replace("/", "_")


def fetch_yf_close(
    ticker: str,
    start: str,
    end: str,
    *,
    cache_dir: Optional[Path] = None,
) -> pd.Series:
    """Pull a close-to-close series from yfinance with Parquet cache.

    Identical pattern to :func:`src.validation.multi_asset.load_close` —
    duplicated here to avoid a circular import in the feature module.
    """
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cpath = cache_dir / f"{_safe_ticker(ticker)}_{start}_{end}.parquet"
        if cpath.exists():
            return pd.read_parquet(cpath).iloc[:, 0]

    import yfinance as yf
    raw = yf.download(
        ticker, start=start, end=end, progress=False, auto_adjust=True
    )
    if raw is None or raw.empty:
        raise RuntimeError(
            f"yfinance returned no data for {ticker} on [{start}, {end}]"
        )
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
    raw.columns = [str(c).lower() for c in raw.columns]
    if "close" not in raw.columns:
        raise RuntimeError(
            f"yfinance frame for {ticker} has no 'close' column"
        )
    close = raw["close"].astype(float).dropna()
    close.name = ticker

    if cache_dir is not None:
        cpath = cache_dir / f"{_safe_ticker(ticker)}_{start}_{end}.parquet"
        close.to_frame().to_parquet(cpath)
    return close


def fetch_vix(start: str, end: str, *, cache_dir: Optional[Path] = None) -> pd.Series:
    """CBOE VIX index close (yfinance ticker ``^VIX``)."""
    return fetch_yf_close("^VIX", start, end, cache_dir=cache_dir)


def fetch_vix3m(start: str, end: str, *, cache_dir: Optional[Path] = None) -> pd.Series:
    """CBOE 3-month VIX (yfinance ticker ``^VIX3M``)."""
    return fetch_yf_close("^VIX3M", start, end, cache_dir=cache_dir)


# ---------------------------------------------------------------------------
# FRED macro fetcher
# ---------------------------------------------------------------------------


def fetch_fred_series(
    series_id: str,
    start: str,
    end: str,
    *,
    api_key: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    timeout: float = 10.0,
) -> pd.Series:
    """Pull a daily FRED series via the official observations endpoint.

    Parameters
    ----------
    series_id : str
        FRED series code (e.g., ``T10Y2Y``, ``BAA10Y``, ``ICSA``).
    start, end : str
        ``YYYY-MM-DD`` date range (inclusive of start, exclusive of end —
        same convention as yfinance).
    api_key : str, optional
        FRED API key. Falls back to ``FRED_API_KEY`` env var. ``RuntimeError``
        if neither is set.
    cache_dir :
        Parquet cache dir.
    timeout :
        HTTP timeout in seconds.

    Returns
    -------
    pd.Series
        DatetimeIndex (date), float values. FRED encodes missing values as
        ``"."`` — these are coerced to NaN.

    Raises
    ------
    RuntimeError
        On missing API key, HTTP failure, malformed response, or empty
        result.
    """
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cpath = cache_dir / f"fred_{series_id}_{start}_{end}.parquet"
        if cpath.exists():
            return pd.read_parquet(cpath).iloc[:, 0]

    key = api_key or os.environ.get("FRED_API_KEY")
    if not key:
        raise RuntimeError(
            f"FRED API key required for {series_id} — pass api_key or set "
            "FRED_API_KEY env var"
        )

    params = {
        "series_id": series_id,
        "api_key": key,
        "file_type": "json",
        "observation_start": start,
        "observation_end": end,
    }
    try:
        resp = requests.get(FRED_BASE_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
    except (requests.RequestException, ValueError) as exc:
        raise RuntimeError(f"FRED fetch failed for {series_id}: {exc}") from exc

    obs = payload.get("observations") or []
    if not obs:
        raise RuntimeError(f"FRED returned no observations for {series_id}")

    dates = pd.to_datetime([o["date"] for o in obs])
    values = [
        float(o["value"]) if o["value"] not in (".", "", None) else np.nan
        for o in obs
    ]
    series = pd.Series(values, index=dates, name=series_id).dropna()

    if cache_dir is not None:
        cpath = cache_dir / f"fred_{series_id}_{start}_{end}.parquet"
        series.to_frame().to_parquet(cpath)
    return series


# ---------------------------------------------------------------------------
# Convenience: fetch all aux data for a single date range
# ---------------------------------------------------------------------------


def fetch_aux_data_bundle(
    start: str,
    end: str,
    *,
    cache_dir: Optional[Path] = None,
    fred_api_key: Optional[str] = None,
) -> dict:
    """One call → dict of every aux series Brief 2.1.2 consumes.

    Returns
    -------
    dict
        Keys (any of which may be missing if a fetch failed):
          - ``vix``           : pd.Series (^VIX close)
          - ``vix3m``         : pd.Series (^VIX3M close)
          - ``tlt``           : pd.Series (TLT close)
          - ``gld``           : pd.Series (GLD close)
          - ``term_spread``   : pd.Series (T10Y2Y from FRED)
          - ``credit_spread`` : pd.Series (BAA10Y from FRED)

        Failed fetches are stored as ``None`` so the caller can detect
        them. None of the failures here are fatal — the v2 feature
        builder will fill the missing columns with NaN, and XGBoost
        handles NaN natively.
    """
    bundle: dict = {}
    for name, fn in [
        ("vix",   lambda: fetch_vix(start, end, cache_dir=cache_dir)),
        ("vix3m", lambda: fetch_vix3m(start, end, cache_dir=cache_dir)),
        ("tlt",   lambda: fetch_yf_close("TLT", start, end, cache_dir=cache_dir)),
        ("gld",   lambda: fetch_yf_close("GLD", start, end, cache_dir=cache_dir)),
    ]:
        try:
            bundle[name] = fn()
        except Exception as exc:  # noqa: BLE001 — intentionally broad
            bundle[name] = None
            bundle[f"_{name}_error"] = str(exc)

    for name, series_id in [
        ("term_spread",   "T10Y2Y"),
        ("credit_spread", "BAA10Y"),
    ]:
        try:
            bundle[name] = fetch_fred_series(
                series_id, start, end,
                api_key=fred_api_key, cache_dir=cache_dir,
            )
        except Exception as exc:  # noqa: BLE001
            bundle[name] = None
            bundle[f"_{name}_error"] = str(exc)

    return bundle


__all__ = [
    "fetch_yf_close",
    "fetch_vix",
    "fetch_vix3m",
    "fetch_fred_series",
    "fetch_aux_data_bundle",
]