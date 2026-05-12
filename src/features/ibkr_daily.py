"""IBKR daily bar fetcher — production live-data source.

Provides the same interface as ``aux_data.fetch_aux_data_bundle`` but sources
data from IB Gateway via ib_insync instead of yfinance. Intended for the
daily live regime update (``scripts/update_live_regime.py``); backtests
continue to use the yfinance / Parquet cache path.

Contract mapping
----------------
  ``^VIX``    → IND / VIX  / CBOE / USD  (whatToShow='MIDPOINT')
  ``^VIX3M``  → IND / VIX3M/ CBOE / USD  (whatToShow='MIDPOINT')
  otherwise   → STK / symbol / SMART / USD (whatToShow='TRADES')

IBKR client IDs in use
-----------------------
  21  live_trader
  22  bar_fetcher (1-min)
  30  ibkr_daily (this module)      ← new
  98  ibgw_healthcheck

FRED data (T10Y2Y, BAA10Y) is NOT available via IBKR and is fetched from the
same FRED HTTP endpoint as ``aux_data.fetch_fred_series``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Contract spec helpers
# ---------------------------------------------------------------------------

_IND_TICKERS = {"^VIX", "^VIX3M"}
_TICKER_TO_IBKR_SYMBOL = {"^VIX": "VIX", "^VIX3M": "VIX3M"}


def _make_contract(ticker: str):
    """Return an ib_insync Contract for *ticker*."""
    from ib_insync import Contract

    if ticker in _IND_TICKERS:
        return Contract(
            symbol=_TICKER_TO_IBKR_SYMBOL[ticker],
            secType="IND",
            exchange="CBOE",
            currency="USD",
        )
    # Strip exchange suffix for FX pairs like JPY=X
    clean = ticker.replace("=X", "")
    return Contract(
        symbol=clean,
        secType="STK",
        exchange="SMART",
        currency="USD",
    )


def _what_to_show(ticker: str) -> str:
    return "MIDPOINT" if ticker in _IND_TICKERS else "TRADES"


# ---------------------------------------------------------------------------
# Context manager client
# ---------------------------------------------------------------------------


class IbkrDailyClient:
    """Context manager that connects / disconnects from IB Gateway."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 4004,
        client_id: int = 30,
        timeout: float = 20.0,
    ) -> None:
        self.host = host
        self.port = port
        self.client_id = client_id
        self.timeout = timeout
        self._ib = None

    def __enter__(self) -> "IbkrDailyClient":
        from ib_insync import IB, util

        util.patchAsyncio()
        ib = IB()
        ib.connect(
            self.host,
            self.port,
            clientId=self.client_id,
            readonly=True,
            timeout=self.timeout,
        )
        self._ib = ib
        return self

    def __exit__(self, *_) -> None:
        if self._ib is not None:
            try:
                self._ib.disconnect()
            except Exception:
                pass
            self._ib = None

    def fetch_daily_close(self, ticker: str, n_bars: int = 300) -> pd.Series:
        """Fetch *n_bars* of daily close prices for *ticker*.

        Returns a pd.Series with a DatetimeIndex (business days), name=ticker.
        Raises RuntimeError if IBKR returns no bars.
        """
        contract = _make_contract(ticker)
        self._ib.qualifyContracts(contract)

        bars = self._ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=f"{n_bars} D",
            barSizeSetting="1 day",
            whatToShow=_what_to_show(ticker),
            useRTH=True,
            formatDate=1,
        )
        if not bars:
            raise RuntimeError(f"[ibkr_daily] no bars returned for {ticker!r}")

        close = pd.Series(
            {pd.Timestamp(b.date): b.close for b in bars},
            name=ticker,
            dtype=float,
        )
        close.index = pd.DatetimeIndex(close.index)
        return close.sort_index()

    def fetch_daily_ohlcv(self, ticker: str, n_bars: int = 300) -> pd.DataFrame:
        """Fetch *n_bars* of daily OHLCV for *ticker*.

        Returns a DataFrame with columns open/high/low/close/volume and a
        DatetimeIndex. Used by ``scripts/update_live_regime.py`` to obtain
        volume for the Amihud cost model.
        """
        contract = _make_contract(ticker)
        self._ib.qualifyContracts(contract)

        bars = self._ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=f"{n_bars} D",
            barSizeSetting="1 day",
            whatToShow=_what_to_show(ticker),
            useRTH=True,
            formatDate=1,
        )
        if not bars:
            raise RuntimeError(f"[ibkr_daily] no bars returned for {ticker!r}")

        rows = [
            {
                "datetime": pd.Timestamp(b.date),
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume,
            }
            for b in bars
        ]
        df = pd.DataFrame(rows).set_index("datetime")
        df.index = pd.DatetimeIndex(df.index)
        return df.sort_index()


# ---------------------------------------------------------------------------
# Standalone fetch helpers (manage their own connection)
# ---------------------------------------------------------------------------


def fetch_ibkr_daily(
    ticker: str,
    n_bars: int = 300,
    *,
    host: str | None = None,
    port: int | None = None,
    client_id: int = 30,
) -> pd.Series:
    """Fetch *n_bars* of daily close prices from IB Gateway.

    Parameters
    ----------
    ticker : str
        Yahoo Finance–style ticker (``^VIX``, ``SPY``, ``BTC-USD``, …).
        ``^VIX`` / ``^VIX3M`` are mapped to IBKR IND contracts; everything
        else is treated as a STK on SMART.
    n_bars : int
        Trading-day bar count (default 300 ≈ 14 months).
    host, port : str / int
        IB Gateway address. Default to ``$IB_HOST`` / ``$IB_PORT`` env vars,
        then ``127.0.0.1:4004``.
    client_id : int
        IBKR client ID. Default 30 (reserved for this module).

    Returns
    -------
    pd.Series
        DatetimeIndex, dtype float, name=ticker.

    Raises
    ------
    RuntimeError
        If IBKR is unreachable or returns no bars.
    """
    host = host or os.environ.get("IB_HOST", "127.0.0.1")
    port = port or int(os.environ.get("IB_PORT", "4004"))

    with IbkrDailyClient(host=host, port=port, client_id=client_id) as client:
        return client.fetch_daily_close(ticker, n_bars=n_bars)


def fetch_ibkr_aux_bundle(
    n_bars: int = 300,
    *,
    host: str | None = None,
    port: int | None = None,
    client_id: int = 30,
    fred_api_key: str | None = None,
    cache_dir: Path | None = None,
) -> dict:
    """Fetch the same auxiliary bundle as ``aux_data.fetch_aux_data_bundle``.

    Price series (VIX, VIX3M, TLT, GLD) come from IB Gateway; FRED macro
    series (T10Y2Y, BAA10Y) are fetched via the existing FRED HTTP endpoint
    since IBKR has no macro data. Any series that fails is returned as None
    (matching ``fetch_aux_data_bundle`` graceful-degradation contract).

    Returns
    -------
    dict with keys: ``vix``, ``vix3m``, ``tlt``, ``gld``,
    ``term_spread``, ``credit_spread``.
    Each value is a pd.Series or None.
    """
    from src.features.aux_data import fetch_fred_series

    host = host or os.environ.get("IB_HOST", "127.0.0.1")
    port = port or int(os.environ.get("IB_PORT", "4004"))
    api_key = fred_api_key or os.environ.get("FRED_API_KEY")

    bundle: dict = {
        "vix": None, "vix3m": None, "tlt": None, "gld": None,
        "term_spread": None, "credit_spread": None,
    }

    # --- IBKR price series ---
    price_map = {"^VIX": "vix", "^VIX3M": "vix3m", "TLT": "tlt", "GLD": "gld"}
    with IbkrDailyClient(host=host, port=port, client_id=client_id) as client:
        for ticker, key in price_map.items():
            try:
                bundle[key] = client.fetch_daily_close(ticker, n_bars=n_bars)
            except Exception as exc:
                bundle[f"_{key}_error"] = str(exc)

    # --- FRED macro series (reuse aux_data fetcher, 1-year lookback) ---
    import datetime as _dt
    end = _dt.date.today().isoformat()
    start = (_dt.date.today() - _dt.timedelta(days=n_bars + 30)).isoformat()

    fred_map = {"T10Y2Y": "term_spread", "BAA10Y": "credit_spread"}
    for series_id, key in fred_map.items():
        try:
            bundle[key] = fetch_fred_series(
                series_id, start, end,
                api_key=api_key,
                cache_dir=cache_dir,
            )
        except Exception as exc:
            bundle[f"_{key}_error"] = str(exc)

    return bundle
