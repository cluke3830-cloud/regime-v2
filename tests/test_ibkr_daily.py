"""Integration tests for src.features.ibkr_daily — uses live IB Gateway.

These tests connect to the real IB Gateway (paper or live account).
They are skipped automatically when the gateway is unreachable so CI
on machines without IBKR still passes.

To run on EC2 / locally with gateway running:
    pytest tests/test_ibkr_daily.py -v -m ibkr

Environment variables (with defaults):
    IB_HOST          127.0.0.1
    IB_PORT          4004
    IBKR_TEST_CLIENT 31   (offset from the module default 30 to avoid conflicts)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.ibkr_daily import (  # noqa: E402
    IbkrDailyClient,
    _make_contract,
    _what_to_show,
    fetch_ibkr_aux_bundle,
    fetch_ibkr_daily,
)


# ---------------------------------------------------------------------------
# Connection fixture — skips entire module if IBKR unreachable
# ---------------------------------------------------------------------------

_IB_HOST = os.environ.get("IB_HOST", "127.0.0.1")
_IB_PORT = int(os.environ.get("IB_PORT", "4004"))
_CLIENT_ID = int(os.environ.get("IBKR_TEST_CLIENT", "31"))


def _ibkr_reachable() -> bool:
    """Return True if IB Gateway is up and accepting connections."""
    try:
        with IbkrDailyClient(
            host=_IB_HOST, port=_IB_PORT, client_id=_CLIENT_ID
        ) as client:
            # Fetch 3 bars of SPY as a smoke test
            client.fetch_daily_close("SPY", n_bars=3)
        return True
    except Exception:
        return False


ibkr = pytest.mark.skipif(
    not _ibkr_reachable(),
    reason="IB Gateway not reachable — skipping IBKR integration tests",
)


# ---------------------------------------------------------------------------
# Contract spec helpers (no IBKR needed — pure logic)
# ---------------------------------------------------------------------------


def test_vix_makes_ind_contract():
    c = _make_contract("^VIX")
    assert c.secType == "IND"
    assert c.symbol == "VIX"
    assert c.exchange == "CBOE"


def test_vix3m_makes_ind_contract():
    c = _make_contract("^VIX3M")
    assert c.secType == "IND"
    assert c.symbol == "VIX3M"


def test_spy_makes_stk_contract():
    c = _make_contract("SPY")
    assert c.secType == "STK"
    assert c.symbol == "SPY"
    assert c.exchange == "SMART"


def test_fx_ticker_stripped():
    c = _make_contract("JPY=X")
    assert c.symbol == "JPY"
    assert c.secType == "STK"


def test_what_to_show_ind():
    assert _what_to_show("^VIX") == "MIDPOINT"
    assert _what_to_show("^VIX3M") == "MIDPOINT"


def test_what_to_show_stk():
    assert _what_to_show("SPY") == "TRADES"
    assert _what_to_show("QQQ") == "TRADES"


# ---------------------------------------------------------------------------
# Live integration tests — require IBKR gateway
# ---------------------------------------------------------------------------


@ibkr
def test_live_fetch_daily_close_spy():
    """Fetch 20 daily close bars for SPY from live gateway."""
    with IbkrDailyClient(host=_IB_HOST, port=_IB_PORT, client_id=_CLIENT_ID) as c:
        s = c.fetch_daily_close("SPY", n_bars=20)
    assert isinstance(s, pd.Series)
    assert len(s) >= 10  # may be fewer on short history / holidays
    assert s.name == "SPY"
    assert pd.api.types.is_datetime64_any_dtype(s.index)
    assert (s > 0).all(), "SPY close prices must be positive"


@ibkr
def test_live_fetch_daily_close_vix():
    """Fetch VIX index bars (IND contract, MIDPOINT)."""
    with IbkrDailyClient(host=_IB_HOST, port=_IB_PORT, client_id=_CLIENT_ID) as c:
        s = c.fetch_daily_close("^VIX", n_bars=20)
    assert len(s) >= 5
    assert (s > 0).all()
    assert s.max() < 100  # VIX historically < 90


@ibkr
def test_live_fetch_daily_ohlcv_has_all_columns():
    with IbkrDailyClient(host=_IB_HOST, port=_IB_PORT, client_id=_CLIENT_ID) as c:
        df = c.fetch_daily_ohlcv("SPY", n_bars=10)
    for col in ("open", "high", "low", "close", "volume"):
        assert col in df.columns, f"Missing column: {col}"
    assert (df["volume"] > 0).all(), "SPY volume should be positive"
    assert (df["high"] >= df["low"]).all()


@ibkr
def test_live_fetch_ibkr_daily_standalone():
    """Standalone helper — opens and closes connection internally."""
    s = fetch_ibkr_daily(
        "QQQ", n_bars=15,
        host=_IB_HOST, port=_IB_PORT, client_id=_CLIENT_ID,
    )
    assert isinstance(s, pd.Series)
    assert len(s) >= 5
    assert (s > 0).all()


@ibkr
def test_live_aux_bundle_all_keys_present():
    """fetch_ibkr_aux_bundle returns all six expected keys."""
    bundle = fetch_ibkr_aux_bundle(
        n_bars=30,
        host=_IB_HOST,
        port=_IB_PORT,
        client_id=_CLIENT_ID,
        fred_api_key=os.environ.get("FRED_API_KEY"),
    )
    required_keys = {"vix", "vix3m", "tlt", "gld", "term_spread", "credit_spread"}
    assert required_keys.issubset(bundle.keys())


@ibkr
def test_live_aux_bundle_vix_is_series():
    bundle = fetch_ibkr_aux_bundle(
        n_bars=20,
        host=_IB_HOST,
        port=_IB_PORT,
        client_id=_CLIENT_ID,
    )
    vix = bundle["vix"]
    assert vix is not None, "VIX should succeed from live gateway"
    assert isinstance(vix, pd.Series)
    assert (vix > 0).all()


@ibkr
def test_live_close_prices_are_sorted():
    """Returned series must be in chronological order."""
    with IbkrDailyClient(host=_IB_HOST, port=_IB_PORT, client_id=_CLIENT_ID) as c:
        s = c.fetch_daily_close("GLD", n_bars=30)
    assert s.index.is_monotonic_increasing, "Index must be chronologically sorted"


@ibkr
def test_live_different_tickers_have_different_prices():
    """SPY and TLT should have meaningfully different price levels."""
    with IbkrDailyClient(host=_IB_HOST, port=_IB_PORT, client_id=_CLIENT_ID) as c:
        spy = c.fetch_daily_close("SPY", n_bars=5)
        tlt = c.fetch_daily_close("TLT", n_bars=5)
    # SPY is typically 400-600; TLT is typically 70-120
    assert abs(spy.mean() - tlt.mean()) > 50, "SPY and TLT should have different price levels"