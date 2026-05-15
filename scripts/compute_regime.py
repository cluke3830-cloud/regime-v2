#!/usr/bin/env python3
"""
On-demand regime computation for any US stock ticker.

Usage:
    python scripts/compute_regime.py AAPL
    python scripts/compute_regime.py NVDA --out-dir /path/to/regimes/

Fetches ~5 years of daily close data via yfinance, runs the full Regime_v2
pipeline (rule-baseline + GMM-HMM + TVTP-MSAR + log-opinion-pool fusion),
and writes a JSON payload to:
    <out-dir>/<safe-ticker>.json

The JSON schema matches AssetPayload in dashboard/lib/types.ts.
Prints the output path and a one-line summary on success.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse the payload builder from the existing build script (safe: main() is
# guarded by if __name__ == '__main__' so importing it is side-effect-free).
from scripts.build_dashboard_data import (  # noqa: E402
    _build_asset_payload,
    ASSET_NAMES,
)
from src.features.aux_data import fetch_aux_data_bundle  # noqa: E402
from src.validation.multi_asset import load_close  # noqa: E402

DEFAULT_OUT_DIR = ROOT / "dashboard" / "public" / "data" / "regimes"
LOOKBACK_YEARS = 5
IBKR_N_BARS = 1260  # ~5 years of daily bars (252 trading days × 5)


def safe_name(ticker: str) -> str:
    return ticker.replace("=", "_").replace("/", "_")


def compute(ticker: str, out_dir: Path, backend: str = "yfinance") -> dict:
    """Compute regime payload for *ticker*.

    Parameters
    ----------
    ticker : str
        Uppercase ticker symbol.
    out_dir : Path
        Directory where ``{safe}.json`` is written.
    backend : str
        ``"yfinance"`` (default) — cached yfinance data.
        ``"ibkr"``    — live daily bars from IB Gateway (port 4004 on EC2).
    """
    cache_dir = ROOT / "data" / "cache"

    if backend == "ibkr":
        from src.features.ibkr_daily import (  # noqa: PLC0415
            fetch_ibkr_aux_bundle,
            fetch_ibkr_daily,
        )
        print(f"[compute_regime] {ticker}  backend=ibkr  n_bars={IBKR_N_BARS}", flush=True)
        print("[compute_regime] fetching price data from IBKR...", flush=True)
        close = fetch_ibkr_daily(ticker, n_bars=IBKR_N_BARS)
        print("[compute_regime] fetching aux bundle from IBKR + FRED...", flush=True)
        aux = fetch_ibkr_aux_bundle(
            n_bars=IBKR_N_BARS,
            fred_api_key=os.environ.get("FRED_API_KEY"),
        )
    else:
        end = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")
        start = (datetime.now(timezone.utc) - timedelta(days=365 * LOOKBACK_YEARS + 60)).strftime(
            "%Y-%m-%d"
        )
        print(f"[compute_regime] {ticker}  backend=yfinance  {start} → {end}", flush=True)
        print("[compute_regime] fetching price data...", flush=True)
        close = load_close(ticker, start, end, cache_dir=cache_dir)
        print("[compute_regime] fetching aux bundle (VIX, FRED)...", flush=True)
        aux = fetch_aux_data_bundle(start, end, cache_dir=cache_dir)

    print("[compute_regime] running regime models...", flush=True)
    payload = _build_asset_payload(ticker, close, aux)

    # Override name if not in the known-universe dict (will fall back to ticker)
    if ticker not in ASSET_NAMES:
        # Try to get company name from yfinance metadata
        try:
            import yfinance as yf
            info = yf.Ticker(ticker).info
            name = info.get("shortName") or info.get("longName") or ticker
            payload["name"] = name
        except Exception:
            payload["name"] = ticker

    out_dir.mkdir(parents=True, exist_ok=True)
    safe = safe_name(ticker)
    out_file = out_dir / f"{safe}.json"
    with open(out_file, "w") as fh:
        json.dump(payload, fh, separators=(",", ":"))

    kb = out_file.stat().st_size / 1024
    print(
        f"[compute_regime] wrote {out_file.relative_to(ROOT)}  "
        f"({kb:.1f} KB)  "
        f"regime={payload['current_regime']['name']}  "
        f"tvtp_pos={payload['current_tvtp']['position']:+.2f}",
        flush=True,
    )
    return payload


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="On-demand regime computation for any US stock ticker")
    p.add_argument("ticker", type=str.upper, help="Ticker symbol (e.g. AAPL, NVDA, MSFT)")
    p.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help=f"Output directory for JSON (default: {DEFAULT_OUT_DIR})",
    )
    p.add_argument(
        "--backend",
        choices=["yfinance", "ibkr"],
        default="yfinance",
        help="Data backend: 'yfinance' (default, cached) or 'ibkr' (live IB Gateway on EC2)",
    )
    args = p.parse_args()

    try:
        compute(args.ticker, Path(args.out_dir), backend=args.backend)
    except Exception as exc:
        print(f"[compute_regime] ERROR: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())