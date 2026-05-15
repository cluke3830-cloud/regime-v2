#!/usr/bin/env python3
"""
Regime detection HTTP API server.  Runs on EC2 so Vercel can proxy to it.

Usage:
    python scripts/regime_api_server.py [--port 8051] [--backend yfinance|ibkr]

Endpoints:
    GET /health                 → {"status": "ok"}
    GET /regime/<TICKER>        → AssetPayload JSON (matches dashboard/lib/types.ts)

Caching:
    Results cached in memory for CACHE_TTL seconds (default 24 h).
    First request for any ticker takes ~5–15 s; subsequent requests return
    the cached payload instantly.

Security:
    Ticker symbols are validated against /^[A-Z0-9.\-^]{1,20}$/ before any
    computation is performed.

Dependencies:
    pip install flask          (not in requirements.txt — server-only dep)
    All other deps already in Regime_v2/requirements.txt
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from flask import Flask, Response, jsonify, request
except ImportError:
    print("[regime-api] flask not found. Install with: pip install flask", file=sys.stderr)
    sys.exit(1)

# Import the payload builder — safe because main() is guarded by __name__
from scripts.build_dashboard_data import (  # noqa: E402
    _build_asset_payload,
    ASSET_NAMES,
)
from src.features.aux_data import fetch_aux_data_bundle  # noqa: E402
from src.validation.multi_asset import load_close  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_TTL = int(os.environ.get("REGIME_CACHE_TTL", 86_400))  # default 24 h
LOOKBACK_YEARS = 5
IBKR_N_BARS = 1260  # ~5 years × 252 trading days

TICKER_RE = re.compile(r"^[A-Z0-9.\-^]{1,20}$")

app = Flask(__name__)

# In-memory caches — ticker → (payload_dict, unix_timestamp)
_payload_cache: dict[str, tuple[dict, float]] = {}
# Shared aux bundle cache — refreshed once per CACHE_TTL
_aux_cache: dict[str, tuple[Any, float]] = {}
_aux_lock = __import__("threading").Lock()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_aux_bundle(backend: str, fred_api_key: Optional[str] = None) -> Any:
    key = f"aux_{backend}"
    with _aux_lock:
        entry = _aux_cache.get(key)
        if entry and time.time() - entry[1] < CACHE_TTL:
            return entry[0]

    end = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")
    start = (
        datetime.now(timezone.utc) - timedelta(days=365 * LOOKBACK_YEARS + 60)
    ).strftime("%Y-%m-%d")

    if backend == "ibkr":
        from src.features.ibkr_daily import fetch_ibkr_aux_bundle  # noqa: PLC0415
        bundle = fetch_ibkr_aux_bundle(n_bars=IBKR_N_BARS, fred_api_key=fred_api_key)
    else:
        cache_dir = ROOT / "data" / "cache"
        bundle = fetch_aux_data_bundle(
            start, end, cache_dir=cache_dir, fred_api_key=fred_api_key
        )

    with _aux_lock:
        _aux_cache[key] = (bundle, time.time())
    return bundle


def _compute(ticker: str, backend: str) -> dict:
    fred_key = os.environ.get("FRED_API_KEY")
    cache_dir = ROOT / "data" / "cache"

    print(f"[regime-api] computing {ticker}  backend={backend}", flush=True)
    t0 = time.time()

    aux = _get_aux_bundle(backend, fred_key)

    if backend == "ibkr":
        from src.features.ibkr_daily import fetch_ibkr_daily  # noqa: PLC0415
        close = fetch_ibkr_daily(ticker, n_bars=IBKR_N_BARS)
    else:
        end = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")
        start = (
            datetime.now(timezone.utc) - timedelta(days=365 * LOOKBACK_YEARS + 60)
        ).strftime("%Y-%m-%d")
        close = load_close(ticker, start, end, cache_dir=cache_dir)

    payload = _build_asset_payload(ticker, close, aux)

    # Add company name for tickers not in the fixed universe
    if ticker not in ASSET_NAMES:
        try:
            import yfinance as yf  # noqa: PLC0415
            info = yf.Ticker(ticker).info
            payload["name"] = info.get("shortName") or info.get("longName") or ticker
        except Exception:
            payload["name"] = ticker

    elapsed = time.time() - t0
    print(
        f"[regime-api] {ticker} done in {elapsed:.1f}s  "
        f"regime={payload['current_regime']['name']}  "
        f"tvtp_pos={payload['current_tvtp']['position']:+.2f}",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/health")
def health():
    return jsonify({"status": "ok", "uptime": time.time()})


@app.route("/regime/<path:ticker>")
def get_regime(ticker: str):
    ticker = ticker.upper().strip()
    if not TICKER_RE.match(ticker):
        return jsonify({"error": "Invalid ticker symbol"}), 400

    entry = _payload_cache.get(ticker)
    if entry and time.time() - entry[1] < CACHE_TTL:
        print(f"[regime-api] cache hit: {ticker}", flush=True)
        return Response(
            json.dumps(entry[0], separators=(",", ":")),
            mimetype="application/json",
        )

    backend = app.config.get("BACKEND", "yfinance")
    try:
        payload = _compute(ticker, backend)
        _payload_cache[ticker] = (payload, time.time())
        return Response(
            json.dumps(payload, separators=(",", ":")),
            mimetype="application/json",
        )
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Regime API server for EC2")
    p.add_argument("--port", type=int, default=int(os.environ.get("REGIME_API_PORT", 8051)))
    p.add_argument(
        "--backend",
        choices=["yfinance", "ibkr"],
        default=os.environ.get("REGIME_BACKEND", "yfinance"),
        help="Data backend (default: yfinance; use 'ibkr' on EC2 with running IB Gateway)",
    )
    p.add_argument("--host", default="0.0.0.0")
    args = p.parse_args()

    app.config["BACKEND"] = args.backend
    print(
        f"[regime-api] starting  host={args.host}  port={args.port}  backend={args.backend}",
        flush=True,
    )
    # Pre-warm the aux bundle in the background so first ticker request is faster
    import threading
    threading.Thread(
        target=lambda: _get_aux_bundle(args.backend, os.environ.get("FRED_API_KEY")),
        daemon=True,
    ).start()

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()