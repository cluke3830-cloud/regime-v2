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
from src.alerts.dispatcher import (  # noqa: E402
    add_subscriber,
    count_active_subscribers,
    remove_subscriber,
)
from src.regime.consensus import compute_market_consensus  # noqa: E402
from src.validation.multi_asset import DEFAULT_UNIVERSE, load_close  # noqa: E402

SUBSCRIBER_STORE = ROOT / "data" / "subscribers.json"

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


def _get_or_compute_payload(ticker: str) -> dict:
    """Cache-aware payload accessor used by both /regime/<T> and
    /regime/<T>/history endpoints. Raises on compute failure."""
    entry = _payload_cache.get(ticker)
    if entry and time.time() - entry[1] < CACHE_TTL:
        return entry[0]
    backend = app.config.get("BACKEND", "yfinance")
    payload = _compute(ticker, backend)
    _payload_cache[ticker] = (payload, time.time())
    return payload


@app.route("/regime/<path:ticker>")
def get_regime(ticker: str):
    ticker = ticker.upper().strip()
    if not TICKER_RE.match(ticker):
        return jsonify({"error": "Invalid ticker symbol"}), 400

    if _payload_cache.get(ticker):
        print(f"[regime-api] cache hit: {ticker}", flush=True)

    try:
        payload = _get_or_compute_payload(ticker)
        return Response(
            json.dumps(payload, separators=(",", ":")),
            mimetype="application/json",
        )
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


# Phase 2 — query-able history slice without forcing a full-payload re-build.
# History is already bundled in the asset payload (HISTORY_BARS=504 = ~2y);
# this endpoint just slices the tail and returns it as a thinner response.
HISTORY_DAYS_MAX = 504  # matches HISTORY_BARS in build_dashboard_data.py
HISTORY_DAYS_DEFAULT = 252  # ~1 year


@app.route("/regime/<path:ticker>/history")
def get_regime_history(ticker: str):
    ticker = ticker.upper().strip()
    if not TICKER_RE.match(ticker):
        return jsonify({"error": "Invalid ticker symbol"}), 400

    try:
        days = int(request.args.get("days", HISTORY_DAYS_DEFAULT))
    except (TypeError, ValueError):
        return jsonify({"error": "'days' must be an integer"}), 400
    if days < 1:
        return jsonify({"error": "'days' must be >= 1"}), 400
    days = min(days, HISTORY_DAYS_MAX)

    try:
        payload = _get_or_compute_payload(ticker)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500

    history = payload.get("history", [])
    sliced = history[-days:] if days < len(history) else history

    response = {
        "ticker":   payload.get("ticker", ticker),
        "name":     payload.get("name", ticker),
        "as_of":    payload.get("as_of"),
        "days_requested": days,
        "days_returned":  len(sliced),
        "history":  sliced,
    }
    return Response(
        json.dumps(response, separators=(",", ":")),
        mimetype="application/json",
    )


# Phase 4 — cross-asset market consensus. Aggregates per-asset fusion
# labels across DEFAULT_UNIVERSE (13 tickers after Phase 7 Forex expansion).
# Each call uses the cached
# per-asset payloads when warm; cold starts trigger 10 ticker computes
# which can take ~30-60s before the cache warms up.


@app.route("/regime/market")
def get_market_consensus():
    universe = DEFAULT_UNIVERSE
    asset_payloads = []
    errors = []
    for ticker in universe:
        try:
            asset_payloads.append(_get_or_compute_payload(ticker))
        except Exception as exc:  # noqa: BLE001
            errors.append({"ticker": ticker, "error": str(exc)})
            print(f"[regime-api] market: {ticker} compute failed: {exc}", flush=True)

    if not asset_payloads:
        return jsonify({
            "error": "no assets could be loaded for the market consensus",
            "failed": errors,
        }), 500

    consensus = compute_market_consensus(asset_payloads)
    consensus["errors"] = errors
    consensus["universe"] = list(universe)
    return Response(
        json.dumps(consensus, separators=(",", ":")),
        mimetype="application/json",
    )


# Phase 6 — Subscription management.
# Subscribers stored in data/subscribers.json (gitignored, contains PII).
# POST /subscribe  — add email and/or webhook
# DELETE /subscribe — soft-delete by email
# GET /subscribers/count — public count (no PII)

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@app.route("/subscribe", methods=["POST"])
def post_subscribe():
    body = request.get_json(silent=True) or {}
    email = (body.get("email") or "").strip() or None
    webhook_url = (body.get("webhook_url") or "").strip() or None
    tickers = body.get("tickers") or ["*"]

    if not email and not webhook_url:
        return jsonify({"error": "provide at least one of email or webhook_url"}), 400
    if email and not EMAIL_RE.match(email):
        return jsonify({"error": "invalid email address"}), 400
    if not isinstance(tickers, list):
        return jsonify({"error": "tickers must be a list"}), 400

    try:
        sub = add_subscriber(
            email=email,
            webhook_url=webhook_url,
            tickers=tickers,
            notify_consensus=bool(body.get("notify_consensus", True)),
            path=SUBSCRIBER_STORE,
        )
        return jsonify({"status": "subscribed", "email": sub.get("email"),
                        "webhook_url": sub.get("webhook_url")}), 201
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/subscribe", methods=["DELETE"])
def delete_subscribe():
    body = request.get_json(silent=True) or {}
    email = (body.get("email") or "").strip()
    if not email:
        return jsonify({"error": "email required"}), 400
    found = remove_subscriber(email, path=SUBSCRIBER_STORE)
    if found:
        return jsonify({"status": "unsubscribed", "email": email})
    return jsonify({"error": "email not found"}), 404


@app.route("/subscribers/count")
def get_subscriber_count():
    n = count_active_subscribers(SUBSCRIBER_STORE)
    return jsonify({"count": n})


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