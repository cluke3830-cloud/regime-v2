"""Snapshot generator for the Regime_v2 web dashboard.

Reads cached parquet for the 10-asset DEFAULT_UNIVERSE, runs the
rule-baseline 5-regime classifier and TVTP-MSAR champion on each asset,
and emits per-asset JSON snapshots that the Vercel-hosted Next.js
front-end ships at build time.

Output layout (all paths relative to repo root):
    dashboard/public/data/assets.json          # universe metadata
    dashboard/public/data/regimes/{ticker}.json  # one per asset
    dashboard/public/data/summary.json         # all-asset latest-bar grid

Why static JSON instead of a live API?
    TVTP-MSAR fits statsmodels MarkovAutoregression on every refresh —
    ~3-5s per asset, plus yfinance/FRED IO. Serverless functions on
    Vercel cap at 10s and don't carry statsmodels. The dashboard is
    "live as of the last build"; a daily GitHub Actions cron (or manual
    `make dashboard-data`) rebuilds the JSON and triggers a redeploy.

Acceptance:
    `python scripts/build_dashboard_data.py` produces 12 JSON files
    (10 assets + assets.json + summary.json) in dashboard/public/data/,
    each file < 500KB, no NaN/Infinity, valid JSON.
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines.tvtp_msar import MarkovSwitchingAR  # noqa: E402
from src.features.aux_data import fetch_aux_data_bundle  # noqa: E402
from src.features.price_features import compute_features_v2  # noqa: E402
from src.regime.rule_baseline import (  # noqa: E402
    REGIME_ALLOC,
    REGIME_NAMES,
    compute_rule_regime_sequence,
)
from src.validation.multi_asset import DEFAULT_UNIVERSE, load_close  # noqa: E402


# Pretty names for the asset cards
ASSET_NAMES: Dict[str, str] = {
    "SPY":     "S&P 500",
    "QQQ":     "Nasdaq 100",
    "DIA":     "Dow Jones",
    "IWM":     "Russell 2000",
    "EFA":     "Developed ex-US",
    "EEM":     "Emerging Markets",
    "GLD":     "Gold",
    "TLT":     "20Y Treasuries",
    "BTC-USD": "Bitcoin",
    "JPY=X":   "USD / JPY",
}

# 5-regime taxonomy color palette (Bloomberg dark)
REGIME_COLORS = {
    0: "#22c55e",  # Full Bull — green
    1: "#84cc16",  # Half Bull — lime
    2: "#a3a3a3",  # Chop      — grey
    3: "#f97316",  # Half Bear — orange
    4: "#ef4444",  # Full Bear — red
}

# Validation report metrics (frozen — they don't move bar-to-bar).
# Pulled from validation_report.md.
ASSET_BACKTEST_STATS: Dict[str, Dict[str, float]] = {
    "SPY":     {"sharpe_p05": 1.487, "sharpe_p50": 2.462, "sharpe_p95": 3.158, "max_dd_p50": -0.036, "dsr_sharpe": 2.366},
    "QQQ":     {"sharpe_p05": 1.749, "sharpe_p50": 2.391, "sharpe_p95": 3.175, "max_dd_p50": -0.053, "dsr_sharpe": 2.391},
    "DIA":     {"sharpe_p05": 1.180, "sharpe_p50": 2.086, "sharpe_p95": 3.038, "max_dd_p50": -0.048, "dsr_sharpe": 2.086},
    "IWM":     {"sharpe_p05": 0.021, "sharpe_p50": 0.949, "sharpe_p95": 1.989, "max_dd_p50": -0.109, "dsr_sharpe": 0.949},
    "EFA":     {"sharpe_p05": 0.021, "sharpe_p50": 0.836, "sharpe_p95": 1.808, "max_dd_p50": -0.091, "dsr_sharpe": 0.836},
    "EEM":     {"sharpe_p05": -0.405, "sharpe_p50": 0.908, "sharpe_p95": 1.902, "max_dd_p50": -0.113, "dsr_sharpe": 0.908},
    "GLD":     {"sharpe_p05": -0.220, "sharpe_p50": 1.042, "sharpe_p95": 2.620, "max_dd_p50": -0.091, "dsr_sharpe": 1.042},
    "TLT":     {"sharpe_p05": -1.510, "sharpe_p50": 0.240, "sharpe_p95": 1.313, "max_dd_p50": -0.114, "dsr_sharpe": 0.000},
    "BTC-USD": {"sharpe_p05": 0.160, "sharpe_p50": 1.149, "sharpe_p95": 2.599, "max_dd_p50": -0.209, "dsr_sharpe": 1.149},
    "JPY=X":   {"sharpe_p05": 0.049, "sharpe_p50": 0.937, "sharpe_p95": 1.529, "max_dd_p50": -0.029, "dsr_sharpe": 0.937},
}


HISTORY_BARS = 504  # ~2 years of trading days kept in the per-asset JSON.


def _finite(x: Any) -> Optional[float]:
    """Coerce numpy/float to plain Python float, mapping NaN/Inf → None.
    JSON spec doesn't allow Infinity/NaN; the front-end's chart libs
    misbehave on them too."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _compute_tvtp_probs(
    close: pd.Series, train_frac: float = 0.7
) -> Optional[pd.DataFrame]:
    """Fit MS-AR on early train portion, filter on full series.

    Returns DataFrame indexed by `close.index[1:]` with columns
    {p_low_vol, p_high_vol, position}.

    Returns None if the fit fails.
    """
    returns = np.log(close).diff().dropna()
    if len(returns) < 200:
        return None
    cut = int(len(returns) * train_frac)
    train_ret = returns.iloc[:cut]

    model = MarkovSwitchingAR(k_regimes=2, order=1, switching_variance=True)
    model.fit(train_ret)
    if model.params_ is None:
        return None
    try:
        probs = model.predict_proba(returns)
    except Exception:
        return None
    # state 0 = low-vol bull (+1.00), state 1 = high-vol stress (-0.30)
    pos = probs["p_low_vol"] * 1.00 + probs["p_high_vol"] * (-0.30)
    probs = probs.copy()
    probs["position"] = pos
    return probs


def _build_asset_payload(
    ticker: str, close: pd.Series, aux_bundle: dict
) -> Dict[str, Any]:
    """Build the per-asset JSON-ready dict."""

    # Features (causally-clean per compute_features_v2 docstring)
    features = compute_features_v2(
        close,
        vix=aux_bundle.get("vix"),
        vix3m=aux_bundle.get("vix3m"),
        tlt=aux_bundle.get("tlt"),
        gld=aux_bundle.get("gld"),
        term_spread=aux_bundle.get("term_spread"),
        credit_spread=aux_bundle.get("credit_spread"),
    )

    # Rule-baseline regime sequence
    rule_seq = compute_rule_regime_sequence(features)

    # TVTP-MSAR (the champion)
    tvtp = _compute_tvtp_probs(close)

    # Align everything to the rule_seq index
    aligned = rule_seq.copy()
    aligned["close"] = close.reindex(aligned.index)
    if tvtp is not None:
        aligned["tvtp_low_vol"] = tvtp["p_low_vol"].reindex(aligned.index)
        aligned["tvtp_high_vol"] = tvtp["p_high_vol"].reindex(aligned.index)
        aligned["tvtp_position"] = tvtp["position"].reindex(aligned.index)
    else:
        aligned["tvtp_low_vol"] = np.nan
        aligned["tvtp_high_vol"] = np.nan
        aligned["tvtp_position"] = 0.0

    # ------------------------------------------------------------------
    # Walk-forward equity curves (each at unit notional, t-1 signal → t return)
    # ------------------------------------------------------------------
    log_ret = np.log(aligned["close"]).diff().fillna(0.0)
    simple_ret = aligned["close"].pct_change().fillna(0.0)

    # 1) TVTP-MSAR champion (log-space cumsum, position in [-0.3, +1.0])
    tvtp_pos_lag = aligned["tvtp_position"].shift(1).fillna(0.0)
    tvtp_returns = tvtp_pos_lag * log_ret
    tvtp_equity = np.exp(tvtp_returns.cumsum())

    # 2) Buy-and-hold (log-space cumsum)
    bh_equity = np.exp(log_ret.cumsum())

    # 3) 3x-Bull Bet: only hold a 3x-leveraged-equity-ETF proxy when the
    #    rule classifier's P(Full Bull) + P(Half Bull) > 50%. Otherwise
    #    sit in cash (return = 0). Daily compounding of 3 × simple return
    #    captures vol drag the way real leveraged ETFs (UPRO/TQQQ/...) do.
    #    Clip per-day return at -99% to keep equity > 0 in a black-swan
    #    bar — real LETFs reset daily but can be wiped out the same way.
    bull3x_signal = aligned["label"].isin([0, 1]).astype(float)
    bull3x_signal_lag = bull3x_signal.shift(1).fillna(0.0)
    bull3x_daily = (bull3x_signal_lag * 3.0 * simple_ret).clip(lower=-0.99)
    bull3x_equity = (1.0 + bull3x_daily).cumprod()

    # Truncate to last HISTORY_BARS bars for the chart
    tail = aligned.tail(HISTORY_BARS)
    tail_tvtp_eq = tvtp_equity.tail(HISTORY_BARS)
    tail_bh_eq = bh_equity.tail(HISTORY_BARS)
    tail_bull3x_eq = bull3x_equity.tail(HISTORY_BARS)
    tail_bull3x_signal = bull3x_signal.tail(HISTORY_BARS)
    # Re-base equity curves to 1.0 at the start of the window
    if len(tail_tvtp_eq) > 0:
        tail_tvtp_eq = tail_tvtp_eq / tail_tvtp_eq.iloc[0]
        tail_bh_eq = tail_bh_eq / tail_bh_eq.iloc[0]
        tail_bull3x_eq = tail_bull3x_eq / tail_bull3x_eq.iloc[0]

    history = []
    for ts, row in tail.iterrows():
        date_str = ts.strftime("%Y-%m-%d")
        history.append({
            "date":  date_str,
            "close": _finite(row["close"]),
            "label": int(row["label"]),
            "regime": REGIME_NAMES[int(row["label"])],
            "alloc": _finite(row["position"]),
            "p0": _finite(row["p_0"]),
            "p1": _finite(row["p_1"]),
            "p2": _finite(row["p_2"]),
            "p3": _finite(row["p_3"]),
            "p4": _finite(row["p_4"]),
            "tvtp_low":  _finite(row["tvtp_low_vol"]),
            "tvtp_high": _finite(row["tvtp_high_vol"]),
            "tvtp_pos":  _finite(row["tvtp_position"]),
            "eq_tvtp":     _finite(tail_tvtp_eq.get(ts, np.nan)),
            "eq_bull3x":   _finite(tail_bull3x_eq.get(ts, np.nan)),
            "eq_bh":       _finite(tail_bh_eq.get(ts, np.nan)),
            "bull3x_in":   int(tail_bull3x_signal.get(ts, 0.0)),
        })

    # Latest-bar snapshot for the asset card
    last = aligned.iloc[-1]
    last_label = int(last["label"])
    last_probs = [
        _finite(last["p_0"]), _finite(last["p_1"]), _finite(last["p_2"]),
        _finite(last["p_3"]), _finite(last["p_4"]),
    ]
    last_tvtp_pos = _finite(last.get("tvtp_position", 0.0)) or 0.0
    last_tvtp_low = _finite(last.get("tvtp_low_vol", 0.0)) or 0.0
    last_tvtp_high = _finite(last.get("tvtp_high_vol", 0.0)) or 0.0

    # Transition matrix on the rule-baseline label sequence (3y)
    trans_window = aligned["label"].tail(min(len(aligned), 252 * 3)).astype(int)
    transition_matrix = _empirical_transition_matrix(trans_window.to_numpy(), n_states=5)

    return {
        "ticker": ticker,
        "name":   ASSET_NAMES.get(ticker, ticker),
        "as_of":  aligned.index[-1].strftime("%Y-%m-%d"),
        "current_close": _finite(last["close"]),
        "current_regime": {
            "label": last_label,
            "name":  REGIME_NAMES[last_label],
            "alloc": REGIME_ALLOC[last_label],
            "color": REGIME_COLORS[last_label],
            "probs": last_probs,
        },
        "current_tvtp": {
            "p_low_vol":  last_tvtp_low,
            "p_high_vol": last_tvtp_high,
            "position":   last_tvtp_pos,
            "state":      "Low-Vol Bull" if last_tvtp_pos > 0 else "High-Vol Defense",
        },
        "stats": ASSET_BACKTEST_STATS.get(ticker, {}),
        "transition_matrix": transition_matrix,
        "history": history,
    }


def _empirical_transition_matrix(labels: np.ndarray, n_states: int) -> List[List[float]]:
    """Empirical p(s' | s) over the label sequence. Rows sum to 1."""
    counts = np.zeros((n_states, n_states), dtype=float)
    for i in range(1, len(labels)):
        a, b = int(labels[i - 1]), int(labels[i])
        if 0 <= a < n_states and 0 <= b < n_states:
            counts[a, b] += 1.0
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    probs = counts / row_sums
    return [[round(float(v), 4) for v in row] for row in probs]


def main(out_dir: Optional[Path] = None) -> int:
    if out_dir is None:
        out_dir = ROOT / "dashboard" / "public" / "data"
    regimes_dir = out_dir / "regimes"
    regimes_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = ROOT / "data" / "cache"
    start = "2015-01-01"
    # yfinance treats `end` as exclusive — push it 1 day past today so we
    # capture today's bar once it settles after market close.
    end = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")
    # GHA runs in a fresh container with no cache; locally each day creates
    # a new parquet under `data/cache/`. Set REGIME_V2_NO_CACHE=1 to force
    # a re-fetch instead of relying on yesterday's cached file.
    if os.environ.get("REGIME_V2_NO_CACHE") == "1":
        cache_dir = None
    print(f"[snapshot] window: {start} → {end} (cache={'on' if cache_dir else 'off'})")

    print(f"[snapshot] fetching aux bundle (VIX, VIX3M, TLT, GLD, FRED)...")
    aux_bundle = fetch_aux_data_bundle(
        start, end,
        cache_dir=cache_dir,
        fred_api_key=os.environ.get("FRED_API_KEY"),
    )

    summary_assets: List[Dict[str, Any]] = []
    for i, ticker in enumerate(DEFAULT_UNIVERSE, 1):
        print(f"[snapshot] [{i:>2}/{len(DEFAULT_UNIVERSE)}] {ticker}...", flush=True)
        try:
            close = load_close(ticker, start, end, cache_dir=cache_dir)
        except Exception as exc:
            print(f"  ! load failed: {exc}")
            continue

        try:
            payload = _build_asset_payload(ticker, close, aux_bundle)
        except Exception as exc:
            print(f"  ! payload build failed: {exc}")
            import traceback
            traceback.print_exc()
            continue

        # Write per-asset
        safe = ticker.replace("=", "_").replace("/", "_")
        asset_path = regimes_dir / f"{safe}.json"
        with open(asset_path, "w") as fh:
            json.dump(payload, fh, separators=(",", ":"))
        print(f"  ok  {asset_path.relative_to(ROOT)}  "
              f"({asset_path.stat().st_size / 1024:.1f} KB)  "
              f"regime={payload['current_regime']['name']}  "
              f"tvtp_pos={payload['current_tvtp']['position']:+.2f}")

        # Summary entry
        summary_assets.append({
            "ticker": ticker,
            "safe":   safe,
            "name":   payload["name"],
            "as_of":  payload["as_of"],
            "close":  payload["current_close"],
            "regime": payload["current_regime"],
            "tvtp":   payload["current_tvtp"],
            "sharpe_p50": payload["stats"].get("sharpe_p50"),
            "max_dd_p50": payload["stats"].get("max_dd_p50"),
        })

    # assets.json — front-end uses this to render the universe selector
    assets_index = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "universe": [
            {"ticker": t, "safe": t.replace("=", "_").replace("/", "_"),
             "name": ASSET_NAMES.get(t, t)}
            for t in DEFAULT_UNIVERSE
        ],
    }
    with open(out_dir / "assets.json", "w") as fh:
        json.dump(assets_index, fh, indent=2)
    print(f"[snapshot] wrote {out_dir / 'assets.json'}")

    # summary.json — multi-asset grid + headline stats
    summary = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_assets": len(summary_assets),
        "regime_names": REGIME_NAMES,
        "regime_alloc": REGIME_ALLOC,
        "regime_colors": REGIME_COLORS,
        "assets": summary_assets,
    }
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[snapshot] wrote {out_dir / 'summary.json'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
