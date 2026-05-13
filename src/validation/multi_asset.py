"""Multi-asset robustness harness (Brief 1.5).

The audit (§4.4 / §8.1 / Brief 1.5) flags single-asset validation as a
universe-bias risk: a strategy can earn its Sharpe on SPY 2015-2024 by
implicitly betting on US-equity beta, then collapse on a different
universe with different microstructure (JPY, BTC), different drift sign
(treasuries during a hiking cycle), or different correlation regime (EFA
vs SPY during the 2022 dollar shock).

This module runs the existing CPCV harness across a heterogeneous asset
universe so the report carries an honest portability signal alongside the
in-universe metrics.

Acceptance soft gate (audit Brief 1.5):
    The strategy passes if  mean(per-asset OOS Sharpe p50) > 0  AND
    fraction of assets with positive OOS Sharpe p50 >= 0.70.

Design split:
    - ``evaluate_close``    — given a pre-loaded close series, run CPCV
      and return a metrics dict. Hermetic, no network — used by tests.
    - ``evaluate_one_asset`` — yfinance-download wrapper around
      ``evaluate_close``. Caches downloaded data to ``data/cache/`` as
      Parquet to make re-runs cheap.
    - ``evaluate_multi_asset`` — loop over the universe, calling
      ``evaluate_one_asset`` on each ticker. Failed downloads emit an
      error entry but do NOT abort the loop.
    - ``multi_asset_summary`` — aggregate fraction-positive + mean Sharpe.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.features.aux_data import fetch_aux_data_bundle
from src.features.price_features import (
    compute_features_v1,
    compute_features_v2,
)
from src.labels.triple_barrier import triple_barrier_labels
from src.validation.cpcv_runner import (
    StrategyFn,
    run_cpcv_validation,
)


DEFAULT_UNIVERSE: List[str] = [
    "SPY",      # US large-cap equity (in-sample baseline)
    "QQQ",      # US tech-heavy
    "DIA",      # US blue-chip
    "IWM",      # US small-cap
    "EFA",      # Developed-market ex-US equity
    "EEM",      # Emerging-market equity
    "GLD",      # Gold
    "TLT",      # Long-duration US treasuries
    "BTC-USD",  # Bitcoin
    "JPY=X",    # USD/JPY currency
]


# ---------------------------------------------------------------------------
# Feature engineering — kept identical to scripts/make_validation_report.py
# ---------------------------------------------------------------------------


def default_feature_fn(
    close: pd.Series,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Causally-clean Tier-1 features (Brief 2.1.1).

    Delegates to :func:`src.features.price_features.compute_features_v1`
    for the 14-feature audit-§5.2 set plus ``close``. Returns
    ``(features, log_returns)`` aligned to the surviving (non-NaN) index.

    Note: this stays at v1 features for hermeticity (no network, no aux
    data). The v2-with-macro version is wired in via
    :func:`make_feature_fn_v2` below — production reports use that
    factory; the tests stay on this hermetic v1 default.
    """
    features = compute_features_v1(close)
    log_ret = np.log(close).diff().loc[features.index]
    return features, log_ret


def make_feature_fn_v2(
    aux_bundle: dict,
    ohlc: Optional[pd.DataFrame] = None,
) -> Callable[[pd.Series], Tuple[pd.DataFrame, pd.Series]]:
    """Build a feature_fn that closes over a pre-fetched aux bundle.

    The bundle is intentionally fetched ONCE per report run (in the
    script) and shared across all assets — VIX/FRED don't depend on the
    asset being traded; only the asset's own close and the cross-asset
    closes (TLT/GLD) participate in per-asset features.

    Parameters
    ----------
    aux_bundle : dict
        Output of :func:`src.features.aux_data.fetch_aux_data_bundle`.
        Keys ``vix``, ``vix3m``, ``tlt``, ``gld``, ``term_spread``,
        ``credit_spread``. Any of which may be ``None`` (graceful
        degradation — the corresponding columns become 0).

    Returns
    -------
    Callable
        Signature matches :class:`default_feature_fn` — takes ``close``,
        returns ``(features, log_returns)``.
    """

    def feature_fn_v2(close: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        # OHLC only attaches if its index matches close (i.e. same ticker as
        # the one ohlc was fetched for). Mismatched-index assets get None →
        # yang_zhang_vol degrades to NaN → constant zero column.
        local_ohlc = ohlc if (ohlc is not None and len(ohlc.index.intersection(close.index)) > 0.9 * len(close)) else None
        features = compute_features_v2(
            close,
            ohlc=local_ohlc,
            vix=aux_bundle.get("vix"),
            vix3m=aux_bundle.get("vix3m"),
            vix6m=aux_bundle.get("vix6m"),
            vix9d=aux_bundle.get("vix9d"),
            skew=aux_bundle.get("skew"),
            vvix=aux_bundle.get("vvix"),
            tlt=aux_bundle.get("tlt"),
            gld=aux_bundle.get("gld"),
            term_spread=aux_bundle.get("term_spread"),
            credit_spread=aux_bundle.get("credit_spread"),
        )
        log_ret = np.log(close).diff().loc[features.index]
        return features, log_ret

    return feature_fn_v2


# ---------------------------------------------------------------------------
# Core (hermetic) — given a close series, run CPCV
# ---------------------------------------------------------------------------


def evaluate_close(
    close: pd.Series,
    strategy_fn: StrategyFn,
    *,
    feature_fn: Callable[[pd.Series], Tuple[pd.DataFrame, pd.Series]] = default_feature_fn,
    n_splits: int = 10,
    n_test_groups: int = 2,
    embargo_pct: float = 0.01,
    n_trials: int = 100,
    pi_up: float = 2.0,
    horizon: int = 10,
    seed: int = 42,
    strategy_name: str = "strategy",
) -> Dict[str, Any]:
    """Evaluate ``strategy_fn`` on a pre-loaded close series via CPCV.

    Parameters
    ----------
    close : pd.Series
        Pre-loaded close-to-close prices. Index defines the time order.
    strategy_fn :
        See ``cpcv_runner.StrategyFn``.
    feature_fn :
        Callable that maps ``close → (features_df, log_returns)``. Must
        return both with matching indices.
    n_splits, n_test_groups, embargo_pct, n_trials, seed :
        Forwarded to ``run_cpcv_validation``.
    pi_up, horizon :
        Triple-barrier params for per-sample label horizons (the CPCV
        purge uses these so training samples whose label window peeks
        into the test region are dropped).

    Returns
    -------
    dict
        ``{
            "n_bars": int,
            "n_paths": int,
            "sharpe_p05": float, "sharpe_p50": float, "sharpe_p95": float,
            "sharpe_mean": float,
            "max_dd_p05": float, "max_dd_p50": float, "max_dd_p95": float,
            "dsr_p_value": float, "dsr_observed_sharpe": float,
        }``
    """
    features, log_returns = feature_fn(close)

    labels = triple_barrier_labels(
        close=close.loc[features.index],
        vol=features["vol_ewma"],
        pi_up=pi_up,
        horizon=horizon,
    )
    label_horizons = (
        labels["t1"].to_numpy() - np.arange(len(labels))
    ).astype(np.int64)

    report = run_cpcv_validation(
        strategy_fn=strategy_fn,
        features_df=features,
        returns_series=log_returns,
        strategy_name=strategy_name,
        n_splits=n_splits,
        n_test_groups=n_test_groups,
        embargo_pct=embargo_pct,
        label_horizons=label_horizons,
        n_trials=n_trials,
        seed=seed,
    )

    return {
        "n_bars": len(features),
        "n_paths": report.n_paths,
        "sharpe_p05": report.sharpe_p05,
        "sharpe_p50": report.sharpe_p50,
        "sharpe_p95": report.sharpe_p95,
        "sharpe_mean": report.sharpe_mean,
        "max_dd_p05": report.max_dd_p05,
        "max_dd_p50": report.max_dd_p50,
        "max_dd_p95": report.max_dd_p95,
        "dsr_p_value": report.dsr_p_value,
        "dsr_observed_sharpe": report.dsr_observed_sharpe,
    }


# ---------------------------------------------------------------------------
# yfinance loader with Parquet cache
# ---------------------------------------------------------------------------


def _cache_path(ticker: str, start: str, end: str, cache_dir: Path) -> Path:
    safe_ticker = ticker.replace("=", "_").replace("/", "_")
    return cache_dir / f"{safe_ticker}_{start}_{end}.parquet"


def load_close(
    ticker: str,
    start: str,
    end: str,
    *,
    cache_dir: Optional[Path] = None,
) -> pd.Series:
    """Load close-to-close prices for ``ticker`` from yfinance with Parquet cache.

    The yfinance v0.2.x MultiIndex column scheme is flattened (matches
    the convention used elsewhere in the repo, see
    [regime_dashboard.py:541](regime_dashboard.py#L541)).

    Cached files live at ``cache_dir / "{ticker}_{start}_{end}.parquet"``.
    Set ``cache_dir=None`` to bypass the cache entirely.

    Raises
    ------
    RuntimeError
        If yfinance returns an empty frame (ticker delisted / wrong
        symbol / network failure).
    """
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cpath = _cache_path(ticker, start, end, cache_dir)
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
            f"yfinance frame for {ticker} has no 'close' column "
            f"(got {list(raw.columns)})"
        )
    close = raw["close"].astype(float).dropna()
    close.name = ticker

    if cache_dir is not None:
        # Save as 1-col DataFrame so column name (ticker) round-trips
        cpath = _cache_path(ticker, start, end, cache_dir)
        close.to_frame().to_parquet(cpath)
    return close


def evaluate_one_asset(
    ticker: str,
    strategy_fn: StrategyFn,
    start: str,
    end: str,
    *,
    cache_dir: Optional[Path] = None,
    **kwargs,
) -> Dict[str, Any]:
    """yfinance-loading wrapper around ``evaluate_close``.

    Any keyword args are forwarded to ``evaluate_close``. On a download
    failure, returns an error dict instead of raising — the multi-asset
    loop should continue past single-asset failures.

    Returns
    -------
    dict
        On success: same shape as ``evaluate_close`` + ``{"ticker": str,
        "error": None}``.
        On download failure: ``{"ticker": str, "error": str, ...keys
        present but float('nan')}``.
    """
    try:
        close = load_close(ticker, start, end, cache_dir=cache_dir)
    except Exception as exc:  # noqa: BLE001 — intentional broad catch
        return _failed_asset_result(ticker, str(exc))

    if len(close) < kwargs.get("n_splits", 10) * 50:
        return _failed_asset_result(
            ticker,
            f"too few bars ({len(close)}) for CPCV after burn-in",
        )

    try:
        metrics = evaluate_close(close, strategy_fn, **kwargs)
    except Exception as exc:  # noqa: BLE001
        return _failed_asset_result(ticker, f"evaluate_close failed: {exc}")
    metrics["ticker"] = ticker
    metrics["error"] = None
    return metrics


def _failed_asset_result(ticker: str, err: str) -> Dict[str, Any]:
    nan = float("nan")
    return {
        "ticker": ticker,
        "error": err,
        "n_bars": 0,
        "n_paths": 0,
        "sharpe_p05": nan, "sharpe_p50": nan, "sharpe_p95": nan,
        "sharpe_mean": nan,
        "max_dd_p05": nan, "max_dd_p50": nan, "max_dd_p95": nan,
        "dsr_p_value": nan, "dsr_observed_sharpe": nan,
    }


# ---------------------------------------------------------------------------
# Multi-asset loop + aggregation
# ---------------------------------------------------------------------------


def evaluate_multi_asset(
    strategy_fn: StrategyFn,
    asset_universe: Optional[List[str]] = None,
    *,
    start: str = "2015-01-01",
    end: str = "2025-01-01",
    cache_dir: Optional[Path] = None,
    progress: bool = False,
    **kwargs,
) -> Dict[str, Dict[str, Any]]:
    """Run ``strategy_fn`` across an asset universe via CPCV.

    Sequential loop — yfinance is rate-limited and the per-asset cost is
    dominated by CPCV path arithmetic, not network IO. Failed downloads
    don't abort the loop; their metrics dict carries an ``error`` field.

    Parameters
    ----------
    strategy_fn :
        Same strategy applied to every asset.
    asset_universe :
        Defaults to :data:`DEFAULT_UNIVERSE` (10 tickers).
    start, end :
        yfinance date range.
    cache_dir :
        Optional Parquet cache directory. Set to ``Path("data/cache")``
        in production.
    progress :
        If True, prints one line per asset evaluated.
    **kwargs :
        Forwarded to ``evaluate_close`` (n_splits, n_trials, embargo_pct,
        seed, etc.).

    Returns
    -------
    dict[str, dict]
        Keyed by ticker.
    """
    if asset_universe is None:
        asset_universe = list(DEFAULT_UNIVERSE)

    results: Dict[str, Dict[str, Any]] = {}
    for i, ticker in enumerate(asset_universe, 1):
        if progress:
            print(f"  [{i:>2}/{len(asset_universe)}] {ticker} ...",
                  end="", flush=True)
        metrics = evaluate_one_asset(
            ticker, strategy_fn, start, end,
            cache_dir=cache_dir, **kwargs,
        )
        results[ticker] = metrics
        if progress:
            if metrics["error"]:
                print(f" FAILED: {metrics['error']}")
            else:
                print(f" sharpe_p50={metrics['sharpe_p50']:+.3f}  "
                      f"DSR={metrics['dsr_p_value']:.3f}  "
                      f"n_bars={metrics['n_bars']}")
    return results


def multi_asset_summary(
    results: Dict[str, Dict[str, Any]],
    *,
    soft_gate_fraction: float = 0.70,
) -> Dict[str, Any]:
    """Aggregate per-asset results into the audit's soft-gate summary.

    Returns
    -------
    dict
        ``{
            "n_assets": int,
            "n_evaluated": int,
            "n_failed": int,
            "n_positive_sharpe": int,
            "fraction_positive_sharpe": float,
            "mean_sharpe_p50": float,
            "median_sharpe_p50": float,
            "passes_gate": bool,    # True iff fraction >= soft_gate_fraction
                                    #            AND mean_sharpe_p50 > 0
        }``
    """
    n_total = len(results)
    evaluated = [
        m for m in results.values()
        if m.get("error") is None and np.isfinite(m.get("sharpe_p50", float("nan")))
    ]
    n_eval = len(evaluated)
    sharpes = np.array([m["sharpe_p50"] for m in evaluated], dtype=float)

    if n_eval == 0:
        return {
            "n_assets": n_total,
            "n_evaluated": 0,
            "n_failed": n_total,
            "n_positive_sharpe": 0,
            "fraction_positive_sharpe": 0.0,
            "mean_sharpe_p50": float("nan"),
            "median_sharpe_p50": float("nan"),
            "passes_gate": False,
        }

    n_positive = int((sharpes > 0).sum())
    fraction_positive = n_positive / n_eval
    mean_sharpe = float(sharpes.mean())
    median_sharpe = float(np.median(sharpes))
    passes_gate = (fraction_positive >= soft_gate_fraction) and (mean_sharpe > 0)

    return {
        "n_assets": n_total,
        "n_evaluated": n_eval,
        "n_failed": n_total - n_eval,
        "n_positive_sharpe": n_positive,
        "fraction_positive_sharpe": fraction_positive,
        "mean_sharpe_p50": mean_sharpe,
        "median_sharpe_p50": median_sharpe,
        "passes_gate": passes_gate,
    }


__all__ = [
    "DEFAULT_UNIVERSE",
    "default_feature_fn",
    "evaluate_close",
    "load_close",
    "evaluate_one_asset",
    "evaluate_multi_asset",
    "multi_asset_summary",
]