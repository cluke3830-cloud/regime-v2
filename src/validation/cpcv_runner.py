"""CPCV validation harness — the integrator for Briefs 1.1, 1.2, 1.3.

Brief 1.4 of the regime_dashboard upgrade plan.

This module replaces the single-split walk-forward + raw-Sharpe validation
at regime_dashboard.py:2802-3051 with López de Prado's combinatorial purged
cross-validation: 45 OOS paths instead of 1, deflated Sharpe ratio instead
of raw Sharpe, and probability of backtest overfitting when multiple
strategy variants are compared.

Public surface:

  - ``ValidationReport`` — dataclass holding per-path metrics + aggregate
    distributions + DSR + PBO + OOS return series (for downstream
    re-analysis without re-running the 45 paths).

  - ``run_cpcv_validation(strategy_fn, features_df, returns_series, ...)``
    Runs ONE strategy through CPCV. The strategy_fn signature is
    ``(features_train, features_test) -> positions_test`` — a 1-D array
    of positions aligned with the test indices. The harness multiplies
    positions × bar-returns to get strategy returns, then computes
    Sharpe / Sortino / max-DD / Calmar per path.

  - ``run_cpcv_multi_strategy(strategies, ...)`` — runs N strategies
    through the *same* folds so PBO is well-defined.

  - ``emit_markdown_report(reports, output_path)`` — writes
    ``validation_report.md`` with the audit-prescribed sections.

Design choice — strategy_fn returns POSITIONS, not returns. Two reasons:
  (1) The harness owns the multiplication by bar-returns, so there's no
      ambiguity about what "returns" mean (gross vs net, slippage applied
      or not). The strategy declares its position; the harness applies
      it to the underlying.
  (2) It cleanly handles regime-based strategies whose output is a
      probability-weighted long/short allocation per bar (which is
      exactly what regime_dashboard._compute_raw_signal does).

Causal hygiene — the harness does NOT inspect features_df for leakage.
The features must already be computed causally (features[t] uses only
data through t). When the regime_dashboard.compute_features pipeline
is wired through, that invariant is enforced upstream. The live-replay
test (Brief 5.3) is what verifies the invariant in production.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from src.validation.cv_purged import CombinatorialPurgedKFold
from src.validation.deflated_sharpe import (
    annualised_sharpe,
    deflated_sharpe,
    probability_of_backtest_overfitting,
)


# A strategy is any callable that accepts (features_train, features_test)
# and returns a 1-D numpy array of positions whose length matches
# len(features_test). Position convention: +1 = full long, -1 = full short,
# 0 = flat. Fractional values for risk-scaled positions are fine.
StrategyFn = Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]

from .strategy_registry import N_TRIALS_REGISTERED

ANN_FACTOR_DAILY = 252

# Transaction cost (one-way, in basis points) applied to |Δposition| per bar.
# 2 bps is the industry-standard liquid-ETF/large-cap-stock commission +
# spread cost at retail brokers (IBKR Tier-1, Fidelity zero-commission with
# realistic spreads). Per-bar cost = (COST_BPS_DEFAULT / 1e4) × |Δposition_t|.
# A full +1 → -1 flip costs 4 bps; a +1 → 0 unwind costs 2 bps.
# Set ``cost_bps=0`` in the runner to recover the frictionless backtest.
COST_BPS_DEFAULT = 2.0


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PathMetrics:
    """Metrics for a single CPCV path."""

    path_id: int
    n_bars: int
    total_return: float          # cumulative log return over the path
    sharpe: float                # annualised
    sortino: float               # annualised; downside-only denom
    max_drawdown: float          # negative number, e.g. -0.15 = 15% DD
    calmar: float                # annualised return / |max_drawdown|

    def as_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ValidationReport:
    """Aggregate CPCV validation report for a single strategy.

    All Sharpe / Sortino / max-DD / Calmar / DSR figures are NET of a
    one-way transaction cost of ``COST_BPS_DEFAULT`` (currently 2 bps)
    applied to |Δposition| per bar. Override at the runner via
    ``cost_bps=0`` to recover the frictionless backtest.

    The OOS return series across all paths are kept on the report so the
    caller can recompute alternative aggregations (rolling, regime-
    conditional, etc.) without re-running CPCV.
    """

    strategy_name: str
    n_paths: int
    n_trials: int                              # for DSR deflation
    path_metrics: List[PathMetrics]

    # Sharpe distribution across paths
    sharpe_p05: float
    sharpe_p50: float
    sharpe_p95: float
    sharpe_mean: float
    sharpe_std: float

    # Drawdown distribution
    max_dd_p05: float
    max_dd_p50: float
    max_dd_p95: float

    # Deflated Sharpe on concatenated OOS returns
    dsr_p_value: float
    dsr_observed_sharpe: float

    # PBO (filled in by run_cpcv_multi_strategy; None for single strategy)
    pbo: Optional[float] = None

    # OOS strategy-return series per path (path_id -> Series)
    oos_returns: Dict[int, pd.Series] = field(default_factory=dict)

    # IS / OOS sharpe matrices (filled by multi-strategy runner)
    is_sharpe_per_path: Optional[np.ndarray] = None
    oos_sharpe_per_path: Optional[np.ndarray] = None

    # Reproducibility
    seed: int = 42
    embargo_pct: float = 0.01
    n_splits: int = 10
    n_test_groups: int = 2
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# Per-path metrics
# ---------------------------------------------------------------------------


def _compute_path_metrics(
    path_id: int,
    strategy_returns: np.ndarray,
    ann_factor: int = ANN_FACTOR_DAILY,
) -> PathMetrics:
    """Compute Sharpe / Sortino / Max-DD / Calmar for one path."""
    r = strategy_returns[~np.isnan(strategy_returns)]
    n = len(r)
    if n < 2:
        return PathMetrics(path_id, n, 0.0, float("nan"), float("nan"),
                           float("nan"), float("nan"))

    total_ret = float(np.sum(r))
    sigma = float(np.std(r, ddof=1))
    sharpe = float(r.mean() / sigma * np.sqrt(ann_factor)) if sigma > 0 else 0.0

    downside = r[r < 0]
    downside_sigma = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
    sortino = (
        float(r.mean() / downside_sigma * np.sqrt(ann_factor))
        if downside_sigma > 0 else float("inf") if r.mean() > 0 else 0.0
    )

    equity = np.exp(np.cumsum(r))
    running_max = np.maximum.accumulate(equity)
    drawdown = equity / running_max - 1.0
    max_dd = float(drawdown.min())

    ann_ret = float(r.mean() * ann_factor)
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else float("inf") if ann_ret > 0 else 0.0

    return PathMetrics(
        path_id=path_id,
        n_bars=n,
        total_return=total_ret,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_dd,
        calmar=calmar,
    )


# ---------------------------------------------------------------------------
# Single-strategy CPCV runner
# ---------------------------------------------------------------------------


def run_cpcv_validation(
    strategy_fn: StrategyFn,
    features_df: pd.DataFrame,
    returns_series: pd.Series,
    *,
    strategy_name: str = "strategy",
    n_splits: int = 10,
    n_test_groups: int = 2,
    embargo_pct: float = 0.01,
    label_horizons: Optional[np.ndarray] = None,
    n_trials: int = N_TRIALS_REGISTERED,
    seed: int = 42,
    ann_factor: int = ANN_FACTOR_DAILY,
    cost_bps: float = COST_BPS_DEFAULT,
) -> ValidationReport:
    """Run a single strategy through CPCV and return its ValidationReport.

    Parameters
    ----------
    strategy_fn : callable
        ``strategy_fn(features_train, features_test) -> positions_test``.
        ``positions_test`` must be a 1-D array whose length matches
        ``len(features_test)``. Values are interpreted as positions —
        position * bar-return = strategy return.
    features_df : pd.DataFrame
        Causally-computed features. Index must align with ``returns_series``.
    returns_series : pd.Series
        Per-bar log returns of the underlying asset (close-to-close).
        ``returns_series.iloc[t]`` is the log return from bar t-1 to bar t.
    strategy_name : str
        Label for the report.
    n_splits, n_test_groups, embargo_pct, label_horizons :
        Forwarded to :class:`CombinatorialPurgedKFold`.
    n_trials : int
        Number of model variants explored — used by DSR deflation.
        Pre-register this number; do not tune. Defaults to
        ``strategy_registry.N_TRIALS_REGISTERED`` (currently 200,
        derived from 15 strategies + 10 universes + 36 HPO grid + slack).
    seed : int
        For any RNG used inside strategy_fn (the harness itself is
        deterministic).
    ann_factor : int
        Annualisation factor (252 for daily).
    cost_bps : float
        One-way transaction cost in basis points, applied to |Δposition|
        per bar. Default 2 bps (see ``COST_BPS_DEFAULT``). All downstream
        metrics (Sharpe, Sortino, max-DD, Calmar, DSR) are net-of-cost.
        Set ``cost_bps=0`` to recover the frictionless backtest.

    Returns
    -------
    ValidationReport
        See dataclass docs. The aggregate Sharpe percentiles are computed
        over the per-path Sharpes. The DSR p-value is computed on the
        concatenated OOS returns across all paths.

    Raises
    ------
    ValueError
        If ``features_df`` and ``returns_series`` are misaligned.
    """
    if len(features_df) != len(returns_series):
        raise ValueError(
            f"features_df len {len(features_df)} != returns_series len "
            f"{len(returns_series)}"
        )
    if not features_df.index.equals(returns_series.index):
        raise ValueError("features_df and returns_series must share an index")

    cv = CombinatorialPurgedKFold(
        n_splits=n_splits,
        n_test_groups=n_test_groups,
        embargo_pct=embargo_pct,
        label_horizons=label_horizons,
    )

    path_metrics: List[PathMetrics] = []
    oos_returns: Dict[int, pd.Series] = {}

    for path_id, (train_idx, test_idx) in enumerate(cv.split(features_df)):
        f_train = features_df.iloc[train_idx]
        f_test = features_df.iloc[test_idx]
        bar_returns = returns_series.iloc[test_idx].to_numpy()

        positions = strategy_fn(f_train, f_test)
        positions = np.asarray(positions, dtype=float)
        if len(positions) != len(test_idx):
            raise ValueError(
                f"strategy_fn returned {len(positions)} positions for "
                f"{len(test_idx)} test bars on path {path_id}"
            )

        gross_returns = positions * bar_returns
        # Transaction cost: cost_bps × |Δposition| per bar. prepend=0 charges
        # the cost of entering the very first position on the path.
        if cost_bps > 0:
            tc = (cost_bps / 1e4) * np.abs(np.diff(positions, prepend=0.0))
            strategy_returns = gross_returns - tc
        else:
            strategy_returns = gross_returns
        oos_returns[path_id] = pd.Series(
            strategy_returns,
            index=features_df.index[test_idx],
            name=f"path_{path_id}",
        )
        path_metrics.append(
            _compute_path_metrics(path_id, strategy_returns, ann_factor)
        )

    sharpes = np.array([pm.sharpe for pm in path_metrics])
    sharpes = sharpes[~np.isnan(sharpes)]
    dds = np.array([pm.max_drawdown for pm in path_metrics])
    dds = dds[~np.isnan(dds)]

    # DSR on the median-Sharpe path's returns — NOT the concat of all 45
    # paths. Concatenation pseudo-replicates bars (each bar appears in up to
    # 9 paths), inflating T ~9× and saturating the z-score. The median path
    # is a conservative, iid-valid single sample.
    valid_paths = [(pm.sharpe, pm.path_id) for pm in path_metrics if not np.isnan(pm.sharpe)]
    if valid_paths:
        valid_paths.sort(key=lambda x: x[0])
        median_path_id = valid_paths[len(valid_paths) // 2][1]
        median_returns = oos_returns[median_path_id].to_numpy()
        if len(median_returns) >= 30 and np.std(median_returns) > 0:
            dsr_p, dsr_sr = deflated_sharpe(
                median_returns, n_trials=n_trials, ann_factor=ann_factor
            )
        else:
            dsr_p, dsr_sr = float("nan"), float("nan")
    else:
        dsr_p, dsr_sr = float("nan"), float("nan")

    return ValidationReport(
        strategy_name=strategy_name,
        n_paths=len(path_metrics),
        n_trials=n_trials,
        path_metrics=path_metrics,
        sharpe_p05=float(np.percentile(sharpes, 5)) if len(sharpes) else float("nan"),
        sharpe_p50=float(np.percentile(sharpes, 50)) if len(sharpes) else float("nan"),
        sharpe_p95=float(np.percentile(sharpes, 95)) if len(sharpes) else float("nan"),
        sharpe_mean=float(sharpes.mean()) if len(sharpes) else float("nan"),
        sharpe_std=float(sharpes.std(ddof=1)) if len(sharpes) > 1 else float("nan"),
        max_dd_p05=float(np.percentile(dds, 5)) if len(dds) else float("nan"),
        max_dd_p50=float(np.percentile(dds, 50)) if len(dds) else float("nan"),
        max_dd_p95=float(np.percentile(dds, 95)) if len(dds) else float("nan"),
        dsr_p_value=dsr_p,
        dsr_observed_sharpe=dsr_sr,
        pbo=None,
        oos_returns=oos_returns,
        seed=seed,
        embargo_pct=embargo_pct,
        n_splits=n_splits,
        n_test_groups=n_test_groups,
    )


# ---------------------------------------------------------------------------
# Multi-strategy CPCV runner (enables PBO)
# ---------------------------------------------------------------------------


def run_cpcv_multi_strategy(
    strategies: Dict[str, StrategyFn],
    features_df: pd.DataFrame,
    returns_series: pd.Series,
    *,
    n_splits: int = 10,
    n_test_groups: int = 2,
    embargo_pct: float = 0.01,
    label_horizons: Optional[np.ndarray] = None,
    n_trials: int = N_TRIALS_REGISTERED,
    seed: int = 42,
    ann_factor: int = ANN_FACTOR_DAILY,
    cost_bps: float = COST_BPS_DEFAULT,
) -> Dict[str, ValidationReport]:
    """Run N strategies through the SAME CPCV folds → PBO is well-defined.

    The same (train_idx, test_idx) split is consumed by every strategy,
    so the IS and OOS performance matrices are aligned and PBO is
    computable. All strategies share ``n_trials`` and reproducibility
    settings.

    Returns
    -------
    Dict[str, ValidationReport]
        Keyed by strategy name. Each report has its ``pbo`` field filled
        with the SAME population PBO value (PBO is a property of the
        strategy population, not of any single strategy).
    """
    if len(strategies) < 1:
        raise ValueError("Provide at least one strategy")
    if len(features_df) != len(returns_series):
        raise ValueError("features_df / returns_series length mismatch")
    if not features_df.index.equals(returns_series.index):
        raise ValueError("features_df / returns_series index mismatch")

    cv = CombinatorialPurgedKFold(
        n_splits=n_splits,
        n_test_groups=n_test_groups,
        embargo_pct=embargo_pct,
        label_horizons=label_horizons,
    )

    names = list(strategies.keys())
    n_strategies = len(names)
    splits = list(cv.split(features_df))
    n_paths = len(splits)

    is_sharpes = np.zeros((n_paths, n_strategies))
    oos_sharpes = np.zeros((n_paths, n_strategies))
    per_strategy_metrics: Dict[str, List[PathMetrics]] = {n: [] for n in names}
    per_strategy_oos: Dict[str, Dict[int, pd.Series]] = {n: {} for n in names}
    per_strategy_concat: Dict[str, List[float]] = {n: [] for n in names}

    for path_id, (train_idx, test_idx) in enumerate(splits):
        f_train = features_df.iloc[train_idx]
        f_test = features_df.iloc[test_idx]
        r_train = returns_series.iloc[train_idx].to_numpy()
        r_test = returns_series.iloc[test_idx].to_numpy()

        for s_idx, name in enumerate(names):
            fn = strategies[name]
            # IS positions: same model trained on train, evaluated on train.
            # Apply cost on IS too so the IS Sharpe (feeds PBO) is consistent
            # with the OOS Sharpe — otherwise PBO inflates artificially.
            is_positions = np.asarray(fn(f_train, f_train), dtype=float)
            is_returns = is_positions * r_train
            if cost_bps > 0:
                is_tc = (cost_bps / 1e4) * np.abs(np.diff(is_positions, prepend=0.0))
                is_returns = is_returns - is_tc
            is_sharpes[path_id, s_idx] = annualised_sharpe(is_returns, ann_factor)

            oos_positions = np.asarray(fn(f_train, f_test), dtype=float)
            oos_returns_arr = oos_positions * r_test
            if cost_bps > 0:
                oos_tc = (cost_bps / 1e4) * np.abs(np.diff(oos_positions, prepend=0.0))
                oos_returns_arr = oos_returns_arr - oos_tc
            oos_sharpes[path_id, s_idx] = annualised_sharpe(oos_returns_arr, ann_factor)

            per_strategy_oos[name][path_id] = pd.Series(
                oos_returns_arr,
                index=features_df.index[test_idx],
                name=f"path_{path_id}",
            )
            per_strategy_concat[name].extend(oos_returns_arr.tolist())
            per_strategy_metrics[name].append(
                _compute_path_metrics(path_id, oos_returns_arr, ann_factor)
            )

    pbo = (
        probability_of_backtest_overfitting(is_sharpes, oos_sharpes)
        if n_strategies >= 2 else None
    )

    reports: Dict[str, ValidationReport] = {}
    for s_idx, name in enumerate(names):
        sharpes = np.array([pm.sharpe for pm in per_strategy_metrics[name]])
        sharpes = sharpes[~np.isnan(sharpes)]
        dds = np.array([pm.max_drawdown for pm in per_strategy_metrics[name]])
        dds = dds[~np.isnan(dds)]
        concat = np.array(per_strategy_concat[name], dtype=float)

        if len(concat) >= 30 and concat.std() > 0:
            dsr_p, dsr_sr = deflated_sharpe(
                concat, n_trials=n_trials, ann_factor=ann_factor
            )
        else:
            dsr_p, dsr_sr = float("nan"), float("nan")

        reports[name] = ValidationReport(
            strategy_name=name,
            n_paths=n_paths,
            n_trials=n_trials,
            path_metrics=per_strategy_metrics[name],
            sharpe_p05=float(np.percentile(sharpes, 5)) if len(sharpes) else float("nan"),
            sharpe_p50=float(np.percentile(sharpes, 50)) if len(sharpes) else float("nan"),
            sharpe_p95=float(np.percentile(sharpes, 95)) if len(sharpes) else float("nan"),
            sharpe_mean=float(sharpes.mean()) if len(sharpes) else float("nan"),
            sharpe_std=float(sharpes.std(ddof=1)) if len(sharpes) > 1 else float("nan"),
            max_dd_p05=float(np.percentile(dds, 5)) if len(dds) else float("nan"),
            max_dd_p50=float(np.percentile(dds, 50)) if len(dds) else float("nan"),
            max_dd_p95=float(np.percentile(dds, 95)) if len(dds) else float("nan"),
            dsr_p_value=dsr_p,
            dsr_observed_sharpe=dsr_sr,
            pbo=pbo,
            oos_returns=per_strategy_oos[name],
            is_sharpe_per_path=is_sharpes[:, s_idx],
            oos_sharpe_per_path=oos_sharpes[:, s_idx],
            seed=seed,
            embargo_pct=embargo_pct,
            n_splits=n_splits,
            n_test_groups=n_test_groups,
        )

    return reports


# ---------------------------------------------------------------------------
# Markdown report emitter
# ---------------------------------------------------------------------------


def _fmt(v: float, decimals: int = 3) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:.{decimals}f}"


def emit_markdown_report(
    reports: Dict[str, ValidationReport],
    output_path: str | Path,
    *,
    label_balance: Optional[Dict[int, float]] = None,
    multi_asset_results: Optional[Dict[str, Dict[str, float]]] = None,
    title: str = "CPCV Validation Report",
) -> Path:
    """Write a publication-grade Markdown report to ``output_path``.

    Sections (audit Brief 1.4 acceptance test):
      - Per-strategy Sharpe distribution (5/50/95 percentiles)
      - DSR with n_trials and resulting p-value
      - PBO percentage (when ≥ 2 strategies)
      - Triple-barrier label class balance (when provided)
      - Multi-asset robustness panel (when provided — populated by Brief 1.5)

    Parameters
    ----------
    reports : dict[str, ValidationReport]
        One report per strategy. Typically obtained from
        ``run_cpcv_multi_strategy``.
    output_path : str | Path
        Destination for the markdown file.
    label_balance : dict[int, float], optional
        Triple-barrier class fractions, e.g. ``{-1: 0.33, 0: 0.34, +1: 0.33}``.
    multi_asset_results : dict, optional
        Filled in by Brief 1.5; passed through verbatim.
    title : str
        Report title.

    Returns
    -------
    Path
        Absolute path of the written report.
    """
    path = Path(output_path).resolve()
    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    any_report = next(iter(reports.values()))
    lines.append(f"_Generated {any_report.timestamp} UTC_")
    lines.append("")
    lines.append("## CPCV configuration")
    lines.append("")
    lines.append(f"- n_splits: **{any_report.n_splits}**")
    lines.append(f"- n_test_groups: **{any_report.n_test_groups}**")
    lines.append(f"- embargo: **{any_report.embargo_pct:.2%}** of sample")
    lines.append(f"- paths per strategy: **{any_report.n_paths}**")
    lines.append(f"- seed: **{any_report.seed}**")
    lines.append(f"- n_trials (DSR deflation): **{any_report.n_trials}**")
    lines.append("")

    lines.append("## Per-strategy results")
    lines.append("")
    lines.append(
        "| strategy | paths | Sharpe p05 | Sharpe p50 | Sharpe p95 | "
        "Sharpe mean | Max-DD p50 | DSR p-value | DSR Sharpe |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for name, r in reports.items():
        lines.append(
            f"| `{name}` | {r.n_paths} | "
            f"{_fmt(r.sharpe_p05)} | {_fmt(r.sharpe_p50)} | "
            f"{_fmt(r.sharpe_p95)} | {_fmt(r.sharpe_mean)} | "
            f"{_fmt(r.max_dd_p50)} | {_fmt(r.dsr_p_value, 4)} | "
            f"{_fmt(r.dsr_observed_sharpe)} |"
        )
    lines.append("")

    if len(reports) >= 2:
        pbo = next(iter(reports.values())).pbo
        lines.append("## Probability of Backtest Overfitting (PBO)")
        lines.append("")
        if pbo is None:
            lines.append("_PBO not computed (single strategy)._")
        else:
            lines.append(f"**PBO: {pbo:.2%}**")
            lines.append("")
            if pbo < 0.5:
                lines.append("✅ Strategy selection generalises (PBO < 0.50).")
            elif pbo < 0.7:
                lines.append("⚠️ Borderline (0.50 ≤ PBO < 0.70). Investigate.")
            else:
                lines.append("❌ Severe overfitting (PBO ≥ 0.70). Backtest "
                             "is not informative.")
        lines.append("")

    if label_balance is not None:
        lines.append("## Triple-barrier label balance")
        lines.append("")
        lines.append("| label | fraction |")
        lines.append("|---:|---:|")
        for cls in sorted(label_balance.keys()):
            lines.append(f"| {cls:+d} | {label_balance[cls]:.2%} |")
        lines.append("")

    if multi_asset_results is not None:
        lines.append("## Multi-asset robustness (Brief 1.5)")
        lines.append("")
        lines.append(
            "| asset | bars | Sharpe p05 | Sharpe p50 | Sharpe p95 | "
            "Max-DD p50 | DSR p-value | OOS positive? | notes |"
        )
        lines.append(
            "|---|---:|---:|---:|---:|---:|---:|:-:|---|"
        )
        evaluated = []
        for asset, metrics in multi_asset_results.items():
            err = metrics.get("error")
            n_bars = metrics.get("n_bars", 0)
            sharpe_p05 = metrics.get("sharpe_p05", float("nan"))
            sharpe_p50 = metrics.get("sharpe_p50", float("nan"))
            sharpe_p95 = metrics.get("sharpe_p95", float("nan"))
            max_dd = metrics.get("max_dd_p50", float("nan"))
            dsr = metrics.get("dsr_p_value", float("nan"))
            ok = err is None and np.isfinite(sharpe_p50)
            tick = "✓" if (ok and sharpe_p50 > 0) else (
                "✗" if ok else "—"
            )
            note = "" if err is None else f"FAILED: {err[:60]}"
            lines.append(
                f"| `{asset}` | {n_bars} | {_fmt(sharpe_p05)} | "
                f"{_fmt(sharpe_p50)} | {_fmt(sharpe_p95)} | "
                f"{_fmt(max_dd)} | {_fmt(dsr, 4)} | {tick} | {note} |"
            )
            if ok:
                evaluated.append(sharpe_p50)

        n_eval = len(evaluated)
        n_pos = sum(1 for s in evaluated if s > 0)
        frac_pos = n_pos / n_eval if n_eval else 0.0
        mean_sharpe = float(np.mean(evaluated)) if evaluated else float("nan")
        passes_gate = (frac_pos >= 0.70) and (mean_sharpe > 0)

        lines.append("")
        lines.append(
            f"**{n_pos}/{n_eval} evaluated assets show positive OOS Sharpe** "
            f"({frac_pos:.0%}, target ≥ 70%). "
            f"Mean p50 Sharpe = {mean_sharpe:+.3f}."
        )
        lines.append("")
        if passes_gate:
            lines.append("✅ **Soft gate PASSED** — strategy generalises "
                         "across the asset universe.")
        else:
            lines.append("⚠️ **Soft gate FAILED** — fewer than 70% positive "
                         "OR mean Sharpe ≤ 0. Strategy is universe-specific; "
                         "investigate.")
        lines.append("")

    lines.append("## Reproducibility")
    lines.append("")
    lines.append("Same seed + same input data + same n_trials → identical "
                 "report. Run `make validate` to regenerate.")
    lines.append("")

    path.write_text("\n".join(lines))
    return path


__all__ = [
    "PathMetrics",
    "ValidationReport",
    "StrategyFn",
    "run_cpcv_validation",
    "run_cpcv_multi_strategy",
    "emit_markdown_report",
]