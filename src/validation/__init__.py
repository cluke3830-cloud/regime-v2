"""Validation harness — CPCV, Deflated Sharpe, PBO, multi-asset robustness.

Phase 1 of the regime upgrade roadmap. The audit (§5.11, §8.1) flagged the
existing single-split walk-forward + raw-Sharpe validation as the binding
constraint on publishability. This package replaces it.
"""

from src.validation.cv_purged import CombinatorialPurgedKFold
from src.validation.deflated_sharpe import (
    annualised_sharpe,
    deflated_sharpe,
    probability_of_backtest_overfitting,
)
from src.validation.cpcv_runner import (
    PathMetrics,
    ValidationReport,
    StrategyFn,
    run_cpcv_validation,
    run_cpcv_multi_strategy,
    emit_markdown_report,
)
from src.validation.live_replay import (
    replay_strategy_bar_by_bar,
    verify_no_lookahead,
)
from src.validation.multi_asset import (
    DEFAULT_UNIVERSE,
    default_feature_fn,
    evaluate_close,
    load_close,
    evaluate_one_asset,
    evaluate_multi_asset,
    multi_asset_summary,
)

__all__ = [
    "CombinatorialPurgedKFold",
    "annualised_sharpe",
    "deflated_sharpe",
    "probability_of_backtest_overfitting",
    "PathMetrics",
    "ValidationReport",
    "StrategyFn",
    "run_cpcv_validation",
    "run_cpcv_multi_strategy",
    "emit_markdown_report",
    "DEFAULT_UNIVERSE",
    "default_feature_fn",
    "evaluate_close",
    "load_close",
    "evaluate_one_asset",
    "evaluate_multi_asset",
    "multi_asset_summary",
    "verify_no_lookahead",
    "replay_strategy_bar_by_bar",
]
