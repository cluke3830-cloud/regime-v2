"""Benchmark strategies and concrete strategy_fn implementations.

These slot into the CPCV harness (``src.validation.cpcv_runner``) via the
:class:`StrategyFn` protocol — ``(features_train, features_test) ->
positions_test``.

``benchmarks`` ships the simple-but-not-trivial baselines that every more
sophisticated regime strategy must beat (audit §4.2 Tier-1 sanity).
"""

from src.strategies.benchmarks import (
    buy_and_hold,
    momentum_20d,
    flat,
)

__all__ = ["buy_and_hold", "momentum_20d", "flat"]