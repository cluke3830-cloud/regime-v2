"""Pre-registered strategy / variant inventory for DSR multiple-testing.

Bailey & López de Prado's Deflated Sharpe Ratio penalises the observed
Sharpe by the *number of model variants tested* (``n_trials``). The
honest input to that deflation is the count of variants the researcher
ever evaluated against the same dataset — not just the ones in the
final report — because hidden trials inflate the Type-1 error rate on
the survivors.

This module is the single source of truth for that count. Bump
``N_TRIALS_REGISTERED`` whenever a new strategy variant or
hyperparameter sweep is run against SPY (or the multi-asset universe).

Derivation
----------
Main strategies in ``scripts/make_validation_report.py``:
    flat, buy_and_hold, momentum_20d,
    rule_baseline,
    xgb_v1, xgb_v2, xgb_tuned,
    meta_equal, meta_ridge,
    transition_gated,
    tvtp_msar, hsmm, ms_garch,
    patchtst, conformal_xgb
        → 15 named strategies (1 trial each)

Multi-asset robustness: the winning strategy is re-evaluated on
``DEFAULT_UNIVERSE`` (10 tickers). Each ticker is a separate trial
of the same model on a fresh dataset.
        → 10 additional trials

Hyperparameter sweeps already absorbed into the named variants:
    xgb_tuned is a 36-combo grid (max_depth × eta × n_estimators ×
    subsample), but the CPCV harness picks one combo per outer fold.
    Audit-conservative accounting counts every combo as a separate trial.
        → 36 grid combos

Subtotal: 15 + 10 + 36 = 61.
Audit safety margin (informal pre-registration drift, exploratory runs
during Brief 1–4 development, prior dashboard iterations): round to 200.

Rationale: 200 is the smallest "safe" round number above the
defensible count for which the DSR z-threshold (≈ 2.42σ at n=200)
forces any survivor to clear an honest bar.
"""

from __future__ import annotations

# Named strategies in the final validation report (one trial each).
STRATEGY_REGISTRY: tuple[str, ...] = (
    "flat",
    "buy_and_hold",
    "momentum_20d",
    "rule_baseline",
    "xgb_v1",
    "xgb_v2",
    "xgb_tuned",
    "meta_equal",
    "meta_ridge",
    "transition_gated",
    "tvtp_msar",
    "hsmm",
    "ms_garch",
    "patchtst",
    "conformal_xgb",
)

# Multi-asset robustness universe (DEFAULT_UNIVERSE in
# src/validation/multi_asset.py).
N_MULTI_ASSET: int = 10

# xgb_tuned full HPO grid size (max_depth × eta × n_estimators × subsample).
# See src/regime/xgb_tuning.py.
N_HPO_GRID: int = 36

# Subtotal of countable trials.
_DEFENSIBLE_SUBTOTAL: int = len(STRATEGY_REGISTRY) + N_MULTI_ASSET + N_HPO_GRID

# The number passed to ``deflated_sharpe(..., n_trials=N_TRIALS_REGISTERED)``.
# Rounded up from _DEFENSIBLE_SUBTOTAL with a safety margin for exploratory
# trials run during development. See module docstring for the audit trail.
N_TRIALS_REGISTERED: int = 200


__all__ = [
    "STRATEGY_REGISTRY",
    "N_MULTI_ASSET",
    "N_HPO_GRID",
    "N_TRIALS_REGISTERED",
]
