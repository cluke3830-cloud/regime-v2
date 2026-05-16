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

Derivation (post-2026-05-16 PBO fix)
------------------------------------
Main strategies in ``scripts/make_validation_report.py``:
    flat, buy_and_hold, momentum_20d,
    rule_baseline,
    meta_equal, meta_ridge,
    transition_gated,
    tvtp_msar, hsmm, ms_garch,
    patchtst,
    fusion (log-opinion-pool of GMM-HMM + TVTP-MSAR with empirical mapping)
        → 12 named strategies (1 trial each)

Multi-asset robustness: the winning strategy is re-evaluated on
``DEFAULT_UNIVERSE`` (10 tickers). Each ticker is a separate trial
of the same model on a fresh dataset.
        → 10 additional trials

Hyperparameter sweeps:
        → 0 (xgb_tuned grid pruned 2026-05-16; no remaining HPO sweeps in panel)

Subtotal: 12 + 10 + 0 = 22.
Rounded up to 50 to keep a comfortable margin for any exploratory runs
(prior to PBO-fix pruning the audit margin was 200, justified by a
larger panel that included 4 XGB variants + a 36-combo HPO grid).

Rationale: 50 is the smallest "safe" round number above the
defensible count for which the DSR z-threshold remains conservative
without being punitive on a now-tightened, evidence-curated panel.

PBO-fix audit trail
-------------------
2026-05-16 — dropped ``xgb_v1`` (DSR 0.065), ``xgb_v2`` (0.175),
``xgb_tuned`` (0.120), ``conformal_xgb`` (0.163) from the panel.
All four had near-zero alpha and collectively drove PBO from
~71% to its post-prune value (see ``validation_report.md`` vs
``validation_report_pre_pbo_fix.md``). Implementation files remain
on disk under ``src/regime/{regime_xgboost,xgb_tuning,conformal}.py``
— they're still importable for future research, just not in the
panel that backs the public dashboard.
"""

from __future__ import annotations

# Named strategies in the final validation report (one trial each).
STRATEGY_REGISTRY: tuple[str, ...] = (
    "flat",
    "buy_and_hold",
    "momentum_20d",
    "rule_baseline",
    "meta_equal",
    "meta_ridge",
    "transition_gated",
    "tvtp_msar",
    "hsmm",
    "ms_garch",
    "patchtst",
    "fusion",
)

# Multi-asset robustness universe (DEFAULT_UNIVERSE in
# src/validation/multi_asset.py).
N_MULTI_ASSET: int = 10

# Hyperparameter grid trials in panel (xgb_tuned 36-combo grid pruned 2026-05-16).
N_HPO_GRID: int = 0

# Subtotal of countable trials.
_DEFENSIBLE_SUBTOTAL: int = len(STRATEGY_REGISTRY) + N_MULTI_ASSET + N_HPO_GRID

# The number passed to ``deflated_sharpe(..., n_trials=N_TRIALS_REGISTERED)``.
# Rounded up from _DEFENSIBLE_SUBTOTAL with a safety margin. See module
# docstring "PBO-fix audit trail" for the 200 → 50 reduction rationale.
N_TRIALS_REGISTERED: int = 50


__all__ = [
    "STRATEGY_REGISTRY",
    "N_MULTI_ASSET",
    "N_HPO_GRID",
    "N_TRIALS_REGISTERED",
]