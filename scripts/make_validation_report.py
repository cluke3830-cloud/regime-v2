"""Generate the validation_report.md on real data.

Operational entry point — `python scripts/make_validation_report.py` — that
wires Phase 1 Briefs 1.1-1.5 AND Phase 2 Brief 2.1 together:

    yfinance SPY (10-year window)
        → log returns + EWMA-vol + 20-day momentum (causally computed)
        → triple-barrier labels for context (Brief 1.3)
        → CPCV with 45 paths (Brief 1.1)
        → multi-strategy run on {flat, buy_and_hold, momentum_20d, xgb_v1}
          (Brief 1.4 + Brief 2.1)
        → DSR + PBO on the OOS path Sharpe matrix (Brief 1.2)
        → multi-asset robustness across 10 tickers, on the WINNING
          strategy (Brief 1.5)
        → validation_report.md at Regime_v2/

n_trials is pre-registered at 100 per the audit's recommendation
(changelog v1-v11 + hyperparameter sweeps in the existing dashboard).
Do NOT raise n_trials retroactively to make DSR look better.

Reproducibility — same seed + same data → identical report. yfinance
adjusts splits/dividends behind the scenes; downloads are Parquet-cached
in ``data/cache/`` so re-runs are byte-stable.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines.hsmm import make_hsmm_strategy  # noqa: E402
from src.baselines.ms_garch import make_ms_garch_strategy  # noqa: E402
from src.baselines.tvtp_msar import make_tvtp_msar_strategy  # noqa: E402
from src.regime.conformal import (  # noqa: E402
    make_conformal_calibrated_strategy,
    regime_xgboost_proba_fn,
)
from src.regime.patchtst import make_patchtst_strategy  # noqa: E402
from src.features.aux_data import fetch_aux_data_bundle  # noqa: E402
from src.labels.triple_barrier import triple_barrier_labels  # noqa: E402
from src.regime.meta_stacker import (  # noqa: E402
    make_equal_weight_stacked_strategy,
    make_ridge_stacked_strategy,
)
from src.regime.regime_xgboost import make_regime_xgboost_strategy  # noqa: E402
from src.regime.rule_baseline import rule_baseline_strategy  # noqa: E402
from src.regime.transition_detector import (  # noqa: E402
    make_transition_gated_strategy,
)
from src.regime.xgb_tuning import (  # noqa: E402
    DEFAULT_PARAM_GRID_SMALL,
    make_tuned_regime_xgboost_strategy,
)
from src.strategies.benchmarks import (  # noqa: E402
    buy_and_hold,
    flat,
    momentum_20d,
)
from src.validation.cpcv_runner import (  # noqa: E402
    emit_markdown_report,
    run_cpcv_multi_strategy,
)
from src.validation.multi_asset import (  # noqa: E402
    DEFAULT_UNIVERSE,
    evaluate_multi_asset,
    load_close,
    make_feature_fn_v2,
    multi_asset_summary,
)
from src.validation.strategy_registry import N_TRIALS_REGISTERED  # noqa: E402


CACHE_DIR = ROOT / "data" / "cache"
START = "2015-01-01"
END = "2025-01-01"


def _build_feature_fn_v2():
    """Fetch the Tier-2 aux bundle ONCE and return a v2 feature_fn.

    yfinance handles VIX/^VIX3M/TLT/GLD; FRED key comes from env var
    ``FRED_API_KEY``. Failed fetches degrade to zero-columns inside
    ``compute_features_v2`` — XGBoost handles the constants fine.
    """
    print("[aux] Fetching VIX / VIX3M / TLT / GLD + FRED bundle ...", flush=True)
    bundle = fetch_aux_data_bundle(
        start=START,
        end=END,
        cache_dir=CACHE_DIR,
        fred_api_key=os.environ.get("FRED_API_KEY"),
    )
    for name in ("vix", "vix3m", "tlt", "gld", "term_spread", "credit_spread"):
        s = bundle.get(name)
        if s is None:
            err = bundle.get(f"_{name}_error", "unknown error")
            print(f"      ⚠️  {name}: NOT AVAILABLE  ({err[:80]})")
        else:
            print(f"      ✓ {name}: {len(s)} obs  "
                  f"[{s.index.min().date()} → {s.index.max().date()}]")
    return make_feature_fn_v2(bundle), bundle


def main() -> int:
    feature_fn_v2, aux_bundle = _build_feature_fn_v2()

    print(f"[1/6] Loading SPY ({START}..{END}) [yfinance, cache={CACHE_DIR}] ...",
          flush=True)
    close = load_close("SPY", START, END, cache_dir=CACHE_DIR)
    features, log_returns = feature_fn_v2(close)
    print(f"      {len(features)} bars, "
          f"{features.index[0].date()} → {features.index[-1].date()}, "
          f"{features.shape[1] - 1} features (v2)")

    print("[2/6] Computing triple-barrier labels (π=2.0, h=10) for context ...",
          flush=True)
    labels = triple_barrier_labels(
        close=close.loc[features.index],
        vol=features["vol_ewma"],
        pi_up=2.0,
        horizon=10,
    )
    label_balance = (
        labels["label"].value_counts(normalize=True).sort_index().to_dict()
    )
    label_balance = {int(k): float(v) for k, v in label_balance.items()}
    label_horizons = (labels["t1"].to_numpy() - np.arange(len(labels))).astype(
        np.int64
    )

    print("[3/6] Running CPCV multi-strategy on SPY "
          "(10 splits, 2 test groups, 45 paths) ...", flush=True)
    # Brief 2.1.2 XGBoost — Tier-2 features (21 columns: 14 price + 7 macro).
    # Two variants pre-registered side-by-side:
    #   xgb_v1: audit §8.2.1 default hparams (depth=4, n_est=200, L2=1)
    #   xgb_v2: regularization-heavy for the higher-dimensional feature
    #           set (depth=3, n_est=100, L2=2). Forces feature selection
    #           via shallower trees, ridge prior.
    # Both are SINGLE configs (no in-script grid search). n_trials=120
    # accounts for the two pre-registered specs + 100-spec prior budget.
    xgb_v1 = make_regime_xgboost_strategy(
        pi_up=2.0,
        horizon=10,
        max_depth=4,
        eta=0.05,
        n_estimators=200,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.1,
        seed=42,
        n_jobs=1,
    )
    xgb_v2 = make_regime_xgboost_strategy(
        pi_up=2.0,
        horizon=10,
        max_depth=3,           # shallower — forces feature selection
        eta=0.05,
        n_estimators=100,      # fewer rounds — less time to memorise noise
        subsample=0.7,
        colsample_bytree=0.7,  # randomly drop features per tree
        reg_lambda=2.0,        # heavier L2 — push noise features to 0
        reg_alpha=0.2,
        seed=42,
        n_jobs=1,
    )
    # Brief 2.1.3 — nested CPCV grid search over (max_depth, n_estimators).
    # Inner CV picks the best hparams PER OUTER FOLD on inner-OOS log-loss.
    # Grid is the SMALL 4-combo default; expanding to FULL bumps wall-clock
    # ~9× and needs n_trials ≥ 220 to stay honest under DSR deflation.
    xgb_tuned = make_tuned_regime_xgboost_strategy(
        param_grid=DEFAULT_PARAM_GRID_SMALL,
        inner_n_splits=5,
        inner_n_test_groups=1,
        inner_embargo_pct=0.01,
        pi_up=2.0,
        horizon=10,
        # Fixed kwargs (held across the grid)
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.1,
        seed=42,
        n_jobs=1,
    )
    # Brief 2.3 — stacked meta-learners over the three deterministic bases.
    meta_equal = make_equal_weight_stacked_strategy({
        "buy_and_hold": buy_and_hold,
        "momentum_20d": momentum_20d,
        "rule_baseline": rule_baseline_strategy,
    })
    meta_ridge = make_ridge_stacked_strategy(
        {
            "buy_and_hold": buy_and_hold,
            "momentum_20d": momentum_20d,
            "rule_baseline": rule_baseline_strategy,
        },
        alpha=1.0, non_negative=True, position_scale=100.0,
    )
    # Brief 2.4 — rule_baseline gated by transition detector (smooth gate).
    transition_gated = make_transition_gated_strategy(
        horizon=5, n_estimators=80, max_depth=4, smooth_gate=True, seed=42,
    )
    # Brief 3.1 — Hamilton MS-AR(1) baseline w/ low-vol/high-vol states.
    tvtp_msar = make_tvtp_msar_strategy()
    # Brief 3.2 — 4-state Gaussian HMM w/ Weibull duration distributions.
    hsmm = make_hsmm_strategy(k_states=4)
    # Brief 3.3 — GARCH(1,1) vol-conditional sizing.
    ms_garch = make_ms_garch_strategy(target_ann_vol=0.14)
    # Brief 4.1 — PatchTST deep ensemble (lightweight settings for CPCV
    # wall-clock; n_seeds=2, epochs=20, seq_len=20).
    patchtst = make_patchtst_strategy(
        pi_up=2.0, horizon=10, seq_len=20,
        n_seeds=2, epochs=20, batch_size=64,
    )
    # Brief 4.2 — Adaptive Conformal Inference wrapping xgb_v2's
    # probability output. Calibrates to 90% target coverage.
    conformal_xgb = make_conformal_calibrated_strategy(
        base_proba_fn=regime_xgboost_proba_fn(
            max_depth=3, eta=0.05, n_estimators=100,
            subsample=0.7, colsample_bytree=0.7,
            reg_lambda=2.0, reg_alpha=0.2, seed=42, n_jobs=1,
        ),
        alpha=0.10, gamma=0.005, window=500,
    )

    strategies = {
        "flat": flat,
        "buy_and_hold": buy_and_hold,
        "momentum_20d": momentum_20d,
        "rule_baseline": rule_baseline_strategy,  # Brief 2.2
        "xgb_v1": xgb_v1,
        "xgb_v2": xgb_v2,
        "xgb_tuned": xgb_tuned,
        "meta_equal": meta_equal,                  # Brief 2.3a
        "meta_ridge": meta_ridge,                  # Brief 2.3b
        "transition_gated": transition_gated,      # Brief 2.4
        "tvtp_msar": tvtp_msar,                    # Brief 3.1
        "hsmm": hsmm,                              # Brief 3.2
        "ms_garch": ms_garch,                      # Brief 3.3
        "patchtst": patchtst,                      # Brief 4.1
        "conformal_xgb": conformal_xgb,            # Brief 4.2
    }
    reports = run_cpcv_multi_strategy(
        strategies=strategies,
        features_df=features,
        returns_series=log_returns,
        n_splits=10,
        n_test_groups=2,
        embargo_pct=0.01,
        label_horizons=label_horizons,
        # Trial accounting lives in src/validation/strategy_registry.py.
        # Bump N_TRIALS_REGISTERED there whenever a new variant is added
        # — never adjust it inline here.
        n_trials=N_TRIALS_REGISTERED,
        seed=42,
    )

    print("[4/6] SPY aggregate Sharpe percentiles:")
    for name, r in reports.items():
        print(f"      {name:>14}  "
              f"p05={r.sharpe_p05:+.3f}  "
              f"p50={r.sharpe_p50:+.3f}  "
              f"p95={r.sharpe_p95:+.3f}  "
              f"DSR={r.dsr_p_value:.3f}")
    pbo = next(iter(reports.values())).pbo
    print(f"      PBO = {pbo:.2%}" if pbo is not None else "      PBO = n/a")
    print(f"      Triple-barrier balance: {label_balance}")

    # Multi-asset robustness on the WINNING strategy. Pick whichever
    # non-trivial strategy has the best SPY Sharpe p50.
    candidates = [
        ("xgb_v1",           reports["xgb_v1"].sharpe_p50),
        ("xgb_v2",           reports["xgb_v2"].sharpe_p50),
        ("xgb_tuned",        reports["xgb_tuned"].sharpe_p50),
        ("rule_baseline",    reports["rule_baseline"].sharpe_p50),
        ("meta_equal",       reports["meta_equal"].sharpe_p50),
        ("meta_ridge",       reports["meta_ridge"].sharpe_p50),
        ("transition_gated", reports["transition_gated"].sharpe_p50),
        ("tvtp_msar",        reports["tvtp_msar"].sharpe_p50),
        ("hsmm",             reports["hsmm"].sharpe_p50),
        ("ms_garch",         reports["ms_garch"].sharpe_p50),
        ("patchtst",         reports["patchtst"].sharpe_p50),
        ("conformal_xgb",    reports["conformal_xgb"].sharpe_p50),
        ("momentum_20d",     reports["momentum_20d"].sharpe_p50),
    ]
    winner_name, winner_sharpe = max(candidates, key=lambda x: x[1])
    winner_fn = strategies[winner_name]
    print(f"[5/6] Multi-asset robustness on '{winner_name}' "
          f"(SPY p50 Sharpe = {winner_sharpe:+.3f}) "
          f"across {len(DEFAULT_UNIVERSE)} tickers ...", flush=True)
    multi_results = evaluate_multi_asset(
        winner_fn,
        DEFAULT_UNIVERSE,
        start=START,
        end=END,
        cache_dir=CACHE_DIR,
        n_splits=10,
        n_test_groups=2,
        embargo_pct=0.01,
        n_trials=100,
        seed=42,
        progress=True,
        feature_fn=feature_fn_v2,  # Brief 2.1.2 — same v2 features per asset
    )
    summary = multi_asset_summary(multi_results)
    print(f"      → {summary['n_evaluated']}/{summary['n_assets']} evaluated, "
          f"{summary['n_failed']} failed")
    print(f"      → fraction positive Sharpe: "
          f"{summary['fraction_positive_sharpe']:.0%}")
    print(f"      → mean p50 Sharpe: {summary['mean_sharpe_p50']:+.3f}")
    print(f"      → soft gate (≥70% positive AND mean>0): "
          f"{'PASS' if summary['passes_gate'] else 'FAIL'}")

    print("[6/6] Writing validation_report.md ...", flush=True)
    out = emit_markdown_report(
        reports,
        ROOT / "validation_report.md",
        label_balance=label_balance,
        multi_asset_results=multi_results,
        title=f"CPCV Validation Report — Phase 1 + Phase 2.1 XGBoost "
              f"(SPY 2015-2024, robustness on '{winner_name}')",
    )
    print(f"      Report written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())