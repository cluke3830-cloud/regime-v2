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
from src.validation.cost_model import CostModel  # noqa: E402
from src.validation.cpcv_runner import (  # noqa: E402
    emit_markdown_report,
    run_cpcv_multi_strategy,
)
from src.validation.risk_layer import RiskControls  # noqa: E402
from src.validation.multi_asset import (  # noqa: E402
    DEFAULT_UNIVERSE,
    evaluate_multi_asset,
    load_close,
    make_feature_fn_v2,
    multi_asset_summary,
)
from src.validation.strategy_registry import N_TRIALS_REGISTERED  # noqa: E402


CACHE_DIR = ROOT / "data" / "cache"
# Extended to 2000 to include 2000-02 dot-com and 2008 GFC — the two canonical
# Bear stress tests missing from the 2015+ window.
START = "2000-01-01"
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
    for name in ("vix", "vix3m", "vix6m", "vix9d", "skew", "vvix",
                 "tlt", "gld", "term_spread", "credit_spread"):
        s = bundle.get(name)
        if s is None:
            err = bundle.get(f"_{name}_error", "unknown error")
            print(f"      ⚠️  {name}: NOT AVAILABLE  ({err[:80]})")
        else:
            print(f"      ✓ {name}: {len(s)} obs  "
                  f"[{s.index.min().date()} → {s.index.max().date()}]")

    # SPY OHLC for Yang-Zhang volatility. Only attached to the primary asset
    # (SPY); multi-asset robustness loop sees ohlc=None and degrades gracefully.
    spy_ohlc = None
    try:
        import yfinance as yf  # noqa: PLC0415
        _raw = yf.download("SPY", start=START, end=END, progress=False, auto_adjust=True)
        # yfinance occasionally returns a MultiIndex on .columns; flatten it.
        if hasattr(_raw.columns, "nlevels") and _raw.columns.nlevels > 1:
            _raw.columns = [c[0] for c in _raw.columns]
        spy_ohlc = _raw[["Open", "High", "Low", "Close"]].rename(
            columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"}
        )
        print(f"      ✓ SPY OHLC: {len(spy_ohlc)} bars (for Yang-Zhang vol)")
    except Exception as exc:
        print(f"      ⚠️  SPY OHLC fetch failed ({exc}); Yang-Zhang vol degrades to 0.")

    return make_feature_fn_v2(bundle, ohlc=spy_ohlc), bundle


def main() -> int:
    feature_fn_v2, aux_bundle = _build_feature_fn_v2()

    print(f"[1/6] Loading SPY ({START}..{END}) [yfinance, cache={CACHE_DIR}] ...",
          flush=True)
    close = load_close("SPY", START, END, cache_dir=CACHE_DIR)
    features, log_returns = feature_fn_v2(close)
    print(f"      {len(features)} bars, "
          f"{features.index[0].date()} → {features.index[-1].date()}, "
          f"{features.shape[1] - 1} features (v2)")

    # Fetch daily volume for Amihud cost scaling. Volume is stored in the
    # same yfinance download as close; use a direct download to get the
    # "Volume" column and align it to features.index.
    try:
        import yfinance as yf  # noqa: PLC0415
        _raw = yf.download("SPY", start=START, end=END, progress=False, auto_adjust=True)
        _vol_col = "Volume" if "Volume" in _raw.columns else _raw.columns[_raw.columns.str.lower() == "volume"][0]
        spy_volume = _raw[_vol_col].reindex(features.index)
        print(f"      volume: {spy_volume.notna().sum()} bars  "
              f"median={spy_volume.median()/1e6:.1f}M shares/day")
    except Exception as exc:
        spy_volume = None
        print(f"      ⚠️  volume fetch failed ({exc}); falling back to flat cost")

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
    # Multi-model log-opinion-pool — fuses GMM-HMM + TVTP-MSAR via an
    # empirical TVTP→3-class mapping learned from rule_baseline label
    # frequencies in the training fold. Tests the fused regime signal
    # as an actual strategy rather than a dashboard artefact.
    from src.strategies.fusion import make_fusion_strategy  # noqa: PLC0415
    fusion = make_fusion_strategy()

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
        "fusion": fusion,                          # log-opinion-pool of GMM + TVTP
    }
    # Real cost model (SPY-specific bps + Amihud volume adjustment) and
    # portfolio risk controls (15% DD circuit-breaker + 95% VaR ≤ 2% NAV).
    cost_model = CostModel(ticker="SPY")
    risk_controls = RiskControls(dd_limit=0.15, dd_reentry=0.075, var_nav_pct=0.02)
    print(f"      cost model: SPY base={cost_model.base_bps()} bps + Amihud  |  "
          f"risk: DD≤{risk_controls.dd_limit:.0%}, VaR≤{risk_controls.var_nav_pct:.0%}")

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
        cost_model=cost_model,
        risk_controls=risk_controls,
        volume_series=spy_volume,
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

    print("[6/6] Computing regime diagnostics (NBER / ECE / stability / "
          "concordance / BIC-AIC) ...", flush=True)
    diagnostics_md = _compute_diagnostics_section(close, features)

    print("[7/7] Writing validation_report.md ...", flush=True)
    out = emit_markdown_report(
        reports,
        ROOT / "validation_report.md",
        label_balance=label_balance,
        multi_asset_results=multi_results,
        title=f"CPCV Validation Report — Phase 1 + Phase 2.1 XGBoost "
              f"(SPY {START[:4]}-{END[:4]}, robustness on '{winner_name}')",
    )
    # Append diagnostics section to the same file
    with open(out, "a", encoding="utf-8") as f:
        f.write("\n\n")
        f.write(diagnostics_md)
    print(f"      Report written: {out}")
    return 0


def _compute_diagnostics_section(close: pd.Series, features: pd.DataFrame) -> str:
    """Run all four regime diagnostics + BIC/AIC sweep, return markdown string."""
    import os
    from src.regime.rule_baseline import compute_rule_regime_sequence
    from src.regime.gmm_hmm import compute_gmm_hmm_sequence, select_hmm_k
    from src.baselines.tvtp_msar import MarkovSwitchingAR
    from src.validation.regime_diagnostics import (
        nber_alignment, reliability_diagram, regime_stability,
        cross_model_concordance,
    )

    lines = ["## Regime Diagnostics\n"]

    # Run all three models on the full series
    rule_seq = compute_rule_regime_sequence(features)
    gmm = compute_gmm_hmm_sequence(close)
    returns = np.log(close).diff().dropna()
    cut = int(len(returns) * 0.7)
    tvtp_model = MarkovSwitchingAR(k_regimes=2, order=1, switching_variance=True)
    tvtp_model.fit(returns.iloc[:cut])
    tvtp_probs = tvtp_model.predict_proba(returns) if tvtp_model.params_ is not None else None

    # ---- BIC/AIC across K
    lines.append("### Model selection — BIC/AIC across K\n")
    k_results = select_hmm_k(close, k_range=(2, 5))
    if k_results:
        lines.append("| K | log-likelihood | n_params | AIC | BIC |")
        lines.append("|---:|---:|---:|---:|---:|")
        for k, r in sorted(k_results.items()):
            lines.append(f"| {k} | {r['log_likelihood']} | {r['n_params']} "
                         f"| {r['aic']} | {r['bic']} |")
        best_bic_k = min(k_results, key=lambda k: k_results[k]["bic"])
        best_aic_k = min(k_results, key=lambda k: k_results[k]["aic"])
        lines.append(f"\n**Best K by BIC: {best_bic_k}**, **best K by AIC: {best_aic_k}**.\n")
    else:
        lines.append("_GMM-HMM K-selection failed (insufficient data?)._\n")

    # ---- NBER alignment (rule_baseline labels)
    lines.append("\n### NBER recession alignment (rule_baseline Bear vs USREC)\n")
    nber = nber_alignment(rule_seq["label"], fred_api_key=os.environ.get("FRED_API_KEY"))
    if nber.get("usrec_available"):
        lines.append(f"- Precision (Bear bars in NBER recession): **{nber['precision']:.1%}**")
        lines.append(f"- Recall (NBER recession bars labeled Bear): **{nber['recall']:.1%}**")
        lines.append(f"- F1: **{nber['f1']:.2f}**")
        lines.append(f"- Median lag from recession start to first Bear: "
                     f"**{nber['median_days_to_bear']} days**")
        lines.append(f"- Sample: {nber['n_recession_bars']} recession bars, "
                     f"{nber['n_bear_bars']} Bear bars, {nber['n_overlap']} overlap.")
    else:
        lines.append(f"_NBER fetch unavailable: {nber.get('error', 'no FRED_API_KEY?')}_\n")

    # ---- Regime stability
    lines.append("\n### Regime stability (rule_baseline)\n")
    stab = regime_stability(rule_seq["label"])
    lines.append(f"- Global flip rate: **{stab['flip_rate']:.3f}** "
                 f"(transitions per bar — `< 0.05` is healthy)")
    lines.append(f"- Dominant regime: **{stab['dominant_regime']}**")
    lines.append("\n| Regime | Mean duration (bars) | Median | # episodes | % time |")
    lines.append("|---:|---:|---:|---:|---:|")
    for lbl, s in sorted(stab["per_regime"].items()):
        lines.append(f"| {lbl} | {s['mean_duration_bars']} | "
                     f"{s['median_duration_bars']} | {s['n_episodes']} | "
                     f"{s['pct_time']:.1%} |")

    # ---- Calibration (GMM-HMM probs vs rule_baseline labels)
    lines.append("\n### Calibration — Expected Calibration Error\n")
    if gmm is not None:
        gmm_aligned = gmm.reindex(rule_seq.index).dropna()
        ref = rule_seq.loc[gmm_aligned.index, "label"].to_numpy()
        proba = gmm_aligned[["p_0", "p_1", "p_2"]].to_numpy()
        cal = reliability_diagram(proba, ref, n_bins=10)
        lines.append(f"- GMM-HMM mean ECE (vs rule_baseline labels): "
                     f"**{cal['mean_ece']:.3f}**")
        lines.append(f"- Per-class ECE: {cal['ece_per_class']}")
        lines.append("(0 = perfectly calibrated; < 0.05 is well-calibrated; "
                     "> 0.10 indicates systematic over/under-confidence.)")
    else:
        lines.append("_GMM-HMM fit failed; calibration skipped._\n")

    # ---- Cross-model concordance
    lines.append("\n### Cross-model concordance\n")
    if gmm is not None and tvtp_probs is not None:
        conc = cross_model_concordance(rule_seq["label"], gmm["label"], tvtp_probs)
        lines.append(f"- Rule vs GMM exact agreement: **{conc['rule_gmm_agreement']:.1%}**")
        lines.append(f"- Rule vs TVTP (Bear vs non-Bear): **{conc['rule_tvtp_agreement']:.1%}**")
        lines.append(f"- GMM vs TVTP (Bear vs non-Bear): **{conc['gmm_tvtp_agreement']:.1%}**")
        lines.append(f"- **Consensus score (≥2 of 3 agree): "
                     f"{conc['consensus_score']:.1%}**")
        lines.append(f"- Cohen's κ (rule, GMM): **{conc['cohen_kappa_rule_gmm']:.3f}**")
        lines.append("\nConfusion matrix (row=rule, col=GMM, labels 0/1/2):\n")
        lines.append("| | gmm=0 | gmm=1 | gmm=2 |")
        lines.append("|---:|---:|---:|---:|")
        for i, row in enumerate(conc["confusion_rule_gmm"]):
            lines.append(f"| **rule={i}** | {row[0]} | {row[1]} | {row[2]} |")
    else:
        lines.append("_GMM or TVTP fit failed; concordance skipped._\n")

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())