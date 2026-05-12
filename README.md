# Regime_v2 — Legendary-tier regime detector

A clean-room rewrite of a six-regime market regime classifier, built
against the prescriptions of a 56-page forensic audit of the original
implementation. The audit scored the legacy classifier at **67.4 / 100**
on a four-axis rubric (mathematical correctness, causal hygiene,
statistical defensibility, ensemble contribution). The target for this
project is **≥ 90 / 100** — the audit's "legendary" tier.

## What's in the box

```
Regime_v2/
├── src/
│   ├── features/
│   │   ├── aux_data.py            # yfinance + FRED loaders w/ Parquet cache
│   │   └── price_features.py      # 14 price features (v1) + 7 macro (v2)
│   ├── labels/
│   │   └── triple_barrier.py      # López de Prado §3.2 labels
│   ├── regime/
│   │   ├── regime_xgboost.py      # 3-class XGBoost classifier + sample weights
│   │   ├── xgb_tuning.py          # Nested CPCV grid search
│   │   └── rule_baseline.py       # 3-regime hand-tuned rule classifier (+ riskoff gate)
│   ├── strategies/
│   │   └── benchmarks.py          # buy_and_hold / flat / momentum_20d
│   └── validation/
│       ├── cv_purged.py           # Combinatorial Purged K-Fold (López de Prado §7.4)
│       ├── deflated_sharpe.py     # Deflated Sharpe Ratio + PBO (Bailey 2014)
│       ├── cpcv_runner.py         # Multi-strategy CPCV harness + markdown report
│       └── multi_asset.py         # 10-asset robustness loop + soft gate
├── tests/                         # 124 acceptance tests, all green
├── scripts/
│   └── make_validation_report.py  # End-to-end report generator
└── pyproject.toml                 # pytest + lint config
```

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the test suite
python -m pytest tests/ -q
# expected: 124 passed

# 3. Generate the validation report on SPY 2015-2024
export FRED_API_KEY=your_fred_api_key_here
python scripts/make_validation_report.py
# writes validation_report.md at repo root, takes ~10-15 min on first run
# (subsequent runs are faster — yfinance + FRED responses are Parquet-cached)
```

## What the validation report contains

For each strategy in `{flat, buy_and_hold, momentum_20d, rule_baseline,
xgb_v1, xgb_v2, xgb_tuned, meta_equal, meta_ridge, transition_gated}`:

- **45 OOS CPCV paths** (10 splits × 2 test groups per combination).
- Per-path Sharpe / Sortino / Calmar / max-drawdown.
- 5th / 50th / 95th percentile of OOS Sharpe distribution.
- **Deflated Sharpe Ratio** with explicit `n_trials` (audit §8.1.2).
- **Probability of Backtest Overfitting** across strategy variants.

Plus a **multi-asset robustness panel**: the winning strategy's
performance across `{SPY, QQQ, DIA, IWM, EFA, EEM, GLD, TLT, BTC-USD,
JPY=X}`. The soft gate is ≥ 70% positive OOS Sharpe + mean positive.

## Current empirical results (SPY 2015-2024, 45 CPCV paths, n_trials=200, **2 bps one-way cost**)

All Sharpes are **net** of a 2 bps × |Δposition| transaction cost applied per
bar by the CPCV runner. This deliberately punishes high-turnover strategies
that look great on paper but burn the edge on commissions and spread. The
audit-grade trial count lives in [`src/validation/strategy_registry.py`](src/validation/strategy_registry.py)
(15 named strategies + 10 multi-asset universes + 36-combo HPO grid → `N_TRIALS_REGISTERED = 200`).

| Strategy | Sharpe p50 | DSR Sharpe | Max-DD p50 | Verdict |
|---|---:|---:|---:|---|
| `flat` | 0.000 | — | 0.000 | reference |
| `buy_and_hold` | +0.806 | +0.761 | -19.2% | bull-market baseline |
| `momentum_20d` | +0.914 | +0.949 | -8.5% | simple-MA gold standard |
| `rule_baseline` (3-regime, hand-tuned) | +0.881 | +0.843 | -10.5% | matches momentum after cost |
| `xgb_v1` (21 feat, audit defaults) | +0.065 | +0.075 | -11.0% | turnover-heavy, cost eats edge |
| `xgb_v2` (21 feat, reg-heavy) | +0.175 | +0.110 | -7.6% | better than v1, still cost-sensitive |
| `xgb_tuned` (nested CPCV grid) | +0.120 | +0.085 | -7.9% | tuner can't overcome cost |
| `meta_equal` (rule + momentum + B&H, equal blend) | +1.004 | +1.066 | -9.4% | runner-up |
| `meta_ridge` (rule + momentum + B&H, Ridge-weighted) | +0.844 | +0.791 | -1.0% | low DD, modest edge |
| `transition_gated` (rule × `(1 - P(transition))`) | +0.807 | +0.811 | -9.4% | gate ≈ neutral |
| `tvtp_msar` (Hamilton 2-state MS-AR + vol-regime sizing) | +0.717 | +0.712 | -8.4% | high turnover penalty |
| `hsmm` (4-state Gaussian HMM + Weibull durations) | +0.135 | +0.259 | -9.3% | diversifier |
| **`ms_garch`** (GARCH(1,1) vol-conditional sizing) | **+1.025** | **+0.973** | **-11.3%** | **🏆 Champion** |
| `patchtst` (Transformer deep ensemble, 2 seeds) | +0.737 | +0.097 | -2.9% | DSR flags as overfit |
| `conformal_xgb` (xgb_v2 + Gibbs-Candès calibration) | +0.163 | +0.079 | -8.1% | calibration ≠ alpha |

**Soft gate:** Multi-asset robustness on `ms_garch` across 10 tickers — **9/10 positive OOS Sharpe (90%), mean +0.544** ✅. Best assets: SPY +1.025, QQQ +0.958, DIA +0.937 (US large-caps remain the most regime-tractable). Crypto holds up too: BTC-USD +0.930. Worst: TLT -0.082 (the lone OOS-negative; 20Y Treasuries' single-regime drift makes vol-conditional sizing unhelpful).

**PBO: 71.11%** ⚠️ — the population-level overfitting probability tripped under cost. Before the 2 bps charge, PBO sat at 13.3% and TVTP-MSAR led at +2.46 Sharpe; after, the high-turnover models concentrated at the bottom of the ranking and MS-GARCH took the crown. The honest read: **with realistic costs, the strategy population is dominated by low-turnover vol/regime models**, and the IS/OOS rank consistency that PBO measures degrades. The multi-asset soft gate (90% positive across 10 universes) is what carries the rigor story here — model selection on SPY alone is no longer the trustworthy axis.

**Why MS-GARCH wins under cost:** the GARCH(1,1) vol forecast turns over only at the persistence horizon of the volatility process (weeks, not days), so the 2 bps × |Δposition| drag is roughly an order of magnitude smaller than for the Hamilton-AR or XGBoost strategies that re-position every bar. Its position mapping (vol-scaled long, no shorting) sidesteps the symmetric-cost problem that hurts the long/short variants.

## 3-regime taxonomy

The rule classifier and the position-mapping layer use this taxonomy (the
legacy 5-regime split was simplified to 3 after we found Half Bull / Half
Bear were rarely distinguishable from Bull / Bear without the riskoff gate):

| Label | Name    | Allocation | Trigger |
|------:|---------|-----------:|---------|
| 0     | Bull    | +1.00      | Uptrend, low vol, shallow DD |
| 1     | Neutral |  0.00      | Sideways or ambiguous, flat the book |
| 2     | Bear    | -0.50      | Severe stress (|drawdown| > 15%, |shock_z| > 3.5σ); promoted by `_riskoff_confirm` gate |

## Brief status

| Phase | Brief | Status |
|------:|-------|--------|
| 1 | 1.1 CombinatorialPurgedKFold      | ✅ |
| 1 | 1.2 Deflated Sharpe + PBO         | ✅ |
| 1 | 1.3 Triple-barrier labels         | ✅ |
| 1 | 1.4 CPCV runner + report          | ✅ |
| 1 | 1.5 Multi-asset robustness        | ✅ |
| 2 | 2.1 XGBoost classifier            | ✅ |
| 2 | 2.1.1 Tier-1 price features (14)  | ✅ |
| 2 | 2.1.2 Tier-2 macro features (+7)  | ✅ |
| 2 | 2.1.3 Nested CPCV grid search     | ✅ |
| 2 | 2.2 Rule baseline (3 regimes)     | ✅ |
| 2 | 2.3 Stacked meta-learner          | ✅ |
| 2 | 2.4 Transition detector rebuild   | ✅ |
| 3 | 3.1 TVTP-MSAR baseline            | ✅ |
| 3 | 3.2 HSMM with Weibull durations   | ✅ |
| 3 | 3.3 MS-GARCH                      | ✅ |
| 4 | 4.1 PatchTST deep ensemble        | ✅ |
| 4 | 4.2 Adaptive conformal calibration| ✅ |
| 5 | 5.1 Drift monitor                 | ✅ |
| 5 | 5.2 Intraday realised variance    | ✅ |
| 5 | 5.3 Live-replay test harness      | ✅ |
| 5 | 5.4 Numba HMM forward filter      | ✅ |

**🏆 All 17 audit briefs shipped. 225/225 tests green.**

## References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*.
  Wiley. — CPCV (§7.4), triple-barrier labels (§3.2), sample weights
  (§4.3–4.5), PBO (§11.6).
- Bailey, D. and López de Prado, M. (2014). *The Deflated Sharpe Ratio*.
  Journal of Portfolio Management, 40(5), 94-107.
- The 56-page forensic audit (`regime_audit_report.pdf` — kept private)
  that prescribed this rewrite.
