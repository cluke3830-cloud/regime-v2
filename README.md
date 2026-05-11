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
│   │   └── rule_baseline.py       # 5-regime hand-tuned rule classifier
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

## Current empirical results (SPY 2015-2024, 45 CPCV paths, n_trials=160)

| Strategy | Sharpe p50 | DSR Sharpe | Max-DD p50 | Verdict |
|---|---:|---:|---:|---|
| `flat` | 0.000 | — | 0.000 | reference |
| `buy_and_hold` | +0.807 | +0.761 | -19.2% | bull-market baseline |
| `momentum_20d` | +0.959 | +0.984 | -8.3% | simple-MA gold standard |
| `rule_baseline` (5-regime, hand-tuned) | +0.792 | +0.796 | -10.5% | matches buy-and-hold |
| `xgb_v1` (21 feat, audit defaults) | +0.129 | +0.130 | -10.9% | underfit on this regime mix |
| `xgb_v2` (21 feat, reg-heavy) | +0.259 | +0.152 | -7.6% | best xgb variant |
| `xgb_tuned` (nested CPCV grid) | +0.216 | +0.128 | -7.9% | tuner ≠ silver bullet |
| **`meta_equal`** (rule + momentum + B&H, equal blend) | **+1.137** | **+1.077** | -9.4% | **🏆 winner** |
| `meta_ridge` (rule + momentum + B&H, Ridge-weighted) | +0.953 | +0.769 | -1.0% | matches momentum |
| `transition_gated` (rule × `(1 - P(transition))`) | +0.773 | +0.742 | -8.7% | gate slightly hurts |

**Soft gate:** Multi-asset robustness on `meta_equal` across 10 tickers — **9/10 positive OOS Sharpe (90%), mean +0.610** ✅. Best asset: BTC-USD +1.262; worst: TLT -0.019 (rate-hike regime hurts equal-weight).

**Caveat — PBO 71.11%:** Strategy-selection across this universe of variants is overfitting (audit §8.1.3 alarm zone is ≥ 0.70). Several of the 10 strategies are highly correlated (meta_equal is literally a linear combination of three other strategies), so IS-best rank flips OOS easily. The PBO is a *strategy-selection* warning, not a *strategy-performance* one — `meta_equal` itself has no tunable parameter to overfit. Phase 3 (econometric baselines) and Phase 4 (deep frontier) will introduce de-correlated voters that should bring PBO back under 0.50.

## 5-regime taxonomy

The rule classifier and the position-mapping layer use this taxonomy:

| Label | Name        | Allocation | Trigger |
|------:|-------------|-----------:|---------|
| 0     | Full Bull   | +1.00      | Strong uptrend, low vol, shallow DD |
| 1     | Half Bull   | +0.70      | Moderate uptrend or slow grind |
| 2     | Chop        | +0.20      | Sideways, no directional edge |
| 3     | Half Bear   | -0.20      | Moderate downtrend, elevated vol |
| 4     | Full Bear   | -0.50      | Severe stress (DD > 15%, shock > 3.5σ) |

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
| 2 | 2.2 Rule baseline (5 regimes)     | ✅ |
| 2 | 2.3 Stacked meta-learner          | ✅ |
| 2 | 2.4 Transition detector rebuild   | ✅ |
| 3 | 3.1 TVTP-MSAR baseline            | ⏳ |
| 3 | 3.2 HSMM with Weibull durations   | ⏳ |
| 3 | 3.3 MS-GARCH                      | ⏳ |
| 4 | 4.1 PatchTST deep ensemble        | ⏳ |
| 4 | 4.2 Adaptive conformal calibration| ⏳ |
| 5 | 5.1 Drift monitor                 | ⏳ |
| 5 | 5.2 Intraday realised variance    | ⏳ |
| 5 | 5.3 Live-replay test harness      | ⏳ |
| 5 | 5.4 Numba HMM forward filter      | ⏳ |

## References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*.
  Wiley. — CPCV (§7.4), triple-barrier labels (§3.2), sample weights
  (§4.3–4.5), PBO (§11.6).
- Bailey, D. and López de Prado, M. (2014). *The Deflated Sharpe Ratio*.
  Journal of Portfolio Management, 40(5), 94-107.
- The 56-page forensic audit (`regime_audit_report.pdf` — kept private)
  that prescribed this rewrite.
