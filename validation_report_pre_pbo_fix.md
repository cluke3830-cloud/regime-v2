# CPCV Validation Report — Phase 1 + Phase 2.1 XGBoost (SPY 2015-2024, robustness on 'ms_garch')

_Generated 2026-05-12T22:46:39.624682+00:00 UTC_

## CPCV configuration

- n_splits: **10**
- n_test_groups: **2**
- embargo: **1.00%** of sample
- paths per strategy: **45**
- seed: **42**
- n_trials (DSR deflation): **200**

## Per-strategy results

| strategy | paths | Sharpe p05 | Sharpe p50 | Sharpe p95 | Sharpe mean | Max-DD p50 | DSR p-value | DSR Sharpe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `flat` | 45 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | — | — |
| `buy_and_hold` | 45 | -0.041 | 0.806 | 2.188 | 0.945 | -0.192 | 0.0448 | 0.806 |
| `momentum_20d` | 45 | 0.232 | 0.914 | 1.725 | 0.989 | -0.085 | 0.0581 | 0.914 |
| `rule_baseline` | 45 | -0.189 | 0.881 | 2.108 | 0.872 | -0.105 | 0.0541 | 0.881 |
| `xgb_v1` | 45 | -0.504 | 0.065 | 1.349 | 0.232 | -0.110 | 0.0037 | 0.065 |
| `xgb_v2` | 45 | -0.308 | 0.175 | 1.306 | 0.312 | -0.076 | 0.0057 | 0.175 |
| `xgb_tuned` | 45 | -0.372 | 0.120 | 1.423 | 0.288 | -0.079 | 0.0046 | 0.120 |
| `meta_equal` | 45 | 0.136 | 1.004 | 2.001 | 1.085 | -0.094 | 0.0669 | 1.004 |
| `meta_ridge` | 45 | -0.041 | 0.844 | 1.943 | 0.921 | -0.010 | 0.0470 | 0.844 |
| `transition_gated` | 45 | -0.273 | 0.807 | 1.959 | 0.814 | -0.094 | 0.0455 | 0.807 |
| `tvtp_msar` | 45 | 0.062 | 0.717 | 1.585 | 0.736 | -0.084 | 0.0341 | 0.717 |
| `hsmm` | 45 | -0.810 | 0.135 | 1.552 | 0.259 | -0.093 | 0.0048 | 0.135 |
| `ms_garch` | 45 | 0.083 | 1.025 | 2.098 | 1.029 | -0.113 | 0.0780 | 1.025 |
| `patchtst` | 45 | -0.645 | 0.737 | 1.664 | 0.575 | -0.029 | 0.0372 | 0.737 |
| `conformal_xgb` | 45 | -0.393 | 0.163 | 1.322 | 0.298 | -0.081 | 0.0054 | 0.163 |

## Probability of Backtest Overfitting (PBO)

**PBO: 71.11%**

❌ Severe overfitting (PBO ≥ 0.70). Backtest is not informative.

## Triple-barrier label balance

| label | fraction |
|---:|---:|
| -1 | 29.92% |
| +0 | 23.77% |
| +1 | 46.31% |

## Multi-asset robustness (Brief 1.5)

| asset | bars | Sharpe p05 | Sharpe p50 | Sharpe p95 | Max-DD p50 | DSR p-value | OOS positive? | notes |
|---|---:|---:|---:|---:|---:|---:|:-:|---|
| `SPY` | 2263 | 0.083 | 1.025 | 2.098 | -0.113 | 0.1183 | ✓ |  |
| `QQQ` | 2263 | 0.019 | 0.958 | 1.987 | -0.128 | 0.1014 | ✓ |  |
| `DIA` | 2263 | -0.069 | 0.937 | 2.066 | -0.123 | 0.0979 | ✓ |  |
| `IWM` | 2263 | -0.664 | 0.284 | 1.342 | -0.157 | 0.0157 | ✓ |  |
| `EFA` | 2263 | -0.485 | 0.259 | 1.388 | -0.150 | 0.0144 | ✓ |  |
| `EEM` | 2263 | -0.902 | 0.153 | 1.293 | -0.175 | 0.0100 | ✓ |  |
| `GLD` | 2263 | -0.046 | 0.650 | 1.496 | -0.153 | 0.0487 | ✓ |  |
| `TLT` | 2263 | -0.964 | -0.082 | 1.017 | -0.205 | 0.0041 | ✗ |  |
| `BTC-USD` | 3400 | -0.306 | 0.930 | 2.044 | -0.230 | 0.1516 | ✓ |  |
| `JPY=X` | 2353 | -0.726 | 0.328 | 1.282 | -0.129 | 0.0186 | ✓ |  |

**9/10 evaluated assets show positive OOS Sharpe** (90%, target ≥ 70%). Mean p50 Sharpe = +0.544.

✅ **Soft gate PASSED** — strategy generalises across the asset universe.

## Reproducibility

Same seed + same input data + same n_trials → identical report. Run `make validate` to regenerate.
