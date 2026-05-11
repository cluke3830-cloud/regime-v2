# CPCV Validation Report — Phase 1 + Phase 2.1 XGBoost (SPY 2015-2024, robustness on 'meta_equal')

_Generated 2026-05-11T19:17:48.518061+00:00 UTC_

## CPCV configuration

- n_splits: **10**
- n_test_groups: **2**
- embargo: **1.00%** of sample
- paths per strategy: **45**
- seed: **42**
- n_trials (DSR deflation): **160**

## Per-strategy results

| strategy | paths | Sharpe p05 | Sharpe p50 | Sharpe p95 | Sharpe mean | Max-DD p50 | DSR p-value | DSR Sharpe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `flat` | 45 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | — | — |
| `buy_and_hold` | 45 | -0.040 | 0.807 | 2.189 | 0.946 | -0.192 | 1.0000 | 0.761 |
| `momentum_20d` | 45 | 0.269 | 0.959 | 1.760 | 1.026 | -0.083 | 1.0000 | 0.984 |
| `rule_baseline` | 45 | -0.361 | 0.792 | 2.044 | 0.793 | -0.105 | 1.0000 | 0.796 |
| `xgb_v1` | 45 | -0.417 | 0.129 | 1.459 | 0.313 | -0.109 | 1.0000 | 0.130 |
| `xgb_v2` | 45 | -0.261 | 0.259 | 1.398 | 0.381 | -0.076 | 1.0000 | 0.152 |
| `xgb_tuned` | 45 | -0.333 | 0.216 | 1.545 | 0.359 | -0.079 | 1.0000 | 0.128 |
| `meta_equal` | 45 | 0.072 | 1.137 | 2.000 | 1.078 | -0.094 | 1.0000 | 1.077 |
| `meta_ridge` | 45 | -0.040 | 0.953 | 1.906 | 0.915 | -0.010 | 1.0000 | 0.769 |
| `transition_gated` | 45 | -0.268 | 0.773 | 1.659 | 0.702 | -0.087 | 1.0000 | 0.742 |

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
| `SPY` | 2263 | 0.072 | 1.137 | 2.000 | -0.094 | 1.0000 | ✓ |  |
| `QQQ` | 2263 | 0.169 | 1.143 | 2.065 | -0.122 | 1.0000 | ✓ |  |
| `DIA` | 2263 | -0.378 | 0.886 | 2.227 | -0.102 | 1.0000 | ✓ |  |
| `IWM` | 2263 | -0.620 | 0.367 | 1.420 | -0.118 | 1.0000 | ✓ |  |
| `EFA` | 2263 | -0.756 | 0.394 | 1.592 | -0.105 | 1.0000 | ✓ |  |
| `EEM` | 2263 | -1.038 | 0.049 | 1.226 | -0.129 | 1.0000 | ✓ |  |
| `GLD` | 2263 | -0.053 | 0.472 | 1.192 | -0.115 | 1.0000 | ✓ |  |
| `TLT` | 2263 | -0.943 | -0.019 | 0.975 | -0.122 | 0.0002 | ✗ |  |
| `BTC-USD` | 3400 | 0.208 | 1.262 | 1.965 | -0.346 | 1.0000 | ✓ |  |
| `JPY=X` | 2353 | -0.672 | 0.411 | 1.247 | -0.073 | 1.0000 | ✓ |  |

**9/10 evaluated assets show positive OOS Sharpe** (90%, target ≥ 70%). Mean p50 Sharpe = +0.610.

✅ **Soft gate PASSED** — strategy generalises across the asset universe.

## Reproducibility

Same seed + same input data + same n_trials → identical report. Run `make validate` to regenerate.
