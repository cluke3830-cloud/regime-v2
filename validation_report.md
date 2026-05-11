# CPCV Validation Report — Phase 1 + Phase 2.1 XGBoost (SPY 2015-2024, robustness on 'tvtp_msar')

_Generated 2026-05-11T22:07:06.376744+00:00 UTC_

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
| `buy_and_hold` | 45 | -0.040 | 0.807 | 2.189 | 0.946 | -0.192 | 1.0000 | 0.761 |
| `momentum_20d` | 45 | 0.269 | 0.959 | 1.760 | 1.026 | -0.083 | 1.0000 | 0.984 |
| `rule_baseline` | 45 | -0.361 | 0.792 | 2.044 | 0.793 | -0.105 | 1.0000 | 0.796 |
| `xgb_v1` | 45 | -0.417 | 0.129 | 1.459 | 0.313 | -0.109 | 1.0000 | 0.130 |
| `xgb_v2` | 45 | -0.261 | 0.259 | 1.398 | 0.381 | -0.076 | 1.0000 | 0.152 |
| `xgb_tuned` | 45 | -0.333 | 0.216 | 1.545 | 0.359 | -0.079 | 1.0000 | 0.128 |
| `meta_equal` | 45 | 0.072 | 1.137 | 2.000 | 1.078 | -0.094 | 1.0000 | 1.077 |
| `meta_ridge` | 45 | -0.040 | 0.953 | 1.906 | 0.915 | -0.010 | 1.0000 | 0.769 |
| `transition_gated` | 45 | -0.268 | 0.773 | 1.659 | 0.702 | -0.087 | 1.0000 | 0.742 |
| `tvtp_msar` | 45 | 1.487 | 2.462 | 3.158 | 2.414 | -0.036 | 1.0000 | 2.366 |
| `hsmm` | 45 | 0.150 | 0.752 | 1.922 | 0.882 | -0.072 | 1.0000 | 0.774 |
| `ms_garch` | 45 | 0.101 | 1.038 | 2.111 | 1.043 | -0.112 | 1.0000 | 0.987 |
| `patchtst` | 45 | -0.626 | 0.787 | 1.705 | 0.620 | -0.029 | 1.0000 | 0.124 |
| `conformal_xgb` | 45 | -0.354 | 0.264 | 1.416 | 0.367 | -0.081 | 1.0000 | 0.118 |

## Probability of Backtest Overfitting (PBO)

**PBO: 13.33%**

✅ Strategy selection generalises (PBO < 0.50).

## Triple-barrier label balance

| label | fraction |
|---:|---:|
| -1 | 29.92% |
| +0 | 23.77% |
| +1 | 46.31% |

## Multi-asset robustness (Brief 1.5)

| asset | bars | Sharpe p05 | Sharpe p50 | Sharpe p95 | Max-DD p50 | DSR p-value | OOS positive? | notes |
|---|---:|---:|---:|---:|---:|---:|:-:|---|
| `SPY` | 2263 | 1.487 | 2.462 | 3.158 | -0.036 | 1.0000 | ✓ |  |
| `QQQ` | 2263 | 1.749 | 2.391 | 3.175 | -0.053 | 1.0000 | ✓ |  |
| `DIA` | 2263 | 1.180 | 2.086 | 3.038 | -0.048 | 1.0000 | ✓ |  |
| `IWM` | 2263 | 0.021 | 0.949 | 1.989 | -0.109 | 1.0000 | ✓ |  |
| `EFA` | 2263 | 0.021 | 0.836 | 1.808 | -0.091 | 1.0000 | ✓ |  |
| `EEM` | 2263 | -0.405 | 0.908 | 1.902 | -0.113 | 1.0000 | ✓ |  |
| `GLD` | 2263 | -0.220 | 1.042 | 2.620 | -0.091 | 1.0000 | ✓ |  |
| `TLT` | 2263 | -1.510 | 0.240 | 1.313 | -0.114 | 0.0000 | ✓ |  |
| `BTC-USD` | 3400 | 0.160 | 1.149 | 2.599 | -0.209 | 1.0000 | ✓ |  |
| `JPY=X` | 2353 | 0.049 | 0.937 | 1.529 | -0.029 | 1.0000 | ✓ |  |

**10/10 evaluated assets show positive OOS Sharpe** (100%, target ≥ 70%). Mean p50 Sharpe = +1.300.

✅ **Soft gate PASSED** — strategy generalises across the asset universe.

## Reproducibility

Same seed + same input data + same n_trials → identical report. Run `make validate` to regenerate.
