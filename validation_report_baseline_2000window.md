# CPCV Validation Report — Phase 1 + Phase 2.1 XGBoost (SPY 2000-2025, robustness on 'momentum_20d')

_Generated 2026-05-16T19:52:49.637061+00:00 UTC_

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
| `buy_and_hold` | 45 | -0.369 | 0.699 | 1.684 | 0.688 | -0.157 | 0.0527 | 0.699 |
| `momentum_20d` | 45 | 0.157 | 0.767 | 1.352 | 0.714 | -0.088 | 0.0676 | 0.767 |
| `rule_baseline` | 45 | -0.069 | 0.668 | 1.499 | 0.642 | -0.104 | 0.0483 | 0.668 |
| `xgb_v1` | 45 | -0.903 | 0.071 | 1.079 | 0.067 | -0.091 | 0.0040 | 0.071 |
| `xgb_v2` | 45 | -0.946 | 0.150 | 1.063 | 0.071 | -0.064 | 0.0059 | 0.150 |
| `xgb_tuned` | 45 | -0.949 | 0.078 | 1.006 | 0.052 | -0.068 | 0.0042 | 0.078 |
| `meta_equal` | 45 | -0.071 | 0.759 | 1.437 | 0.754 | -0.101 | 0.0643 | 0.759 |
| `meta_ridge` | 45 | -0.463 | 0.672 | 1.510 | 0.613 | -0.011 | 0.0482 | 0.672 |
| `transition_gated` | 45 | -0.130 | 0.576 | 1.148 | 0.581 | -0.096 | 0.0352 | 0.576 |
| `tvtp_msar` | 45 | -0.002 | 0.627 | 1.419 | 0.639 | -0.092 | 0.0405 | 0.627 |
| `hsmm` | 45 | -0.735 | 0.252 | 1.238 | 0.225 | -0.112 | 0.0095 | 0.252 |
| `ms_garch` | 45 | -0.328 | 0.751 | 1.629 | 0.735 | -0.130 | 0.0610 | 0.751 |
| `patchtst` | 45 | -0.665 | 0.347 | 1.694 | 0.381 | -0.044 | 0.0143 | 0.347 |
| `conformal_xgb` | 45 | -0.959 | 0.125 | 1.060 | 0.067 | -0.067 | 0.0052 | 0.125 |
| `fusion` | 45 | -0.003 | 0.733 | 1.664 | 0.770 | -0.111 | 0.0584 | 0.733 |

## Probability of Backtest Overfitting (PBO)

**PBO: 68.89%**

⚠️ Borderline (0.50 ≤ PBO < 0.70). Investigate.

## Triple-barrier label balance

| label | fraction |
|---:|---:|
| -1 | 31.72% |
| +0 | 23.54% |
| +1 | 44.73% |

## Multi-asset robustness (Brief 1.5)

| asset | bars | Sharpe p05 | Sharpe p50 | Sharpe p95 | Max-DD p50 | DSR p-value | OOS positive? | notes |
|---|---:|---:|---:|---:|---:|---:|:-:|---|
| `SPY` | 3521 | 0.151 | 0.749 | 1.354 | -0.115 | 0.0957 | ✓ |  |
| `QQQ` | 3521 | 0.506 | 0.915 | 1.519 | -0.135 | 0.1492 | ✓ |  |
| `DIA` | 3521 | 0.009 | 0.446 | 1.535 | -0.129 | 0.0362 | ✓ |  |
| `IWM` | 3521 | -0.376 | 0.016 | 0.739 | -0.155 | 0.0061 | ✓ |  |
| `EFA` | 3521 | -0.607 | 0.128 | 0.799 | -0.156 | 0.0103 | ✓ |  |
| `EEM` | 3521 | -0.768 | -0.114 | 0.556 | -0.211 | 0.0033 | ✗ |  |
| `GLD` | 3521 | -0.765 | -0.041 | 0.699 | -0.184 | 0.0047 | ✗ |  |
| `TLT` | 3521 | -0.585 | 0.243 | 0.764 | -0.130 | 0.0168 | ✓ |  |
| `BTC-USD` | 3506 | 0.047 | 1.178 | 1.990 | -0.371 | 0.2632 | ✓ |  |
| `JPY=X` | 3646 | -0.913 | 0.274 | 1.227 | -0.106 | 0.0195 | ✓ |  |

**8/10 evaluated assets show positive OOS Sharpe** (80%, target ≥ 70%). Mean p50 Sharpe = +0.379.

✅ **Soft gate PASSED** — strategy generalises across the asset universe.

## Reproducibility

Same seed + same input data + same n_trials → identical report. Run `make validate` to regenerate.


## Regime Diagnostics

### Model selection — BIC/AIC across K

| K | log-likelihood | n_params | AIC | BIC |
|---:|---:|---:|---:|---:|
| 2 | 47403.5 | 13 | -94781.0 | -94693.3 |
| 3 | 49609.85 | 23 | -99173.71 | -99018.55 |
| 4 | 49678.91 | 35 | -99287.82 | -99051.7 |

**Best K by BIC: 4**, **best K by AIC: 4**.


### NBER recession alignment (rule_baseline Bear vs USREC)

- Precision (Bear bars in NBER recession): **7.0%**
- Recall (NBER recession bars labeled Bear): **93.0%**
- F1: **0.13**
- Median lag from recession start to first Bear: **0 days**
- Sample: 43 recession bars, 571 Bear bars, 40 overlap.

### Regime stability (rule_baseline)

- Global flip rate: **0.026** (transitions per bar — `< 0.05` is healthy)
- Dominant regime: **0**

| Regime | Mean duration (bars) | Median | # episodes | % time |
|---:|---:|---:|---:|---:|
| 0 | 73.61 | 46.5 | 36 | 75.3% |
| 1 | 8.33 | 6.0 | 36 | 8.5% |
| 2 | 27.19 | 15.0 | 21 | 16.2% |

### Calibration — Expected Calibration Error

- GMM-HMM mean ECE (vs rule_baseline labels): **0.241**
- Per-class ECE: [0.2478, 0.3176, 0.1584]
(0 = perfectly calibrated; < 0.05 is well-calibrated; > 0.10 indicates systematic over/under-confidence.)

### Cross-model concordance

- Rule vs GMM exact agreement: **62.8%**
- Rule vs TVTP (Bear vs non-Bear): **87.2%**
- GMM vs TVTP (Bear vs non-Bear): **82.3%**
- **Consensus score (≥2 of 3 agree): 92.5%**
- Cohen's κ (rule, GMM): **0.182**

Confusion matrix (row=rule, col=GMM, labels 0/1/2):

| | gmm=0 | gmm=1 | gmm=2 |
|---:|---:|---:|---:|
| **rule=0** | 2079 | 488 | 83 |
| **rule=1** | 252 | 44 | 4 |
| **rule=2** | 79 | 402 | 90 |
