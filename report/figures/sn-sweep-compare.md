# Sweep Comparison Report

Generated: 2026-04-19T21:26:59

## Best Run Rejection-Curve Comparison
- Sweep A best run: outputs\caco2_wang\sv_dkl\sweeps\2026-04-18_12-54-35\164
- Sweep B best run: outputs\caco2_wang\sv_dkl\sweeps\2026-04-18_18-29-50\176
- Plot path: sweep_compare_2026-04-18_12-54-35_vs_2026-04-18_18-29-50_rejection_curve.pdf

## Top-K Test Metrics

### Sweep: outputs\caco2_wang\sv_dkl\sweeps\2026-04-18_12-54-35
- Setting: with spectral norm
- Top 5 jobs: 164, 103, 79, 141, 41

| Metric | Mean | Std |
|---|---:|---:|
| MAE | 0.380000 | 0.014983 |
| NLL | 0.663307 | 0.021364 |
| PICP_95 | 0.970330 | 0.008333 |
| AURC | 0.326565 | 0.022328 |

### Sweep: outputs\caco2_wang\sv_dkl\sweeps\2026-04-18_18-29-50
- Setting: without spectral norm
- Top 5 jobs: 176, 181, 153, 115, 163

| Metric | Mean | Std |
|---|---:|---:|
| MAE | 0.403800 | 0.026939 |
| NLL | 0.753412 | 0.085778 |
| PICP_95 | 0.915385 | 0.028444 |
| AURC | 0.368790 | 0.018803 |

## Comparison (without_sn - with_sn)
| Metric | Delta Mean | Delta Std |
|---|---:|---:|
| MAE | +0.023800 | +0.011955 |
| NLL | +0.090105 | +0.064414 |
| PICP_95 | -0.054945 | +0.020112 |
| AURC | +0.042225 | -0.003525 |
