# Results of the shipped run

Primary model: `lgbm_history_weighted` (version `lgbm_history_weighted-20261031T043636Z`), interval method `cqr`. Generated 2026-10-31T04:37:20.269367+00:00. Total training time 231.3 s.

## Data and partitions

| Partition | Events | Rows | Safe | Elevated | Critical (distance) | Critical (Pc) |
|---|---|---|---|---|---|---|
| dev | 7893 | 97282 | 4719 | 2089 | 993 | 92 |
| policy_cal | 1973 | 24339 | 1180 | 522 | 248 | 23 |
| conformal_cal | 1316 | 16531 | 787 | 348 | 165 | 16 |
| eval | 1972 | 24482 | 1179 | 522 | 248 | 23 |

Training file: 162634 rows, 13154 events, 41.3% of rows at the −30 floor, 716 rows with risk ≥ −4. Test file: 24484 rows, 2167 events, rows stop at 2.00 days before TCA.

## Latest CDM per event (evaluation partition)

| Model | Critical recall | Critical precision | Critical F2 | Elev+ F2 | Macro-F1 | Pc-only Crit recall | Pc-only Crit precision | MAE | MAE non-floor | MAE high-risk | Coverage | Median width | Auto-clear | True Crit → auto-clear |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_median | 1.000 | 1.000 | 1.000 | 0.957 | 0.985 | undefined | undefined | 10.45 | 7.89 | 12.69 | 0.982 | 24.4 | 0.0% | 0 |
| baseline_linear | 1.000 | 1.000 | 1.000 | 0.957 | 0.985 | undefined | undefined | 9.70 | 9.45 | 10.95 | 0.950 | 24.0 | 45.4% | 0 |
| lgbm_single_cdm | 1.000 | 0.985 | 0.997 | 0.974 | 0.985 | undefined | 0.000 | 2.44 | 4.22 | 2.45 | 0.972 | 14.9 | 66.6% | 0 |
| lgbm_physics | 1.000 | 0.992 | 0.998 | 0.971 | 0.985 | undefined | 0.000 | 2.40 | 4.24 | 2.58 | 0.967 | 14.6 | 68.1% | 0 |
| lgbm_history | 1.000 | 0.985 | 0.997 | 0.981 | 0.987 | undefined | 0.000 | 2.10 | 3.61 | 2.38 | 0.973 | 12.8 | 68.0% | 0 |
| lgbm_history_weighted | 1.000 | 1.000 | 1.000 | 0.984 | 0.992 | undefined | 0.000 | 2.29 | 3.30 | 1.16 | 0.973 | 12.8 | 68.1% | 0 |
| lgbm_history_l2 | 1.000 | 0.956 | 0.991 | 0.976 | 0.979 | undefined | 0.000 | 2.18 | 3.85 | 2.72 | 0.973 | 12.8 | 68.1% | 0 |
| lgbm_history_l2_monotone | 1.000 | 0.949 | 0.989 | 0.975 | 0.977 | undefined | 0.000 | 2.21 | 3.91 | 2.82 | 0.973 | 12.8 | 68.1% | 0 |

View size: 1972 rows, 1972 events, 131 true Critical rows, 0 Pc-only true positives for the primary model. Nominal interval coverage 0.9.
Event-bootstrap 95% intervals for the primary model: Critical recall [1.000, 1.000], Critical precision [1.000, 1.000], MAE [2.127, 2.450], coverage [0.965, 0.979].

Undefined metrics in this view (no positives in the subset): tier [], Pc-only ['critical_recall', 'critical_f2'].

## All CDMs (evaluation partition)

| Model | Critical recall | Critical precision | Critical F2 | Elev+ F2 | Macro-F1 | Pc-only Crit recall | Pc-only Crit precision | MAE | MAE non-floor | MAE high-risk | Coverage | Median width | Auto-clear | True Crit → auto-clear |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_median | 0.966 | 1.000 | 0.973 | 0.922 | 0.964 | 0.000 | undefined | 9.50 | 7.84 | 13.00 | 0.950 | 24.4 | 0.0% | 0 |
| baseline_linear | 0.966 | 1.000 | 0.973 | 0.922 | 0.964 | 0.000 | undefined | 9.22 | 8.59 | 10.70 | 0.969 | 24.8 | 34.2% | 0 |
| lgbm_single_cdm | 0.980 | 0.974 | 0.979 | 0.955 | 0.971 | 0.451 | 0.485 | 3.54 | 3.81 | 2.13 | 0.970 | 20.7 | 59.9% | 0 |
| lgbm_physics | 0.977 | 0.985 | 0.979 | 0.956 | 0.973 | 0.507 | 0.621 | 3.30 | 3.61 | 1.86 | 0.969 | 20.6 | 62.1% | 0 |
| lgbm_history | 0.979 | 0.968 | 0.976 | 0.956 | 0.969 | 0.423 | 0.390 | 2.92 | 3.29 | 1.84 | 0.967 | 20.2 | 62.8% | 0 |
| lgbm_history_weighted | 0.977 | 0.989 | 0.979 | 0.970 | 0.976 | 0.465 | 0.629 | 3.03 | 3.03 | 0.83 | 0.967 | 20.2 | 62.8% | 0 |
| lgbm_history_l2 | 0.980 | 0.903 | 0.963 | 0.955 | 0.953 | 0.423 | 0.203 | 2.94 | 3.34 | 2.15 | 0.967 | 20.2 | 62.7% | 0 |
| lgbm_history_l2_monotone | 0.981 | 0.905 | 0.965 | 0.954 | 0.953 | 0.493 | 0.234 | 2.95 | 3.34 | 2.08 | 0.967 | 20.2 | 62.7% | 0 |

View size: 24482 rows, 1972 events, 1677 true Critical rows, 66 Pc-only true positives for the primary model. Nominal interval coverage 0.9.
Event-bootstrap 95% intervals for the primary model: Critical recall [0.962, 0.992], Critical precision [0.979, 0.997], MAE [2.907, 3.162], coverage [0.961, 0.972].

## Latest CDM per event, restricted to ≥ 2 days before TCA (matches the test horizon)

| Model | Critical recall | Critical precision | Critical F2 | Elev+ F2 | Macro-F1 | Pc-only Crit recall | Pc-only Crit precision | MAE | MAE non-floor | MAE high-risk | Coverage | Median width | Auto-clear | True Crit → auto-clear |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_median | 0.991 | 1.000 | 0.993 | 0.939 | 0.976 | 0.000 | undefined | 9.61 | 7.85 | 12.90 | 0.962 | 24.4 | 0.0% | 0 |
| baseline_linear | 0.991 | 1.000 | 0.993 | 0.939 | 0.976 | 0.000 | undefined | 9.40 | 8.86 | 10.74 | 0.962 | 24.5 | 38.3% | 0 |
| lgbm_single_cdm | 0.991 | 0.967 | 0.986 | 0.962 | 0.974 | 0.000 | 0.000 | 3.79 | 4.89 | 3.36 | 0.964 | 20.3 | 64.5% | 0 |
| lgbm_physics | 0.991 | 0.983 | 0.990 | 0.959 | 0.976 | 0.333 | 0.400 | 3.51 | 4.35 | 2.55 | 0.963 | 20.1 | 66.4% | 0 |
| lgbm_history | 0.991 | 0.983 | 0.990 | 0.970 | 0.982 | 0.167 | 0.143 | 3.07 | 3.99 | 2.67 | 0.961 | 19.2 | 66.5% | 0 |
| lgbm_history_weighted | 0.991 | 0.991 | 0.991 | 0.975 | 0.985 | 0.333 | 0.400 | 3.08 | 3.50 | 1.37 | 0.961 | 19.2 | 66.6% | 0 |
| lgbm_history_l2 | 0.991 | 0.921 | 0.976 | 0.966 | 0.965 | 0.333 | 0.105 | 3.13 | 3.82 | 2.33 | 0.962 | 19.2 | 66.5% | 0 |
| lgbm_history_l2_monotone | 1.000 | 0.921 | 0.983 | 0.969 | 0.968 | 0.500 | 0.158 | 3.13 | 3.81 | 2.30 | 0.961 | 19.2 | 66.6% | 0 |

View size: 1778 rows, 1778 events, 117 true Critical rows, 2 Pc-only true positives for the primary model. Nominal interval coverage 0.9.
Event-bootstrap 95% intervals for the primary model: Critical recall [0.970, 1.000], Critical precision [0.973, 1.000], MAE [2.890, 3.264], coverage [0.952, 0.969].

## Triage mix of the primary model (evaluation partition)

| View | Maneuver planning review | Monitor | Human review | Auto-clear candidate | Insufficient data |
|---|---|---|---|---|---|
| latest_cdm | 131 | 335 | 163 | 1343 | 0 |
| row_level | 1656 | 5133 | 2317 | 15376 | 0 |

## Interval method comparison (primary model, evaluation partition)

| Method | View | Coverage | Median width | Auto-clear | Human review | True Crit → auto-clear |
|---|---|---|---|---|---|---|
| residual | row_level | 0.979 | 24.1 | 30.7% | 56.2% | 0 |
| residual | latest_cdm | 0.983 | 18.2 | 48.2% | 35.5% | 0 |
| cqr | row_level | 0.967 | 20.2 | 62.8% | 9.5% | 0 |
| cqr | latest_cdm | 0.973 | 12.8 | 68.1% | 8.3% | 0 |

## Optional classifier P(risk ≥ −4)

Sigmoid-calibrated on the policy-calibration partition (106 positives); disagreement threshold tuned there to 0.15 (policy-cal F2 0.847, optimistic because the same partition fitted the sigmoid). Evaluation partition, row level: PR-AUC 0.663, Brier 0.0026, positives 142.

Reliability (policy-cal): bin | mean predicted | observed fraction | count

| bin | mean pred | observed | n |
|---|---|---|---|
| 0.0–0.1 | 0.000 | 0.000 | 24145 |
| 0.1–0.2 | 0.168 | 0.154 | 13 |
| 0.2–0.3 | 0.246 | 0.400 | 10 |
| 0.3–0.4 | 0.343 | 0.125 | 8 |
| 0.4–0.5 | 0.475 | 0.385 | 13 |
| 0.5–0.6 | 0.565 | 0.492 | 61 |
| 0.6–0.7 | 0.631 | 0.713 | 87 |
| 0.7–0.8 | 0.795 | 0.000 | 1 |
| 0.8–0.9 | 0.825 | 0.000 | 1 |

## External check: delivered test file (labelled, ≥ 2 days before TCA)

| Model | View | Critical recall | Critical precision | Pc-only Crit recall | Pc-only Crit precision | MAE | MAE high-risk | Coverage | Auto-clear | True Crit → auto-clear |
|---|---|---|---|---|---|---|---|---|---|---|
| lgbm_history_weighted | latest_cdm | 0.978 | 0.995 | 0.609 | 0.700 | 3.01 | 1.47 | 0.964 | 62.7% | 0 |
| lgbm_history_weighted | row_level | 0.972 | 0.993 | 0.571 | 0.704 | 3.27 | 0.67 | 0.973 | 53.9% | 0 |
| baseline_median | latest_cdm | 0.962 | 1.000 | 0.000 | undefined | 10.06 | 13.13 | 0.937 | 0.0% | 0 |
| baseline_median | row_level | 0.959 | 1.000 | 0.000 | undefined | 9.27 | 12.99 | 0.903 | 0.0% | 0 |
| baseline_linear | latest_cdm | 0.962 | 1.000 | 0.000 | undefined | 9.08 | 9.98 | 0.975 | 31.4% | 0 |
| baseline_linear | row_level | 0.959 | 1.000 | 0.000 | undefined | 8.83 | 10.54 | 0.985 | 27.3% | 0 |

## Model selection trace

Grouped CV (5 folds) over 10 sampled configurations; rule: 0.5*rank(MAE on non-floor rows) + 0.5*rank(Pc-only Critical F2), lower is better. Best: `{"objective": "l1", "learning_rate": 0.04572128610553908, "num_leaves": 23, "min_child_samples": 69, "feature_fraction": 0.8900771875791135, "lambda_l2": 10.0}`, mean best iteration 999, OOF MAE non-floor 3.263, OOF Pc-only Critical F2 0.452.

Candidate selection on the policy-calibration partition:

| Candidate | MAE non-floor | MAE high-risk | Pc-only Crit recall | Pc-only Crit precision | Pc-only Crit F2 | Selection score |
|---|---|---|---|---|---|---|
| lgbm_single_cdm | 3.829 | 2.288 | 0.528 | 0.448 | 0.510 | 4.00 |
| lgbm_physics | 3.534 | 2.006 | 0.415 | 0.518 | 0.432 | 4.50 |
| lgbm_history | 3.208 | 1.864 | 0.462 | 0.422 | 0.454 | 2.50 |
| lgbm_history_weighted | 2.973 | 0.942 | 0.594 | 0.787 | 0.625 | 1.00 |
| lgbm_history_l2 | 3.225 | 2.207 | 0.453 | 0.187 | 0.352 | 4.50 |
| lgbm_history_l2_monotone | 3.246 | 2.168 | 0.481 | 0.198 | 0.374 | 4.50 |

## Illustrative cases (primary model, evaluation partition)

### Missed Critical (true Critical, predicted lower tier): 3 shown

| row_id | time to TCA (d) | miss (m) | true risk | pred risk | interval | tier true → pred | action | reasons |
|---|---|---|---|---|---|---|---|---|
| kelvins_train:128469 | 2.52 | 2088 | -3.70 | -8.13 | [-30.0, -3.5] | Critical → Elevated | Human review | DISTANCE_ELEVATED, UPPER_BOUND_CRITICAL, MODEL_DISAGREEMENT |
| kelvins_train:133157 | 0.79 | 1018 | -3.59 | -6.72 | [-30.0, -2.9] | Critical → Elevated | Human review | DISTANCE_ELEVATED, UPPER_BOUND_CRITICAL, MODEL_DISAGREEMENT |
| kelvins_train:133155 | 1.37 | 1688 | -3.63 | -6.69 | [-30.0, -3.4] | Critical → Elevated | Human review | DISTANCE_ELEVATED, UPPER_BOUND_CRITICAL, MODEL_DISAGREEMENT |

### False alarm (predicted Critical, true lower tier): 3 shown

| row_id | time to TCA (d) | miss (m) | true risk | pred risk | interval | tier true → pred | action | reasons |
|---|---|---|---|---|---|---|---|---|
| kelvins_train:87255 | 3.41 | 1480 | -11.69 | -3.78 | [-30.0, -3.1] | Elevated → Critical | Maneuver planning review | PC_CRITICAL, UPPER_BOUND_CRITICAL |
| kelvins_train:87254 | 3.69 | 1364 | -5.56 | -3.99 | [-30.0, -3.2] | Elevated → Critical | Maneuver planning review | PC_CRITICAL, UPPER_BOUND_CRITICAL |
| kelvins_train:14177 | 2.38 | 3502 | -5.17 | -3.82 | [-30.0, -3.0] | Elevated → Critical | Maneuver planning review | PC_CRITICAL, UPPER_BOUND_CRITICAL |

### Routed to human review with the widest intervals: 3 shown

| row_id | time to TCA (d) | miss (m) | true risk | pred risk | interval | tier true → pred | action | reasons |
|---|---|---|---|---|---|---|---|---|
| kelvins_train:133153 | 1.91 | 1280 | -3.51 | -4.76 | [-30.0, -2.6] | Critical → Elevated | Human review | DISTANCE_ELEVATED, PC_ELEVATED, UPPER_BOUND_CRITICAL, MODEL_DISAGREEMENT |
| kelvins_train:62357 | 3.03 | 2666 | -3.82 | -4.11 | [-30.0, -2.6] | Critical → Elevated | Human review | DISTANCE_ELEVATED, PC_ELEVATED, UPPER_BOUND_CRITICAL, MODEL_DISAGREEMENT |
| kelvins_train:116585 | 5.72 | 5609 | -3.61 | -4.33 | [-30.0, -2.6] | Critical → Elevated | Human review | PC_ELEVATED, UPPER_BOUND_CRITICAL, MODEL_DISAGREEMENT |

### Invalid inputs: 0 shown

None in this partition.

## Batch inference

Delivered test file: 24484 rows in 18.542 s including explanations (breakdown: {"features": 0.1106, "point_model": 0.2018, "policy": 0.1129, "explanations": 17.6877, "total": 18.4963}). Status counts {"valid": 22499, "warning": 1985}; tier counts {"Safe": 15814, "Elevated": 6490, "Critical": 2180}; triage counts {"Auto-clear candidate": 13188, "Monitor": 6029, "Human review": 3087, "Maneuver planning review": 2180}.

## Notes

- Intervals are conformalized quantile regression intervals (0.05/0.95 LightGBM quantile heads, crossing-corrected, shifted by the finite-sample quantile of per-event max scores) around the dataset risk target (log10 Pc); they describe uncertainty of the dataset label, not every physical uncertainty of an encounter.
- Coverage is reported empirically on evaluation events; no universal or subgroup guarantee is claimed.
- Pc-only metrics isolate model skill from the deterministic distance rule.
- Zero observed misses in a subset would not imply zero operational risk; see event-bootstrap intervals.
