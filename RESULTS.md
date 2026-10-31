# Results of the shipped run

Task: forecast each event's **final** tier (Safe / Elevated / Critical) from CDMs at >= 2 days before TCA. Primary model: `lgbm_multiclass` (version `lgbm_multiclass-20261031T062033Z`). Generated 2026-10-31T06:21:26.853989+00:00. Total training time 76.0 s.

Holdout: 2389 events (1779 real forecasts, 610 trivial), split seed 42, holdout fraction 0.200, split hash `d18e932157a7c887`. Final-tier counts on the holdout: {"Safe": 1790, "Elevated": 432, "Critical": 167}.

Subsets: `all_usable` = every holdout event; `trivial` = the event's final CDM was itself at >= 2 days (the answer is visible in the input); `real_forecast` = everything else and the headline subset.

## Headline: real-forecast holdout

Headline metric: **Critical recall on real_forecast events**, read together with Critical precision / F2. `carry_forward_risk` is the official naive baseline (the latest known tier carried forward).

| Model | Events | Critical recall | Critical precision | Critical F2 | Elev+ F2 | Macro-F1 | Brier | Log loss | Auto-clear | Final Crit auto-cleared | Watch share | Escalation recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carry_forward_risk | 1779 | 0.797 | 0.764 | 0.790 | 0.918 | 0.858 | 0.142 | 1.957 | 0.0% | 0 | 0.0% | 0.000 |
| **lgbm_multiclass** | 1779 | 0.907 | 0.677 | 0.849 | 0.923 | 0.841 | 0.104 | 0.202 | 16.4% | 0 | 4.9% | 0.792 |
| carry_forward_mre | 1779 | 0.797 | 0.686 | 0.772 | 0.813 | 0.700 | 0.527 | 7.284 | 0.0% | 0 | 0.0% | 0.000 |
| majority | 1779 | 0.000 | undefined | 0.000 | 0.000 | 0.284 | 0.512 | 7.067 | 0.0% | 0 | 0.0% | 0.000 |
| logistic | 1779 | 0.890 | 0.640 | 0.825 | 0.895 | 0.813 | 0.143 | 0.277 | 30.4% | 2 | 5.4% | 0.792 |
| lgbm_ordinal | 1779 | 0.890 | 0.709 | 0.847 | 0.922 | 0.849 | 0.105 | 0.202 | 18.2% | 1 | 4.9% | 0.750 |

`lgbm_multiclass` beats `carry_forward_risk` on Critical F2 (real_forecast): 0.849 vs 0.790.

### All usable holdout events

| Model | Events | Critical recall | Critical precision | Critical F2 | Elev+ F2 | Macro-F1 | Brier | Log loss | Auto-clear | Final Crit auto-cleared | Watch share | Escalation recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carry_forward_risk | 2389 | 0.856 | 0.831 | 0.851 | 0.938 | 0.894 | 0.105 | 1.457 | 0.0% | 0 | 0.0% | 0.000 |
| **lgbm_multiclass** | 2389 | 0.934 | 0.696 | 0.874 | 0.936 | 0.853 | 0.093 | 0.183 | 14.1% | 0 | 5.0% | 0.792 |
| carry_forward_mre | 2389 | 0.856 | 0.769 | 0.837 | 0.835 | 0.745 | 0.454 | 6.269 | 0.0% | 0 | 0.0% | 0.000 |
| majority | 2389 | 0.000 | undefined | 0.000 | 0.000 | 0.286 | 0.501 | 6.928 | 0.0% | 0 | 0.0% | 0.000 |
| logistic | 2389 | 0.922 | 0.658 | 0.854 | 0.883 | 0.812 | 0.143 | 0.271 | 32.5% | 2 | 5.0% | 0.792 |
| lgbm_ordinal | 2389 | 0.922 | 0.726 | 0.875 | 0.930 | 0.857 | 0.096 | 0.186 | 16.8% | 1 | 5.0% | 0.750 |

### Trivial events (answer visible in the input)

| Model | Events | Critical recall | Critical precision | Critical F2 | Elev+ F2 | Macro-F1 | Brier | Log loss | Auto-clear | Final Crit auto-cleared | Watch share | Escalation recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carry_forward_risk | 610 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | -0.000 | 0.0% | 0 | 0.0% | undefined |
| **lgbm_multiclass** | 610 | 1.000 | 0.742 | 0.935 | 0.976 | 0.887 | 0.059 | 0.124 | 7.2% | 0 | 5.2% | undefined |
| carry_forward_mre | 610 | 1.000 | 1.000 | 1.000 | 0.908 | 0.879 | 0.239 | 3.307 | 0.0% | 0 | 0.0% | undefined |
| majority | 610 | 0.000 | undefined | 0.000 | 0.000 | 0.289 | 0.472 | 6.523 | 0.0% | 0 | 0.0% | undefined |
| logistic | 610 | 1.000 | 0.700 | 0.921 | 0.841 | 0.794 | 0.141 | 0.252 | 38.9% | 0 | 3.9% | undefined |
| lgbm_ordinal | 610 | 1.000 | 0.766 | 0.942 | 0.953 | 0.875 | 0.072 | 0.140 | 12.6% | 0 | 5.1% | undefined |

## Confusion matrices

### `carry_forward_risk` on real_forecast (1779 events)

Counts (rows = true final tier, columns = predicted):

| true \ pred | Safe | Elevated | Critical |
|---|---|---|---|
| Safe | 1281 | 40 | 3 |
| Elevated | 33 | 278 | 26 |
| Critical | 3 | 21 | 94 |

Row-normalised (share of each true tier):

| true \ pred | Safe | Elevated | Critical |
|---|---|---|---|
| Safe | 96.8% | 3.0% | 0.2% |
| Elevated | 9.8% | 82.5% | 7.7% |
| Critical | 2.5% | 17.8% | 79.7% |

### `carry_forward_risk` on all_usable (2389 events)

Counts (rows = true final tier, columns = predicted):

| true \ pred | Safe | Elevated | Critical |
|---|---|---|---|
| Safe | 1747 | 40 | 3 |
| Elevated | 33 | 373 | 26 |
| Critical | 3 | 21 | 143 |

Row-normalised (share of each true tier):

| true \ pred | Safe | Elevated | Critical |
|---|---|---|---|
| Safe | 97.6% | 2.2% | 0.2% |
| Elevated | 7.6% | 86.3% | 6.0% |
| Critical | 1.8% | 12.6% | 85.6% |

### `lgbm_multiclass` on real_forecast (1779 events)

Counts (rows = true final tier, columns = predicted):

| true \ pred | Safe | Elevated | Critical |
|---|---|---|---|
| Safe | 1249 | 69 | 6 |
| Elevated | 23 | 269 | 45 |
| Critical | 3 | 8 | 107 |

Row-normalised (share of each true tier):

| true \ pred | Safe | Elevated | Critical |
|---|---|---|---|
| Safe | 94.3% | 5.2% | 0.5% |
| Elevated | 6.8% | 79.8% | 13.4% |
| Critical | 2.5% | 6.8% | 90.7% |

### `lgbm_multiclass` on all_usable (2389 events)

Counts (rows = true final tier, columns = predicted):

| true \ pred | Safe | Elevated | Critical |
|---|---|---|---|
| Safe | 1697 | 86 | 7 |
| Elevated | 23 | 348 | 61 |
| Critical | 3 | 8 | 156 |

Row-normalised (share of each true tier):

| true \ pred | Safe | Elevated | Critical |
|---|---|---|---|
| Safe | 94.8% | 4.8% | 0.4% |
| Elevated | 5.3% | 80.6% | 14.1% |
| Critical | 1.8% | 4.8% | 93.4% |

## Escalation watch

Among events that are not Critical at the forecast time: how many end Critical, how many the forecast flags (predicted Critical or placed on the escalation watch list), and how many of those flags are hits. Among events that are Critical now: how many de-escalate and how the forecast handles them.

| Model | Subset | Not-Critical now | End Critical | Flagged | Caught | Escalation recall | Flag precision | Watch size | Watch hits | Critical now | De-escalate | Pred. de-esc. | Correct | False de-esc. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carry_forward_risk | real_forecast | 1656 | 24 | 0 | 0 | 0.000 | undefined | 0 | 0 | 123 | 29 | 0 | 0 | 0 |
| carry_forward_risk | all_usable | 2217 | 24 | 0 | 0 | 0.000 | undefined | 0 | 0 | 172 | 29 | 0 | 0 | 0 |
| lgbm_multiclass | real_forecast | 1656 | 24 | 123 | 19 | 0.792 | 0.154 | 87 | 6 | 123 | 29 | 1 | 1 | 0 |
| lgbm_multiclass | all_usable | 2217 | 24 | 172 | 19 | 0.792 | 0.110 | 119 | 6 | 172 | 29 | 1 | 1 | 0 |
| carry_forward_mre | real_forecast | 1656 | 24 | 14 | 0 | 0.000 | 0.000 | 0 | 0 | 123 | 29 | 0 | 0 | 0 |
| carry_forward_mre | all_usable | 2217 | 24 | 14 | 0 | 0.000 | 0.000 | 0 | 0 | 172 | 29 | 0 | 0 | 0 |
| majority | real_forecast | 1656 | 24 | 0 | 0 | 0.000 | undefined | 0 | 0 | 123 | 29 | 123 | 29 | 94 |
| majority | all_usable | 2217 | 24 | 0 | 0 | 0.000 | undefined | 0 | 0 | 172 | 29 | 172 | 29 | 143 |
| logistic | real_forecast | 1656 | 24 | 137 | 19 | 0.792 | 0.139 | 94 | 7 | 123 | 29 | 2 | 1 | 1 |
| logistic | all_usable | 2217 | 24 | 182 | 19 | 0.792 | 0.104 | 118 | 7 | 172 | 29 | 2 | 1 | 1 |
| lgbm_ordinal | real_forecast | 1656 | 24 | 113 | 18 | 0.750 | 0.159 | 87 | 7 | 123 | 29 | 1 | 1 | 0 |
| lgbm_ordinal | all_usable | 2217 | 24 | 159 | 18 | 0.750 | 0.113 | 118 | 7 | 172 | 29 | 1 | 1 | 0 |

## Auto-clear and triage mix (primary model)

| Subset | Events | Auto-clear share | Final-Critical auto-cleared | Final Critical | Watch share | Final-Critical in watch | Maneuver review | Human review | Monitor |
|---|---|---|---|---|---|---|---|---|---|
| real_forecast | 1779 | 16.4% | 0 | 118 | 4.9% | 6 | 8.9% | 54.6% | 15.1% |
| all_usable | 2389 | 14.1% | 0 | 167 | 5.0% | 6 | 9.4% | 57.3% | 14.2% |
| trivial | 610 | 7.2% | 0 | 49 | 5.2% | 0 | 10.8% | 65.2% | 11.5% |

No event that ended Critical was auto-cleared on the holdout.

Action counts (primary model, all usable holdout events):

| Action | Events |
|---|---|
| Auto-clear candidate | 336 |
| Escalation watch | 120 |
| Human review | 1370 |
| Maneuver planning review | 224 |
| Monitor | 339 |

## Calibration

| Model | Subset | n | Brier (multiclass) | Log loss | Brier (Critical) | Critical PR-AUC | Critical ROC-AUC | ECE (Critical) |
|---|---|---|---|---|---|---|---|---|
| carry_forward_risk | real_forecast | 1779 | 0.1417 | 1.9570 | 0.0298 | 0.622 | 0.890 | 0.0298 |
| carry_forward_risk | all_usable | 2389 | 0.1055 | 1.4573 | 0.0222 | 0.722 | 0.922 | 0.0222 |
| lgbm_multiclass | real_forecast | 1779 | 0.1043 | 0.2025 | 0.0191 | 0.888 | 0.980 | 0.0129 |
| lgbm_multiclass | all_usable | 2389 | 0.0926 | 0.1826 | 0.0170 | 0.909 | 0.985 | 0.0118 |
| carry_forward_mre | real_forecast | 1779 | 0.5273 | 7.2844 | 0.0377 | 0.560 | 0.885 | 0.0377 |
| carry_forward_mre | all_usable | 2389 | 0.4537 | 6.2687 | 0.0280 | 0.668 | 0.918 | 0.0280 |
| majority | real_forecast | 1779 | 0.5115 | 7.0670 | 0.0663 | 0.066 | 0.500 | 0.0663 |
| majority | all_usable | 2389 | 0.5015 | 6.9280 | 0.0699 | 0.070 | 0.500 | 0.0699 |
| logistic | real_forecast | 1779 | 0.1434 | 0.2773 | 0.0247 | 0.790 | 0.973 | 0.0159 |
| logistic | all_usable | 2389 | 0.1428 | 0.2710 | 0.0232 | 0.837 | 0.980 | 0.0155 |
| lgbm_ordinal | real_forecast | 1779 | 0.1048 | 0.2016 | 0.0195 | 0.885 | 0.975 | 0.0121 |
| lgbm_ordinal | all_usable | 2389 | 0.0963 | 0.1860 | 0.0176 | 0.904 | 0.982 | 0.0109 |

Reliability of P(Critical) for `lgbm_multiclass` on real_forecast holdout events (10 equal-width bins):

| Bin | Mean predicted | Observed Critical share | Events |
|---|---|---|---|
| 0.0-0.1 | 0.006 | 0.004 | 1611 |
| 0.1-0.2 | 0.153 | 0.400 | 10 |
| 0.2-0.3 | 0.241 | 0.200 | 10 |
| 0.3-0.4 | 0.334 | 0.273 | 11 |
| 0.4-0.5 | 0.442 | 0.167 | 12 |
| 0.5-0.6 | 0.551 | 0.143 | 7 |
| 0.6-0.7 | 0.629 | 0.667 | 3 |
| 0.7-0.8 | 0.764 | 0.556 | 9 |
| 0.8-0.9 | 0.870 | 0.727 | 11 |
| 0.9-1.0 | 0.966 | 0.884 | 95 |

Calibration method: `isotonic` fitted on dev out-of-fold predictions (dev OOF Brier undefined, log loss undefined).

## Bootstrap confidence intervals

Point estimate followed by the 95 % event-bootstrap interval (2.5 / 97.5 percentiles).

| Model | Subset | Resamples | Critical recall | Critical precision | Macro-F1 | Auto-clear share |
|---|---|---|---|---|---|---|
| lgbm_multiclass | real_forecast | 100 | 0.907 [0.860, 0.946] | 0.677 [0.617, 0.743] | 0.841 [0.819, 0.864] | 16.4% [0.143, 0.187] |
| lgbm_multiclass | all_usable | 100 | 0.934 [0.900, 0.976] | 0.696 [0.637, 0.748] | 0.853 [0.834, 0.869] | 14.1% [0.129, 0.152] |
| carry_forward_risk | real_forecast | 100 | 0.797 [0.734, 0.878] | 0.764 [0.687, 0.825] | 0.858 [0.833, 0.883] | 0.0% [0.000, 0.000] |
| carry_forward_risk | all_usable | 100 | 0.856 [0.803, 0.910] | 0.831 [0.773, 0.885] | 0.894 [0.873, 0.913] | 0.0% [0.000, 0.000] |

## Breakdowns (primary model)

### Lead time of the latest snapshot, real_forecast

| Days to TCA | Events | Final Critical | Predicted Critical | Critical recall | Critical precision | Critical F2 | Macro-F1 | Accuracy |
|---|---|---|---|---|---|---|---|---|
| [2,3) | 1701 | 113 | 154 | 0.920 | 0.675 | 0.858 | 0.849 | 0.918 |
| [3,4) | 30 | 3 | 2 | 0.667 | 1.000 | 0.714 | 0.560 | 0.800 |
| [4,5) | 18 | 1 | 0 | 0.000 | undefined | 0.000 | 0.323 | 0.889 |
| [5,inf) | 30 | 1 | 2 | 1.000 | 0.500 | 0.833 | 0.523 | 0.800 |

### Object type, real_forecast

| Object type | Events | Final Critical | Predicted Critical | Critical recall | Critical precision | Critical F2 | Macro-F1 | Accuracy |
|---|---|---|---|---|---|---|---|---|
| DEBRIS | 991 | 74 | 103 | 0.919 | 0.660 | 0.852 | 0.847 | 0.910 |
| PAYLOAD | 224 | 9 | 9 | 0.778 | 0.778 | 0.778 | 0.850 | 0.951 |
| ROCKET BODY | 20 | 5 | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| UNKNOWN | 543 | 30 | 41 | 0.900 | 0.659 | 0.839 | 0.810 | 0.901 |
| other (TBA) | 1 | 0 | 0 | undefined | undefined | undefined | 1.000 | 1.000 |

### Lead time of the latest snapshot, all_usable

| Days to TCA | Events | Final Critical | Predicted Critical | Critical recall | Critical precision | Critical F2 | Macro-F1 | Accuracy |
|---|---|---|---|---|---|---|---|---|
| [2,3) | 1802 | 124 | 167 | 0.927 | 0.689 | 0.867 | 0.854 | 0.921 |
| [3,4) | 183 | 17 | 20 | 0.941 | 0.800 | 0.909 | 0.834 | 0.913 |
| [4,5) | 160 | 12 | 16 | 0.917 | 0.688 | 0.859 | 0.844 | 0.944 |
| [5,inf) | 244 | 14 | 21 | 1.000 | 0.667 | 0.909 | 0.855 | 0.918 |

### Object type, all_usable

| Object type | Events | Final Critical | Predicted Critical | Critical recall | Critical precision | Critical F2 | Macro-F1 | Accuracy |
|---|---|---|---|---|---|---|---|---|
| DEBRIS | 1291 | 105 | 145 | 0.943 | 0.683 | 0.876 | 0.856 | 0.916 |
| PAYLOAD | 275 | 11 | 11 | 0.818 | 0.818 | 0.818 | 0.856 | 0.949 |
| ROCKET BODY | 28 | 5 | 6 | 1.000 | 0.833 | 0.962 | 0.939 | 0.964 |
| UNKNOWN | 792 | 46 | 62 | 0.935 | 0.694 | 0.874 | 0.841 | 0.919 |
| other (TBA) | 3 | 0 | 0 | undefined | undefined | undefined | 1.000 | 1.000 |

### Current tier at forecast time, real_forecast

| Current tier | Events | Final Critical | Predicted Critical | Critical recall | Critical precision | Critical F2 | Macro-F1 | Accuracy |
|---|---|---|---|---|---|---|---|---|
| Critical | 123 | 94 | 122 | 1.000 | 0.770 | 0.944 | 0.315 | 0.772 |
| Elevated | 339 | 21 | 36 | 0.619 | 0.361 | 0.542 | 0.465 | 0.802 |
| Safe | 1317 | 3 | 0 | 0.000 | undefined | 0.000 | 0.413 | 0.955 |

## Ablations (real_forecast holdout)

| Model | Events | Critical recall | Critical precision | Critical F2 | Elev+ F2 | Macro-F1 | Brier | Log loss | Auto-clear | Final Crit auto-cleared | Watch share | Escalation recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lgbm_multiclass (primary) | 1779 | 0.907 | 0.677 | 0.849 | 0.923 | 0.841 | 0.104 | 0.202 | 16.4% | 0 | 4.9% | 0.792 |
| use_past_risk_off | 1779 | 0.898 | 0.707 | 0.852 | 0.912 | 0.835 | 0.110 | 0.218 | 20.1% | 1 | 5.4% | 0.792 |
| latest_only_training | 1779 | 0.864 | 0.718 | 0.831 | 0.927 | 0.849 | 0.113 | 0.226 | 1.6% | 0 | 5.4% | 0.750 |
| no_history_features | 1779 | 0.864 | 0.723 | 0.832 | 0.922 | 0.854 | 0.111 | 0.222 | 27.8% | 1 | 5.7% | 0.792 |

## Mission-held-out cross-validation

Out-of-fold predictions over all usable events with folds grouped by `mission_id` (mission_id is never a feature). This checks that the model is not memorising missions.

| Model | Events | Critical recall | Critical precision | Critical F2 | Elev+ F2 | Macro-F1 | Brier | Log loss | Auto-clear | Final Crit auto-cleared | Watch share | Escalation recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mission_heldout_oof | 8892 | 0.903 | 0.673 | 0.845 | 0.922 | 0.835 | 0.112 | 0.214 | 12.0% | 0 | 5.4% | 0.812 |

| Model | Events | Critical recall | Critical precision | Critical F2 | Elev+ F2 | Macro-F1 | Brier | Log loss | Auto-clear | Final Crit auto-cleared | Watch share | Escalation recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mission_heldout_oof | 11942 | 0.930 | 0.698 | 0.872 | 0.932 | 0.849 | 0.100 | 0.193 | 9.7% | 0 | 5.5% | 0.812 |

## Conformal prediction sets

Nominal coverage 0.900, empirical holdout coverage undefined, average set size undefined, q_hat 0.3098.
On real_forecast events: coverage 0.933, average set size 1.00, singleton share 100.0%.

## Illustrative cases (primary model, holdout)

### Missed Critical (ended Critical, forecast lower, not on the watch list): 3 shown

| Event | Lead time (d) | Miss (m) | CDMs | Current | Final | Forecast | P(Critical) | Set | Action | Reasons |
|---|---|---|---|---|---|---|---|---|---|---|
| kelvins_train:2958 | 2.54 | 34514 | 1 | Safe | Critical | Safe | 0.000 | Safe | Human review | NO_HISTORY, FORECAST_SAFE, LOW_CONFIDENCE_SAFE |
| kelvins_train:8732 | 2.34 | 28737 | 14 | Safe | Critical | Safe | 0.000 | Safe | Human review | FORECAST_SAFE, RISK_RISING, MISS_SHRINKING, SIGMA_GROWING, HIGH_VOLATILITY, LOW_CONFIDENCE_SAFE |
| kelvins_train:3183 | 3.03 | 8524 | 6 | Safe | Critical | Safe | 0.001 | Safe | Human review | FORECAST_SAFE, HIGH_VOLATILITY, LOW_CONFIDENCE_SAFE |

### Caught by the escalation watch (ended Critical, forecast lower, on the watch list): 3 shown

| Event | Lead time (d) | Miss (m) | CDMs | Current | Final | Forecast | P(Critical) | Set | Action | Reasons |
|---|---|---|---|---|---|---|---|---|---|---|
| kelvins_train:12558 | 2.02 | 1990 | 1 | Elevated | Critical | Elevated | 0.041 | Elevated | Escalation watch | NO_HISTORY, FORECAST_ELEVATED, CURRENT_TIER_ELEVATED, ESCALATION_RISK |
| kelvins_train:7848 | 2.27 | 1426 | 6 | Elevated | Critical | Elevated | 0.043 | Elevated | Escalation watch | FORECAST_ELEVATED, CURRENT_TIER_ELEVATED, MISS_SHRINKING, HIGH_VOLATILITY, ESCALATION_RISK |
| kelvins_train:5025 | 2.01 | 1261 | 5 | Elevated | Critical | Elevated | 0.123 | Elevated | Escalation watch | FORECAST_ELEVATED, CURRENT_TIER_ELEVATED, RISK_RISING, MISS_SHRINKING, SIGMA_GROWING, HIGH_VOLATILITY, ESCALATION_RISK |

### False alarm (forecast Critical, ended lower): 3 shown

| Event | Lead time (d) | Miss (m) | CDMs | Current | Final | Forecast | P(Critical) | Set | Action | Reasons |
|---|---|---|---|---|---|---|---|---|---|---|
| kelvins_train:2960 | 2.43 | 189 | 4 | Critical | Elevated | Critical | 0.984 | Critical | Maneuver planning review | FORECAST_CRITICAL, CURRENT_TIER_CRITICAL, MISS_SHRINKING |
| kelvins_train:9929 | 2.25 | 392 | 2 | Critical | Safe | Critical | 0.976 | Critical | Maneuver planning review | FORECAST_CRITICAL, CURRENT_TIER_CRITICAL |
| kelvins_train:8273 | 2.65 | 521 | 3 | Critical | Elevated | Critical | 0.969 | Critical | Maneuver planning review | FORECAST_CRITICAL, CURRENT_TIER_CRITICAL, RISK_RISING, MISS_SHRINKING, SIGMA_GROWING, HIGH_VOLATILITY |

### False de-escalation (Critical now, forecast lower, ended Critical): 0 shown

None on the holdout.

### Most uncertain forecasts: 3 shown

| Event | Lead time (d) | Miss (m) | CDMs | Current | Final | Forecast | P(Critical) | Set | Action | Reasons |
|---|---|---|---|---|---|---|---|---|---|---|
| kelvins_train:7754 | 2.28 | 786 | 4 | Critical | Critical | Critical | 0.403 | Critical | Maneuver planning review | FORECAST_CRITICAL, CURRENT_TIER_CRITICAL, RISK_RISING, MISS_SHRINKING, HIGH_VOLATILITY |
| kelvins_train:11606 | 2.18 | 587 | 11 | Critical | Elevated | Critical | 0.464 | Critical | Maneuver planning review | FORECAST_CRITICAL, CURRENT_TIER_CRITICAL, MISS_SHRINKING, HIGH_VOLATILITY |
| kelvins_train:4479 | 2.28 | 766 | 10 | Critical | Safe | Critical | 0.467 | Critical | Maneuver planning review | FORECAST_CRITICAL, CURRENT_TIER_CRITICAL, MISS_SHRINKING, HIGH_VOLATILITY |

## Test-set predictions (Kelvins test file)

Predicted class distribution not available in manifest.json.

The Kelvins test set was enriched with high-risk events, so its predicted class distribution is expected to be shifted toward Elevated/Critical relative to the training labels.

Batch run: status counts {"valid": 2002, "warning": 165}; triage counts {"Human review": 1086, "Monitor": 347, "Auto-clear candidate": 321, "Maneuver planning review": 240, "Escalation watch": 173}; batch inference 6.2 s.

## Data and label audit

| Quantity | Value |
|---|---|
| Events in the labels workbook | 13154 |
| Usable events (>= 1 CDM at >= 2 days) | 11942 |
| Unusable events | 1212 |
| Trivial events (final CDM itself at >= 2 days) | 3050 |
| Real-forecast events | 8892 |
| Usable class counts (Safe / Elevated / Critical) | 8946 / 2162 / 834 |
| Labels match recomputed tiers | True |

| File | Rows | Events | Kept columns | Dropped columns | sha256 (prefix) |
|---|---|---|---|---|---|
| train | 162634 | undefined | 14 | 89 | ba47ce80580d |
| test | 24484 | undefined | 14 | 89 | fc4110e979e7 |

`test_horizon`: `{"min_time_to_tca_days": 2.000213078703704, "test_rows_stop_at_or_after_horizon": true}`

## Model selection trace

Training rows: `snapshots` chosen on dev OOF Critical F2 (real_forecast): snapshots = {'critical_recall': 0.8559322033898306, 'critical_precision': 0.7481481481481481, 'critical_f2': 0.8319604612850082, 'macro_f1': 0.8543082496974018}, latest = {'critical_recall': 0.8283898305084746, 'critical_precision': 0.7696850393700787, 'critical_f2': 0.8159432387312188, 'macro_f1': 0.8580496303826025}.

Grouped CV search (undefined folds, 2 configurations, rule: max OOF Critical F2 on real_forecast latest snapshots (t=0.5), tie-break macro-F1). Best params `{"learning_rate": 0.050958241942238525, "num_leaves": 23, "min_child_samples": 105, "feature_fraction": 0.9575793759734148, "lambda_l2": 0.0}`, scores `{"critical_recall": 0.8559322033898306, "critical_precision": 0.7481481481481481, "critical_f2": 0.8319604612850082, "macro_f1": 0.8543082496974018}`.

Decision thresholds (tuned on dev OOF only): `t_crit` = 0.200, `t_elev` = 0.300, `t_watch` = 0.037, `t_clear` = 0.999.

Policy tuning summary: `{"t_watch": 0.03660673261890372, "t_clear": 0.9989702722681502, "watch": {"watch_max_share": 0.05, "watch_share": 0.04982727938867371, "n_watch": 476, "n_escalations_caught": 26, "n_escalations_total": 49, "recall": 0.5306122448979592, "n_events": 9553}, "auto_clear": {"n_eligible": 6536, "n_eligible_final_critical": 8, "auto_clear_share": 0.15314560870930596, "n_auto_clear": 1463, "n_final_critical_cleared": 0, "zero_missed_achievable": true, "note": "zero final-Critical cleared at t_clear", "n_events": 9553}}`


## Notes

- Metrics are computed on the 20% event holdout of train_data only; test_data has no final outcomes and is used solely for demo predictions.
- Trivial events (final CDM itself at >= 2 days) are reported separately from real_forecast events; headline numbers use real_forecast.
- Probabilities are per-class isotonic-calibrated on dev out-of-fold predictions and renormalised; decision and triage thresholds are tuned on dev OOF only.
- risk of CDMs at >= 2 days is used as a past input when features.use_past_risk is true (the carry-forward baseline uses the same information).
- Zero final-Critical events auto-cleared on dev OOF is a tuning target, not a guarantee; the holdout count is reported.
