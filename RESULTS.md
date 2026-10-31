# Results of the shipped run

Task: forecast each event's **final** tier (Safe / Elevated / Critical) from CDMs at >= 2 days before TCA. Primary model: `lgbm_ordinal` (version `lgbm_ordinal-20261031T062654Z`). Generated 2026-10-31T06:29:56.250411+00:00. Total training time 419.3 s.

Holdout: 2389 events (1779 real forecasts, 610 trivial), split seed 42, holdout fraction 0.200, split hash `d18e932157a7c887`. Final-tier counts on the holdout: {"Safe": 1790, "Elevated": 432, "Critical": 167}.

Subsets: `all_usable` = every holdout event; `trivial` = the event's final CDM was itself at >= 2 days (the answer is visible in the input); `real_forecast` = everything else and the headline subset.

## Headline: real-forecast holdout

Headline metric: **Critical recall on real_forecast events**, read together with Critical precision / F2. `carry_forward_risk` is the official naive baseline (the latest known tier carried forward).

| Model | Events | Critical recall | Critical precision | Critical F2 | Elev+ F2 | Macro-F1 | Brier | Log loss | Auto-clear | Human review | Final Crit auto-cleared | Watch share | Escalation recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carry_forward_risk | 1779 | 0.797 | 0.764 | 0.790 | 0.918 | 0.858 | 0.142 | 1.957 | 0.0% | 74.0% | 0 | 0.0% | 0.000 |
| **lgbm_ordinal** | 1779 | 0.907 | 0.673 | 0.848 | 0.919 | 0.838 | 0.108 | 0.204 | 17.0% | 54.4% | 0 | 4.7% | 0.750 |
| carry_forward_mre | 1779 | 0.797 | 0.686 | 0.772 | 0.813 | 0.700 | 0.527 | 7.284 | 0.0% | 54.6% | 0 | 0.0% | 0.000 |
| majority | 1779 | 0.000 | undefined | 0.000 | 0.000 | 0.284 | 0.512 | 7.067 | 0.0% | 100.0% | 0 | 0.0% | 0.000 |
| logistic | 1779 | 0.864 | 0.689 | 0.823 | 0.902 | 0.825 | 0.144 | 0.278 | 32.3% | 39.1% | 2 | 5.8% | 0.792 |
| lgbm_multiclass | 1779 | 0.881 | 0.722 | 0.844 | 0.921 | 0.849 | 0.107 | 0.209 | 27.4% | 44.0% | 0 | 4.9% | 0.792 |

`lgbm_ordinal` beats `carry_forward_risk` on Critical F2 (real_forecast): 0.848 vs 0.790.

### All usable holdout events

| Model | Events | Critical recall | Critical precision | Critical F2 | Elev+ F2 | Macro-F1 | Brier | Log loss | Auto-clear | Human review | Final Crit auto-cleared | Watch share | Escalation recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carry_forward_risk | 2389 | 0.856 | 0.831 | 0.851 | 0.938 | 0.894 | 0.105 | 1.457 | 0.0% | 74.6% | 0 | 0.0% | 0.000 |
| **lgbm_ordinal** | 2389 | 0.934 | 0.696 | 0.874 | 0.930 | 0.850 | 0.098 | 0.187 | 14.4% | 57.2% | 0 | 4.6% | 0.750 |
| carry_forward_mre | 2389 | 0.856 | 0.769 | 0.837 | 0.835 | 0.745 | 0.454 | 6.269 | 0.0% | 57.1% | 0 | 0.0% | 0.000 |
| majority | 2389 | 0.000 | undefined | 0.000 | 0.000 | 0.286 | 0.501 | 6.928 | 0.0% | 100.0% | 0 | 0.0% | 0.000 |
| logistic | 2389 | 0.904 | 0.706 | 0.856 | 0.890 | 0.826 | 0.142 | 0.270 | 34.9% | 37.6% | 2 | 5.5% | 0.792 |
| lgbm_multiclass | 2389 | 0.916 | 0.754 | 0.878 | 0.935 | 0.868 | 0.095 | 0.186 | 24.4% | 47.5% | 0 | 4.7% | 0.792 |

### Trivial events (answer visible in the input)

| Model | Events | Critical recall | Critical precision | Critical F2 | Elev+ F2 | Macro-F1 | Brier | Log loss | Auto-clear | Human review | Final Crit auto-cleared | Watch share | Escalation recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carry_forward_risk | 610 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | -0.000 | 0.0% | 76.4% | 0 | 0.0% | undefined |
| **lgbm_ordinal** | 610 | 1.000 | 0.754 | 0.939 | 0.965 | 0.882 | 0.069 | 0.137 | 7.0% | 65.4% | 0 | 4.4% | undefined |
| carry_forward_mre | 610 | 1.000 | 1.000 | 1.000 | 0.908 | 0.879 | 0.239 | 3.307 | 0.0% | 64.4% | 0 | 0.0% | undefined |
| majority | 610 | 0.000 | undefined | 0.000 | 0.000 | 0.289 | 0.472 | 6.523 | 0.0% | 100.0% | 0 | 0.0% | undefined |
| logistic | 610 | 1.000 | 0.742 | 0.935 | 0.849 | 0.816 | 0.139 | 0.248 | 42.5% | 33.3% | 0 | 4.6% | undefined |
| lgbm_multiclass | 610 | 1.000 | 0.831 | 0.961 | 0.980 | 0.921 | 0.058 | 0.121 | 15.4% | 57.7% | 0 | 4.1% | undefined |

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

### `lgbm_ordinal` on real_forecast (1779 events)

Counts (rows = true final tier, columns = predicted):

| true \ pred | Safe | Elevated | Critical |
|---|---|---|---|
| Safe | 1248 | 70 | 6 |
| Elevated | 25 | 266 | 46 |
| Critical | 3 | 8 | 107 |

Row-normalised (share of each true tier):

| true \ pred | Safe | Elevated | Critical |
|---|---|---|---|
| Safe | 94.3% | 5.3% | 0.5% |
| Elevated | 7.4% | 78.9% | 13.6% |
| Critical | 2.5% | 6.8% | 90.7% |

### `lgbm_ordinal` on all_usable (2389 events)

Counts (rows = true final tier, columns = predicted):

| true \ pred | Safe | Elevated | Critical |
|---|---|---|---|
| Safe | 1692 | 91 | 7 |
| Elevated | 26 | 345 | 61 |
| Critical | 3 | 8 | 156 |

Row-normalised (share of each true tier):

| true \ pred | Safe | Elevated | Critical |
|---|---|---|---|
| Safe | 94.5% | 5.1% | 0.4% |
| Elevated | 6.0% | 79.9% | 14.1% |
| Critical | 1.8% | 4.8% | 93.4% |

## Escalation watch

Among events that are not Critical at the forecast time: how many end Critical, how many the forecast flags (predicted Critical or placed on the escalation watch list), and how many of those flags are hits. Among events that are Critical now: how many de-escalate and how the forecast handles them.

| Model | Subset | Not-Critical now | End Critical | Flagged | Caught | Escalation recall | Flag precision | Watch size | Watch hits | Critical now | De-escalate | Pred. de-esc. | Correct | False de-esc. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carry_forward_risk | real_forecast | 1656 | 24 | 0 | 0 | 0.000 | undefined | 0 | 0 | 123 | 29 | 0 | 0 | 0 |
| carry_forward_risk | all_usable | 2217 | 24 | 0 | 0 | 0.000 | undefined | 0 | 0 | 172 | 29 | 0 | 0 | 0 |
| lgbm_ordinal | real_forecast | 1656 | 24 | 120 | 18 | 0.750 | 0.150 | 84 | 5 | 123 | 29 | 0 | 0 | 0 |
| lgbm_ordinal | all_usable | 2217 | 24 | 163 | 18 | 0.750 | 0.110 | 111 | 5 | 172 | 29 | 0 | 0 | 0 |
| carry_forward_mre | real_forecast | 1656 | 24 | 14 | 0 | 0.000 | 0.000 | 0 | 0 | 123 | 29 | 0 | 0 | 0 |
| carry_forward_mre | all_usable | 2217 | 24 | 14 | 0 | 0.000 | 0.000 | 0 | 0 | 172 | 29 | 0 | 0 | 0 |
| majority | real_forecast | 1656 | 24 | 0 | 0 | 0.000 | undefined | 0 | 0 | 123 | 29 | 123 | 29 | 94 |
| majority | all_usable | 2217 | 24 | 0 | 0 | 0.000 | undefined | 0 | 0 | 172 | 29 | 172 | 29 | 143 |
| logistic | real_forecast | 1656 | 24 | 129 | 19 | 0.792 | 0.147 | 102 | 10 | 123 | 29 | 2 | 1 | 1 |
| logistic | all_usable | 2217 | 24 | 174 | 19 | 0.792 | 0.109 | 130 | 10 | 172 | 29 | 2 | 1 | 1 |
| lgbm_multiclass | real_forecast | 1656 | 24 | 108 | 19 | 0.792 | 0.176 | 85 | 9 | 123 | 29 | 2 | 2 | 0 |
| lgbm_multiclass | all_usable | 2217 | 24 | 143 | 19 | 0.792 | 0.133 | 110 | 9 | 172 | 29 | 2 | 2 | 0 |

## Auto-clear and triage mix (primary model)

| Subset | Events | Auto-clear share | Final-Critical auto-cleared | Final Critical | Watch share | Final-Critical in watch | Maneuver review | Human review | Monitor | Low-priority monitor | Final-Critical not urgent |
|---|---|---|---|---|---|---|---|---|---|---|---|
| real_forecast | 1779 | 17.0% | 0 | 118 | 4.7% | 5 | 8.9% | 54.4% | 15.0% | 0.0% | n/a (older artifacts) |
| all_usable | 2389 | 14.4% | 0 | 167 | 4.6% | 5 | 9.4% | 57.2% | 14.4% | 0.0% | n/a (older artifacts) |
| trivial | 610 | 7.0% | 0 | 49 | 4.4% | 0 | 10.7% | 65.4% | 12.5% | 0.0% | n/a (older artifacts) |

Review workload = Maneuver planning review + Escalation watch + Human review. Auto-clear and low-priority monitor shares are *potential* workload reductions on the holdout, not demonstrated labour savings; an operator study has not been run.

No event that ended Critical was auto-cleared on the holdout.

Action counts (primary model, all usable holdout events):

| Action | Events |
|---|---|
| Auto-clear candidate | 345 |
| Escalation watch | 111 |
| Human review | 1366 |
| Maneuver planning review | 224 |
| Monitor | 343 |

## Calibration

| Model | Subset | n | Brier (multiclass) | Log loss | Brier (Critical) | Critical PR-AUC | Critical ROC-AUC | ECE (Critical) |
|---|---|---|---|---|---|---|---|---|
| carry_forward_risk | real_forecast | 1779 | 0.1417 | 1.9570 | 0.0298 | 0.622 | 0.890 | 0.0298 |
| carry_forward_risk | all_usable | 2389 | 0.1055 | 1.4573 | 0.0222 | 0.722 | 0.922 | 0.0222 |
| lgbm_ordinal | real_forecast | 1779 | 0.1077 | 0.2042 | 0.0203 | 0.882 | 0.979 | 0.0108 |
| lgbm_ordinal | all_usable | 2389 | 0.0979 | 0.1871 | 0.0178 | 0.903 | 0.984 | 0.0096 |
| carry_forward_mre | real_forecast | 1779 | 0.5273 | 7.2844 | 0.0377 | 0.560 | 0.885 | 0.0377 |
| carry_forward_mre | all_usable | 2389 | 0.4537 | 6.2687 | 0.0280 | 0.668 | 0.918 | 0.0280 |
| majority | real_forecast | 1779 | 0.5115 | 7.0670 | 0.0663 | 0.066 | 0.500 | 0.0663 |
| majority | all_usable | 2389 | 0.5015 | 6.9280 | 0.0699 | 0.070 | 0.500 | 0.0699 |
| logistic | real_forecast | 1779 | 0.1436 | 0.2777 | 0.0247 | 0.791 | 0.974 | 0.0149 |
| logistic | all_usable | 2389 | 0.1424 | 0.2701 | 0.0230 | 0.838 | 0.981 | 0.0145 |
| lgbm_multiclass | real_forecast | 1779 | 0.1073 | 0.2086 | 0.0203 | 0.887 | 0.978 | 0.0123 |
| lgbm_multiclass | all_usable | 2389 | 0.0947 | 0.1861 | 0.0180 | 0.910 | 0.983 | 0.0112 |

Reliability of P(Critical) for `lgbm_ordinal` on real_forecast holdout events (10 equal-width bins):

| Bin | Mean predicted | Observed Critical share | Events |
|---|---|---|---|
| 0.0-0.1 | 0.005 | 0.006 | 1612 |
| 0.1-0.2 | 0.169 | 0.250 | 8 |
| 0.2-0.3 | 0.253 | 0.158 | 19 |
| 0.3-0.4 | 0.321 | 0.333 | 3 |
| 0.4-0.5 | 0.440 | 0.455 | 11 |
| 0.5-0.6 | 0.535 | 0.375 | 8 |
| 0.6-0.7 | 0.606 | 0.200 | 5 |
| 0.7-0.8 | 0.759 | 0.500 | 2 |
| 0.8-0.9 | 0.860 | 0.545 | 11 |
| 0.9-1.0 | 0.952 | 0.870 | 100 |

Calibration method: `isotonic` fitted on dev out-of-fold predictions (dev OOF Brier undefined, log loss undefined).

## Bootstrap confidence intervals

Point estimate followed by the 95 % event-bootstrap interval (2.5 / 97.5 percentiles).

| Model | Subset | Resamples | Critical recall | Critical precision | Macro-F1 | Auto-clear share |
|---|---|---|---|---|---|---|
| lgbm_ordinal | real_forecast | 1000 | 0.907 [0.852, 0.959] | 0.673 [0.601, 0.747] | 0.838 [0.813, 0.863] | 17.0% [0.153, 0.188] |
| lgbm_ordinal | all_usable | 1000 | 0.934 [0.891, 0.970] | 0.696 [0.637, 0.753] | 0.850 [0.829, 0.870] | 14.4% [0.131, 0.160] |
| carry_forward_risk | real_forecast | 1000 | 0.797 [0.724, 0.865] | 0.764 [0.683, 0.836] | 0.858 [0.833, 0.884] | 0.0% [0.000, 0.000] |
| carry_forward_risk | all_usable | 1000 | 0.856 [0.801, 0.907] | 0.831 [0.776, 0.886] | 0.894 [0.875, 0.914] | 0.0% [0.000, 0.000] |

## Breakdowns (primary model)

### Lead time of the latest snapshot, real_forecast

| Days to TCA | Events | Final Critical | Predicted Critical | Critical recall | Critical precision | Critical F2 | Macro-F1 | Accuracy |
|---|---|---|---|---|---|---|---|---|
| [2,3) | 1701 | 113 | 152 | 0.912 | 0.678 | 0.853 | 0.844 | 0.914 |
| [3,4) | 30 | 3 | 2 | 0.667 | 1.000 | 0.714 | 0.567 | 0.833 |
| [4,5) | 18 | 1 | 1 | 1.000 | 1.000 | 1.000 | 0.657 | 0.944 |
| [5,inf) | 30 | 1 | 4 | 1.000 | 0.250 | 0.625 | 0.434 | 0.800 |

### Object type, real_forecast

| Object type | Events | Final Critical | Predicted Critical | Critical recall | Critical precision | Critical F2 | Macro-F1 | Accuracy |
|---|---|---|---|---|---|---|---|---|
| DEBRIS | 991 | 74 | 107 | 0.932 | 0.645 | 0.856 | 0.840 | 0.905 |
| PAYLOAD | 224 | 9 | 9 | 0.778 | 0.778 | 0.778 | 0.850 | 0.951 |
| ROCKET BODY | 20 | 5 | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| UNKNOWN | 543 | 30 | 38 | 0.867 | 0.684 | 0.823 | 0.813 | 0.902 |
| other (TBA) | 1 | 0 | 0 | undefined | undefined | undefined | 1.000 | 1.000 |

### Lead time of the latest snapshot, all_usable

| Days to TCA | Events | Final Critical | Predicted Critical | Critical recall | Critical precision | Critical F2 | Macro-F1 | Accuracy |
|---|---|---|---|---|---|---|---|---|
| [2,3) | 1802 | 124 | 165 | 0.919 | 0.691 | 0.862 | 0.848 | 0.916 |
| [3,4) | 183 | 17 | 19 | 0.941 | 0.842 | 0.920 | 0.864 | 0.929 |
| [4,5) | 160 | 12 | 16 | 1.000 | 0.750 | 0.938 | 0.885 | 0.956 |
| [5,inf) | 244 | 14 | 24 | 1.000 | 0.583 | 0.875 | 0.820 | 0.898 |

### Object type, all_usable

| Object type | Events | Final Critical | Predicted Critical | Critical recall | Critical precision | Critical F2 | Macro-F1 | Accuracy |
|---|---|---|---|---|---|---|---|---|
| DEBRIS | 1291 | 105 | 148 | 0.952 | 0.676 | 0.880 | 0.851 | 0.911 |
| PAYLOAD | 275 | 11 | 11 | 0.818 | 0.818 | 0.818 | 0.866 | 0.956 |
| ROCKET BODY | 28 | 5 | 6 | 1.000 | 0.833 | 0.962 | 0.939 | 0.964 |
| UNKNOWN | 792 | 46 | 59 | 0.913 | 0.712 | 0.864 | 0.837 | 0.914 |
| other (TBA) | 3 | 0 | 0 | undefined | undefined | undefined | 1.000 | 1.000 |

### Current tier at forecast time, real_forecast

| Current tier | Events | Final Critical | Predicted Critical | Critical recall | Critical precision | Critical F2 | Macro-F1 | Accuracy |
|---|---|---|---|---|---|---|---|---|
| Critical | 123 | 94 | 123 | 1.000 | 0.764 | 0.942 | 0.289 | 0.764 |
| Elevated | 339 | 21 | 36 | 0.619 | 0.361 | 0.542 | 0.448 | 0.799 |
| Safe | 1317 | 3 | 0 | 0.000 | undefined | 0.000 | 0.397 | 0.954 |

## Ablations (real_forecast holdout)

| Model | Events | Critical recall | Critical precision | Critical F2 | Elev+ F2 | Macro-F1 | Brier | Log loss | Auto-clear | Human review | Final Crit auto-cleared | Watch share | Escalation recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lgbm_ordinal (primary) | 1779 | 0.907 | 0.673 | 0.848 | 0.919 | 0.838 | 0.108 | 0.204 | 17.0% | 54.4% | 0 | 4.7% | 0.750 |
| use_past_risk_off | 1779 | 0.890 | 0.714 | 0.848 | 0.912 | 0.843 | 0.111 | 0.218 | 21.7% | 50.0% | 0 | 5.7% | 0.792 |
| latest_only_training | 1779 | 0.847 | 0.719 | 0.818 | 0.922 | 0.853 | 0.116 | 0.231 | 27.7% | 45.3% | 1 | 5.1% | 0.750 |
| no_history_features | 1779 | 0.881 | 0.712 | 0.841 | 0.920 | 0.853 | 0.113 | 0.226 | 24.7% | 48.3% | 1 | 5.6% | 0.750 |

## Mission-held-out cross-validation

Out-of-fold predictions over all usable events with folds grouped by `mission_id` (mission_id is never a feature). This checks that the model is not memorising missions.

| Model | Events | Critical recall | Critical precision | Critical F2 | Elev+ F2 | Macro-F1 | Brier | Log loss | Auto-clear | Human review | Final Crit auto-cleared | Watch share | Escalation recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mission_heldout_oof | 8892 | 0.902 | 0.673 | 0.844 | 0.920 | 0.835 | 0.111 | 0.208 | 11.3% | 59.6% | 0 | 5.4% | 0.759 |

| Model | Events | Critical recall | Critical precision | Critical F2 | Elev+ F2 | Macro-F1 | Brier | Log loss | Auto-clear | Human review | Final Crit auto-cleared | Watch share | Escalation recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mission_heldout_oof | 11942 | 0.929 | 0.700 | 0.872 | 0.930 | 0.848 | 0.100 | 0.189 | 9.4% | 62.2% | 0 | 5.1% | 0.759 |

## Conformal prediction sets

Nominal coverage 0.900, q_hat 0.3155; empirical holdout coverage 0.938, average set size 1.00 (all usable events).

real_forecast: marginal coverage 0.929, average set size 1.00, singleton share 100.0%.

all_usable: marginal coverage 0.938, average set size 1.00, singleton share 100.0%.

When every set is a singleton the sets carry no information beyond the point decision; marginal coverage above nominal then comes from the majority Safe class, and the class-conditional table above is the honest statement of what the sets guarantee. No class-conditional coverage guarantee is claimed.

## Illustrative cases (primary model, holdout)

### Missed Critical (ended Critical, forecast lower, not on the watch list): 3 shown

| Event | Lead time (d) | Miss (m) | CDMs | Current | Final | Forecast | P(Critical) | Set | Action | Reasons |
|---|---|---|---|---|---|---|---|---|---|---|
| kelvins_train:2958 | 2.54 | 34514 | 1 | Safe | Critical | Safe | 0.000 | Safe | Human review | NO_HISTORY, FORECAST_SAFE, LOW_CONFIDENCE_SAFE |
| kelvins_train:8732 | 2.34 | 28737 | 14 | Safe | Critical | Safe | 0.000 | Safe | Human review | FORECAST_SAFE, RISK_RISING, MISS_SHRINKING, SIGMA_GROWING, HIGH_VOLATILITY, LOW_CONFIDENCE_SAFE |
| kelvins_train:3183 | 3.03 | 8524 | 6 | Safe | Critical | Safe | 0.005 | Safe | Human review | FORECAST_SAFE, HIGH_VOLATILITY, LOW_CONFIDENCE_SAFE |

### Caught by the escalation watch (ended Critical, forecast lower, on the watch list): 3 shown

| Event | Lead time (d) | Miss (m) | CDMs | Current | Final | Forecast | P(Critical) | Set | Action | Reasons |
|---|---|---|---|---|---|---|---|---|---|---|
| kelvins_train:7848 | 2.27 | 1426 | 6 | Elevated | Critical | Elevated | 0.060 | Elevated | Escalation watch | FORECAST_ELEVATED, CURRENT_TIER_ELEVATED, MISS_SHRINKING, HIGH_VOLATILITY, ESCALATION_RISK |
| kelvins_train:11943 | 2.29 | 1653 | 15 | Elevated | Critical | Elevated | 0.060 | Elevated | Escalation watch | FORECAST_ELEVATED, CURRENT_TIER_ELEVATED, MISS_SHRINKING, HIGH_VOLATILITY, ESCALATION_RISK |
| kelvins_train:5025 | 2.01 | 1261 | 5 | Elevated | Critical | Elevated | 0.065 | Elevated | Escalation watch | FORECAST_ELEVATED, CURRENT_TIER_ELEVATED, RISK_RISING, MISS_SHRINKING, SIGMA_GROWING, HIGH_VOLATILITY, ESCALATION_RISK |

### False alarm (forecast Critical, ended lower): 3 shown

| Event | Lead time (d) | Miss (m) | CDMs | Current | Final | Forecast | P(Critical) | Set | Action | Reasons |
|---|---|---|---|---|---|---|---|---|---|---|
| kelvins_train:2960 | 2.43 | 189 | 4 | Critical | Elevated | Critical | 0.963 | Critical | Maneuver planning review | FORECAST_CRITICAL, CURRENT_TIER_CRITICAL, MISS_SHRINKING |
| kelvins_train:11851 | 2.05 | 325 | 15 | Critical | Elevated | Critical | 0.961 | Critical | Maneuver planning review | FORECAST_CRITICAL, CURRENT_TIER_CRITICAL, RISK_RISING |
| kelvins_train:9929 | 2.25 | 392 | 2 | Critical | Safe | Critical | 0.961 | Critical | Maneuver planning review | FORECAST_CRITICAL, CURRENT_TIER_CRITICAL |

### False de-escalation (Critical now, forecast lower, ended Critical): 0 shown

None on the holdout.

### Most uncertain forecasts: 3 shown

| Event | Lead time (d) | Miss (m) | CDMs | Current | Final | Forecast | P(Critical) | Set | Action | Reasons |
|---|---|---|---|---|---|---|---|---|---|---|
| kelvins_train:9574 | 4.71 | 1153 | 1 | Elevated | Critical | Critical | 0.262 | Elevated | Maneuver planning review | NO_HISTORY, FORECAST_CRITICAL, CURRENT_TIER_ELEVATED |
| kelvins_train:11630 | 3.10 | 617 | 8 | Critical | Critical | Critical | 0.463 | Critical | Maneuver planning review | FORECAST_CRITICAL, CURRENT_TIER_CRITICAL, RISK_RISING, MISS_SHRINKING |
| kelvins_train:1655 | 5.24 | 727 | 2 | Critical | Elevated | Critical | 0.468 | Critical | Maneuver planning review | FORECAST_CRITICAL, CURRENT_TIER_CRITICAL, RISK_RISING, MISS_SHRINKING, HIGH_VOLATILITY |

## Test-set predictions (Kelvins test file)

| Predicted final tier | Events | Share |
|---|---|---|
| Safe | 1438 | 66.4% |
| Elevated | 489 | 22.6% |
| Critical | 240 | 11.1% |

The Kelvins test set was enriched with high-risk events, so its predicted class distribution is expected to be shifted toward Elevated/Critical relative to the training labels.

Batch run: 2167 events / 24484 snapshots; status counts {"valid": 2002, "warning": 165}; triage counts {"Human review": 1074, "Auto-clear candidate": 352, "Monitor": 333, "Maneuver planning review": 240, "Escalation watch": 168}; batch inference 9.7 s.

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

Training rows: `snapshots` chosen on dev OOF Critical F2 (real_forecast): snapshots = {'critical_recall': 0.8728813559322034, 'critical_precision': 0.7370304114490162, 'critical_f2': 0.8418471597874949, 'macro_f1': 0.8515030912165972}, latest = {'critical_recall': 0.8241525423728814, 'critical_precision': 0.7764471057884231, 'critical_f2': 0.8141481791544579, 'macro_f1': 0.8600351533370483}.

Grouped CV search (5 folds, 12 configurations, rule: max OOF Critical F2 on real_forecast latest snapshots (t=0.5), tie-break macro-F1). Best params `{"learning_rate": 0.04679255978730041, "num_leaves": 12, "min_child_samples": 190, "feature_fraction": 0.9498034588173512, "lambda_l2": 1.0}`, scores `{"critical_recall": 0.8728813559322034, "critical_precision": 0.7370304114490162, "critical_f2": 0.8418471597874949, "macro_f1": 0.8515030912165972}`.

Decision thresholds (tuned on dev OOF only): `t_crit` = 0.200, `t_elev` = 0.300, `t_watch` = 0.030, `t_clear` = 0.999.

Policy tuning summary: `{"t_watch": 0.02976306754188084, "t_clear": 0.9994625934229522, "watch": {"watch_max_share": 0.05, "watch_share": 0.04993195854705328, "n_watch": 477, "n_escalations_caught": 24, "n_escalations_total": 46, "recall": 0.5217391304347826, "n_events": 9553}, "auto_clear": {"n_eligible": 6524, "n_eligible_final_critical": 6, "auto_clear_share": 0.1299068355490422, "n_auto_clear": 1241, "n_final_critical_cleared": 0, "zero_missed_achievable": true, "note": "zero final-Critical cleared at t_clear", "n_events": 9553}}`


## Notes

- **These artifacts predate triage rule 7 (Low-priority monitor).** Under the current policy the predicted-Safe events below the auto-clear bar that appear here as Human review are routed to Low-priority monitor; rerun `train` and `predict` with the data in place to refresh the numbers.
- Metrics are computed on the 20% event holdout of train_data only; test_data has no final outcomes and is used solely for demo predictions.
- Trivial events (final CDM itself at >= 2 days) are reported separately from real_forecast events; headline numbers use real_forecast.
- Probabilities are per-class isotonic-calibrated on dev out-of-fold predictions and renormalised; decision and triage thresholds are tuned on dev OOF only.
- risk of CDMs at >= 2 days is used as a past input when features.use_past_risk is true (the carry-forward baseline uses the same information).
- Zero final-Critical events auto-cleared on dev OOF is a tuning target, not a guarantee; the holdout count is reported.
