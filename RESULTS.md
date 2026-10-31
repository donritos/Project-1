# Results of the shipped run

Task: forecast each event's **final** tier (Safe / Elevated / Critical) from CDMs at >= 2 days before TCA. Primary model: `lgbm_ordinal` (version `lgbm_ordinal-20261031T084334Z`). Generated 2026-10-31T08:44:12.807763+00:00. Total training time 261.6 s.

Holdout: 2389 events (1779 real forecasts, 610 trivial), split seed 42, holdout fraction 0.200, split hash `d18e932157a7c887`. Final-tier counts on the holdout: {"Safe": 1790, "Elevated": 432, "Critical": 167}.

Subsets: `all_usable` = every holdout event; `trivial` = the event's final CDM was itself at >= 2 days (the answer is visible in the input); `real_forecast` = everything else and the headline subset.

## Headline: real-forecast holdout

Headline metric: **Critical recall on real_forecast events**, read together with Critical precision / F2. `carry_forward_risk` is the official naive baseline (the latest known tier carried forward).

| Model | Events | Critical recall | Critical precision | Critical F2 | Elev+ F2 | Macro-F1 | Brier | Log loss | Auto-clear | Low-priority monitor | Human review | Final Crit auto-cleared | Final Crit not urgent | Watch share | Escalation recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carry_forward_risk | 1779 | 0.797 | 0.764 | 0.790 | 0.918 | 0.858 | 0.142 | 1.957 | 0.0% | 70.5% | 3.5% | 0 | 24 | 0.0% | 0.000 |
| **lgbm_ordinal** | 1779 | 0.890 | 0.724 | 0.851 | 0.919 | 0.850 | 0.109 | 0.204 | 19.2% | 48.8% | 3.5% | 0 | 6 | 4.9% | 0.750 |
| carry_forward_mre | 1779 | 0.797 | 0.686 | 0.772 | 0.813 | 0.700 | 0.527 | 7.284 | 0.0% | 51.1% | 3.5% | 0 | 24 | 0.0% | 0.000 |
| majority | 1779 | 0.000 | undefined | 0.000 | 0.000 | 0.284 | 0.512 | 7.067 | 0.0% | 70.5% | 29.5% | 0 | 3 | 0.0% | 0.000 |
| logistic | 1779 | 0.864 | 0.689 | 0.823 | 0.902 | 0.825 | 0.144 | 0.278 | 32.3% | 35.0% | 4.2% | 2 | 5 | 5.8% | 0.792 |
| lgbm_multiclass | 1779 | 0.881 | 0.727 | 0.846 | 0.920 | 0.849 | 0.106 | 0.202 | 11.6% | 56.1% | 3.5% | 0 | 5 | 4.4% | 0.792 |

`lgbm_ordinal` beats `carry_forward_risk` on Critical F2 (real_forecast): 0.851 vs 0.790.

### All usable holdout events

| Model | Events | Critical recall | Critical precision | Critical F2 | Elev+ F2 | Macro-F1 | Brier | Log loss | Auto-clear | Low-priority monitor | Human review | Final Crit auto-cleared | Final Crit not urgent | Watch share | Escalation recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carry_forward_risk | 2389 | 0.856 | 0.831 | 0.851 | 0.938 | 0.894 | 0.105 | 1.457 | 0.0% | 71.2% | 3.4% | 0 | 24 | 0.0% | 0.000 |
| **lgbm_ordinal** | 2389 | 0.922 | 0.748 | 0.881 | 0.930 | 0.862 | 0.098 | 0.186 | 16.4% | 51.9% | 3.4% | 0 | 6 | 4.6% | 0.750 |
| carry_forward_mre | 2389 | 0.856 | 0.769 | 0.837 | 0.835 | 0.745 | 0.454 | 6.269 | 0.0% | 53.7% | 3.4% | 0 | 24 | 0.0% | 0.000 |
| majority | 2389 | 0.000 | undefined | 0.000 | 0.000 | 0.286 | 0.501 | 6.928 | 0.0% | 71.2% | 28.8% | 0 | 3 | 0.0% | 0.000 |
| logistic | 2389 | 0.904 | 0.706 | 0.856 | 0.890 | 0.826 | 0.142 | 0.270 | 34.9% | 33.0% | 4.6% | 2 | 5 | 5.5% | 0.792 |
| lgbm_multiclass | 2389 | 0.916 | 0.757 | 0.879 | 0.934 | 0.868 | 0.093 | 0.180 | 10.5% | 57.8% | 3.3% | 0 | 5 | 4.6% | 0.792 |

### Trivial events (answer visible in the input)

| Model | Events | Critical recall | Critical precision | Critical F2 | Elev+ F2 | Macro-F1 | Brier | Log loss | Auto-clear | Low-priority monitor | Human review | Final Crit auto-cleared | Final Crit not urgent | Watch share | Escalation recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carry_forward_risk | 610 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | -0.000 | 0.0% | 73.3% | 3.1% | 0 | 0 | 0.0% | undefined |
| **lgbm_ordinal** | 610 | 1.000 | 0.803 | 0.953 | 0.964 | 0.895 | 0.068 | 0.135 | 8.2% | 61.0% | 3.1% | 0 | 0 | 3.8% | undefined |
| carry_forward_mre | 610 | 1.000 | 1.000 | 1.000 | 0.908 | 0.879 | 0.239 | 3.307 | 0.0% | 61.3% | 3.1% | 0 | 0 | 0.0% | undefined |
| majority | 610 | 0.000 | undefined | 0.000 | 0.000 | 0.289 | 0.472 | 6.523 | 0.0% | 73.3% | 26.7% | 0 | 0 | 0.0% | undefined |
| logistic | 610 | 1.000 | 0.742 | 0.935 | 0.849 | 0.816 | 0.139 | 0.248 | 42.5% | 27.2% | 6.1% | 0 | 0 | 4.6% | undefined |
| lgbm_multiclass | 610 | 1.000 | 0.831 | 0.961 | 0.980 | 0.921 | 0.055 | 0.118 | 7.5% | 62.6% | 2.8% | 0 | 0 | 4.9% | undefined |

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
| Safe | 1248 | 73 | 3 |
| Elevated | 25 | 275 | 37 |
| Critical | 3 | 10 | 105 |

Row-normalised (share of each true tier):

| true \ pred | Safe | Elevated | Critical |
|---|---|---|---|
| Safe | 94.3% | 5.5% | 0.2% |
| Elevated | 7.4% | 81.6% | 11.0% |
| Critical | 2.5% | 8.5% | 89.0% |

### `lgbm_ordinal` on all_usable (2389 events)

Counts (rows = true final tier, columns = predicted):

| true \ pred | Safe | Elevated | Critical |
|---|---|---|---|
| Safe | 1691 | 96 | 3 |
| Elevated | 26 | 357 | 49 |
| Critical | 3 | 10 | 154 |

Row-normalised (share of each true tier):

| true \ pred | Safe | Elevated | Critical |
|---|---|---|---|
| Safe | 94.5% | 5.4% | 0.2% |
| Elevated | 6.0% | 82.6% | 11.3% |
| Critical | 1.8% | 6.0% | 92.2% |

## Escalation watch

Among events that are not Critical at the forecast time: how many end Critical, how many the forecast flags (predicted Critical or placed on the escalation watch list), and how many of those flags are hits. Among events that are Critical now: how many de-escalate and how the forecast handles them.

| Model | Subset | Not-Critical now | End Critical | Flagged | Caught | Escalation recall | Flag precision | Watch size | Watch hits | Critical now | De-escalate | Pred. de-esc. | Correct | False de-esc. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carry_forward_risk | real_forecast | 1656 | 24 | 0 | 0 | 0.000 | undefined | 0 | 0 | 123 | 29 | 0 | 0 | 0 |
| carry_forward_risk | all_usable | 2217 | 24 | 0 | 0 | 0.000 | undefined | 0 | 0 | 172 | 29 | 0 | 0 | 0 |
| lgbm_ordinal | real_forecast | 1656 | 24 | 109 | 18 | 0.750 | 0.165 | 85 | 7 | 123 | 29 | 2 | 2 | 0 |
| lgbm_ordinal | all_usable | 2217 | 24 | 144 | 18 | 0.750 | 0.125 | 108 | 7 | 172 | 29 | 2 | 2 | 0 |
| carry_forward_mre | real_forecast | 1656 | 24 | 14 | 0 | 0.000 | 0.000 | 0 | 0 | 123 | 29 | 0 | 0 | 0 |
| carry_forward_mre | all_usable | 2217 | 24 | 14 | 0 | 0.000 | 0.000 | 0 | 0 | 172 | 29 | 0 | 0 | 0 |
| majority | real_forecast | 1656 | 24 | 0 | 0 | 0.000 | undefined | 0 | 0 | 123 | 29 | 123 | 29 | 94 |
| majority | all_usable | 2217 | 24 | 0 | 0 | 0.000 | undefined | 0 | 0 | 172 | 29 | 172 | 29 | 143 |
| logistic | real_forecast | 1656 | 24 | 129 | 19 | 0.792 | 0.147 | 102 | 10 | 123 | 29 | 2 | 1 | 1 |
| logistic | all_usable | 2217 | 24 | 174 | 19 | 0.792 | 0.109 | 130 | 10 | 172 | 29 | 2 | 1 | 1 |
| lgbm_multiclass | real_forecast | 1656 | 24 | 99 | 19 | 0.792 | 0.192 | 77 | 9 | 123 | 29 | 2 | 2 | 0 |
| lgbm_multiclass | all_usable | 2217 | 24 | 139 | 19 | 0.792 | 0.137 | 107 | 9 | 172 | 29 | 2 | 2 | 0 |

## Auto-clear and triage mix (primary model)

| Subset | Events | Auto-clear share | Final-Critical auto-cleared | Final Critical | Watch share | Final-Critical in watch | Maneuver review | Human review | Monitor | Low-priority monitor | Final-Critical not urgent |
|---|---|---|---|---|---|---|---|---|---|---|---|
| real_forecast | 1779 | 19.2% | 0 | 118 | 4.9% | 7 | 8.2% | 3.5% | 15.4% | 48.8% | 6 |
| all_usable | 2389 | 16.4% | 0 | 167 | 4.6% | 7 | 8.6% | 3.4% | 15.0% | 51.9% | 6 |
| trivial | 610 | 8.2% | 0 | 49 | 3.8% | 0 | 10.0% | 3.1% | 13.9% | 61.0% | 0 |

Review workload = Maneuver planning review + Escalation watch + Human review. Auto-clear and low-priority monitor shares are *potential* workload reductions on the holdout, not demonstrated labour savings; an operator study has not been run.

True final tier by triage action, real_forecast (every routing failure is visible here, not only final-Critical events that were auto-cleared):

| Action | Safe | Elevated | Critical | Total |
|---|---|---|---|---|
| Maneuver planning review | 3 | 37 | 105 | 145 |
| Escalation watch | 11 | 69 | 7 | 87 |
| Human review | 60 | 2 | 0 | 62 |
| Monitor | 65 | 206 | 3 | 274 |
| Low-priority monitor | 843 | 23 | 3 | 869 |
| Auto-clear candidate | 342 | 0 | 0 | 342 |

True final tier by triage action, all_usable (every routing failure is visible here, not only final-Critical events that were auto-cleared):

| Action | Safe | Elevated | Critical | Total |
|---|---|---|---|---|
| Maneuver planning review | 3 | 49 | 154 | 206 |
| Escalation watch | 15 | 88 | 7 | 110 |
| Human review | 78 | 3 | 0 | 81 |
| Monitor | 87 | 269 | 3 | 359 |
| Low-priority monitor | 1215 | 23 | 3 | 1241 |
| Auto-clear candidate | 392 | 0 | 0 | 392 |

No event that ended Critical was auto-cleared on the holdout.

Action counts (primary model, all usable holdout events):

| Action | Events |
|---|---|
| Auto-clear candidate | 392 |
| Escalation watch | 110 |
| Human review | 81 |
| Low-priority monitor | 1241 |
| Maneuver planning review | 206 |
| Monitor | 359 |

## Calibration

| Model | Subset | n | Brier (multiclass) | Log loss | Brier (Critical) | Critical PR-AUC | Critical ROC-AUC | ECE (Critical) |
|---|---|---|---|---|---|---|---|---|
| carry_forward_risk | real_forecast | 1779 | 0.1417 | 1.9570 | 0.0298 | 0.622 | 0.890 | 0.0298 |
| carry_forward_risk | all_usable | 2389 | 0.1055 | 1.4573 | 0.0222 | 0.722 | 0.922 | 0.0222 |
| lgbm_ordinal | real_forecast | 1779 | 0.1086 | 0.2037 | 0.0208 | 0.885 | 0.978 | 0.0112 |
| lgbm_ordinal | all_usable | 2389 | 0.0982 | 0.1862 | 0.0182 | 0.906 | 0.984 | 0.0092 |
| carry_forward_mre | real_forecast | 1779 | 0.5273 | 7.2844 | 0.0377 | 0.560 | 0.885 | 0.0377 |
| carry_forward_mre | all_usable | 2389 | 0.4537 | 6.2687 | 0.0280 | 0.668 | 0.918 | 0.0280 |
| majority | real_forecast | 1779 | 0.5115 | 7.0670 | 0.0663 | 0.066 | 0.500 | 0.0663 |
| majority | all_usable | 2389 | 0.5015 | 6.9280 | 0.0699 | 0.070 | 0.500 | 0.0699 |
| logistic | real_forecast | 1779 | 0.1436 | 0.2777 | 0.0247 | 0.791 | 0.974 | 0.0149 |
| logistic | all_usable | 2389 | 0.1424 | 0.2701 | 0.0230 | 0.838 | 0.981 | 0.0145 |
| lgbm_multiclass | real_forecast | 1779 | 0.1062 | 0.2019 | 0.0200 | 0.890 | 0.974 | 0.0114 |
| lgbm_multiclass | all_usable | 2389 | 0.0932 | 0.1804 | 0.0176 | 0.913 | 0.981 | 0.0102 |

Reliability of P(Critical) for `lgbm_ordinal` on real_forecast holdout events (10 equal-width bins):

| Bin | Mean predicted | Observed Critical share | Events |
|---|---|---|---|
| 0.0-0.1 | 0.005 | 0.006 | 1610 |
| 0.1-0.2 | 0.147 | 0.200 | 15 |
| 0.2-0.3 | 0.212 | 0.111 | 9 |
| 0.3-0.4 | 0.367 | 0.308 | 13 |
| 0.4-0.5 | 0.478 | 0.444 | 9 |
| 0.5-0.6 | 0.551 | 0.500 | 4 |
| 0.7-0.8 | 0.777 | 0.000 | 6 |
| 0.8-0.9 | 0.847 | 0.562 | 16 |
| 0.9-1.0 | 0.954 | 0.887 | 97 |

Calibration method: `isotonic` fitted on dev out-of-fold predictions (dev OOF Brier 0.0978, log loss 0.1817).

## Bootstrap confidence intervals

Point estimate followed by the 95 % event-bootstrap interval (2.5 / 97.5 percentiles).

| Model | Subset | Resamples | Critical recall | Critical precision | Macro-F1 | Auto-clear share |
|---|---|---|---|---|---|---|
| lgbm_ordinal | real_forecast | 1000 | 0.890 [0.833, 0.943] | 0.724 [0.654, 0.799] | 0.850 [0.826, 0.874] | 19.2% [0.175, 0.210] |
| lgbm_ordinal | all_usable | 1000 | 0.922 [0.878, 0.960] | 0.748 [0.689, 0.804] | 0.862 [0.841, 0.882] | 16.4% [0.149, 0.180] |
| carry_forward_risk | real_forecast | 1000 | 0.797 [0.724, 0.865] | 0.764 [0.683, 0.836] | 0.858 [0.833, 0.884] | 0.0% [0.000, 0.000] |
| carry_forward_risk | all_usable | 1000 | 0.856 [0.801, 0.907] | 0.831 [0.776, 0.886] | 0.894 [0.875, 0.914] | 0.0% [0.000, 0.000] |

## Breakdowns (primary model)

### Lead time of the latest snapshot, real_forecast

| Days to TCA | Events | Final Critical | Predicted Critical | Critical recall | Critical precision | Critical F2 | Macro-F1 | Accuracy |
|---|---|---|---|---|---|---|---|---|
| [2,3) | 1701 | 113 | 140 | 0.894 | 0.721 | 0.853 | 0.855 | 0.918 |
| [3,4) | 30 | 3 | 2 | 0.667 | 1.000 | 0.714 | 0.567 | 0.833 |
| [4,5) | 18 | 1 | 1 | 1.000 | 1.000 | 1.000 | 0.657 | 0.944 |
| [5,inf) | 30 | 1 | 2 | 1.000 | 0.500 | 0.833 | 0.523 | 0.800 |

### Object type, real_forecast

| Object type | Events | Final Critical | Predicted Critical | Critical recall | Critical precision | Critical F2 | Macro-F1 | Accuracy |
|---|---|---|---|---|---|---|---|---|
| DEBRIS | 991 | 74 | 94 | 0.905 | 0.713 | 0.859 | 0.855 | 0.910 |
| PAYLOAD | 224 | 9 | 9 | 0.778 | 0.778 | 0.778 | 0.850 | 0.951 |
| ROCKET BODY | 20 | 5 | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| UNKNOWN | 543 | 30 | 37 | 0.867 | 0.703 | 0.828 | 0.821 | 0.906 |
| other (TBA) | 1 | 0 | 0 | undefined | undefined | undefined | 1.000 | 1.000 |

### Lead time of the latest snapshot, all_usable

| Days to TCA | Events | Final Critical | Predicted Critical | Critical recall | Critical precision | Critical F2 | Macro-F1 | Accuracy |
|---|---|---|---|---|---|---|---|---|
| [2,3) | 1802 | 124 | 153 | 0.903 | 0.732 | 0.863 | 0.858 | 0.920 |
| [3,4) | 183 | 17 | 18 | 0.941 | 0.889 | 0.930 | 0.863 | 0.918 |
| [4,5) | 160 | 12 | 15 | 1.000 | 0.800 | 0.952 | 0.906 | 0.963 |
| [5,inf) | 244 | 14 | 20 | 1.000 | 0.700 | 0.921 | 0.853 | 0.910 |

### Object type, all_usable

| Object type | Events | Final Critical | Predicted Critical | Critical recall | Critical precision | Critical F2 | Macro-F1 | Accuracy |
|---|---|---|---|---|---|---|---|---|
| DEBRIS | 1291 | 105 | 133 | 0.933 | 0.737 | 0.886 | 0.865 | 0.916 |
| PAYLOAD | 275 | 11 | 11 | 0.818 | 0.818 | 0.818 | 0.866 | 0.956 |
| ROCKET BODY | 28 | 5 | 6 | 1.000 | 0.833 | 0.962 | 0.939 | 0.964 |
| UNKNOWN | 792 | 46 | 56 | 0.913 | 0.750 | 0.875 | 0.849 | 0.918 |
| other (TBA) | 3 | 0 | 0 | undefined | undefined | undefined | 1.000 | 1.000 |

### Current tier at forecast time, real_forecast

| Current tier | Events | Final Critical | Predicted Critical | Critical recall | Critical precision | Critical F2 | Macro-F1 | Accuracy |
|---|---|---|---|---|---|---|---|---|
| Critical | 123 | 94 | 121 | 1.000 | 0.777 | 0.946 | 0.339 | 0.780 |
| Elevated | 339 | 21 | 24 | 0.524 | 0.458 | 0.509 | 0.460 | 0.811 |
| Safe | 1317 | 3 | 0 | 0.000 | undefined | 0.000 | 0.406 | 0.954 |

## Ablations (real_forecast holdout)

| Model | Events | Critical recall | Critical precision | Critical F2 | Elev+ F2 | Macro-F1 | Brier | Log loss | Auto-clear | Low-priority monitor | Human review | Final Crit auto-cleared | Final Crit not urgent | Watch share | Escalation recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lgbm_ordinal (primary) | 1779 | 0.890 | 0.724 | 0.851 | 0.919 | 0.850 | 0.109 | 0.204 | 19.2% | 48.8% | 3.5% | 0 | 6 | 4.9% | 0.750 |
| use_past_risk_off | 1779 | 0.890 | 0.700 | 0.844 | 0.914 | 0.837 | 0.111 | 0.215 | 17.7% | 31.6% | 21.4% | 0 | 6 | 5.3% | 0.750 |
| latest_only_training | 1779 | 0.822 | 0.735 | 0.803 | 0.920 | 0.851 | 0.118 | 0.234 | 25.9% | 43.6% | 3.5% | 1 | 6 | 5.3% | 0.750 |
| no_history_features | 1779 | 0.881 | 0.712 | 0.841 | 0.920 | 0.853 | 0.113 | 0.226 | 24.7% | 44.6% | 3.7% | 1 | 6 | 5.6% | 0.750 |

## Mission-held-out cross-validation

Out-of-fold predictions over all usable events with folds grouped by `mission_id` (mission_id is never a feature). This checks that the model is not memorising missions.

| Model | Events | Critical recall | Critical precision | Critical F2 | Elev+ F2 | Macro-F1 | Brier | Log loss | Auto-clear | Low-priority monitor | Human review | Final Crit auto-cleared | Final Crit not urgent | Watch share | Escalation recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mission_heldout_oof | 8892 | 0.881 | 0.704 | 0.839 | 0.920 | 0.842 | 0.111 | 0.208 | 12.8% | 54.6% | 3.6% | 1 | 28 | 5.3% | 0.750 |

| Model | Events | Critical recall | Critical precision | Critical F2 | Elev+ F2 | Macro-F1 | Brier | Log loss | Auto-clear | Low-priority monitor | Human review | Final Crit auto-cleared | Final Crit not urgent | Watch share | Escalation recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mission_heldout_oof | 11942 | 0.915 | 0.735 | 0.872 | 0.930 | 0.858 | 0.100 | 0.189 | 10.6% | 57.4% | 3.6% | 1 | 28 | 5.0% | 0.750 |

## Conformal prediction sets

Nominal coverage 0.900, q_hat 0.3210; empirical holdout coverage 0.941, average set size 1.00 (all usable events).

real_forecast: marginal coverage 0.931, average set size 1.00, singleton share 100.0%, decision outside set 3.4%.

| True final tier | Coverage | Average set size |
|---|---|---|
| Safe | 0.968 | 1.00 |
| Elevated | 0.819 | 1.00 |
| Critical | 0.839 | 1.00 |

all_usable: marginal coverage 0.941, average set size 1.00, singleton share 100.0%, decision outside set 3.9%.

| True final tier | Coverage | Average set size |
|---|---|---|
| Safe | 0.974 | 1.00 |
| Elevated | 0.826 | 1.00 |
| Critical | 0.880 | 1.00 |

When every set is a singleton the sets carry no information beyond the point decision; marginal coverage above nominal then comes from the majority Safe class, and the class-conditional table above is the honest statement of what the sets guarantee. No class-conditional coverage guarantee is claimed.

## Illustrative cases (primary model, holdout)

### Missed Critical (ended Critical, forecast lower, not on the watch list): 3 shown

| Event | Lead time (d) | Miss (m) | CDMs | Current | Final | Forecast | P(Critical) | Set | Action | Reasons |
|---|---|---|---|---|---|---|---|---|---|---|
| kelvins_train:2958 | 2.54 | 34514 | 1 | Safe | Critical | Safe | 0.000 | Safe | Low-priority monitor | NO_HISTORY, FORECAST_SAFE, LOW_CONFIDENCE_SAFE |
| kelvins_train:8732 | 2.34 | 28737 | 14 | Safe | Critical | Safe | 0.000 | Safe | Low-priority monitor | FORECAST_SAFE, RISK_RISING, MISS_SHRINKING, SIGMA_GROWING, HIGH_VOLATILITY, LOW_CONFIDENCE_SAFE |
| kelvins_train:3183 | 3.03 | 8524 | 6 | Safe | Critical | Safe | 0.006 | Safe | Low-priority monitor | FORECAST_SAFE, HIGH_VOLATILITY, LOW_CONFIDENCE_SAFE |

### Caught by the escalation watch (ended Critical, forecast lower, on the watch list): 3 shown

| Event | Lead time (d) | Miss (m) | CDMs | Current | Final | Forecast | P(Critical) | Set | Action | Reasons |
|---|---|---|---|---|---|---|---|---|---|---|
| kelvins_train:7848 | 2.27 | 1426 | 6 | Elevated | Critical | Elevated | 0.033 | Elevated | Escalation watch | FORECAST_ELEVATED, CURRENT_TIER_ELEVATED, MISS_SHRINKING, HIGH_VOLATILITY, ESCALATION_RISK |
| kelvins_train:11943 | 2.29 | 1653 | 15 | Elevated | Critical | Elevated | 0.066 | Elevated | Escalation watch | FORECAST_ELEVATED, CURRENT_TIER_ELEVATED, MISS_SHRINKING, HIGH_VOLATILITY, ESCALATION_RISK |
| kelvins_train:5025 | 2.01 | 1261 | 5 | Elevated | Critical | Elevated | 0.067 | Elevated | Escalation watch | FORECAST_ELEVATED, CURRENT_TIER_ELEVATED, RISK_RISING, MISS_SHRINKING, SIGMA_GROWING, HIGH_VOLATILITY, ESCALATION_RISK |

### False alarm (forecast Critical, ended lower): 3 shown

| Event | Lead time (d) | Miss (m) | CDMs | Current | Final | Forecast | P(Critical) | Set | Action | Reasons |
|---|---|---|---|---|---|---|---|---|---|---|
| kelvins_train:9929 | 2.25 | 392 | 2 | Critical | Safe | Critical | 0.966 | Critical | Maneuver planning review | FORECAST_CRITICAL, CURRENT_TIER_CRITICAL |
| kelvins_train:11851 | 2.05 | 325 | 15 | Critical | Elevated | Critical | 0.966 | Critical | Maneuver planning review | FORECAST_CRITICAL, CURRENT_TIER_CRITICAL, RISK_RISING |
| kelvins_train:2960 | 2.43 | 189 | 4 | Critical | Elevated | Critical | 0.962 | Critical | Maneuver planning review | FORECAST_CRITICAL, CURRENT_TIER_CRITICAL, MISS_SHRINKING |

### False de-escalation (Critical now, forecast lower, ended Critical): 0 shown

None on the holdout.

### Most uncertain forecasts: 3 shown

| Event | Lead time (d) | Miss (m) | CDMs | Current | Final | Forecast | P(Critical) | Set | Action | Reasons |
|---|---|---|---|---|---|---|---|---|---|---|
| kelvins_train:11630 | 3.10 | 617 | 8 | Critical | Critical | Critical | 0.430 | Critical | Maneuver planning review | FORECAST_CRITICAL, CURRENT_TIER_CRITICAL, RISK_RISING, MISS_SHRINKING |
| kelvins_train:7754 | 2.28 | 786 | 4 | Critical | Critical | Critical | 0.430 | Critical | Maneuver planning review | FORECAST_CRITICAL, CURRENT_TIER_CRITICAL, RISK_RISING, MISS_SHRINKING, HIGH_VOLATILITY |
| kelvins_train:9787 | 2.07 | 732 | 13 | Critical | Elevated | Critical | 0.454 | Critical | Maneuver planning review | FORECAST_CRITICAL, CURRENT_TIER_CRITICAL, RISK_RISING, MISS_SHRINKING, SIGMA_GROWING, HIGH_VOLATILITY |

## Test-set predictions (Kelvins test file)

| Predicted final tier | Events | Share |
|---|---|---|
| Safe | 1439 | 66.4% |
| Elevated | 510 | 23.5% |
| Critical | 218 | 10.1% |

The Kelvins test set was enriched with high-risk events, so its predicted class distribution is expected to be shifted toward Elevated/Critical relative to the training labels.

Batch run: 2167 events / 24484 snapshots; status counts {"valid": 2002, "warning": 165}; triage counts {"Low-priority monitor": 983, "Auto-clear candidate": 384, "Monitor": 352, "Maneuver planning review": 218, "Escalation watch": 171, "Human review": 59}; batch inference 1.3 s.

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

Training rows: `snapshots` chosen on dev OOF Critical F2 (real_forecast): snapshots = {'critical_recall': 0.8728813559322034, 'critical_precision': 0.7330960854092526, 'critical_f2': 0.8408163265306123, 'macro_f1': 0.8506237291339422}, latest = {'critical_recall': 0.8220338983050848, 'critical_precision': 0.7729083665338645, 'critical_f2': 0.8117154811715482, 'macro_f1': 0.8582591602128004}.

Grouped CV search (5 folds, 12 configurations, rule: max OOF Critical F2 on real_forecast latest snapshots (t=0.5), tie-break macro-F1). Best params `{"learning_rate": 0.04679255978730041, "num_leaves": 12, "min_child_samples": 190, "feature_fraction": 0.9498034588173512, "lambda_l2": 1.0}`, scores `{"critical_recall": 0.8728813559322034, "critical_precision": 0.7330960854092526, "critical_f2": 0.8408163265306123, "macro_f1": 0.8506237291339422}`.

Decision thresholds (tuned on dev OOF only): `t_crit` = 0.240, `t_elev` = 0.300, `t_watch` = 0.028, `t_clear` = 0.999.

Policy tuning summary: `{"t_watch": 0.027596627078912576, "t_clear": 0.9994272636625692, "watch": {"watch_max_share": 0.05, "watch_share": 0.04993195854705328, "n_watch": 477, "n_escalations_caught": 29, "n_escalations_total": 55, "recall": 0.5272727272727272, "n_events": 9553}, "auto_clear": {"n_eligible": 6522, "n_eligible_final_critical": 6, "auto_clear_share": 0.12875536480686695, "n_auto_clear": 1230, "n_final_critical_cleared": 0, "zero_missed_achievable": true, "note": "zero final-Critical cleared at t_clear", "n_events": 9553}}`


## Notes

- Metrics are computed on the 20% event holdout of train_data only; test_data has no final outcomes and is used solely for demo predictions.
- Trivial events (final CDM itself at >= 2 days) are reported separately from real_forecast events; headline numbers use real_forecast.
- Probabilities are per-class isotonic-calibrated on dev out-of-fold predictions and renormalised; decision and triage thresholds are tuned on dev OOF only.
- risk of CDMs at >= 2 days is used as a past input when features.use_past_risk is true (the carry-forward baseline uses the same information).
- Zero final-Critical events auto-cleared on dev OOF is a tuning target, not a guarantee; the holdout count is reported.
