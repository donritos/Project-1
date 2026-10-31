# Shared schema: event-forecast frame (final-tier task)

One row per **snapshot** (a CDM with `time_to_tca >= 2.0` days). The row whose
`is_latest_snapshot` is True is the event forecast used for evaluation, test export, and triage.
Policy, evaluation and explanation modules consume this frame; `pipeline.py` produces it.

| Column | Type | Meaning |
|---|---|---|
| `dataset_id` | str | file/dataset the event came from (`kelvins_train`, `kelvins_test`, uploads) |
| `event_id` | int | event id, unique only within a dataset; never join datasets on it |
| `row_id` | str | snapshot CDM row id (`dataset_id:source_row_index`) |
| `time_to_tca_days` | float | snapshot's time to closest approach |
| `miss_distance_m` | float | snapshot's miss distance |
| `is_latest_snapshot` | bool | True on the event's last CDM at >= 2 days |
| `n_cdms_2d` | int | CDMs at >= 2 days seen so far (incl. snapshot) |
| `has_history` | int (0/1) | 0 when the snapshot is the first CDM of the event |
| `current_tier` | str | official tier of the snapshot from (`risk` if `use_past_risk` else `max_risk_estimate`) and `miss_distance` |
| `current_tier_source` | str | `"risk"` or `"mre"` |
| `latest_risk` | float, nullable | snapshot's own `risk`; NaN when `use_past_risk` is off or unavailable (test demo keeps it) |
| `prediction_status` | str | `valid` / `warning` / `invalid` |
| `quality_flags` | list[str] | from `data.validate_rows` |
| `p_safe`, `p_elevated`, `p_critical` | float | calibrated class probabilities, sum to 1 (NaN for invalid rows) |
| `predicted_final_tier` | str, nullable | result of the frozen decision rule (`Safe`/`Elevated`/`Critical`); None if invalid |
| `prediction_set` | list[str], nullable | optional conformal prediction set at 90 % |
| `slope_log_miss` | float | least-squares slope of log10 miss distance vs `time_to_tca` over snapshots so far (per day of time_to_tca; **negative slope = miss shrinking as TCA approaches**... see sign note) |
| `slope_risk_or_mre` | float | slope of `risk` (or `max_risk_estimate` when past risk is off) vs `time_to_tca` |
| `slope_log_sigma` | float | slope of log10 combined radial sigma vs `time_to_tca` |
| `volatility_mre` | float | std of `max_risk_estimate` over snapshots so far |
| `volatility_risk` | float, nullable | std of `risk` over snapshots so far (NaN if past risk off) |
| `carry_forward_tier` | str | = `current_tier` of the latest snapshot (the official naive baseline) |
| `triage_action` | str | assigned by `policy.assign_triage` |
| `reason_codes` | list[str] | assigned by `policy.assign_triage` |
| `explanation_id` | str, nullable | link to explanation record |
| `model_version`, `feature_version`, `policy_version` | str | provenance |

**Evaluation-only columns (train/holdout frames):**

| Column | Type | Meaning |
|---|---|---|
| `final_tier` | str | ground truth from the labels workbook |
| `final_risk_log10_pc` | float | ground truth final risk |
| `final_miss_distance_m` | float | ground truth final miss distance |
| `is_trivial` | bool | event's final CDM is itself at >= 2 days (answer visible in input) |
| `object_type` | str | `c_object_type` of the snapshot (for breakdowns) |
| `mission_id` | int | from the labels file, **never a feature**; for mission-held-out CV only |
| `split` | str | `dev` or `holdout` |

**Sign note for slopes.** Slopes are taken against `time_to_tca`, which *decreases* as the
event progresses. So `slope_log_miss > 0` means miss distance was larger earlier, i.e.
**shrinking** toward TCA; `slope_risk_or_mre < 0` means risk was lower earlier, i.e. **rising**.
Policy thresholds in `configs/default.yaml` under `triage.trends` use these conventions:
`MISS_SHRINKING` when `slope_log_miss >= miss_shrink_slope`, `RISK_RISING` when
`slope_risk_or_mre <= risk_rise_slope`, `SIGMA_GROWING` when `slope_log_sigma <= sigma_grow_slope`,
`HIGH_VOLATILITY` when `volatility_mre >= mre_volatility` (or `volatility_risk >= risk_volatility`).

**Subsets for every metric:** `all_usable` (all events), `trivial` (`is_trivial`), `real_forecast` (`~is_trivial`).

**Decision thresholds** live in the bundle meta (`t_crit`, `t_elev`, `t_watch`, `t_clear`) and are
tuned on dev out-of-fold predictions only.
