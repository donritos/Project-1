# Collision-risk triage engine (Autonomous SG 2026, Airbus Challenge #1)

Prediction engine for a satellite conjunction triage product. For every conjunction
data message (CDM) it produces an estimated collision probability, a `log10(Pc)` point
prediction with a calibrated prediction interval, the official challenge risk tier, a
separate recommended triage action, reason codes, data-quality flags, feature
contributions, and full provenance (model, feature, and policy versions).

The user is a satellite operations analyst. The system supports their decision; it does
not plan or execute manoeuvres.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) (`pip install uv`). uv fetches Python 3.11 and
installs the locked dependencies.

```bash
uv sync --group dev
```

Place the challenge files at `data/raw/train_data.csv` and `data/raw/test_data.csv`, then:

```bash
uv run collision-risk audit
```

```bash
uv run collision-risk train
```

```bash
uv run collision-risk predict
```

```bash
uv run collision-risk evaluate
```

Or everything at once (`--quick` runs a tiny search for smoke testing):

```bash
uv run collision-risk run-all
```

Tests (leakage, rule boundaries, features, inference, contracts):

```bash
uv run pytest
```

`predict` accepts any CSV with the permitted columns, e.g.
`uv run collision-risk predict --input my_cdms.csv --dataset-id ops_2026_10 --out-dir artifacts/ops`.

## What the pipeline does

| Stage | Module | Output |
|---|---|---|
| Audit | `data.py` | `artifacts/audit.json`, allowlisted parquet cache |
| Features | `features.py` | one pipeline for training, batch and uploads |
| Splits and models | `train.py` | event-level stratified partition, baselines, grouped-CV LightGBM search, candidates |
| Intervals | `calibration.py` | conformalized quantile regression (CQR) and residual split conformal, one score per event |
| Tiers and triage | `policy.py` | official tier rules (single implementation) and the review policy |
| Evaluation | `evaluate.py` | `artifacts/evaluation.json` with row-level and latest-CDM views, bootstrap, failure cases |
| Explanations | `explain.py` | native LightGBM contributions, `artifacts/explanations.parquet` |
| Serving | `service.py` | `load_bundle`, `predict_batch`, `get_event_history`, `get_explanation`, `get_evaluation_report` |
| Orchestration | `pipeline.py`, `cli.py` | documented commands |

### Input restrictions

Only these source columns may enter feature engineering:
`event_id, time_to_tca, miss_distance, relative_speed, relative_position_r/t/n, c_object_type, c_sigma_r, t_sigma_r, max_risk_estimate, F10, AP`.
`risk = log10(Pc)` is used only as the target. `event_id` is used for grouping and
traceability, never as a numeric feature. The allowlist is applied at load time and
again at the predictor's entry point; the feature pipeline raises if any other column is
present (the delivered files carry 89 extra columns, all dropped and listed in the audit).

### Audit findings that shaped the design (see `artifacts/audit.json`)

- Units are metres, m/s and days as expected; the RTN vector norm matches miss distance
  within a 1 percent relative tolerance on 99.95 percent of rows.
- `max_risk_estimate` is already `log10(Pc)` (range −9.8 to −1.1). It is **not** a strict
  upper bound: the target exceeds it in 0.35 percent of rows, so predictions are not
  clipped against it. Both facts are stored in `configs/default.yaml`.
- 41 percent of training rows sit at the −30 floor (no meaningful probability).
- `event_id` values are reused between the train and test files, so events are only
  unique within a file. All identifiers are `(dataset_id, event_id)`; row ids are
  `dataset_id:source_row_index`.
- Test rows stop two days before TCA, so the report includes a matching horizon slice.
- Some `c_sigma_r` and `t_sigma_r` values equal Earth's radius (placeholders); they are
  flagged `SUSPICIOUS_SIGMA` and left to the model with the flag.
- F10 and AP are missing on about 4 percent of rows; missingness is a feature and the
  only quality flag that does not block auto-clear.

### Official tier rules (`policy.assign_risk_tier`)

Applied most severe first, in metres:

1. `risk_pred >= -4` or `miss_distance < 1000` → Critical
2. `risk_pred >= -6` or `1000 <= miss_distance <= 5000` → Elevated (the band is inclusive at both ends)
3. otherwise → Safe

Invalid rows (missing required geometry, non-positive miss distance) get a null tier and
`prediction_status = "invalid"`; they are never classified Safe.

### Triage policy (`policy.assign_triage`), separate from the tier

1. invalid data → Human review - insufficient data
2. Critical tier → Maneuver planning review
3. interval upper bound ≥ −4, or calibrated classifier ≥ its tuned threshold while the regressor disagrees → Human review
4. Elevated tier → Monitor
5. Safe tier, interval upper bound < −6, acceptable data quality → Auto-clear candidate
6. everything else → Human review

A missing interval can never qualify a row for auto-clear. Official thresholds are
fixed; only the review policy is tunable.

### Features

Permitted telemetry plus object type, then physics-inspired features (log miss distance,
combined radial sigma assuming independent errors, radial separation / combined sigma as
a proxy rather than a Mahalanobis distance, sigma ratio, RTN geometry fractions, log
speed, missingness indicators), then causal history features (messages seen so far,
elapsed time, deltas, running extrema, max-risk trend). Within an event a row only sees
messages with strictly larger `time_to_tca`; tied timestamps never see each other. A
standalone CDM uses the documented no-history representation (`n_cdms_so_far = 1`,
deltas and trend missing, `has_history = 0`, reason code `NO_HISTORY`).

### Validation design

Events are split 60 / 15 / 10 / 15 into development, policy calibration, conformal
calibration, and final evaluation, stratified by the maximum observed label tier (with
Pc-driven Critical events as their own stratum). Hyperparameters are chosen by 5-fold
grouped CV inside development. Candidates are compared on the policy-calibration
partition; the evaluation partition is touched once, after every decision is frozen, and
the evaluated bundle is the one shipped in `artifacts/model`. The delivered test file,
which carries labels, is reported as a separate external check.

### Uncertainty

Intervals are conformalized quantile regression: 0.05 and 0.95 quantile LightGBM heads
(crossing-corrected around the point prediction) shifted by a finite-sample corrected
quantile of per-event scores, where the event score is the maximum over its CDMs. This is
conservative by construction; measured coverage and widths are in the report. A
constant-width residual split-conformal calibrator is fitted too and compared in
`interval_method_comparison_primary`. Intervals describe uncertainty around the dataset
risk label, not every physical uncertainty of a real encounter, and no universal or
subgroup guarantee is claimed. If too few calibration events exist the calibrator reports
`insufficient_calibration` and intervals are null.

`p_pc_critical` is a separate sigmoid-calibrated classifier estimate that the reported Pc
crosses 1e-4; it is not another estimate of the collision probability.

### Explanations

Native LightGBM contributions in `log10(Pc)` units (+1 is a factor of ten in Pc). Each
record names the component explained (`lgbm_point_regressor`), the baseline, the raw
output, the top positive and negative contributions with feature values and labels, the
remaining total, and a note that clipping, tier rules, and triage are applied separately.
Text is a deterministic template that states associations, not causes.

## Artifacts (`artifacts/`)

| File | Content |
|---|---|
| `predictions.parquet` | one record per input CDM (contract in `contracts.PredictionRecord`) |
| `events.parquet` | latest available CDM per event plus per-event counters |
| `explanations.parquet` | contributions keyed by `explanation_id` |
| `evaluation.json` | metrics for every baseline and candidate, selection trace, failure cases |
| `evaluation_labels.parquet` | ground truth for the evaluation partition, kept separate |
| `manifest.json` | versions, hashes, units, split hash, thresholds, counts, timing |
| `submission.csv` | predictions in original test-row order |
| `audit.json`, `splits.json` | data audit and the event partition |
| `model/` | shipped bundle: `model.txt`, quantile heads, `classifier.txt`, `bundle.json`, config snapshot |
| `candidates/` | every other evaluated candidate and both baselines |

`submission.csv` columns: `row_id, source_row_index, event_id, time_to_tca_days,
miss_distance_m, risk, pc_pred, risk_log10_lower, risk_log10_upper, risk_tier,
triage_action, prediction_status`. If the organiser publishes a format, adapt the column
selection in `pipeline.stage_predict`.

## Results

See [RESULTS.md](RESULTS.md) for the headline numbers from the shipped run, including
the baseline comparison, Pc-only versus combined Critical detection, interval coverage,
triage mix, and illustrative missed and uncertain cases.

## Known limitations

- Pc-only Critical recall is modest; most Critical detections at the latest CDM come
  from the deterministic distance rule. The report separates the two so the rule does not
  hide model weakness.
- The −30 floor makes log-space regression errors large on floor rows; the optional
  hurdle model (not yet built) is the planned mitigation.
- The classifier threshold is tuned on the same partition used to fit its sigmoid, so
  its policy-calibration F2 is optimistic; evaluation-partition PR-AUC and Brier are the
  honest numbers.
- Monotone constraints could only be tested with the L2 objective (LightGBM does not
  support them with L1); the comparison is a matched L2 pair.
- Final-event risk forecasting is a stretch feature and is not implemented.

## Project structure

```
src/collision_risk/   contracts, config, data, features, train, calibration, policy,
                      evaluate, explain, service, pipeline, cli
configs/default.yaml  thresholds, features, split, model, conformal, triage settings
tests/                pytest suites (policy, calibration, explain, evaluate, data/features, pipeline/service)
artifacts/            outputs of the documented commands
app/                  dashboard (Part 2)
```
