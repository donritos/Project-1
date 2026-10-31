# Collision-risk final-tier forecasting (Autonomous SG 2026, Airbus Challenge #1)

Prediction engine for a satellite conjunction triage product. For every conjunction **event**
it forecasts the event's **final risk tier** (Safe / Elevated / Critical, the tier of the last
conjunction data message before closest approach) using only CDMs at **2 or more days before
TCA**, with calibrated class probabilities, a recommended triage action, reason codes,
data-quality flags, feature contributions, and provenance.

The user is a satellite operations analyst. The system supports their decision; it does not
plan or execute manoeuvres.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) (`pip install uv`). uv fetches Python 3.11 and installs
the locked dependencies.

```bash
uv sync --group dev
```

Place `train_data.csv`, `test_data.csv` and, when available, `challenge1_event_labels.xlsx` in
`data/raw/`, then:

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

Or everything at once (`--quick` is a smoke run that finishes in under two minutes):

```bash
uv run collision-risk run-all
```

Tests (horizon leakage, labels, splits, policy precedence, decision rule, calibration,
explanations, evaluation, end-to-end smoke):

```bash
uv run pytest
```

Regenerate the results page from the artifacts:

```bash
uv run python scripts/make_results.py
```

### Dashboard

The analyst dashboard lives in [`satellite_dashboard/`](satellite_dashboard/README.md) and runs on
the shipped bundle in `artifacts/model` (it carries its own copy of the bundle plus the held-out and
test forecasts, so it works offline). From that folder:

```bash
cd satellite_dashboard
uv sync                      # installs the app and the model package from ..
uv run streamlit run app.py  # http://localhost:8501
uv run pytest                # dashboard acceptance tests
```

After retraining (`collision-risk run-all`), refresh the dashboard's artifacts with
`uv run python scripts/build_artifacts.py` followed by `uv run python scripts/validate_bundle.py`;
the build fails if the dashboard's forecasts differ from the model's exported predictions.

`predict` accepts any CSV of CDMs with the permitted columns, e.g.
`uv run collision-risk predict --input my_cdms.csv --dataset-id ops_2026_11 --out-dir artifacts/ops`.
The CSV and in-memory (`service.predict_batch`) paths share one normalisation routine: optional
columns may be omitted (they become missing-value flags), a missing `event_id` makes the row a
standalone encounter, and a non-finite or fractional `event_id` is rejected with a clear error
instead of being silently truncated. CDMs under two days are dropped before any feature is built.

## The task

The organisers' update makes the deliverable explicit: predict each event's **final** tier from
early CDMs. Tiers can be read straight from the current `risk`, so the challenge is the forecast,
not the rule. The official tier rules (applied most severe first, metres) are unchanged:

1. `risk >= -4` or `miss_distance < 1000` → Critical
2. otherwise `-6 <= risk < -4` or `1000 <= miss_distance <= 5000` → Elevated (band inclusive at both ends)
3. otherwise → Safe

`risk = -30` is a placeholder for a probability too small to compute (Pc ≈ 0).

**Labels.** One row per training event: final risk, final miss distance, final tier, and a
usable flag. The loader reads the official workbook (`train_event_labels` sheet, header on row 3)
when it is present and cross-checks every event's final tier, usable flag and count of CDMs at
2 or more days against the values recomputed from the event's CDMs, failing loudly on any
difference. When the workbook is absent it derives the same table from `train_data.csv` and
records `labels_source = derived_from_train_data` in the audit.

**Status of the shipped run: the workbook was not available** (`audit.json` records
`workbook_present: false`), so the shipped labels are the derived ones and **no comparison with
the official workbook has been made yet**. The derived definition (tier of the last CDM) is the
one the organisers describe; the comparison must be run once the file from the organisers'
Drive folder is placed in `data/raw/`. The yes/no text of the usable column is parsed explicitly
because pandas 3 reads it as a string dtype, where a naive boolean cast would mark every event
usable.

**Usable events.** Of 13,154 training events, 11,942 have at least one CDM at 2 or more days
(8,946 Safe, 2,162 Elevated, 834 Critical). The other 1,212 are excluded from training and
evaluation and the count is reported. In 3,050 usable events the final CDM is itself at 2 or more
days, so the answer is visible in the input (**trivial**). The other 8,892 are **real forecasts**.
Every metric is reported on `all_usable`, `trivial` and `real_forecast`; headline numbers use
`real_forecast`.

**Test data** has no final outcomes. Every test CDM is 2 to 7 days out. It is used only to
produce demo forecasts; no metric is computed on it anywhere in the code.

## What the pipeline does

| Stage | Module | Output |
|---|---|---|
| Audit | `data.py` | `artifacts/audit.json`: units, encodings, duplicates, label cross-check, usable/trivial counts, horizon check |
| Horizon and snapshots | `data.apply_horizon`, `features.build_event_snapshots` | one row per CDM at ≥ 2 days with per-CDM physics features, causal history features and event-so-far features |
| Split | `train.make_holdout_split` | 80/20 event split stratified on final tier × trivial, saved with its hash |
| Models | `train.py`, `pipeline.py` | grouped out-of-fold search, LightGBM multiclass and ordinal candidates, baselines |
| Calibration and decisions | `calibration.py` | per-class isotonic calibration, decision thresholds, conformal prediction sets |
| Triage | `policy.py` | official tier rules (single implementation) and the event-level review policy |
| Evaluation | `evaluate.py` | holdout metrics per subset, escalation metrics, triage workload, bootstrap, breakdowns |
| Explanations | `explain.py` | vectorised Critical-class contributions and grouped drivers |
| Serving | `service.py` | `load_bundle`, `predict_batch`, `get_event_forecast`, `get_event_history`, `get_explanation`, `get_evaluation_report` |

### Columns used

Recommended inputs: `event_id, time_to_tca, miss_distance, relative_speed, relative_position_r/t/n,
c_object_type, c_sigma_r, t_sigma_r, max_risk_estimate, F10, AP`. The allowlist is enforced at
load time and again at the predictor's entry point; the delivered files carry 89 other columns,
all dropped and listed in the audit.

Beyond the recommended list the model uses **one** extra column, under a config flag:

- **`risk` of CDMs at 2 or more days** (`features.use_past_risk`, default `true`). Past risk is
  known information at prediction time, and the official naive baseline is precisely "carry the
  latest known risk forward", so withholding it would handicap the model against its own
  baseline. Results are reported with the flag on and off (ablation `use_past_risk_off`).

`mission_id` is **never** a feature; a model could memorise satellites. It is read from the labels
file only for the mission-held-out robustness check. `event_id` is used for grouping and
traceability only. Event ids restart in each file, so every identifier is a `(dataset_id, event_id)`
pair and train and test are never joined on `event_id`.

### Features

Per snapshot: the permitted telemetry, log miss distance, combined radial sigma (independence
assumed), radial separation over combined sigma (a proxy, not a Mahalanobis distance), sigma
ratio, RTN geometry fractions, log speed, missingness indicators, and causal history features
(messages so far, deltas, running extrema, trend). Event-so-far features use only snapshot CDMs
at the same or an earlier time: count of CDMs at ≥ 2 days, span, first-to-current changes,
least-squares slopes against `time_to_tca`, volatility, up/down move counts, current-tier
proxies from `risk` and from `max_risk_estimate`, margins to the −4 and −6 thresholds and to the
1 km and 5 km distances, and a floor flag. A first CDM uses the documented no-history
representation (`has_history = 0`, reason code `NO_HISTORY`).

### Validation design

- 80/20 event split over usable events, stratified on final tier × trivial, fixed seed, hash in
  the manifest.
- Everything that needs tuning uses **5-fold grouped out-of-fold predictions inside the 80 percent
  development set**: hyperparameters, the snapshot-versus-latest training-row choice, isotonic
  calibration, decision thresholds, triage thresholds, conformal sets.
- The 20 percent holdout is evaluated **once**, after every choice is frozen. The evaluated
  bundle is the one shipped in `artifacts/model`.
- Training examples are every snapshot of every development event labelled with the event's
  final tier, weighted `1 / n_snapshots` so each event counts equally, times a capped
  inverse-class-frequency weight. Latest-snapshot-only training is compared and the better OOF
  Critical F2 on real forecasts is kept.

### Models and baselines

- `carry_forward_risk`: final tier = current tier of the latest 2-day-plus CDM (official naive baseline).
- `carry_forward_mre`: same using `max_risk_estimate`.
- `majority`: always Safe.
- `logistic`: multinomial logistic regression on a few standardised features.
- `lgbm_multiclass`: LightGBM 3-class model, random search with grouped CV, selection by OOF
  Critical F2 on real forecasts with macro-F1 as tie-break.
- `lgbm_ordinal`: two binary heads, P(final ≥ Elevated) and P(final = Critical), with
  P(Critical) ≤ P(≥ Elevated) enforced.

Probabilities are isotonic-calibrated per class on development OOF predictions and renormalised.
**`p_critical` is the calibrated probability that the event's final official tier is Critical.
It is not a collision probability (Pc)**; the same holds for `p_safe` and `p_elevated`.
The decision rule predicts Critical when P(Critical) ≥ `t_crit`, else Elevated when
P(Elevated) + P(Critical) ≥ `t_elev`, else Safe; both thresholds are tuned on OOF and frozen in
the bundle. Split-conformal prediction sets at 90 percent are calibrated on OOF and reported
with marginal and class-conditional coverage, set sizes, and the share of events whose
threshold decision falls outside the set. Calibration, decision thresholds, triage thresholds and
the conformal quantile are all fitted on the same development OOF predictions, so the sets carry
only an empirical, not a formal, coverage guarantee.

### Triage policy (`policy.assign_triage`), in precedence order

1. Invalid data or failed quality → Human review - insufficient data
2. Predicted Critical → Maneuver planning review
3. Not predicted Critical but P(Critical) ≥ `t_watch` → **Escalation watch**. `t_watch` is tuned so the list holds about 5 percent of development events; it is a threshold, not a hard cap, so the share on new data is reported rather than enforced
4. Human review: model and carry-forward disagree by two tiers, the prediction set holds two non-adjacent classes, or the current tier is Critical and the model forecasts lower (a de-escalation claim, never auto-cleared)
5. Predicted Elevated → Monitor
6. Predicted Safe with P(Safe) ≥ `t_clear`, current tier Safe and acceptable quality → Auto-clear candidate (`t_clear` is the smallest bar with zero OOF final-Critical events cleared, which puts it near 0.999)
7. Predicted Safe below that bar, current tier Safe and acceptable quality → **Low-priority monitor**. Without this rule roughly half of all events, most of them predicted Safe with P(Safe) above 0.99, fell through to Human review and swamped the review queue
8. Everything else (predicted Safe with a non-Safe current tier or blocking quality flags) → Human review

Review workload is counted as Maneuver planning review + Escalation watch + Human review.
Auto-clear and low-priority monitor shares are *potential* workload reductions measured on the
holdout, not demonstrated labour savings; no operator study has been run. The evaluation reports
a true-final-tier by triage-action table so that any final-Critical event parked in Monitor,
Low-priority monitor or Auto-clear is visible, not only the auto-clear count.

For each snapshot the carry-forward tier used by rule 4 is the tier known at that snapshot, so a
later CDM can never change an earlier snapshot's action or reason codes (covered by a regression
test).

Reason codes record every applicable condition, including trend codes (`RISK_RISING`,
`MISS_SHRINKING`, `SIGMA_GROWING`, `HIGH_VOLATILITY`) derived from the event-so-far slopes.

### Explanations

Native LightGBM contributions for the **Critical-class** output, in log-odds, computed
vectorised (a few seconds for the 24,484 test snapshots). Each record names the component, the
baseline, the raw output, the top positive and negative drivers with values, the remainder,
contributions summed by driver group (geometry, uncertainty, risk trend, risk level, space
weather, object type, history), and a templated sentence that describes associations, not causes.
The calibrated probability is reported alongside and is not part of the additive decomposition.

## Artifacts (`artifacts/`)

| File | Content |
|---|---|
| `event_forecasts.parquet` | one row per test event (latest snapshot): current tier, probabilities, predicted final tier, prediction set, triage action, reason codes, explanation id, RTN position, versions |
| `snapshot_forecasts.parquet` | the same fields for every snapshot CDM, including miss distance and RTN position against time to TCA (dashboard timeline) |
| `explanations.parquet` | Critical-class contributions keyed by `explanation_id` |
| `submission.csv` | one row per test event sorted by `event_id`: `event_id, predicted_final_tier, p_safe, p_elevated, p_critical, triage_action` |
| `evaluation.json` | holdout metrics for every candidate and subset, selection trace, thresholds, ablations, mission-held-out CV, illustrative cases |
| `holdout_forecasts.parquet`, `evaluation_labels.parquet` | holdout forecasts and the ground truth, kept separate |
| `manifest.json` | versions, hashes, units, split hash, thresholds, predicted class mix on test, timing; provenance fields come from the bundle's own saved configuration, and `submission.csv` is written with LF line endings so its hash is reproducible after checkout |
| `audit.json`, `splits.json` | data and label audit, the event partition |
| `model/` | shipped bundle; `candidates/` | the other evaluated candidates and baselines |

## Results

See [RESULTS.md](RESULTS.md), generated by `scripts/make_results.py` from the artifacts. It leads
with the real-forecast holdout comparison against `carry_forward_risk`, the confusion matrices,
escalation-watch results, auto-clear share with the number of final-Critical events cleared,
calibration, bootstrap intervals, lead-time and object-type breakdowns, ablations, the
mission-held-out check, and illustrative missed and caught cases. The test-set class mix is
reported from the manifest with the caveat that the Kelvins test set was deliberately enriched
with high-risk events.

## Known limitations

- Almost all final-Critical events are Critical because the final miss distance is under
  1,000 m; the model is mostly forecasting miss-distance evolution, and its advantage over
  carry-forward comes from escalations it anticipates and de-escalations it catches.
- The model trades precision for recall: on real forecasts it catches more final-Critical
  events than carry-forward but raises more false Critical alarms (lower Critical precision and
  macro-F1). In collision avoidance that trade is deliberate, and it is stated rather than hidden.
- The mission-held-out check is reported, not used for selection, and it is not fully
  independent: it reuses the hyperparameters, isotonic calibrator and decision thresholds fitted
  on all development events. Performance on an unseen satellite may be lower than it suggests.
- Zero final-Critical events auto-cleared is a tuning target on development data; the holdout
  count is reported and must be read alongside the bootstrap intervals.
- In the shipped run every conformal prediction set was a singleton, so the sets add nothing
  beyond the point decision; the class-conditional coverage table in RESULTS.md is the honest
  statement of what they provide.
- The official labels workbook has not yet been compared with the derived labels (see Labels).
- Events with no CDM at 2 or more days cannot be forecast and are excluded.
- The parquet artifacts (`event_forecasts`, `snapshot_forecasts`, `explanations`,
  `holdout_forecasts`, `evaluation_labels`) are gitignored. The artifact readers in `service.py`
  need them, so run `predict` (and `train` for the holdout files) with the data in place before
  demoing; the JSON artifacts and the model bundle are committed. The dashboard ships its own
  copies under `satellite_dashboard/artifacts/`.
- Final risk regression is out of scope for this repository state. The dashboard plots risk and
  miss distance against time to TCA; `snapshot_forecasts.parquet` carries the RTN coordinates.

## Project structure

```
src/collision_risk/   contracts, config, data, features, train, calibration, policy,
                      evaluate, explain, service, pipeline, cli
configs/default.yaml  horizon, allowlist, thresholds, features, split, model, calibration, triage
docs/forecast_schema.md  shared frame schema consumed by policy, evaluation and the dashboard
tests/                pytest suites
scripts/              make_results.py, check_baseline_sanity.py
artifacts/            outputs of the documented commands
satellite_dashboard/  Streamlit analyst dashboard (own README, tests and artifact bundle)
```
