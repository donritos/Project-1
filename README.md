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

`predict` accepts any CSV of CDMs with the permitted columns, e.g.
`uv run collision-risk predict --input my_cdms.csv --dataset-id ops_2026_11 --out-dir artifacts/ops`.
CDMs under two days are dropped before any feature is built.

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
when it is present and cross-checks every tier against the tier recomputed from the event's last
CDM, failing loudly on any difference. When the workbook is absent it derives the identical table
from `train_data.csv` and records `labels_source = derived_from_train_data` in the audit.
Recomputed tiers match the workbook for 100 percent of events.

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
The decision rule predicts Critical when P(Critical) ≥ `t_crit`, else Elevated when
P(Elevated) + P(Critical) ≥ `t_elev`, else Safe; both thresholds are tuned on OOF and frozen in
the bundle. Split-conformal prediction sets at 90 percent are calibrated on OOF and reported
with their empirical coverage.

### Triage policy (`policy.assign_triage`), in precedence order

1. Invalid data or failed quality → Human review - insufficient data
2. Predicted Critical → Maneuver planning review
3. Not predicted Critical but P(Critical) ≥ `t_watch` → **Escalation watch** (watch list capped at 5 percent of events, tuned on OOF)
4. Human review: model and carry-forward disagree by two tiers, the prediction set holds two non-adjacent classes, or the current tier is Critical and the model forecasts lower (a de-escalation claim, never auto-cleared)
5. Predicted Elevated → Monitor
6. Predicted Safe with P(Safe) ≥ `t_clear`, current tier Safe and acceptable quality → Auto-clear candidate (`t_clear` is the largest auto-clear share with zero OOF final-Critical events cleared)
7. Everything else → Human review

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
| `event_forecasts.parquet` | one row per test event (latest snapshot): current tier, probabilities, predicted final tier, prediction set, triage action, reason codes, explanation id, versions |
| `snapshot_forecasts.parquet` | the same fields for every snapshot CDM (dashboard timeline) |
| `explanations.parquet` | Critical-class contributions keyed by `explanation_id` |
| `submission.csv` | one row per test event sorted by `event_id`: `event_id, predicted_final_tier, p_safe, p_elevated, p_critical, triage_action` |
| `evaluation.json` | holdout metrics for every candidate and subset, selection trace, thresholds, ablations, mission-held-out CV, illustrative cases |
| `holdout_forecasts.parquet`, `evaluation_labels.parquet` | holdout forecasts and the ground truth, kept separate |
| `manifest.json` | versions, hashes, units, split hash, thresholds, predicted class mix on test, timing |
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
- The mission-held-out check is reported, not used for selection; performance on an unseen
  satellite may be lower than the random-event holdout suggests.
- Zero final-Critical events auto-cleared is a tuning target on development data; the holdout
  count is reported and must be read alongside the bootstrap intervals.
- Events with no CDM at 2 or more days cannot be forecast and are excluded.
- Final risk regression and the dashboard are out of scope for this repository state.

## Project structure

```
src/collision_risk/   contracts, config, data, features, train, calibration, policy,
                      evaluate, explain, service, pipeline, cli
configs/default.yaml  horizon, allowlist, thresholds, features, split, model, calibration, triage
docs/forecast_schema.md  shared frame schema consumed by policy, evaluation and the dashboard
tests/                pytest suites
scripts/              make_results.py, check_baseline_sanity.py
artifacts/            outputs of the documented commands
app/                  dashboard (not built here)
```
