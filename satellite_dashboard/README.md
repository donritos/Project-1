# Conjunction forecast review dashboard

A local Streamlit app that helps an analyst:

**find events forecast to finish Critical → inspect the evidence available at forecast time → understand the
uncertainty → record a review decision.**

It sits on top of the trained `collision-risk` model in the parent folder (Autonomous SG 2026, Airbus Challenge #1).
Forecasts use only CDMs with `time_to_tca ≥ 2 days` and predict each event's **final tier**, which is the tier of its
last CDM before closest approach. This is decision support. Review priorities describe the next analyst action;
nothing here calculates or authorises a manoeuvre.

## Run it

You need [uv](https://docs.astral.sh/uv/) (`pip install uv`). From this folder:

```bash
uv sync                      # installs the app, Python 3.11 and the model package from ..
uv run streamlit run app.py
```

Without uv, use Python 3.11 and run `pip install -e .. -e .`, then `streamlit run app.py`.

The app opens at http://localhost:8501 and works offline. The predictions, explanations and model bundle it needs
are already in `artifacts/`.

## Screens

| Screen | Answers |
|---|---|
| **Overview** | Which events are forecast to finish Critical? Shows KPI cards (event-based counts), a review queue with one row per event, how forecasts move between the latest observed tier and the predicted final tier, and filters in the sidebar. |
| **Event detail** | What was known when the forecast was made, and how did the forecast change as eligible CDMs arrived? Available evidence, the forecast and the historical outcome are kept in three separate panels. Tabs cover forecast history (each point uses only CDMs received up to then), measurements, model contributions, encounter geometry, review notes and the report export. |
| **Model evidence** | How well does the model do on unseen events? Held-out, event-level final-tier results against the carry-forward baseline: confusion matrices, Critical recall / precision / F2, macro-F1, calibration of P(Critical), escalations, and genuine failure cases. |
| **Upload & predict** | CSV upload of one or more event histories. Rows inside the 2-day cutoff are identified and excluded, then the rest goes through the same model pipeline. Gives a row-level validation report, event predictions and CSV downloads. A single-CDM form is also available, since the model supports single-message inference with no history. |

Datasets (sidebar):
- **Held-out training events**: the model's 20% event split, never used for training, tuning or calibration. Observed outcomes appear only where they are labelled *evaluation only*.
- **Unlabelled test events**: no outcomes exist, so no test-set performance is shown anywhere.
- **Synthetic demo data**: hand-made fixtures, shown with a permanent banner. They cover Critical, Elevated, Safe, invalid-input, insufficient-data and missing-capability cases.

## What the numbers mean

- **P(final = Critical)** is the model's calibrated probability that the event's *final tier* is Critical. It is
  **not** a collision probability.
- **Latest observed miss distance** is measured at the latest eligible CDM. It is not a forecast of the final distance.
- **Risk = log10(Pc).** −30 is the dataset placeholder for a probability too small to compute (Pc ≈ 0). It is shown
  as "≈ 0 (placeholder −30)", not as 10⁻³⁰.
- **Uncertainty interval unavailable.** This model does not produce intervals, so the app shows nothing in their place.
  Its 90% prediction sets always contain exactly one class in this run, so the manifest marks them as uninformative
  and they are hidden.
- **Contributions** are LightGBM contributions to the Critical-class score, in log-odds. They describe associations
  the model learned, not causes.
- **Review priority** comes from the model's review policy: *Priority review* (forecast Critical), *Data review*
  (no usable input), *Escalation watch*, *Analyst review*, *Monitor*, *Low-priority monitor* (predicted Safe, but below
  the auto-clear confidence bar), and *Lower review priority*. The last one is
  shown instead of the model's "auto-clear" label, because it is not a clearance decision.

## How it is put together

```
app.py                          navigation and sidebar
src/satellite_dashboard/
  contracts.py                  PredictionRecord, Capabilities, Manifest (Pydantic): the only fields views use
  services/model_adapter.py     the ONLY module that knows the model's column names; also runs uploads
  services/artifact_loader.py   reads artifacts and fails loudly (never falls back to fixtures)
  services/input_validation.py  upload checks: columns, types, units, ids, size, 2-day exclusions
  services/review_store.py      append-only SQLite review log keyed to event + prediction id
  services/metrics.py           event-level final-tier metrics
  services/reports.py           event report (Markdown / JSON)
  views/ components/            screens, charts (Plotly), tables, banners
artifacts/
  manifest.json                 model version, eligibility rule, capabilities, split, dataset counts
  model/                        model bundle used for uploads
  predictions/<dataset>/        events, snapshots (per-prefix forecasts), inputs, explanations, outcomes (held-out only)
  demo/                         synthetic fixtures (+ their own manifest, flagged synthetic)
data/sample_inputs/             sample_upload.csv (real held-out histories, including CDMs inside the cutoff)
data/local/                     reviews.sqlite (git-ignored)
scripts/
  build_artifacts.py            re-export predictions from the trained model (run after retraining)
  validate_bundle.py            pre-demo check of the whole bundle
  make_fixtures.py              regenerate synthetic fixtures
```

**After retraining the model:** run `collision-risk run-all` in the parent folder, then
`uv run python scripts/build_artifacts.py`, then `uv run python scripts/validate_bundle.py`. The build script reruns
the model's own forecasting pipeline over the held-out and test events and **fails if its output differs from the
model's exported predictions**. If the official labels workbook is in `../data/raw/`, it also cross-checks the
labels against it (and parses the `yes`/`no` column correctly).

To switch to a different model, replace `services/model_adapter.py` and the artifact bundle. The views stay the same.

## Tests

```bash
uv run pytest
```

These cover the acceptance checks:
- Ineligible CDMs never reach the model, and an input exactly 2 days out is eligible.
- Historical forecasts recomputed from history prefixes match the stored ones, so no forecast sees later messages.
- Train and test events with the same id stay distinct.
- Counts are event-based.
- Outcomes stay out of the prediction path.
- Missing predictions never appear as Safe or zero.
- Units and the −30 placeholder display correctly.
- Missing capabilities degrade gracefully.
- Reviews survive a restart and stay tied to their prediction.
- Uploads reproduce the precomputed predictions.
- Every page renders for every dataset (Streamlit AppTest).
- Evidence metrics match the model's own `evaluation.json`.
