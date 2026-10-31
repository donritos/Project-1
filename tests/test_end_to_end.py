"""End-to-end smoke run on synthetic data: all artifacts exist with the expected schemas, no metric
touches the test file, probabilities sum to 1, save/reload preserves forecasts."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from collision_risk import pipeline
from collision_risk.contracts import EVENT_FORECAST_COLUMNS, EventForecastRecord
from collision_risk.service import forecast_frame, load_bundle, predict_batch


@pytest.fixture(scope="module")
def run(synthetic_cfg):
    pipeline.stage_audit(synthetic_cfg)
    ev = pipeline.stage_train(synthetic_cfg, quick=True)
    man = pipeline.stage_predict(synthetic_cfg)
    return {"cfg": synthetic_cfg, "evaluation": ev, "manifest": man}


def test_artifacts_exist_with_expected_schema(run):
    from pathlib import Path
    art = Path(run["cfg"]["paths"]["artifacts_dir"])
    for f in ["audit.json", "splits.json", "evaluation.json", "manifest.json", "event_forecasts.parquet", "snapshot_forecasts.parquet",
              "explanations.parquet", "submission.csv", "holdout_forecasts.parquet", "evaluation_labels.parquet", "model/bundle.json", "model/config_snapshot.json"]:
        assert (art / f).exists(), f
    ev = pd.read_parquet(art / "event_forecasts.parquet")
    for c in EVENT_FORECAST_COLUMNS:
        assert c in ev.columns, c
    assert ev["event_id"].is_unique
    sub = pd.read_csv(art / "submission.csv")
    assert list(sub.columns) == ["event_id", "predicted_final_tier", "p_safe", "p_elevated", "p_critical", "triage_action"]
    assert sub["event_id"].is_monotonic_increasing and len(sub) == len(ev)
    P = ev[["p_safe", "p_elevated", "p_critical"]].to_numpy(float)
    ok = np.isfinite(P).all(axis=1)
    assert np.allclose(P[ok].sum(axis=1), 1.0, atol=1e-6)
    for i in range(min(50, len(ev))):
        d = {k: (None if isinstance(v, float) and np.isnan(v) else (v.item() if isinstance(v, np.generic) else v)) for k, v in ev.iloc[i].to_dict().items() if k in EVENT_FORECAST_COLUMNS}
        EventForecastRecord.model_validate(d)


def test_evaluation_json_structure_and_no_test_metrics(run):
    e = run["evaluation"]
    assert "models" in e and "carry_forward_risk" in e["models"] and e["primary_model"] in e["models"]
    for name, res in e["models"].items():
        for subset in ("all_usable", "trivial", "real_forecast"):
            assert subset in res["subsets"], (name, subset)
            cls = res["subsets"][subset]["classification"]
            assert "confusion" in cls and "critical_recall" in cls
        assert "bootstrap" in res["subsets"]["real_forecast"]
    assert all(k in e["decision_thresholds"] for k in ("t_crit", "t_elev", "t_watch", "t_clear"))
    assert "ablations" in e and "mission_heldout_cv" in e
    s = json.dumps(e)
    assert "kelvins_test" not in s and "test_data" not in json.dumps(e["models"])
    assert run["manifest"]["n_events"] > 0 and "predicted_class_distribution" in run["manifest"]


def test_save_reload_preserves_forecasts_and_horizon_is_enforced(run, synthetic_raw):
    from pathlib import Path
    cfg = run["cfg"]
    b = load_bundle(Path(cfg["paths"]["artifacts_dir"]) / "model")
    rows = synthetic_raw[synthetic_raw["event_id"].isin([1, 2, 3])]
    out1 = predict_batch(b, rows, dataset_id="up")
    out2 = predict_batch(load_bundle(Path(cfg["paths"]["artifacts_dir"]) / "model"), rows, dataset_id="up")
    np.testing.assert_allclose(out1.events[["p_safe", "p_elevated", "p_critical"]].to_numpy(float), out2.events[["p_safe", "p_elevated", "p_critical"]].to_numpy(float))
    assert out1.events["predicted_final_tier"].tolist() == out2.events["predicted_final_tier"].tolist()
    assert (out1.snapshots["time_to_tca_days"] >= 2.0).all()
    assert out1.info["n_rows_below_horizon_dropped"] == int((rows["time_to_tca"] < 2.0).sum())
    assert len(out1.events) == rows[rows["time_to_tca"] >= 2.0]["event_id"].nunique()
    # a de-escalation claim can never be auto-cleared; no final-Critical check is possible here, but current Critical rows must not auto-clear
    snaps = out1.snapshots
    assert not ((snaps["current_tier"] == "Critical") & (snaps["triage_action"] == "Auto-clear candidate")).any()


def test_invalid_rows_get_insufficient_data(run, synthetic_raw):
    from pathlib import Path
    b = load_bundle(Path(run["cfg"]["paths"]["artifacts_dir"]) / "model")
    rows = synthetic_raw[synthetic_raw["event_id"] == 10].copy()  # event 0 already carries injected invalid rows
    rows.loc[rows["time_to_tca"].idxmax(), "miss_distance"] = np.nan
    out = predict_batch(b, rows, dataset_id="bad")
    inv = out.snapshots[out.snapshots["prediction_status"] == "invalid"]
    assert len(inv) == 1 and (inv["triage_action"] == "Human review - insufficient data").all()
    assert inv["predicted_final_tier"].isna().all() and inv["p_critical"].isna().all()
