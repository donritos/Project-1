"""End-to-end smoke run on synthetic data: all artifacts exist with the expected schemas, no metric
touches the test file, probabilities sum to 1, save/reload preserves forecasts."""
from __future__ import annotations

import json
from pathlib import Path

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


def _bundle(run):
    from pathlib import Path
    return load_bundle(Path(run["cfg"]["paths"]["artifacts_dir"]) / "model")


def test_later_cdm_never_changes_earlier_snapshot_forecasts_actions_or_reasons(run, synthetic_raw):
    """Regression test for the carry-forward leak: appending a later CDM must leave every earlier
    snapshot's probabilities, decision, triage action and reason codes untouched."""
    b = _bundle(run)
    cols = ["row_id", "p_safe", "p_elevated", "p_critical", "predicted_final_tier", "triage_action", "reason_codes", "carry_forward_tier"]
    for ev_id in (2, 5, 11):
        ev = synthetic_raw[(synthetic_raw["event_id"] == ev_id) & (synthetic_raw["time_to_tca"] >= 2.0)].sort_values("time_to_tca", ascending=False)
        if len(ev) < 3:
            continue
        early = ev.iloc[:-1].copy()
        later = ev.iloc[[-1]].copy()
        # make the later CDM dramatically different so any leak shows
        later["miss_distance"] = 50.0; later["relative_position_r"] = 10.0; later["relative_position_t"] = 48.0; later["relative_position_n"] = 5.0
        later["max_risk_estimate"] = -1.5
        if "risk" in later.columns:
            later["risk"] = -1.5
        before = predict_batch(b, early, dataset_id="e", with_explanations=False).snapshots
        after = predict_batch(b, pd.concat([early, later]), dataset_id="e", with_explanations=False).snapshots
        after_early = after[after["row_id"].isin(before["row_id"])].sort_values("row_id").reset_index(drop=True)
        before = before.sort_values("row_id").reset_index(drop=True)
        pd.testing.assert_frame_equal(before[cols], after_early[cols])
        # the previous latest snapshot keeps its own current tier as carry-forward, not the new latest tier
        assert (after_early["carry_forward_tier"] == after_early["current_tier"]).all()


def test_snapshot_frame_carries_rtn_geometry_for_the_dashboard(run, synthetic_raw):
    b = _bundle(run)
    rows = synthetic_raw[synthetic_raw["event_id"].isin([1, 2])]
    out = predict_batch(b, rows, dataset_id="rtn", with_explanations=False)
    for c in ("relative_position_r_m", "relative_position_t_m", "relative_position_n_m"):
        assert c in out.snapshots.columns and c in out.events.columns
        assert out.snapshots[c].notna().any()
    merged = out.snapshots.merge(rows.assign(row_id=lambda d: "rtn:" + d.reset_index(drop=True).index.astype(str)), on="row_id", how="left")
    assert np.allclose(merged["relative_position_r_m"], merged["relative_position_r"], equal_nan=True)


def test_explanations_carry_their_own_dataset_id_in_a_mixed_batch(run, synthetic_raw):
    from collision_risk.data import prepare_uploaded
    b = _bundle(run)
    rows = synthetic_raw[synthetic_raw["event_id"].isin([1, 2])]
    a, _ = prepare_uploaded(rows, "dsA", b.cfg)
    c, _ = prepare_uploaded(rows, "dsB", b.cfg)
    out = forecast_frame(b, pd.concat([a, c], ignore_index=True), with_explanations=True)
    ex = out.explanations
    assert ex is not None and set(ex["dataset_id"]) == {"dsA", "dsB"}
    assert ex["explanation_id"].is_unique
    joined = out.snapshots.dropna(subset=["explanation_id"]).merge(ex, on="explanation_id", suffixes=("", "_x"))
    assert (joined["dataset_id"] == joined["dataset_id_x"]).all()


def test_csv_and_api_paths_give_identical_forecasts_when_optional_columns_are_omitted(run, synthetic_raw, tmp_path):
    from collision_risk.data import load_raw
    b = _bundle(run)
    rows = synthetic_raw[synthetic_raw["event_id"].isin([3, 4])].drop(columns=["relative_speed", "F10"])
    p = tmp_path / "upload.csv"; rows.to_csv(p, index=False)
    df, _ = load_raw(p, "u", b.cfg)
    via_csv = forecast_frame(b, df, with_explanations=False).events
    via_api = predict_batch(b, rows, dataset_id="u", with_explanations=False).events
    cols = ["event_id", "p_safe", "p_elevated", "p_critical", "predicted_final_tier", "triage_action"]
    pd.testing.assert_frame_equal(via_csv[cols].reset_index(drop=True), via_api[cols].reset_index(drop=True))


def test_submission_csv_uses_lf_line_endings_and_manifest_provenance_comes_from_bundle(run):
    from pathlib import Path
    art = Path(run["cfg"]["paths"]["artifacts_dir"])
    raw = (art / "submission.csv").read_bytes()
    assert b"\r\n" not in raw and raw.count(b"\n") == len(pd.read_csv(art / "submission.csv")) + 1
    man = run["manifest"]
    b = _bundle(run)
    assert man["policy_version"] == b.cfg["policy_version"] and man["bundle_path"] == str(b.path)
    assert "predicted_class_distribution" in man and sum(man["predicted_class_distribution"].values()) == man["n_events"]


def test_low_priority_monitor_never_holds_current_critical_and_tier_by_action_table_balances(run):
    e = run["evaluation"]
    for name, res in e["models"].items():
        t = res["subsets"]["all_usable"]["triage"]
        table = t["action_by_final_tier"]
        assert sum(sum(v.values()) for v in table.values()) == t["n_events"]
        assert t["n_final_critical_not_urgent"] == sum(table.get(a, {}).get("Critical", 0) for a in ("Monitor", "Low-priority monitor", "Auto-clear candidate"))
    snaps = pd.read_parquet(Path(run["cfg"]["paths"]["artifacts_dir"]) / "snapshot_forecasts.parquet")
    assert not ((snaps["current_tier"] == "Critical") & (snaps["triage_action"] == "Low-priority monitor")).any()
