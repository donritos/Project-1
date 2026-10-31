"""The 2-day rule: ineligible CDMs never reach feature generation, and historical forecasts never see later CDMs."""
import io

import numpy as np
import pandas as pd

from conftest import needs_model, needs_real
from satellite_dashboard.config import MIN_TIME_TO_TCA_DAYS
from satellite_dashboard.services.input_validation import validate_upload

MODEL_COLS = {"time_to_tca_days": "time_to_tca", "miss_distance_m": "miss_distance", "relative_speed_mps": "relative_speed",
              "rel_pos_r_m": "relative_position_r", "rel_pos_t_m": "relative_position_t", "rel_pos_n_m": "relative_position_n",
              "chaser_sigma_r_m": "c_sigma_r", "target_sigma_r_m": "t_sigma_r", "max_risk_estimate_log10": "max_risk_estimate",
              "risk_log10": "risk", "object_type": "c_object_type", "f10": "F10", "ap": "AP"}


def _csv(df: pd.DataFrame) -> bytes:
    b = io.StringIO(); df.to_csv(b, index=False); return b.getvalue().encode()


def _row(eid, t, **kw):
    r = {"event_id": eid, "time_to_tca": t, "miss_distance": 1500.0, "relative_position_r": 100.0, "relative_position_t": -1400.0,
         "relative_position_n": 520.0, "relative_speed": 9000.0, "c_object_type": "DEBRIS", "c_sigma_r": 120.0, "t_sigma_r": 12.0,
         "max_risk_estimate": -4.5, "F10": 90.0, "AP": 7.0, "risk": -5.5}
    r.update(kw)
    return r


def test_exactly_two_days_is_eligible_and_just_under_is_excluded():
    res = validate_upload(_csv(pd.DataFrame([_row(1, 2.0), _row(1, 1.9999), _row(1, 3.0)])))
    assert res.ok
    assert list(res.row_report["row_status"]) == ["Eligible", "Excluded: inside 2-day cutoff", "Eligible"]


def test_schema_errors_reject_rows_and_missing_columns_reject_file():
    res = validate_upload(_csv(pd.DataFrame([_row("x", 3.0), _row(1, "soon")])))
    assert (res.row_report["row_status"] == "Rejected").all()
    res = validate_upload(b"event_id,time_to_tca\n1,3\n")
    assert not res.ok and "Missing required columns" in res.errors[0]
    assert not validate_upload(_csv(pd.DataFrame([_row(1, 3.0, risk=2.0)]))).ok  # risk must be log10 Pc


@needs_real
def test_artifact_forecast_inputs_are_all_eligible(heldout, test_view):
    for v in (heldout, test_view):
        assert (v.snapshots["time_to_tca_days"] >= MIN_TIME_TO_TCA_DAYS).all()
        assert (v.inputs["time_to_tca_days"] >= MIN_TIME_TO_TCA_DAYS).all()
        ok = v.events[v.events["prediction_status"] == "Successful"]
        assert (ok["latest_input_time_to_tca_days"] >= MIN_TIME_TO_TCA_DAYS).all()


def _history(view, eid) -> pd.DataFrame:
    return view.event_inputs(eid).rename(columns=MODEL_COLS).drop(columns=["dataset_id", "row_id"])


@needs_model
def test_ineligible_rows_never_reach_the_model(adapter, test_view):
    eid = int(test_view.events.iloc[0]["event_id"])
    hist = _history(test_view, eid)
    noisy = pd.concat([hist, pd.DataFrame([_row(eid, 1.2, miss_distance=10.0, risk=-1.5), _row(eid, 0.3, miss_distance=5.0, risk=-1.0)])])
    a = validate_upload(_csv(hist)); b = validate_upload(_csv(noisy))
    va, _ = adapter.predict_upload(a.rows, a.dataset_id, "a")
    vb, _ = adapter.predict_upload(b.rows, b.dataset_id, "b")
    pa, pb = va.events.iloc[0], vb.events.iloc[0]
    assert pa["predicted_final_tier"] == pb["predicted_final_tier"]
    assert np.isclose(pa["p_critical"], pb["p_critical"], atol=1e-12)
    assert pb["n_excluded_cdms"] == 2 and pa["n_eligible_cdms"] == pb["n_eligible_cdms"]


@needs_model
def test_upload_reproduces_precomputed_prediction(adapter, test_view):
    for eid in test_view.events["event_id"].iloc[:5].astype(int):
        res = validate_upload(_csv(_history(test_view, eid)))
        v, _ = adapter.predict_upload(res.rows, res.dataset_id, "x")
        ref = test_view.event(eid)
        assert v.events.iloc[0]["predicted_final_tier"] == ref["predicted_final_tier"]
        assert np.isclose(v.events.iloc[0]["p_critical"], ref["p_critical"], atol=1e-9)


@needs_model
def test_historical_forecasts_cannot_see_later_messages(adapter, heldout):
    """Recompute the forecast from a history PREFIX and compare with the stored snapshot forecast at that point."""
    multi = heldout.events[heldout.events["n_eligible_cdms"] >= 5]["event_id"].astype(int).iloc[:3]
    for eid in multi:
        snaps = heldout.event_snapshots(eid)
        hist = _history(heldout, eid)
        for k in (1, len(snaps) // 2):
            cut = snaps.iloc[k - 1]["time_to_tca_days"]
            prefix = hist[hist["time_to_tca"] >= cut]
            res = validate_upload(_csv(prefix))
            v, _ = adapter.predict_upload(res.rows, res.dataset_id, "prefix")
            assert np.isclose(v.events.iloc[0]["p_critical"], snaps.iloc[k - 1]["p_critical"], atol=1e-3), (eid, k)


@needs_model
def test_event_without_eligible_rows_is_insufficient_not_safe(adapter):
    res = validate_upload(_csv(pd.DataFrame([_row(7, 1.5), _row(7, 0.5), _row(8, 3.0)])))
    v, _ = adapter.predict_upload(res.rows, res.dataset_id, "x")
    e7 = v.event(7)
    assert e7["prediction_status"] == "Insufficient eligible data"
    assert e7["predicted_final_tier"] is None and e7["n_excluded_cdms"] == 2 and e7["review_priority"] == "Data review"
