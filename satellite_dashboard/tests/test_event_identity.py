"""Event identity, event-based counts, outcome confinement and review persistence."""
from conftest import needs_real
from satellite_dashboard.contracts import PredictionRecord
from satellite_dashboard.services.review_store import ReviewStore, review_status_column

OUTCOME_COLUMNS = {"final_tier", "final_risk_log10_pc", "final_miss_distance_m", "observed_final_tier", "is_trivial", "mission_id"}


@needs_real
def test_train_and_test_events_with_the_same_id_stay_distinct(heldout, test_view):
    shared = set(heldout.events["event_id"]) & set(test_view.events["event_id"])
    assert shared, "expected overlapping ids (ids restart in each file)"
    eid = sorted(shared)[0]
    a, b = heldout.event(eid), test_view.event(eid)
    assert a["dataset_id"] != b["dataset_id"]
    assert a["prediction_id"] != b["prediction_id"]
    assert set(a["source_row_ids"]).isdisjoint(b["source_row_ids"])
    assert test_view.outcome(eid) is None and heldout.outcome(eid) is not None


@needs_real
def test_overview_counts_are_event_based(heldout, test_view):
    for v in (heldout, test_view):
        assert v.events["event_id"].is_unique
        assert len(v.events) == v.info.n_events
        assert len(v.snapshots) > len(v.events)


@needs_real
def test_final_labels_stay_out_of_the_prediction_path(heldout):
    for frame in (heldout.events, heldout.snapshots, heldout.inputs):
        assert not OUTCOME_COLUMNS & set(frame.columns)
    assert set(PredictionRecord.model_fields).isdisjoint(OUTCOME_COLUMNS)


def test_missing_predictions_never_appear_as_safe(synthetic):
    bad = synthetic.events[synthetic.events["prediction_status"] != "Successful"]
    assert len(bad) == 2
    assert bad["predicted_final_tier"].isna().all() and bad["p_critical"].isna().all()
    assert set(bad["review_priority"]) == {"Data review"}


def test_reviews_survive_restart_and_stay_tied_to_their_prediction(tmp_path, synthetic):
    path = tmp_path / "r.sqlite"
    ev = synthetic.events.iloc[0]
    before = synthetic.events["predicted_final_tier"].value_counts().to_dict()
    ReviewStore(path).add(dataset_id=ev["dataset_id"], event_id=int(ev["event_id"]), prediction_id=ev["prediction_id"],
                          model_version=ev["model_version"], status="Acknowledged", note="checked")
    store = ReviewStore(path)  # "restart"
    hist = store.history(ev["dataset_id"], int(ev["event_id"]))
    assert len(hist) == 1 and hist.iloc[0]["prediction_id"] == ev["prediction_id"]
    status = review_status_column(synthetic.events, store.latest_status(ev["dataset_id"]))
    assert status.iloc[0] == "Acknowledged" and (status.iloc[1:] == "Not reviewed").all()
    # a review of an older prediction is not carried over to a new one
    newer = synthetic.events.copy(); newer.loc[newer.index[0], "prediction_id"] = "new-prediction"
    assert review_status_column(newer, store.latest_status(ev["dataset_id"])).iloc[0] == "Reviewed earlier prediction"
    # acknowledging does not change the prediction or the Critical count
    assert synthetic.events["predicted_final_tier"].value_counts().to_dict() == before
