"""Required checks on the integrated pipeline: event-disjoint splits, save/reload identity,
one ordered record per input row, missing data never auto-clears, explanation provenance."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from collision_risk import features as FE
from collision_risk import train as T
from collision_risk.calibration import CQRCalibrator, SplitConformalCalibrator, fix_quantile_crossing
from collision_risk.contracts import PredictionRecord
from collision_risk.service import Bundle, load_bundle, predict_batch, predict_frame, save_bundle


@pytest.fixture(scope="module")
def tiny_bundle(synthetic_allowlisted, cfg, tmp_path_factory):
    """Train a tiny model on the synthetic data (dev events), calibrate on other events, save and return paths."""
    df, _ = synthetic_allowlisted
    splits = T.make_event_splits(df, cfg)
    F = FE.build_features(df, cfg)
    F["split"] = T.attach_split(F, splits).to_numpy()
    dev = F[F["split"] == "dev"]
    cal = F[F["split"] == "conformal_cal"]
    y = dev["risk"].to_numpy(float)
    X = FE.feature_matrix(dev, "history")
    params = {"objective": "l1", "learning_rate": 0.1, "num_leaves": 7, "min_child_samples": 10, "feature_fraction": 1.0, "lambda_l2": 0.0}
    booster, _ = T.fit_lgb(X, y, params, cfg, num_rounds=40)
    heads = T.fit_quantile_heads(X, y, params, cfg, num_rounds=40)
    meta = {"name": "tiny", "kind": "lgbm", "model_version": "tiny-test", "feature_version": cfg["feature_version"], "policy_version": cfg["policy_version"],
            "feature_set": "history", "feature_columns": FE.feature_columns("history"), "object_type_categories": cfg["features"]["object_type_categories"],
            "split_hash": splits["split_hash"], "num_iteration": 40, "feature_ranges": FE.compute_feature_ranges(X)}
    b = Bundle(path=None, meta=meta, cfg=cfg, booster=booster, classifier=None, conformal=None, sigmoid=None, quantile_heads=heads, interval_method="cqr")
    raw_cal = df[df["row_id"].isin(cal["row_id"])]
    out = predict_frame(b, raw_cal, with_explanations=False).predictions.merge(df[["row_id", "risk"]], on="row_id")
    Xc = FE.feature_matrix(cal, "history")
    q_lo, q_hi = fix_quantile_crossing(out["risk_log10_pred"].to_numpy(), T.postprocess(heads[0.05].predict(Xc), cfg), T.postprocess(heads[0.95].predict(Xc), cfg))
    cqr = CQRCalibrator(coverage=0.9, min_events=5).fit(out["risk"].to_numpy(), q_lo, q_hi, out["event_id"].to_numpy())
    conf = SplitConformalCalibrator(coverage=0.9, min_events=5).fit(out["risk"].to_numpy(), out["risk_log10_pred"].to_numpy(), out["event_id"].to_numpy())
    b.cqr, b.conformal = cqr, conf
    path = tmp_path_factory.mktemp("bundle") / "model"
    save_bundle(path, meta=meta, cfg=cfg, booster=booster, classifier=None, conformal=conf, sigmoid=None, quantile_heads=heads, cqr=cqr, interval_method="cqr")
    return {"path": path, "bundle": b, "df": df, "splits": splits}


def test_splits_are_event_disjoint_and_complete(tiny_bundle):
    splits = tiny_bundle["splits"]
    sets = {k: set(v) for k, v in splits["events"].items()}
    names = list(sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert not (sets[names[i]] & sets[names[j]]), f"{names[i]} overlaps {names[j]}"
    assert set().union(*sets.values()) == set(tiny_bundle["df"]["event_id"].unique())
    # mapping back to rows keeps every row
    s = T.attach_split(tiny_bundle["df"], splits)
    assert s.notna().all()


def test_save_reload_preserves_predictions(tiny_bundle):
    b = tiny_bundle["bundle"]
    df = tiny_bundle["df"]
    loaded = load_bundle(tiny_bundle["path"])
    a = predict_frame(b, df, with_explanations=True)
    c = predict_frame(loaded, df, with_explanations=True)
    for col in ["risk_log10_pred", "risk_log10_lower", "risk_log10_upper"]:
        np.testing.assert_allclose(a.predictions[col].to_numpy(float), c.predictions[col].to_numpy(float), rtol=0, atol=1e-9, equal_nan=True)
    assert a.predictions["risk_tier"].tolist() == c.predictions["risk_tier"].tolist()
    assert a.predictions["triage_action"].tolist() == c.predictions["triage_action"].tolist()
    assert loaded.interval_method == "cqr" and loaded.cqr.status == "ok"


def test_one_ordered_record_per_input_row_and_contract(tiny_bundle):
    b, df = tiny_bundle["bundle"], tiny_bundle["df"]
    out = predict_frame(b, df, with_explanations=True).predictions
    assert len(out) == len(df)
    assert out["row_id"].tolist() == df["row_id"].tolist()
    assert (out["source_row_index"].to_numpy() == df["source_row_index"].to_numpy()).all()
    for i in range(len(out)):
        d = out.iloc[i].to_dict()
        d = {k: (None if isinstance(v, float) and np.isnan(v) else (v.item() if isinstance(v, np.generic) else v)) for k, v in d.items()}
        PredictionRecord.model_validate(d)


def test_invalid_rows_null_tier_and_never_auto_clear(tiny_bundle):
    b, df = tiny_bundle["bundle"], tiny_bundle["df"]
    out = predict_frame(b, df, with_explanations=False).predictions
    inv = out[out["prediction_status"] == "invalid"]
    assert len(inv) >= 2
    assert inv["risk_tier"].isna().all()
    assert (inv["triage_action"] == "Human review - insufficient data").all()
    assert inv["risk_log10_pred"].isna().all() and inv["pc_pred"].isna().all()
    assert inv["explanation_id"].isna().all()
    # warnings with blocking flags never auto-clear
    warn = out[(out["prediction_status"] == "warning")]
    blocking = warn[warn["quality_flags"].apply(lambda fl: any(f not in b.cfg["triage"]["auto_clear_allowed_flags"] for f in fl))]
    assert not (blocking["triage_action"] == "Auto-clear candidate").any()


def test_missing_interval_never_auto_clears(tiny_bundle):
    b, df = tiny_bundle["bundle"], tiny_bundle["df"]
    stripped = Bundle(path=None, meta=b.meta, cfg=b.cfg, booster=b.booster, classifier=None, conformal=None, sigmoid=None, interval_method="residual")
    out = predict_frame(stripped, df, with_explanations=False).predictions
    assert out["risk_log10_upper"].isna().all()
    assert not (out["triage_action"] == "Auto-clear candidate").any()
    safe = out[out["risk_tier"] == "Safe"]
    assert safe["reason_codes"].apply(lambda rc: "INTERVAL_UNAVAILABLE" in rc).all()


def test_explanations_identify_component_and_are_additive(tiny_bundle):
    b, df = tiny_bundle["bundle"], tiny_bundle["df"]
    res = predict_frame(b, df, with_explanations=True)
    ex = res.explanations
    assert (ex["component"] == "lgbm_point_regressor").all()
    assert (ex["output_units"] == "log10(Pc)").all()
    # explained output + postprocessing note; final_output equals the clipped point prediction
    joined = res.predictions.merge(ex[["explanation_id", "explained_output", "final_output", "baseline", "top_positive", "top_negative", "remaining_contribution"]], on="explanation_id")
    np.testing.assert_allclose(joined["final_output"].to_numpy(float), joined["risk_log10_pred"].to_numpy(float), atol=1e-9)
    for _, r in joined.head(20).iterrows():
        shown = sum(c["contribution"] for c in json.loads(r["top_positive"])) + sum(c["contribution"] for c in json.loads(r["top_negative"]))
        assert abs(r["baseline"] + shown + r["remaining_contribution"] - r["explained_output"]) < 1e-6
    assert set(res.predictions.loc[res.predictions["prediction_status"] != "invalid", "explanation_id"]) == set(ex["explanation_id"])


def test_predict_batch_upload_path_with_and_without_history(tiny_bundle, synthetic_raw):
    b = tiny_bundle["bundle"]
    ev = synthetic_raw[synthetic_raw["event_id"] == 4].sort_values("time_to_tca", ascending=False)
    latest = ev.tail(1)
    hist = ev.iloc[:-1]
    standalone = predict_batch(b, latest, history=None, dataset_id="up").predictions
    with_hist = predict_batch(b, latest, history=hist, dataset_id="up").predictions
    assert len(standalone) == 1 and len(with_hist) == 1
    assert "NO_HISTORY" in standalone["reason_codes"].iloc[0]
    assert "NO_HISTORY" not in with_hist["reason_codes"].iloc[0]


def test_feature_pipeline_identity_full_vs_single_event(tiny_bundle):
    """Predictions for an event are identical whether predicted alone or inside the full batch."""
    b, df = tiny_bundle["bundle"], tiny_bundle["df"]
    full = predict_frame(b, df, with_explanations=False).predictions
    one = predict_frame(b, df[df["event_id"] == 11], with_explanations=False).predictions
    m = full[full["event_id"] == 11].reset_index(drop=True)
    np.testing.assert_allclose(m["risk_log10_pred"].to_numpy(float), one["risk_log10_pred"].to_numpy(float), atol=1e-9)


@pytest.mark.parametrize("bad_time", [None, "not-a-time", np.inf, -np.inf])
def test_predict_batch_invalid_time_preserves_rows_and_valid_predictions(tiny_bundle, bad_time):
    b, df = tiny_bundle["bundle"], tiny_bundle["df"]
    good = df[df["event_id"] == 4].copy()
    bad = good.iloc[[0]].copy()
    bad["row_id"] = "invalid-time"
    bad["time_to_tca"] = bad_time
    bad["max_risk_estimate"] = -0.1
    rows = pd.concat([bad, good], ignore_index=True)
    result = predict_batch(b, rows, dataset_id="check", with_explanations=True)
    out = result.predictions
    assert out["row_id"].tolist() == rows["row_id"].tolist()
    invalid = out.iloc[0]
    assert invalid["prediction_status"] == "invalid"
    assert pd.isna(invalid["risk_tier"]) and pd.isna(invalid["explanation_id"])
    assert invalid["triage_action"] == "Human review - insufficient data"
    assert "MISSING_REQUIRED_GEOMETRY" in invalid["quality_flags"]
    assert invalid[["risk_log10_pred", "pc_pred", "risk_log10_lower", "risk_log10_upper"]].isna().all()
    expected = predict_batch(b, good, dataset_id="check", with_explanations=True)
    pd.testing.assert_frame_equal(out.iloc[1:].reset_index(drop=True), expected.predictions)
    assert "invalid-time" not in set(result.explanations["row_id"])


def test_predict_batch_missing_time_column_returns_invalid_records(tiny_bundle):
    b, df = tiny_bundle["bundle"], tiny_bundle["df"]
    rows = df[df["event_id"] == 4].drop(columns="time_to_tca")
    result = predict_batch(b, rows, with_explanations=True)
    assert result.predictions["row_id"].tolist() == rows["row_id"].tolist()
    assert result.predictions["prediction_status"].eq("invalid").all()
    assert result.predictions["triage_action"].eq("Human review - insufficient data").all()
    assert result.predictions["risk_log10_pred"].isna().all()
    assert result.explanations is None


def test_mixed_dataset_predictions_match_separate_predictions(tiny_bundle):
    b, df = tiny_bundle["bundle"], tiny_bundle["df"]
    a = df[df["event_id"] == 4].copy()
    other = a.copy()
    other["dataset_id"] = "other"
    other["row_id"] = "other:" + other["source_row_index"].astype(str)
    other["max_risk_estimate"] += 0.5
    rows = pd.concat([a, other], ignore_index=True).sample(frac=1, random_state=8).reset_index(drop=True)
    actual = predict_frame(b, rows, with_explanations=False).predictions
    expected = pd.concat([predict_frame(b, x, with_explanations=False).predictions for x in (a, other)])
    assert actual["row_id"].tolist() == rows["row_id"].tolist()
    pd.testing.assert_frame_equal(actual.set_index("row_id").sort_index(), expected.set_index("row_id").sort_index())
