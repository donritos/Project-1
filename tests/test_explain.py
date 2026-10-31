import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from collision_risk.contracts import ExplanationRecord, FeatureContribution
from collision_risk.explain import (
    EXPLANATION_COLUMNS,
    FEATURE_LABELS,
    compute_contributions,
    explanations_to_frame,
    feature_label,
    frame_to_explanations,
    render_text,
)

NUMERIC = ["log_miss_distance", "time_to_tca", "max_risk_estimate", "radial_sep_ratio", "F10"]
CATEGORICAL = "c_object_type"
FEATURES = NUMERIC + [CATEGORICAL]
OBJECT_TYPES = ["DEBRIS", "PAYLOAD", "UNKNOWN"]


def _make_data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "log_miss_distance": rng.uniform(2.0, 5.0, n),
            "time_to_tca": rng.uniform(0.0, 7.0, n),
            "max_risk_estimate": rng.uniform(-9.0, -2.0, n),
            "radial_sep_ratio": rng.uniform(0.0, 10.0, n),
            "F10": rng.uniform(60.0, 200.0, n),
            CATEGORICAL: pd.Series(rng.choice(OBJECT_TYPES, n)).astype("category"),
        }
    )
    cat_effect = X[CATEGORICAL].map({"DEBRIS": 0.5, "PAYLOAD": -0.5, "UNKNOWN": 0.0}).astype(float)
    y = (
        -2.0
        - 1.5 * X["log_miss_distance"]
        + 0.6 * X["max_risk_estimate"]
        - 0.2 * X["radial_sep_ratio"]
        + cat_effect.to_numpy()
        + rng.normal(0, 0.2, n)
    )
    # a few NaNs in a numeric feature to exercise value=None
    X.loc[X.index[:3], "F10"] = np.nan
    return X, y


@pytest.fixture(scope="module")
def booster_and_data():
    X, y = _make_data()
    ds = lgb.Dataset(X, label=y, categorical_feature=[CATEGORICAL], free_raw_data=False)
    params = {
        "objective": "regression",
        "learning_rate": 0.1,
        "num_leaves": 7,
        "min_data_in_leaf": 10,
        "verbose": -1,
        "seed": 0,
        "deterministic": True,
    }
    booster = lgb.train(params, ds, num_boost_round=30)
    return booster, X, y


@pytest.fixture(scope="module")
def records(booster_and_data):
    booster, X, _ = booster_and_data
    Xs = X.iloc[:50]
    row_ids = [f"r{i}" for i in range(len(Xs))]
    raw = booster.predict(Xs)
    final = np.maximum(raw, -6.0)  # pretend a floor was applied
    return compute_contributions(
        booster,
        Xs,
        row_ids=row_ids,
        dataset_id="unit",
        component="lgbm_point_regressor",
        model_version="test-0",
        final_outputs=final,
        postprocessing_note="floored at -6",
        top_k=3,
    )


def test_additivity_matches_raw_prediction(booster_and_data, records):
    booster, X, _ = booster_and_data
    Xs = X.iloc[:50]
    raw = booster.predict(Xs)
    contrib = booster.predict(Xs, pred_contrib=True)
    assert contrib.shape == (len(Xs), len(FEATURES) + 1)
    for i, rec in enumerate(records):
        all_contrib_sum = contrib[i, :-1].sum()
        assert rec.baseline == pytest.approx(contrib[i, -1])
        assert rec.explained_output == pytest.approx(rec.baseline + all_contrib_sum, abs=1e-9)
        assert rec.explained_output == pytest.approx(raw[i], abs=1e-6)
        assert rec.output_units == "log10(Pc)"


def test_top_lists_sizes_sorting_and_remaining(records):
    for rec in records:
        assert len(rec.top_positive) <= 3
        assert len(rec.top_negative) <= 3
        pos = [fc.contribution for fc in rec.top_positive]
        neg = [fc.contribution for fc in rec.top_negative]
        assert all(c > 0 for c in pos)
        assert all(c < 0 for c in neg)
        assert pos == sorted(pos, reverse=True)
        assert neg == sorted(neg)
        shown_feats = [fc.feature for fc in rec.top_positive + rec.top_negative]
        assert len(shown_feats) == len(set(shown_feats))
        shown_sum = sum(pos) + sum(neg)
        assert shown_sum + rec.remaining_contribution == pytest.approx(
            rec.explained_output - rec.baseline, abs=1e-9
        )


def test_top_k_truncation_vs_full(booster_and_data):
    booster, X, _ = booster_and_data
    Xs = X.iloc[:10]
    ids = [str(i) for i in range(10)]
    full = compute_contributions(
        booster, Xs, row_ids=ids, dataset_id="d", component="c", model_version="v", top_k=100
    )
    for rec in full:
        # with top_k large everything is shown -> remaining is 0
        assert rec.remaining_contribution == pytest.approx(0.0, abs=1e-12)
        assert len(rec.top_positive) + len(rec.top_negative) <= len(FEATURES)
        assert rec.final_output is None
    one = compute_contributions(
        booster, Xs, row_ids=ids, dataset_id="d", component="c", model_version="v", top_k=1
    )
    for rec in one:
        assert len(rec.top_positive) <= 1 and len(rec.top_negative) <= 1


def test_feature_values_and_labels(records, booster_and_data):
    _, X, _ = booster_and_data
    Xs = X.iloc[:50]
    for i, rec in enumerate(records):
        for fc in rec.top_positive + rec.top_negative:
            if fc.feature == CATEGORICAL:
                assert fc.value is None
                assert fc.label.startswith("Object type = ")
                assert fc.label.split("= ")[1] == str(Xs[CATEGORICAL].iloc[i])
            else:
                raw = Xs[fc.feature].iloc[i]
                if np.isnan(raw):
                    assert fc.value is None
                else:
                    assert fc.value == pytest.approx(float(raw))
                assert fc.label == FEATURE_LABELS[fc.feature]
    assert feature_label("not_a_feature") == "not_a_feature"
    assert FEATURE_LABELS["log_miss_distance"] == "Miss distance (log10 m)"
    assert FEATURE_LABELS["c_sigma_r"] == "Chaser radial sigma (m)"
    assert FEATURE_LABELS["max_risk_estimate"] == "Max risk estimate (log10 Pc)"
    assert FEATURE_LABELS["radial_sep_ratio"] == "Radial separation / combined sigma"
    assert FEATURE_LABELS["n_cdms_so_far"] == "CDMs seen so far"


def test_explanation_ids_unique_and_formatted(records):
    ids = [r.explanation_id for r in records]
    assert len(ids) == len(set(ids))
    assert ids[0] == "exp:unit:r0"
    assert all(r.dataset_id == "unit" for r in records)
    assert all(r.component == "lgbm_point_regressor" for r in records)


def test_render_text_content(records):
    for rec in records:
        assert rec.text == render_text(rec)
        assert "lgbm_point_regressor" in rec.text
        assert "associated" in rec.text
        assert "cause" not in rec.text.lower()
        assert "factor of 10" in rec.text
        assert len(rec.text) <= 600
        if rec.final_output != pytest.approx(rec.explained_output):
            assert "floored at -6" in rec.text
        else:
            assert "floored at -6" not in rec.text


def test_render_text_with_no_drivers():
    rec = ExplanationRecord(
        explanation_id="exp:d:1",
        dataset_id="d",
        row_id="1",
        component="comp",
        baseline=-5.0,
        explained_output=-5.0,
        final_output=None,
        postprocessing_note=None,
        top_positive=[],
        top_negative=[],
        remaining_contribution=0.0,
        text="",
        model_version="v",
    )
    txt = render_text(rec)
    assert "comp" in txt and "associated" in txt


def test_frame_roundtrip(records):
    df = explanations_to_frame(records)
    assert list(df.columns) == EXPLANATION_COLUMNS
    assert len(df) == len(records)
    # JSON-serialised lists
    parsed = json.loads(df["top_positive"].iloc[0])
    assert isinstance(parsed, list)
    for item in parsed:
        FeatureContribution.model_validate(item)
    back = frame_to_explanations(df)
    assert len(back) == len(records)
    for a, b in zip(records, back):
        assert a.model_dump() == b.model_dump()


def test_frame_roundtrip_through_parquet(records, tmp_path):
    pytest.importorskip("pyarrow")
    df = explanations_to_frame(records)
    path = tmp_path / "explanations.parquet"
    df.to_parquet(path, index=False)
    back = frame_to_explanations(pd.read_parquet(path))
    for a, b in zip(records, back):
        assert a.model_dump() == b.model_dump()


def test_input_validation(booster_and_data):
    booster, X, _ = booster_and_data
    with pytest.raises(ValueError):
        compute_contributions(
            booster, X.iloc[:5], row_ids=["a"], dataset_id="d", component="c", model_version="v"
        )
    with pytest.raises(ValueError):
        compute_contributions(
            booster,
            X.iloc[:5].drop(columns=["F10"]),
            row_ids=list("abcde"),
            dataset_id="d",
            component="c",
            model_version="v",
        )
    assert (
        compute_contributions(
            booster, X.iloc[:0], row_ids=[], dataset_id="d", component="c", model_version="v"
        )
        == []
    )
