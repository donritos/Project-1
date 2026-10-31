import json
import re
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from collision_risk.explain import (
    EXPLANATION_COLUMNS,
    FEATURE_GROUPS,
    FEATURE_LABELS,
    GROUP_NAMES,
    UNITS_BINARY,
    UNITS_MULTICLASS,
    ForecastExplanationRecord,
    critical_class_contributions,
    explain_forecasts,
    feature_group,
    feature_label,
    frame_to_records,
    render_forecast_text,
)

NUMERIC = [
    "log_miss_distance",
    "time_to_tca",
    "max_risk_estimate",
    "slope_risk",
    "margin_to_critical",
    "F10",
]
CATEGORICAL = "c_object_type"
FEATURES = NUMERIC + [CATEGORICAL]
OBJECT_TYPES = ["DEBRIS", "PAYLOAD", "UNKNOWN"]
CRITICAL_INDEX = 2
CAUSAL_WORDS = ("cause", "because", "leads to")


def _make_data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "log_miss_distance": rng.uniform(2.0, 5.0, n),
            "time_to_tca": rng.uniform(0.0, 7.0, n),
            "max_risk_estimate": rng.uniform(-9.0, -2.0, n),
            "slope_risk": rng.normal(0.0, 1.0, n),
            "margin_to_critical": rng.uniform(-6.0, 3.0, n),
            "F10": rng.uniform(60.0, 200.0, n),
            CATEGORICAL: pd.Series(rng.choice(OBJECT_TYPES, n)).astype("category"),
        }
    )
    cat_effect = X[CATEGORICAL].map({"DEBRIS": 0.8, "PAYLOAD": -0.8, "UNKNOWN": 0.0}).astype(float)
    score = (
        -1.5 * (X["log_miss_distance"] - 3.5)
        + 0.6 * (X["max_risk_estimate"] + 5.5)
        + 0.5 * X["slope_risk"]
        + 0.4 * X["margin_to_critical"]
        + cat_effect.to_numpy()
        + rng.normal(0, 0.5, n)
    )
    y3 = np.digitize(score, np.quantile(score, [0.5, 0.8]))  # 0 Safe, 1 Elevated, 2 Critical
    y_bin = (y3 == CRITICAL_INDEX).astype(int)
    # a few NaNs in a numeric feature to exercise value=None
    X.loc[X.index[:3], "F10"] = np.nan
    return X, y3, y_bin


_BASE_PARAMS = {
    "learning_rate": 0.1,
    "num_leaves": 7,
    "min_data_in_leaf": 10,
    "verbose": -1,
    "seed": 0,
    "deterministic": True,
    "num_threads": 1,
}


@pytest.fixture(scope="module")
def data():
    return _make_data()


@pytest.fixture(scope="module")
def multiclass_booster(data):
    X, y3, _ = data
    ds = lgb.Dataset(X, label=y3, categorical_feature=[CATEGORICAL], free_raw_data=False)
    params = {**_BASE_PARAMS, "objective": "multiclass", "num_class": 3}
    return lgb.train(params, ds, num_boost_round=40)


@pytest.fixture(scope="module")
def binary_booster(data):
    X, _, y_bin = data
    ds = lgb.Dataset(X, label=y_bin, categorical_feature=[CATEGORICAL], free_raw_data=False)
    params = {**_BASE_PARAMS, "objective": "binary"}
    return lgb.train(params, ds, num_boost_round=40)


def _explain(booster, X, class_index, **kw):
    n = len(X)
    row_ids = [f"r{i}" for i in range(n)]
    if class_index is None:
        p = booster.predict(X)
        tiers = np.where(p >= 0.5, "Critical", "Elevated")
    else:
        proba = booster.predict(X)
        p = proba[:, class_index]
        tiers = np.array(["Safe", "Elevated", "Critical"])[proba.argmax(axis=1)]
    kwargs = dict(
        class_index=class_index,
        row_ids=row_ids,
        dataset_id="ds-test",
        model_version="m-test",
        p_critical=p,
        predicted_tier=tiers,
    )
    kwargs.update(kw)
    return explain_forecasts(booster, X, **kwargs)


@pytest.fixture(scope="module")
def mc_frame(multiclass_booster, data):
    X = data[0].iloc[:80]
    return _explain(multiclass_booster, X, CRITICAL_INDEX)


# --------------------------------------------------------------------------- #
# labels / groups
# --------------------------------------------------------------------------- #
def test_labels_and_groups():
    assert feature_label("log_miss_distance") == FEATURE_LABELS["log_miss_distance"]
    assert feature_label("not_a_feature") == "not_a_feature"
    for f in ("n_cdms_2d", "span_days", "slope_risk", "margin_to_critical", "latest_risk_is_floor", "risk"):
        assert f in FEATURE_LABELS
    assert FEATURE_LABELS["margin_to_critical"] == "Margin to Critical (risk + 4)"
    assert set(FEATURE_GROUPS.values()) <= set(GROUP_NAMES)
    assert feature_group("c_object_type") == "object_type"
    assert feature_group("F10") == "space_weather"
    assert feature_group("mystery") == "other"
    # every labelled feature has a group
    assert not (set(FEATURE_LABELS) - set(FEATURE_GROUPS))


# --------------------------------------------------------------------------- #
# contributions are exact
# --------------------------------------------------------------------------- #
def test_multiclass_contribs_match_raw_score(multiclass_booster, data):
    X = data[0].iloc[:200]
    contribs, bias, names = critical_class_contributions(multiclass_booster, X, class_index=CRITICAL_INDEX)
    assert names == list(multiclass_booster.feature_name())
    assert contribs.shape == (len(X), len(names))
    assert bias.shape == (len(X),)
    assert contribs.dtype == np.float64 and bias.dtype == np.float64
    raw = multiclass_booster.predict(X, raw_score=True)[:, CRITICAL_INDEX]
    np.testing.assert_allclose(bias + contribs.sum(axis=1), raw, rtol=1e-6, atol=1e-6)


def test_multiclass_other_class_block(multiclass_booster, data):
    X = data[0].iloc[:50]
    for k in range(3):
        contribs, bias, _ = critical_class_contributions(multiclass_booster, X, class_index=k)
        raw = multiclass_booster.predict(X, raw_score=True)[:, k]
        np.testing.assert_allclose(bias + contribs.sum(axis=1), raw, rtol=1e-6, atol=1e-6)


def test_binary_contribs_match_raw_score(binary_booster, data):
    X = data[0].iloc[:200]
    contribs, bias, names = critical_class_contributions(binary_booster, X, class_index=None)
    assert contribs.shape == (len(X), len(names))
    raw = binary_booster.predict(X, raw_score=True)
    np.testing.assert_allclose(bias + contribs.sum(axis=1), raw, rtol=1e-6, atol=1e-6)


def test_reorders_columns(multiclass_booster, data):
    X = data[0].iloc[:30]
    shuffled = X[list(reversed(X.columns))]
    a, ba, _ = critical_class_contributions(multiclass_booster, X, class_index=CRITICAL_INDEX)
    b, bb, _ = critical_class_contributions(multiclass_booster, shuffled, class_index=CRITICAL_INDEX)
    np.testing.assert_allclose(a, b)
    np.testing.assert_allclose(ba, bb)


def test_shape_and_feature_errors(multiclass_booster, binary_booster, data):
    X = data[0].iloc[:10]
    with pytest.raises(ValueError, match="missing model features"):
        critical_class_contributions(multiclass_booster, X.drop(columns=["F10"]), class_index=CRITICAL_INDEX)
    # multiclass booster explained as binary -> wrong width
    with pytest.raises(ValueError):
        critical_class_contributions(multiclass_booster, X, class_index=None)
    # class index out of range
    with pytest.raises(ValueError, match="out of range"):
        critical_class_contributions(multiclass_booster, X, class_index=7)
    with pytest.raises(ValueError):
        critical_class_contributions(binary_booster, X, class_index=-1)


def test_empty_input(multiclass_booster, data):
    X = data[0].iloc[:0]
    contribs, bias, names = critical_class_contributions(multiclass_booster, X, class_index=CRITICAL_INDEX)
    assert contribs.shape == (0, len(names)) and bias.shape == (0,)
    df = explain_forecasts(
        multiclass_booster, X, class_index=CRITICAL_INDEX, row_ids=[], dataset_id="d",
        model_version="m", p_critical=np.array([]), predicted_tier=[],
    )
    assert len(df) == 0 and list(df.columns) == EXPLANATION_COLUMNS


# --------------------------------------------------------------------------- #
# explain_forecasts frame
# --------------------------------------------------------------------------- #
def test_frame_columns_ids_and_units(mc_frame, binary_booster, data):
    df = mc_frame
    assert list(df.columns) == EXPLANATION_COLUMNS
    assert df["explanation_id"].is_unique
    assert (df["explanation_id"] == "exp:" + df["row_id"]).all()
    assert (df["output_units"] == UNITS_MULTICLASS).all()
    assert (df["component"] == "lgbm_multiclass_critical_logit").all()
    assert (df["dataset_id"] == "ds-test").all() and (df["model_version"] == "m-test").all()
    assert set(df["predicted_tier"]) <= {"Safe", "Elevated", "Critical"}
    assert df["p_critical"].between(0, 1).all()

    bdf = _explain(binary_booster, data[0].iloc[:20], None, component="lgbm_binary_critical_logit",
                   explanation_id_prefix="bexp")
    assert (bdf["output_units"] == UNITS_BINARY).all()
    assert (bdf["explanation_id"] == "bexp:" + bdf["row_id"]).all()


def test_frame_explained_output_matches_raw(mc_frame, multiclass_booster, data):
    X = data[0].iloc[:80]
    raw = multiclass_booster.predict(X, raw_score=True)[:, CRITICAL_INDEX]
    np.testing.assert_allclose(mc_frame["explained_output"].to_numpy(), raw, rtol=1e-6, atol=1e-6)


def test_top_lists_sizes_sorting_and_accounting(mc_frame, multiclass_booster, data):
    X = data[0].iloc[:80]
    contribs, bias, names = critical_class_contributions(multiclass_booster, X, class_index=CRITICAL_INDEX)
    recs = frame_to_records(mc_frame)
    for i, rec in enumerate(recs):
        pos, neg = rec["top_positive"], rec["top_negative"]
        assert len(pos) <= 5 and len(neg) <= 5
        pc = [fc["contribution"] for fc in pos]
        nc = [fc["contribution"] for fc in neg]
        assert all(c > 0 for c in pc) and pc == sorted(pc, reverse=True)
        assert all(c < 0 for c in nc) and nc == sorted(nc)
        # exactly the k largest positives / most negative
        row = contribs[i]
        n_pos = min(5, int((row > 0).sum()))
        n_neg = min(5, int((row < 0).sum()))
        assert len(pos) == n_pos and len(neg) == n_neg
        np.testing.assert_allclose(pc, np.sort(row[row > 0])[::-1][:5])
        np.testing.assert_allclose(nc, np.sort(row[row < 0])[:5])
        # no feature in both lists, features exist, groups match
        pf = {fc["feature"] for fc in pos}
        nf = {fc["feature"] for fc in neg}
        assert not (pf & nf) and (pf | nf) <= set(names)
        for fc in pos + neg:
            assert fc["group"] == feature_group(fc["feature"])
            assert set(fc) == {"feature", "label", "value", "contribution", "group"}
        # shown + remaining == explained - baseline
        shown = sum(pc) + sum(nc)
        assert np.isclose(shown + rec["remaining_contribution"], rec["explained_output"] - rec["baseline"], atol=1e-9)
        # group sums == sum of all contribs
        gc = rec["group_contributions"]
        assert set(gc) <= set(GROUP_NAMES)
        assert np.isclose(sum(gc.values()), row.sum(), atol=1e-9)
        assert np.isclose(gc["object_type"], row[names.index(CATEGORICAL)], atol=1e-12)


def test_top_k_override(multiclass_booster, data):
    X = data[0].iloc[:20]
    df = _explain(multiclass_booster, X, CRITICAL_INDEX, top_k=2)
    recs = frame_to_records(df)
    assert all(len(r["top_positive"]) <= 2 and len(r["top_negative"]) <= 2 for r in recs)
    df0 = _explain(multiclass_booster, X, CRITICAL_INDEX, top_k=0)
    for r in frame_to_records(df0):
        assert r["top_positive"] == [] and r["top_negative"] == []
        assert np.isclose(r["remaining_contribution"], r["explained_output"] - r["baseline"])
    with pytest.raises(ValueError):
        _explain(multiclass_booster, X, CRITICAL_INDEX, top_k=-1)


def test_values_numeric_nan_and_categorical(mc_frame, data):
    X = data[0].iloc[:80]
    recs = frame_to_records(mc_frame)
    seen_cat = seen_nan = False
    for i, rec in enumerate(recs):
        for fc in rec["top_positive"] + rec["top_negative"]:
            f = fc["feature"]
            if f == CATEGORICAL:
                seen_cat = True
                assert fc["value"] is None
                assert " = " in fc["label"]
                assert fc["label"] == f"{feature_label(f)} = {X[CATEGORICAL].iloc[i]}"
            else:
                assert fc["label"] == feature_label(f)
                raw = X[f].iloc[i]
                if np.isnan(raw):
                    seen_nan = True
                    assert fc["value"] is None
                else:
                    assert fc["value"] == pytest.approx(float(raw))
    assert seen_cat, "categorical feature never appeared in a top list"
    assert seen_nan, "NaN numeric never appeared in a top list"


def test_length_mismatch_errors(multiclass_booster, data):
    X = data[0].iloc[:5]
    base = dict(class_index=CRITICAL_INDEX, dataset_id="d", model_version="m")
    with pytest.raises(ValueError, match="row_ids"):
        explain_forecasts(multiclass_booster, X, row_ids=["a"], p_critical=np.zeros(5),
                          predicted_tier=["Safe"] * 5, **base)
    with pytest.raises(ValueError, match="p_critical"):
        explain_forecasts(multiclass_booster, X, row_ids=list("abcde"), p_critical=np.zeros(2),
                          predicted_tier=["Safe"] * 5, **base)
    with pytest.raises(ValueError, match="predicted_tier"):
        explain_forecasts(multiclass_booster, X, row_ids=list("abcde"), p_critical=np.zeros(5),
                          predicted_tier=["Safe"], **base)


# --------------------------------------------------------------------------- #
# text
# --------------------------------------------------------------------------- #
def test_render_text_content(mc_frame):
    recs = frame_to_records(mc_frame)
    for rec in recs:
        txt = rec["text"]
        assert txt == render_forecast_text(rec)
        assert txt.startswith(f"Forecast {rec['predicted_tier']} (P = {rec['p_critical']:.2f}).")
        assert "associated" in txt
        assert "P = " in txt
        assert "contributions in log-odds; the probability is calibrated separately" in txt.lower()
        assert txt.count("log-odds") == 1
        assert len(txt) <= 600
        low = txt.lower()
        for w in CAUSAL_WORDS:
            assert w not in low, (w, txt)
        if rec["top_positive"]:
            assert rec["top_positive"][0]["label"] in txt
        if rec["top_negative"]:
            assert rec["top_negative"][0]["label"] in txt


def test_render_text_accepts_json_strings(mc_frame):
    row = mc_frame.iloc[0].to_dict()
    assert isinstance(row["top_positive"], str)
    assert render_forecast_text(row) == row["text"]


def test_render_text_no_drivers_and_missing_prob():
    txt = render_forecast_text(
        {"predicted_tier": None, "p_critical": float("nan"), "top_positive": [], "top_negative": [],
         "remaining_contribution": 0.0}
    )
    assert txt.startswith("Forecast n/a (P = n/a).")
    assert "No feature is associated with a shift from the baseline." in txt
    assert "Other features" not in txt
    assert "log-odds" in txt


def test_render_text_truncates():
    pos = [{"feature": f"f{i}", "label": "A very long feature label " * 3, "value": 1.0,
            "contribution": 0.5, "group": "other"} for i in range(10)]
    txt = render_forecast_text({"predicted_tier": "Critical", "p_critical": 0.9, "top_positive": pos,
                                "top_negative": [], "remaining_contribution": 0.1})
    assert len(txt) <= 600 and txt.endswith("...")


def test_no_causal_words_in_module_templates():
    import inspect

    import collision_risk.explain as mod

    src = inspect.getsource(mod.render_forecast_text) + inspect.getsource(mod._driver_phrase)
    strings = re.findall(r'"([^"]*)"', src)
    for s in strings:
        for w in CAUSAL_WORDS:
            assert w not in s.lower()


# --------------------------------------------------------------------------- #
# roundtrip
# --------------------------------------------------------------------------- #
def test_frame_to_records_roundtrip(mc_frame, tmp_path):
    recs = frame_to_records(mc_frame)
    assert len(recs) == len(mc_frame)
    for rec in recs:
        assert isinstance(rec["top_positive"], list) and isinstance(rec["group_contributions"], dict)
        ForecastExplanationRecord.model_validate(rec)  # dict validates against the contract
    # json columns are plain strings that decode to the same payloads
    assert json.loads(mc_frame["top_positive"].iloc[0]) == recs[0]["top_positive"]
    # survives parquet
    path = tmp_path / "expl.parquet"
    mc_frame.to_parquet(path, index=False)
    back = frame_to_records(pd.read_parquet(path))
    assert back == recs


def test_frame_to_records_handles_missing():
    df = pd.DataFrame(
        {
            "explanation_id": ["exp:a"], "dataset_id": ["d"], "row_id": ["a"], "component": ["c"],
            "output_units": ["log-odds"], "baseline": [0.0], "explained_output": [0.0],
            "p_critical": [np.nan], "predicted_tier": [None], "top_positive": [None], "top_negative": ["[]"],
            "remaining_contribution": [0.0], "group_contributions": [None], "text": [""], "model_version": ["m"],
        }
    )
    rec = frame_to_records(df)[0]
    assert rec["p_critical"] is None and rec["predicted_tier"] is None
    assert rec["top_positive"] == [] and rec["group_contributions"] == {}
    ForecastExplanationRecord.model_validate(rec)


# --------------------------------------------------------------------------- #
# performance
# --------------------------------------------------------------------------- #
def test_timing_5000_rows(multiclass_booster, data):
    X, _, _ = data
    n = 5000
    rng = np.random.default_rng(1)
    idx = rng.integers(0, len(X), n)
    Xb = X.iloc[idx].reset_index(drop=True)
    proba = multiclass_booster.predict(Xb)
    row_ids = [f"r{i}" for i in range(n)]
    tiers = np.array(["Safe", "Elevated", "Critical"])[proba.argmax(axis=1)]
    t0 = time.perf_counter()
    df = explain_forecasts(
        multiclass_booster, Xb, class_index=CRITICAL_INDEX, row_ids=row_ids, dataset_id="d",
        model_version="m", p_critical=proba[:, CRITICAL_INDEX], predicted_tier=tiers,
    )
    elapsed = time.perf_counter() - t0
    print(f"\nexplain_forecasts: {n} rows in {elapsed:.3f} s ({n / elapsed:,.0f} rows/s)")
    assert len(df) == n
    assert elapsed < 5.0, f"too slow: {elapsed:.2f} s for {n} rows"
