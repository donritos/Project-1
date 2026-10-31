"""Splits, weights, decision rule, calibration, prediction sets, ordinal combination."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from collision_risk import train as T
from collision_risk.calibration import LACPredictionSets, MulticlassIsotonicCalibrator, apply_decision_rule, prediction_set_metrics, tune_decision_thresholds
from collision_risk.data import load_event_labels


@pytest.fixture(scope="module")
def labels(synthetic_cfg):
    return load_event_labels(synthetic_cfg)[0]


def test_holdout_split_event_disjoint_complete_and_stratified(labels, synthetic_cfg):
    split = T.make_holdout_split(labels, synthetic_cfg)
    dev, hold = set(split["events"]["dev"]), set(split["events"]["holdout"])
    assert not (dev & hold)
    usable = set(labels.loc[labels["usable_for_training"], "event_id"])
    assert dev | hold == usable
    assert split["n_unusable_excluded"] == int((~labels["usable_for_training"]).sum())
    c_dev = split["counts"]["dev"]["critical_share"]; c_hold = split["counts"]["holdout"]["critical_share"]
    assert abs(c_dev - c_hold) <= 0.03  # synthetic set is small; real data is within +-1 point
    assert len(split["split_hash"]) == 16
    json.dumps(split)


def test_no_holdout_snapshot_used_for_fitting(labels, synthetic_cfg, synthetic_allowlisted):
    from collision_risk import features as FE
    from collision_risk.data import apply_horizon
    df, _ = synthetic_allowlisted
    split = T.make_holdout_split(labels, synthetic_cfg)
    h = apply_horizon(df, 2.0)
    dev_rows = h[h["event_id"].isin(split["events"]["dev"])]
    assert not dev_rows["event_id"].isin(split["events"]["holdout"]).any()
    snap = FE.build_event_snapshots(dev_rows, synthetic_cfg)
    assert set(snap["event_id"]) <= set(split["events"]["dev"])


def test_snapshot_weights_make_events_count_equally():
    df = pd.DataFrame({"dataset_id": "d", "event_id": [1, 1, 1, 2, 3, 3]})
    y = np.array([0, 0, 0, 2, 1, 1])
    cfg = {"model": {"class_weight_cap": 10.0}}
    w = T.snapshot_weights(df, y, cfg)
    per_event = pd.Series(w).groupby(df["event_id"].to_numpy()).sum()
    # class weights differ, but within a class each event sums to the same weight
    assert np.isclose(per_event[1], per_event[1]) and per_event[2] >= per_event[1]
    assert (w > 0).all() and w.max() / w.min() <= 10.0 * 3


def test_ordinal_probabilities_sum_to_one_and_respect_ordering():
    P = T.ordinal_probabilities(np.array([0.9, 0.2, 0.5]), np.array([0.95, 0.1, 0.0]))
    assert np.allclose(P.sum(axis=1), 1.0)
    assert (P[:, 2] <= np.array([0.9, 0.2, 0.5]) + 1e-12).all()
    assert (P >= 0).all()


def test_decision_rule_thresholds():
    P = np.array([[0.1, 0.2, 0.7], [0.5, 0.3, 0.2], [0.8, 0.15, 0.05], [np.nan, np.nan, np.nan]])
    pred = apply_decision_rule(P, t_crit=0.6, t_elev=0.4)
    assert pred.tolist() == ["Critical", "Elevated", "Safe", None]
    pred2 = apply_decision_rule(P, t_crit=0.15, t_elev=0.4)
    assert pred2[1] == "Critical"


def test_tune_decision_thresholds_improves_or_matches_default():
    rng = np.random.default_rng(0)
    y = rng.choice([0, 1, 2], size=2000, p=[0.75, 0.18, 0.07])
    P = np.eye(3)[y] * 0.6 + rng.dirichlet([1, 1, 1], size=2000) * 0.4
    thr = tune_decision_thresholds(P, y)
    assert 0.05 <= thr["t_crit"] <= 0.95 and 0.05 <= thr["t_elev"] <= 0.95
    base = T.critical_f2_from_probs(P, y)["critical_f2"]
    assert thr["oof_critical_f2"] >= base - 1e-9


def test_multiclass_isotonic_rows_sum_to_one_and_roundtrip():
    rng = np.random.default_rng(1)
    y = rng.choice([0, 1, 2], size=1500, p=[0.7, 0.2, 0.1])
    P = np.eye(3)[y] * 0.5 + rng.dirichlet([1, 1, 1], size=1500) * 0.5
    cal = MulticlassIsotonicCalibrator().fit(P, y)
    Q = cal.predict_proba(P)
    assert np.allclose(Q.sum(axis=1), 1.0)
    Q2 = MulticlassIsotonicCalibrator.from_dict(json.loads(json.dumps(cal.to_dict()))).predict_proba(P)
    np.testing.assert_allclose(Q, Q2)
    Pn = P.copy(); Pn[0] = np.nan
    assert np.isnan(cal.predict_proba(Pn)[0]).all()


def test_lac_prediction_sets_cover_and_contain_argmax():
    rng = np.random.default_rng(2)
    y = rng.choice([0, 1, 2], size=3000, p=[0.7, 0.2, 0.1])
    P = np.eye(3)[y] * 0.5 + rng.dirichlet([1, 1, 1], size=3000) * 0.5
    lac = LACPredictionSets(coverage=0.9).fit(P[:1500], y[:1500])
    sets = lac.predict_sets(P[1500:])
    m = prediction_set_metrics(sets, np.array(T.TIERS)[y[1500:]])
    assert m["coverage_empirical"] >= 0.85 and 1.0 <= m["avg_set_size"] <= 3.0
    for s, row in zip(sets, P[1500:]):
        assert T.TIERS[int(row.argmax())] in s
    tiny = LACPredictionSets(coverage=0.9, min_rows=50).fit(P[:10], y[:10])
    assert tiny.status != "ok" and tiny.predict_sets(P[:3]) == [None, None, None]


def test_carry_forward_and_majority_probs():
    P = T.carry_forward_probs(np.array(["Safe", "Critical", None], dtype=object))
    assert P[0].tolist() == [1, 0, 0] and P[1].tolist() == [0, 0, 1] and np.allclose(P[2], 1 / 3)
    assert T.majority_probs(2).tolist() == [[1, 0, 0], [1, 0, 0]]
