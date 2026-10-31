"""Tests for collision_risk.evaluate (final-tier forecast evaluation) and scripts/make_results.py."""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, log_loss

from collision_risk.config import load_config
from collision_risk.evaluate import (
    TIERS,
    bootstrap_cis,
    breakdown_by,
    classification_metrics,
    compare_models,
    escalation_metrics,
    evaluate_forecasts,
    find_illustrative_cases,
    latest_snapshot_view,
    lead_time_breakdown,
    probability_metrics,
    to_jsonable,
    triage_metrics,
)

ROOT = Path(__file__).resolve().parents[1]

AUTO = "Auto-clear candidate"
HUMAN = "Human review"
HUMAN_INSUFF = "Human review - insufficient data"
MANEUVER = "Maneuver planning review"
MONITOR = "Monitor"
WATCH = "Escalation watch"
S, E, C = "Safe", "Elevated", "Critical"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config()


def _triage(pred, cur, p_safe, p_crit):
    if pred is None:
        return HUMAN_INSUFF
    if pred == C:
        return MANEUVER
    if cur == C:
        return HUMAN
    if p_crit >= 0.2:
        return WATCH
    if pred == S and p_safe >= 0.9:
        return AUTO
    if pred == E:
        return MONITOR
    return HUMAN


def make_forecast_frame(n_events: int = 600, seed: int = 7, n_invalid: int = 5) -> pd.DataFrame:
    """Synthetic latest-snapshot frame: ~7 % final Critical, current tier mostly equal
    to the final tier with some escalations / de-escalations, ~25 % trivial events,
    probabilities roughly consistent with the predictions, rule-based triage actions,
    and a few invalid rows (no prediction, NaN probabilities)."""
    rng = np.random.default_rng(seed)
    n = n_events
    final = rng.choice(TIERS, size=n, p=[0.68, 0.25, 0.07]).astype(object)
    current = final.copy()
    for i in np.flatnonzero(rng.random(n) < 0.2):
        current[i] = rng.choice([t for t in TIERS if t != final[i]])
    is_trivial = rng.random(n) < 0.25
    current[is_trivial] = final[is_trivial]

    final_idx = np.asarray([TIERS.index(t) for t in final])
    P = rng.dirichlet([1.0, 1.0, 1.0], size=n)
    boost = np.where(rng.random(n) < 0.75, final_idx, rng.integers(0, 3, size=n))
    P[np.arange(n), boost] += rng.uniform(1.0, 3.0, size=n)
    P /= P.sum(axis=1, keepdims=True)
    pred = np.asarray([TIERS[i] for i in P.argmax(axis=1)], dtype=object)

    ttca = rng.uniform(2.0, 7.0, size=n)
    obj = rng.choice(["DEBRIS", "PAYLOAD", "UNKNOWN", "ROCKET BODY", "TBA"], size=n, p=[0.6, 0.2, 0.15, 0.04, 0.01])

    df = pd.DataFrame({
        "dataset_id": "kelvins_train",
        "event_id": np.arange(n),
        "row_id": [f"kelvins_train:{i}" for i in range(n)],
        "time_to_tca_days": ttca,
        "miss_distance_m": 10 ** rng.uniform(2.5, 4.8, size=n),
        "is_latest_snapshot": True,
        "n_cdms_2d": rng.integers(1, 12, size=n),
        "current_tier": current,
        "carry_forward_tier": current,
        "prediction_status": "valid",
        "p_safe": P[:, 0], "p_elevated": P[:, 1], "p_critical": P[:, 2],
        "predicted_final_tier": pred,
        "final_tier": final,
        "is_trivial": is_trivial,
        "object_type": obj,
        "mission_id": rng.integers(1, 6, size=n),
        "split": "holdout",
    })
    df["prediction_set"] = [[t for t, p in zip(TIERS, row) if 1.0 - p <= 0.7] for row in P]
    inv = rng.choice(n, size=n_invalid, replace=False)
    df.loc[inv, ["p_safe", "p_elevated", "p_critical"]] = np.nan
    df["predicted_final_tier"] = df["predicted_final_tier"].astype(object)
    df.loc[inv, "predicted_final_tier"] = None
    df.loc[inv, "prediction_status"] = "invalid"
    df["prediction_set"] = df["prediction_set"].astype(object)
    df.loc[inv, "prediction_set"] = None
    df["triage_action"] = [
        _triage(p, c, ps, pc)
        for p, c, ps, pc in zip(df["predicted_final_tier"], df["current_tier"], df["p_safe"], df["p_critical"])
    ]
    df["reason_codes"] = [["FORECAST_" + str(p).upper()] if p else ["INVALID_INPUT"] for p in df["predicted_final_tier"]]
    return df


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return make_forecast_frame()


# --------------------------------------------------------------------------- #
# Views
# --------------------------------------------------------------------------- #
def test_latest_snapshot_view_uses_flag_or_min_time():
    df = pd.DataFrame({
        "dataset_id": ["a"] * 3 + ["b"] * 2,
        "event_id": [1, 1, 1, 1, 1],
        "time_to_tca_days": [5.0, 3.0, 2.5, 4.0, 2.0],
        "is_latest_snapshot": [False, False, True, False, True],
    })
    v = latest_snapshot_view(df)
    assert v.index.tolist() == [2, 4]
    v2 = latest_snapshot_view(df.drop(columns=["is_latest_snapshot"]))
    assert v2.index.tolist() == [2, 4]
    assert len(latest_snapshot_view(df.iloc[0:0])) == 0


# --------------------------------------------------------------------------- #
# Classification metrics
# --------------------------------------------------------------------------- #
def test_classification_metrics_hand_computed():
    y_true = [C, C, C, E, E, S, S, S]
    y_pred = [C, E, C, E, C, S, S, E]
    m = classification_metrics(y_true, y_pred)
    assert m["confusion"]["labels"] == TIERS
    assert m["confusion"]["counts"] == [[2, 1, 0], [0, 1, 1], [0, 1, 2]]
    assert m["confusion"]["row_normalised"][0] == pytest.approx([2 / 3, 1 / 3, 0.0])
    assert m["critical_recall"] == pytest.approx(2 / 3)
    assert m["critical_precision"] == pytest.approx(2 / 3)
    assert m["critical_f2"] == pytest.approx(2 / 3)
    assert m["accuracy"] == pytest.approx(5 / 8)
    assert m["elevated_or_above_recall"] == pytest.approx(1.0)
    assert m["elevated_or_above_precision"] == pytest.approx(5 / 6)
    assert m["elevated_or_above_f2"] == pytest.approx(25 / 26)
    assert m["per_class"]["Critical"]["support"] == 3
    assert m["per_class"]["Safe"]["precision"] == pytest.approx(1.0)
    assert m["macro_f1"] == pytest.approx(f1_score(y_true, y_pred, average="macro"))
    assert m["undefined"] == []


def test_classification_invalid_predictions_count_as_wrong_not_safe():
    y_true = [S, S, C, E]
    y_pred = [None, float("nan"), C, "Invalid"]
    m = classification_metrics(y_true, y_pred)
    assert m["confusion"]["pred_labels"] == TIERS + ["Invalid"]
    assert m["n_invalid_pred"] == 3
    counts = np.asarray(m["confusion"]["counts"])
    assert counts.shape == (3, 4)
    assert counts[0].tolist() == [0, 0, 0, 2]  # Safe truths NOT counted as Safe predictions
    assert m["accuracy"] == pytest.approx(0.25)
    assert m["per_class"]["Safe"]["recall"] == 0.0
    assert m["per_class"]["Safe"]["precision"] is None
    assert "per_class.Safe.precision" in m["undefined"]


def test_classification_no_critical_is_undefined():
    m = classification_metrics([S, E, S, E], [S, E, E, S])
    assert m["critical_recall"] is None and m["critical_precision"] is None and m["critical_f2"] is None
    assert {"critical_recall", "critical_precision", "critical_f2"} <= set(m["undefined"])
    assert m["confusion"]["row_normalised"][2] is None
    assert m["macro_f1"] is not None  # averaged over the tiers that occur
    # empty input: everything undefined, nothing crashes
    e = classification_metrics([], [])
    assert e["n"] == 0 and e["accuracy"] is None and e["macro_f1"] is None
    assert e["confusion"]["counts"] == [[0, 0, 0]] * 3


# --------------------------------------------------------------------------- #
# Probability metrics
# --------------------------------------------------------------------------- #
def test_probability_metrics_match_manual_and_sklearn(frame):
    P = frame[["p_safe", "p_elevated", "p_critical"]].to_numpy()
    y = frame["final_tier"].to_numpy()
    m = probability_metrics(P, y)
    valid = ~np.isnan(P).any(axis=1)
    assert m["n_used"] == int(valid.sum()) and m["n_excluded"] == int((~valid).sum())
    Pv, yv = P[valid], y[valid]
    onehot = np.zeros_like(Pv)
    onehot[np.arange(len(yv)), [TIERS.index(t) for t in yv]] = 1.0
    assert m["brier"] == pytest.approx(float(np.mean(((Pv - onehot) ** 2).sum(axis=1))))
    # sklearn wants lexicographic label order: Critical, Elevated, Safe
    assert m["log_loss"] == pytest.approx(log_loss(yv, Pv[:, [2, 1, 0]], labels=sorted(TIERS)), rel=1e-6)
    assert sum(m["per_class_brier"].values()) == pytest.approx(m["brier"])
    rel = m["reliability"]["Critical"]
    assert sum(rel["count"]) == m["n_used"]
    assert len(rel["mean_pred"]) == 10
    for mp, fp, n in zip(rel["mean_pred"], rel["frac_pos"], rel["count"]):
        assert (n == 0) == (mp is None) == (fp is None)
    assert 0.0 <= rel["ece"] <= 1.0
    assert 0.0 <= m["critical_pr_auc"] <= 1.0
    assert m["undefined"] == []


def test_probability_metrics_empty_and_clipping():
    m = probability_metrics(np.zeros((0, 3)), [])
    assert m["n_used"] == 0 and m["brier"] is None and "log_loss" in m["undefined"]
    P = np.array([[1.0, 0.0, 0.0]])
    m2 = probability_metrics(P, [C])  # p_true = 0 -> clipped to 1e-12
    assert m2["log_loss"] == pytest.approx(-math.log(1e-12))
    assert m2["brier"] == pytest.approx(2.0)
    with pytest.raises(ValueError):
        probability_metrics(np.zeros((2, 2)), [S, S])


# --------------------------------------------------------------------------- #
# Escalation and triage
# --------------------------------------------------------------------------- #
def test_escalation_metrics_hand_checked():
    df = pd.DataFrame({
        #                a  b  c  d  e  f  g  h  i  j
        "current_tier":  [S, S, E, S, E, S, C, C, C, C],
        "final_tier":    [C, C, C, S, E, S, C, S, C, E],
        "predicted_final_tier": [C, E, S, S, C, S, C, S, E, C],
        "triage_action": [MANEUVER, WATCH, MONITOR, WATCH, MANEUVER, AUTO, MANEUVER, HUMAN, HUMAN, MANEUVER],
    })
    m = escalation_metrics(df)
    nc, cc = m["non_critical_now"], m["critical_now"]
    assert nc["n_total"] == 6 and nc["n_end_critical"] == 3
    assert nc["n_flagged"] == 4 and nc["n_caught"] == 2
    assert nc["recall_of_escalations"] == pytest.approx(2 / 3)
    assert nc["precision_of_flags"] == pytest.approx(0.5)
    assert nc["watch_list_size"] == 2 and nc["watch_list_hits"] == 1
    assert cc["n_total"] == 4 and cc["n_deescalate"] == 2
    assert cc["n_predicted_deescalation"] == 2 and cc["n_correct_deescalation"] == 1 and cc["n_false_deescalation"] == 1
    assert m["undefined"] == []
    # without triage column: watch-list numbers undefined, flags come from the forecast only
    m2 = escalation_metrics(df.drop(columns=["triage_action"]))
    assert m2["non_critical_now"]["watch_list_size"] is None
    assert m2["non_critical_now"]["n_flagged"] == 2  # a, e
    assert "non_critical_now.watch_list_size" in m2["undefined"]
    # no Critical-now events: de-escalation ratios undefined
    m3 = escalation_metrics(df[df["current_tier"] != C])
    assert m3["critical_now"]["deescalation_recall"] is None


def test_triage_metrics_shares_sum_to_one(frame):
    t = triage_metrics(frame)
    assert t is not None
    assert sum(t["action_counts"].values()) == len(frame)
    assert sum(t["action_shares"].values()) == pytest.approx(1.0)
    assert t["auto_clear_share"] == pytest.approx(t["action_shares"].get(AUTO, 0.0))
    assert t["human_review_share"] == pytest.approx(t["action_shares"].get(HUMAN, 0.0) + t["action_shares"].get(HUMAN_INSUFF, 0.0))
    assert t["n_final_critical_auto_cleared"] == int(((frame["final_tier"] == C) & (frame["triage_action"] == AUTO)).sum())
    assert triage_metrics(frame.drop(columns=["triage_action"])) is None


# --------------------------------------------------------------------------- #
# Bootstrap and breakdowns
# --------------------------------------------------------------------------- #
def test_bootstrap_brackets_point_estimates_and_is_deterministic(frame):
    b1 = bootstrap_cis(frame, n_bootstrap=300, seed=1)
    b2 = bootstrap_cis(frame, n_bootstrap=300, seed=1)
    assert b1 == b2
    cm = classification_metrics(frame["final_tier"], frame["predicted_final_tier"])
    tr = triage_metrics(frame)
    for key, point in (("critical_recall", cm["critical_recall"]), ("critical_precision", cm["critical_precision"]),
                       ("macro_f1", cm["macro_f1"]), ("auto_clear_share", tr["auto_clear_share"])):
        assert b1[key]["lower"] <= point <= b1[key]["upper"], key
        assert b1[key]["lower"] < b1[key]["upper"]
    assert b1["undefined"] == []
    b3 = bootstrap_cis(frame, n_bootstrap=300, seed=2)
    assert b3 != b1
    empty = bootstrap_cis(frame.iloc[0:0], n_bootstrap=10)
    assert set(empty["undefined"]) == {"critical_recall", "critical_precision", "macro_f1", "auto_clear_share"}
    nocrit = bootstrap_cis(frame[frame["final_tier"] != C].assign(predicted_final_tier=S), n_bootstrap=20)
    assert nocrit["critical_recall"]["lower"] is None and "critical_recall" in nocrit["undefined"]


def test_breakdown_merges_small_groups(frame):
    bd = breakdown_by(frame, "object_type", min_rows=30)
    assert sum(s["n"] for s in bd.values()) == len(frame)
    for name, s in bd.items():
        if name != "other":
            assert s["n"] >= 30
    assert "other" in bd and set(bd["other"]["merged_groups"]) >= {"TBA"}
    assert "TBA" not in bd
    assert breakdown_by(frame, "no_such_column") == {}
    # missing values become their own group
    df = frame.copy()
    df["object_type"] = df["object_type"].astype(object)
    df.loc[df.index[:40], "object_type"] = None
    assert "missing" in breakdown_by(df, "object_type", min_rows=20)


def test_lead_time_buckets_partition_rows(frame):
    df = frame.copy()
    df.loc[df.index[:3], "time_to_tca_days"] = [1.5, np.nan, 2.0]
    lt = lead_time_breakdown(df)
    assert sum(s["n"] for s in lt.values()) == len(df)
    assert list(lt.keys())[:4] == ["[2,3)", "[3,4)", "[4,5)", "[5,inf)"]
    assert lt["<2_or_missing"]["n"] == 2
    assert lt["[2,3)"]["n"] == int(((df["time_to_tca_days"] >= 2) & (df["time_to_tca_days"] < 3)).sum())
    assert lead_time_breakdown(df.drop(columns=["time_to_tca_days"])) == {}


# --------------------------------------------------------------------------- #
# Full report
# --------------------------------------------------------------------------- #
def test_evaluate_forecasts_is_json_safe_with_three_subsets(frame, cfg):
    res = evaluate_forecasts(frame, cfg, n_bootstrap=50, seed=0, scope={"model": "test", "n": np.int64(3)})
    json.dumps(res, allow_nan=False)
    assert set(res["subsets"]) == {"all_usable", "trivial", "real_forecast"}
    assert res["n_events"] == len(frame)
    assert res["n_trivial"] + res["n_real_forecast"] == res["n_events"]
    assert res["subsets"]["trivial"]["n_events"] == res["n_trivial"]
    assert res["subsets"]["real_forecast"]["n_events"] == res["n_real_forecast"]
    assert sum(res["class_counts"].values()) == res["n_events"]
    assert res["scope"] == {"model": "test", "n": 3}
    rf = res["subsets"]["real_forecast"]
    for key in ("classification", "probability", "escalation", "triage", "bootstrap", "prediction_sets"):
        assert rf[key] is not None, key
    assert rf["classification"]["critical_recall"] is not None
    assert rf["prediction_sets"]["coverage_empirical"] is not None
    assert set(res["by_object_type"]) == {"real_forecast", "all_usable"}
    assert set(res["by_lead_time"]) == {"real_forecast", "all_usable"}
    assert sum(s["n"] for s in res["by_lead_time"]["real_forecast"].values()) == res["n_real_forecast"]
    # the Invalid predictions show up as the extra confusion column, never as Safe
    assert "Invalid" in res["subsets"]["all_usable"]["classification"]["confusion"]["pred_labels"]


def test_evaluate_forecasts_undefined_when_subset_has_no_critical(frame, cfg):
    df = frame.copy()
    df.loc[df["is_trivial"], "final_tier"] = S
    df.loc[df["is_trivial"], "predicted_final_tier"] = S
    res = evaluate_forecasts(df, cfg, n_bootstrap=20)
    json.dumps(res, allow_nan=False)
    tri = res["subsets"]["trivial"]["classification"]
    assert tri["critical_recall"] is None and "critical_recall" in tri["undefined"]
    assert tri["confusion"]["row_normalised"][2] is None
    assert res["subsets"]["trivial"]["probability"]["critical_pr_auc"] is None
    assert "critical_recall" in res["subsets"]["trivial"]["bootstrap"]["undefined"]
    # frame without is_trivial: everything is a real forecast, with a note
    res2 = evaluate_forecasts(frame.drop(columns=["is_trivial"]), cfg, n_bootstrap=5)
    assert res2["n_trivial"] == 0 and res2["subsets"]["trivial"]["n_events"] == 0 and res2["notes"]


def test_compare_models_one_row_per_model(frame, cfg):
    base = frame.copy()
    base["predicted_final_tier"] = base["current_tier"]
    results = {
        "carry_forward_risk": evaluate_forecasts(base, cfg, n_bootstrap=5),
        "lgbm_multiclass": evaluate_forecasts(frame, cfg, n_bootstrap=5),
        "broken": {},
    }
    table = compare_models(results, subset="real_forecast")
    assert table.index.tolist() == ["carry_forward_risk", "lgbm_multiclass", "broken"]
    for col in ("critical_recall", "critical_precision", "critical_f2", "elevated_or_above_f2", "macro_f1", "brier",
                "log_loss", "auto_clear_share", "n_final_critical_auto_cleared", "watch_share", "escalation_recall"):
        assert col in table.columns
    assert table.loc["broken"].isna().all() or all(v is None for v in table.loc["broken"].tolist())
    assert table.loc["lgbm_multiclass", "critical_recall"] == results["lgbm_multiclass"]["subsets"]["real_forecast"]["classification"]["critical_recall"]


def test_find_illustrative_cases_returns_missed_critical(frame):
    df = frame.copy()
    df["predicted_final_tier"] = df["predicted_final_tier"].astype(object)
    df.loc[df.index[0], ["final_tier", "predicted_final_tier", "current_tier", "triage_action", "p_critical"]] = [C, S, S, AUTO, 0.01]
    cases = find_illustrative_cases(df, k=3)
    assert set(cases) == {"missed_critical", "caught_by_watch", "false_alarm_critical", "false_deescalation", "uncertain"}
    missed = cases["missed_critical"]
    assert 1 <= len(missed) <= 3
    assert missed[0]["final_tier"] == C and missed[0]["predicted_final_tier"] != C
    assert missed[0]["triage_action"] != WATCH
    assert missed[0]["event_id"] == 0  # lowest p_critical first
    for rows in cases.values():
        json.dumps(rows, allow_nan=False)
        assert len(rows) <= 3
    for r in cases["caught_by_watch"]:
        assert r["triage_action"] == WATCH and r["final_tier"] == C
    for r in cases["false_deescalation"]:
        assert r["current_tier"] == C and r["final_tier"] == C and r["predicted_final_tier"] != C
    assert cases["uncertain"] and len(cases["uncertain"][0]["prediction_set"]) >= 1


def test_to_jsonable_handles_numpy_and_nan():
    out = to_jsonable({"a": np.float64("nan"), "b": np.int32(2), "c": [np.bool_(True), pd.NA], "d": np.array([1.0, np.inf])})
    assert out == {"a": None, "b": 2, "c": [True, None], "d": [1.0, None]}


# --------------------------------------------------------------------------- #
# RESULTS.md renderer
# --------------------------------------------------------------------------- #
def _load_make_results():
    spec = importlib.util.spec_from_file_location("make_results", ROOT / "scripts" / "make_results.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_make_results_main_writes_report(tmp_path, frame, cfg):
    base = frame.copy()
    base["predicted_final_tier"] = base["current_tier"]
    cf = evaluate_forecasts(base, cfg, n_bootstrap=5)
    pm = evaluate_forecasts(frame, cfg, n_bootstrap=5)
    ev = {
        "generated_at": "2026-10-31T00:00:00",
        "primary_model": "lgbm_multiclass",
        "model_version": "test",
        "split": {"seed": 42, "holdout_fraction": 0.2, "split_hash": "abc", "counts": {"dev": {}, "holdout": {}}},
        "label_audit": {"n_events_total": 10, "n_usable": 9, "n_unusable": 1, "n_trivial": 2, "n_real_forecast": 7,
                        "class_counts_usable": {"Safe": 6, "Elevated": 2, "Critical": 1}, "labels_match_recomputed": True},
        "decision_thresholds": {"t_crit": 0.3, "t_elev": 0.4, "t_watch": 0.15, "t_clear": 0.9},
        "models": {"carry_forward_risk": cf, "lgbm_multiclass": pm},
        "ablations": {"use_past_risk_off": pm},
        "mission_heldout_cv": pm,
        "conformal_sets": None,
        "illustrative_cases": find_illustrative_cases(frame),
        "notes": ["synthetic"],
        "total_train_seconds": 1.5,
    }
    (tmp_path / "evaluation.json").write_text(json.dumps(ev, allow_nan=False), encoding="utf-8")
    (tmp_path / "audit.json").write_text(json.dumps({"files": {"train": {"n_rows": 5}}}), encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps({"tier_counts": {"Safe": 100, "Elevated": 40, "Critical": 20}}), encoding="utf-8")
    mod = _load_make_results()
    out = mod.main(tmp_path, tmp_path / "RESULTS.md")
    text = Path(out).read_text(encoding="utf-8")
    assert "## Headline: real-forecast holdout" in text
    assert "| Model | Events | Critical recall | Critical precision | Critical F2 |" in text
    assert "carry_forward_risk" in text and "**lgbm_multiclass**" in text
    assert "enriched with high-risk events" in text
    assert "Row-normalised" in text
    assert "## Escalation watch" in text
    # tolerant to an almost-empty evaluation.json
    (tmp_path / "evaluation.json").write_text("{}", encoding="utf-8")
    out2 = mod.main(tmp_path, tmp_path / "RESULTS2.md")
    assert "undefined" in Path(out2).read_text(encoding="utf-8")
