"""Tests for collision_risk.evaluate."""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from collision_risk.config import load_config
from collision_risk.evaluate import (
    CONFUSION_LABELS,
    compare_models,
    evaluate_predictions,
    find_illustrative_cases,
    latest_cdm_view,
    tier_metrics,
    to_jsonable,
)

FLOOR = -30.0
CRIT = -4.0
ELEV = -6.0
AUTO = "Auto-clear candidate"
HUMAN = "Human review"
HUMAN_INSUFF = "Human review - insufficient data"
MANEUVER = "Maneuver planning review"
MONITOR = "Monitor"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config()


def _tier_from_rule(risk: float, miss: float) -> str:
    if risk >= CRIT or miss < 1000.0:
        return "Critical"
    if risk >= ELEV or miss <= 5000.0:
        return "Elevated"
    return "Safe"


def make_frame(n_events: int = 200, seed: int = 123) -> pd.DataFrame:
    """Synthetic prediction frame: ~10 rows per event with distinct time_to_tca,
    a floor mass at -30, a handful of true Critical events, NaN intervals on some
    rows, a few invalid rows, and a predictor that is noisy but informative."""
    rng = np.random.default_rng(seed)
    rows = []
    for ev in range(n_events):
        n_cdm = int(rng.integers(6, 14))
        ttca = np.sort(rng.uniform(0.1, 7.0, size=n_cdm))[::-1]  # decreasing towards TCA
        kind = rng.choice(["floor", "safe", "elevated", "critical"], p=[0.41, 0.40, 0.15, 0.04])
        if kind == "floor":
            base_true = FLOOR
        elif kind == "safe":
            base_true = rng.uniform(-12.0, -6.5)
        elif kind == "elevated":
            base_true = rng.uniform(-5.9, -4.3)
        else:
            base_true = rng.uniform(-3.9, -2.0)
        for j, t in enumerate(ttca):
            if base_true == FLOOR:
                risk_true = FLOOR
                risk_pred = rng.choice([FLOOR, rng.uniform(-14.0, -9.0)], p=[0.7, 0.3])
            else:
                risk_true = base_true + rng.normal(0, 0.15)
                risk_pred = risk_true + rng.normal(0, 0.6)
            miss = float(rng.uniform(6000.0, 60000.0))
            has_int = rng.random() > 0.15
            lower = risk_pred - rng.uniform(0.5, 2.5) if has_int else np.nan
            upper = risk_pred + rng.uniform(0.5, 2.5) if has_int else np.nan
            rows.append(
                {
                    "event_id": ev,
                    "time_to_tca_days": float(t),
                    "miss_distance_m": miss,
                    "risk_true": float(risk_true),
                    "risk_log10_pred": float(risk_pred),
                    "risk_log10_lower": lower,
                    "risk_log10_upper": upper,
                    "prediction_status": "valid",
                    "p_pc_critical": float(np.clip(1.0 / (1.0 + np.exp(-(risk_pred + 4.0) * 2.0)) + rng.normal(0, 0.05), 0, 1)),
                }
            )
    df = pd.DataFrame(rows)

    # A few hard-coded geometry cases: distance rule drives Critical on some rows.
    df.loc[df.index[:5], "miss_distance_m"] = 500.0
    # A few invalid rows
    inv = df.index[10:15]
    df.loc[inv, "prediction_status"] = "invalid"

    df["risk_tier_true"] = [
        _tier_from_rule(r, m) for r, m in zip(df["risk_true"], df["miss_distance_m"])
    ]
    df["risk_tier"] = [
        None if s == "invalid" else _tier_from_rule(r, m)
        for r, m, s in zip(df["risk_log10_pred"], df["miss_distance_m"], df["prediction_status"])
    ]
    df.loc[inv, "risk_log10_pred"] = np.nan
    df.loc[inv, ["risk_log10_lower", "risk_log10_upper"]] = np.nan

    def action(row):
        if row["prediction_status"] == "invalid":
            return HUMAN_INSUFF
        if row["risk_tier"] == "Critical":
            return MANEUVER
        if not math.isnan(row["risk_log10_upper"]) and row["risk_log10_upper"] >= CRIT:
            return HUMAN
        if row["risk_tier"] == "Elevated":
            return MONITOR
        if math.isnan(row["risk_log10_upper"]) or row["risk_log10_upper"] >= ELEV:
            return MONITOR
        return AUTO

    df["triage_action"] = df.apply(action, axis=1)
    return df


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return make_frame()


@pytest.fixture(scope="module")
def result(frame, cfg) -> dict:
    return evaluate_predictions(frame, cfg, n_bootstrap=100, seed=1, scope={"model": "test"})


# --------------------------------------------------------------------------- #
# latest_cdm_view
# --------------------------------------------------------------------------- #
def test_frame_shape(frame):
    assert 1500 <= len(frame) <= 2800
    assert frame["event_id"].nunique() == 200


def test_latest_cdm_view_picks_min_ttca_per_event(frame):
    view = latest_cdm_view(frame)
    assert len(view) == frame["event_id"].nunique()
    assert view["event_id"].is_unique
    assert list(view.columns) == list(frame.columns)
    expected = frame.groupby("event_id")["time_to_tca_days"].min()
    got = view.set_index("event_id")["time_to_tca_days"]
    pd.testing.assert_series_equal(got.sort_index(), expected.sort_index(), check_names=False)


def test_latest_cdm_view_tie_breaks_last_occurrence():
    df = pd.DataFrame(
        {
            "event_id": [1, 1, 1, 2, 2],
            "time_to_tca_days": [2.0, 1.0, 1.0, 3.0, np.nan],
            "tag": ["a", "b", "c", "d", "e"],
        }
    )
    view = latest_cdm_view(df)
    assert view.set_index("event_id")["tag"].to_dict() == {1: "c", 2: "d"}


def test_latest_cdm_view_empty():
    df = pd.DataFrame({"event_id": [], "time_to_tca_days": []})
    assert len(latest_cdm_view(df)) == 0


# --------------------------------------------------------------------------- #
# tier_metrics
# --------------------------------------------------------------------------- #
def test_tier_metrics_hand_computed():
    y_true = ["Critical", "Critical", "Elevated", "Safe", "Safe", None]
    y_pred = ["Critical", "Elevated", "Elevated", "Safe", "Critical", np.nan]
    m = tier_metrics(y_true, y_pred)

    assert m["critical_recall"] == pytest.approx(0.5)
    assert m["critical_precision"] == pytest.approx(0.5)
    assert m["critical_f2"] == pytest.approx(0.5)  # 5*.25/(4*.5+.5)
    assert m["elevated_or_above_recall"] == pytest.approx(1.0)
    assert m["elevated_or_above_precision"] == pytest.approx(0.75)
    assert m["elevated_or_above_f2"] == pytest.approx(0.9375)  # 5*.75/(3+1)
    # per-class f1: Critical .5, Elevated 2/3, Safe 2/3
    assert m["macro_f1"] == pytest.approx((0.5 + 2 / 3 + 2 / 3) / 3)
    assert m["confusion"]["labels"] == CONFUSION_LABELS == ["Critical", "Elevated", "Safe", "Invalid"]
    assert m["confusion"]["matrix"] == [
        [1, 1, 0, 0],
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 0, 0, 1],
    ]
    assert m["undefined"] == []


def test_tier_metrics_undefined_when_no_true_critical():
    y_true = ["Safe", "Elevated", "Safe"]
    y_pred = ["Safe", "Elevated", "Safe"]
    m = tier_metrics(y_true, y_pred)
    assert m["critical_recall"] is None
    assert m["critical_precision"] is None
    assert m["critical_f2"] is None
    assert set(m["undefined"]) == {"critical_recall", "critical_precision", "critical_f2"}
    assert m["elevated_or_above_recall"] == 1.0


def test_evaluate_all_none_with_undefined_list_when_no_critical(cfg):
    df = make_frame(n_events=40, seed=7)
    df = df[df["risk_tier_true"] != "Critical"].copy()
    df["risk_tier"] = df["risk_tier"].where(df["risk_tier"] != "Critical", "Elevated")
    df["risk_log10_pred"] = df["risk_log10_pred"].clip(upper=-4.5)
    res = evaluate_predictions(df, cfg, n_bootstrap=20)
    for view in ("row_level", "latest_cdm", "horizon_slice", "horizon_latest_cdm"):
        t = res[view]["tier"]
        assert res[view]["n_critical_true"] == 0
        assert t["critical_recall"] is None and t["critical_precision"] is None and t["critical_f2"] is None
        assert "critical_recall" in t["undefined"]
        assert res[view]["pc_only"]["critical_recall"] is None
        assert "critical_recall" in res[view]["pc_only"]["undefined"]
        assert res[view]["triage"]["true_critical_routed_to_auto_clear_fraction"] is None
        assert "true_critical_routed_to_auto_clear_fraction" in res[view]["triage"]["undefined"]
        assert res[view]["bootstrap"]["critical_recall"] == {"lower": None, "upper": None}
        assert "critical_recall" in res[view]["bootstrap"]["undefined"]
    json.dumps(res)


# --------------------------------------------------------------------------- #
# pc_only vs tier
# --------------------------------------------------------------------------- #
def test_pc_only_differs_when_distance_rule_drives_critical(cfg):
    # Distance rule makes rows Critical in both true and predicted tier even
    # though risk is far below -4 -> tier metrics are perfect, pc_only undefined.
    df = pd.DataFrame(
        {
            "event_id": [1, 1, 2, 2, 3, 3],
            "time_to_tca_days": [3.0, 1.0, 3.0, 1.0, 3.0, 1.0],
            "miss_distance_m": [500.0, 500.0, 20000.0, 20000.0, 500.0, 20000.0],
            "risk_true": [-9.0, -9.0, -10.0, -10.0, -8.0, -3.5],
            "risk_log10_pred": [-9.5, -8.8, -10.2, -9.9, -8.5, -5.0],
            "risk_tier_true": ["Critical", "Critical", "Safe", "Safe", "Critical", "Critical"],
            "risk_tier": ["Critical", "Critical", "Safe", "Safe", "Critical", "Elevated"],
            "risk_log10_lower": [np.nan] * 6,
            "risk_log10_upper": [np.nan] * 6,
            "triage_action": [MANEUVER, MANEUVER, AUTO, AUTO, MANEUVER, MONITOR],
            "prediction_status": ["valid"] * 6,
        }
    )
    res = evaluate_predictions(df, cfg, n_bootstrap=10)
    rl = res["row_level"]
    assert rl["tier"]["critical_recall"] == pytest.approx(3 / 4)
    assert rl["tier"]["critical_precision"] == pytest.approx(1.0)
    # pc_only: only row 5 is truly >= -4; predicted -5.0 is not -> recall 0, no predicted positives
    assert rl["pc_only"]["critical_recall"] == 0.0
    assert rl["pc_only"]["critical_precision"] is None
    assert rl["pc_only"]["critical_f2"] == 0.0
    assert rl["pc_only"]["n_true_positive_rows"] == 0
    assert rl["pc_only"]["critical_recall"] != rl["tier"]["critical_recall"]
    assert "critical_precision" in rl["pc_only"]["undefined"]


# --------------------------------------------------------------------------- #
# Regression / interval / triage metrics
# --------------------------------------------------------------------------- #
def test_regression_metrics_match_numpy(frame, result):
    rl = result["row_level"]["regression"]
    valid = frame["risk_true"].notna() & frame["risk_log10_pred"].notna()
    err = (frame.loc[valid, "risk_log10_pred"] - frame.loc[valid, "risk_true"]).to_numpy()
    assert rl["mae"] == pytest.approx(np.mean(np.abs(err)))
    assert rl["rmse"] == pytest.approx(np.sqrt(np.mean(err ** 2)))

    rt = frame.loc[valid, "risk_true"].to_numpy()
    nf = rt > FLOOR
    hr = rt >= ELEV
    assert rl["n_non_floor"] == int(nf.sum())
    assert rl["n_high_risk"] == int(hr.sum())
    assert rl["mae_non_floor"] == pytest.approx(np.mean(np.abs(err[nf])))
    assert rl["rmse_non_floor"] == pytest.approx(np.sqrt(np.mean(err[nf] ** 2)))
    assert rl["mae_high_risk"] == pytest.approx(np.mean(np.abs(err[hr])))
    assert rl["rmse_high_risk"] == pytest.approx(np.sqrt(np.mean(err[hr] ** 2)))
    assert rl["undefined"] == []
    # floor rows exist and are excluded from non-floor
    assert result["counts"]["n_floor_rows"] > 0
    assert rl["n_non_floor"] < result["counts"]["n_rows"]


def test_regression_subsets_hand_example(cfg):
    df = pd.DataFrame(
        {
            "event_id": [1, 2, 3, 4],
            "time_to_tca_days": [1.0, 1.0, 1.0, 1.0],
            "miss_distance_m": [1e4] * 4,
            "risk_true": [-30.0, -8.0, -5.0, -3.0],
            "risk_log10_pred": [-29.0, -10.0, -5.5, -3.5],
            "risk_tier_true": ["Safe", "Safe", "Elevated", "Critical"],
            "risk_tier": ["Safe", "Safe", "Elevated", "Critical"],
            "risk_log10_lower": [np.nan] * 4,
            "risk_log10_upper": [np.nan] * 4,
            "triage_action": [AUTO, AUTO, MONITOR, MANEUVER],
            "prediction_status": ["valid"] * 4,
        }
    )
    r = evaluate_predictions(df, cfg, n_bootstrap=5)["row_level"]["regression"]
    assert r["mae"] == pytest.approx((1 + 2 + 0.5 + 0.5) / 4)
    assert r["n_non_floor"] == 3
    assert r["mae_non_floor"] == pytest.approx((2 + 0.5 + 0.5) / 3)
    assert r["n_high_risk"] == 2
    assert r["mae_high_risk"] == pytest.approx(0.5)
    assert r["rmse_high_risk"] == pytest.approx(0.5)


def test_interval_coverage_excludes_nan(cfg):
    df = pd.DataFrame(
        {
            "event_id": [1, 2, 3, 4, 5],
            "time_to_tca_days": [1.0] * 5,
            "miss_distance_m": [1e4] * 5,
            "risk_true": [-8.0, -8.0, -8.0, -8.0, -8.0],
            "risk_log10_pred": [-8.0] * 5,
            "risk_tier_true": ["Safe"] * 5,
            "risk_tier": ["Safe"] * 5,
            "risk_log10_lower": [-9.0, -9.0, -7.5, np.nan, -9.0],
            "risk_log10_upper": [-7.0, -7.0, -6.0, -7.0, np.nan],
            "triage_action": [AUTO] * 5,
            "prediction_status": ["valid"] * 5,
        }
    )
    res = evaluate_predictions(df, cfg, n_bootstrap=5)
    iv = res["row_level"]["interval"]
    assert iv["nominal_coverage"] == pytest.approx(cfg["conformal"]["coverage"])
    assert iv["n_with_interval"] == 3
    assert iv["fraction_with_interval"] == pytest.approx(3 / 5)
    assert iv["coverage"] == pytest.approx(2 / 3)
    assert iv["median_width"] == pytest.approx(2.0)  # widths 2, 2, 1.5


def test_interval_coverage_matches_manual(frame, result):
    iv = result["row_level"]["interval"]
    has = frame["risk_log10_lower"].notna() & frame["risk_log10_upper"].notna()
    cov = (frame.loc[has, "risk_log10_lower"] <= frame.loc[has, "risk_true"]) & (
        frame.loc[has, "risk_true"] <= frame.loc[has, "risk_log10_upper"]
    )
    assert iv["n_with_interval"] == int(has.sum())
    assert iv["fraction_with_interval"] == pytest.approx(has.mean())
    assert iv["coverage"] == pytest.approx(cov.mean())
    assert 0 < iv["fraction_with_interval"] < 1


def test_triage_metrics(cfg):
    df = pd.DataFrame(
        {
            "event_id": [1, 2, 3, 4, 5, 6],
            "time_to_tca_days": [1.0] * 6,
            "miss_distance_m": [1e4] * 6,
            "risk_true": [-3.0, -3.5, -8.0, -8.0, -5.0, -8.0],
            "risk_log10_pred": [-3.0, -9.0, -8.0, -8.0, -5.0, -8.0],
            "risk_tier_true": ["Critical", "Critical", "Safe", "Safe", "Elevated", None],
            "risk_tier": ["Critical", "Safe", "Safe", "Safe", "Elevated", None],
            "risk_log10_lower": [np.nan] * 6,
            "risk_log10_upper": [np.nan] * 6,
            "triage_action": [MANEUVER, AUTO, AUTO, AUTO, HUMAN, HUMAN_INSUFF],
            "prediction_status": ["valid"] * 5 + ["invalid"],
        }
    )
    tr = evaluate_predictions(df, cfg, n_bootstrap=5)["row_level"]["triage"]
    assert tr["action_counts"] == {AUTO: 3, HUMAN: 1, HUMAN_INSUFF: 1, MANEUVER: 1}
    assert tr["auto_clear_fraction"] == pytest.approx(0.5)
    assert tr["true_critical_routed_to_auto_clear_n"] == 1
    assert tr["true_critical_routed_to_auto_clear_fraction"] == pytest.approx(0.5)
    assert tr["maneuver_review_fraction"] == pytest.approx(1 / 6)
    assert tr["human_review_fraction"] == pytest.approx(2 / 6)


def test_triage_on_synthetic_frame(frame, result):
    tr = result["row_level"]["triage"]
    n_auto = int((frame["triage_action"] == AUTO).sum())
    assert tr["action_counts"][AUTO] == n_auto
    assert tr["auto_clear_fraction"] == pytest.approx(n_auto / len(frame))
    n_crit_auto = int(((frame["triage_action"] == AUTO) & (frame["risk_tier_true"] == "Critical")).sum())
    assert tr["true_critical_routed_to_auto_clear_n"] == n_crit_auto
    assert sum(tr["action_counts"].values()) == len(frame)


# --------------------------------------------------------------------------- #
# Full report structure, counts, classifier, bootstrap, JSON
# --------------------------------------------------------------------------- #
def test_report_structure_and_counts(frame, result, cfg):
    assert result["scope"] == {"model": "test"}
    assert set(result) == {"scope", "counts", "row_level", "latest_cdm", "horizon_slice", "horizon_latest_cdm"}
    c = result["counts"]
    assert c["n_rows"] == len(frame)
    assert c["n_events"] == 200
    assert c["n_critical_rows"] == int((frame["risk_tier_true"] == "Critical").sum())
    assert c["n_critical_events"] == frame.loc[frame["risk_tier_true"] == "Critical", "event_id"].nunique()
    assert c["n_elevated_rows"] == int((frame["risk_tier_true"] == "Elevated").sum())
    assert c["n_floor_rows"] == int((frame["risk_true"] <= FLOOR).sum())
    assert c["n_invalid_rows"] == 5

    for view in ("row_level", "latest_cdm", "horizon_slice", "horizon_latest_cdm"):
        v = result[view]
        assert set(v) == {
            "n_rows", "n_events", "n_critical_true", "tier", "pc_only", "regression",
            "interval", "triage", "classifier", "bootstrap",
        }
        for group in ("tier", "pc_only", "regression", "interval", "triage", "classifier", "bootstrap"):
            assert "undefined" in v[group]

    assert result["latest_cdm"]["n_rows"] == 200
    assert result["latest_cdm"]["n_events"] == 200
    hmin = cfg["split"]["test_horizon_min_days"]
    hz = frame[frame["time_to_tca_days"] >= hmin]
    assert result["horizon_slice"]["n_rows"] == len(hz)
    assert result["horizon_latest_cdm"]["n_rows"] == hz["event_id"].nunique()
    assert result["horizon_slice"]["n_rows"] < result["row_level"]["n_rows"]


def test_classifier_metrics(frame, result):
    cl = result["row_level"]["classifier"]
    assert cl is not None
    from sklearn.metrics import average_precision_score, brier_score_loss

    ok = frame["p_pc_critical"].notna() & frame["risk_true"].notna()
    y = (frame.loc[ok, "risk_true"] >= CRIT).astype(int)
    p = frame.loc[ok, "p_pc_critical"]
    assert cl["n_pos"] == int(y.sum()) > 0
    assert cl["pr_auc"] == pytest.approx(average_precision_score(y, p))
    assert cl["brier"] == pytest.approx(brier_score_loss(y, p))


def test_classifier_none_when_column_absent(frame, cfg):
    res = evaluate_predictions(frame.drop(columns=["p_pc_critical"]), cfg, n_bootstrap=5)
    assert res["row_level"]["classifier"] is None


def test_bootstrap_brackets_point_estimates(result):
    for view in ("row_level", "latest_cdm"):
        v = result[view]
        bs = v["bootstrap"]
        point = {
            "critical_recall": v["tier"]["critical_recall"],
            "critical_precision": v["tier"]["critical_precision"],
            "mae": v["regression"]["mae"],
            "coverage": v["interval"]["coverage"],
        }
        for name, pt in point.items():
            assert pt is not None
            lo, hi = bs[name]["lower"], bs[name]["upper"]
            assert lo is not None and hi is not None
            assert lo <= pt <= hi, (view, name, lo, pt, hi)
            assert lo <= hi
        # continuous metrics must have a strictly positive width
        assert bs["mae"]["lower"] < bs["mae"]["upper"]
        assert bs["coverage"]["lower"] < bs["coverage"]["upper"]
        assert bs["undefined"] == []


def test_bootstrap_is_deterministic(frame, cfg):
    a = evaluate_predictions(frame, cfg, n_bootstrap=30, seed=5)["row_level"]["bootstrap"]
    b = evaluate_predictions(frame, cfg, n_bootstrap=30, seed=5)["row_level"]["bootstrap"]
    assert a == b


def test_result_is_json_serialisable(result):
    text = json.dumps(result, allow_nan=False)
    assert "NaN" not in text

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        else:
            assert o is None or isinstance(o, (bool, int, float, str)), type(o)
            assert not isinstance(o, (np.generic,))
            if isinstance(o, float):
                assert not math.isnan(o)

    walk(result)


def test_to_jsonable():
    out = to_jsonable(
        {
            "a": np.int64(3),
            "b": np.float32(1.5),
            "c": np.nan,
            "d": [np.bool_(True), np.array([1.0, np.nan])],
            "e": pd.NA,
            "f": ("x", np.str_("y")),
            7: None,
        }
    )
    assert out == {"a": 3, "b": 1.5, "c": None, "d": [True, [1.0, None]], "e": None, "f": ["x", "y"], "7": None}
    assert type(out["a"]) is int and type(out["b"]) is float and type(out["d"][0]) is bool
    json.dumps(out, allow_nan=False)


# --------------------------------------------------------------------------- #
# compare_models / find_illustrative_cases
# --------------------------------------------------------------------------- #
def test_compare_models(result, frame, cfg):
    other = evaluate_predictions(frame.drop(columns=["p_pc_critical"]), cfg, n_bootstrap=5)
    table = compare_models({"champion": result, "challenger": other, "broken": {}})
    assert list(table.index) == ["champion", "challenger", "broken"]
    assert len(table) == 3
    expected_cols = [
        "critical_recall", "critical_precision", "critical_f2", "pc_only_critical_recall",
        "pc_only_critical_precision", "macro_f1", "mae", "mae_non_floor", "mae_high_risk",
        "coverage", "median_width", "auto_clear_fraction", "true_critical_routed_to_auto_clear_n",
    ]
    assert list(table.columns) == expected_cols
    assert table.loc["champion", "mae"] == pytest.approx(result["latest_cdm"]["regression"]["mae"])
    assert table.loc["champion", "critical_recall"] == pytest.approx(result["latest_cdm"]["tier"]["critical_recall"])
    assert all(v is None for v in table.loc["broken"].tolist())
    # other view
    t2 = compare_models({"champion": result}, view="row_level")
    assert t2.loc["champion", "mae"] == pytest.approx(result["row_level"]["regression"]["mae"])


def test_find_illustrative_cases(frame):
    df = frame.copy()
    # Guarantee at least one missed critical and one false alarm exist.
    # pick a risk-driven Critical (risk_true >= -4) so a -12 prediction is the
    # largest under-prediction in the frame and therefore sorts first
    crit_idx = df.index[(df["risk_tier_true"] == "Critical") & (df["risk_true"] >= CRIT)][0]
    df.loc[crit_idx, "risk_tier"] = "Safe"
    df.loc[crit_idx, "risk_log10_pred"] = -12.0
    safe_idx = df.index[df["risk_tier_true"] == "Safe"][0]
    df.loc[safe_idx, "risk_tier"] = "Critical"

    cases = find_illustrative_cases(df, k=3)
    assert set(cases) == {"missed_critical", "false_alarm_critical", "uncertain", "invalid"}
    assert 1 <= len(cases["missed_critical"]) <= 3
    assert all(r["risk_tier_true"] == "Critical" and r["risk_tier"] != "Critical" for r in cases["missed_critical"])
    # worst (most under-predicted) first
    assert cases["missed_critical"][0]["event_id"] == int(df.loc[crit_idx, "event_id"])
    assert 1 <= len(cases["false_alarm_critical"]) <= 3
    assert all(r["risk_tier"] == "Critical" and r["risk_tier_true"] != "Critical" for r in cases["false_alarm_critical"])
    assert 1 <= len(cases["invalid"]) <= 3
    assert all(r["prediction_status"] == "invalid" for r in cases["invalid"])
    assert all(r["triage_action"] == HUMAN for r in cases["uncertain"])
    if len(cases["uncertain"]) > 1:
        widths = [r["risk_log10_upper"] - r["risk_log10_lower"] for r in cases["uncertain"]]
        assert widths == sorted(widths, reverse=True)
    # invalid rows have NaN pred -> must be None, and the whole thing serialises
    assert all(r["risk_log10_pred"] is None for r in cases["invalid"])
    json.dumps(cases, allow_nan=False)


def test_find_illustrative_cases_empty_categories(cfg):
    df = pd.DataFrame(
        {
            "event_id": [1],
            "time_to_tca_days": [1.0],
            "miss_distance_m": [1e4],
            "risk_true": [-8.0],
            "risk_log10_pred": [-8.0],
            "risk_tier_true": ["Safe"],
            "risk_tier": ["Safe"],
            "risk_log10_lower": [-9.0],
            "risk_log10_upper": [-7.0],
            "triage_action": [AUTO],
            "prediction_status": ["valid"],
        }
    )
    cases = find_illustrative_cases(df)
    assert cases == {"missed_critical": [], "false_alarm_critical": [], "uncertain": [], "invalid": []}



def test_latest_cdm_scopes_reused_ids_by_dataset():
    rows = pd.DataFrame({
        "dataset_id": ["a", "b", "a", "b", "a", "b"],
        "event_id": [1, 1, 1, 1, 1, 1],
        "time_to_tca_days": [4.0, 3.0, 2.0, 1.0, 2.0, np.nan],
        "tag": list("abcdef"),
    }, index=[10, 20, 30, 40, 50, 60])
    latest = latest_cdm_view(rows)
    assert latest["tag"].tolist() == ["d", "e"]
    assert latest.index.tolist() == [40, 50]
    assert latest.columns.tolist() == rows.columns.tolist()


def test_mixed_dataset_evaluation_matches_globally_unique_ids(cfg):
    a = make_frame(n_events=20, seed=17).assign(dataset_id="a")
    b = make_frame(n_events=20, seed=23).assign(dataset_id="b")
    mixed = pd.concat([a, b], ignore_index=True)
    remapped = mixed.copy()
    remapped.loc[remapped["dataset_id"] == "b", "event_id"] += 20
    remapped = remapped.drop(columns="dataset_id")
    actual = evaluate_predictions(mixed, cfg, n_bootstrap=40, seed=5)
    expected = evaluate_predictions(remapped, cfg, n_bootstrap=40, seed=5)
    assert actual == expected
    assert actual["counts"]["n_events"] == 40
    assert actual["latest_cdm"]["n_rows"] == 40
