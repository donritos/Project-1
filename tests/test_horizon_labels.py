"""Required checks: horizon leakage, labels, columns, snapshot features."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from collision_risk import features as FE
from collision_risk.data import (apply_horizon, cross_check_labels, derive_event_labels, enforce_allowlist,
                                 load_event_labels, parse_usable_flag, parse_workbook_labels)
from collision_risk.policy import tier_from_true_risk

SNAP_COLS = FE.EVENT_SO_FAR_BASE + FE.EVENT_SO_FAR_RISK + FE.HISTORY + FE.PHYSICS


def test_apply_horizon_keeps_only_2_day_plus_rows(synthetic_allowlisted):
    df, _ = synthetic_allowlisted
    h = apply_horizon(df, 2.0)
    assert (h["time_to_tca"] >= 2.0).all()
    assert len(h) < len(df)
    assert set(h["row_id"]) <= set(df["row_id"])


def test_snapshot_features_identical_with_and_without_sub_horizon_rows(synthetic_allowlisted, cfg):
    """Rows under 2 days must never influence any snapshot feature."""
    df, _ = synthetic_allowlisted
    with_late = df.copy()
    h = apply_horizon(df, 2.0)
    a = FE.build_event_snapshots(h, cfg)
    # add the sub-horizon rows back with absurd values; the horizon filter must remove them before features
    late = with_late[with_late["time_to_tca"] < 2.0].copy()
    late["miss_distance"] = 1.0; late["max_risk_estimate"] = -1.0
    if "risk" in late.columns:
        late["risk"] = -1.0
    b = FE.build_event_snapshots(apply_horizon(pd.concat([h, late], ignore_index=True), 2.0), cfg)
    pd.testing.assert_frame_equal(a[SNAP_COLS].reset_index(drop=True), b[SNAP_COLS].reset_index(drop=True), check_categorical=False)


def test_no_snapshot_feature_uses_a_later_cdm(synthetic_allowlisted, cfg):
    df, _ = synthetic_allowlisted
    h = apply_horizon(df, 2.0)
    ev = h[h["event_id"] == 5].copy()
    base = FE.build_event_snapshots(ev, cfg)
    latest_idx = ev["time_to_tca"].idxmin()
    mod = ev.copy()
    mod.loc[latest_idx, ["miss_distance", "max_risk_estimate", "c_sigma_r"]] = [11.0, -1.5, 5.0]
    if "risk" in mod.columns:
        mod.loc[latest_idx, "risk"] = -1.0
    changed = FE.build_event_snapshots(mod, cfg)
    earlier = ev.index != latest_idx
    pd.testing.assert_frame_equal(base.loc[earlier, SNAP_COLS].reset_index(drop=True), changed.loc[earlier, SNAP_COLS].reset_index(drop=True), check_categorical=False)
    assert base.loc[latest_idx, "is_latest_snapshot"] and changed.loc[latest_idx, "is_latest_snapshot"]
    assert base.loc[latest_idx, "d_log_miss_first"] != changed.loc[latest_idx, "d_log_miss_first"]


def test_is_latest_snapshot_one_per_event_and_order_preserved(synthetic_allowlisted, cfg):
    df, _ = synthetic_allowlisted
    h = apply_horizon(df, 2.0).sample(frac=1.0, random_state=3).reset_index(drop=True)
    snap = FE.build_event_snapshots(h, cfg)
    assert (snap["row_id"].to_numpy() == h["row_id"].to_numpy()).all()
    per_event = snap.groupby("event_id")["is_latest_snapshot"].sum()
    assert (per_event == 1).all()
    lat = snap[snap["is_latest_snapshot"]].set_index("event_id")["time_to_tca"]
    assert np.allclose(lat.sort_index().to_numpy(), h.groupby("event_id")["time_to_tca"].min().sort_index().to_numpy())
    # n_cdms_2d counts snapshots so far, inclusive
    ev = snap[snap["event_id"] == 2].sort_values("time_to_tca", ascending=False)
    assert ev["n_cdms_2d"].tolist() == list(range(1, len(ev) + 1))
    assert ev.iloc[0]["has_history"] == 0 and np.isnan(ev.iloc[0]["slope_log_miss"]) and ev.iloc[0]["span_days"] == 0


def test_derived_labels_match_recomputed_final_tier(synthetic_paths, cfg):
    labels = derive_event_labels(synthetic_paths["train"], cfg)
    raw = pd.read_csv(synthetic_paths["train"])
    last = raw.sort_values("time_to_tca").groupby("event_id").head(1).set_index("event_id").loc[labels["event_id"]]
    recomputed = tier_from_true_risk(last["risk"].to_numpy(float), last["miss_distance"].to_numpy(float), cfg["thresholds"])
    assert (labels["final_tier"].to_numpy() == recomputed).all()
    assert labels["usable_for_training"].dtype == bool
    assert (labels["cdm_count_2d_or_more"] > 0).equals(labels["usable_for_training"])
    assert set(labels["final_tier"].unique()) <= {"Safe", "Elevated", "Critical"}


def test_load_event_labels_falls_back_to_derived_and_reports_counts(synthetic_cfg):
    labels, info = load_event_labels(synthetic_cfg)
    assert info["labels_source"] == "derived_from_train_data" and info["labels_match_recomputed"]
    assert info["n_usable"] + info["n_unusable"] == info["n_events_total"]
    assert info["n_trivial"] + info["n_real_forecast"] == info["n_usable"]
    assert "is_trivial" in labels.columns and "mission_id" in labels.columns


def test_mission_id_and_forbidden_columns_never_in_feature_matrix(synthetic_raw, synthetic_allowlisted, cfg):
    df, meta = synthetic_allowlisted
    assert "mission_id" in meta["dropped_columns"] and "mahalanobis_distance" in meta["dropped_columns"]
    snap = FE.build_event_snapshots(apply_horizon(df, 2.0), cfg)
    for fs in ("final_tier", "final_tier_no_past_risk", "final_tier_no_history"):
        X = FE.feature_matrix(snap, fs)
        assert "mission_id" not in X.columns and "event_id" not in X.columns and "row_id" not in X.columns
        assert "mahalanobis_distance" not in X.columns
    with pytest.raises(ValueError):
        FE.build_event_snapshots(synthetic_raw, cfg)  # un-allowlisted frame carries forbidden columns


def test_risk_present_only_when_use_past_risk(cfg):
    assert "risk" in FE.feature_columns(FE.final_tier_feature_set(cfg))
    off = copy.deepcopy(cfg); off["features"]["use_past_risk"] = False
    cols = FE.feature_columns(FE.final_tier_feature_set(off))
    assert "risk" not in cols and not any(c in cols for c in FE.EVENT_SO_FAR_RISK)


def test_train_and_test_are_never_joined_on_event_id(synthetic_allowlisted, cfg):
    """Snapshots for two datasets sharing event ids must stay separate."""
    df, _ = synthetic_allowlisted
    h = apply_horizon(df[df["event_id"] < 5], 2.0)
    other = h.copy(); other["dataset_id"] = "other"; other["row_id"] = other["row_id"].str.replace("synth", "other")
    other["miss_distance"] = other["miss_distance"] * 10
    both = pd.concat([h, other], ignore_index=True)
    snap = FE.build_event_snapshots(both, cfg)
    a = snap[snap["dataset_id"] == "synth"].sort_values("row_id")
    alone = FE.build_event_snapshots(h, cfg).sort_values("row_id")
    pd.testing.assert_frame_equal(a[SNAP_COLS].reset_index(drop=True), alone[SNAP_COLS].reset_index(drop=True), check_categorical=False)
    assert snap.groupby(["dataset_id", "event_id"])["is_latest_snapshot"].sum().eq(1).all()


def test_enforce_allowlist_keeps_risk_but_drops_mission(synthetic_raw, cfg):
    kept, dropped = enforce_allowlist(synthetic_raw, cfg)
    assert "risk" in kept.columns and "mission_id" in dropped


# --------------------------------------------------------------------------- #
# Workbook parsing: the usable flag must survive pandas 3 string dtypes
# --------------------------------------------------------------------------- #
def _with_unusable(derived: pd.DataFrame, n: int = 7) -> pd.DataFrame:
    """The synthetic fixture has a CDM at >= 2 days for every event; mark the first n as unusable so the
    usable-flag path is exercised (as 1,212 of 13,154 real events are)."""
    d = derived.copy()
    d.loc[d.index[:n], "cdm_count_2d_or_more"] = 0
    d.loc[d.index[:n], "usable_for_training"] = False
    return d


def _as_workbook(derived: pd.DataFrame) -> pd.DataFrame:
    """Mimic the official sheet: yes/no text for the usable flag, lower-case tiers, string columns."""
    wb = derived.drop(columns=["is_trivial"], errors="ignore").copy()
    wb["usable_for_training"] = np.where(wb["usable_for_training"], "yes", "no")
    wb["final_tier"] = wb["final_tier"].str.lower()
    return wb


def test_parse_usable_flag_text_is_not_truthy_by_dtype():
    text = pd.Series(["yes", "no", " No ", "YES"])
    assert text.dtype != object or True  # pandas 3 uses the str dtype; the parser must not care
    assert parse_usable_flag(text).tolist() == [True, False, False, True]
    assert parse_usable_flag(pd.Series([True, False])).tolist() == [True, False]
    assert parse_usable_flag(pd.Series([1, 0, 2])).tolist() == [True, False, True]
    with pytest.raises(ValueError):
        parse_usable_flag(pd.Series(["yes", "maybe"]))


def test_parse_workbook_labels_matches_derived_and_cross_check_passes(synthetic_paths, cfg):
    derived = _with_unusable(derive_event_labels(synthetic_paths["train"], cfg))
    assert (~derived["usable_for_training"]).any()
    lab = parse_workbook_labels(_as_workbook(derived))
    assert lab["usable_for_training"].dtype == bool
    assert lab["usable_for_training"].tolist() == derived["usable_for_training"].tolist()
    assert lab["final_tier"].tolist() == derived["final_tier"].tolist()
    info = cross_check_labels(lab, derived)
    assert info["labels_match_recomputed"] and info["n_usable_flag_mismatches"] == 0


def test_cross_check_catches_usable_flag_corruption_not_only_tiers(synthetic_paths, cfg):
    derived = _with_unusable(derive_event_labels(synthetic_paths["train"], cfg))
    wb = _as_workbook(derived)
    wb["usable_for_training"] = "yes"  # the failure mode: every event marked usable, tiers untouched
    lab = parse_workbook_labels(wb)
    info = cross_check_labels(lab, derived)
    assert info["n_tier_mismatches"] == 0
    assert info["n_usable_flag_mismatches"] == int((~derived["usable_for_training"]).sum()) > 0
    assert not info["labels_match_recomputed"]


def test_load_event_labels_reads_workbook_and_cross_checks(synthetic_paths, cfg, tmp_path):
    pytest.importorskip("openpyxl")
    import copy
    derived = derive_event_labels(synthetic_paths["train"], cfg)
    xlsx = tmp_path / "challenge1_event_labels.xlsx"
    with pd.ExcelWriter(xlsx) as xw:
        # two title rows above the header, as in the official workbook (header on row 3)
        pd.DataFrame([["Challenge 1 event labels"], [""]]).to_excel(xw, sheet_name="train_event_labels", index=False, header=False)
        _as_workbook(derived).to_excel(xw, sheet_name="train_event_labels", index=False, startrow=2)
    c = copy.deepcopy(cfg)
    c["paths"].update({"raw_train": str(synthetic_paths["train"]), "event_labels": str(xlsx)})
    labels, info = load_event_labels(c)
    assert info["labels_source"] == "workbook" and info["labels_match_recomputed"]
    assert info["n_usable"] == int(derived["usable_for_training"].sum())


def test_event_so_far_features_are_batch_independent_to_the_last_bit(synthetic_allowlisted, cfg):
    """A flat series must have volatility exactly 0, and an event's features must be bit-identical
    whether it is scored alone or inside a large batch (no cross-event cumsum rounding)."""
    df, _ = synthetic_allowlisted
    h = apply_horizon(df, 2.0).reset_index(drop=True)
    flat = h[h["event_id"] == 4].copy()
    flat["max_risk_estimate"] = -7.1234567
    if "risk" in flat.columns:
        flat["risk"] = -7.1234567
    big = pd.concat([h[h["event_id"] != 4], flat], ignore_index=True)
    snap_big = FE.build_event_snapshots(big, cfg)
    snap_alone = FE.build_event_snapshots(flat.reset_index(drop=True), cfg)
    a = snap_big[snap_big["event_id"] == 4].sort_values("row_id").reset_index(drop=True)
    b = snap_alone.sort_values("row_id").reset_index(drop=True)
    assert (a["volatility_mre"].fillna(0) == 0).all() and (b["volatility_mre"].fillna(0) == 0).all()
    for c in FE.EVENT_SO_FAR_BASE + FE.EVENT_SO_FAR_RISK:
        np.testing.assert_array_equal(a[c].to_numpy(), b[c].to_numpy(), err_msg=c)
