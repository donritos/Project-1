"""Required checks: horizon leakage, labels, columns, snapshot features."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from collision_risk import features as FE
from collision_risk.data import apply_horizon, derive_event_labels, enforce_allowlist, load_event_labels
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
