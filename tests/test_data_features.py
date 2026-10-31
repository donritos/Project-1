"""Required checks: allowlist, validation, feature causality, order preservation, pipeline identity."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from collision_risk import features as FE
from collision_risk.data import assert_no_forbidden_columns, enforce_allowlist, prepare_uploaded, validate_rows


def test_forbidden_columns_are_dropped_and_reported(synthetic_allowlisted, cfg):
    df, meta = synthetic_allowlisted
    assert "mahalanobis_distance" in meta["dropped_columns"]
    assert "mission_id" in meta["dropped_columns"]
    assert "max_risk_scaling" in meta["dropped_columns"]
    forbidden = set(df.columns) - set(cfg["allowed_input_columns"]) - {"risk", "dataset_id", "row_id", "source_row_index"}
    assert not forbidden


def test_forbidden_columns_cannot_reach_feature_pipeline(synthetic_raw, cfg):
    with pytest.raises(ValueError):
        assert_no_forbidden_columns(synthetic_raw.columns, cfg)
    with pytest.raises(ValueError):
        FE.build_features(synthetic_raw, cfg)  # un-allowlisted frame carries mahalanobis_distance


def test_target_never_in_feature_matrix(synthetic_allowlisted, cfg):
    df, _ = synthetic_allowlisted
    F = FE.build_features(df, cfg)
    for fs in FE.FEATURE_SETS:
        X = FE.feature_matrix(F, fs)
        assert "risk" not in X.columns and "event_id" not in X.columns and "row_id" not in X.columns


def test_row_ids_stable_and_order_preserved(synthetic_allowlisted, cfg):
    df, _ = synthetic_allowlisted
    assert (df["source_row_index"].to_numpy() == np.arange(len(df))).all()
    assert df["row_id"].is_unique
    shuffled = df.sample(frac=1.0, random_state=1).reset_index(drop=True)
    F = FE.build_features(shuffled, cfg)
    assert (F["row_id"].to_numpy() == shuffled["row_id"].to_numpy()).all()


def test_validation_flags_invalid_geometry(synthetic_allowlisted, cfg):
    df, _ = synthetic_allowlisted
    status, flags = validate_rows(df, cfg)
    assert status[3] == "invalid" and "MISSING_REQUIRED_GEOMETRY" in flags[3]
    assert status[7] == "invalid"
    assert (status == "valid").sum() > 0


def test_history_features_are_causal(synthetic_allowlisted, cfg):
    """Changing a LATER message (smaller time_to_tca) must not change any earlier row's features."""
    df, _ = synthetic_allowlisted
    ev = df[df["event_id"] == 5].copy()
    base = FE.build_features(ev, cfg)
    latest_idx = ev["time_to_tca"].idxmin()
    mod = ev.copy()
    mod.loc[latest_idx, ["miss_distance", "max_risk_estimate", "c_sigma_r"]] = [11.0, -1.5, 5.0]
    changed = FE.build_features(mod, cfg)
    hist_cols = FE.HISTORY
    earlier = ev.index != latest_idx
    pd.testing.assert_frame_equal(base.loc[earlier, hist_cols].reset_index(drop=True), changed.loc[earlier, hist_cols].reset_index(drop=True))
    # the modified latest row itself does change
    assert not np.allclose(base.loc[latest_idx, "delta_log_miss_distance"], changed.loc[latest_idx, "delta_log_miss_distance"])


def test_history_count_and_standalone_representation(synthetic_allowlisted, cfg):
    df, _ = synthetic_allowlisted
    F = FE.build_features(df, cfg)
    ev = F[F["event_id"] == 2].sort_values("time_to_tca", ascending=False)
    assert ev["n_cdms_so_far"].tolist() == list(range(1, len(ev) + 1))
    first = ev.iloc[0]
    assert first["has_history"] == 0 and first["days_since_first_cdm"] == 0 and np.isnan(first["delta_log_miss_distance"]) and np.isnan(first["mre_trend"])
    assert first["run_min_log_miss"] == first["log_miss_distance"]


def test_tied_time_to_tca_rows_do_not_see_each_other(cfg):
    base = {"dataset_id": "t", "relative_speed": 1000.0, "relative_position_r": 10.0, "relative_position_t": 100.0,
            "relative_position_n": 100.0, "c_object_type": "DEBRIS", "c_sigma_r": 10.0, "t_sigma_r": 2.0, "F10": 70.0, "AP": 5.0}
    rows = [
        {**base, "row_id": "t:0", "source_row_index": 0, "event_id": 1, "time_to_tca": 5.0, "miss_distance": 1000.0, "max_risk_estimate": -7.0},
        {**base, "row_id": "t:1", "source_row_index": 1, "event_id": 1, "time_to_tca": 3.0, "miss_distance": 2000.0, "max_risk_estimate": -6.0},
        {**base, "row_id": "t:2", "source_row_index": 2, "event_id": 1, "time_to_tca": 3.0, "miss_distance": 500.0, "max_risk_estimate": -5.0},
    ]
    F = FE.build_features(pd.DataFrame(rows), cfg)
    # both tied rows have exactly one earlier message (the t=5 row), not each other
    assert F.loc[1, "n_cdms_so_far"] == 2 and F.loc[2, "n_cdms_so_far"] == 2
    assert np.isclose(F.loc[1, "delta_max_risk"], -6.0 - (-7.0)) and np.isclose(F.loc[2, "delta_max_risk"], -5.0 - (-7.0))
    assert np.isclose(F.loc[1, "run_min_log_miss"], np.log10(1000.0)) and np.isclose(F.loc[2, "run_min_log_miss"], np.log10(500.0))


def test_feature_definitions_identical_train_vs_inference(synthetic_allowlisted, cfg):
    """Features computed on the full frame equal features computed through the upload path row-group by row-group."""
    df, _ = synthetic_allowlisted
    F_all = FE.build_features(df, cfg)
    ev = df[df["event_id"] == 9]
    up, _ = prepare_uploaded(ev.drop(columns=["row_id", "source_row_index", "dataset_id"]), "synth", cfg)
    up["row_id"] = ev["row_id"].to_numpy()
    F_up = FE.build_features(up, cfg)
    cols = FE.feature_columns("history")
    a = F_all[F_all["event_id"] == 9][cols].reset_index(drop=True)
    b = F_up[cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_categorical=False)


def test_invalid_sigma_becomes_nan_not_plausible(cfg):
    base = {"dataset_id": "t", "row_id": "t:0", "source_row_index": 0, "event_id": 1, "time_to_tca": 2.0, "miss_distance": 1000.0,
            "relative_speed": 0.0, "relative_position_r": 10.0, "relative_position_t": 100.0, "relative_position_n": 100.0,
            "c_object_type": "WEIRD", "c_sigma_r": -3.0, "t_sigma_r": 2.0, "max_risk_estimate": -7.0, "F10": 70.0, "AP": 5.0}
    df = pd.DataFrame([base])
    status, flags = validate_rows(df, cfg)
    assert status[0] == "warning"
    assert {"INVALID_SIGMA", "INVALID_SPEED", "UNSEEN_OBJECT_TYPE"} <= set(flags[0])
    F = FE.build_features(df, cfg)
    assert np.isnan(F.loc[0, "log_c_sigma_r"]) and np.isnan(F.loc[0, "log_relative_speed"]) and pd.isna(F.loc[0, "c_object_type"])


def test_enforce_allowlist_keeps_only_allowed(synthetic_raw, cfg):
    kept, dropped = enforce_allowlist(synthetic_raw, cfg)
    assert set(kept.columns) <= set(cfg["allowed_input_columns"]) | {"risk"}
    assert "mahalanobis_distance" in dropped


@pytest.mark.parametrize("bad_time", [np.nan, np.inf, -np.inf])
def test_nonfinite_time_is_standalone_and_cannot_contaminate_history(synthetic_allowlisted, cfg, bad_time):
    df, _ = synthetic_allowlisted
    good = df[df["event_id"] == 5].copy()
    invalid = good.iloc[[0]].copy()
    invalid["time_to_tca"] = bad_time
    invalid["max_risk_estimate"] = -0.1
    mixed = pd.concat([invalid, good, invalid], ignore_index=True)
    features = FE.build_features(mixed, cfg)
    expected = FE.build_features(good, cfg)
    pd.testing.assert_frame_equal(
        features.iloc[1:-1][FE.HISTORY].reset_index(drop=True),
        expected[FE.HISTORY].reset_index(drop=True),
    )
    for pos in (0, len(features) - 1):
        assert features.iloc[pos]["n_cdms_so_far"] == 1
        assert features.iloc[pos]["has_history"] == 0
        assert features.iloc[pos]["days_since_first_cdm"] == 0
        assert np.isnan(features.iloc[pos]["delta_max_risk"])
        assert np.isnan(features.iloc[pos]["mre_trend"])
    np.testing.assert_equal(features["time_to_tca"].to_numpy(), mixed["time_to_tca"].to_numpy())


def test_history_and_quality_flags_are_isolated_by_dataset(synthetic_allowlisted, cfg):
    df, _ = synthetic_allowlisted
    a = df[df["event_id"] == 5].copy()
    b = a.copy()
    b["dataset_id"] = "other"
    b["row_id"] = "other:" + b["source_row_index"].astype(str)
    mixed = pd.concat([a, b], ignore_index=True).sample(frac=1, random_state=12).reset_index(drop=True)
    actual = FE.build_features(mixed, cfg).set_index("row_id")[FE.HISTORY].sort_index()
    expected = pd.concat([FE.build_features(a, cfg), FE.build_features(b, cfg)]).set_index("row_id")[FE.HISTORY].sort_index()
    pd.testing.assert_frame_equal(actual, expected)
    _, flags = validate_rows(mixed, cfg)
    assert all("TIED_TIME_TO_TCA" not in f and "DUPLICATE_ROW" not in f for f in flags)
    _, flags = validate_rows(pd.concat([a.iloc[[0]], a.iloc[[0]]], ignore_index=True), cfg)
    assert all("TIED_TIME_TO_TCA" in f for f in flags)
    assert "DUPLICATE_ROW" in flags[1]
