"""CSV and in-memory inputs share one normalisation routine: optional columns may be omitted,
event ids are validated, identifiers must be unique."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from collision_risk.data import load_raw, prepare_uploaded, validate_rows


def _rows(**overrides) -> pd.DataFrame:
    base = dict(event_id=[1, 1, 2], time_to_tca=[5.0, 3.0, 4.0], miss_distance=[12000.0, 8000.0, 900.0],
                relative_position_r=[100.0, 80.0, 50.0], relative_position_t=[11999.0, 7999.0, 898.0],
                relative_position_n=[100.0, 90.0, 20.0], c_object_type=["debris", "DEBRIS", "payload "],
                c_sigma_r=[50.0, 40.0, 30.0], t_sigma_r=[20.0, 20.0, 10.0], max_risk_estimate=[-7.0, -6.5, -4.0],
                F10=[70.0, 70.0, 72.0], AP=[5.0, 5.0, 6.0], relative_speed=[12000.0, 12000.0, 9000.0])
    base.update(overrides)
    return pd.DataFrame(base)


def _csv(tmp_path, df: pd.DataFrame):
    p = tmp_path / "in.csv"
    df.to_csv(p, index=False)
    return p


def test_csv_path_accepts_omitted_optional_columns_like_the_api(tmp_path, cfg):
    df = _rows().drop(columns=["relative_speed", "F10", "AP"])
    via_csv, meta = load_raw(_csv(tmp_path, df), "t", cfg)
    via_api, _ = prepare_uploaded(df, "t", cfg)
    for c in cfg["allowed_input_columns"]:
        assert c in via_csv.columns and c in via_api.columns
    assert via_csv["relative_speed"].isna().all()
    pd.testing.assert_frame_equal(via_csv.reset_index(drop=True), via_api.reset_index(drop=True), check_dtype=False)
    status, flags = validate_rows(via_csv, cfg)
    assert all("MISSING_SPEED" in f and "MISSING_SPACE_WEATHER" in f for f in flags)
    assert meta["n_kept_columns"] == len(df.columns)
    assert via_csv["c_object_type"].tolist() == ["DEBRIS", "DEBRIS", "PAYLOAD"]


def test_missing_required_geometry_column_becomes_invalid_rows_not_a_crash(tmp_path, cfg):
    df = _rows().drop(columns=["time_to_tca"])
    via_csv, _ = load_raw(_csv(tmp_path, df), "t", cfg)
    status, flags = validate_rows(via_csv, cfg)
    assert (status == "invalid").all()


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), 1.5, "abc"])
def test_malformed_event_ids_are_rejected_on_both_paths(tmp_path, cfg, bad):
    df = _rows(event_id=[1, bad, 2])
    with pytest.raises(ValueError, match="event_id"):
        prepare_uploaded(df, "t", cfg)
    with pytest.raises(ValueError, match="event_id"):
        load_raw(_csv(tmp_path, df), "t", cfg)


def test_missing_event_id_is_standalone_minus_one(tmp_path, cfg):
    df = _rows(event_id=[1, np.nan, 2])
    out, _ = prepare_uploaded(df, "t", cfg)
    assert out["event_id"].tolist() == [1, -1, 2]
    out, _ = load_raw(_csv(tmp_path, df), "t", cfg)
    assert out["event_id"].tolist() == [1, -1, 2]
    # integer-valued floats (as read from CSV) are fine
    out, _ = prepare_uploaded(_rows(event_id=[1.0, 1.0, 2.0]), "t", cfg)
    assert out["event_id"].dtype == "int64" and out["event_id"].tolist() == [1, 1, 2]


def test_caller_supplied_row_ids_must_be_unique(cfg):
    df = _rows(); df["row_id"] = ["a", "a", "b"]
    with pytest.raises(ValueError, match="row_id"):
        prepare_uploaded(df, "t", cfg)
    df["row_id"] = ["a", "b", "c"]
    out, _ = prepare_uploaded(df, "t", cfg)
    assert out["row_id"].tolist() == ["a", "b", "c"]


def test_csv_loader_ignores_stale_id_columns(tmp_path, cfg):
    df = _rows(); df["row_id"] = ["x", "x", "x"]; df["dataset_id"] = "old"
    out, _ = load_raw(_csv(tmp_path, df), "new", cfg)
    assert out["row_id"].tolist() == ["new:0", "new:1", "new:2"] and (out["dataset_id"] == "new").all()
