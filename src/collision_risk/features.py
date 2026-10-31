"""One reusable feature pipeline for training, batch prediction, and uploaded CDMs.

Design rules:
- Only allowlisted source columns enter (`assert_no_forbidden_columns` guards this).
- Every derived quantity uses documented floors; invalid inputs are flagged upstream
  (data.validate_rows) and become NaN here rather than plausible numbers.
- History features are strictly causal: a row may only use messages with a strictly
  larger time_to_tca (earlier in the event). Tied time_to_tca values never see each other.
- Original row order is preserved.
- A standalone CDM (no earlier messages) uses the documented "no history" representation:
  n_cdms_so_far = 1, days_since_first_cdm = 0, deltas and trend NaN, running stats = own values,
  has_history = 0.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .data import assert_no_forbidden_columns, event_group_codes

BASE_NUMERIC = [
    "time_to_tca", "miss_distance", "relative_speed", "relative_position_r", "relative_position_t",
    "relative_position_n", "c_sigma_r", "t_sigma_r", "max_risk_estimate", "F10", "AP",
]
CATEGORICAL = ["c_object_type"]
PHYSICS = [
    "log_miss_distance", "combined_sigma_r", "log_combined_sigma_r", "radial_sep_ratio", "sigma_ratio",
    "log_c_sigma_r", "log_t_sigma_r", "frac_r", "frac_t", "frac_n", "log_relative_speed",
    "missing_c_sigma_r", "missing_space_weather", "missing_max_risk", "sigma_suspicious",
]
HISTORY = [
    "n_cdms_so_far", "days_since_first_cdm", "delta_log_miss_distance", "delta_max_risk",
    "delta_log_combined_sigma", "run_min_log_miss", "run_max_max_risk", "run_min_max_risk", "mre_trend", "has_history",
]

FEATURE_SETS: dict[str, list[str]] = {
    "single_cdm": BASE_NUMERIC + CATEGORICAL,
    "physics": BASE_NUMERIC + CATEGORICAL + PHYSICS,
    "history": BASE_NUMERIC + CATEGORICAL + PHYSICS + HISTORY,
}


def feature_columns(feature_set: str) -> list[str]:
    return list(FEATURE_SETS[feature_set])


# --------------------------------------------------------------------------- #
def _log10_floor(x: np.ndarray, floor: float) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=float)
    ok = np.isfinite(x)
    out[ok] = np.log10(np.maximum(x[ok], floor))
    return out


def build_single_cdm_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Physics-inspired single-message features. Returns a new frame with added columns."""
    assert_no_forbidden_columns([c for c in df.columns if c not in ("risk",)], cfg)
    f = cfg["features"]
    out = df.copy()
    num = {c: pd.to_numeric(out[c], errors="coerce").to_numpy(dtype=float) for c in BASE_NUMERIC}

    # Invalid sigma / speed values become NaN (flagged upstream), never plausible numbers.
    for c in ("c_sigma_r", "t_sigma_r"):
        bad = np.isfinite(num[c]) & (num[c] <= 0)
        num[c] = np.where(bad, np.nan, num[c])
    bad_speed = np.isfinite(num["relative_speed"]) & (num["relative_speed"] <= 0)
    num["relative_speed"] = np.where(bad_speed, np.nan, num["relative_speed"])
    bad_miss = np.isfinite(num["miss_distance"]) & (num["miss_distance"] <= 0)
    miss = np.where(bad_miss, np.nan, num["miss_distance"])

    out["log_miss_distance"] = _log10_floor(miss, f["epsilon_m"])
    cs, ts = num["c_sigma_r"], num["t_sigma_r"]
    combined = np.sqrt(cs ** 2 + ts ** 2)  # independence assumption, documented
    out["combined_sigma_r"] = combined
    out["log_combined_sigma_r"] = _log10_floor(combined, f["sigma_floor_m"])
    with np.errstate(invalid="ignore", divide="ignore"):
        out["radial_sep_ratio"] = np.where(np.isfinite(combined) & (combined > 0), np.abs(num["relative_position_r"]) / combined, np.nan)
        out["sigma_ratio"] = np.where(np.isfinite(ts) & (ts > 0), cs / ts, np.nan)
        out["frac_r"] = np.where(np.isfinite(miss), np.abs(num["relative_position_r"]) / np.maximum(miss, f["epsilon_m"]), np.nan)
        out["frac_t"] = np.where(np.isfinite(miss), np.abs(num["relative_position_t"]) / np.maximum(miss, f["epsilon_m"]), np.nan)
        out["frac_n"] = np.where(np.isfinite(miss), np.abs(num["relative_position_n"]) / np.maximum(miss, f["epsilon_m"]), np.nan)
    out["log_c_sigma_r"] = _log10_floor(cs, f["sigma_floor_m"])
    out["log_t_sigma_r"] = _log10_floor(ts, f["sigma_floor_m"])
    out["log_relative_speed"] = _log10_floor(num["relative_speed"], f["speed_floor_mps"])

    # max_risk_estimate: verified log10 encoding (config). No further transform.
    if cfg["max_risk_estimate_encoding"] != "log10":
        raise ValueError("Unsupported max_risk_estimate encoding; audit must verify log10 before training.")

    out["missing_c_sigma_r"] = (~np.isfinite(cs)).astype(np.int8)
    out["missing_space_weather"] = (~np.isfinite(num["F10"]) | ~np.isfinite(num["AP"])).astype(np.int8)
    out["missing_max_risk"] = (~np.isfinite(num["max_risk_estimate"])).astype(np.int8)
    out["sigma_suspicious"] = ((np.isfinite(cs) & (cs > f["sigma_suspicious_m"])) | (np.isfinite(ts) & (ts > f["sigma_suspicious_m"]))).astype(np.int8)

    # Categorical with fixed category set; unseen -> NaN (flagged upstream as UNSEEN_OBJECT_TYPE)
    cats = list(f["object_type_categories"])
    ot = out["c_object_type"].astype("object")
    ot = ot.where(ot.isin(cats), np.nan)  # unseen categories -> NaN (flagged upstream)
    out["c_object_type"] = pd.Categorical(ot, categories=cats)
    return out


# --------------------------------------------------------------------------- #
def build_history_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Causal per-event history features. Requires build_single_cdm_features to have run.

    Ordering key within an event is time_to_tca descending (larger = earlier). Rows with
    equal time_to_tca form one 'time group'; a row's history is the set of rows in strictly
    earlier time groups, so tied messages never use one another. Encounters are scoped by
    (dataset_id, event_id). Rows with event_id < 0 or non-finite time_to_tca are standalone."""
    out = df.copy()
    n = len(out)
    key_cols = ["event_id", "time_to_tca", "log_miss_distance", "max_risk_estimate", "log_combined_sigma_r"]
    if "dataset_id" in out.columns:
        key_cols.insert(0, "dataset_id")
    w = out[key_cols].copy()
    w["_pos"] = np.arange(n)
    w["event_id"] = w["event_id"].fillna(-1).astype("int64")
    w["_event"] = event_group_codes(w)
    time_values = pd.to_numeric(w["time_to_tca"], errors="coerce").to_numpy(dtype=float)
    invalid_time = ~np.isfinite(time_values)
    # Invalid timestamps cannot contribute to another row's history. The internal
    # placeholder only permits ranking a singleton; the original input stays invalid.
    standalone = (w["event_id"] < 0) | invalid_time
    w.loc[standalone, "_event"] = -1 - np.arange(int(standalone.sum()))
    w["time_to_tca"] = np.where(invalid_time, 0.0, time_values)
    w["mre"] = pd.to_numeric(w["max_risk_estimate"], errors="coerce")

    w = w.sort_values(["_event", "time_to_tca"], ascending=[True, False], kind="stable")
    # dense rank of distinct time values within event: 1 = earliest message
    w["tg"] = w.groupby("_event")["time_to_tca"].rank(method="dense", ascending=False).astype("int64")

    # Per (event, time-group) aggregates, then exclusive cumulative over time groups.
    g = w.groupby(["_event", "tg"], sort=True).agg(
        cnt=("_pos", "size"),
        t=("time_to_tca", "first"),
        lm_min=("log_miss_distance", "min"),
        lm_mean=("log_miss_distance", "mean"),
        mre_max=("mre", "max"),
        mre_min=("mre", "min"),
        mre_mean=("mre", "mean"),
        ls_mean=("log_combined_sigma_r", "mean"),
    ).reset_index()
    ge = g.groupby("_event", sort=False)
    # Inclusive cumulative stats per time group, then shift by one group -> strictly earlier
    g["c_cnt"] = ge["cnt"].cumsum()
    g["c_lm_min"] = ge["lm_min"].cummin()
    g["c_mre_max"] = ge["mre_max"].cummax()
    g["c_mre_min"] = ge["mre_min"].cummin()
    g["first_t"] = ge["t"].transform("first")
    # trend accumulators (mean mre per group vs its time) for least-squares slope
    mre_ok = g["mre_mean"].notna()
    g["_n"] = mre_ok.astype(float)
    g["_st"] = np.where(mre_ok, g["t"], 0.0)
    g["_sm"] = np.where(mre_ok, g["mre_mean"], 0.0)
    g["_stm"] = np.where(mre_ok, g["t"] * g["mre_mean"], 0.0)
    g["_stt"] = np.where(mre_ok, g["t"] ** 2, 0.0)
    for c in ("_n", "_st", "_sm", "_stm", "_stt"):
        g["c" + c] = g.groupby("_event", sort=False)[c].cumsum()
    shift_cols = ["c_cnt", "c_lm_min", "c_mre_max", "c_mre_min", "lm_mean", "mre_mean", "ls_mean", "c_n", "c_st", "c_sm", "c_stm", "c_stt"]
    prev = g.groupby("_event", sort=False)[shift_cols].shift(1)
    prev.columns = ["p_" + c for c in shift_cols]
    g = pd.concat([g, prev], axis=1)

    m = w.merge(g[["_event", "tg", "first_t"] + list(prev.columns)], on=["_event", "tg"], how="left")
    m = m.sort_values("_pos", kind="stable")  # back to original order
    assert (m["_pos"].to_numpy() == np.arange(n)).all()

    p_cnt = m["p_c_cnt"].fillna(0).to_numpy(dtype=float)
    has_hist = p_cnt > 0
    lm = m["log_miss_distance"].to_numpy(dtype=float)
    mre = m["mre"].to_numpy(dtype=float)
    ls = m["log_combined_sigma_r"].to_numpy(dtype=float)
    t = m["time_to_tca"].to_numpy(dtype=float)

    out["n_cdms_so_far"] = (p_cnt + 1).astype(np.int32)
    out["has_history"] = has_hist.astype(np.int8)
    out["days_since_first_cdm"] = np.where(has_hist, m["first_t"].to_numpy(dtype=float) - t, 0.0)
    out["delta_log_miss_distance"] = np.where(has_hist, lm - m["p_lm_mean"].to_numpy(dtype=float), np.nan)
    out["delta_max_risk"] = np.where(has_hist, mre - m["p_mre_mean"].to_numpy(dtype=float), np.nan)
    out["delta_log_combined_sigma"] = np.where(has_hist, ls - m["p_ls_mean"].to_numpy(dtype=float), np.nan)
    out["run_min_log_miss"] = np.where(has_hist, np.fmin(lm, m["p_c_lm_min"].to_numpy(dtype=float)), lm)
    out["run_max_max_risk"] = np.where(has_hist, np.fmax(mre, m["p_c_mre_max"].to_numpy(dtype=float)), mre)
    out["run_min_max_risk"] = np.where(has_hist, np.fmin(mre, m["p_c_mre_min"].to_numpy(dtype=float)), mre)

    # Trend: least-squares slope of max_risk_estimate vs time_to_tca over strictly earlier
    # groups plus the current row (needs >= 2 points). Sign convention: slope per day of
    # time_to_tca; negative slope means the estimate increases as TCA approaches.
    cur_ok = np.isfinite(mre) & np.isfinite(t)
    nn = m["p_c_n"].fillna(0).to_numpy(dtype=float) + cur_ok
    st = m["p_c_st"].fillna(0).to_numpy(dtype=float) + np.where(cur_ok, t, 0.0)
    sm = m["p_c_sm"].fillna(0).to_numpy(dtype=float) + np.where(cur_ok, mre, 0.0)
    stm = m["p_c_stm"].fillna(0).to_numpy(dtype=float) + np.where(cur_ok, t * mre, 0.0)
    stt = m["p_c_stt"].fillna(0).to_numpy(dtype=float) + np.where(cur_ok, t ** 2, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        denom = nn * stt - st ** 2
        slope = (nn * stm - st * sm) / denom
    out["mre_trend"] = np.where((nn >= 2) & np.isfinite(slope) & (np.abs(denom) > 1e-12), slope, np.nan)
    return out


# --------------------------------------------------------------------------- #
def build_features(df: pd.DataFrame, cfg: dict, *, feature_set: str = "history") -> pd.DataFrame:
    """Full pipeline. Input must already be allowlisted (data.load_raw / prepare_uploaded)."""
    out = build_single_cdm_features(df, cfg)
    if feature_set == "history":
        out = build_history_features(out, cfg)
    return out


def feature_matrix(df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    cols = feature_columns(feature_set)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Feature matrix missing columns: {missing}")
    X = df[cols].copy()
    for c in cols:
        if c in CATEGORICAL:
            continue
        X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")
    return X


def compute_feature_ranges(X: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Training-range envelope used to flag out-of-range inference inputs."""
    ranges: dict[str, dict[str, float]] = {}
    for c in X.columns:
        if c in CATEGORICAL:
            continue
        s = pd.to_numeric(X[c], errors="coerce")
        if s.notna().any():
            ranges[c] = {"min": float(s.min()), "max": float(s.max())}
    return ranges


def out_of_range_mask(X: pd.DataFrame, ranges: dict[str, dict[str, float]], columns: list[str] | None = None) -> np.ndarray:
    cols = columns or [c for c in BASE_NUMERIC if c in ranges]
    mask = np.zeros(len(X), dtype=bool)
    for c in cols:
        if c not in X.columns or c not in ranges:
            continue
        v = pd.to_numeric(X[c], errors="coerce").to_numpy(dtype=float)
        mask |= np.isfinite(v) & ((v < ranges[c]["min"]) | (v > ranges[c]["max"]))
    return mask


def describe_feature_set(feature_set: str) -> dict[str, Any]:
    return {"feature_set": feature_set, "columns": feature_columns(feature_set), "categorical": CATEGORICAL,
            "history_representation_for_standalone": {"n_cdms_so_far": 1, "days_since_first_cdm": 0.0, "deltas": "NaN", "mre_trend": "NaN", "has_history": 0}}
