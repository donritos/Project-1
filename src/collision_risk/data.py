"""Data loading, input allowlist enforcement, row validation, and the pre-training audit.

Conventions (verified by `audit`): distances and sigmas in metres, speed in m/s,
time_to_tca in days, risk and max_risk_estimate in log10(Pc).
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .contracts import PredictionStatus, QualityFlag, RiskTier

ID_COLUMNS = ["dataset_id", "row_id", "source_row_index"]


def event_group_codes(df: pd.DataFrame) -> np.ndarray:
    """Row-aligned encounter codes, scoped by dataset when available.

    Legacy single-dataset frames may omit dataset_id. Frames without event_id
    treat each row as an independent observation.
    """
    if "event_id" not in df.columns:
        return np.arange(len(df), dtype=np.int64)
    columns = ["dataset_id", "event_id"] if "dataset_id" in df.columns else ["event_id"]
    keys = pd.MultiIndex.from_frame(df[columns])
    return pd.factorize(keys, sort=False)[0]


# --------------------------------------------------------------------------- #
# Loading and allowlist
# --------------------------------------------------------------------------- #
def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def enforce_allowlist(df: pd.DataFrame, cfg: dict, *, allow_target: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """Keep only allowed source columns (+ target if present and allowed, + id columns).

    Returns (filtered frame, list of dropped columns). Extra columns never pass silently."""
    allowed = set(cfg["allowed_input_columns"])
    if allow_target:
        allowed.add(cfg["target_column"])
    allowed |= set(ID_COLUMNS)
    keep = [c for c in df.columns if c in allowed]
    dropped = [c for c in df.columns if c not in allowed]
    return df[keep].copy(), dropped


def assert_no_forbidden_columns(columns: Iterable[str], cfg: dict) -> None:
    """Raise if any column outside the allowlist (target excluded) is present in a feature-pipeline input."""
    allowed = set(cfg["allowed_input_columns"]) | set(ID_COLUMNS)
    bad = [c for c in columns if c not in allowed]
    if bad:
        raise ValueError(f"Forbidden columns reached the feature pipeline: {bad}")


def _normalise_object_type(s: pd.Series) -> pd.Series:
    s = s.astype("object")
    return s.where(s.isna(), s.astype(str).str.strip().str.upper())


def load_raw(path: str | Path, dataset_id: str, cfg: dict) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a CSV, keep only allowed columns, attach stable ids preserving original order."""
    path = Path(path)
    raw = pd.read_csv(path)
    n_source_cols = raw.shape[1]
    raw.insert(0, "source_row_index", np.arange(len(raw), dtype=np.int64))
    raw.insert(0, "row_id", [f"{dataset_id}:{i}" for i in range(len(raw))])
    raw.insert(0, "dataset_id", dataset_id)
    df, dropped = enforce_allowlist(raw, cfg, allow_target=True)
    if "c_object_type" in df.columns:
        df["c_object_type"] = _normalise_object_type(df["c_object_type"])
    df["event_id"] = df["event_id"].astype("int64")
    meta = {
        "path": str(path),
        "sha256": file_sha256(path),
        "n_rows": int(len(df)),
        "n_source_columns": int(n_source_cols),
        "n_kept_columns": int(df.shape[1] - len(ID_COLUMNS)),
        "dropped_columns": dropped,
        "has_target": cfg["target_column"] in df.columns,
    }
    return df, meta


def prepare_uploaded(df: pd.DataFrame, dataset_id: str, cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    """Same treatment as load_raw for an in-memory frame (e.g. dashboard upload or API rows).

    Missing optional columns are added as NaN; missing event_id becomes -1 (no grouping)."""
    raw = df.copy().reset_index(drop=True)
    if "source_row_index" not in raw.columns:
        raw.insert(0, "source_row_index", np.arange(len(raw), dtype=np.int64))
    if "row_id" not in raw.columns:
        raw.insert(0, "row_id", [f"{dataset_id}:{i}" for i in range(len(raw))])
    raw["dataset_id"] = dataset_id
    out, dropped = enforce_allowlist(raw, cfg, allow_target=True)
    if "event_id" not in out.columns:
        out["event_id"] = -1
    out["event_id"] = pd.to_numeric(out["event_id"], errors="coerce").fillna(-1).astype("int64")
    for c in cfg["allowed_input_columns"]:
        if c not in out.columns:
            out[c] = np.nan
    out["c_object_type"] = _normalise_object_type(out["c_object_type"])
    for c in cfg["allowed_input_columns"]:
        if c not in ("event_id", "c_object_type"):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out, dropped


# --------------------------------------------------------------------------- #
# Validation -> prediction_status and quality_flags
# --------------------------------------------------------------------------- #
def validate_rows(df: pd.DataFrame, cfg: dict, *, known_object_types: list[str] | None = None) -> tuple[np.ndarray, list[list[str]]]:
    """Return (prediction_status array, per-row quality flags).

    Invalid: required geometry missing / non-finite, or miss_distance <= 0.
    Warning: any other quality flag. Valid: no flags."""
    n = len(df)
    flags: list[list[str]] = [[] for _ in range(n)]
    f = cfg["features"]
    req = cfg["required_geometry_columns"]
    known = set(known_object_types or f["object_type_categories"])

    def add(mask, flag: QualityFlag):
        for i in np.flatnonzero(np.asarray(mask, dtype=bool)):
            flags[i].append(flag.value)

    num = {
        c: (pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float) if c in df.columns else np.full(n, np.nan))
        for c in cfg["allowed_input_columns"] if c not in ("event_id", "c_object_type")
    }

    invalid = np.zeros(n, dtype=bool)
    geom_missing = np.zeros(n, dtype=bool)
    for c in req:
        geom_missing |= ~np.isfinite(num[c])
    add(geom_missing, QualityFlag.MISSING_REQUIRED_GEOMETRY)
    invalid |= geom_missing

    bad_miss = np.isfinite(num["miss_distance"]) & (num["miss_distance"] <= 0)
    add(bad_miss, QualityFlag.INVALID_MISS_DISTANCE)
    invalid |= bad_miss

    with np.errstate(invalid="ignore", divide="ignore"):
        rtn = np.sqrt(num["relative_position_r"] ** 2 + num["relative_position_t"] ** 2 + num["relative_position_n"] ** 2)
        rel = np.abs(rtn - num["miss_distance"]) / np.maximum(num["miss_distance"], 1.0)
    add(~invalid & np.isfinite(rel) & (rel > f["rtn_norm_tolerance"]), QualityFlag.RTN_NORM_MISMATCH)

    add(np.isfinite(num["time_to_tca"]) & (num["time_to_tca"] < 0), QualityFlag.NEGATIVE_TIME_TO_TCA)

    for c in ("c_sigma_r", "t_sigma_r"):
        add(~np.isfinite(num[c]), QualityFlag.MISSING_SIGMA)
        add(np.isfinite(num[c]) & (num[c] <= 0), QualityFlag.INVALID_SIGMA)
        add(np.isfinite(num[c]) & (num[c] > f["sigma_suspicious_m"]), QualityFlag.SUSPICIOUS_SIGMA)

    add(~np.isfinite(num["max_risk_estimate"]), QualityFlag.MISSING_MAX_RISK)
    add(~np.isfinite(num["F10"]) | ~np.isfinite(num["AP"]), QualityFlag.MISSING_SPACE_WEATHER)
    add(~np.isfinite(num["relative_speed"]), QualityFlag.MISSING_SPEED)
    add(np.isfinite(num["relative_speed"]) & (num["relative_speed"] <= 0), QualityFlag.INVALID_SPEED)

    if "c_object_type" in df.columns:
        ot = df["c_object_type"]
        add(ot.isna().to_numpy(), QualityFlag.MISSING_OBJECT_TYPE)
        add((~ot.isna() & ~ot.isin(list(known))).to_numpy(), QualityFlag.UNSEEN_OBJECT_TYPE)

    if "event_id" in df.columns and "time_to_tca" in df.columns:
        dataset_cols = ["dataset_id"] if "dataset_id" in df.columns else []
        tied = df.duplicated(dataset_cols + ["event_id", "time_to_tca"], keep=False).to_numpy() & (df["event_id"].to_numpy() >= 0)
        add(tied, QualityFlag.TIED_TIME_TO_TCA)
        cols = dataset_cols + [c for c in cfg["allowed_input_columns"] if c in df.columns]
        add(df.duplicated(cols, keep="first").to_numpy(), QualityFlag.DUPLICATE_ROW)

    flags = [list(dict.fromkeys(fl)) for fl in flags]
    has_flag = np.array([len(fl) > 0 for fl in flags], dtype=bool)
    status = np.where(invalid, PredictionStatus.INVALID.value,
                      np.where(has_flag, PredictionStatus.WARNING.value, PredictionStatus.VALID.value))
    return status.astype(object), flags


# --------------------------------------------------------------------------- #
# Audit
# --------------------------------------------------------------------------- #
def _tier_counts(risk: pd.Series, miss: pd.Series, th: dict) -> dict[str, int]:
    crit = (risk >= th["critical_risk"]) | (miss < th["critical_distance_m"])
    elev = ~crit & ((risk >= th["elevated_risk"]) | ((miss >= th["critical_distance_m"]) & (miss <= th["elevated_distance_max_m"])))
    safe = ~crit & ~elev
    return {"Critical": int(crit.sum()), "Elevated": int(elev.sum()), "Safe": int(safe.sum())}


def _f(x) -> float | None:
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(xf) else xf


def _describe(s: pd.Series) -> dict[str, Any]:
    s = pd.to_numeric(s, errors="coerce")
    q = s.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "n_missing": int(s.isna().sum()), "min": _f(s.min()), "max": _f(s.max()), "mean": _f(s.mean()), "std": _f(s.std()),
        **{f"p{int(round(k * 100)):02d}": _f(v) for k, v in q.items()},
    }


def _ks_stat(a: pd.Series, b: pd.Series) -> float | None:
    a = pd.to_numeric(a, errors="coerce").dropna().to_numpy()
    b = pd.to_numeric(b, errors="coerce").dropna().to_numpy()
    if len(a) == 0 or len(b) == 0:
        return None
    from scipy.stats import ks_2samp
    return _f(ks_2samp(a, b).statistic)


def audit(train: pd.DataFrame, test: pd.DataFrame | None, cfg: dict, train_meta: dict, test_meta: dict | None) -> dict[str, Any]:
    """Machine-readable audit of the delivered files (section 4 of the spec)."""
    th = cfg["thresholds"]
    tgt = cfg["target_column"]
    f = cfg["features"]
    floor = cfg["model"]["target_floor"]
    horizon = cfg["split"]["test_horizon_min_days"]
    num_cols = [c for c in cfg["allowed_input_columns"] if c not in ("event_id", "c_object_type")]
    rep: dict[str, Any] = {"files": {"train": train_meta, "test": test_meta}, "units_assumed": cfg["units"]}

    def basic(df: pd.DataFrame) -> dict[str, Any]:
        d: dict[str, Any] = {"n_rows": int(len(df)), "n_events": int(df["event_id"].nunique())}
        per_event = df.groupby("event_id").size()
        d["cdms_per_event"] = {
            "min": int(per_event.min()), "median": _f(per_event.median()), "mean": _f(per_event.mean()), "max": int(per_event.max()),
            "histogram": {str(k): int(v) for k, v in per_event.value_counts().sort_index().items()},
        }
        d["time_to_tca_days"] = _describe(df["time_to_tca"])
        d["rows_with_time_to_tca_below_horizon"] = {"horizon_days": horizon, "n": int((df["time_to_tca"] < horizon).sum())}
        d["rows_with_negative_time_to_tca"] = int((df["time_to_tca"] < 0).sum())
        d["missing_values"] = {c: int(df[c].isna().sum()) for c in df.columns if df[c].isna().any()}
        d["invalid_values"] = {
            "miss_distance_le_0": int((df["miss_distance"] <= 0).sum()),
            "relative_speed_le_0": int((df["relative_speed"] <= 0).sum()),
            "c_sigma_r_le_0": int((df["c_sigma_r"] <= 0).sum()),
            "t_sigma_r_le_0": int((df["t_sigma_r"] <= 0).sum()),
            "c_sigma_r_suspicious": {"threshold_m": f["sigma_suspicious_m"], "n": int((df["c_sigma_r"] > f["sigma_suspicious_m"]).sum())},
            "t_sigma_r_suspicious": {"threshold_m": f["sigma_suspicious_m"], "n": int((df["t_sigma_r"] > f["sigma_suspicious_m"]).sum())},
            "max_risk_estimate_positive": int((df["max_risk_estimate"] > 0).sum()),
        }
        cols = [c for c in cfg["allowed_input_columns"] if c in df.columns]
        dup_pairs = df.duplicated(["event_id", "time_to_tca"], keep=False)
        conflicting = 0
        if dup_pairs.any():
            diff = df[dup_pairs].groupby(["event_id", "time_to_tca"])["miss_distance"].nunique()
            conflicting = int((diff > 1).sum())
        d["duplicates"] = {
            "exact_duplicate_rows": int(df.duplicated(cols).sum()),
            "duplicate_event_time_pairs": int(df.duplicated(["event_id", "time_to_tca"]).sum()),
            "conflicting_duplicate_timestamps": conflicting,
        }
        d["feature_summary"] = {c: _describe(df[c]) for c in num_cols}
        d["object_type_counts"] = {str(k): int(v) for k, v in df["c_object_type"].value_counts(dropna=False).items()}
        rtn = np.sqrt(df["relative_position_r"] ** 2 + df["relative_position_t"] ** 2 + df["relative_position_n"] ** 2)
        rel = (rtn - df["miss_distance"]).abs() / df["miss_distance"].clip(lower=1.0)
        d["rtn_norm_vs_miss_distance"] = {
            "tolerance_relative": f["rtn_norm_tolerance"], "median_rel_error": _f(rel.median()), "p99_rel_error": _f(rel.quantile(0.99)),
            "n_exceeding_tolerance": int((rel > f["rtn_norm_tolerance"]).sum()), "fraction_exceeding_tolerance": _f((rel > f["rtn_norm_tolerance"]).mean()),
        }
        if tgt in df.columns:
            r = df[tgt]
            d["target"] = {
                **_describe(r), "floor_value": floor, "n_at_floor": int((r <= floor + 1e-9).sum()), "fraction_at_floor": _f((r <= floor + 1e-9).mean()),
                "n_ge_minus4": int((r >= -4).sum()), "n_ge_minus6": int((r >= -6).sum()),
                "histogram_1unit_bins": {str(k): int(v) for k, v in pd.cut(r, bins=list(range(-31, 1, 1))).value_counts().sort_index().items()},
            }
            d["tier_counts_row_level"] = _tier_counts(r, df["miss_distance"], th)
            ev = df.groupby("event_id").agg(risk=(tgt, "max"), miss=("miss_distance", "min"))
            d["tier_counts_event_level_max_severity"] = _tier_counts(ev["risk"], ev["miss"], th)
            latest = df.sort_values("time_to_tca", kind="stable").groupby("event_id").head(1)
            d["tier_counts_latest_cdm"] = _tier_counts(latest[tgt], latest["miss_distance"], th)
            mre = df["max_risk_estimate"]
            nf = r > floor
            d["max_risk_estimate_encoding"] = {
                "all_non_positive": bool((mre.dropna() <= 0).all()), "min": _f(mre.min()), "max": _f(mre.max()),
                "inferred_encoding": "log10" if (mre.dropna() <= 0).all() and mre.min() < -1 else "unknown",
                "configured_encoding": cfg["max_risk_estimate_encoding"],
                "pearson_corr_with_risk": _f(mre.corr(r)),
                "pearson_corr_with_risk_non_floor": _f(mre[nf].corr(r[nf])),
                "n_rows_risk_exceeds_mre": int((r > mre + 1e-9).sum()),
                "fraction_rows_risk_exceeds_mre": _f((r > mre + 1e-9).mean()),
                "is_strict_upper_bound": bool((r <= mre + 1e-9).all()),
            }
        return d

    rep["train"] = basic(train)
    if test is not None:
        rep["test"] = basic(test)
        tr_ev, te_ev = set(train["event_id"]), set(test["event_id"])
        rep["train_test_event_id_overlap"] = {
            "n_overlapping_ids": len(tr_ev & te_ev),
            "n_test_events": len(te_ev),
            "note": "event_id values are reused across files; events are unique only within a file. All joins use (dataset_id, event_id).",
        }
        rep["test_horizon"] = {
            "min_time_to_tca_days": _f(test["time_to_tca"].min()),
            "test_rows_stop_at_or_after_horizon": bool(test["time_to_tca"].min() >= horizon - 1e-9),
        }
        tr_types, te_types = set(train["c_object_type"].dropna()), set(test["c_object_type"].dropna())
        rep["unseen_object_types_in_test"] = sorted(te_types - tr_types)
        rep["train_test_distribution_shift_ks"] = {c: _ks_stat(train[c], test[c]) for c in num_cols}
    enc = rep["train"].get("max_risk_estimate_encoding", {})
    rep["checks"] = {
        "units_consistent_with_metres": bool(train["miss_distance"].median() > 100),
        "max_risk_estimate_is_log10": enc.get("inferred_encoding") == "log10",
        "max_risk_estimate_matches_config": enc.get("inferred_encoding") == cfg["max_risk_estimate_encoding"],
        "max_risk_estimate_is_strict_upper_bound": enc.get("is_strict_upper_bound"),
        "rtn_norm_fraction_exceeding_tolerance": rep["train"]["rtn_norm_vs_miss_distance"]["fraction_exceeding_tolerance"],
        "test_horizon_restricted": rep.get("test_horizon", {}).get("test_rows_stop_at_or_after_horizon"),
    }
    return rep

# --------------------------------------------------------------------------- #
# Final-tier task: event labels and the 2-day horizon
# --------------------------------------------------------------------------- #
LABEL_COLUMNS = ["event_id", "mission_id", "cdm_count", "cdm_count_2d_or_more", "final_time_to_tca_days",
                 "final_risk_log10_pc", "final_miss_distance_m", "final_tier", "usable_for_training"]


def apply_horizon(df: pd.DataFrame, min_days: float) -> pd.DataFrame:
    """Keep only CDMs with time_to_tca >= min_days. Must run BEFORE any feature engineering so
    that history features can never see later CDMs. Rows with non-finite time_to_tca are dropped."""
    t = pd.to_numeric(df["time_to_tca"], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(t) & (t >= float(min_days) - 1e-12)
    return df.loc[keep].copy()


def derive_event_labels(train_csv: str | Path, cfg: dict) -> pd.DataFrame:
    """Recompute the labels workbook content from the raw training file: one row per event with the
    tier of its LAST CDM (smallest time_to_tca). Used to verify the workbook, or as the label source
    when the workbook is not available (recorded in the audit as labels_source='derived')."""
    from .policy import tier_from_true_risk
    cols = ["event_id", "mission_id", "time_to_tca", "risk", "miss_distance"]
    raw = pd.read_csv(train_csv, usecols=cols)
    raw = raw[np.isfinite(pd.to_numeric(raw["time_to_tca"], errors="coerce"))]
    last = raw.sort_values(["event_id", "time_to_tca"], kind="stable").groupby("event_id").head(1).set_index("event_id")
    g = raw.groupby("event_id")
    out = pd.DataFrame({
        "event_id": last.index.to_numpy(),
        "mission_id": last["mission_id"].to_numpy(),
        "cdm_count": g.size().reindex(last.index).to_numpy(),
        "cdm_count_2d_or_more": g["time_to_tca"].apply(lambda s: int((s >= cfg["horizon"]["min_days"]).sum())).reindex(last.index).to_numpy(),
        "final_time_to_tca_days": last["time_to_tca"].to_numpy(),
        "final_risk_log10_pc": last["risk"].to_numpy(),
        "final_miss_distance_m": last["miss_distance"].to_numpy(),
    })
    out["final_tier"] = tier_from_true_risk(out["final_risk_log10_pc"].to_numpy(float), out["final_miss_distance_m"].to_numpy(float), cfg["thresholds"])
    out["usable_for_training"] = out["cdm_count_2d_or_more"] > 0
    return out.sort_values("event_id").reset_index(drop=True)


def load_event_labels(cfg: dict) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the official workbook sheet `train_event_labels` (header on row 3) if present, validate it,
    and cross-check it against labels recomputed from train_data.csv. Falls back to the recomputed labels
    when the workbook is missing. Returns (labels, info). Raises if the workbook disagrees with the data."""
    from .config import PROJECT_ROOT
    xlsx = Path(cfg["paths"]["event_labels"])
    xlsx = xlsx if xlsx.is_absolute() else PROJECT_ROOT / xlsx
    train_csv = Path(cfg["paths"]["raw_train"])
    train_csv = train_csv if train_csv.is_absolute() else PROJECT_ROOT / train_csv
    derived = derive_event_labels(train_csv, cfg)
    info: dict[str, Any] = {"workbook_path": str(xlsx), "workbook_present": xlsx.exists()}
    if xlsx.exists():
        lab = pd.read_excel(xlsx, sheet_name="train_event_labels", header=2)
        lab.columns = [str(c).strip() for c in lab.columns]
        missing = [c for c in LABEL_COLUMNS if c not in lab.columns]
        if missing:
            raise ValueError(f"labels workbook missing columns: {missing}")
        lab = lab[LABEL_COLUMNS].copy()
        lab["event_id"] = lab["event_id"].astype("int64")
        lab["final_tier"] = lab["final_tier"].astype(str).str.strip().str.capitalize()
        valid_tiers = {t.value for t in RiskTier}
        bad = sorted(set(lab["final_tier"]) - valid_tiers)
        if bad:
            raise ValueError(f"unexpected final_tier values in workbook: {bad}")
        u = lab["usable_for_training"]
        lab["usable_for_training"] = u.astype(str).str.strip().str.lower().isin(["yes", "true", "1", "y"]) if u.dtype == object else u.astype(bool)
        lab = lab.sort_values("event_id").reset_index(drop=True)
        m = lab.merge(derived, on="event_id", how="outer", suffixes=("", "_recomputed"), indicator=True)
        info["n_events_only_in_workbook"] = int((m["_merge"] == "left_only").sum())
        info["n_events_only_in_data"] = int((m["_merge"] == "right_only").sum())
        both = m[m["_merge"] == "both"]
        mismatch = int((both["final_tier"] != both["final_tier_recomputed"]).sum())
        info["n_tier_mismatches"] = mismatch
        info["labels_match_recomputed"] = mismatch == 0 and info["n_events_only_in_workbook"] == 0 and info["n_events_only_in_data"] == 0
        info["labels_source"] = "workbook"
        if not info["labels_match_recomputed"]:
            raise RuntimeError(f"Labels workbook disagrees with train_data.csv: {info}")
        labels = lab
    else:
        labels = derived
        info["labels_source"] = "derived_from_train_data"
        info["labels_match_recomputed"] = True
        info["n_tier_mismatches"] = 0
    labels["is_trivial"] = labels["final_time_to_tca_days"] >= float(cfg["horizon"]["min_days"]) - 1e-12
    labels.loc[~labels["usable_for_training"], "is_trivial"] = False
    info.update({
        "n_events_total": int(len(labels)),
        "n_usable": int(labels["usable_for_training"].sum()),
        "n_unusable": int((~labels["usable_for_training"]).sum()),
        "n_trivial": int(labels.loc[labels["usable_for_training"], "is_trivial"].sum()),
        "n_real_forecast": int((~labels.loc[labels["usable_for_training"], "is_trivial"]).sum()),
        "class_counts_all": labels["final_tier"].value_counts().to_dict(),
        "class_counts_usable": labels.loc[labels["usable_for_training"], "final_tier"].value_counts().to_dict(),
        "class_counts_trivial": labels.loc[labels["usable_for_training"] & labels["is_trivial"], "final_tier"].value_counts().to_dict(),
        "class_counts_real_forecast": labels.loc[labels["usable_for_training"] & ~labels["is_trivial"], "final_tier"].value_counts().to_dict(),
        "n_missions": int(labels["mission_id"].nunique()),
    })
    return labels, info
