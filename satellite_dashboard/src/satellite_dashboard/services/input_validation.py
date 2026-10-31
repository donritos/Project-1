"""Schema-level checks for uploaded CDM histories, before anything reaches the model.

The model applies its own row-level quality checks afterwards (quality_flags). This module checks that the
file is usable at all: columns, types, units, event identity, size, and which rows the 2-day rule excludes."""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import ELIGIBILITY_EPS, MAX_UPLOAD_MB, MAX_UPLOAD_ROWS, MIN_TIME_TO_TCA_DAYS

REQUIRED = ["event_id", "time_to_tca", "miss_distance", "relative_position_r", "relative_position_t", "relative_position_n"]
RECOMMENDED = ["relative_speed", "c_object_type", "c_sigma_r", "t_sigma_r", "max_risk_estimate", "F10", "AP"]
PAST_RISK = "risk"
NUMERIC = ["time_to_tca", "miss_distance", "relative_speed", "relative_position_r", "relative_position_t", "relative_position_n",
           "c_sigma_r", "t_sigma_r", "max_risk_estimate", "F10", "AP", "risk"]
UNITS = {"time_to_tca": "days", "miss_distance": "m", "relative_position_r": "m", "relative_position_t": "m",
         "relative_position_n": "m", "relative_speed": "m/s", "c_sigma_r": "m", "t_sigma_r": "m",
         "max_risk_estimate": "log10 Pc", "risk": "log10 Pc"}


@dataclass
class ValidationResult:
    ok: bool
    dataset_id: str
    rows: pd.DataFrame                       # cleaned rows that may go to the model (with row_id)
    row_report: pd.DataFrame                 # one line per uploaded row: status + reason
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    ignored_columns: list[str] = field(default_factory=list)

    @property
    def n_eligible(self) -> int:
        return int((self.row_report["row_status"] == "Eligible").sum())


def dataset_id_for(content: bytes) -> str:
    return "upload:" + hashlib.sha1(content).hexdigest()[:10]


def validate_upload(content: bytes, *, use_past_risk: bool = True) -> ValidationResult:
    did = dataset_id_for(content)
    empty = pd.DataFrame()
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        return ValidationResult(False, did, empty, empty, [f"File is larger than {MAX_UPLOAD_MB} MB."])
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        return ValidationResult(False, did, empty, empty, [f"Could not read the file as CSV: {exc}"])
    if len(df) == 0:
        return ValidationResult(False, did, empty, empty, ["The file has no rows."])
    if len(df) > MAX_UPLOAD_ROWS:
        return ValidationResult(False, did, empty, empty, [f"The file has {len(df):,} rows; the limit is {MAX_UPLOAD_ROWS:,}."])
    df.columns = [str(c).strip() for c in df.columns]
    errors, warnings = [], []
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {', '.join(missing)}.")
        return ValidationResult(False, did, empty, empty, errors)
    rec_missing = [c for c in RECOMMENDED if c not in df.columns]
    if rec_missing:
        warnings.append(f"Recommended columns not supplied (the model treats them as missing): {', '.join(rec_missing)}.")
    if use_past_risk and PAST_RISK not in df.columns:
        warnings.append("No `risk` column. This model was trained with the risk of eligible CDMs as an input, "
                        "so forecasts without it are less reliable.")
    known = set(REQUIRED + RECOMMENDED + [PAST_RISK])
    ignored = [c for c in df.columns if c not in known]

    report = pd.DataFrame({"source_line": np.arange(2, len(df) + 2)})  # line 1 is the header
    reasons = [[] for _ in range(len(df))]
    ev = pd.to_numeric(df["event_id"], errors="coerce")
    bad_ev = ev.isna() | (ev != np.floor(ev)) | (ev < 0)
    for i in np.flatnonzero(bad_ev):
        reasons[i].append("event_id is not a non-negative integer")
    for c in NUMERIC:
        if c in df.columns:
            raw = df[c]
            num = pd.to_numeric(raw, errors="coerce")
            bad = raw.notna() & num.isna()
            for i in np.flatnonzero(bad):
                reasons[i].append(f"{c} is not numeric")
            df[c] = num
    t = df["time_to_tca"]
    for i in np.flatnonzero(t.isna()):
        reasons[i].append("time_to_tca is missing")
    md = df["miss_distance"]
    if md.notna().any() and float(md.median()) < 50:
        warnings.append("miss_distance values look like kilometres (median below 50). The model expects metres.")
    if "risk" in df.columns and (df["risk"] > 0).any():
        errors.append("risk must be log10(Pc) (never positive). Some values are positive.")
    if "max_risk_estimate" in df.columns and (df["max_risk_estimate"] > 0).any():
        errors.append("max_risk_estimate must be log10(Pc) (never positive). Some values are positive.")

    unusable = np.array([len(r) > 0 for r in reasons])
    eligible = ~unusable & (t >= MIN_TIME_TO_TCA_DAYS - ELIGIBILITY_EPS).to_numpy()
    status = np.where(unusable, "Rejected", np.where(eligible, "Eligible", "Excluded: inside 2-day cutoff"))
    report["event_id"] = ev.astype("Int64")
    report["time_to_tca_days"] = t
    report["row_status"] = status
    report["reason"] = ["; ".join(r) if r else ("" if s == "Eligible" else f"time_to_tca < {MIN_TIME_TO_TCA_DAYS:g} days")
                        for r, s in zip(reasons, status)]

    keep = ~unusable
    rows = df.loc[keep, [c for c in df.columns if c in known]].copy()
    rows["event_id"] = ev[keep].astype("int64")
    rows.insert(0, "row_id", [f"{did}:line{n}" for n in report.loc[keep, "source_line"]])
    rows.insert(1, "source_row_index", report.loc[keep, "source_line"].to_numpy())
    if unusable.any():
        warnings.append(f"{int(unusable.sum())} row(s) were rejected and will not reach the model (see the row report).")
    if keep.sum() == 0:
        errors.append("No usable rows remain after validation.")
    return ValidationResult(not errors, did, rows.reset_index(drop=True), report, errors, warnings, ignored)
