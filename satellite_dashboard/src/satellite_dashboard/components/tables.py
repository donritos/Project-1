"""Display tables built from standard records (kilometres and days only at this layer)."""
from __future__ import annotations

import pandas as pd

from ..formatting import FLAG_TEXT, quality_status

QUEUE_COLUMNS = {
    "event_id": "Event ID",
    "predicted_final_tier": "Predicted final tier",
    "p_critical": "P(final = Critical)",
    "latest_input_time_to_tca_days": "Latest input (d)",
    "latest_miss_km": "Latest observed miss (km)",
    "n_eligible_cdms": "Eligible CDMs",
    "data_quality": "Data quality",
    "review_priority": "Review priority",
    "review_status": "Analyst review",
}


def queue_table(events: pd.DataFrame, *, show_outcome: bool = False) -> pd.DataFrame:
    t = events.copy()
    t["predicted_final_tier"] = [tier if isinstance(tier, str) else status for tier, status in
                                 zip(t["predicted_final_tier"], t["prediction_status"])]
    t["latest_miss_km"] = t["latest_miss_distance_m"] / 1000
    for c in ("p_critical", "latest_input_time_to_tca_days", "latest_miss_km"):
        t[c] = pd.to_numeric(t[c], errors="coerce")
    t["p_critical"] = t["p_critical"] * 100  # percent, for display only
    t["data_quality"] = [quality_status(s, f) for s, f in zip(t["prediction_status"], t["quality_flags"])]
    cols = list(QUEUE_COLUMNS)
    if "review_status" not in t.columns:
        t["review_status"] = "Not reviewed"
    out = t[cols].rename(columns=QUEUE_COLUMNS)
    if show_outcome and "observed_final_tier" in t.columns:
        out.insert(3, "Observed final tier (evaluation only)", t["observed_final_tier"].fillna("–"))
    return out


def queue_column_config() -> dict:
    """Fixed pixel widths so the queue fits a laptop screen on every supported Streamlit version."""
    import streamlit as st
    return {
        "Event ID": st.column_config.NumberColumn(format="%d", width=70),
        "Predicted final tier": st.column_config.TextColumn(width=118),
        "Observed final tier (evaluation only)": st.column_config.TextColumn(width=132),
        "P(final = Critical)": st.column_config.ProgressColumn("P(final = Critical)", min_value=0.0, max_value=100.0, format="%.0f%%", width=116,
                                                               help="Model probability that the FINAL TIER is Critical. Not a collision probability."),
        "Latest input (d)": st.column_config.NumberColumn(format="%.2f", width=104, help="Time to TCA of the latest eligible CDM (≥ 2 days)."),
        "Latest observed miss (km)": st.column_config.NumberColumn(format="%.3f", width=150,
                                                                   help="Latest observed miss distance, measured at the latest eligible CDM. Not a forecast."),
        "Eligible CDMs": st.column_config.NumberColumn(width=96),
        "Data quality": st.column_config.TextColumn(width=88),
        "Review priority": st.column_config.TextColumn(width=136),
        "Analyst review": st.column_config.TextColumn(width=118),
    }


def inputs_table(inputs: pd.DataFrame) -> pd.DataFrame:
    t = inputs.copy()
    t["miss_km"] = t["miss_distance_m"] / 1000
    cols = {"time_to_tca_days": "Time to TCA (d)", "miss_km": "Miss distance (km)", "rel_pos_r_m": "R (m)", "rel_pos_t_m": "T (m)",
            "rel_pos_n_m": "N (m)", "relative_speed_mps": "Rel. speed (m/s)", "chaser_sigma_r_m": "Chaser σR (m)",
            "target_sigma_r_m": "Target σR (m)", "risk_log10": "Risk (log10 Pc)", "max_risk_estimate_log10": "Max risk est. (log10 Pc)",
            "object_type": "Object type", "f10": "F10.7", "ap": "AP"}
    out = t[[c for c in cols if c in t.columns]].rename(columns=cols)
    if "Risk (log10 Pc)" in out.columns:
        out["Risk (log10 Pc)"] = out["Risk (log10 Pc)"].map(lambda v: "≈0 (−30 placeholder)" if pd.notna(v) and v <= -30 + 1e-9
                                                            else ("–" if pd.isna(v) else f"{v:.2f}"))
    return out


def missing_fields(inputs: pd.DataFrame) -> list[str]:
    labels = {"relative_speed_mps": "relative speed", "chaser_sigma_r_m": "chaser σR", "target_sigma_r_m": "target σR",
              "max_risk_estimate_log10": "max risk estimate", "risk_log10": "risk", "object_type": "object type", "f10": "F10.7", "ap": "AP"}
    out = []
    for c, lab in labels.items():
        if c in inputs.columns and inputs[c].isna().any():
            n = int(inputs[c].isna().sum())
            out.append(f"{lab} missing in {n} of {len(inputs)} CDMs")
    return out


def flags_list(flags) -> list[str]:
    return [FLAG_TEXT.get(f, f) for f in (flags or [])]
