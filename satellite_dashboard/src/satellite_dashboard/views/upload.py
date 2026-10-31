"""Upload event histories (CSV) and run them through the same model pipeline used for the precomputed predictions."""
from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from .. import state
from ..components.status import kpi
from ..components.tables import queue_column_config, queue_table
from ..config import MAX_UPLOAD_MB, MIN_TIME_TO_TCA_DAYS, SAMPLE_INPUTS_DIR
from ..services.artifact_loader import ArtifactError
from ..services.input_validation import RECOMMENDED, REQUIRED, UNITS, validate_upload
from .overview import export_frame


def _open_upload(eid: int) -> None:
    st.session_state["dataset_key"] = "upload"
    state.select_event(eid)


def _run(content: bytes, label: str) -> None:
    adapter = state.adapter()
    manifest = adapter.manifest()
    res = validate_upload(content, use_past_risk=bool(manifest.model.get("use_past_risk")))
    st.session_state["upload_validation"] = res
    st.session_state["upload_view"] = None
    st.session_state["upload_error"] = None
    if not res.ok:
        return
    try:
        view, info = adapter.predict_upload(res.rows, res.dataset_id, label)
    except ImportError:
        st.session_state["upload_error"] = ("The model package (collision-risk) is not installed in this environment, so uploads "
                                            "cannot be predicted. Install the app with `uv sync` (or `pip install -e .. -e .`).")
        return
    except (ArtifactError, Exception) as exc:  # surface any model failure; never fake a result
        st.session_state["upload_error"] = f"The model could not produce predictions: {exc}"
        return
    st.session_state["upload_view"] = view
    st.session_state["upload_info"] = info


def render() -> None:
    st.title("Upload & predict")
    st.caption("Upload the CDM history of one or more events. Rows inside the 2-day cutoff are identified and excluded before the "
               "model builds any feature; the remaining rows are grouped by event and passed to the same model pipeline as the "
               "precomputed predictions.")
    manifest = state.adapter().manifest()
    with st.expander("File format", expanded=False):
        st.markdown("One row per CDM. Required columns: " + ", ".join(f"`{c}`" for c in REQUIRED) + ".  \nRecommended: "
                    + ", ".join(f"`{c}`" for c in RECOMMENDED) + ", and `risk` (this model uses the risk of eligible CDMs as an input).")
        st.dataframe(pd.DataFrame({"Column": list(UNITS), "Unit": list(UNITS.values())}), hide_index=True)
        st.markdown(f"Limits: {MAX_UPLOAD_MB} MB per file. `event_id` identifies events **within this file only**; it is never matched "
                    "against training or test events.")
        sample = SAMPLE_INPUTS_DIR / "sample_upload.csv"
        if sample.exists():
            st.download_button("Download sample file", sample.read_bytes(), file_name="sample_upload.csv", mime="text/csv")

    up = st.file_uploader("Event-history CSV", type=["csv"], accept_multiple_files=False)
    c1, c2 = st.columns([1, 3])
    if up is not None and c1.button("Validate and predict", type="primary"):
        with st.spinner("Running the model…"):
            _run(up.getvalue(), f"Upload: {up.name}")
    sample = SAMPLE_INPUTS_DIR / "sample_upload.csv"
    if up is None and sample.exists() and c1.button("Try the sample file"):
        with st.spinner("Running the model…"):
            _run(sample.read_bytes(), "Upload: sample_upload.csv")

    if manifest.capabilities.single_cdm_inference:
        with st.expander("Single CDM (no history)"):
            st.caption(manifest.capabilities.single_cdm_inference_note or "")
            with st.form("single_cdm"):
                cols = st.columns(4)
                t = cols[0].number_input("time_to_tca (days)", value=2.5, step=0.1, format="%.3f")
                md = cols[1].number_input("miss_distance (m)", value=1500.0, step=50.0)
                rr = cols[2].number_input("relative_position_r (m)", value=150.0)
                rt = cols[3].number_input("relative_position_t (m)", value=-1400.0)
                cols = st.columns(4)
                rn = cols[0].number_input("relative_position_n (m)", value=-500.0)
                spd = cols[1].number_input("relative_speed (m/s)", value=11000.0)
                risk = cols[2].number_input("risk (log10 Pc)", value=-5.5, max_value=0.0)
                mre = cols[3].number_input("max_risk_estimate (log10 Pc)", value=-4.5, max_value=0.0)
                cols = st.columns(4)
                cs = cols[0].number_input("c_sigma_r (m)", value=120.0)
                ts = cols[1].number_input("t_sigma_r (m)", value=15.0)
                ot = cols[2].selectbox("c_object_type", ["DEBRIS", "PAYLOAD", "ROCKET BODY", "UNKNOWN"])
                f10 = cols[3].number_input("F10", value=90.0)
                ap = st.number_input("AP", value=8.0)
                if st.form_submit_button("Predict single CDM"):
                    row = pd.DataFrame([{"event_id": 1, "time_to_tca": t, "miss_distance": md, "relative_position_r": rr,
                                         "relative_position_t": rt, "relative_position_n": rn, "relative_speed": spd, "risk": risk,
                                         "max_risk_estimate": mre, "c_sigma_r": cs, "t_sigma_r": ts, "c_object_type": ot, "F10": f10, "AP": ap}])
                    buf = io.StringIO(); row.to_csv(buf, index=False)
                    with st.spinner("Running the model…"):
                        _run(buf.getvalue().encode(), "Upload: single CDM")
    else:
        st.caption("Single-CDM entry is not offered: this model needs an event history.")

    res = st.session_state.get("upload_validation")
    if res is None:
        return
    st.divider()
    st.subheader("1 · Validation")
    for e in res.errors:
        st.error(e)
    for w in res.warnings:
        st.warning(w)
    if res.ignored_columns:
        st.caption("Ignored columns (not model inputs): " + ", ".join(f"`{c}`" for c in res.ignored_columns))
    rep = res.row_report
    if len(rep):
        counts = rep["row_status"].value_counts()
        c = st.columns(4)
        with c[0]:
            kpi("Rows in file", f"{len(rep):,}")
        with c[1]:
            kpi("Eligible rows", f"{int(counts.get('Eligible', 0)):,}", f"time_to_tca ≥ {MIN_TIME_TO_TCA_DAYS:g} days")
        with c[2]:
            kpi("Excluded by the 2-day rule", f"{int(counts.get('Excluded: inside 2-day cutoff', 0)):,}", "Never reach the model")
        with c[3]:
            kpi("Rejected rows", f"{int(counts.get('Rejected', 0)):,}", "Schema or type errors")
        with st.expander("Row-level validation results"):
            st.dataframe(rep, hide_index=True, width="stretch")
        st.download_button("Download row validation (CSV)", rep.to_csv(index=False).encode(), file_name="upload_row_validation.csv", mime="text/csv")
    if st.session_state.get("upload_error"):
        st.error(st.session_state["upload_error"])
    view = st.session_state.get("upload_view")
    if view is None:
        return
    st.subheader("2 · Event predictions")
    info = st.session_state.get("upload_info") or {}
    st.caption(f"{len(view.events)} event(s) · model `{view.model_version}` · "
               f"{info.get('n_rows_below_horizon_dropped', 0)} row(s) dropped by the model's horizon filter · "
               f"runtime {info.get('t_total', 0):.2f} s.")
    ev = state.events_with_review(view)
    st.dataframe(queue_table(ev), hide_index=True, width="stretch", column_config=queue_column_config())
    c = st.columns([1, 1, 2])
    c[0].download_button("Download predictions (CSV)", export_frame(ev).to_csv(index=False).encode(), file_name="upload_predictions.csv",
                         mime="text/csv")
    ids = ev["event_id"].astype(int).tolist()
    pick = c[1].selectbox("Open event", ids, label_visibility="collapsed")
    if c[2].button("Open in event detail →", on_click=_open_upload, args=(int(pick),)):
        state.go_to("detail")
