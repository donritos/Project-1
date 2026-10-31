"""Overview: the review queue. One row per event."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import state
from ..components import charts
from ..components.status import show_chart, dataset_banner, kpi
from ..components.tables import queue_column_config, queue_table
from ..config import MIN_TIME_TO_TCA_DAYS
from ..formatting import PRIORITY_HELP
from ..services.metrics import evaluation_frame


def export_frame(ev: pd.DataFrame) -> pd.DataFrame:
    cols = ["dataset_id", "event_id", "prediction_id", "prediction_status", "predicted_final_tier", "p_safe", "p_elevated", "p_critical",
            "latest_input_time_to_tca_days", "latest_miss_distance_m", "n_eligible_cdms", "n_excluded_cdms", "latest_observed_tier",
            "review_priority", "reason_codes", "quality_flags", "review_status", "model_version", "schema_version"]
    out = ev[[c for c in cols if c in ev.columns]].copy()
    for c in ("reason_codes", "quality_flags"):
        out[c] = out[c].map(lambda v: ";".join(v) if isinstance(v, (list, tuple)) else "")
    return out


def _open(event_id: int) -> None:
    state.select_event(event_id)


def render() -> None:
    view = state.current_view()
    st.title("Conjunction forecast review")
    st.caption(f"Which events are forecast to **finish Critical**? Forecasts use only CDMs at least {MIN_TIME_TO_TCA_DAYS:g} days "
               "before closest approach (TCA) and predict the tier of each event's last CDM.")
    dataset_banner(view)

    all_ev = state.events_with_review(view)
    ok = all_ev["prediction_status"] == "Successful"
    c = st.columns(5)
    with c[0]:
        kpi("Total events", f"{len(all_ev):,}", view.info.label)
    with c[1]:
        kpi("Valid predictions", f"{int(ok.sum()):,}", f"{ok.mean():.0%} of events" if len(all_ev) else "")
    with c[2]:
        n_crit = int((all_ev["predicted_final_tier"] == "Critical").sum())
        kpi("Predicted final tier Critical", f"{n_crit:,}", "Acknowledged events still count")
    with c[3]:
        kpi("Requiring review", f"{int(all_ev['needs_review'].sum()):,}", "Priority, watch, analyst or data review")
    with c[4]:
        kpi("Insufficient / invalid input", f"{int((~ok).sum()):,}", "Never shown as Safe")

    ev = state.filtered_events(view)
    show_outcome = False
    if view.outcomes is not None:
        show_outcome = st.toggle("Show observed final tier (evaluation only)", value=False,
                                 help="Outcomes are kept outside the prediction path. Shown here only for held-out evaluation.")
        if show_outcome:
            ev = evaluation_frame(ev, view.outcomes)

    top_l, top_r = st.columns([1, 1.25], gap="large")
    with top_l:
        st.subheader("Where forecasts move")
        st.caption("Tier of each event's latest eligible CDM against its predicted final tier. Cells off the diagonal are "
                   "forecast escalations (right of the diagonal) and de-escalations (left).")
        if ok.any():
            show_chart(charts.tier_transition(all_ev[ok]))
    with top_r:
        st.subheader("Review priority")
        counts = all_ev["review_priority"].value_counts()
        rows = [(p, int(counts.get(p, 0)), h) for p, h in PRIORITY_HELP.items() if counts.get(p, 0)]
        md = "| Priority | Events | Meaning |\n|---|---:|---|\n" + "\n".join(f"| **{p}** | {n:,} | {h} |" for p, n, h in rows)
        st.markdown(md)
        st.caption("Priorities describe the next analyst action. They come from the model's review policy and do not clear, "
                   "plan or authorise anything.")

    st.subheader("Review queue")
    st.caption(f"{len(ev):,} of {len(all_ev):,} events shown. Ordered by review priority, then probability of a Critical "
               "final tier, then latest eligible input closest to TCA. Select a row, then open it.")
    table = queue_table(ev, show_outcome=show_outcome)
    sel = st.dataframe(
        table, hide_index=True, width="stretch", height=min(520, 36 * (len(table) + 1) + 3), on_select="rerun", selection_mode="single-row",
        key=f"queue_{view.key}",
        column_config=queue_column_config())
    rows = sel.selection.rows if sel and sel.selection else []
    bcols = st.columns([1.2, 1, 3])
    if rows:
        eid = int(table.iloc[rows[0]]["Event ID"])
        with bcols[0]:
            if st.button(f"Open event {eid} →", type="primary", on_click=_open, args=(eid,)):
                state.go_to("detail")
    else:
        bcols[0].button("Open selected event →", disabled=True)
    with bcols[1]:
        st.download_button("Download CSV", export_frame(ev).to_csv(index=False).encode(), file_name=f"predictions_{view.key}.csv",
                           mime="text/csv", help="Filtered event predictions with model version and review status.")
