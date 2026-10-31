"""Event detail: available evidence, forecast, historical outcome (kept apart), forecast history,
explanation, geometry, review and export."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from .. import state
from ..components import charts
from ..components.status import show_chart, badge, chips, dataset_banner, panel
from ..components.tables import flags_list, inputs_table, missing_fields
from ..config import MIN_TIME_TO_TCA_DAYS
from ..formatting import GROUP_TEXT, PRIORITY_HELP, REASON_TEXT, days, km
from ..services.reports import event_report, report_markdown
from ..services.review_store import REVIEW_STATUSES

PLOT_CFG = {"displayModeBar": False}


def _label(row) -> str:
    tier = row["predicted_final_tier"] if isinstance(row["predicted_final_tier"], str) else row["prediction_status"]
    return f"Event {int(row['event_id'])} · {tier} · {row['review_priority']}"


def render() -> None:
    view = state.current_view()
    st.title("Event detail")
    dataset_banner(view)
    ev_all = state.filtered_events(view)
    if ev_all.empty:
        st.info("No events match the current filters.")
        return
    ids = ev_all["event_id"].astype(int).tolist()
    sel = st.session_state.get("selected_event_id")
    if sel not in ids:
        sel = ids[0]
    labels = {int(r.event_id): _label(r._asdict()) for r in ev_all[["event_id", "predicted_final_tier", "prediction_status", "review_priority"]].itertuples()}
    eid = st.selectbox("Event", ids, index=ids.index(sel), format_func=lambda i: labels[i], key=f"detail_pick_{view.key}")
    state.select_event(eid)
    ev = ev_all[ev_all["event_id"] == eid].iloc[0]
    snaps = view.event_snapshots(eid)
    inputs = view.event_inputs(eid)
    caps = view.capabilities
    ok = ev["prediction_status"] == "Successful"

    # ---------------------------------------------------------------- three separate concepts
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        excl = ev["n_excluded_cdms"]
        excl_txt = "" if excl is None or pd.isna(excl) else f"<div class='note'>{int(excl)} CDM(s) inside the 2-day cutoff were excluded and are not shown.</div>"
        body = (f"<div class='big'>Latest eligible CDM: {days(ev['latest_input_time_to_tca_days'])} before TCA</div>"
                f"<div class='note'>{int(ev['n_eligible_cdms'])} eligible CDM(s) · latest observed miss distance {km(ev['latest_miss_distance_m'], 3)}"
                f" · latest observed tier {ev['latest_observed_tier'] or '–'}</div>{excl_txt}") if ev["n_eligible_cdms"] else \
               f"<div class='big'>No eligible CDM</div><div class='note'>All {int(excl or 0)} CDM(s) are inside the 2-day cutoff.</div>"
        panel("Available evidence", body)
    with c2:
        if ok:
            body = (f"<div class='big'>Predicted final tier: {badge(ev['predicted_final_tier'])}</div>"
                    f"<div class='note'>Review priority: <b>{ev['review_priority']}</b> — {PRIORITY_HELP.get(ev['review_priority'], '')}</div>")
        else:
            body = (f"<div class='big'>{badge(None, ev['prediction_status'])}</div>"
                    f"<div class='note'>No forecast was made. {'; '.join(flags_list(ev['quality_flags']))}</div>")
        panel("Forecast", body)
    with c3:
        out = view.outcome(eid)
        if view.outcomes is None:
            body = "<div class='big'>No outcome available</div><div class='note'>Final outcomes are not provided for these events.</div>"
        elif out is None:
            body = "<div class='big'>Outcome not recorded</div>"
        else:
            body = (f"<div class='big'>Observed final tier: {badge(out['final_tier'])}</div>"
                    "<div class='note'><b>Evaluation only.</b> Not available to the model or to an analyst at prediction time."
                    + (" The event's final CDM was itself ≥ 2 days out, so the answer was visible in the input." if bool(out.get("is_trivial")) else "")
                    + "</div>")
        panel("Historical outcome", body)

    if ok and caps.class_probabilities and ev["p_critical"] is not None and not pd.isna(ev["p_critical"]):
        show_chart(charts.probability_bar(ev["p_safe"], ev["p_elevated"], ev["p_critical"]))
        st.caption("Calibrated model probability of each **final tier**. This is not a collision probability: an 80 % "
                   "probability of a Critical final tier is not an 80 % chance of collision.")
    if not caps.uncertainty_intervals:
        st.caption("Uncertainty interval unavailable — this model does not produce one.")

    snap_ids = snaps["row_id"].tolist()
    picked = snap_ids[-1] if snap_ids else None
    if len(snap_ids) > 1:
        tlabels = {r: f"{t:.2f} d before TCA ({n} CDM{'s' if n > 1 else ''})" for r, t, n in
                   zip(snaps["row_id"], snaps["time_to_tca_days"], snaps["n_eligible_so_far"])}
        picked = st.select_slider("Selected eligible CDM (used by the explanation and geometry tabs)", options=snap_ids,
                                  value=snap_ids[-1], format_func=lambda r: tlabels[r], key=f"snap_{view.key}_{eid}")

    tabs = st.tabs(["Forecast history", "Input evidence", "Explanation", "Encounter geometry", "Review & export"])
    with tabs[0]:
        if len(snaps) == 0:
            st.info("No eligible CDMs, so there is no forecast history.")
        else:
            if caps.snapshot_history and len(snaps) > 1 and snaps["p_critical"].notna().any():
                st.markdown("**Forecast history** — each point is the final-tier forecast made from only the CDMs received up to that "
                            "time (no later message is used).")
                show_chart(charts.probability_history(snaps, picked))
            else:
                st.info("Only one forecast exists for this event (a single eligible CDM), so there is no forecast trajectory. "
                        "Measurements are shown below.")
            st.markdown("**Measurement history** (observed values at each eligible CDM)")
            show_chart(charts.measurement_history(inputs, picked))
            st.caption("▼ marks a risk value of −30, the dataset placeholder for a probability too small to compute (Pc ≈ 0); "
                       "it is drawn at the bottom of the axis rather than at −30.")
            with st.expander("Forecast history as a table"):
                t = snaps[["time_to_tca_days", "n_eligible_so_far", "observed_tier", "predicted_final_tier", "p_safe", "p_elevated",
                           "p_critical", "prediction_status"]].rename(columns={
                    "time_to_tca_days": "Time to TCA (d)", "n_eligible_so_far": "CDMs used", "observed_tier": "Observed tier at CDM",
                    "predicted_final_tier": "Predicted final tier", "p_safe": "P(Safe)", "p_elevated": "P(Elevated)",
                    "p_critical": "P(Critical)", "prediction_status": "Status"})
                for c in ("P(Safe)", "P(Elevated)", "P(Critical)"):
                    t[c] = t[c] * 100
                st.dataframe(t, hide_index=True, width="stretch",
                             column_config={c: st.column_config.NumberColumn(format="%.1f%%") for c in ("P(Safe)", "P(Elevated)", "P(Critical)")})

    with tabs[1]:
        st.markdown(f"**{int(ev['n_eligible_cdms'])} eligible CDM(s)** at ≥ {MIN_TIME_TO_TCA_DAYS:g} days before TCA. "
                    "Distances are in metres in the source data and shown here in kilometres where marked.")
        miss = missing_fields(inputs)
        if miss:
            st.markdown("Missing fields: " + chips(miss), unsafe_allow_html=True)
        st.markdown("Quality flags on the forecast: " + chips(flags_list(ev["quality_flags"])), unsafe_allow_html=True)
        if len(inputs):
            st.dataframe(inputs_table(inputs), hide_index=True, width="stretch")
            st.caption("Risk = log10(Pc). The value −30 is the dataset's placeholder for a probability too small to compute (Pc ≈ 0), "
                       "not a measured 10⁻³⁰.")

    with tabs[2]:
        st.markdown("**Why the model forecasts this tier**")
        codes = list(ev["reason_codes"])
        if caps.reason_codes and codes:
            st.markdown(chips([REASON_TEXT.get(c, c) for c in codes]), unsafe_allow_html=True)
        srow = snaps[snaps["row_id"] == picked].iloc[0] if picked is not None and len(snaps) else None
        expl = view.explanation(srow["explanation_id"]) if srow is not None else None
        if not caps.feature_contributions:
            st.info(f"Feature contributions unavailable. {caps.feature_contributions_note or ''}")
        elif expl is None:
            st.info("No explanation was produced for this forecast.")
        else:
            st.caption(f"Contributions to the model's **Critical-class score** ({expl.get('output_units', 'log-odds')}) for the forecast made "
                       f"at {days(srow['time_to_tca_days'])} before TCA. Positive values raise the score. They describe associations "
                       "learned by the model, not causes, and do not explain a collision probability.")
            fig = charts.contribution_groups(expl.get("group_contributions"))
            if fig is not None:
                show_chart(fig)
            drivers = []
            for side in ("top_positive", "top_negative"):
                try:
                    drivers += json.loads(expl.get(side) or "[]")
                except ValueError:
                    pass
            if drivers:
                d = pd.DataFrame(drivers)
                d["group"] = d["group"].map(lambda g: GROUP_TEXT.get(g, g))
                d = d.reindex(d["contribution"].abs().sort_values(ascending=False).index)
                st.dataframe(d.rename(columns={"label": "Model input", "value": "Value", "contribution": "Contribution (log-odds)",
                                               "group": "Group"})[["Model input", "Value", "Contribution (log-odds)", "Group"]],
                             hide_index=True, width="stretch",
                             column_config={"Contribution (log-odds)": st.column_config.NumberColumn(format="%+.3f"),
                                            "Value": st.column_config.NumberColumn(format="%.3f")})
        st.caption("Latest observed measurements are supporting context. A small current miss distance does not by itself explain "
                   "why the model forecasts a Critical final tier.")

    with tabs[3]:
        st.markdown("**Predicted encounter geometry from the selected CDM** — position of the secondary object relative to the primary "
                    "satellite at the predicted closest approach, in metres, in the radial / transverse / normal frame.")
        if len(inputs) and inputs[["rel_pos_r_m", "rel_pos_t_m", "rel_pos_n_m"]].notna().any().any():
            show_chart(charts.geometry(inputs, picked))
            if picked is not None and (inputs["row_id"] == picked).any():
                r = inputs[inputs["row_id"] == picked].iloc[0]
                st.caption(f"Selected CDM ({days(r['time_to_tca_days'])} before TCA): R {r['rel_pos_r_m']:,.0f} m · "
                           f"T {r['rel_pos_t_m']:,.0f} m · N {r['rel_pos_n_m']:,.0f} m · miss distance {km(r['miss_distance_m'], 3)}.")
        else:
            st.info("No relative-position values available.")

    with tabs[4]:
        _review(view, ev)


def _review(view, ev) -> None:
    store = state.review_store()
    st.markdown("**Analyst review**")
    st.caption("Notes are stored locally and tied to this event and this specific prediction. Acknowledging an event does not "
               "change its predicted tier or remove it from any count.")
    with st.form(f"review_{view.key}_{int(ev['event_id'])}", clear_on_submit=True):
        c1, c2 = st.columns([1, 1])
        status = c1.selectbox("Review status", REVIEW_STATUSES[1:])
        reviewer = c2.text_input("Reviewer", placeholder="Name or initials")
        note = st.text_area("Note", placeholder="What did you check, and what did you conclude?")
        follow = st.text_input("Follow-up", placeholder="e.g. re-check after the next CDM")
        if st.form_submit_button("Save review", type="primary"):
            store.add(dataset_id=ev["dataset_id"], event_id=int(ev["event_id"]), prediction_id=ev["prediction_id"],
                      model_version=ev["model_version"], status=status, note=note, follow_up=follow, reviewer=reviewer,
                      predicted_final_tier=ev["predicted_final_tier"] if isinstance(ev["predicted_final_tier"], str) else None)
            st.success("Review saved.")
    hist = store.history(ev["dataset_id"], int(ev["event_id"]))
    if len(hist):
        h = hist.assign(prediction=lambda d: ["current" if p == ev["prediction_id"] else "earlier prediction" for p in d["prediction_id"]])
        st.dataframe(h[["created_at", "status", "reviewer", "note", "follow_up", "prediction", "model_version"]].rename(columns={
            "created_at": "Saved (UTC)", "status": "Status", "reviewer": "Reviewer", "note": "Note", "follow_up": "Follow-up",
            "prediction": "Prediction", "model_version": "Model"}), hide_index=True, width="stretch")
    else:
        st.caption("No reviews recorded yet.")
    rep = event_report(view, int(ev["event_id"]), hist)
    c1, c2, _ = st.columns([1, 1, 2])
    c1.download_button("Event report (Markdown)", report_markdown(rep).encode(), file_name=f"event_{int(ev['event_id'])}_report.md",
                       mime="text/markdown")
    c2.download_button("Event report (JSON)", json.dumps(rep, indent=2, default=str).encode(),
                       file_name=f"event_{int(ev['event_id'])}_report.json", mime="application/json")
