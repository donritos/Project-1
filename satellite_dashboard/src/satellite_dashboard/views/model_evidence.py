"""Held-out evaluation: one final-tier prediction per held-out event against its observed final tier."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from .. import state
from ..components import charts
from ..components.status import show_chart, dataset_banner, kpi
from ..config import ARTIFACTS_DIR, MIN_TIME_TO_TCA_DAYS
from ..services.metrics import brier, confusion, evaluation_frame, reliability, tier_metrics

PLOT_CFG = {"displayModeBar": False}
SUBSETS = {
    "real": "Real forecasts — final CDM < 2 days before TCA",
    "all": "All held-out events",
    "trivial": "Answer visible in input — final CDM itself ≥ 2 days out",
}


def _fmt(v) -> str:
    return "–" if v is None else f"{v:.3f}"


def _open(eid: int) -> None:
    state.select_event(eid)


def render() -> None:
    view = state.current_view()
    st.title("Model evidence")
    dataset_banner(view)
    if view.outcomes is None:
        st.info("Performance can only be measured on events with known outcomes. Choose **Held-out training events** in the sidebar. "
                "Test events have no outcomes, so no test-set accuracy is reported anywhere in this app.")
        return
    if view.kind == "synthetic":
        st.warning("These numbers are computed on hand-made fixtures. They are **not** measured model performance.")

    m = view.manifest
    ef = evaluation_frame(view.events, view.outcomes)
    usable = ef["prediction_status"] == "Successful"
    n_unusable_split = (m.split or {}).get("n_unusable_excluded")
    c = st.columns(5)
    with c[0]:
        kpi("Held-out events", f"{len(ef):,}", "Never used for training, tuning or calibration")
    with c[1]:
        cc = ef["observed_final_tier"].value_counts()
        kpi("Observed final tier Critical", f"{int(cc.get('Critical', 0)):,}",
            f"Elevated {int(cc.get('Elevated', 0)):,} · Safe {int(cc.get('Safe', 0)):,}")
    with c[2]:
        kpi("Without a usable prediction", f"{int((~usable).sum()):,}", "Invalid or insufficient input (excluded from metrics)")
    with c[3]:
        kpi("Input horizon", f"≥ {MIN_TIME_TO_TCA_DAYS:g} days", "Latest eligible CDM per event")
    with c[4]:
        kpi("Model", m.model.get("kind", "–"), m.model_version)
    st.caption(f"Split: {m.split.get('method', '–')} (split hash `{m.split.get('split_hash', '–')}`)."
               + (f" {n_unusable_split:,} training events had no CDM at ≥ 2 days and were excluded before splitting." if n_unusable_split else ""))

    key = st.radio("Events evaluated", list(SUBSETS), format_func=SUBSETS.get, horizontal=True, key="ev_subset")
    sub = ef[usable].copy()
    if key == "real":
        sub = sub[~sub["is_trivial"].astype(bool)]
    elif key == "trivial":
        sub = sub[sub["is_trivial"].astype(bool)]
    if sub.empty:
        st.info("No events in this subset.")
        return
    y = sub["observed_final_tier"]
    model = tier_metrics(y, sub["predicted_final_tier"])
    base = tier_metrics(y, sub["baseline_tier"])
    st.markdown(f"**{len(sub):,} events** · observed final tiers: "
                + ", ".join(f"{t} {int((y == t).sum())}" for t in ("Critical", "Elevated", "Safe")))

    rows = [("Critical recall", "critical_recall"), ("Critical precision", "critical_precision"), ("Critical F2", "critical_f2"),
            ("Elevated-or-above F2", "elevated_plus_f2"), ("Macro-F1", "macro_f1"), ("Accuracy", "accuracy")]
    tbl = pd.DataFrame({"Metric": [r[0] for r in rows],
                        "Baseline (carry forward)": [_fmt(base[k]) for _, k in rows],
                        "Final-tier model": [_fmt(model[k]) for _, k in rows]})
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.subheader("Baseline versus model")
        st.dataframe(tbl, hide_index=True, width="stretch")
        st.caption("Baseline: the tier of each event's latest eligible CDM, carried forward unchanged to TCA.")
        st.caption("**Critical F2** treats Critical versus the other two tiers as a binary problem and weights recall twice as much "
                   "as precision: F2 = 5·P·R / (4·P + R). A missed Critical event costs more than a false alarm.")
        if key == "trivial":
            st.caption("On these events the latest eligible CDM *is* the final CDM, so carrying it forward is exact by construction.")
    with right:
        st.subheader("Escalations")
        not_crit = sub[sub["latest_observed_tier"] != "Critical"]
        esc = not_crit[not_crit["observed_final_tier"] == "Critical"]
        caught = esc[esc["predicted_final_tier"] == "Critical"]
        watched = esc[(esc["predicted_final_tier"] != "Critical") & (esc["review_priority"] == "Escalation watch")]
        st.markdown(f"{len(esc)} event(s) were **not** Critical at their latest eligible CDM but ended Critical. "
                    f"The baseline misses all of them by definition.")
        cc = st.columns(3)
        with cc[0]:
            kpi("Forecast Critical", f"{len(caught)}", "of escalations")
        with cc[1]:
            kpi("On escalation watch", f"{len(watched)}", "flagged, not Critical")
        with cc[2]:
            kpi("Not flagged", f"{len(esc) - len(caught) - len(watched)}", "missed")

    st.subheader("Confusion matrices")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        show_chart(charts.confusion_heatmap(confusion(y, sub["baseline_tier"]), "Baseline (carry forward)"))
    with c2:
        show_chart(charts.confusion_heatmap(confusion(y, sub["predicted_final_tier"]), "Final-tier model"))
    st.caption("Counts and share of each observed tier (rows sum to 100 %).")

    if view.capabilities.class_probabilities and sub["p_critical"].notna().any():
        st.subheader("Calibration of P(final = Critical)")
        is_c = (y == "Critical").astype(float)
        rel = reliability(sub["p_critical"], is_c)
        c1, c2 = st.columns([1.3, 1], gap="large")
        with c1:
            show_chart(charts.reliability_chart(rel))
        with c2:
            b = brier(sub["p_critical"], is_c)
            st.markdown(f"Brier score (Critical vs not): **{_fmt(b)}**")
            rel_t = rel.assign(mean_predicted=rel["mean_predicted"] * 100, observed=rel["observed"] * 100)
            st.dataframe(rel_t[["bin_label", "mean_predicted", "observed", "events"]].rename(columns={
                "bin_label": "Predicted bin", "mean_predicted": "Mean predicted", "observed": "Observed share", "events": "Events"}),
                hide_index=True, width="stretch",
                column_config={"Mean predicted": st.column_config.NumberColumn(format="%.1f%%"),
                               "Observed share": st.column_config.NumberColumn(format="%.1f%%")})
            st.caption("Bins with few events are noisy. Points above the diagonal mean the model under-states the chance of a Critical "
                       "final tier; below means it over-states it.")

    st.subheader("Failure cases")
    st.caption("Genuine errors on held-out events, so you can judge where the model fails — not only where it succeeds.")
    missed = sub[(y == "Critical") & (sub["predicted_final_tier"] != "Critical")].sort_values("p_critical")
    false_alarm = sub[(y != "Critical") & (sub["predicted_final_tier"] == "Critical")].sort_values("p_critical", ascending=False)
    incomplete = ef[~usable | ef["quality_flags"].map(len).gt(0) | ef["n_eligible_cdms"].eq(1)]
    t1, t2, t3 = st.tabs([f"Missed Critical ({len(missed)})", f"False alarms ({len(false_alarm)})", f"Incomplete input ({len(incomplete)})"])
    for i, (tab, frame, note) in enumerate(((t1, missed, "Observed Critical, forecast lower."), (t2, false_alarm, "Forecast Critical, observed lower."),
                             (t3, incomplete, "Invalid input, quality warnings, or a single eligible CDM."))):
        with tab:
            st.caption(note)
            if frame.empty:
                st.info("None in this subset.")
                continue
            show = frame.head(200)
            disp = pd.DataFrame({"Event ID": show["event_id"].astype(int),
                                 "Predicted final tier": show["predicted_final_tier"].fillna(show["prediction_status"]),
                                 "Observed final tier (evaluation only)": show["observed_final_tier"],
                                 "P(final = Critical)": show["p_critical"] * 100,
                                 "Latest eligible input (d)": show["latest_input_time_to_tca_days"],
                                 "Latest observed miss (km)": show["latest_miss_distance_m"] / 1000,
                                 "Latest observed tier": show["latest_observed_tier"], "Eligible CDMs": show["n_eligible_cdms"],
                                 "Review priority": show["review_priority"]})
            st.dataframe(disp, hide_index=True, width="stretch", height=min(38 * (len(disp) + 1), 360),
                         column_config={"P(final = Critical)": st.column_config.NumberColumn(format="%.1f%%"),
                                        "Latest eligible input (d)": st.column_config.NumberColumn(format="%.2f"),
                                        "Latest observed miss (km)": st.column_config.NumberColumn(format="%.3f")})
            pick = st.selectbox("Open an event", disp["Event ID"].tolist(), key=f"fail_pick_{key}_{i}")
            if st.button("Open in event detail →", key=f"fail_open_{key}_{i}", on_click=_open, args=(int(pick),)):
                state.go_to("detail")

    if view.kind == "real":
        _crosscheck(sub, key, model)


def _crosscheck(sub: pd.DataFrame, key: str, model: dict) -> None:
    path = ARTIFACTS_DIR / "evaluation.json"
    if not path.exists():
        return
    try:
        ev = json.loads(path.read_text())
        sname = {"real": "real_forecast", "all": "all_usable", "trivial": "trivial"}[key]
        primary = ev.get("primary_model")
        ref = ev["models"][primary]["subsets"][sname]["classification"]
    except Exception:
        return
    rows = []
    for lab, k_app, k_ref in (("Critical recall", "critical_recall", "critical_recall"),
                              ("Critical precision", "critical_precision", "critical_precision"),
                              ("Critical F2", "critical_f2", "critical_f2"), ("Elevated-or-above F2", "elevated_plus_f2", "elevated_or_above_f2"),
                              ("Macro-F1", "macro_f1", "macro_f1"), ("Accuracy", "accuracy", "accuracy")):
        a, b = model.get(k_app), ref.get(k_ref) if isinstance(ref, dict) else None
        if isinstance(b, (int, float)):
            rows.append((lab, a, b, abs((a or 0) - b) < 1e-6))
    if rows:
        with st.expander("Cross-check against the model's own evaluation report"):
            st.dataframe(pd.DataFrame(rows, columns=["Metric", "Computed here", "evaluation.json", "Match"]), hide_index=True)
