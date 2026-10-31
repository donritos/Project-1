"""Conjunction forecast review dashboard.  Run:  streamlit run app.py"""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Conjunction Forecast Review", page_icon=":material/satellite_alt:", layout="wide")

from satellite_dashboard import state  # noqa: E402
from satellite_dashboard.components.status import inject_css  # noqa: E402
from satellite_dashboard.config import MIN_TIME_TO_TCA_DAYS  # noqa: E402
from satellite_dashboard.contracts import ReviewPriority  # noqa: E402
from satellite_dashboard.views import event_detail, model_evidence, overview, upload  # noqa: E402

inject_css()

PAGES = {
    "overview": st.Page(overview.render, title="Overview", icon=":material/list_alt:", url_path="overview", default=True),
    "detail": st.Page(event_detail.render, title="Event detail", icon=":material/troubleshoot:", url_path="event"),
    "evidence": st.Page(model_evidence.render, title="Model evidence", icon=":material/fact_check:", url_path="evidence"),
    "upload": st.Page(upload.render, title="Upload & predict", icon=":material/upload_file:", url_path="upload"),
}
state.PAGES.update(PAGES)
nav = st.navigation(list(PAGES.values()))

with st.sidebar:
    opts = state.dataset_options()
    if st.session_state.get("dataset_key") not in opts:
        st.session_state["dataset_key"] = "heldout"
    st.radio("Dataset", list(opts), format_func=opts.get, key="dataset_key")
    view = state.current_view()
    st.markdown(f"**Model version**  \n`{view.model_version}`")
    st.markdown(f"**Eligibility rule**  \nOnly CDMs with `time_to_tca ≥ {MIN_TIME_TO_TCA_DAYS:g} days` reach the model.")
    st.divider()
    st.markdown("**Filters**")
    st.text_input("Search event ID", key="flt_search", placeholder="e.g. 849")
    st.multiselect("Predicted final tier", ["Critical", "Elevated", "Safe", "Invalid input", "Insufficient eligible data"], key="flt_tier")
    st.multiselect("Review priority", [p.value for p in ReviewPriority], key="flt_priority")
    st.multiselect("Data quality", ["OK", "Warnings", "Invalid input", "Insufficient eligible data"], key="flt_quality")
    st.toggle("Only events needing review", key="flt_needs_review")
    st.caption("Decision support only. Review priorities describe analyst actions; nothing here calculates or authorises a manoeuvre.")

nav.run()
