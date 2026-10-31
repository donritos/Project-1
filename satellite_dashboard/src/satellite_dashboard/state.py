"""Session state, cached loading and filters shared by every page."""
from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

from .formatting import quality_status
from .services.artifact_loader import ArtifactError
from .services.model_adapter import DatasetView, ModelAdapter
from .services.review_store import ReviewStore, review_status_column

DATASET_OPTIONS = {"heldout": "Held-out training events", "test": "Unlabelled test events", "synthetic": "Synthetic demo data"}
PAGES: dict = {}


@st.cache_resource(show_spinner=False)
def adapter() -> ModelAdapter:
    return ModelAdapter()


@st.cache_resource(show_spinner="Loading predictions…")
def _load(key: str) -> DatasetView:
    return adapter().load(key)


@st.cache_resource(show_spinner=False)
def review_store() -> ReviewStore:
    return ReviewStore()


def dataset_options() -> dict[str, str]:
    opts = dict(DATASET_OPTIONS)
    if st.session_state.get("upload_view") is not None:
        opts["upload"] = "Uploaded events (this session)"
    return opts


def current_view() -> DatasetView:
    key = st.session_state.get("dataset_key", "heldout")
    if key == "upload":
        v = st.session_state.get("upload_view")
        if v is None:
            st.session_state["dataset_key"] = "heldout"
            return current_view()
        return v
    try:
        return _load(key)
    except ArtifactError as exc:
        st.error(f"**Could not load {DATASET_OPTIONS.get(key, key)}.** {exc}")
        st.caption("Real artifacts never fall back to synthetic data. Fix the artifacts, or pick “Synthetic demo data” explicitly.")
        st.stop()


def events_with_review(view: DatasetView) -> pd.DataFrame:
    ev = view.events.copy()
    latest = review_store().latest_status(ev["dataset_id"].iloc[0]) if len(ev) else pd.DataFrame()
    ev["review_status"] = review_status_column(ev, latest) if len(ev) else []
    ev["data_quality"] = [quality_status(s, f) for s, f in zip(ev["prediction_status"], ev["quality_flags"])]
    return ev


def filtered_events(view: DatasetView) -> pd.DataFrame:
    ev = events_with_review(view)
    f = st.session_state
    tiers = f.get("flt_tier") or []
    if tiers:
        label = [t if isinstance(t, str) else s for t, s in zip(ev["predicted_final_tier"], ev["prediction_status"])]
        ev = ev[pd.Series(label, index=ev.index).isin(tiers)]
    pr = f.get("flt_priority") or []
    if pr:
        ev = ev[ev["review_priority"].isin(pr)]
    dq = f.get("flt_quality") or []
    if dq:
        ev = ev[ev["data_quality"].isin(dq)]
    if f.get("flt_needs_review"):
        ev = ev[ev["needs_review"]]
    q = str(f.get("flt_search") or "").strip()
    if q:
        ev = ev[ev["event_id"].astype(str).str.contains(q, regex=False)]
    return ev


def select_event(event_id: Optional[int]) -> None:
    st.session_state["selected_event_id"] = None if event_id is None else int(event_id)


def go_to(page: str) -> None:
    if page in PAGES:
        st.switch_page(PAGES[page])
