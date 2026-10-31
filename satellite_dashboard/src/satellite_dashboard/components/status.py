"""Banners, KPI cards and small status widgets."""
from __future__ import annotations

import streamlit as st

from ..formatting import INK, tier_badge

CSS = f"""
<style>
.block-container {{padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px;}}
.kpi {{border:1px solid #2e2e2b; border-radius:10px; padding:12px 14px; background:#1a1a19; height:100%;}}
.kpi .label {{color:{INK['secondary']}; font-size:0.8rem; letter-spacing:0.02em;}}
.kpi .value {{color:{INK['primary']}; font-size:1.7rem; font-weight:650; line-height:1.25; font-variant-numeric: tabular-nums;}}
.kpi .sub {{color:{INK['muted']}; font-size:0.78rem;}}
.panel {{border:1px solid #2e2e2b; border-radius:10px; padding:14px 16px; background:#1a1a19; height:100%;}}
.panel h4 {{margin:0 0 6px 0; font-size:0.78rem; color:{INK['secondary']}; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;}}
.panel .big {{font-size:1.15rem; color:{INK['primary']}; font-weight:600; margin:4px 0;}}
.panel .note {{font-size:0.82rem; color:{INK['muted']};}}
.chip {{display:inline-block; padding:1px 8px; margin:2px 4px 2px 0; border-radius:6px; background:#26261f; color:{INK['secondary']}; font-size:0.8rem;}}
.banner {{border-radius:8px; padding:8px 12px; margin-bottom:10px; font-size:0.88rem;}}
.banner.synthetic {{background:#3a2a00; border:1px solid #fab219; color:#fde7b0;}}
.banner.test {{background:#14243a; border:1px solid #3987e5; color:#cde2fb;}}
.banner.heldout {{background:#1c2a1c; border:1px solid #0ca30c; color:#cfe8cf;}}
.banner.upload {{background:#2a1f2e; border:1px solid #9085e9; color:#e2dcf7;}}
</style>
"""


def _plotly_takes_width() -> bool:
    import inspect
    return "width" in inspect.signature(st.plotly_chart).parameters


_PLOTLY_WIDTH = _plotly_takes_width()


def show_chart(fig, **kw) -> None:
    """Full-width Plotly chart on any supported Streamlit version (no toolbar)."""
    kw.setdefault("config", {"displayModeBar": False})
    if _PLOTLY_WIDTH:
        st.plotly_chart(fig, width="stretch", **kw)
    else:
        st.plotly_chart(fig, use_container_width=True, **kw)


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def dataset_banner(view) -> None:
    if view.kind == "synthetic":
        st.markdown("<div class='banner synthetic'><b>Synthetic demo data.</b> Hand-made fixtures for testing the interface. "
                    "Nothing on this screen is model output or a measured result.</div>", unsafe_allow_html=True)
    elif view.kind == "upload":
        st.markdown(f"<div class='banner upload'><b>Uploaded events.</b> {view.info.note}</div>", unsafe_allow_html=True)
    elif view.info.has_outcomes:
        st.markdown("<div class='banner heldout'><b>Held-out training events.</b> The model never trained on these events. "
                    "Observed final tiers are shown only where marked <i>evaluation only</i>.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='banner test'><b>Unlabelled test events.</b> No final outcomes exist for these events; "
                    "predictions are for demonstration.</div>", unsafe_allow_html=True)


def kpi(label: str, value: str, sub: str = "") -> None:
    st.markdown(f"<div class='kpi'><div class='label'>{label}</div><div class='value'>{value}</div>"
                f"<div class='sub'>{sub}</div></div>", unsafe_allow_html=True)


def panel(title: str, body_html: str) -> None:
    st.markdown(f"<div class='panel'><h4>{title}</h4>{body_html}</div>", unsafe_allow_html=True)


def chips(items: list[str]) -> str:
    return "".join(f"<span class='chip'>{i}</span>" for i in items) if items else "<span class='note'>None</span>"


def badge(tier, text=None) -> str:
    return tier_badge(tier, text)
