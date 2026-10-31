"""Plotly figures. One y-scale per panel (no dual axes); tier colours always come with text labels;
time to TCA runs right-to-left in value (8 -> 2 days) so that time advances toward TCA from left to right."""
from __future__ import annotations

import json
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..config import (CRITICAL_DISTANCE_M, CRITICAL_RISK, ELEVATED_DISTANCE_MAX_M, ELEVATED_RISK, MIN_TIME_TO_TCA_DAYS,
                      RISK_FLOOR)
from ..contracts import TIER_ORDER
from ..formatting import GROUP_TEXT, INK, SERIES_1, SERIES_2, TIER_COLORS

FONT = dict(family="Inter, Segoe UI, system-ui, sans-serif", size=12, color=INK["secondary"])
SEQ_BLUE = ["#1a1a19", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4"]
RISK_AXIS_MIN = -12.0


def _base(fig: go.Figure, height: int, *, legend: bool = True) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=8, r=16, t=36, b=8), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=FONT, showlegend=legend,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=INK["secondary"])),
                      hoverlabel=dict(bgcolor="#26261f", bordercolor="#4a4a46", font=dict(color=INK["primary"])))
    fig.update_xaxes(gridcolor=INK["grid"], linecolor=INK["axis"], zeroline=False, tickfont=dict(color=INK["muted"]))
    fig.update_yaxes(gridcolor=INK["grid"], linecolor=INK["axis"], zeroline=False, tickfont=dict(color=INK["muted"]))
    return fig


def _time_axis(fig: go.Figure, tmax: float, **kw) -> None:
    hi = max(tmax + 0.3, MIN_TIME_TO_TCA_DAYS + 1.0)
    fig.update_xaxes(range=[hi, MIN_TIME_TO_TCA_DAYS - 0.15], title_text="Time to TCA (days) — advancing toward TCA →",
                     title_font=dict(color=INK["secondary"], size=12), **kw)


def _cutoff(fig: go.Figure, row=None, col=None, label=True) -> None:
    kw = {} if row is None else dict(row=row, col=col)
    fig.add_vline(x=MIN_TIME_TO_TCA_DAYS, line=dict(color=INK["muted"], dash="dot", width=1), **kw)
    if label:
        fig.add_annotation(x=MIN_TIME_TO_TCA_DAYS, y=0.5, yref="paper", text="2-day input cutoff", showarrow=False, textangle=-90,
                           xanchor="left", yanchor="middle", xshift=2, font=dict(color=INK["muted"], size=10))


def probability_bar(p_safe: float, p_elev: float, p_crit: float) -> go.Figure:
    fig = go.Figure()
    for name, v in (("Safe", p_safe), ("Elevated", p_elev), ("Critical", p_crit)):
        fig.add_trace(go.Bar(y=["Final tier"], x=[v], name=name, orientation="h", marker=dict(color=TIER_COLORS[name],
                             line=dict(color="#141413", width=2)), text=[f"{name} {v:.0%}" if v >= 0.12 else ""],
                             textposition="inside", insidetextanchor="middle", textfont=dict(color="#141413", size=12),
                             hovertemplate=f"{name}: %{{x:.1%}}<extra></extra>"))
    fig.update_layout(barmode="stack", bargap=0.1, legend_traceorder="normal")
    fig.update_xaxes(range=[0, 1], tickformat=".0%", showgrid=False)
    fig.update_yaxes(showticklabels=False)
    _base(fig, 128, legend=True)
    fig.update_layout(margin=dict(l=8, r=24, t=28, b=24))
    return fig


def probability_history(snaps: pd.DataFrame, selected_row: Optional[str] = None) -> go.Figure:
    s = snaps.sort_values("time_to_tca_days", ascending=False)
    fig = go.Figure()
    for tier, col in (("Safe", "p_safe"), ("Elevated", "p_elevated"), ("Critical", "p_critical")):
        fig.add_trace(go.Scatter(
            x=s["time_to_tca_days"], y=s[col], name=f"P(final = {tier})", mode="lines+markers",
            line=dict(color=TIER_COLORS[tier], width=2), marker=dict(size=8, color=TIER_COLORS[tier], line=dict(color="#141413", width=2)),
            customdata=np.c_[s["n_eligible_so_far"], s["predicted_final_tier"].fillna("no forecast")],
            hovertemplate=f"{tier}: %{{y:.1%}}<br>%{{x:.2f}} d before TCA · CDMs used: %{{customdata[0]}}<br>"
                          "Forecast: %{customdata[1]}<extra></extra>"))
    # one direct label: the forecast tier at the latest snapshot (the legend names all three lines)
    last = s[s["p_critical"].notna()]
    if len(last):
        last = last.iloc[-1]
        tier = last["predicted_final_tier"] if isinstance(last["predicted_final_tier"], str) else None
        if tier:
            col = {"Safe": "p_safe", "Elevated": "p_elevated", "Critical": "p_critical"}[tier]
            fig.add_annotation(x=last["time_to_tca_days"], y=last[col], text=f"Forecast {tier} {last[col]:.0%}", showarrow=False,
                               xanchor="right", yanchor="top" if last[col] > 0.5 else "bottom", xshift=-8, yshift=-6 if last[col] > 0.5 else 6,
                               font=dict(color=INK["primary"], size=11))
    if selected_row is not None and (s["row_id"] == selected_row).any():
        x = float(s.loc[s["row_id"] == selected_row, "time_to_tca_days"].iloc[0])
        fig.add_vline(x=x, line=dict(color=INK["secondary"], width=1))
    _time_axis(fig, float(s["time_to_tca_days"].max()))
    fig.update_yaxes(range=[-0.02, 1.02], tickformat=".0%", title_text="Model probability of final tier",
                     title_font=dict(color=INK["secondary"], size=12))
    _cutoff(fig)
    return _base(fig, 320)


def measurement_history(inputs: pd.DataFrame, selected_row: Optional[str] = None) -> go.Figure:
    s = inputs.sort_values("time_to_tca_days", ascending=False)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
                        subplot_titles=("Observed miss distance (km)", "Observed risk, log10 Pc"))
    fig.add_trace(go.Scatter(x=s["time_to_tca_days"], y=s["miss_distance_m"] / 1000, name="Miss distance", mode="lines+markers",
                             line=dict(color=SERIES_1, width=2), marker=dict(size=8, color=SERIES_1, line=dict(color="#141413", width=2)),
                             hovertemplate="Miss distance %{y:.3f} km<br>%{x:.2f} d before TCA<extra></extra>", showlegend=False), row=1, col=1)
    for y, lab in ((CRITICAL_DISTANCE_M / 1000, "1 km (Critical)"), (ELEVATED_DISTANCE_MAX_M / 1000, "5 km (Elevated)")):
        fig.add_hline(y=y, line=dict(color=INK["muted"], dash="dash", width=1), row=1, col=1)
        fig.add_annotation(x=0, xref="x domain", y=np.log10(y), yref="y", text=lab, showarrow=False, xanchor="left",
                           yanchor="bottom", font=dict(color=INK["muted"], size=10))
    real_vals = pd.concat([s[c] for c in ("risk_log10", "max_risk_estimate_log10") if c in s.columns]).astype(float)
    real_vals = real_vals[real_vals > RISK_FLOOR + 1e-9]
    axis_min = min(RISK_AXIS_MIN, float(np.floor(real_vals.min())) - 1.0) if len(real_vals) else RISK_AXIS_MIN
    for col, name, color in (("risk_log10", "Risk (log10 Pc)", SERIES_1), ("max_risk_estimate_log10", "Max risk estimate (log10 Pc)", SERIES_2)):
        if col not in s.columns or s[col].isna().all():
            continue
        v = s[col].astype(float)
        floor = v <= RISK_FLOOR + 1e-9
        real = v.where(~floor)  # gaps at placeholder values, so no line is drawn into the floor
        fig.add_trace(go.Scatter(x=s["time_to_tca_days"], y=real, name=name, mode="lines+markers", connectgaps=False,
                                 line=dict(color=color, width=2), marker=dict(size=8, color=color, line=dict(color="#141413", width=2)),
                                 legendgroup=col, hovertemplate=f"{name}: %{{y:.2f}}<br>%{{x:.2f}} d before TCA<extra></extra>"),
                      row=2, col=1)
        if floor.any():
            fig.add_trace(go.Scatter(x=s.loc[floor, "time_to_tca_days"], y=[axis_min] * int(floor.sum()), mode="markers",
                                     name=f"{name}: placeholder −30", legendgroup=col, showlegend=False,
                                     marker=dict(size=9, symbol="triangle-down", color=color, line=dict(color="#141413", width=1)),
                                     hovertemplate=f"{name}: ≈ 0 (placeholder −30)<br>%{{x:.2f}} d before TCA<extra></extra>"),
                          row=2, col=1)
    for y, lab, ya in ((CRITICAL_RISK, "−4 (Critical)", "bottom"), (ELEVATED_RISK, "−6 (Elevated)", "top")):
        fig.add_hline(y=y, line=dict(color=INK["muted"], dash="dash", width=1), row=2, col=1)
        fig.add_annotation(x=0, xref="x2 domain", y=y, yref="y2", text=lab, showarrow=False, xanchor="left", yanchor=ya,
                           font=dict(color=INK["muted"], size=10))
    if selected_row is not None and (s["row_id"] == selected_row).any():
        x = float(s.loc[s["row_id"] == selected_row, "time_to_tca_days"].iloc[0])
        for r in (1, 2):
            fig.add_vline(x=x, line=dict(color=INK["secondary"], width=1), row=r, col=1)
    fig.update_yaxes(type="log", row=1, col=1, tickvals=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100],
                     ticktext=["0.01", "0.02", "0.05", "0.1", "0.2", "0.5", "1", "2", "5", "10", "20", "50", "100"])
    fig.update_yaxes(range=[axis_min - 0.6, 0], row=2, col=1)
    tmax = float(s["time_to_tca_days"].max()) if len(s) else MIN_TIME_TO_TCA_DAYS + 1
    hi = max(tmax + 0.3, MIN_TIME_TO_TCA_DAYS + 1.0)
    fig.update_xaxes(range=[hi, MIN_TIME_TO_TCA_DAYS - 0.15])
    fig.update_xaxes(title_text="Time to TCA (days) — advancing toward TCA →", row=2, col=1, title_font=dict(color=INK["secondary"], size=12))
    for r in (1, 2):
        fig.add_vline(x=MIN_TIME_TO_TCA_DAYS, line=dict(color=INK["muted"], dash="dot", width=1), row=r, col=1)
    fig.update_annotations(font=dict(color=INK["secondary"], size=12), selector=dict(text="Observed miss distance (km)"))
    fig.update_annotations(font=dict(color=INK["secondary"], size=12), selector=dict(text="Observed risk, log10 Pc"))
    _base(fig, 470)
    fig.update_layout(margin=dict(l=8, r=16, t=48, b=8),
                      legend=dict(orientation="h", x=1, xanchor="right", y=0.455, yanchor="bottom", font=dict(color=INK["secondary"], size=11)))
    return fig


def contribution_groups(group_json: Optional[str]) -> Optional[go.Figure]:
    if not group_json:
        return None
    try:
        g = json.loads(group_json)
    except (TypeError, ValueError):
        return None
    items = sorted(((GROUP_TEXT.get(k, k), float(v)) for k, v in g.items() if abs(float(v)) >= 0.005), key=lambda kv: kv[1])
    if not items:
        return None
    labels, vals = zip(*items)
    colors = ["#e66767" if v > 0 else SERIES_1 for v in vals]
    fig = go.Figure(go.Bar(x=list(vals), y=list(labels), orientation="h", marker=dict(color=colors, line=dict(color="#141413", width=2)),
                           text=[f"{v:+.2f}" for v in vals], textposition="outside", textfont=dict(color=INK["secondary"]),
                           hovertemplate="%{y}: %{x:+.3f} log-odds<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=INK["axis"], width=1))
    pad = max(abs(min(vals)), abs(max(vals))) * 1.35 + 0.1
    fig.update_xaxes(range=[-pad, pad], title_text="← lowers Critical score · raises Critical score → (log-odds)",
                     title_font=dict(color=INK["secondary"], size=12))
    return _base(fig, 60 + 34 * len(items), legend=False)


def geometry(inputs: pd.DataFrame, selected_row: Optional[str]) -> go.Figure:
    s = inputs.dropna(subset=["rel_pos_r_m", "rel_pos_t_m", "rel_pos_n_m"]).sort_values("time_to_tca_days", ascending=False)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Radial vs transverse (m)", "Normal vs transverse (m)"), horizontal_spacing=0.12)
    sel = s[s["row_id"] == selected_row] if selected_row is not None else s.iloc[0:0]
    other = s[s["row_id"] != selected_row] if selected_row is not None else s
    for col, (ycol, ylab) in enumerate((("rel_pos_r_m", "Radial"), ("rel_pos_n_m", "Normal")), start=1):
        fig.add_trace(go.Scatter(x=other["rel_pos_t_m"], y=other[ycol], mode="markers", name="Other eligible CDMs",
                                 marker=dict(size=8, color=INK["muted"], line=dict(color="#141413", width=2)), showlegend=col == 1,
                                 customdata=other["time_to_tca_days"],
                                 hovertemplate=f"T %{{x:,.0f}} m · {ylab} %{{y:,.0f}} m<br>%{{customdata:.2f}} d before TCA<extra></extra>"),
                      row=1, col=col)
        if len(sel):
            fig.add_trace(go.Scatter(x=sel["rel_pos_t_m"], y=sel[ycol], mode="markers", name="Selected CDM",
                                     marker=dict(size=13, color=SERIES_1, line=dict(color="#141413", width=2)), showlegend=col == 1,
                                     customdata=sel["time_to_tca_days"],
                                     hovertemplate=f"Selected · T %{{x:,.0f}} m · {ylab} %{{y:,.0f}} m<br>%{{customdata:.2f}} d before TCA<extra></extra>"),
                          row=1, col=col)
        fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers+text", name="Primary satellite", text=["Primary"], textposition="top center",
                                 marker=dict(size=10, symbol="cross", color=INK["primary"]), textfont=dict(color=INK["secondary"], size=10),
                                 showlegend=col == 1, hovertemplate="Primary satellite (origin)<extra></extra>"), row=1, col=col)
        fig.update_xaxes(title_text="Transverse (m)", row=1, col=col, title_font=dict(color=INK["secondary"], size=11))
        fig.update_yaxes(title_text=f"{ylab} (m)", row=1, col=col, title_font=dict(color=INK["secondary"], size=11))
    fig.update_annotations(font=dict(color=INK["secondary"], size=12))
    # equal metres on both axes so the geometry is not distorted
    fig.update_yaxes(scaleanchor="x", scaleratio=1, constrain="domain", row=1, col=1)
    fig.update_yaxes(scaleanchor="x2", scaleratio=1, constrain="domain", row=1, col=2)
    fig.update_xaxes(constrain="domain")
    _base(fig, 420)
    fig.update_layout(legend=dict(orientation="h", y=-0.22, yanchor="top", x=0, xanchor="left"), margin=dict(l=8, r=16, t=40, b=60))
    return fig


def confusion_heatmap(cm: pd.DataFrame, title: str) -> go.Figure:
    z = cm.to_numpy().astype(float)
    row_tot = z.sum(axis=1, keepdims=True)
    share = np.divide(z, row_tot, out=np.zeros_like(z), where=row_tot > 0)
    text = [[f"{int(z[i, j])}<br>{share[i, j]:.0%}" for j in range(3)] for i in range(3)]
    fig = go.Figure(go.Heatmap(z=share, x=TIER_ORDER, y=TIER_ORDER, text=text, texttemplate="%{text}", colorscale=[
        [0, SEQ_BLUE[0]], [0.25, SEQ_BLUE[1]], [0.5, SEQ_BLUE[2]], [0.75, SEQ_BLUE[3]], [1, SEQ_BLUE[4]]], zmin=0, zmax=1,
        showscale=False, xgap=2, ygap=2, textfont=dict(color=INK["primary"], size=13),
        hovertemplate="Observed %{y} → predicted %{x}<br>%{text}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", title_text="Observed final tier", title_font=dict(color=INK["secondary"]))
    fig.update_xaxes(title_text="Predicted final tier", side="bottom", title_font=dict(color=INK["secondary"]))
    fig.update_layout(title=dict(text=title, font=dict(color=INK["primary"], size=14), x=0, xanchor="left"))
    _base(fig, 330, legend=False)
    fig.update_layout(margin=dict(l=8, r=8, t=44, b=8))
    return fig


def tier_transition(events: pd.DataFrame) -> go.Figure:
    ok = events[events["predicted_final_tier"].notna() & events["latest_observed_tier"].notna()]
    cm = pd.crosstab(pd.Categorical(ok["latest_observed_tier"], TIER_ORDER), pd.Categorical(ok["predicted_final_tier"], TIER_ORDER),
                     dropna=False).reindex(index=TIER_ORDER, columns=TIER_ORDER, fill_value=0)
    z = cm.to_numpy().astype(float)
    fig = go.Figure(go.Heatmap(z=np.log1p(z), x=TIER_ORDER, y=TIER_ORDER, text=z.astype(int), texttemplate="%{text}", showscale=False,
                               colorscale=[[0, SEQ_BLUE[0]], [0.5, SEQ_BLUE[2]], [1, SEQ_BLUE[4]]], xgap=2, ygap=2,
                               textfont=dict(color=INK["primary"], size=13),
                               hovertemplate="Latest observed %{y} → forecast %{x}: %{text} events<extra></extra>"))
    fig.update_yaxes(autorange="reversed", title_text="Latest eligible CDM tier", title_font=dict(color=INK["secondary"], size=11))
    fig.update_xaxes(title_text="Predicted final tier", title_font=dict(color=INK["secondary"], size=11))
    _base(fig, 260, legend=False)
    fig.update_layout(margin=dict(l=8, r=8, t=10, b=8))
    return fig


def reliability_chart(rel: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=INK["muted"], dash="dash", width=1),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=rel["mean_predicted"], y=rel["observed"], mode="lines+markers", line=dict(color=SERIES_1, width=2),
                             marker=dict(size=np.clip(np.sqrt(rel["events"]) * 1.6, 8, 22), color=SERIES_1, line=dict(color="#141413", width=2)),
                             customdata=np.c_[rel["events"], rel["bin_label"]], showlegend=False,
                             hovertemplate="Bin %{customdata[1]}<br>Mean predicted %{x:.1%} · observed %{y:.1%}<br>%{customdata[0]} events<extra></extra>"))
    fig.add_annotation(x=0.98, y=0.9, text="perfect calibration", showarrow=False, xanchor="right", font=dict(color=INK["muted"], size=10))
    fig.update_xaxes(range=[0, 1], tickformat=".0%", title_text="Predicted P(final = Critical)", title_font=dict(color=INK["secondary"]))
    fig.update_yaxes(range=[0, 1], tickformat=".0%", title_text="Observed share Critical", title_font=dict(color=INK["secondary"]))
    return _base(fig, 330, legend=False)
