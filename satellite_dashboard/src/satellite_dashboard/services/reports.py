"""Structured event report (JSON and Markdown): inputs used, prediction, explanation, model version, notes."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from ..formatting import FLAG_TEXT, REASON_TEXT, days, km, log10_pc, pct


def _clean(v):
    if isinstance(v, float) and v != v:
        return None
    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:
            return v
    return v


def event_report(view, event_id: int, reviews: pd.DataFrame) -> dict:
    ev = view.event(event_id)
    inputs = view.event_inputs(event_id)
    expl = view.explanation(ev.get("explanation_id"))
    caps = view.capabilities
    rep = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {"key": view.key, "label": view.label, "dataset_id": ev["dataset_id"], "synthetic": view.kind == "synthetic"},
        "model": {"model_version": view.model_version, "schema_version": ev["schema_version"]},
        "eligibility_rule": "Only CDMs with time_to_tca >= 2 days contribute to the forecast.",
        "event_id": int(ev["event_id"]),
        "prediction": {
            "prediction_id": ev["prediction_id"], "prediction_status": ev["prediction_status"],
            "predicted_final_tier": ev["predicted_final_tier"],
            "class_probabilities": None if ev["p_critical"] is None or pd.isna(ev["p_critical"]) else
                {"Safe": _clean(ev["p_safe"]), "Elevated": _clean(ev["p_elevated"]), "Critical": _clean(ev["p_critical"])},
            "class_probability_note": caps.class_probabilities_note,
            "review_priority": ev["review_priority"], "reason_codes": list(ev["reason_codes"]),
            "quality_flags": list(ev["quality_flags"]),
            "uncertainty_interval": "unavailable" if not caps.uncertainty_intervals else "see model output",
        },
        "available_evidence": {
            "n_eligible_cdms": int(ev["n_eligible_cdms"]), "n_excluded_cdms": _clean(ev["n_excluded_cdms"]),
            "latest_input_time_to_tca_days": _clean(ev["latest_input_time_to_tca_days"]),
            "latest_observed_miss_distance_m": _clean(ev["latest_miss_distance_m"]),
            "latest_observed_tier": ev["latest_observed_tier"],
            "source_row_ids": list(ev["source_row_ids"]),
        },
        "inputs_used": [{k: _clean(v) for k, v in r.items()} for r in inputs.to_dict("records")],
        "explanation": None if expl is None else {"component": expl.get("component"), "units": expl.get("output_units"),
                                                  "text": expl.get("text"), "group_contributions": json.loads(expl["group_contributions"])
                                                  if isinstance(expl.get("group_contributions"), str) else None},
        "analyst_reviews": [{k: _clean(v) for k, v in r.items()} for r in reviews.to_dict("records")],
    }
    return rep


def report_markdown(rep: dict) -> str:
    p, e = rep["prediction"], rep["available_evidence"]
    lines = [f"# Conjunction event {rep['event_id']} — review report", "",
             f"*Generated {rep['generated_at']} · {rep['dataset']['label']} · model `{rep['model']['model_version']}`*", ""]
    if rep["dataset"]["synthetic"]:
        lines += ["> **Synthetic demo data — not model output.**", ""]
    lines += ["## Forecast", f"- Predicted final tier: **{p['predicted_final_tier'] or p['prediction_status']}**",
              f"- Prediction status: {p['prediction_status']}"]
    if p["class_probabilities"]:
        cp = p["class_probabilities"]
        lines.append(f"- Model probability of final tier: Safe {pct(cp['Safe'])}, Elevated {pct(cp['Elevated'])}, Critical {pct(cp['Critical'])}")
        lines.append(f"  - {p['class_probability_note'] or 'Class probability, not a collision probability.'}")
    lines += [f"- Uncertainty interval: {p['uncertainty_interval']}", f"- Review priority: {p['review_priority']}",
              "- Reasons: " + ("; ".join(REASON_TEXT.get(r, r) for r in p["reason_codes"]) or "none"),
              "- Quality flags: " + ("; ".join(FLAG_TEXT.get(f, f) for f in p["quality_flags"]) or "none"), "",
              "## Available evidence", f"- {rep['eligibility_rule']}",
              f"- Eligible CDMs used: {e['n_eligible_cdms']}" + (f" (excluded inside cutoff: {e['n_excluded_cdms']})" if e["n_excluded_cdms"] is not None else ""),
              f"- Latest eligible CDM: {days(e['latest_input_time_to_tca_days'])} before TCA",
              f"- Latest observed miss distance: {km(e['latest_observed_miss_distance_m'])}",
              f"- Latest observed tier: {e['latest_observed_tier'] or '–'}", "",
              "| Time to TCA (d) | Miss (km) | Risk | Max risk est. |", "|---|---|---|---|"]
    for r in rep["inputs_used"]:
        lines.append(f"| {r.get('time_to_tca_days', 0):.2f} | {km(r.get('miss_distance_m'), 3).replace(' km', '')} | "
                     f"{log10_pc(r.get('risk_log10'))} | {log10_pc(r.get('max_risk_estimate_log10'))} |")
    lines += ["", "## Explanation"]
    if rep["explanation"]:
        lines += [rep["explanation"]["text"] or "", f"*({rep['explanation']['component']}, {rep['explanation']['units']}; associations, not causes)*"]
    else:
        lines.append("Explanation unavailable for this prediction.")
    lines += ["", "## Analyst reviews"]
    if rep["analyst_reviews"]:
        for r in rep["analyst_reviews"]:
            lines.append(f"- {r['created_at']} · {r['status']} · {r.get('reviewer') or 'unnamed'}: {r.get('note') or ''}"
                         + (f" (follow-up: {r['follow_up']})" if r.get("follow_up") else ""))
    else:
        lines.append("No reviews recorded.")
    lines += ["", "*Decision support only. This report does not calculate or authorise a manoeuvre.*"]
    return "\n".join(lines)
