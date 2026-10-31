"""Render RESULTS.md from artifacts/evaluation.json, audit.json and manifest.json.

Usage: uv run python scripts/make_results.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"


def f(x, nd=3):
    if x is None:
        return "undefined"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def pct(x):
    return "undefined" if x is None else f"{100 * x:.1f}%"


def ci(b, key):
    d = (b or {}).get(key) or {}
    if d.get("lower") is None:
        return ""
    return f" [{f(d['lower'])}, {f(d['upper'])}]"


def main() -> None:
    ev = json.loads((ART / "evaluation.json").read_text(encoding="utf-8"))
    audit = json.loads((ART / "audit.json").read_text(encoding="utf-8"))
    man = json.loads((ART / "manifest.json").read_text(encoding="utf-8"))
    pm = ev["primary_model"]
    models = ev["models"]
    order = ["baseline_median", "baseline_linear", "lgbm_single_cdm", "lgbm_physics", "lgbm_history", "lgbm_history_weighted",
             "lgbm_history_l2", "lgbm_history_l2_monotone"]
    order = [m for m in order if m in models] + [m for m in models if m not in order]

    out = []
    out.append("# Results of the shipped run\n")
    out.append(f"Primary model: `{pm}` (version `{ev['model_version']}`), interval method `{ev['interval_method']}`. "
               f"Generated {ev['generated_at']}. Total training time {ev['total_train_seconds']} s.\n")
    sc = ev["scope"]["split_counts"]
    out.append("## Data and partitions\n")
    out.append("| Partition | Events | Rows | Safe | Elevated | Critical (distance) | Critical (Pc) |\n|---|---|---|---|---|---|---|")
    for k, v in sc.items():
        s = v["strata"]
        out.append(f"| {k} | {v['n_events']} | {v['n_rows']} | {s.get('Safe', 0)} | {s.get('Elevated', 0)} | {s.get('Critical', 0)} | {s.get('Critical_pc', 0)} |")
    tr = audit["train"]
    out.append(f"\nTraining file: {tr['n_rows']} rows, {tr['n_events']} events, {pct(tr['target']['fraction_at_floor'])} of rows at the −30 floor, "
               f"{tr['target']['n_ge_minus4']} rows with risk ≥ −4. Test file: {audit['test']['n_rows']} rows, {audit['test']['n_events']} events, "
               f"rows stop at {f(audit['test_horizon']['min_time_to_tca_days'], 2)} days before TCA.\n")

    def table(view: str, title: str):
        out.append(f"## {title}\n")
        out.append("| Model | Critical recall | Critical precision | Critical F2 | Elev+ F2 | Macro-F1 | Pc-only Crit recall | Pc-only Crit precision | MAE | MAE non-floor | MAE high-risk | Coverage | Median width | Auto-clear | True Crit → auto-clear |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for m in order:
            v = models[m][view]
            t, p, r, i, g = v["tier"], v["pc_only"], v["regression"], v["interval"], v["triage"]
            out.append(f"| {m} | {f(t['critical_recall'])} | {f(t['critical_precision'])} | {f(t['critical_f2'])} | {f(t['elevated_or_above_f2'])} | {f(t['macro_f1'])} | "
                       f"{f(p['critical_recall'])} | {f(p['critical_precision'])} | {f(r['mae'], 2)} | {f(r['mae_non_floor'], 2)} | {f(r['mae_high_risk'], 2)} | "
                       f"{f(i['coverage'])} | {f(i['median_width'], 1)} | {pct(g['auto_clear_fraction'])} | {g['true_critical_routed_to_auto_clear_n']} |")
        v = models[pm][view]
        out.append(f"\nView size: {v['n_rows']} rows, {v['n_events']} events, {v['n_critical_true']} true Critical rows, "
                   f"{v['pc_only'].get('n_true_positive_rows', 0)} Pc-only true positives for the primary model. Nominal interval coverage {v['interval']['nominal_coverage']}.")
        b = v.get("bootstrap") or {}
        out.append(f"Event-bootstrap 95% intervals for the primary model: Critical recall{ci(b, 'critical_recall')}, "
                   f"Critical precision{ci(b, 'critical_precision')}, MAE{ci(b, 'mae')}, coverage{ci(b, 'coverage')}.\n")
        if v["tier"].get("undefined") or v["pc_only"].get("undefined"):
            out.append(f"Undefined metrics in this view (no positives in the subset): tier {v['tier'].get('undefined')}, Pc-only {v['pc_only'].get('undefined')}.\n")

    table("latest_cdm", "Latest CDM per event (evaluation partition)")
    table("row_level", "All CDMs (evaluation partition)")
    table("horizon_latest_cdm", "Latest CDM per event, restricted to ≥ 2 days before TCA (matches the test horizon)")

    out.append("## Triage mix of the primary model (evaluation partition)\n")
    out.append("| View | Maneuver planning review | Monitor | Human review | Auto-clear candidate | Insufficient data |\n|---|---|---|---|---|---|")
    for view in ("latest_cdm", "row_level"):
        a = models[pm][view]["triage"]["action_counts"]
        out.append(f"| {view} | {a.get('Maneuver planning review', 0)} | {a.get('Monitor', 0)} | {a.get('Human review', 0)} | {a.get('Auto-clear candidate', 0)} | {a.get('Human review - insufficient data', 0)} |")

    out.append("\n## Interval method comparison (primary model, evaluation partition)\n")
    out.append("| Method | View | Coverage | Median width | Auto-clear | Human review | True Crit → auto-clear |\n|---|---|---|---|---|---|---|")
    for meth, views in ev.get("interval_method_comparison_primary", {}).items():
        for view, d in views.items():
            out.append(f"| {meth} | {view} | {f(d['coverage'])} | {f(d['median_width'], 1)} | {pct(d['auto_clear_fraction'])} | {pct(d['human_review_fraction'])} | {d['true_critical_routed_to_auto_clear_n']} |")

    c = ev["classifier"]
    cl = models[pm]["row_level"].get("classifier") or {}
    out.append("\n## Optional classifier P(risk ≥ −4)\n")
    out.append(f"Sigmoid-calibrated on the policy-calibration partition ({c['n_pos_policy_cal']} positives); disagreement threshold tuned there to {c['threshold']:.2f} "
               f"(policy-cal F2 {f(c['policy_cal_f2_at_threshold'])}, optimistic because the same partition fitted the sigmoid). "
               f"Evaluation partition, row level: PR-AUC {f(cl.get('pr_auc'))}, Brier {f(cl.get('brier'), 4)}, positives {cl.get('n_pos')}.\n")
    rel = c.get("reliability_policy_cal") or {}
    if rel:
        out.append("Reliability (policy-cal): bin | mean predicted | observed fraction | count")
        out.append("")
        out.append("| bin | mean pred | observed | n |\n|---|---|---|---|")
        for lo, hi, mp, fp, n in zip(rel["bin_lower"], rel["bin_upper"], rel["mean_pred"], rel["frac_pos"], rel["count"]):
            if n:
                out.append(f"| {lo:.1f}–{hi:.1f} | {f(mp)} | {f(fp)} | {n} |")

    dt = ev.get("delivered_test_file") or {}
    if dt:
        out.append("\n## External check: delivered test file (labelled, ≥ 2 days before TCA)\n")
        out.append("| Model | View | Critical recall | Critical precision | Pc-only Crit recall | Pc-only Crit precision | MAE | MAE high-risk | Coverage | Auto-clear | True Crit → auto-clear |\n|---|---|---|---|---|---|---|---|---|---|---|")
        for m, r in dt.items():
            for view in ("latest_cdm", "row_level"):
                v = r[view]
                out.append(f"| {m} | {view} | {f(v['tier']['critical_recall'])} | {f(v['tier']['critical_precision'])} | {f(v['pc_only']['critical_recall'])} | "
                           f"{f(v['pc_only']['critical_precision'])} | {f(v['regression']['mae'], 2)} | {f(v['regression']['mae_high_risk'], 2)} | {f(v['interval']['coverage'])} | "
                           f"{pct(v['triage']['auto_clear_fraction'])} | {v['triage']['true_critical_routed_to_auto_clear_n']} |")

    out.append("\n## Model selection trace\n")
    best = ev["selection"]["cv_search"]["best"]
    out.append(f"Grouped CV ({ev['selection']['cv_search']['n_folds']} folds) over {len(ev['selection']['cv_search']['results'])} sampled configurations; "
               f"rule: {ev['selection']['cv_search']['selection_rule']}. Best: `{json.dumps(best['params'])}`, mean best iteration {best['mean_best_iteration']}, "
               f"OOF MAE non-floor {f(best['scores']['mae_non_floor'])}, OOF Pc-only Critical F2 {f(best['scores']['critical_f2_pc_only'])}.\n")
    out.append("Candidate selection on the policy-calibration partition:\n")
    out.append("| Candidate | MAE non-floor | MAE high-risk | Pc-only Crit recall | Pc-only Crit precision | Pc-only Crit F2 | Selection score |\n|---|---|---|---|---|---|---|")
    for r in ev["selection"]["candidate_selection_on_policy_cal"]:
        out.append(f"| {r['name']} | {f(r['mae_non_floor'])} | {f(r['mae_high_risk'])} | {f(r['critical_recall_pc_only'])} | {f(r['critical_precision_pc_only'])} | {f(r['critical_f2_pc_only'])} | {f(r['selection_score'], 2)} |")

    out.append("\n## Illustrative cases (primary model, evaluation partition)\n")
    cases = ev["illustrative_cases"]
    for key, title in (("missed_critical", "Missed Critical (true Critical, predicted lower tier)"), ("false_alarm_critical", "False alarm (predicted Critical, true lower tier)"),
                       ("uncertain", "Routed to human review with the widest intervals"), ("invalid", "Invalid inputs")):
        rows = cases.get(key) or []
        out.append(f"### {title}: {len(rows)} shown\n")
        if not rows:
            out.append("None in this partition.\n")
            continue
        out.append("| row_id | time to TCA (d) | miss (m) | true risk | pred risk | interval | tier true → pred | action | reasons |\n|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            interval = f"[{f(r.get('risk_log10_lower'), 1)}, {f(r.get('risk_log10_upper'), 1)}]"
            out.append(f"| {r['row_id']} | {f(r.get('time_to_tca_days'), 2)} | {f(r.get('miss_distance_m'), 0)} | {f(r.get('risk_true'), 2)} | {f(r.get('risk_log10_pred'), 2)} | {interval} | "
                       f"{r.get('risk_tier_true')} → {r.get('risk_tier')} | {r.get('triage_action')} | {', '.join(r.get('reason_codes') or [])} |")
        out.append("")

    out.append("## Batch inference\n")
    out.append(f"Delivered test file: {man['n_predictions']} rows in {man['batch_inference_seconds']} s including explanations "
               f"(breakdown: {json.dumps(man['timing_breakdown_seconds'])}). Status counts {json.dumps(man['status_counts'])}; "
               f"tier counts {json.dumps(man['tier_counts'])}; triage counts {json.dumps(man['triage_counts'])}.\n")
    out.append("## Notes\n")
    for n in ev["notes"]:
        out.append(f"- {n}")
    (ROOT / "RESULTS.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("wrote RESULTS.md")


if __name__ == "__main__":
    main()
