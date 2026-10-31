"""Render RESULTS.md from artifacts/evaluation.json, audit.json and manifest.json.

Usage: uv run python scripts/make_results.py [ARTIFACTS_DIR] [OUTPUT_MD]

Every lookup is tolerant (`.get` with defaults) so a partially written
evaluation.json still renders; undefined metrics print as "undefined".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from collision_risk.evaluate import TIERS, compare_models  # noqa: E402

CARRY_FORWARD = "carry_forward_risk"
MODEL_ORDER = [
    "carry_forward_risk", "carry_forward_mre", "majority", "logistic",
    "lgbm_multiclass", "lgbm_ordinal", "lgbm_residual",
]
SUBSETS = ("real_forecast", "all_usable", "trivial")
TEST_ENRICHMENT_NOTE = (
    "The Kelvins test set was enriched with high-risk events, so its predicted class "
    "distribution is expected to be shifted toward Elevated/Critical relative to the training labels."
)


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def f(x: Any, nd: int = 3) -> str:
    if x is None:
        return "undefined"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def pct(x: Any) -> str:
    return "undefined" if x is None else f"{100 * float(x):.1f}%"


def ci(b: Optional[dict], key: str, nd: int = 3) -> str:
    d = (b or {}).get(key) or {}
    if d.get("lower") is None or d.get("upper") is None:
        return "undefined"
    return f"[{f(d['lower'], nd)}, {f(d['upper'], nd)}]"


def g(d: Any, *path: str, default: Any = None) -> Any:
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur or cur[key] is None:
            return default
        cur = cur[key]
    return cur


def load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def md_table(header: list[str], rows: list[list[Any]]) -> list[str]:
    out = ["| " + " | ".join(str(h) for h in header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return out


def order_models(models: dict, primary: Optional[str]) -> list[str]:
    names = list(models.keys())
    ordered = [m for m in MODEL_ORDER if m in names] + [m for m in names if m not in MODEL_ORDER]
    if primary in ordered:
        ordered.remove(primary)
        idx = 1 if ordered and ordered[0] == CARRY_FORWARD else 0
        ordered.insert(idx, primary)
    return ordered


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
def comparison_table(models: dict, order: list[str], subset: str, primary: Optional[str]) -> list[str]:
    if not models:
        return ["No model results available."]
    table = compare_models({m: models[m] for m in order}, subset=subset)
    header = ["Model", "Events", "Critical recall", "Critical precision", "Critical F2", "Elev+ F2", "Macro-F1",
              "Brier", "Log loss", "Auto-clear", "Final Crit auto-cleared", "Watch share", "Escalation recall"]
    rows = []
    for m in order:
        r = table.loc[m]
        name = f"**{m}**" if m == primary else m
        rows.append([
            name, f(r["n_events"]), f(r["critical_recall"]), f(r["critical_precision"]), f(r["critical_f2"]),
            f(r["elevated_or_above_f2"]), f(r["macro_f1"]), f(r["brier"]), f(r["log_loss"]),
            pct(r["auto_clear_share"]), f(r["n_final_critical_auto_cleared"]), pct(r["watch_share"]),
            f(r["escalation_recall"]),
        ])
    return md_table(header, rows)


def confusion_tables(cls: Optional[dict]) -> list[str]:
    conf = g(cls, "confusion") or {}
    counts = conf.get("counts")
    if not counts:
        return ["Confusion matrix unavailable."]
    pred_labels = conf.get("pred_labels") or conf.get("labels") or TIERS
    true_labels = conf.get("labels") or TIERS
    out = ["Counts (rows = true final tier, columns = predicted):", ""]
    out += md_table(["true \\ pred"] + list(pred_labels), [[t] + [str(c) for c in row] for t, row in zip(true_labels, counts)])
    out.append("")
    out.append("Row-normalised (share of each true tier):")
    out.append("")
    norm = conf.get("row_normalised") or [None] * len(true_labels)
    rows = []
    for t, row in zip(true_labels, norm):
        rows.append([t] + (["undefined"] * len(pred_labels) if row is None else [pct(v) for v in row]))
    out += md_table(["true \\ pred"] + list(pred_labels), rows)
    if cls and cls.get("undefined"):
        out.append("")
        out.append(f"Undefined metrics here: {', '.join(cls['undefined'])}.")
    return out


def escalation_section(models: dict, primary: Optional[str], order: list[str]) -> list[str]:
    out: list[str] = []
    header = ["Model", "Subset", "Not-Critical now", "End Critical", "Flagged", "Caught", "Escalation recall",
              "Flag precision", "Watch size", "Watch hits", "Critical now", "De-escalate", "Pred. de-esc.", "Correct", "False de-esc."]
    rows = []
    for m in [x for x in (CARRY_FORWARD, primary) if x in models] + [x for x in order if x not in (CARRY_FORWARD, primary)]:
        for subset in ("real_forecast", "all_usable"):
            e = g(models, m, "subsets", subset, "escalation") or {}
            nc = e.get("non_critical_now") or {}
            cc = e.get("critical_now") or {}
            rows.append([
                m, subset, f(nc.get("n_total")), f(nc.get("n_end_critical")), f(nc.get("n_flagged")), f(nc.get("n_caught")),
                f(nc.get("recall_of_escalations")), f(nc.get("precision_of_flags")), f(nc.get("watch_list_size")),
                f(nc.get("watch_list_hits")), f(cc.get("n_total")), f(cc.get("n_deescalate")),
                f(cc.get("n_predicted_deescalation")), f(cc.get("n_correct_deescalation")), f(cc.get("n_false_deescalation")),
            ])
    if not rows:
        return ["No escalation results available."]
    out.append("Among events that are not Critical at the forecast time: how many end Critical, how many the forecast "
               "flags (predicted Critical or placed on the escalation watch list), and how many of those flags are hits. "
               "Among events that are Critical now: how many de-escalate and how the forecast handles them.")
    out.append("")
    out += md_table(header, rows)
    return out


def triage_section(models: dict, primary: Optional[str]) -> list[str]:
    out: list[str] = []
    pm = g(models, primary) or {}
    rows = []
    for subset in SUBSETS:
        t = g(pm, "subsets", subset, "triage")
        if not t:
            continue
        rows.append([
            subset, f(t.get("n_events")), pct(t.get("auto_clear_share")), f(t.get("n_final_critical_auto_cleared")),
            f(t.get("n_final_critical")), pct(t.get("watch_share")), f(t.get("n_final_critical_in_watch")),
            pct(t.get("maneuver_review_share")), pct(t.get("human_review_share")), pct(t.get("monitor_share")),
        ])
    if not rows:
        return ["No triage results available (triage_action column missing)."]
    out += md_table(["Subset", "Events", "Auto-clear share", "Final-Critical auto-cleared", "Final Critical",
                     "Watch share", "Final-Critical in watch", "Maneuver review", "Human review", "Monitor"], rows)
    n_cleared = g(pm, "subsets", "all_usable", "triage", "n_final_critical_auto_cleared")
    if n_cleared is None:
        pass
    elif n_cleared == 0:
        out.append("")
        out.append("No event that ended Critical was auto-cleared on the holdout.")
    else:
        out.append("")
        out.append(f"**{n_cleared} event(s) that ended Critical were auto-cleared on the holdout.** The auto-clear "
                   "threshold was tuned for zero missed Critical on dev out-of-fold predictions; this holdout count "
                   "is the honest out-of-sample number and should be read with the bootstrap interval of the "
                   "auto-clear share.")
    counts = g(pm, "subsets", "all_usable", "triage", "action_counts") or {}
    if counts:
        out.append("")
        out.append("Action counts (primary model, all usable holdout events):")
        out.append("")
        out += md_table(["Action", "Events"], [[k, v] for k, v in counts.items()])
    return out


def calibration_section(ev: dict, models: dict, order: list[str], primary: Optional[str]) -> list[str]:
    out: list[str] = []
    rows = []
    for m in order:
        for subset in ("real_forecast", "all_usable"):
            p = g(models, m, "subsets", subset, "probability")
            if not p:
                continue
            pcb = p.get("per_class_brier") or {}
            rows.append([m, subset, f(p.get("n_used")), f(p.get("brier"), 4), f(p.get("log_loss"), 4),
                         f(pcb.get("Critical"), 4), f(p.get("critical_pr_auc")), f(p.get("critical_roc_auc")),
                         f(g(p, "reliability", "Critical", "ece"), 4)])
    if rows:
        out += md_table(["Model", "Subset", "n", "Brier (multiclass)", "Log loss", "Brier (Critical)", "Critical PR-AUC",
                         "Critical ROC-AUC", "ECE (Critical)"], rows)
    else:
        out.append("No probability metrics available.")
    rel = g(models, primary, "subsets", "real_forecast", "probability", "reliability", "Critical")
    if rel and rel.get("count"):
        out.append("")
        out.append(f"Reliability of P(Critical) for `{primary}` on real_forecast holdout events (10 equal-width bins):")
        out.append("")
        rrows = []
        for lo, hi, mp, fp, n in zip(rel.get("bin_lower", []), rel.get("bin_upper", []), rel.get("mean_pred", []),
                                     rel.get("frac_pos", []), rel.get("count", [])):
            if n:
                rrows.append([f"{lo:.1f}-{hi:.1f}", f(mp), f(fp), n])
        out += md_table(["Bin", "Mean predicted", "Observed Critical share", "Events"], rrows)
    cal = ev.get("calibration") or {}
    if cal:
        oof = cal.get("dev_oof") or {}
        out.append("")
        out.append(f"Calibration method: `{cal.get('method', 'unknown')}` fitted on dev out-of-fold predictions "
                   f"(dev OOF Brier {f(oof.get('brier'), 4)}, log loss {f(oof.get('log_loss'), 4)}).")
    return out


def bootstrap_section(models: dict, primary: Optional[str]) -> list[str]:
    rows = []
    for m in [x for x in (primary, CARRY_FORWARD) if x in models]:
        for subset in ("real_forecast", "all_usable"):
            b = g(models, m, "subsets", subset, "bootstrap")
            cls = g(models, m, "subsets", subset, "classification") or {}
            tri = g(models, m, "subsets", subset, "triage") or {}
            if not b:
                continue
            rows.append([m, subset, f(b.get("n_bootstrap")),
                         f"{f(cls.get('critical_recall'))} {ci(b, 'critical_recall')}",
                         f"{f(cls.get('critical_precision'))} {ci(b, 'critical_precision')}",
                         f"{f(cls.get('macro_f1'))} {ci(b, 'macro_f1')}",
                         f"{pct(tri.get('auto_clear_share'))} {ci(b, 'auto_clear_share')}"])
    if not rows:
        return ["No bootstrap results available."]
    return ["Point estimate followed by the 95 % event-bootstrap interval (2.5 / 97.5 percentiles).", ""] + md_table(
        ["Model", "Subset", "Resamples", "Critical recall", "Critical precision", "Macro-F1", "Auto-clear share"], rows)


def breakdown_table(bd: Optional[dict], first_col: str) -> list[str]:
    if not bd:
        return ["Not available."]
    rows = []
    for name, s in bd.items():
        label = name
        if s.get("merged_groups"):
            label = f"{name} ({', '.join(s['merged_groups'])})"
        rows.append([label, f(s.get("n")), f(s.get("n_critical")), f(s.get("n_predicted_critical")),
                     f(s.get("critical_recall")), f(s.get("critical_precision")), f(s.get("critical_f2")),
                     f(s.get("macro_f1")), f(s.get("accuracy"))])
    return md_table([first_col, "Events", "Final Critical", "Predicted Critical", "Critical recall", "Critical precision",
                     "Critical F2", "Macro-F1", "Accuracy"], rows)


def cases_section(cases: dict) -> list[str]:
    out: list[str] = []
    titles = (
        ("missed_critical", "Missed Critical (ended Critical, forecast lower, not on the watch list)"),
        ("caught_by_watch", "Caught by the escalation watch (ended Critical, forecast lower, on the watch list)"),
        ("false_alarm_critical", "False alarm (forecast Critical, ended lower)"),
        ("false_deescalation", "False de-escalation (Critical now, forecast lower, ended Critical)"),
        ("uncertain", "Most uncertain forecasts"),
    )
    for key, title in titles:
        rows = (cases or {}).get(key) or []
        out.append(f"### {title}: {len(rows)} shown")
        out.append("")
        if not rows:
            out.append("None on the holdout.")
            out.append("")
            continue
        trows = []
        for r in rows:
            pset = r.get("prediction_set")
            pset_s = "/".join(pset) if isinstance(pset, list) else ""
            reasons = r.get("reason_codes")
            reasons_s = ", ".join(reasons) if isinstance(reasons, list) else ""
            trows.append([
                f"{r.get('dataset_id', '')}:{r.get('event_id', '')}", f(r.get("time_to_tca_days"), 2),
                f(r.get("miss_distance_m"), 0), f(r.get("n_cdms_2d")), r.get("current_tier"), r.get("final_tier"),
                r.get("predicted_final_tier"), f(r.get("p_critical")), pset_s, r.get("triage_action"), reasons_s,
            ])
        out += md_table(["Event", "Lead time (d)", "Miss (m)", "CDMs", "Current", "Final", "Forecast", "P(Critical)",
                         "Set", "Action", "Reasons"], trows)
        out.append("")
    return out


def audit_section(ev: dict, audit: dict) -> list[str]:
    out: list[str] = []
    la = ev.get("label_audit") or {}
    if la:
        cc = la.get("class_counts_usable") or {}
        out += md_table(["Quantity", "Value"], [
            ["Events in the labels workbook", f(la.get("n_events_total"))],
            ["Usable events (>= 1 CDM at >= 2 days)", f(la.get("n_usable"))],
            ["Unusable events", f(la.get("n_unusable"))],
            ["Trivial events (final CDM itself at >= 2 days)", f(la.get("n_trivial"))],
            ["Real-forecast events", f(la.get("n_real_forecast"))],
            ["Usable class counts (Safe / Elevated / Critical)",
             f"{f(cc.get('Safe'))} / {f(cc.get('Elevated'))} / {f(cc.get('Critical'))}"],
            ["Labels match recomputed tiers", f(la.get("labels_match_recomputed"))],
        ])
    else:
        out.append("Label audit not available in evaluation.json.")
    files = audit.get("files") or {}
    if isinstance(files, dict) and files:
        out.append("")
        rows = []
        for name, info in files.items():
            if not isinstance(info, dict):
                continue
            rows.append([name, f(info.get("n_rows")), f(info.get("n_events")), f(info.get("n_kept_columns")),
                         f(len(info.get("dropped_columns") or []) or None), (info.get("sha256") or "")[:12]])
        if rows:
            out += md_table(["File", "Rows", "Events", "Kept columns", "Dropped columns", "sha256 (prefix)"], rows)
    for key in ("test_horizon", "max_risk_estimate", "label_check"):
        if key in audit:
            out.append("")
            out.append(f"`{key}`: `{json.dumps(audit[key], default=str)[:600]}`")
    return out


def selection_section(ev: dict) -> list[str]:
    out: list[str] = []
    trc = ev.get("training_rows_choice") or {}
    if trc:
        scores = trc.get("oof_critical_f2_real_forecast") or {}
        out.append(f"Training rows: `{trc.get('chosen', 'unknown')}` chosen on dev OOF Critical F2 (real_forecast): "
                   + ", ".join(f"{k} = {f(v)}" for k, v in scores.items()) + ".")
        out.append("")
    cv = ev.get("cv_search") or {}
    if cv:
        best = cv.get("best") or {}
        n_cfg = len(cv.get("results") or []) or cv.get("n_configs")
        out.append(f"Grouped CV search ({f(cv.get('n_folds'))} folds, {f(n_cfg)} configurations, rule: "
                   f"{cv.get('selection_rule', 'n/a')}). Best params `{json.dumps(best.get('params', {}), default=str)}`, "
                   f"scores `{json.dumps(best.get('scores', {}), default=str)}`.")
        out.append("")
    thr = ev.get("decision_thresholds") or {}
    if thr:
        out.append("Decision thresholds (tuned on dev OOF only): "
                   + ", ".join(f"`{k}` = {f(v)}" for k, v in thr.items()) + ".")
        out.append("")
    pt = ev.get("policy_tuning") or {}
    if pt:
        out.append(f"Policy tuning summary: `{json.dumps(pt, default=str)[:800]}`")
        out.append("")
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def render(ev: dict, audit: dict, man: dict) -> str:
    models: dict = ev.get("models") or {}
    primary: Optional[str] = ev.get("primary_model")
    if primary not in models and models:
        primary = next((m for m in models if m != CARRY_FORWARD), next(iter(models)))
    order = order_models(models, primary)
    split = ev.get("split") or {}
    hold = g(split, "counts", "holdout") or {}
    prim = models.get(primary) or {}

    out: list[str] = []
    out.append("# Results of the shipped run")
    out.append("")
    out.append(f"Task: forecast each event's **final** tier (Safe / Elevated / Critical) from CDMs at >= 2 days before TCA. "
               f"Primary model: `{primary}` (version `{ev.get('model_version', 'n/a')}`). Generated {ev.get('generated_at', 'n/a')}. "
               f"Total training time {f(ev.get('total_train_seconds'), 1)} s.")
    out.append("")
    out.append(f"Holdout: {f(prim.get('n_events'))} events ({f(prim.get('n_real_forecast'))} real forecasts, "
               f"{f(prim.get('n_trivial'))} trivial), split seed {f(split.get('seed'))}, holdout fraction "
               f"{f(split.get('holdout_fraction'))}, split hash `{split.get('split_hash', 'n/a')}`. "
               f"Final-tier counts on the holdout: {json.dumps(prim.get('class_counts') or hold.get('class_counts') or {})}.")
    out.append("")
    out.append("Subsets: `all_usable` = every holdout event; `trivial` = the event's final CDM was itself at >= 2 days "
               "(the answer is visible in the input); `real_forecast` = everything else and the headline subset.")
    out.append("")

    out.append("## Headline: real-forecast holdout")
    out.append("")
    out.append("Headline metric: **Critical recall on real_forecast events**, read together with Critical precision / F2. "
               f"`{CARRY_FORWARD}` is the official naive baseline (the latest known tier carried forward).")
    out.append("")
    out += comparison_table(models, order, "real_forecast", primary)
    out.append("")
    cf_f2 = g(models, CARRY_FORWARD, "subsets", "real_forecast", "classification", "critical_f2")
    pm_f2 = g(models, primary, "subsets", "real_forecast", "classification", "critical_f2")
    if cf_f2 is not None and pm_f2 is not None:
        if pm_f2 > cf_f2:
            out.append(f"`{primary}` beats `{CARRY_FORWARD}` on Critical F2 (real_forecast): {f(pm_f2)} vs {f(cf_f2)}.")
        else:
            out.append(f"The primary model `{primary}` does not beat the carry-forward baseline on Critical F2 for "
                       f"real forecasts ({f(pm_f2)} vs {f(cf_f2)}). The gain, if any, is in the escalation watch list, "
                       f"calibration and auto-clear safety reported below, not in the hard tier forecast.")
    elif pm_f2 is None:
        out.append("Critical F2 of the primary model is undefined on real_forecast (no final-Critical events in the subset).")
    out.append("")
    out.append("### All usable holdout events")
    out.append("")
    out += comparison_table(models, order, "all_usable", primary)
    out.append("")
    out.append("### Trivial events (answer visible in the input)")
    out.append("")
    out += comparison_table(models, order, "trivial", primary)
    out.append("")

    out.append("## Confusion matrices")
    out.append("")
    for m in [x for x in (CARRY_FORWARD, primary) if x in models]:
        for subset in ("real_forecast", "all_usable"):
            out.append(f"### `{m}` on {subset} ({f(g(models, m, 'subsets', subset, 'n_events'))} events)")
            out.append("")
            out += confusion_tables(g(models, m, "subsets", subset, "classification"))
            out.append("")

    out.append("## Escalation watch")
    out.append("")
    out += escalation_section(models, primary, order)
    out.append("")

    out.append("## Auto-clear and triage mix (primary model)")
    out.append("")
    out += triage_section(models, primary)
    out.append("")

    out.append("## Calibration")
    out.append("")
    out += calibration_section(ev, models, order, primary)
    out.append("")

    out.append("## Bootstrap confidence intervals")
    out.append("")
    out += bootstrap_section(models, primary)
    out.append("")

    out.append("## Breakdowns (primary model)")
    out.append("")
    for subset in ("real_forecast", "all_usable"):
        out.append(f"### Lead time of the latest snapshot, {subset}")
        out.append("")
        out += breakdown_table(g(prim, "by_lead_time", subset), "Days to TCA")
        out.append("")
        out.append(f"### Object type, {subset}")
        out.append("")
        out += breakdown_table(g(prim, "by_object_type", subset), "Object type")
        out.append("")
    if g(prim, "by_current_tier", "real_forecast"):
        out.append("### Current tier at forecast time, real_forecast")
        out.append("")
        out += breakdown_table(g(prim, "by_current_tier", "real_forecast"), "Current tier")
        out.append("")

    out.append("## Ablations (real_forecast holdout)")
    out.append("")
    abl = ev.get("ablations") or {}
    if abl:
        abl_all = dict(abl)
        if primary in models:
            abl_all = {f"{primary} (primary)": models[primary], **abl}
        out += comparison_table(abl_all, list(abl_all.keys()), "real_forecast", None)
    else:
        out.append("No ablations recorded.")
    out.append("")

    out.append("## Mission-held-out cross-validation")
    out.append("")
    mh = ev.get("mission_heldout_cv")
    if mh:
        out.append("Out-of-fold predictions over all usable events with folds grouped by `mission_id` "
                   "(mission_id is never a feature). This checks that the model is not memorising missions.")
        out.append("")
        out += comparison_table({"mission_heldout_oof": mh}, ["mission_heldout_oof"], "real_forecast", None)
        out.append("")
        out += comparison_table({"mission_heldout_oof": mh}, ["mission_heldout_oof"], "all_usable", None)
    else:
        out.append("Not run.")
    out.append("")

    out.append("## Conformal prediction sets")
    out.append("")
    cs = ev.get("conformal_sets")
    if cs:
        out.append(f"Nominal coverage {f(cs.get('coverage_nominal'))}, empirical holdout coverage "
                   f"{f(cs.get('coverage_empirical'))}, average set size {f(cs.get('avg_set_size'), 2)}, q_hat {f(cs.get('q_hat'), 4)}.")
        ps = g(prim, "subsets", "real_forecast", "prediction_sets")
        if ps:
            out.append(f"On real_forecast events: coverage {f(ps.get('coverage_empirical'))}, average set size "
                       f"{f(ps.get('avg_set_size'), 2)}, singleton share {pct(ps.get('singleton_share'))}.")
    else:
        out.append("Disabled or not available.")
    out.append("")

    out.append("## Illustrative cases (primary model, holdout)")
    out.append("")
    out += cases_section(ev.get("illustrative_cases") or {})

    out.append("## Test-set predictions (Kelvins test file)")
    out.append("")
    tier_counts = man.get("tier_counts") or man.get("predicted_tier_counts") or {}
    if tier_counts:
        total = sum(v for v in tier_counts.values() if isinstance(v, (int, float))) or None
        rows = [[k, v, pct(v / total) if total else "undefined"] for k, v in tier_counts.items()]
        out += md_table(["Predicted final tier", "Events", "Share"], rows)
    else:
        out.append("Predicted class distribution not available in manifest.json.")
    out.append("")
    out.append(TEST_ENRICHMENT_NOTE)
    out.append("")
    extra = []
    if man.get("n_predictions") is not None:
        extra.append(f"{man['n_predictions']} predictions")
    if man.get("status_counts"):
        extra.append(f"status counts {json.dumps(man['status_counts'])}")
    if man.get("triage_counts"):
        extra.append(f"triage counts {json.dumps(man['triage_counts'])}")
    if man.get("batch_inference_seconds") is not None:
        extra.append(f"batch inference {f(man['batch_inference_seconds'], 1)} s")
    if extra:
        out.append("Batch run: " + "; ".join(extra) + ".")
        out.append("")

    out.append("## Data and label audit")
    out.append("")
    out += audit_section(ev, audit)
    out.append("")

    out.append("## Model selection trace")
    out.append("")
    sel = selection_section(ev)
    out += sel if sel else ["Not recorded."]
    out.append("")

    out.append("## Notes")
    out.append("")
    notes = ev.get("notes") or []
    for n in notes:
        out.append(f"- {n}")
    for n in prim.get("notes") or []:
        out.append(f"- {n}")
    if not notes and not prim.get("notes"):
        out.append("- none")
    return "\n".join(out) + "\n"


def main(art_dir: Optional[Path | str] = None, out_path: Optional[Path | str] = None) -> Path:
    art = Path(art_dir) if art_dir is not None else ART
    ev = load_json(art / "evaluation.json", {})
    audit = load_json(art / "audit.json", {})
    man = load_json(art / "manifest.json", {})
    text = render(ev if isinstance(ev, dict) else {}, audit if isinstance(audit, dict) else {}, man if isinstance(man, dict) else {})
    target = Path(out_path) if out_path is not None else ROOT / "RESULTS.md"
    target.write_text(text, encoding="utf-8")
    print(f"wrote {target}")
    return target


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args[0] if len(args) > 0 else None, args[1] if len(args) > 1 else None)
