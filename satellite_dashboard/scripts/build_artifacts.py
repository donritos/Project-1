"""Export the dashboard data bundle from the trained collision_risk model.

Run from the satellite_dashboard folder after the model repo has been trained:

    uv run python scripts/build_artifacts.py            # or: python scripts/build_artifacts.py

What it does
1. Loads the shipped model bundle (../artifacts/model) and the event split (../artifacts/splits.json).
2. Runs the SAME forecasting pipeline the model uses (collision_risk.service.forecast_frame) on
   - held-out training events (the 20 % event split the model never trained on), and
   - the unlabelled test file.
   The pipeline drops CDMs under 2 days BEFORE features are built, and every snapshot forecast uses only
   the CDMs available at that snapshot, so the per-snapshot history plotted in the app is leak-free.
3. Checks the regenerated event forecasts against the model's own exported files.
4. Writes model-native frames to artifacts/predictions/<dataset>/ and the held-out outcomes to a
   separate outcomes.parquet (evaluation only; never joined into the prediction path).
5. Copies the model bundle to artifacts/model/ (used for uploads) and writes artifacts/manifest.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_REPO = APP_ROOT.parent

INPUT_COLUMNS = ["time_to_tca", "miss_distance", "relative_speed", "relative_position_r", "relative_position_t",
                 "relative_position_n", "c_object_type", "c_sigma_r", "t_sigma_r", "max_risk_estimate", "F10", "AP", "risk"]
EXPLANATION_COLUMNS = ["explanation_id", "component", "output_units", "baseline", "explained_output", "p_critical",
                       "predicted_tier", "top_positive", "top_negative", "remaining_contribution", "group_contributions", "text"]
BUNDLE_FILES = ["bundle.json", "config_snapshot.json", "model_ge_elevated.txt", "model_critical.txt"]  # model.txt is only used by multiclass bundles


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_workbook_labels(path: Path) -> pd.DataFrame:
    """Official labels workbook. `usable_for_training` is parsed as text ('yes'/'no') explicitly: under
    pandas 3 the column loads as a string dtype, and astype(bool) would turn 'no' into True."""
    lab = pd.read_excel(path, sheet_name="train_event_labels", header=2, engine="openpyxl")
    lab.columns = [str(c).strip() for c in lab.columns]
    lab["usable_for_training"] = lab["usable_for_training"].astype(str).str.strip().str.lower().isin(["yes", "true", "1", "y"])
    lab["final_tier"] = lab["final_tier"].astype(str).str.strip().str.capitalize()
    return lab


def eligible_inputs(df: pd.DataFrame, min_days: float) -> pd.DataFrame:
    t = pd.to_numeric(df["time_to_tca"], errors="coerce")
    keep = t.notna() & (t >= min_days - 1e-12)
    cols = ["dataset_id", "event_id", "row_id"] + [c for c in INPUT_COLUMNS if c in df.columns]
    return df.loc[keep, cols].reset_index(drop=True)


def excluded_counts(df: pd.DataFrame, min_days: float) -> pd.Series:
    t = pd.to_numeric(df["time_to_tca"], errors="coerce")
    return (~(t.notna() & (t >= min_days - 1e-12))).groupby(df["event_id"]).sum().astype(int)


def check_against_reference(regen: pd.DataFrame, ref_path: Path, label: str) -> dict:
    if not ref_path.exists():
        return {"reference": str(ref_path.name), "checked": False}
    ref = pd.read_parquet(ref_path)
    m = regen.merge(ref[["event_id", "p_critical", "predicted_final_tier"]], on="event_id", suffixes=("", "_ref"))
    max_dp = float(np.nanmax(np.abs(m["p_critical"] - m["p_critical_ref"]))) if len(m) else 0.0
    tier_agree = float((m["predicted_final_tier"] == m["predicted_final_tier_ref"]).mean()) if len(m) else 1.0
    out = {"reference": ref_path.name, "checked": True, "n_matched": int(len(m)), "n_reference": int(len(ref)),
           "max_abs_diff_p_critical": max_dp, "tier_agreement": tier_agree}
    if len(m) != len(ref) or tier_agree < 1.0 or max_dp > 1e-6:
        raise RuntimeError(f"{label}: regenerated forecasts disagree with the model's exported file: {out}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-repo", type=Path, default=DEFAULT_MODEL_REPO, help="collision_risk repo root (default: parent folder)")
    ap.add_argument("--train", type=Path, default=None, help="train_data.csv (default: <model-repo>/data/raw/train_data.csv)")
    ap.add_argument("--test", type=Path, default=None, help="test_data.csv (default: <model-repo>/data/raw/test_data.csv)")
    ap.add_argument("--labels", type=Path, default=None, help="challenge1_event_labels.xlsx (optional cross-check)")
    ap.add_argument("--out", type=Path, default=APP_ROOT / "artifacts")
    args = ap.parse_args()

    repo = args.model_repo.resolve()
    sys.path.insert(0, str(repo / "src"))
    from collision_risk.data import load_raw  # noqa: E402
    from collision_risk.service import forecast_frame, load_bundle  # noqa: E402

    train_csv = args.train or repo / "data" / "raw" / "train_data.csv"
    test_csv = args.test or repo / "data" / "raw" / "test_data.csv"
    labels_xlsx = args.labels or repo / "data" / "raw" / "challenge1_event_labels.xlsx"
    bundle_dir = repo / "artifacts" / "model"
    for p in (train_csv, test_csv, bundle_dir / "bundle.json", repo / "artifacts" / "splits.json",
              repo / "artifacts" / "evaluation_labels.parquet"):
        if not Path(p).exists():
            raise FileNotFoundError(f"Required input missing: {p}. Train the model first (collision-risk run-all).")

    bundle = load_bundle(bundle_dir)
    cfg = bundle.cfg
    min_days = float(cfg["horizon"]["min_days"])
    splits = json.loads((repo / "artifacts" / "splits.json").read_text())
    holdout_ids = set(int(e) for e in splits["events"]["holdout"])
    print(f"model {bundle.meta['model_version']} | holdout events {len(holdout_ids)} | horizon >= {min_days} d")

    out = args.out
    pred_dir = out / "predictions"
    out.mkdir(parents=True, exist_ok=True)
    manifest_datasets: dict = {}

    # ----------------------------------------------------------------- held-out training events
    tr, _ = load_raw(train_csv, cfg["dataset_ids"]["train"], cfg)
    tr_h = tr[tr["event_id"].isin(holdout_ids)].reset_index(drop=True)
    labels = pd.read_parquet(repo / "artifacts" / "evaluation_labels.parquet")
    labels = labels[labels["event_id"].isin(holdout_ids)].copy()
    if set(labels["event_id"]) != holdout_ids:
        raise RuntimeError("evaluation_labels.parquet does not cover the holdout split")
    workbook_check = {"present": labels_xlsx.exists()}
    if labels_xlsx.exists():
        wb = read_workbook_labels(labels_xlsx).set_index("event_id")
        cmp_ = labels.set_index("event_id")
        workbook_check.update({
            "tier_mismatches": int((wb.loc[cmp_.index, "final_tier"] != cmp_["final_tier"]).sum()),
            "usable_mismatches": int((wb.loc[cmp_.index, "usable_for_training"] != cmp_["usable_for_training"].astype(bool)).sum()),
        })
        if workbook_check["tier_mismatches"] or workbook_check["usable_mismatches"]:
            raise RuntimeError(f"Labels workbook disagrees with the model's evaluation labels: {workbook_check}")
    out_h = forecast_frame(bundle, tr_h, with_explanations=True, explain_all_snapshots=True)
    chk_h = check_against_reference(out_h.events, repo / "artifacts" / "holdout_forecasts.parquet", "holdout")
    _write_dataset(pred_dir / "heldout", out_h, eligible_inputs(tr_h, min_days), excluded_counts(tr_h, min_days))
    outcomes = labels[["event_id", "final_tier", "final_risk_log10_pc", "final_miss_distance_m", "final_time_to_tca_days",
                       "cdm_count", "cdm_count_2d_or_more", "is_trivial"]].copy()
    outcomes.insert(0, "dataset_id", cfg["dataset_ids"]["train"])
    outcomes.to_parquet(pred_dir / "heldout" / "outcomes.parquet", index=False)
    manifest_datasets["heldout"] = {
        "label": "Held-out training events", "dataset_id": cfg["dataset_ids"]["train"], "has_outcomes": True,
        "n_events": int(out_h.events["event_id"].nunique()), "n_snapshots": int(len(out_h.snapshots)),
        "n_input_rows": int(len(tr_h)), "n_rows_excluded_by_horizon": int(out_h.info["n_rows_below_horizon_dropped"]),
        "reference_check": chk_h, "workbook_check": workbook_check,
        "note": "20 % event split never used for training, tuning or calibration. Outcomes are evaluation-only.",
    }
    print(f"heldout: {manifest_datasets['heldout']['n_events']} events, {len(out_h.snapshots)} snapshots, check {chk_h}")

    # ----------------------------------------------------------------- unlabelled test events
    te, _ = load_raw(test_csv, cfg["dataset_ids"]["test"], cfg)
    out_t = forecast_frame(bundle, te, with_explanations=True, explain_all_snapshots=True)
    chk_t = check_against_reference(out_t.events, repo / "artifacts" / "event_forecasts.parquet", "test")
    _write_dataset(pred_dir / "test", out_t, eligible_inputs(te, min_days), excluded_counts(te, min_days))
    manifest_datasets["test"] = {
        "label": "Unlabelled test events", "dataset_id": cfg["dataset_ids"]["test"], "has_outcomes": False,
        "n_events": int(out_t.events["event_id"].nunique()), "n_snapshots": int(len(out_t.snapshots)),
        "n_input_rows": int(len(te)), "n_rows_excluded_by_horizon": int(out_t.info["n_rows_below_horizon_dropped"]),
        "reference_check": chk_t,
        "note": "No final outcomes are provided for test events; predictions are for demonstration only.",
    }
    print(f"test: {manifest_datasets['test']['n_events']} events, check {chk_t}")
    write_sample_upload(tr, holdout_ids, labels, min_days, APP_ROOT / "data" / "sample_inputs" / "sample_upload.csv")

    # ----------------------------------------------------------------- model bundle copy + evaluation
    model_out = out / "model"
    model_out.mkdir(parents=True, exist_ok=True)
    files = BUNDLE_FILES + (["model.txt"] if bundle.kind == "multiclass" else [])
    for f in files:
        if (bundle_dir / f).exists():
            shutil.copy2(bundle_dir / f, model_out / f)
    for f in ("evaluation.json", "audit.json"):
        if (repo / "artifacts" / f).exists():
            shutil.copy2(repo / "artifacts" / f, out / f)

    sets = out_h.events["prediction_set"].dropna()
    set_sizes = sets.apply(len) if len(sets) else pd.Series(dtype=float)
    sets_informative = bool(len(set_sizes) and (set_sizes > 1).any())
    meta = bundle.meta
    manifest = {
        "schema_version": "dashboard-1.0",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "model": {"model_version": meta["model_version"], "feature_version": meta.get("feature_version"),
                  "policy_version": meta.get("policy_version"), "kind": meta.get("kind"),
                  "model_schema_version": meta.get("schema_version"), "split_hash": meta.get("split_hash"),
                  "use_past_risk": bool(meta.get("use_past_risk")), "thresholds": meta.get("thresholds"),
                  "bundle_sha256": {f: sha256(model_out / f) for f in BUNDLE_FILES if (model_out / f).exists()}},
        "eligibility": {"min_time_to_tca_days": min_days, "rule": f"time_to_tca >= {min_days:g} days"},
        "tier_rules": {"critical": "risk >= -4.0 OR miss_distance < 1000 m",
                       "elevated": "not Critical, and (-6.0 <= risk < -4.0 OR 1000 m <= miss_distance <= 5000 m)",
                       "safe": "risk < -6.0 AND miss_distance > 5000 m",
                       "floor": "risk = -30 is a placeholder for a probability too small to compute (Pc ~ 0)"},
        "split": {"method": "80/20 split by event_id, stratified by final tier x trivial flag; holdout untouched until freeze",
                  "holdout_fraction": splits.get("holdout_fraction"), "split_hash": splits.get("split_hash"),
                  "n_unusable_excluded": splits.get("n_unusable_excluded")},
        "capabilities": {
            "class_probabilities": True,
            "class_probabilities_note": "Calibrated probability of each FINAL TIER. Not a collision probability.",
            "feature_contributions": True,
            "feature_contributions_note": "LightGBM contributions to the Critical-class score, in log-odds.",
            "reason_codes": True,
            "review_recommendation": True,
            "snapshot_history": True,
            "predicted_final_pc": False,
            "predicted_final_miss_distance": False,
            "uncertainty_intervals": False,
            "prediction_sets": sets_informative,
            "prediction_sets_note": None if sets_informative else
                "The model's 90 % prediction sets contain exactly one class for every held-out event in this run, so they add no information and are hidden.",
            "single_cdm_inference": True,
            "single_cdm_inference_note": "Supported, but the forecast then has no history (NO_HISTORY).",
        },
        "datasets": manifest_datasets,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"wrote {out / 'manifest.json'}")
    return 0


def write_sample_upload(tr: pd.DataFrame, holdout_ids: set, labels: pd.DataFrame, min_days: float, path: Path) -> None:
    """A small upload example: four held-out event histories (one per final tier plus one escalation-prone
    event) INCLUDING their CDMs inside the 2-day cutoff, and one event whose CDMs are all inside the cutoff.
    Event ids are renumbered 1..5 because ids only identify events within one file."""
    t = pd.to_numeric(tr["time_to_tca"], errors="coerce")
    n_elig = (t >= min_days).groupby(tr["event_id"]).sum()
    picks = []
    lab = labels.set_index("event_id")
    for tier in ("Critical", "Elevated", "Safe", "Critical"):
        cand = [e for e in lab.index[lab["final_tier"] == tier] if n_elig.get(e, 0) >= 4 and e not in picks]
        if cand:
            picks.append(int(sorted(cand)[len(picks) * 7 % len(cand)]))
    unusable = [int(e) for e, n in n_elig.items() if n == 0 and int(e) not in holdout_ids]
    if unusable:
        picks.append(unusable[0])
    cols = ["event_id"] + [c for c in INPUT_COLUMNS if c in tr.columns]
    sample = tr[tr["event_id"].isin(picks)][cols].copy()
    sample["event_id"] = sample["event_id"].map({e: i + 1 for i, e in enumerate(picks)})
    sample = sample.sort_values(["event_id", "time_to_tca"], ascending=[True, False])
    path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(path, index=False)
    print(f"sample upload: {len(sample)} rows, {sample['event_id'].nunique()} events -> {path}")


def _write_dataset(d: Path, fo, inputs: pd.DataFrame, excluded: pd.Series) -> None:
    d.mkdir(parents=True, exist_ok=True)
    snaps = fo.snapshots.copy()
    events = fo.events.copy()
    events["n_rows_excluded_by_horizon"] = events["event_id"].map(excluded).fillna(0).astype(int)
    for frame in (snaps, events):
        for c in ("quality_flags", "reason_codes", "prediction_set"):
            if c in frame.columns:
                frame[c] = frame[c].apply(lambda v: [] if v is None or (isinstance(v, float) and np.isnan(v)) else [str(x) for x in v])
    snaps.to_parquet(d / "snapshots.parquet", index=False)
    events.to_parquet(d / "events.parquet", index=False)
    inputs.to_parquet(d / "inputs.parquet", index=False)
    if fo.explanations is not None:
        ex = fo.explanations[[c for c in EXPLANATION_COLUMNS if c in fo.explanations.columns]]
        ex.to_parquet(d / "explanations.parquet", index=False)


if __name__ == "__main__":
    raise SystemExit(main())
