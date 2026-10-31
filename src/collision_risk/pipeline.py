"""End-to-end pipeline stages: audit -> train (splits, baselines, search, candidates,
calibration, final evaluation) -> predict/export. Every stage writes artifacts to disk."""
from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import __version__ as PKG_VERSION
from . import features as FE
from . import train as T
from .calibration import CQRCalibrator, SigmoidCalibrator, SplitConformalCalibrator, fix_quantile_crossing, reliability_diagram
from .config import PROJECT_ROOT, load_config, resolve_path
from .contracts import PredictionRecord
from .data import audit as run_data_audit, file_sha256, load_raw, validate_rows
from .evaluate import compare_models, evaluate_predictions, find_illustrative_cases, latest_cdm_view, to_jsonable
from .policy import tier_from_true_risk
from .service import Bundle, load_bundle, predict_frame, save_bundle


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(to_jsonable(obj), fh, indent=2)


# --------------------------------------------------------------------------- #
# Stage 1: audit
# --------------------------------------------------------------------------- #
def stage_audit(cfg: dict) -> dict:
    art = resolve_path(cfg, "artifacts_dir")
    proc = resolve_path(cfg, "processed_dir")
    proc.mkdir(parents=True, exist_ok=True)
    tr, m_tr = load_raw(resolve_path(cfg, "raw_train"), cfg["dataset_ids"]["train"], cfg)
    te, m_te = load_raw(resolve_path(cfg, "raw_test"), cfg["dataset_ids"]["test"], cfg)
    rep = run_data_audit(tr, te, cfg, m_tr, m_te)
    st, fl = validate_rows(tr, cfg)
    rep["train"]["validation"] = {"status_counts": pd.Series(st).value_counts().to_dict(),
                                  "flag_counts": pd.Series([x for f in fl for x in f]).value_counts().to_dict()}
    st, fl = validate_rows(te, cfg)
    rep["test"]["validation"] = {"status_counts": pd.Series(st).value_counts().to_dict(),
                                 "flag_counts": pd.Series([x for f in fl for x in f]).value_counts().to_dict()}
    if not rep["checks"]["max_risk_estimate_matches_config"]:
        raise RuntimeError("Audit: max_risk_estimate encoding does not match configuration; fix config before training.")
    tr.to_parquet(proc / "train.parquet", index=False)
    te.to_parquet(proc / "test.parquet", index=False)
    _write_json(art / "audit.json", rep)
    _log(f"audit written: {art / 'audit.json'}")
    return rep


def _load_processed(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    proc = resolve_path(cfg, "processed_dir")
    if not (proc / "train.parquet").exists():
        stage_audit(cfg)
    return pd.read_parquet(proc / "train.parquet"), pd.read_parquet(proc / "test.parquet")


# --------------------------------------------------------------------------- #
# Stage 2: train
# --------------------------------------------------------------------------- #
def _eval_frame(preds: pd.DataFrame, truth: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Join predictions with ground truth (kept separate in artifacts) for evaluation."""
    df = preds.merge(truth[["row_id", "risk_true"]], on="row_id", how="left")
    df["risk_tier_true"] = tier_from_true_risk(df["risk_true"].to_numpy(float), df["miss_distance_m"].to_numpy(float), cfg["thresholds"])
    return df


def _make_meta(name: str, kind: str, feature_set: str, cfg: dict, splits: dict, **extra) -> dict:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return {
        "name": name, "kind": kind, "model_version": f"{name}-{stamp}", "feature_version": cfg["feature_version"],
        "policy_version": cfg["policy_version"], "feature_set": feature_set, "feature_columns": FE.feature_columns(feature_set),
        "object_type_categories": cfg["features"]["object_type_categories"], "split_hash": splits["split_hash"],
        "trained_at": stamp, "units": cfg["units"], **extra,
    }


def _bundle_in_memory(meta: dict, cfg: dict, booster=None, classifier=None, conformal=None, sigmoid=None, baseline=None) -> Bundle:
    return Bundle(path=Path("."), meta=meta, cfg=cfg, booster=booster, classifier=classifier, conformal=conformal, sigmoid=sigmoid, baseline=baseline)


def stage_train(cfg: dict, *, quick: bool = False) -> dict:
    t_start = time.time()
    art = resolve_path(cfg, "artifacts_dir")
    art.mkdir(parents=True, exist_ok=True)
    if quick:
        cfg = json.loads(json.dumps(cfg))
        cfg["model"].update({"n_search": 2, "cv_folds": 3, "max_rounds": 150})
    tgt = cfg["target_column"]
    train_raw, test_raw = _load_processed(cfg)

    # ---- splits
    splits = T.make_event_splits(train_raw, cfg)
    _write_json(art / "splits.json", splits)
    _log(f"splits: { {k: v['n_events'] for k, v in splits['counts'].items()} }")
    F = FE.build_features(train_raw, cfg, feature_set="history")
    F["split"] = T.attach_split(F, splits).to_numpy()
    part = {name: F[F["split"] == name].copy() for name in T.SPLIT_NAMES}
    truth_train = pd.DataFrame({"row_id": train_raw["row_id"], "risk_true": train_raw[tgt]})
    truth_test = pd.DataFrame({"row_id": test_raw["row_id"], "risk_true": test_raw[tgt]}) if tgt in test_raw.columns else None

    dev = part["dev"]
    y_dev = dev[tgt].to_numpy(float)
    groups_dev = dev["event_id"].to_numpy()

    # ---- baselines (fit on dev)
    median_b = T.MedianBaseline().fit(y_dev)
    linear_b = T.LinearBaseline().fit(dev, y_dev)

    # ---- hyperparameter search on the history feature set (dev only, grouped CV)
    X_dev_hist = FE.feature_matrix(dev, "history")
    _log(f"grouped CV search: {cfg['model']['n_search']} configs x {cfg['model']['cv_folds']} folds")
    search = T.grouped_cv_search(X_dev_hist, y_dev, groups_dev, cfg)
    best = search["best"]
    _log(f"best params: {best['params']} scores={best['scores']}")
    n_rounds = max(int(best["mean_best_iteration"]), 20)

    # ---- candidate point models trained on all of dev with the selected params
    candidates: dict[str, dict] = {}

    def add_candidate(name: str, feature_set: str, *, params: dict | None = None, weights: np.ndarray | None = None,
                      monotone: dict | None = None, note: str = ""):
        params = params or best["params"]
        X = FE.feature_matrix(dev, feature_set)
        t0 = time.time()
        booster, _ = T.fit_lgb(X, y_dev, params, cfg, w=weights, num_rounds=n_rounds, monotone=monotone)
        meta = _make_meta(name, "lgbm", feature_set, cfg, splits, params=params, num_iteration=n_rounds,
                          weighted=weights is not None, monotone=monotone or None, note=note,
                          feature_ranges=FE.compute_feature_ranges(X), train_seconds=round(time.time() - t0, 2))
        b = _bundle_in_memory(meta, cfg, booster=booster)
        b.interval_method = cfg["interval"]["method"]
        if b.interval_method == "cqr":
            b.quantile_heads = T.fit_quantile_heads(X, y_dev, params, cfg, num_rounds=n_rounds, alphas=tuple(cfg["interval"]["quantile_alphas"]))
        candidates[name] = {"bundle": b, "meta": meta}
        _log(f"trained {name} (+{len(b.quantile_heads)} quantile heads) in {round(time.time() - t0, 2)}s")

    add_candidate("lgbm_single_cdm", "single_cdm", note="allowed single-CDM telemetry + object type")
    add_candidate("lgbm_physics", "physics", note="+ physics-inspired features")
    add_candidate("lgbm_history", "history", note="+ causal history features")
    add_candidate("lgbm_history_weighted", "history", weights=T.risk_weights(y_dev, cfg), note="capped risk-aware weights")
    # Monotone constraints are not supported with the L1 objective in LightGBM, so the experiment is a
    # clean pair of L2 models with identical params: unconstrained vs constrained on individual columns.
    l2_params = {**best["params"], "objective": "l2"}
    add_candidate("lgbm_history_l2", "history", params=l2_params, note="L2 objective, same params (control for monotone experiment)")
    add_candidate("lgbm_history_l2_monotone", "history", params=l2_params, monotone=T.default_monotone(FE.feature_columns("history")),
                  note="experimental monotone constraints (max_risk_estimate +, miss distance -) on an L2 objective")

    for name, b in (("baseline_median", median_b), ("baseline_linear", linear_b)):
        kind = "median" if name.endswith("median") else "linear"
        meta = _make_meta(name, kind, "physics", cfg, splits, feature_ranges={})
        candidates[name] = {"bundle": _bundle_in_memory(meta, cfg, baseline=b), "meta": meta}

    # ---- calibrate conformal intervals for EVERY candidate on conformal_cal (full pipeline incl. postprocessing)
    cal = part["conformal_cal"]
    for name, c in candidates.items():
        b = c["bundle"]
        out = predict_frame(b, cal, with_explanations=False).predictions
        ef = _eval_frame(out, truth_train, cfg)
        y_cal = ef["risk_true"].to_numpy(float); pred_cal = ef["risk_log10_pred"].to_numpy(float); ev_cal = ef["event_id"].to_numpy()
        conf = SplitConformalCalibrator(coverage=float(cfg["conformal"]["coverage"]), min_events=int(cfg["conformal"]["min_calibration_events"]))
        conf.fit(y_cal, pred_cal, ev_cal)
        b.conformal = conf
        c["conformal"] = conf.to_dict()
        if b.quantile_heads:
            Xc = FE.feature_matrix(cal, b.feature_set)
            alphas = sorted(b.quantile_heads)
            q_lo = T.postprocess(b.quantile_heads[alphas[0]].predict(Xc), cfg)
            q_hi = T.postprocess(b.quantile_heads[alphas[-1]].predict(Xc), cfg)
            q_lo, q_hi = fix_quantile_crossing(pred_cal, q_lo, q_hi)
            cqr = CQRCalibrator(coverage=float(cfg["conformal"]["coverage"]), min_events=int(cfg["conformal"]["min_calibration_events"]))
            cqr.fit(y_cal, q_lo, q_hi, ev_cal)
            b.cqr = cqr
            c["cqr"] = cqr.to_dict()
        else:
            b.interval_method = "residual"
    _log("interval calibrators fitted on conformal_cal")

    # ---- choose primary candidate on policy_cal (eval stays untouched)
    pc = part["policy_cal"]
    selection_rows = []
    for name, c in candidates.items():
        if c["bundle"].kind != "lgbm":
            continue
        out = predict_frame(c["bundle"], pc, with_explanations=False).predictions
        ef = _eval_frame(out, truth_train, cfg)
        sc = T._oof_scores(ef["risk_true"].to_numpy(float), ef["risk_log10_pred"].to_numpy(float), cfg)
        selection_rows.append({"name": name, **sc})
    sel = pd.DataFrame(selection_rows)
    sel["rank_mae_nf"] = sel["mae_non_floor"].rank()
    sel["rank_f2"] = sel["critical_f2_pc_only"].fillna(0).rank(ascending=False)
    sel["selection_score"] = 0.5 * sel["rank_mae_nf"] + 0.5 * sel["rank_f2"]
    primary_name = str(sel.sort_values(["selection_score", "mae_non_floor"]).iloc[0]["name"])
    _log(f"primary model: {primary_name}")
    primary = candidates[primary_name]

    # ---- optional classifier P(risk >= -4): trained on dev (grouped internal holdout for early stopping),
    #      sigmoid-calibrated on policy_cal, disagreement threshold tuned on policy_cal
    fs = primary["bundle"].feature_set
    X_dev = FE.feature_matrix(dev, fs)
    y_bin = (y_dev >= cfg["thresholds"]["critical_risk"]).astype(int)
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=cfg["split"]["seed"])
    tr_i, va_i = next(gss.split(X_dev, y_bin, groups_dev))
    clf, clf_iter = T.train_classifier(X_dev.iloc[tr_i], y_bin[tr_i], cfg, X_val=X_dev.iloc[va_i], y_val=y_bin[va_i])
    X_pc = FE.feature_matrix(pc, fs)
    s_pc = clf.predict(X_pc, num_iteration=clf_iter)
    y_pc = (pc[tgt].to_numpy(float) >= cfg["thresholds"]["critical_risk"]).astype(int)
    sigmoid = SigmoidCalibrator().fit(s_pc, y_pc)
    p_pc = sigmoid.predict_proba(s_pc)
    # threshold: maximise F2 for critical detection by classifier alone on policy_cal (fallback to cfg)
    best_thr, best_f2 = float(cfg["triage"]["classifier_disagreement_prob"]), -1.0
    for thr in np.linspace(0.05, 0.95, 19):
        tp = ((p_pc >= thr) & (y_pc == 1)).sum(); fp = ((p_pc >= thr) & (y_pc == 0)).sum(); fn = ((p_pc < thr) & (y_pc == 1)).sum()
        if tp + fn == 0:
            break
        prec = tp / max(tp + fp, 1); rec = tp / (tp + fn)
        f2 = 5 * prec * rec / (4 * prec + rec) if (4 * prec + rec) > 0 else 0.0
        if f2 > best_f2:
            best_f2, best_thr = float(f2), float(thr)
    classifier_info = {"num_iteration": clf_iter, "threshold": best_thr, "policy_cal_f2_at_threshold": best_f2,
                       "n_pos_policy_cal": int(y_pc.sum()), "reliability_policy_cal": reliability_diagram(y_pc, p_pc)}
    primary["bundle"].classifier = clf
    primary["bundle"].sigmoid = sigmoid
    primary["meta"]["classifier_num_iteration"] = clf_iter
    primary["meta"]["classifier_threshold"] = best_thr
    _log(f"classifier threshold {best_thr:.2f} (policy_cal F2={best_f2:.3f})")

    # ---- FREEZE: final evaluation on eval partition for every candidate
    ev = part["eval"]
    results: dict[str, dict] = {}
    pred_frames: dict[str, pd.DataFrame] = {}
    for name, c in candidates.items():
        t0 = time.time()
        out = predict_frame(c["bundle"], ev, with_explanations=False)
        ef = _eval_frame(out.predictions, truth_train, cfg)
        res = evaluate_predictions(ef, cfg, scope={"partition": "eval", "dataset": cfg["dataset_ids"]["train"], "model": name,
                                                   "n_events": int(ev["event_id"].nunique())})
        res["batch_inference_seconds"] = round(time.time() - t0, 3)
        res["timing_breakdown_seconds"] = out.timing_seconds
        res["conformal"] = c["conformal"]
        res["cqr"] = c.get("cqr")
        res["interval_method"] = c["bundle"].interval_method
        results[name] = res
        pred_frames[name] = ef

    # interval-method comparison for the primary model (same eval partition)
    interval_comparison = {}
    pb = candidates[primary_name]["bundle"]
    for method in ("residual", "cqr"):
        if method == "cqr" and pb.cqr is None:
            continue
        saved = pb.interval_method
        pb.interval_method = method
        out = predict_frame(pb, ev, with_explanations=False)
        ef = _eval_frame(out.predictions, truth_train, cfg)
        r = evaluate_predictions(ef, cfg, n_bootstrap=50)
        interval_comparison[method] = {view: {"coverage": r[view]["interval"]["coverage"], "median_width": r[view]["interval"]["median_width"],
                                              "auto_clear_fraction": r[view]["triage"]["auto_clear_fraction"],
                                              "human_review_fraction": r[view]["triage"]["human_review_fraction"],
                                              "true_critical_routed_to_auto_clear_n": r[view]["triage"]["true_critical_routed_to_auto_clear_n"]}
                                       for view in ("row_level", "latest_cdm")}
        pb.interval_method = saved
    comparison = compare_models(results, view="latest_cdm")
    comparison_rows = compare_models(results, view="row_level")

    # ---- external check on the delivered test file (labelled, horizon-restricted) for primary + baselines
    test_results: dict[str, dict] = {}
    if truth_test is not None:
        for name in [primary_name, "baseline_median", "baseline_linear"]:
            out = predict_frame(candidates[name]["bundle"], test_raw, with_explanations=False)
            ef = _eval_frame(out.predictions, truth_test, cfg)
            test_results[name] = evaluate_predictions(ef, cfg, scope={"partition": "delivered_test_file", "dataset": cfg["dataset_ids"]["test"], "model": name,
                                                                      "note": "labels present in delivered test file; rows stop at >= 2 days before TCA"})

    cases = find_illustrative_cases(pred_frames[primary_name], k=3)

    # ---- persist primary bundle (the evaluated model is the demo model; no retraining after freeze)
    bundle_dir = art / "model"
    save_bundle(bundle_dir, meta=primary["meta"], cfg=cfg, booster=primary["bundle"].booster, classifier=clf,
                conformal=primary["bundle"].conformal, sigmoid=sigmoid, quantile_heads=primary["bundle"].quantile_heads,
                cqr=primary["bundle"].cqr, interval_method=primary["bundle"].interval_method)
    # save baselines + other candidates for reference (no classifier)
    for name, c in candidates.items():
        if name == primary_name:
            continue
        save_bundle(art / "candidates" / name, meta=c["meta"], cfg=cfg, booster=c["bundle"].booster, classifier=None,
                    conformal=c["bundle"].conformal, sigmoid=None, baseline=c["bundle"].baseline, quantile_heads=c["bundle"].quantile_heads,
                    cqr=c["bundle"].cqr, interval_method=c["bundle"].interval_method)

    # ---- evaluation artifact (labels stay in a separate file)
    labels = pred_frames[primary_name][["row_id", "event_id", "time_to_tca_days", "risk_true", "risk_tier_true"]].copy()
    labels.to_parquet(art / "evaluation_labels.parquet", index=False)
    evaluation = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "primary_model": primary_name,
        "model_version": primary["meta"]["model_version"],
        "scope": {"evaluation_partition": "eval (15% of training events, untouched until freeze)", "split_hash": splits["split_hash"],
                  "split_counts": splits["counts"], "quick_mode": quick},
        "selection": {"cv_search": search, "candidate_selection_on_policy_cal": sel.to_dict(orient="records"),
                      "selection_rule": "0.5*rank(MAE non-floor) + 0.5*rank(Pc-only Critical F2) on policy_cal"},
        "classifier": classifier_info,
        "interval_method": pb.interval_method,
        "interval_method_comparison_primary": interval_comparison,
        "models": results,
        "comparison_latest_cdm": comparison.reset_index().to_dict(orient="records"),
        "comparison_row_level": comparison_rows.reset_index().to_dict(orient="records"),
        "delivered_test_file": test_results,
        "illustrative_cases": cases,
        "notes": [
            ("Intervals are conformalized quantile regression intervals (0.05/0.95 LightGBM quantile heads, crossing-corrected, shifted by the finite-sample quantile of per-event max scores)"
             if pb.interval_method == "cqr" else
             "Intervals are split-conformal residual intervals around the point prediction, one max-|residual| score per calibration event")
            + " around the dataset risk target (log10 Pc); they describe uncertainty of the dataset label, not every physical uncertainty of an encounter.",
            "Coverage is reported empirically on evaluation events; no universal or subgroup guarantee is claimed.",
            "Pc-only metrics isolate model skill from the deterministic distance rule.",
            "Zero observed misses in a subset would not imply zero operational risk; see event-bootstrap intervals.",
        ],
        "total_train_seconds": round(time.time() - t_start, 1),
    }
    _write_json(art / "evaluation.json", evaluation)
    _log(f"evaluation written ({evaluation['total_train_seconds']}s total)")
    return evaluation


# --------------------------------------------------------------------------- #
# Stage 3: predict / export
# --------------------------------------------------------------------------- #
def stage_predict(cfg: dict, *, input_path: str | Path | None = None, dataset_id: str | None = None, out_dir: str | Path | None = None,
                  bundle_path: str | Path | None = None) -> dict:
    art = resolve_path(cfg, "artifacts_dir")
    out_dir = Path(out_dir) if out_dir else art
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle(bundle_path or (art / "model"))
    input_path = Path(input_path) if input_path else resolve_path(cfg, "raw_test")
    dataset_id = dataset_id or cfg["dataset_ids"]["test"]
    df, meta = load_raw(input_path, dataset_id, cfg)
    t0 = time.time()
    out = predict_frame(bundle, df, with_explanations=True)
    secs = time.time() - t0
    preds = out.predictions
    assert len(preds) == len(df) and (preds["source_row_index"].to_numpy() == np.arange(len(df))).all()
    # contract validation on a sample + all invalid rows
    sample_idx = list(range(min(200, len(preds)))) + list(np.flatnonzero(preds["prediction_status"].to_numpy() == "invalid")[:200])
    for i in sorted(set(sample_idx)):
        PredictionRecord.model_validate(_row_to_record(preds.iloc[i]))

    preds.to_parquet(out_dir / "predictions.parquet", index=False)
    events = _events_view(preds)
    events.to_parquet(out_dir / "events.parquet", index=False)
    if out.explanations is not None:
        out.explanations.to_parquet(out_dir / "explanations.parquet", index=False)
    sub = preds[["row_id", "source_row_index", "event_id", "time_to_tca_days", "miss_distance_m", "risk_log10_pred", "pc_pred",
                 "risk_log10_lower", "risk_log10_upper", "risk_tier", "triage_action", "prediction_status"]].copy()
    sub = sub.rename(columns={"risk_log10_pred": "risk"})
    sub.to_csv(out_dir / "submission.csv", index=False)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(), "package_version": PKG_VERSION, "python": platform.python_version(),
        "lightgbm": lgb.__version__, "pandas": pd.__version__, "numpy": np.__version__,
        "schema_version": preds["schema_version"].iloc[0] if len(preds) else None,
        "model_version": bundle.model_version, "feature_version": bundle.meta["feature_version"], "policy_version": cfg["policy_version"],
        "feature_set": bundle.feature_set, "units": cfg["units"], "max_risk_estimate_encoding": cfg["max_risk_estimate_encoding"],
        "thresholds": cfg["thresholds"], "triage": {**cfg["triage"], "classifier_disagreement_prob_effective": bundle.meta.get("classifier_threshold")},
        "conformal": bundle.meta.get("conformal"), "split_hash": bundle.meta.get("split_hash"),
        "input": {"dataset_id": dataset_id, **meta}, "n_predictions": int(len(preds)),
        "status_counts": preds["prediction_status"].value_counts().to_dict(), "tier_counts": preds["risk_tier"].value_counts(dropna=False).to_dict(),
        "triage_counts": preds["triage_action"].value_counts().to_dict(), "batch_inference_seconds": round(secs, 3),
        "timing_breakdown_seconds": out.timing_seconds,
        "artifacts": {f: file_sha256(out_dir / f) for f in ["predictions.parquet", "events.parquet", "explanations.parquet", "submission.csv"] if (out_dir / f).exists()},
        "config_path": cfg.get("_config_path"),
    }
    _write_json(out_dir / "manifest.json", manifest)
    _log(f"predictions exported to {out_dir} ({len(preds)} rows, {secs:.2f}s)")
    return manifest


def _row_to_record(row: pd.Series) -> dict:
    d = row.to_dict()
    for k, v in list(d.items()):
        if isinstance(v, float) and np.isnan(v):
            d[k] = None
        elif isinstance(v, np.generic):
            d[k] = v.item()
        elif isinstance(v, np.ndarray):
            d[k] = v.tolist()
    return d


def _events_view(preds: pd.DataFrame) -> pd.DataFrame:
    """Latest available CDM per (dataset_id, event_id) plus simple per-event counters."""
    latest = latest_cdm_view(preds.assign(_key=preds["dataset_id"].astype(str) + ":" + preds["event_id"].astype(str)).rename(columns={"event_id": "_event_id"}).rename(columns={"_key": "event_id"}))
    latest = latest.rename(columns={"event_id": "event_key", "_event_id": "event_id"})
    g = preds.groupby(["dataset_id", "event_id"]).agg(n_cdms=("row_id", "size"), first_time_to_tca_days=("time_to_tca_days", "max"),
                                                       max_risk_log10_pred=("risk_log10_pred", "max"), min_miss_distance_m=("miss_distance_m", "min")).reset_index()
    sev = preds["risk_tier"].map({"Critical": 2, "Elevated": 1, "Safe": 0})
    worst = preds.assign(_sev=sev).groupby(["dataset_id", "event_id"])["_sev"].max().map({2: "Critical", 1: "Elevated", 0: "Safe"}).rename("worst_tier_any_cdm").reset_index()
    ev = latest.drop(columns=["event_key"]).merge(g, on=["dataset_id", "event_id"], how="left").merge(worst, on=["dataset_id", "event_id"], how="left")
    return ev.reset_index(drop=True)


def run_all(cfg: dict | None = None, *, quick: bool = False) -> None:
    cfg = cfg or load_config()
    stage_audit(cfg)
    stage_train(cfg, quick=quick)
    stage_predict(cfg)
