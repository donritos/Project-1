"""End-to-end stages for the final-tier forecasting task: audit -> train -> predict/export.
Every stage writes artifacts to disk. No metric is ever computed on test_data."""
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
from sklearn.model_selection import GroupKFold

from . import __version__ as PKG_VERSION
from . import features as FE
from . import train as T
from .calibration import LACPredictionSets, MulticlassIsotonicCalibrator, prediction_set_metrics, tune_decision_thresholds
from .config import PROJECT_ROOT, load_config, resolve_path
from .contracts import EventForecastRecord
from .data import apply_horizon, audit as run_data_audit, file_sha256, load_event_labels, load_raw, validate_rows
from .evaluate import compare_models, evaluate_forecasts, find_illustrative_cases, probability_metrics, to_jsonable
from .policy import tune_policy_thresholds
from .service import ForecastBundle, assemble_forecast_frame, forecast_frame, load_bundle, save_bundle


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
    for name, df in (("train", tr), ("test", te)):
        st, fl = validate_rows(df, cfg)
        rep[name]["validation"] = {"status_counts": pd.Series(st).value_counts().to_dict(),
                                   "flag_counts": pd.Series([x for f in fl for x in f]).value_counts().to_dict()}
    if not rep["checks"]["max_risk_estimate_matches_config"]:
        raise RuntimeError("Audit: max_risk_estimate encoding does not match configuration.")
    labels, info = load_event_labels(cfg)
    if not info["labels_match_recomputed"]:
        raise RuntimeError(f"Audit: event labels disagree with recomputed final tiers: {info}")
    horizon = float(cfg["horizon"]["min_days"])
    rep["label_audit"] = info
    rep["horizon"] = {"min_days": horizon, "test_min_time_to_tca": float(te["time_to_tca"].min()),
                      "test_all_rows_at_or_beyond_horizon": bool(te["time_to_tca"].min() >= horizon - 1e-9),
                      "train_rows_at_or_beyond_horizon": int((tr["time_to_tca"] >= horizon).sum())}
    if not rep["horizon"]["test_all_rows_at_or_beyond_horizon"]:
        raise RuntimeError("Audit: test rows below the 2-day horizon found.")
    rep["checks"]["event_ids_restart_per_file"] = True
    rep["checks"]["labels_match_recomputed"] = True
    tr.to_parquet(proc / "train.parquet", index=False)
    te.to_parquet(proc / "test.parquet", index=False)
    labels.to_parquet(proc / "event_labels.parquet", index=False)
    _write_json(art / "audit.json", rep)
    _log(f"audit written: usable events {info['n_usable']} (trivial {info['n_trivial']}, real_forecast {info['n_real_forecast']}), unusable {info['n_unusable']}")
    return rep


def _load_processed(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    proc = resolve_path(cfg, "processed_dir")
    if not (proc / "train.parquet").exists() or not (proc / "event_labels.parquet").exists():
        stage_audit(cfg)
    return pd.read_parquet(proc / "train.parquet"), pd.read_parquet(proc / "test.parquet"), pd.read_parquet(proc / "event_labels.parquet")


# --------------------------------------------------------------------------- #
# Stage 2: train
# --------------------------------------------------------------------------- #
def _meta(name: str, kind: str, feature_set: str, cfg: dict, split: dict, **extra) -> dict:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return {"name": name, "kind": kind, "model_version": f"{name}-{stamp}", "feature_version": cfg["feature_version"],
            "policy_version": cfg["policy_version"], "feature_set": feature_set, "feature_columns": FE.feature_columns(feature_set),
            "object_type_categories": cfg["features"]["object_type_categories"], "split_hash": split["split_hash"],
            "horizon_min_days": cfg["horizon"]["min_days"], "use_past_risk": cfg["features"].get("use_past_risk", True),
            "trained_at": stamp, **extra}


def _truth_merge(frame: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    lab = labels[["event_id", "final_tier", "final_risk_log10_pc", "final_miss_distance_m", "is_trivial", "mission_id"]]
    return frame.merge(lab, on="event_id", how="left")


def _finish_learned_candidate(name: str, kind: str, feature_set: str, cfg: dict, split: dict, dev: pd.DataFrame, X: pd.DataFrame,
                              y: np.ndarray, w: np.ndarray, groups: np.ndarray, params: dict, fit_mask: np.ndarray | None,
                              status: np.ndarray, flags: list[list[str]], labels: pd.DataFrame, *, n_folds: int, quick: bool) -> tuple[ForecastBundle, dict]:
    """OOF -> calibrator -> decision thresholds -> policy thresholds -> prediction sets -> final fit."""
    latest = dev["is_latest_snapshot"].to_numpy(bool)
    if kind == "multiclass":
        oof, iters = T.grouped_oof_multiclass(X, y, groups, w, params, cfg, n_folds=n_folds, fit_mask=fit_mask)
        it = {"num_iteration": max(int(np.mean(iters)), 10)}
    elif kind == "ordinal":
        oof, ie, ic = T.grouped_oof_ordinal(X, y, groups, w, params, cfg, n_folds=n_folds, fit_mask=fit_mask)
        it = {"num_iteration_ge_elev": max(int(np.mean(ie)), 10), "num_iteration_crit": max(int(np.mean(ic)), 10)}
    else:  # logistic
        oof = np.full((len(y), 3), np.nan)
        for tr_i, va_i in GroupKFold(n_splits=n_folds).split(X, y, groups):
            fm = tr_i if fit_mask is None else tr_i[fit_mask[tr_i]]
            oof[va_i] = T.LogisticBaseline().fit(dev.iloc[fm], y[fm], w[fm]).predict_proba(dev.iloc[va_i])
        it = {}
    cal = MulticlassIsotonicCalibrator().fit(oof[latest], y[latest])
    oof_cal = cal.predict_proba(oof)
    thr = tune_decision_thresholds(oof_cal[latest], y[latest])
    sets_model = LACPredictionSets(coverage=float(cfg["conformal_sets"]["coverage"])).fit(oof_cal[latest], y[latest]) if cfg["conformal_sets"].get("enabled", True) else None
    meta = _meta(name, kind, feature_set, cfg, split, params=params, feature_ranges=FE.compute_feature_ranges(X),
                 training_rows="latest" if fit_mask is not None else "snapshots", **it)
    bundle = ForecastBundle(path=None, meta=meta, cfg=cfg, kind=kind, calibrator=cal, sets=sets_model,
                            thresholds={"t_crit": thr["t_crit"], "t_elev": thr["t_elev"], "t_watch": 1.01, "t_clear": 1.01})
    # policy thresholds on the OOF forecast frame (latest snapshots, with truth)
    oof_frame = assemble_forecast_frame(dev, status, flags, oof_cal, bundle, sets=sets_model.predict_sets(oof_cal) if sets_model else None)
    oof_latest = _truth_merge(oof_frame[oof_frame["is_latest_snapshot"]].reset_index(drop=True), labels)
    pol = tune_policy_thresholds(oof_latest, cfg)
    bundle.thresholds.update({"t_watch": float(pol["t_watch"]), "t_clear": float(pol["t_clear"])})
    # final fit on all dev rows (or latest only)
    fm = np.ones(len(y), bool) if fit_mask is None else fit_mask
    if kind == "multiclass":
        bundle.booster, _ = T.fit_multiclass(X[fm], y[fm], params, cfg, w=w[fm], num_rounds=it["num_iteration"])
    elif kind == "ordinal":
        bundle.booster_ge_elev, _ = T.fit_binary(X[fm], (y[fm] >= 1).astype(int), params, cfg, w=w[fm], num_rounds=it["num_iteration_ge_elev"])
        bundle.booster_crit, _ = T.fit_binary(X[fm], (y[fm] == 2).astype(int), params, cfg, w=w[fm], num_rounds=it["num_iteration_crit"])
    else:
        bundle.logistic = T.LogisticBaseline().fit(dev[fm], y[fm], w[fm])
    oof_eval_mask = latest & ~dev["is_trivial"].to_numpy(bool)
    pred_oof = oof_latest["predicted_final_tier"].to_numpy()
    y_lat = oof_latest["final_tier"].map(T.TIER_INDEX).to_numpy()
    rf = ~oof_latest["is_trivial"].to_numpy(bool)
    sc = T.critical_f2_from_probs(np.eye(3)[pd.Series(pred_oof).map(T.TIER_INDEX).fillna(0).astype(int).to_numpy()][rf], y_lat[rf])
    diag = {"oof_decision": thr, "oof_policy": pol, "oof_real_forecast_at_tuned_thresholds": sc,
            "oof_calibration": probability_metrics(oof_cal[latest], np.array(T.TIERS)[y[latest]]),
            "oof_prediction_sets": prediction_set_metrics(sets_model.predict_sets(oof_cal[oof_eval_mask]), np.array(T.TIERS)[y[oof_eval_mask]]) if sets_model else None,
            "iterations": it}
    return bundle, diag


def _baseline_bundle(name: str, kind: str, cfg: dict, split: dict, dev: pd.DataFrame, status: np.ndarray, flags: list[list[str]], labels: pd.DataFrame) -> tuple[ForecastBundle, dict]:
    meta = _meta(name, kind, "final_tier", cfg, split, feature_ranges={})
    bundle = ForecastBundle(path=None, meta=meta, cfg=cfg, kind=kind)
    X = FE.feature_matrix(dev, "final_tier")
    from .service import raw_probabilities
    P = raw_probabilities(bundle, X, dev)
    frame = assemble_forecast_frame(dev, status, flags, P, bundle)
    latest = _truth_merge(frame[frame["is_latest_snapshot"]].reset_index(drop=True), labels)
    pol = tune_policy_thresholds(latest, cfg)
    bundle.thresholds.update({"t_watch": float(pol["t_watch"]), "t_clear": float(pol["t_clear"])})
    return bundle, {"oof_policy": pol}


def _evaluate_bundle(bundle: ForecastBundle, raw_rows: pd.DataFrame, labels: pd.DataFrame, cfg: dict, *, n_bootstrap: int, scope: dict) -> tuple[dict, pd.DataFrame]:
    out = forecast_frame(bundle, raw_rows, with_explanations=False)
    ev = _truth_merge(out.events, labels)
    res = evaluate_forecasts(ev, cfg, n_bootstrap=n_bootstrap, scope=scope)
    res["timing"] = out.info
    return res, ev


def stage_train(cfg: dict, *, quick: bool = False) -> dict:
    t_start = time.time()
    art = resolve_path(cfg, "artifacts_dir")
    art.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(json.dumps({k: v for k, v in cfg.items()}))
    if quick:
        cfg["model"].update({"n_search": 2, "max_rounds": 300, "early_stopping_rounds": 30})
        cfg["split"]["cv_folds"] = 3
    n_boot = 100 if quick else 1000
    n_folds = int(cfg["split"]["cv_folds"])
    train_raw, _test_raw, labels = _load_processed(cfg)
    labels = labels.copy()

    split = T.make_holdout_split(labels, cfg)
    _write_json(art / "splits.json", split)
    _log(f"split: dev {split['counts']['dev']['n_events']} / holdout {split['counts']['holdout']['n_events']} events; {split['n_unusable_excluded']} unusable excluded")

    horizon = float(cfg["horizon"]["min_days"])
    h_all = apply_horizon(train_raw, horizon)
    dev_raw = h_all[h_all["event_id"].isin(split["events"]["dev"])].reset_index(drop=True)
    hold_raw = h_all[h_all["event_id"].isin(split["events"]["holdout"])].reset_index(drop=True)

    # dev snapshots (training frame)
    status, flags = validate_rows(dev_raw, cfg)
    dev = FE.build_event_snapshots(dev_raw, cfg)
    dev = dev.merge(labels[["event_id", "final_tier", "is_trivial"]], on="event_id", how="left")
    y = dev["final_tier"].map(T.TIER_INDEX).to_numpy()
    groups = dev["event_id"].to_numpy()
    w = T.snapshot_weights(dev, y, cfg)
    latest = dev["is_latest_snapshot"].to_numpy(bool)
    eval_mask = latest & ~dev["is_trivial"].to_numpy(bool)
    fs = FE.final_tier_feature_set(cfg)
    X = FE.feature_matrix(dev, fs)

    # ---- hyperparameter search (grouped OOF on dev; selection on real_forecast latest snapshots)
    _log(f"search: {cfg['model']['n_search']} configs x {n_folds} folds")
    search = T.multiclass_search(X, y, groups, w, cfg, eval_mask=eval_mask)
    params = search["best"]["params"]
    _log(f"best params {params} -> {search['best']['scores']}")

    # ---- training rows: snapshots vs latest-only
    rows_choice = {"snapshots": search["best"]["scores"]}
    oof_lat, _ = T.grouped_oof_multiclass(X, y, groups, w, params, cfg, n_folds=n_folds, fit_mask=latest)
    rows_choice["latest"] = {**T.critical_f2_from_probs(oof_lat[eval_mask], y[eval_mask]), "macro_f1": T.macro_f1_from_probs(oof_lat[eval_mask], y[eval_mask])}
    mode = cfg["model"].get("training_rows", "auto")
    if mode == "auto":
        chosen = "latest" if (rows_choice["latest"]["critical_f2"] or 0) > (rows_choice["snapshots"]["critical_f2"] or 0) else "snapshots"
    else:
        chosen = mode
    fit_mask = latest if chosen == "latest" else None
    _log(f"training rows: {chosen} (OOF critical F2 snapshots={rows_choice['snapshots']['critical_f2']:.3f}, latest={rows_choice['latest']['critical_f2']:.3f})")

    # ---- candidates
    candidates: dict[str, ForecastBundle] = {}
    diagnostics: dict[str, dict] = {}
    for name, kind in (("lgbm_multiclass", "multiclass"), ("lgbm_ordinal", "ordinal"), ("logistic", "logistic")):
        t0 = time.time()
        b, d = _finish_learned_candidate(name, kind, fs, cfg, split, dev, X, y, w, groups, params, fit_mask, status, flags, labels, n_folds=n_folds, quick=quick)
        candidates[name], diagnostics[name] = b, d
        _log(f"{name}: OOF real_forecast {d['oof_real_forecast_at_tuned_thresholds']} thresholds {b.thresholds} ({time.time() - t0:.1f}s)")
    for name, kind in (("carry_forward_risk", "carry_forward_risk"), ("carry_forward_mre", "carry_forward_mre"), ("majority", "majority")):
        candidates[name], diagnostics[name] = _baseline_bundle(name, kind, cfg, split, dev, status, flags, labels)

    # primary: best OOF Critical F2 (real_forecast) among learned LightGBM models, tie-break macro-F1 from search
    def _score(n):
        s = diagnostics[n]["oof_real_forecast_at_tuned_thresholds"]
        return (s["critical_f2"] or 0.0)
    primary_name = max(["lgbm_multiclass", "lgbm_ordinal"], key=_score)
    primary = candidates[primary_name]
    _log(f"primary: {primary_name}")

    # ---- FREEZE. Holdout evaluation (once) for every candidate
    results: dict[str, dict] = {}
    ev_frames: dict[str, pd.DataFrame] = {}
    for name, b in candidates.items():
        res, ev = _evaluate_bundle(b, hold_raw, labels, cfg, n_bootstrap=n_boot, scope={"partition": "holdout", "model": name, "n_events": int(hold_raw["event_id"].nunique())})
        results[name], ev_frames[name] = res, ev
    conformal = None
    if primary.sets is not None:
        evp = ev_frames[primary_name]
        rf = ~evp["is_trivial"].to_numpy(bool)
        conformal = {"coverage_nominal": primary.sets.coverage, "q_hat": primary.sets.q_hat,
                     "holdout_all_usable": prediction_set_metrics(evp["prediction_set"].tolist(), evp["final_tier"].to_numpy()),
                     "holdout_real_forecast": prediction_set_metrics(evp.loc[rf, "prediction_set"].tolist(), evp.loc[rf, "final_tier"].to_numpy())}

    # ---- ablations (primary kind, best params; each gets its own OOF calibration; evaluated on holdout)
    ablations: dict[str, dict] = {}
    abl_specs = [("use_past_risk_off", {"use_past_risk": False}, "final_tier_no_past_risk", fit_mask),
                 ("latest_only_training" if chosen == "snapshots" else "snapshot_training", {}, fs, (latest if chosen == "snapshots" else None)),
                 ("no_history_features", {}, "final_tier_no_history", fit_mask)]
    for aname, feat_over, afs, afm in abl_specs:
        acfg = json.loads(json.dumps(cfg)); acfg["features"].update(feat_over)
        adev = dev if not feat_over else FE.build_event_snapshots(dev_raw.drop(columns=["risk"]) if not feat_over.get("use_past_risk", True) else dev_raw, acfg).merge(labels[["event_id", "final_tier", "is_trivial"]], on="event_id", how="left")
        aX = FE.feature_matrix(adev, afs)
        ab, ad = _finish_learned_candidate(f"ablation_{aname}", primary.kind, afs, acfg, split, adev, aX, y, w, groups, params, afm, status, flags, labels, n_folds=n_folds, quick=quick)
        ares, _ = _evaluate_bundle(ab, hold_raw, labels, acfg, n_bootstrap=min(n_boot, 200), scope={"partition": "holdout", "ablation": aname})
        ares["oof"] = ad["oof_real_forecast_at_tuned_thresholds"]
        ablations[aname] = ares
        _log(f"ablation {aname}: holdout real_forecast critical recall {ares['subsets']['real_forecast']['classification'].get('critical_recall')}")

    # ---- mission-held-out robustness check (report only): GroupKFold by mission over all usable events
    all_snap = FE.build_event_snapshots(h_all, cfg).merge(labels[["event_id", "final_tier", "is_trivial", "mission_id"]], on="event_id", how="left")
    ya = all_snap["final_tier"].map(T.TIER_INDEX).to_numpy(); wa = T.snapshot_weights(all_snap, ya, cfg)
    Xa = FE.feature_matrix(all_snap, fs)
    m_oof = np.full((len(ya), 3), np.nan)
    fm_all = np.ones(len(ya), bool) if fit_mask is None else all_snap["is_latest_snapshot"].to_numpy(bool)
    n_m_folds = min(5, int(all_snap["mission_id"].nunique()))
    for tr_i, va_i in GroupKFold(n_splits=n_m_folds).split(Xa, ya, all_snap["mission_id"].to_numpy()):
        tf = tr_i[fm_all[tr_i]]
        if primary.kind == "multiclass":
            bm, best = T.fit_multiclass(Xa.iloc[tf], ya[tf], params, cfg, w=wa[tf], num_rounds=primary.meta.get("num_iteration"))
            m_oof[va_i] = bm.predict(Xa.iloc[va_i])
        else:
            be, _ = T.fit_binary(Xa.iloc[tf], (ya[tf] >= 1).astype(int), params, cfg, w=wa[tf], num_rounds=primary.meta.get("num_iteration_ge_elev"))
            bc, _ = T.fit_binary(Xa.iloc[tf], (ya[tf] == 2).astype(int), params, cfg, w=wa[tf], num_rounds=primary.meta.get("num_iteration_crit"))
            m_oof[va_i] = T.ordinal_probabilities(be.predict(Xa.iloc[va_i]), bc.predict(Xa.iloc[va_i]))
    m_cal = primary.calibrator.predict_proba(m_oof)
    st_a, fl_a = validate_rows(h_all, cfg)
    m_frame = assemble_forecast_frame(all_snap, st_a, fl_a, m_cal, primary, sets=None)
    m_latest = _truth_merge(m_frame[m_frame["is_latest_snapshot"]].reset_index(drop=True), labels)
    mission_cv = evaluate_forecasts(m_latest, cfg, n_bootstrap=min(n_boot, 200), scope={"check": "mission_held_out_cv", "n_folds": n_m_folds, "note": "GroupKFold by mission_id over all usable events; report only"})
    _log("mission-held-out CV done")

    # ---- persist
    save_bundle(art / "model", primary)
    for name, b in candidates.items():
        if name != primary_name:
            save_bundle(art / "candidates" / name, b)
    ev_frames[primary_name].drop(columns=["quality_flags", "reason_codes", "prediction_set"], errors="ignore").to_parquet(art / "holdout_forecasts.parquet", index=False)
    labels[labels["event_id"].isin(split["events"]["holdout"])].to_parquet(art / "evaluation_labels.parquet", index=False)

    evaluation = {
        "generated_at": datetime.now(timezone.utc).isoformat(), "task": "final-tier forecast from CDMs at >= 2 days before TCA",
        "primary_model": primary_name, "model_version": primary.model_version, "quick_mode": quick,
        "split": {"seed": split["seed"], "holdout_fraction": split["holdout_fraction"], "split_hash": split["split_hash"], "counts": split["counts"],
                  "n_unusable_excluded": split["n_unusable_excluded"]},
        "label_audit": json.loads((art / "audit.json").read_text(encoding="utf-8")).get("label_audit") if (art / "audit.json").exists() else None,
        "decision_thresholds": primary.thresholds,
        "policy_tuning": diagnostics[primary_name]["oof_policy"],
        "training_rows_choice": {"chosen": chosen, "oof_critical_f2_real_forecast": rows_choice},
        "cv_search": search,
        "calibration": {"method": cfg["calibration"]["method"], "dev_oof": diagnostics[primary_name]["oof_calibration"]},
        "candidate_diagnostics": diagnostics,
        "models": results,
        "ablations": ablations,
        "mission_heldout_cv": mission_cv,
        "conformal_sets": conformal,
        "illustrative_cases": find_illustrative_cases(ev_frames[primary_name], k=3),
        "comparison_real_forecast": compare_models(results, subset="real_forecast").reset_index().to_dict(orient="records"),
        "comparison_all_usable": compare_models(results, subset="all_usable").reset_index().to_dict(orient="records"),
        "notes": [
            "Metrics are computed on the 20% event holdout of train_data only; test_data has no final outcomes and is used solely for demo predictions.",
            "Trivial events (final CDM itself at >= 2 days) are reported separately from real_forecast events; headline numbers use real_forecast.",
            "Probabilities are per-class isotonic-calibrated on dev out-of-fold predictions and renormalised; decision and triage thresholds are tuned on dev OOF only.",
            "risk of CDMs at >= 2 days is used as a past input when features.use_past_risk is true (the carry-forward baseline uses the same information).",
            "Zero final-Critical events auto-cleared on dev OOF is a tuning target, not a guarantee; the holdout count is reported.",
        ],
        "total_train_seconds": round(time.time() - t_start, 1),
    }
    _write_json(art / "evaluation.json", evaluation)
    _log(f"evaluation written ({evaluation['total_train_seconds']}s)")
    return evaluation


# --------------------------------------------------------------------------- #
# Stage 3: predict / export (test data: demo predictions only, never metrics)
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
    out = forecast_frame(bundle, df, with_explanations=True)
    secs = time.time() - t0
    events, snaps = out.events, out.snapshots
    assert events["event_id"].is_unique
    ps = events[["p_safe", "p_elevated", "p_critical"]].to_numpy(float)
    ok = np.isfinite(ps).all(axis=1)
    assert np.allclose(ps[ok].sum(axis=1), 1.0, atol=1e-6)
    cols = [c for c in EventForecastRecord.model_fields if c in events.columns]
    for i in list(range(min(100, len(events)))) + list(np.flatnonzero(events["prediction_status"].to_numpy() == "invalid")[:50]):
        EventForecastRecord.model_validate(_row_to_record(events.iloc[i][cols]))
    events.to_parquet(out_dir / "event_forecasts.parquet", index=False)
    snaps.to_parquet(out_dir / "snapshot_forecasts.parquet", index=False)
    if out.explanations is not None:
        out.explanations.to_parquet(out_dir / "explanations.parquet", index=False)
    sub = events.sort_values("event_id")[["event_id", "predicted_final_tier", "p_safe", "p_elevated", "p_critical", "triage_action"]]
    # LF line endings on every platform so the recorded sha256 is reproducible after a git checkout
    sub.to_csv(out_dir / "submission.csv", index=False, lineterminator="\n")
    bcfg = bundle.cfg  # provenance comes from the configuration the bundle was trained with
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(), "package_version": PKG_VERSION, "python": platform.python_version(),
        "lightgbm": lgb.__version__, "pandas": pd.__version__, "numpy": np.__version__, "schema_version": SCHEMA_VERSION_OF(events),
        "task": "final-tier forecast", "model_version": bundle.model_version, "model_kind": bundle.kind, "feature_version": bundle.meta.get("feature_version"),
        "policy_version": bcfg["policy_version"], "feature_set": bundle.feature_set, "use_past_risk": bundle.use_past_risk,
        "horizon_min_days": bcfg["horizon"]["min_days"], "thresholds": bundle.thresholds, "split_hash": bundle.meta.get("split_hash"),
        "bundle_path": str(bundle.path), "trained_at": bundle.meta.get("trained_at"),
        "units": bcfg["units"], "input": {"dataset_id": dataset_id, **meta, **out.info},
        "n_events": int(len(events)), "n_snapshots": int(len(snaps)),
        "predicted_class_distribution": events["predicted_final_tier"].value_counts(dropna=False).to_dict(),
        "predicted_class_share": events["predicted_final_tier"].value_counts(normalize=True, dropna=False).round(4).to_dict(),
        "current_tier_distribution": events["current_tier"].value_counts(dropna=False).to_dict(),
        "triage_counts": events["triage_action"].value_counts().to_dict(),
        "status_counts": events["prediction_status"].value_counts().to_dict(),
        "note": "Test predictions are demo output only; no metrics are computed on test_data (no final outcomes). The Kelvins test set was deliberately enriched with high-risk events, so the predicted class mix may differ from train.",
        "batch_inference_seconds": round(secs, 3), "timing": out.info,
        "artifacts": {f: file_sha256(out_dir / f) for f in ["event_forecasts.parquet", "snapshot_forecasts.parquet", "explanations.parquet", "submission.csv"] if (out_dir / f).exists()},
        "config_path": cfg.get("_config_path"),
    }
    _write_json(out_dir / "manifest.json", manifest)
    _log(f"exported {len(events)} event forecasts / {len(snaps)} snapshots to {out_dir} in {secs:.2f}s")
    return manifest


def SCHEMA_VERSION_OF(df: pd.DataFrame) -> str | None:
    return str(df["schema_version"].iloc[0]) if len(df) and "schema_version" in df.columns else None


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


def run_all(cfg: dict | None = None, *, quick: bool = False) -> None:
    cfg = cfg or load_config()
    stage_audit(cfg)
    stage_train(cfg, quick=quick)
    stage_predict(cfg)
