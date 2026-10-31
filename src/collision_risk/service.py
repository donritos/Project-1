"""Python service interface used by the pipeline, the dashboard, and tests.

    bundle = load_bundle("artifacts/model")
    out = predict_batch(bundle, rows_df, history=None)      # -> PredictionOutput
    get_event_history(dataset_id, event_id)                  # from exported artifacts
    get_explanation(explanation_id)
    get_evaluation_report()

`predict_frame` is the single prediction path: features -> validation -> point model ->
postprocessing -> conformal interval -> classifier -> official tiers -> triage policy.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import features as FE
from .calibration import CQRCalibrator, SigmoidCalibrator, SplitConformalCalibrator, fix_quantile_crossing
from .config import PROJECT_ROOT, load_config
from .contracts import PREDICTION_COLUMNS, PredictionStatus, QualityFlag, ReasonCode, SCHEMA_VERSION
from .data import enforce_allowlist, prepare_uploaded, validate_rows
from .explain import compute_contributions, explanations_to_frame
from .policy import assign_triage
from .train import LinearBaseline, MedianBaseline, postprocess


@dataclass
class Bundle:
    path: Path
    meta: dict[str, Any]
    cfg: dict[str, Any]
    booster: lgb.Booster | None
    classifier: lgb.Booster | None
    conformal: SplitConformalCalibrator | None
    sigmoid: SigmoidCalibrator | None
    baseline: MedianBaseline | LinearBaseline | None = None
    quantile_heads: dict[float, lgb.Booster] = field(default_factory=dict)
    cqr: CQRCalibrator | None = None
    interval_method: str = "residual"   # "residual" | "cqr"

    @property
    def active_calibrator(self):
        return self.cqr if self.interval_method == "cqr" else self.conformal

    @property
    def feature_set(self) -> str:
        return self.meta["feature_set"]

    @property
    def model_version(self) -> str:
        return self.meta["model_version"]

    @property
    def kind(self) -> str:
        return self.meta.get("kind", "lgbm")


def save_bundle(path: str | Path, *, meta: dict[str, Any], cfg: dict[str, Any], booster: lgb.Booster | None,
                classifier: lgb.Booster | None, conformal: SplitConformalCalibrator | None, sigmoid: SigmoidCalibrator | None,
                baseline: MedianBaseline | LinearBaseline | None = None, quantile_heads: dict[float, lgb.Booster] | None = None,
                cqr: CQRCalibrator | None = None, interval_method: str = "residual") -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if booster is not None:
        booster.save_model(str(path / "model.txt"))
    if classifier is not None:
        classifier.save_model(str(path / "classifier.txt"))
    meta = dict(meta)
    meta["quantile_alphas"] = []
    for a, h in (quantile_heads or {}).items():
        h.save_model(str(path / f"quantile_{a:.2f}.txt"))
        meta["quantile_alphas"].append(float(a))
    meta["cqr"] = cqr.to_dict() if cqr is not None else None
    meta["interval_method"] = interval_method
    meta["conformal"] = conformal.to_dict() if conformal is not None else None
    meta["sigmoid"] = sigmoid.to_dict() if sigmoid is not None else None
    meta["baseline"] = baseline.to_dict() if baseline is not None else None
    meta["has_booster"] = booster is not None
    meta["has_classifier"] = classifier is not None
    meta["schema_version"] = SCHEMA_VERSION
    with open(path / "bundle.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    with open(path / "config_snapshot.json", "w", encoding="utf-8") as fh:
        json.dump({k: v for k, v in cfg.items() if not k.startswith("_")}, fh, indent=2)
    return path


def load_bundle(path: str | Path) -> Bundle:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path / "bundle.json", "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    with open(path / "config_snapshot.json", "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    booster = lgb.Booster(model_file=str(path / "model.txt")) if meta.get("has_booster") else None
    classifier = lgb.Booster(model_file=str(path / "classifier.txt")) if meta.get("has_classifier") else None
    conformal = SplitConformalCalibrator.from_dict(meta["conformal"]) if meta.get("conformal") else None
    sigmoid = SigmoidCalibrator.from_dict(meta["sigmoid"]) if meta.get("sigmoid") else None
    baseline = None
    b = meta.get("baseline")
    if b:
        if b["type"] == "median":
            baseline = MedianBaseline(median=b["median"])
        else:
            baseline = LinearBaseline(columns=b["columns"], coef_=b["coef"], intercept_=b["intercept"], medians_=b["medians"])
    heads = {float(a): lgb.Booster(model_file=str(path / f"quantile_{a:.2f}.txt")) for a in meta.get("quantile_alphas", [])}
    cqr = CQRCalibrator.from_dict(meta["cqr"]) if meta.get("cqr") else None
    return Bundle(path=path, meta=meta, cfg=cfg, booster=booster, classifier=classifier, conformal=conformal, sigmoid=sigmoid, baseline=baseline,
                  quantile_heads=heads, cqr=cqr, interval_method=meta.get("interval_method", "residual"))


# --------------------------------------------------------------------------- #
@dataclass
class PredictionOutput:
    predictions: pd.DataFrame
    explanations: pd.DataFrame | None
    timing_seconds: dict[str, float] = field(default_factory=dict)


def _point_predict(bundle: Bundle, X: pd.DataFrame, F: pd.DataFrame) -> np.ndarray:
    if bundle.kind == "lgbm":
        return bundle.booster.predict(X, num_iteration=bundle.meta.get("num_iteration"))
    if bundle.kind == "median":
        return bundle.baseline.predict(len(X))
    if bundle.kind == "linear":
        return bundle.baseline.predict(F)
    raise ValueError(f"unknown bundle kind {bundle.kind}")


def predict_frame(bundle: Bundle, df: pd.DataFrame, *, with_explanations: bool = True, explanation_top_k: int = 5) -> PredictionOutput:
    """df must already be allowlisted (data.load_raw / prepare_uploaded) and may contain `risk`
    (ignored for prediction). Returns one record per input row in input order."""
    import time
    cfg = bundle.cfg
    t: dict[str, float] = {}
    t0 = time.time()
    n = len(df)
    df, _dropped = enforce_allowlist(df.reset_index(drop=True), cfg, allow_target=True)

    status, flags = validate_rows(df, cfg, known_object_types=bundle.meta.get("object_type_categories"))
    F = FE.build_features(df, cfg, feature_set=bundle.feature_set)
    X = FE.feature_matrix(F, bundle.feature_set)
    # out-of-training-range flag on raw telemetry
    oor = FE.out_of_range_mask(X, bundle.meta.get("feature_ranges", {}))
    for i in np.flatnonzero(oor):
        if QualityFlag.OUT_OF_TRAINING_RANGE.value not in flags[i]:
            flags[i].append(QualityFlag.OUT_OF_TRAINING_RANGE.value)
    status = np.where((status == PredictionStatus.VALID.value) & oor, PredictionStatus.WARNING.value, status)
    t["features"] = time.time() - t0

    t1 = time.time()
    valid = status != PredictionStatus.INVALID.value
    raw_pred = np.full(n, np.nan)
    if valid.any():
        raw_pred[valid] = _point_predict(bundle, X[valid], F[valid])
    pred = postprocess(raw_pred, cfg)
    t["point_model"] = time.time() - t1

    lower = np.full(n, np.nan); upper = np.full(n, np.nan)
    nominal = None
    if bundle.interval_method == "cqr" and bundle.cqr is not None and bundle.quantile_heads:
        alphas = sorted(bundle.quantile_heads)
        q_lo = np.full(n, np.nan); q_hi = np.full(n, np.nan)
        if valid.any():
            q_lo[valid] = bundle.quantile_heads[alphas[0]].predict(X[valid])
            q_hi[valid] = bundle.quantile_heads[alphas[-1]].predict(X[valid])
        q_lo, q_hi = fix_quantile_crossing(pred, postprocess(q_lo, cfg), postprocess(q_hi, cfg))
        lower, upper = bundle.cqr.predict_interval(q_lo, q_hi)
        lower = postprocess(lower, cfg); upper = postprocess(upper, cfg)
        nominal = bundle.cqr.coverage
    elif bundle.conformal is not None:
        lower, upper = bundle.conformal.predict_interval(pred)
        lower = postprocess(lower, cfg); upper = postprocess(upper, cfg)
        nominal = bundle.conformal.coverage

    p_crit = np.full(n, np.nan)
    if bundle.classifier is not None and valid.any():
        s = bundle.classifier.predict(X[valid], num_iteration=bundle.meta.get("classifier_num_iteration"))
        p_crit[valid] = bundle.sigmoid.predict_proba(s) if bundle.sigmoid is not None else s

    out = pd.DataFrame({
        "schema_version": SCHEMA_VERSION,
        "dataset_id": df["dataset_id"].astype(str).to_numpy(),
        "row_id": df["row_id"].astype(str).to_numpy(),
        "event_id": df["event_id"].astype("int64").to_numpy(),
        "source_row_index": df["source_row_index"].astype("int64").to_numpy(),
        "prediction_status": status,
        "time_to_tca_days": pd.to_numeric(df["time_to_tca"], errors="coerce").to_numpy(float),
        "miss_distance_m": pd.to_numeric(df["miss_distance"], errors="coerce").to_numpy(float),
        "relative_position_r_m": pd.to_numeric(df["relative_position_r"], errors="coerce").to_numpy(float),
        "relative_position_t_m": pd.to_numeric(df["relative_position_t"], errors="coerce").to_numpy(float),
        "relative_position_n_m": pd.to_numeric(df["relative_position_n"], errors="coerce").to_numpy(float),
        "object_type": df["c_object_type"].astype("object").where(df["c_object_type"].notna(), None).to_numpy(),
        "risk_log10_pred": pred,
        "pc_pred": np.where(np.isfinite(pred), np.power(10.0, pred), np.nan),
        "risk_log10_lower": lower,
        "risk_log10_upper": upper,
        "interval_nominal_coverage": nominal,
        "p_pc_critical": p_crit,
        "quality_flags": flags,
    })
    # triage policy (official tiers + review policy). Classifier threshold may be tuned in the bundle.
    pol_cfg = dict(cfg)
    if bundle.meta.get("classifier_threshold") is not None:
        pol_cfg = {**cfg, "triage": {**cfg["triage"], "classifier_disagreement_prob": bundle.meta["classifier_threshold"]}}
    t2 = time.time()
    out = assign_triage(out, pol_cfg)
    t["policy"] = time.time() - t2
    # standalone rows: record NO_HISTORY reason code for transparency
    if bundle.feature_set == "history" and "has_history" in F.columns:
        nh = F["has_history"].to_numpy() == 0
        out["reason_codes"] = [rc + [ReasonCode.NO_HISTORY.value] if (nh[i] and ReasonCode.NO_HISTORY.value not in rc) else rc
                               for i, rc in enumerate(out["reason_codes"])]

    out["explanation_id"] = None
    out["model_version"] = bundle.model_version
    out["feature_version"] = bundle.meta["feature_version"]
    out["policy_version"] = cfg["policy_version"]

    expl_df = None
    if with_explanations and bundle.kind == "lgbm" and valid.any():
        t3 = time.time()
        idx = np.flatnonzero(valid)
        recs = compute_contributions(
            bundle.booster, X.iloc[idx], row_ids=list(out["row_id"].iloc[idx]), dataset_id=str(out["dataset_id"].iloc[0]) if n else "",
            component="lgbm_point_regressor", model_version=bundle.model_version, final_outputs=pred[idx],
            postprocessing_note=f"clipped to [{cfg['model']['target_floor']}, 0] log10(Pc); official tier and triage rules applied separately",
            top_k=explanation_top_k,
        )
        expl_df = explanations_to_frame(recs)
        ids = np.array([None] * n, dtype=object)
        ids[idx] = [r.explanation_id for r in recs]
        out["explanation_id"] = ids
        t["explanations"] = time.time() - t3

    out = out[PREDICTION_COLUMNS]
    t["total"] = time.time() - t0
    return PredictionOutput(predictions=out, explanations=expl_df, timing_seconds={k: round(v, 4) for k, v in t.items()})


def predict_batch(bundle: Bundle, rows: pd.DataFrame, history: pd.DataFrame | None = None, *, dataset_id: str = "upload",
                  with_explanations: bool = True) -> PredictionOutput:
    """Predict for raw CDM rows (any columns; forbidden ones are dropped). `history` holds earlier
    CDMs for the same event_ids; they are used only to build causal history features and are not
    returned. Without history, rows use the documented no-history representation."""
    rows_p, _ = prepare_uploaded(rows, dataset_id, bundle.cfg)
    if history is not None and len(history):
        hist_p, _ = prepare_uploaded(history, dataset_id + ":history", bundle.cfg)
        hist_p["dataset_id"] = dataset_id
        both = pd.concat([hist_p, rows_p], ignore_index=True)
        out = predict_frame(bundle, both, with_explanations=with_explanations)
        keep = out.predictions["row_id"].isin(set(rows_p["row_id"]))
        preds = out.predictions[keep].reset_index(drop=True)
        expl = None
        if out.explanations is not None:
            expl = out.explanations[out.explanations["row_id"].isin(set(rows_p["row_id"]))].reset_index(drop=True)
        return PredictionOutput(predictions=preds, explanations=expl, timing_seconds=out.timing_seconds)
    return predict_frame(bundle, rows_p, with_explanations=with_explanations)


# --------------------------------------------------------------------------- #
# Artifact readers for the dashboard
# --------------------------------------------------------------------------- #
def _artifacts_dir(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    p = Path(cfg["paths"]["artifacts_dir"])
    return p if p.is_absolute() else PROJECT_ROOT / p


def get_event_history(dataset_id: str, event_id: int, artifacts_dir: str | Path | None = None) -> pd.DataFrame:
    d = Path(artifacts_dir) if artifacts_dir else _artifacts_dir()
    preds = pd.read_parquet(d / "predictions.parquet")
    h = preds[(preds["dataset_id"] == dataset_id) & (preds["event_id"] == int(event_id))]
    return h.sort_values("time_to_tca_days", ascending=False).reset_index(drop=True)


def get_explanation(explanation_id: str, artifacts_dir: str | Path | None = None) -> dict | None:
    d = Path(artifacts_dir) if artifacts_dir else _artifacts_dir()
    ex = pd.read_parquet(d / "explanations.parquet")
    m = ex[ex["explanation_id"] == explanation_id]
    if m.empty:
        return None
    rec = m.iloc[0].to_dict()
    for k in ("top_positive", "top_negative"):
        if isinstance(rec.get(k), str):
            rec[k] = json.loads(rec[k])
    return rec


def get_evaluation_report(artifacts_dir: str | Path | None = None) -> dict:
    d = Path(artifacts_dir) if artifacts_dir else _artifacts_dir()
    with open(d / "evaluation.json", "r", encoding="utf-8") as fh:
        return json.load(fh)
