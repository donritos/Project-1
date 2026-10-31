"""Service interface for the final-tier forecasting task.

    bundle = load_bundle("artifacts/model")
    out = predict_batch(bundle, raw_cdm_rows)          # -> ForecastOutput(events, snapshots, explanations)
    get_event_forecast(dataset_id, event_id)           # from exported artifacts
    get_event_history(dataset_id, event_id)            # snapshot forecasts of one event
    get_explanation(explanation_id)
    get_evaluation_report()

`forecast_frame` is the single prediction path used by training-time OOF assembly, holdout
evaluation, the test export and uploads: allowlist -> horizon filter -> validation -> event
snapshots -> probabilities -> calibration -> decision rule -> prediction sets -> triage policy.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import features as FE
from .calibration import LACPredictionSets, MulticlassIsotonicCalibrator, apply_decision_rule
from .config import PROJECT_ROOT, load_config
from .contracts import EVENT_FORECAST_COLUMNS, PredictionStatus, QualityFlag, SCHEMA_VERSION
from .data import apply_horizon, enforce_allowlist, prepare_uploaded, validate_rows
from .explain import explain_forecasts
from .policy import assign_triage
from .train import TIERS, LogisticBaseline, carry_forward_probs, majority_probs, ordinal_probabilities

SEV_TO_TIER = {0.0: "Safe", 1.0: "Elevated", 2.0: "Critical"}
LEARNED_KINDS = ("multiclass", "ordinal", "logistic")


@dataclass
class ForecastBundle:
    path: Path | None
    meta: dict[str, Any]
    cfg: dict[str, Any]
    kind: str                                  # multiclass | ordinal | logistic | carry_forward_risk | carry_forward_mre | majority
    booster: lgb.Booster | None = None         # multiclass
    booster_ge_elev: lgb.Booster | None = None # ordinal head P(final >= Elevated)
    booster_crit: lgb.Booster | None = None    # ordinal head P(final = Critical)
    logistic: LogisticBaseline | None = None
    calibrator: MulticlassIsotonicCalibrator | None = None
    sets: LACPredictionSets | None = None
    thresholds: dict[str, float] = field(default_factory=lambda: {"t_crit": 0.5, "t_elev": 0.5, "t_watch": 1.01, "t_clear": 1.01})

    @property
    def feature_set(self) -> str:
        return self.meta["feature_set"]

    @property
    def model_version(self) -> str:
        return self.meta["model_version"]

    @property
    def use_past_risk(self) -> bool:
        return bool(self.cfg["features"].get("use_past_risk", True))


def save_bundle(path: str | Path, bundle: ForecastBundle) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    meta = dict(bundle.meta)
    meta.update({"kind": bundle.kind, "thresholds": bundle.thresholds, "schema_version": SCHEMA_VERSION,
                 "calibrator": bundle.calibrator.to_dict() if bundle.calibrator else None,
                 "prediction_sets": bundle.sets.to_dict() if bundle.sets else None,
                 "logistic": bundle.logistic.to_dict() if bundle.logistic else None})
    if bundle.booster is not None:
        bundle.booster.save_model(str(path / "model.txt"))
    if bundle.booster_ge_elev is not None:
        bundle.booster_ge_elev.save_model(str(path / "model_ge_elevated.txt"))
    if bundle.booster_crit is not None:
        bundle.booster_crit.save_model(str(path / "model_critical.txt"))
    with open(path / "bundle.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    with open(path / "config_snapshot.json", "w", encoding="utf-8") as fh:
        json.dump({k: v for k, v in bundle.cfg.items() if not k.startswith("_")}, fh, indent=2)
    return path


def load_bundle(path: str | Path) -> ForecastBundle:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    meta = json.loads((path / "bundle.json").read_text(encoding="utf-8"))
    cfg = json.loads((path / "config_snapshot.json").read_text(encoding="utf-8"))
    b = ForecastBundle(path=path, meta=meta, cfg=cfg, kind=meta["kind"], thresholds=dict(meta.get("thresholds") or {}))
    if (path / "model.txt").exists():
        b.booster = lgb.Booster(model_file=str(path / "model.txt"))
    if (path / "model_ge_elevated.txt").exists():
        b.booster_ge_elev = lgb.Booster(model_file=str(path / "model_ge_elevated.txt"))
    if (path / "model_critical.txt").exists():
        b.booster_crit = lgb.Booster(model_file=str(path / "model_critical.txt"))
    if meta.get("logistic"):
        b.logistic = LogisticBaseline.from_dict(meta["logistic"])
    if meta.get("calibrator"):
        b.calibrator = MulticlassIsotonicCalibrator.from_dict(meta["calibrator"])
    if meta.get("prediction_sets"):
        b.sets = LACPredictionSets.from_dict(meta["prediction_sets"])
    return b


# --------------------------------------------------------------------------- #
@dataclass
class ForecastOutput:
    events: pd.DataFrame                 # one row per event (latest snapshot)
    snapshots: pd.DataFrame              # one row per snapshot CDM
    explanations: pd.DataFrame | None
    info: dict[str, Any] = field(default_factory=dict)


def current_tier_columns(snap: pd.DataFrame, use_past_risk: bool) -> tuple[np.ndarray, str]:
    """Current tier of each snapshot from (risk if allowed and present else max_risk_estimate) and miss distance."""
    if use_past_risk and "tier_now_from_risk" in snap.columns and snap["tier_now_from_risk"].notna().any():
        return snap["tier_now_from_risk"].map(SEV_TO_TIER).to_numpy(dtype=object), "risk"
    return snap["tier_now_from_mre"].map(SEV_TO_TIER).to_numpy(dtype=object), "mre"


def raw_probabilities(bundle: ForecastBundle, X: pd.DataFrame, snap: pd.DataFrame) -> np.ndarray:
    """Uncalibrated (n,3) class probabilities in TIERS order."""
    n = len(X)
    if bundle.kind == "multiclass":
        return bundle.booster.predict(X, num_iteration=bundle.meta.get("num_iteration"))
    if bundle.kind == "ordinal":
        pe = bundle.booster_ge_elev.predict(X, num_iteration=bundle.meta.get("num_iteration_ge_elev"))
        pc = bundle.booster_crit.predict(X, num_iteration=bundle.meta.get("num_iteration_crit"))
        return ordinal_probabilities(pe, pc)
    if bundle.kind == "logistic":
        return bundle.logistic.predict_proba(snap)
    if bundle.kind == "carry_forward_risk":
        tier, _ = current_tier_columns(snap, True)
        return carry_forward_probs(tier)
    if bundle.kind == "carry_forward_mre":
        tier, _ = current_tier_columns(snap, False)
        return carry_forward_probs(tier)
    if bundle.kind == "majority":
        return majority_probs(n)
    raise ValueError(f"unknown bundle kind {bundle.kind}")


def assemble_forecast_frame(snap: pd.DataFrame, status: np.ndarray, flags: list[list[str]], P: np.ndarray, bundle: ForecastBundle,
                            *, sets: list[list[str] | None] | None = None) -> pd.DataFrame:
    """Build the schema frame (docs/forecast_schema.md) from snapshots + probabilities, then apply triage."""
    cfg = bundle.cfg
    n = len(snap)
    pred = apply_decision_rule(P, bundle.thresholds.get("t_crit", 0.5), bundle.thresholds.get("t_elev", 0.5))
    cur, src = current_tier_columns(snap, bundle.use_past_risk)
    has_risk = "risk" in snap.columns
    latest_risk = pd.to_numeric(snap["risk"], errors="coerce").to_numpy(float) if has_risk else np.full(n, np.nan)
    ev_key = snap["dataset_id"].astype(str) + ":" + snap["event_id"].astype(str)
    latest_mask = snap["is_latest_snapshot"].to_numpy(bool)
    cf = pd.Series(cur, index=snap.index).where(latest_mask).groupby(ev_key.to_numpy()).transform("first")
    frame = pd.DataFrame({
        "schema_version": SCHEMA_VERSION,
        "dataset_id": snap["dataset_id"].astype(str).to_numpy(),
        "event_id": snap["event_id"].astype("int64").to_numpy(),
        "row_id": snap["row_id"].astype(str).to_numpy(),
        "time_to_tca_days": pd.to_numeric(snap["time_to_tca"], errors="coerce").to_numpy(float),
        "miss_distance_m": pd.to_numeric(snap["miss_distance"], errors="coerce").to_numpy(float),
        "is_latest_snapshot": latest_mask,
        "n_cdms_2d": snap["n_cdms_2d"].fillna(1).astype(int).to_numpy(),
        "has_history": snap["has_history"].fillna(0).astype(int).to_numpy(),
        "current_tier": cur,
        "current_tier_source": src,
        "latest_risk": latest_risk,
        "prediction_status": status,
        "quality_flags": flags,
        "p_safe": P[:, 0], "p_elevated": P[:, 1], "p_critical": P[:, 2],
        "predicted_final_tier": pred,
        "prediction_set": sets if sets is not None else [None] * n,
        "slope_log_miss": snap["slope_log_miss"].to_numpy(float),
        "slope_risk_or_mre": (snap["slope_risk"] if (bundle.use_past_risk and has_risk) else snap["slope_mre"]).to_numpy(float),
        "slope_log_sigma": snap["slope_log_sigma"].to_numpy(float),
        "volatility_mre": snap["volatility_mre"].to_numpy(float),
        "volatility_risk": snap["volatility_risk"].to_numpy(float) if "volatility_risk" in snap.columns else np.full(n, np.nan),
        "carry_forward_tier": cf.to_numpy(dtype=object),
        "object_type": snap["c_object_type"].astype("object").where(snap["c_object_type"].notna(), None).to_numpy(),
    })
    frame = assign_triage(frame, cfg, {"t_watch": bundle.thresholds.get("t_watch", 1.01), "t_clear": bundle.thresholds.get("t_clear", 1.01)})
    frame["explanation_id"] = None
    frame["model_version"] = bundle.model_version
    frame["feature_version"] = bundle.meta.get("feature_version", cfg["feature_version"])
    frame["policy_version"] = cfg["policy_version"]
    return frame


def forecast_frame(bundle: ForecastBundle, df: pd.DataFrame, *, with_explanations: bool = True, explanation_top_k: int = 5,
                   explain_all_snapshots: bool = True) -> ForecastOutput:
    """df: allowlisted CDM rows (may include `risk`). Applies the horizon filter, builds snapshots and
    returns per-snapshot and per-event forecasts in snapshot (input) order."""
    cfg = bundle.cfg
    t0 = time.time()
    info: dict[str, Any] = {}
    df, _ = enforce_allowlist(df.reset_index(drop=True), cfg, allow_target=True)
    h = apply_horizon(df, cfg["horizon"]["min_days"])
    info["n_input_rows"] = int(len(df)); info["n_rows_below_horizon_dropped"] = int(len(df) - len(h))
    info["n_events_input"] = int(df["event_id"].nunique()); info["n_events_with_snapshot"] = int(h["event_id"].nunique())
    if not bundle.use_past_risk and "risk" in h.columns:
        h = h.drop(columns=["risk"])
    if len(h) == 0:
        empty = pd.DataFrame(columns=EVENT_FORECAST_COLUMNS)
        return ForecastOutput(events=empty, snapshots=empty, explanations=None, info=info)

    status, flags = validate_rows(h, cfg, known_object_types=bundle.meta.get("object_type_categories"))
    snap = FE.build_event_snapshots(h, cfg)
    X = FE.feature_matrix(snap, bundle.feature_set)
    oor = FE.out_of_range_mask(X, bundle.meta.get("feature_ranges", {}))
    for i in np.flatnonzero(oor):
        if QualityFlag.OUT_OF_TRAINING_RANGE.value not in flags[i]:
            flags[i].append(QualityFlag.OUT_OF_TRAINING_RANGE.value)
    status = np.where((status == PredictionStatus.VALID.value) & oor, PredictionStatus.WARNING.value, status)
    valid = status != PredictionStatus.INVALID.value
    info["t_features"] = round(time.time() - t0, 3)

    t1 = time.time()
    P = np.full((len(snap), 3), np.nan)
    if valid.any():
        P[valid] = raw_probabilities(bundle, X[valid], snap[valid])
        if bundle.calibrator is not None:
            P[valid] = bundle.calibrator.predict_proba(P[valid])
    sets = bundle.sets.predict_sets(P, TIERS) if bundle.sets is not None else None
    info["t_model"] = round(time.time() - t1, 3)

    frame = assemble_forecast_frame(snap, status, flags, P, bundle, sets=sets)

    expl = None
    if with_explanations and bundle.kind in ("multiclass", "ordinal") and valid.any():
        t2 = time.time()
        idx = np.flatnonzero(valid & (frame["is_latest_snapshot"].to_numpy() | explain_all_snapshots))
        booster = bundle.booster if bundle.kind == "multiclass" else bundle.booster_crit
        expl = explain_forecasts(booster, X.iloc[idx], class_index=2 if bundle.kind == "multiclass" else None,
                                 row_ids=list(frame["row_id"].iloc[idx]), dataset_id=str(frame["dataset_id"].iloc[0]),
                                 model_version=bundle.model_version, p_critical=P[idx, 2], predicted_tier=frame["predicted_final_tier"].iloc[idx].tolist(),
                                 top_k=explanation_top_k,
                                 component="lgbm_multiclass_critical_logit" if bundle.kind == "multiclass" else "lgbm_binary_critical_logit")
        ids = np.array([None] * len(frame), dtype=object)
        ids[idx] = expl["explanation_id"].to_numpy()
        frame["explanation_id"] = ids
        info["t_explanations"] = round(time.time() - t2, 3)

    events = frame[frame["is_latest_snapshot"]].reset_index(drop=True)
    info["t_total"] = round(time.time() - t0, 3)
    return ForecastOutput(events=events, snapshots=frame, explanations=expl, info=info)


def predict_batch(bundle: ForecastBundle, rows: pd.DataFrame, *, dataset_id: str = "upload", with_explanations: bool = True) -> ForecastOutput:
    """Raw CDM rows for one or more events (any columns; forbidden ones dropped). CDMs under the horizon are
    dropped before features are built. Returns event forecasts (latest snapshot per event) and snapshot forecasts."""
    rows_p, _ = prepare_uploaded(rows, dataset_id, bundle.cfg)
    return forecast_frame(bundle, rows_p, with_explanations=with_explanations)


# --------------------------------------------------------------------------- #
# Artifact readers for the dashboard
# --------------------------------------------------------------------------- #
def _artifacts_dir(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    p = Path(cfg["paths"]["artifacts_dir"])
    return p if p.is_absolute() else PROJECT_ROOT / p


def get_event_forecast(dataset_id: str, event_id: int, artifacts_dir: str | Path | None = None) -> dict | None:
    d = Path(artifacts_dir) if artifacts_dir else _artifacts_dir()
    ev = pd.read_parquet(d / "event_forecasts.parquet")
    m = ev[(ev["dataset_id"] == dataset_id) & (ev["event_id"] == int(event_id))]
    return None if m.empty else m.iloc[0].to_dict()


def get_event_history(dataset_id: str, event_id: int, artifacts_dir: str | Path | None = None) -> pd.DataFrame:
    d = Path(artifacts_dir) if artifacts_dir else _artifacts_dir()
    s = pd.read_parquet(d / "snapshot_forecasts.parquet")
    h = s[(s["dataset_id"] == dataset_id) & (s["event_id"] == int(event_id))]
    return h.sort_values("time_to_tca_days", ascending=False).reset_index(drop=True)


def get_explanation(explanation_id: str, artifacts_dir: str | Path | None = None) -> dict | None:
    d = Path(artifacts_dir) if artifacts_dir else _artifacts_dir()
    ex = pd.read_parquet(d / "explanations.parquet")
    m = ex[ex["explanation_id"] == explanation_id]
    if m.empty:
        return None
    rec = m.iloc[0].to_dict()
    for k in ("top_positive", "top_negative", "group_contributions"):
        if isinstance(rec.get(k), str):
            rec[k] = json.loads(rec[k])
    return rec


def get_evaluation_report(artifacts_dir: str | Path | None = None) -> dict:
    d = Path(artifacts_dir) if artifacts_dir else _artifacts_dir()
    return json.loads((d / "evaluation.json").read_text(encoding="utf-8"))
