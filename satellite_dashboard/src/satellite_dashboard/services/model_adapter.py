"""The only module that knows the model's own column names and loading behaviour.

It turns model-native frames (from artifacts or from a live upload) into the standard records in
contracts.py. Swapping the model means replacing this file and the artifact bundle; views do not change.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..config import ARTIFACTS_DIR, DEMO_DIR, ELIGIBILITY_EPS, MIN_TIME_TO_TCA_DAYS
from ..contracts import (NEEDS_REVIEW, PRIORITY_RANK, Capabilities, DatasetInfo, Manifest, PredictionRecord,
                         PredictionStatus, ReviewPriority)
from .artifact_loader import ArtifactError, NativeDataset, load_manifest, load_native_dataset

# model triage action -> analyst-facing review priority
ACTION_MAP = {
    "Maneuver planning review": ReviewPriority.PRIORITY,
    "Escalation watch": ReviewPriority.WATCH,
    "Human review": ReviewPriority.ANALYST,
    "Human review - insufficient data": ReviewPriority.INSUFFICIENT,
    "Monitor": ReviewPriority.MONITOR,
    "Low-priority monitor": ReviewPriority.LOW_PRIORITY,
    "Auto-clear candidate": ReviewPriority.LOWER,
}
STATUS_MAP = {"valid": PredictionStatus.SUCCESS, "warning": PredictionStatus.SUCCESS, "invalid": PredictionStatus.INVALID}

INPUT_RENAME = {
    "time_to_tca": "time_to_tca_days", "miss_distance": "miss_distance_m", "relative_speed": "relative_speed_mps",
    "relative_position_r": "rel_pos_r_m", "relative_position_t": "rel_pos_t_m", "relative_position_n": "rel_pos_n_m",
    "c_sigma_r": "chaser_sigma_r_m", "t_sigma_r": "target_sigma_r_m", "max_risk_estimate": "max_risk_estimate_log10",
    "risk": "risk_log10", "c_object_type": "object_type", "F10": "f10", "AP": "ap",
}
INPUT_STD_COLUMNS = ["dataset_id", "event_id", "row_id"] + list(INPUT_RENAME.values())


@dataclass
class DatasetView:
    key: str
    label: str
    kind: str                       # "real", "synthetic" or "upload"
    manifest: Manifest
    info: DatasetInfo
    events: pd.DataFrame            # one standard record per event
    snapshots: pd.DataFrame         # one row per eligible CDM: the forecast made from history up to that CDM
    inputs: pd.DataFrame            # eligible CDM measurements (standard names, metres)
    explanations: Optional[pd.DataFrame]
    outcomes: Optional[pd.DataFrame]  # evaluation only; never joined into events/snapshots

    @property
    def capabilities(self) -> Capabilities:
        return self.manifest.capabilities

    @property
    def model_version(self) -> str:
        return self.manifest.model_version

    def event(self, event_id: int) -> pd.Series:
        m = self.events[self.events["event_id"] == int(event_id)]
        if m.empty:
            raise KeyError(f"event {event_id} not in {self.key}")
        return m.iloc[0]

    def event_snapshots(self, event_id: int) -> pd.DataFrame:
        s = self.snapshots[self.snapshots["event_id"] == int(event_id)]
        return s.sort_values("time_to_tca_days", ascending=False).reset_index(drop=True)

    def event_inputs(self, event_id: int) -> pd.DataFrame:
        s = self.inputs[self.inputs["event_id"] == int(event_id)]
        return s.sort_values("time_to_tca_days", ascending=False).reset_index(drop=True)

    def outcome(self, event_id: int) -> Optional[pd.Series]:
        if self.outcomes is None:
            return None
        m = self.outcomes[self.outcomes["event_id"] == int(event_id)]
        return None if m.empty else m.iloc[0]

    def explanation(self, explanation_id: Optional[str]) -> Optional[pd.Series]:
        if self.explanations is None or not explanation_id:
            return None
        m = self.explanations[self.explanations["explanation_id"] == explanation_id]
        return None if m.empty else m.iloc[0]


# --------------------------------------------------------------------------- #
def prediction_id(model_version: str, dataset_id: str, event_id: int, latest_row_id: Optional[str]) -> str:
    raw = f"{model_version}|{dataset_id}|{int(event_id)}|{latest_row_id or '-'}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _f(x) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def _list(x) -> list[str]:
    if x is None:
        return []
    if isinstance(x, float) and math.isnan(x):
        return []
    return [str(v) for v in list(x)]


def _tier(x) -> Optional[str]:
    return x if isinstance(x, str) and x in ("Safe", "Elevated", "Critical") else None


def translate(native: NativeDataset, model_version: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Model-native frames -> (standard events, standard snapshots, standard inputs)."""
    ev, sn = native.events, native.snapshots
    src_ids = sn.sort_values("time_to_tca_days", ascending=False).groupby(["dataset_id", "event_id"])["row_id"].apply(list)
    records: list[PredictionRecord] = []
    for r in ev.itertuples(index=False):
        status = STATUS_MAP.get(str(r.prediction_status), PredictionStatus.FAILED)
        ok = status == PredictionStatus.SUCCESS
        action = ACTION_MAP.get(str(r.triage_action)) if isinstance(r.triage_action, str) else None
        if not ok:
            action = ReviewPriority.INSUFFICIENT
        excl = getattr(r, "n_rows_excluded_by_horizon", None)
        records.append(PredictionRecord(
            dataset_id=str(r.dataset_id), event_id=int(r.event_id),
            prediction_id=prediction_id(model_version, r.dataset_id, r.event_id, r.row_id),
            source_row_ids=src_ids.get((r.dataset_id, r.event_id), []),
            latest_input_row_id=str(r.row_id), latest_input_time_to_tca_days=_f(r.time_to_tca_days),
            predicted_final_tier=_tier(r.predicted_final_tier) if ok else None,
            prediction_status=status, quality_flags=_list(r.quality_flags), model_version=str(r.model_version),
            n_eligible_cdms=int(r.n_cdms_2d) if _f(r.n_cdms_2d) is not None else len(src_ids.get((r.dataset_id, r.event_id), [])),
            n_excluded_cdms=None if excl is None or _f(excl) is None else int(excl),
            p_safe=_f(r.p_safe) if ok else None, p_elevated=_f(r.p_elevated) if ok else None,
            p_critical=_f(r.p_critical) if ok else None,
            latest_observed_tier=_tier(r.current_tier), baseline_tier=_tier(r.carry_forward_tier),
            latest_miss_distance_m=_f(r.miss_distance_m), object_type=getattr(r, "object_type", None),
            review_priority=action, model_action=str(r.triage_action) if isinstance(r.triage_action, str) else None,
            reason_codes=_list(r.reason_codes),
            explanation_id=r.explanation_id if isinstance(r.explanation_id, str) else None,
        ))
    if native.ineligible is not None:
        for r in native.ineligible.itertuples(index=False):
            records.append(PredictionRecord(
                dataset_id=str(r.dataset_id), event_id=int(r.event_id),
                prediction_id=prediction_id(model_version, r.dataset_id, r.event_id, None),
                prediction_status=PredictionStatus.INSUFFICIENT, model_version=model_version,
                n_eligible_cdms=0, n_excluded_cdms=int(r.n_excluded_cdms),
                quality_flags=["NO_ELIGIBLE_CDM"], review_priority=ReviewPriority.INSUFFICIENT))
    events = none_for_missing(pd.DataFrame([r.model_dump() for r in records]))
    events["priority_rank"] = events["review_priority"].map({k.value: v for k, v in PRIORITY_RANK.items()}).fillna(9).astype(int)
    events["needs_review"] = events["review_priority"].isin([p.value for p in NEEDS_REVIEW])
    events = order_events(events)

    snaps = sn[["dataset_id", "event_id", "row_id", "time_to_tca_days", "miss_distance_m", "n_cdms_2d", "is_latest_snapshot",
                "current_tier", "prediction_status", "quality_flags", "p_safe", "p_elevated", "p_critical",
                "predicted_final_tier", "triage_action", "reason_codes", "explanation_id"]].copy()
    snaps = snaps.rename(columns={"n_cdms_2d": "n_eligible_so_far", "current_tier": "observed_tier", "is_latest_snapshot": "is_latest"})
    snaps["prediction_status"] = snaps["prediction_status"].map(lambda s: STATUS_MAP.get(str(s), PredictionStatus.FAILED).value)
    snaps.loc[snaps["prediction_status"] != PredictionStatus.SUCCESS.value, ["predicted_final_tier", "p_safe", "p_elevated", "p_critical"]] = np.nan
    snaps["review_priority"] = snaps["triage_action"].map(lambda a: ACTION_MAP[a].value if a in ACTION_MAP else None)
    snaps = none_for_missing(snaps.drop(columns=["triage_action"]))
    bad = snaps["time_to_tca_days"] < MIN_TIME_TO_TCA_DAYS - ELIGIBILITY_EPS
    if bad.any():
        raise ArtifactError(f"{int(bad.sum())} snapshot forecasts use a CDM inside the 2-day cutoff")

    inputs = native.inputs.rename(columns=INPUT_RENAME)
    for c in INPUT_STD_COLUMNS:
        if c not in inputs.columns:
            inputs[c] = np.nan
    inputs = inputs[INPUT_STD_COLUMNS]
    if (inputs["time_to_tca_days"] < MIN_TIME_TO_TCA_DAYS - ELIGIBILITY_EPS).any():
        raise ArtifactError("inputs.parquet contains CDMs inside the 2-day cutoff; only eligible CDMs may be shown as forecast inputs")
    return events, snaps.reset_index(drop=True), inputs


OPTIONAL_TEXT = ["predicted_final_tier", "latest_observed_tier", "baseline_tier", "review_priority", "model_action",
                 "explanation_id", "object_type", "latest_input_row_id", "observed_tier"]


def none_for_missing(df: pd.DataFrame, cols=OPTIONAL_TEXT) -> pd.DataFrame:
    """Missing optional text values are None (never NaN, never a default such as 'Safe'), whatever the pandas version."""
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(object).where(df[c].notna(), None)
    return df


def record_dict(row: dict) -> dict:
    """A standard-event row as plain Python values, ready to re-validate against PredictionRecord."""
    out = {}
    for k in PredictionRecord.model_fields:
        if k not in row:
            continue
        v = row[k]
        if isinstance(v, float) and math.isnan(v):
            v = None
        elif hasattr(v, "tolist") and not isinstance(v, str):
            v = v.tolist()
        out[k] = v
    return out


def order_events(events: pd.DataFrame) -> pd.DataFrame:
    """Documented ordering: review priority (Priority review first), then P(final = Critical) high to low,
    then the latest eligible input closest to TCA, then event id."""
    pc = events["p_critical"].fillna(-1.0)
    t = events["latest_input_time_to_tca_days"].fillna(np.inf)
    out = events.assign(_pc=-pc, _t=t).sort_values(["priority_rank", "_pc", "_t", "event_id"], kind="stable")
    out = out.drop(columns=["_pc", "_t"]).reset_index(drop=True)
    out.insert(0, "queue_position", np.arange(1, len(out) + 1))
    return out


# --------------------------------------------------------------------------- #
class ModelAdapter:
    """Loads datasets for the UI and runs live predictions for uploads through the model package."""

    def __init__(self, artifacts_dir: Path = ARTIFACTS_DIR, demo_dir: Path = DEMO_DIR) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.demo_dir = Path(demo_dir)

    # ---------------------------------------------------------------- datasets
    def manifest(self, synthetic: bool = False) -> Manifest:
        return load_manifest(self.demo_dir if synthetic else self.artifacts_dir)

    def load(self, key: str) -> DatasetView:
        synthetic = key == "synthetic"
        root = self.demo_dir if synthetic else self.artifacts_dir
        manifest = load_manifest(root)
        if synthetic != manifest.synthetic:
            raise ArtifactError(f"{root / 'manifest.json'}: synthetic flag is {manifest.synthetic}, expected {synthetic}")
        native = load_native_dataset(root, key, manifest)
        events, snaps, inputs = translate(native, manifest.model_version)
        versions = set(events["model_version"].unique())
        if versions - {manifest.model_version}:
            raise ArtifactError(f"Predictions were made by {sorted(versions)} but the manifest declares {manifest.model_version}")
        info = manifest.datasets[key]
        return DatasetView(key=key, label=info.label, kind="synthetic" if synthetic else "real", manifest=manifest, info=info,
                           events=events, snapshots=snaps, inputs=inputs, explanations=native.explanations,
                           outcomes=native.outcomes)

    # ---------------------------------------------------------------- live model
    def model_bundle(self):
        return _load_bundle(str(self.artifacts_dir / "model"))

    def predict_upload(self, rows: pd.DataFrame, dataset_id: str, label: str) -> tuple[DatasetView, dict]:
        """rows: validated raw CDM rows (model column names, with row_id). Runs the same pipeline used for the
        precomputed predictions: the model drops CDMs under 2 days before building features."""
        from collision_risk.service import predict_batch  # model package (declared dependency)

        manifest = load_manifest(self.artifacts_dir)
        bundle = self.model_bundle()
        if bundle.meta.get("model_version") != manifest.model_version:
            raise ArtifactError(f"Model bundle {bundle.meta.get('model_version')} does not match manifest {manifest.model_version}")
        fo = predict_batch(bundle, rows, dataset_id=dataset_id, with_explanations=True)
        t = pd.to_numeric(rows["time_to_tca"], errors="coerce")
        eligible = t.notna() & (t >= MIN_TIME_TO_TCA_DAYS - ELIGIBILITY_EPS)
        excluded = (~eligible).groupby(rows["event_id"]).sum().astype(int)
        has_eligible = set(rows.loc[eligible, "event_id"].astype(int))
        inel = pd.DataFrame({"dataset_id": dataset_id, "event_id": [int(e) for e in excluded.index if int(e) not in has_eligible]})
        inel["n_excluded_cdms"] = inel["event_id"].map(excluded).astype(int) if len(inel) else pd.Series(dtype=int)
        events = fo.events.copy()
        if len(events):
            events["n_rows_excluded_by_horizon"] = events["event_id"].map(excluded).fillna(0).astype(int)
        snaps = fo.snapshots if len(fo.snapshots) else pd.DataFrame(columns=events.columns.tolist() + ["is_latest_snapshot"])
        inp_cols = ["dataset_id", "event_id", "row_id"] + [c for c in INPUT_RENAME if c in rows.columns]
        inputs = rows.loc[eligible].assign(dataset_id=dataset_id)[inp_cols]
        native = NativeDataset(events=events, snapshots=snaps, inputs=inputs, explanations=fo.explanations, outcomes=None,
                               ineligible=inel if len(inel) else None)
        if len(events) == 0:
            std_events, std_snaps, std_inputs = _empty_translate(native, manifest.model_version)
        else:
            std_events, std_snaps, std_inputs = translate(native, manifest.model_version)
        info = DatasetInfo(label=label, dataset_id=dataset_id, has_outcomes=False, n_events=int(len(std_events)),
                           note="Uploaded during this session; predicted by the bundled model.",
                           n_rows_excluded_by_horizon=int((~eligible).sum()))
        view = DatasetView(key="upload", label=label, kind="upload", manifest=manifest, info=info, events=std_events,
                           snapshots=std_snaps, inputs=std_inputs, explanations=fo.explanations, outcomes=None)
        return view, fo.info


def _empty_translate(native: NativeDataset, model_version: str):
    recs = []
    if native.ineligible is not None:
        for r in native.ineligible.itertuples(index=False):
            recs.append(PredictionRecord(dataset_id=str(r.dataset_id), event_id=int(r.event_id),
                                         prediction_id=prediction_id(model_version, r.dataset_id, r.event_id, None),
                                         prediction_status=PredictionStatus.INSUFFICIENT, model_version=model_version,
                                         n_eligible_cdms=0, n_excluded_cdms=int(r.n_excluded_cdms),
                                         quality_flags=["NO_ELIGIBLE_CDM"], review_priority=ReviewPriority.INSUFFICIENT).model_dump())
    events = none_for_missing(pd.DataFrame(recs))
    if len(events):
        events["priority_rank"] = PRIORITY_RANK[ReviewPriority.INSUFFICIENT]
        events["needs_review"] = True
        events = order_events(events)
    snaps = pd.DataFrame(columns=["dataset_id", "event_id", "row_id", "time_to_tca_days", "miss_distance_m", "n_eligible_so_far",
                                  "is_latest", "observed_tier", "prediction_status", "quality_flags", "p_safe", "p_elevated",
                                  "p_critical", "predicted_final_tier", "reason_codes", "explanation_id", "review_priority"])
    return events, snaps, pd.DataFrame(columns=INPUT_STD_COLUMNS)


@lru_cache(maxsize=2)
def _load_bundle(path: str):
    from collision_risk.service import load_bundle  # model package (declared dependency)
    p = Path(path)
    if not (p / "bundle.json").exists():
        raise ArtifactError(f"Model bundle not found at {p}. Run scripts/build_artifacts.py.")
    return load_bundle(p)
