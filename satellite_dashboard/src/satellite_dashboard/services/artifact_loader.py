"""Reads the artifact bundle from disk. Fails loudly: a missing or malformed real artifact raises
ArtifactError and is never replaced by synthetic fixtures."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from ..contracts import Manifest

EVENT_COLUMNS = ["dataset_id", "event_id", "row_id", "time_to_tca_days", "miss_distance_m", "n_cdms_2d",
                 "current_tier", "carry_forward_tier", "prediction_status", "quality_flags", "p_safe", "p_elevated",
                 "p_critical", "predicted_final_tier", "triage_action", "reason_codes", "explanation_id", "model_version"]
SNAPSHOT_COLUMNS = ["dataset_id", "event_id", "row_id", "time_to_tca_days", "miss_distance_m", "n_cdms_2d",
                    "is_latest_snapshot", "current_tier", "prediction_status", "quality_flags", "p_safe", "p_elevated",
                    "p_critical", "predicted_final_tier", "triage_action", "reason_codes", "explanation_id"]
INPUT_COLUMNS = ["dataset_id", "event_id", "row_id", "time_to_tca", "miss_distance"]
OUTCOME_COLUMNS = ["dataset_id", "event_id", "final_tier"]
EXPLANATION_COLUMNS = ["explanation_id", "text"]


class ArtifactError(RuntimeError):
    """A required artifact is missing or does not match the expected contract."""


@dataclass
class NativeDataset:
    """Model-native frames for one dataset, exactly as the model exported them."""
    events: pd.DataFrame
    snapshots: pd.DataFrame
    inputs: pd.DataFrame
    explanations: Optional[pd.DataFrame]
    outcomes: Optional[pd.DataFrame]
    ineligible: Optional[pd.DataFrame]  # events present in the input with no eligible CDM


def load_manifest(root: Path) -> Manifest:
    path = Path(root) / "manifest.json"
    if not path.exists():
        raise ArtifactError(f"Artifact manifest not found: {path}. Run scripts/build_artifacts.py (real data) "
                            f"or scripts/make_fixtures.py (synthetic demo).")
    try:
        return Manifest.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except Exception as exc:  # pydantic or json error
        raise ArtifactError(f"Artifact manifest {path} is invalid: {exc}") from exc


def _read(path: Path, required: list[str], *, optional: bool = False) -> Optional[pd.DataFrame]:
    if not path.exists():
        if optional:
            return None
        raise ArtifactError(f"Required artifact missing: {path}")
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        raise ArtifactError(f"Could not read {path}: {exc}") from exc
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ArtifactError(f"{path.name} is missing required columns {missing}")
    return df


def load_native_dataset(root: Path, key: str, manifest: Manifest) -> NativeDataset:
    if key not in manifest.datasets:
        raise ArtifactError(f"Dataset '{key}' is not declared in {Path(root) / 'manifest.json'}")
    d = Path(root) / "predictions" / key
    info = manifest.datasets[key]
    events = _read(d / "events.parquet", EVENT_COLUMNS)
    snapshots = _read(d / "snapshots.parquet", SNAPSHOT_COLUMNS)
    inputs = _read(d / "inputs.parquet", INPUT_COLUMNS)
    explanations = _read(d / "explanations.parquet", EXPLANATION_COLUMNS, optional=True)
    if manifest.capabilities.feature_contributions and explanations is None:
        raise ArtifactError(f"The manifest declares feature contributions but {d / 'explanations.parquet'} is missing. "
                            "Re-run scripts/build_artifacts.py.")
    outcomes = _read(d / "outcomes.parquet", OUTCOME_COLUMNS, optional=not info.has_outcomes)
    ineligible = _read(d / "ineligible_events.parquet", ["dataset_id", "event_id", "n_excluded_cdms"], optional=True)
    for name, df in (("events", events), ("snapshots", snapshots), ("inputs", inputs)):
        ids = set(df["dataset_id"].unique())
        if ids - {info.dataset_id}:
            raise ArtifactError(f"{key}/{name}.parquet contains dataset ids {sorted(ids)}; expected only '{info.dataset_id}'")
    return NativeDataset(events, snapshots, inputs, explanations, outcomes, ineligible)
