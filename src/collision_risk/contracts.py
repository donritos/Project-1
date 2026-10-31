"""Shared data contracts between the prediction engine, evaluation, and dashboard.

Everything the dashboard consumes is defined here. Field semantics:
- risk values are log10(Pc); pc_pred = 10 ** risk_log10_pred.
- Distances are metres, time_to_tca is days.
- Nullable fields are None for invalid records; None is distinct from 0.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator

SCHEMA_VERSION = "2.0.0"


class RiskTier(str, Enum):
    CRITICAL = "Critical"
    ELEVATED = "Elevated"
    SAFE = "Safe"


class TriageAction(str, Enum):
    HUMAN_REVIEW_INSUFFICIENT_DATA = "Human review - insufficient data"
    MANEUVER_PLANNING_REVIEW = "Maneuver planning review"
    HUMAN_REVIEW = "Human review"
    ESCALATION_WATCH = "Escalation watch"
    MONITOR = "Monitor"
    LOW_PRIORITY_MONITOR = "Low-priority monitor"   # predicted Safe, current Safe, good quality, but P(Safe) below the auto-clear bar
    AUTO_CLEAR_CANDIDATE = "Auto-clear candidate"


class PredictionStatus(str, Enum):
    VALID = "valid"
    WARNING = "warning"
    INVALID = "invalid"


class ReasonCode(str, Enum):
    # Official tier rules
    DISTANCE_CRITICAL = "DISTANCE_CRITICAL"        # miss_distance < 1000 m
    PC_CRITICAL = "PC_CRITICAL"                    # risk_pred >= -4
    DISTANCE_ELEVATED = "DISTANCE_ELEVATED"        # 1000 <= miss_distance <= 5000 m
    PC_ELEVATED = "PC_ELEVATED"                    # -6 <= risk_pred < -4
    SAFE_RULE = "SAFE_RULE"                        # risk_pred < -6 and miss_distance > 5000 m
    # Uncertainty / policy conditions
    UPPER_BOUND_CRITICAL = "UPPER_BOUND_CRITICAL"  # interval upper >= -4
    UPPER_BOUND_ELEVATED = "UPPER_BOUND_ELEVATED"  # interval upper >= -6 (blocks auto-clear)
    MODEL_DISAGREEMENT = "MODEL_DISAGREEMENT"      # classifier says critical, regressor does not
    INTERVAL_UNAVAILABLE = "INTERVAL_UNAVAILABLE"  # no interval -> cannot auto-clear
    QUALITY_WARNING = "QUALITY_WARNING"            # quality flags present -> cannot auto-clear
    INVALID_INPUT = "INVALID_INPUT"                # required geometry missing/invalid
    MISSING_TELEMETRY = "MISSING_TELEMETRY"        # optional predictor missing
    NO_HISTORY = "NO_HISTORY"                      # standalone CDM, history features defaulted
    # Event-level forecast conditions (final-tier task)
    FORECAST_CRITICAL = "FORECAST_CRITICAL"        # predicted final tier is Critical
    FORECAST_ELEVATED = "FORECAST_ELEVATED"        # predicted final tier is Elevated
    FORECAST_SAFE = "FORECAST_SAFE"                # predicted final tier is Safe
    CURRENT_TIER_CRITICAL = "CURRENT_TIER_CRITICAL"
    CURRENT_TIER_ELEVATED = "CURRENT_TIER_ELEVATED"
    RISK_RISING = "RISK_RISING"
    MISS_SHRINKING = "MISS_SHRINKING"
    SIGMA_GROWING = "SIGMA_GROWING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    ESCALATION_RISK = "ESCALATION_RISK"            # P(Critical) >= t_watch while not predicted Critical
    DEESCALATION_CLAIM = "DEESCALATION_CLAIM"      # current Critical, predicted lower
    MODEL_CARRY_FORWARD_DISAGREE = "MODEL_CARRY_FORWARD_DISAGREE"  # two-tier disagreement
    AMBIGUOUS_PREDICTION_SET = "AMBIGUOUS_PREDICTION_SET"          # conformal set has non-adjacent classes
    LOW_CONFIDENCE_SAFE = "LOW_CONFIDENCE_SAFE"    # predicted Safe but P(Safe) < t_clear
    CURRENT_TIER_NOT_SAFE = "CURRENT_TIER_NOT_SAFE"  # blocks auto-clear


class QualityFlag(str, Enum):
    MISSING_REQUIRED_GEOMETRY = "MISSING_REQUIRED_GEOMETRY"
    INVALID_MISS_DISTANCE = "INVALID_MISS_DISTANCE"
    RTN_NORM_MISMATCH = "RTN_NORM_MISMATCH"
    MISSING_SIGMA = "MISSING_SIGMA"
    INVALID_SIGMA = "INVALID_SIGMA"              # negative or zero
    SUSPICIOUS_SIGMA = "SUSPICIOUS_SIGMA"        # absurdly large placeholder
    MISSING_MAX_RISK = "MISSING_MAX_RISK"
    MISSING_SPACE_WEATHER = "MISSING_SPACE_WEATHER"
    MISSING_SPEED = "MISSING_SPEED"
    INVALID_SPEED = "INVALID_SPEED"
    UNSEEN_OBJECT_TYPE = "UNSEEN_OBJECT_TYPE"
    MISSING_OBJECT_TYPE = "MISSING_OBJECT_TYPE"
    OUT_OF_TRAINING_RANGE = "OUT_OF_TRAINING_RANGE"
    TIED_TIME_TO_TCA = "TIED_TIME_TO_TCA"
    NEGATIVE_TIME_TO_TCA = "NEGATIVE_TIME_TO_TCA"
    DUPLICATE_ROW = "DUPLICATE_ROW"


class PredictionRecord(BaseModel):
    """One record per input CDM. This is the dashboard contract."""

    schema_version: str = SCHEMA_VERSION
    dataset_id: str
    row_id: str
    event_id: int
    source_row_index: int
    prediction_status: PredictionStatus

    time_to_tca_days: Optional[float]
    miss_distance_m: Optional[float]
    relative_position_r_m: Optional[float]
    relative_position_t_m: Optional[float]
    relative_position_n_m: Optional[float]
    object_type: Optional[str]

    risk_log10_pred: Optional[float]
    pc_pred: Optional[float]
    risk_log10_lower: Optional[float]
    risk_log10_upper: Optional[float]
    interval_nominal_coverage: Optional[float]
    p_pc_critical: Optional[float] = None

    risk_tier: Optional[RiskTier]
    triage_action: TriageAction
    reason_codes: list[str] = Field(default_factory=list)
    quality_flags: list[str] = Field(default_factory=list)

    explanation_id: Optional[str]
    model_version: str
    feature_version: str
    policy_version: str

    @field_validator("pc_pred")
    @classmethod
    def _pc_in_unit_interval(cls, v):
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("pc_pred must be in [0, 1]")
        return v

    @field_validator("risk_tier", mode="before")
    @classmethod
    def _tier_none_only_if_invalid(cls, v, info):
        return v


class FeatureContribution(BaseModel):
    feature: str
    label: str
    value: Optional[float]
    contribution: float  # in log10(Pc) units for the explained output


class ExplanationRecord(BaseModel):
    """Local feature contributions for one prediction. Explains a single model
    component (e.g. the LightGBM point regressor), not the gated/clipped system."""

    explanation_id: str
    dataset_id: str
    row_id: str
    component: str                    # e.g. "lgbm_point_regressor"
    output_units: str = "log10(Pc)"
    baseline: float                   # expected value / bias term
    explained_output: float           # raw model output being decomposed
    final_output: Optional[float]     # output after post-processing (clip/floor), if different
    postprocessing_note: Optional[str]
    top_positive: list[FeatureContribution]
    top_negative: list[FeatureContribution]
    remaining_contribution: float     # sum of contributions not shown
    text: str                         # deterministic template rendering
    model_version: str


# Column order for predictions.parquet (dashboard reads these names).
PREDICTION_COLUMNS: list[str] = list(PredictionRecord.model_fields.keys())


def validate_prediction_rows(rows: list[dict]) -> list[PredictionRecord]:
    return [PredictionRecord.model_validate(r) for r in rows]


class EventForecastRecord(BaseModel):
    """One record per event (or per snapshot) for the final-tier forecasting task.
    Probabilities are calibrated and sum to 1; None for invalid rows."""

    schema_version: str = SCHEMA_VERSION
    dataset_id: str
    event_id: int
    row_id: str                              # snapshot CDM that the forecast was made from
    time_to_tca_days: Optional[float]
    miss_distance_m: Optional[float]
    relative_position_r_m: Optional[float] = None    # RTN geometry of the snapshot, for the dashboard timeline
    relative_position_t_m: Optional[float] = None
    relative_position_n_m: Optional[float] = None
    is_latest_snapshot: bool
    n_cdms_2d: int
    has_history: int
    current_tier: Optional[str]
    current_tier_source: str                 # "risk" | "mre"
    latest_risk: Optional[float]
    prediction_status: PredictionStatus
    quality_flags: list[str] = Field(default_factory=list)
    p_safe: Optional[float]                  # calibrated probability that the event's FINAL tier is Safe
    p_elevated: Optional[float]              # ... is Elevated
    p_critical: Optional[float]              # ... is Critical. NOT a physical collision probability (Pc).
    predicted_final_tier: Optional[RiskTier]
    prediction_set: Optional[list[str]] = None
    triage_action: TriageAction
    reason_codes: list[str] = Field(default_factory=list)
    explanation_id: Optional[str]
    model_version: str
    feature_version: str
    policy_version: str

    @field_validator("p_safe", "p_elevated", "p_critical")
    @classmethod
    def _prob_in_unit_interval(cls, v):
        if v is not None and not (-1e-9 <= v <= 1 + 1e-9):
            raise ValueError("probabilities must be in [0, 1]")
        return v


EVENT_FORECAST_COLUMNS: list[str] = list(EventForecastRecord.model_fields.keys())
