"""The standard records the UI consumes. Views only ever see these field names; the model adapter
translates the model's own output into them."""
from __future__ import annotations

import math
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import ELIGIBILITY_EPS, MIN_TIME_TO_TCA_DAYS

SCHEMA_VERSION = "prediction-record-1.0"
TIER_ORDER = ["Safe", "Elevated", "Critical"]


class Tier(str, Enum):
    SAFE = "Safe"
    ELEVATED = "Elevated"
    CRITICAL = "Critical"


class PredictionStatus(str, Enum):
    SUCCESS = "Successful"
    INSUFFICIENT = "Insufficient eligible data"
    INVALID = "Invalid input"
    FAILED = "Failed"


class ReviewPriority(str, Enum):
    """Analyst actions. These describe who should look at an event next; none of them calculates
    or authorises a manoeuvre."""
    PRIORITY = "Priority review"          # predicted final tier Critical
    WATCH = "Escalation watch"            # not predicted Critical, but P(Critical) above the watch threshold
    ANALYST = "Analyst review"            # model uncertain / disagrees with the latest observed tier
    INSUFFICIENT = "Data review"          # no usable eligible input
    MONITOR = "Monitor"                   # predicted Elevated
    LOW_PRIORITY = "Low-priority monitor"  # predicted Safe, currently Safe, but P(Safe) below the auto-clear bar
    LOWER = "Lower review priority"       # predicted Safe with high confidence (shown instead of "auto-clear")


PRIORITY_RANK = {ReviewPriority.PRIORITY: 1, ReviewPriority.INSUFFICIENT: 2, ReviewPriority.WATCH: 3,
                 ReviewPriority.ANALYST: 4, ReviewPriority.MONITOR: 5, ReviewPriority.LOW_PRIORITY: 6,
                 ReviewPriority.LOWER: 7}
NEEDS_REVIEW = {ReviewPriority.PRIORITY, ReviewPriority.WATCH, ReviewPriority.ANALYST, ReviewPriority.INSUFFICIENT}


class PredictionRecord(BaseModel):
    """One event-level forecast, made from the event's eligible CDMs (time_to_tca >= 2 days)."""
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    # required
    dataset_id: str
    event_id: int
    prediction_id: str
    source_row_ids: list[str] = Field(default_factory=list)
    latest_input_time_to_tca_days: Optional[float] = None
    predicted_final_tier: Optional[Tier] = None
    prediction_status: PredictionStatus
    quality_flags: list[str] = Field(default_factory=list)
    model_version: str
    schema_version: str = SCHEMA_VERSION

    # optional
    latest_input_row_id: Optional[str] = None
    n_eligible_cdms: int = 0
    n_excluded_cdms: Optional[int] = None
    p_safe: Optional[float] = None
    p_elevated: Optional[float] = None
    p_critical: Optional[float] = None
    latest_observed_tier: Optional[Tier] = None
    baseline_tier: Optional[Tier] = None
    latest_miss_distance_m: Optional[float] = None
    object_type: Optional[str] = None
    review_priority: Optional[ReviewPriority] = None
    model_action: Optional[str] = None
    reason_codes: list[str] = Field(default_factory=list)
    explanation_id: Optional[str] = None

    @model_validator(mode="after")
    def _consistent(self) -> "PredictionRecord":
        ok = self.prediction_status == PredictionStatus.SUCCESS.value
        if ok and self.predicted_final_tier is None:
            raise ValueError("a successful prediction needs predicted_final_tier")
        if not ok and self.predicted_final_tier is not None:
            raise ValueError("an unsuccessful prediction must not carry a tier (never default to Safe)")
        probs = [self.p_safe, self.p_elevated, self.p_critical]
        present = [p for p in probs if p is not None and not math.isnan(p)]
        if present:
            if len(present) != 3:
                raise ValueError("class probabilities must be given for all three tiers or none")
            if any(p < -1e-9 or p > 1 + 1e-9 for p in present) or abs(sum(present) - 1.0) > 1e-3:
                raise ValueError(f"class probabilities must lie in [0, 1] and sum to 1, got {present}")
        else:
            self.p_safe = self.p_elevated = self.p_critical = None
        if self.prediction_status == PredictionStatus.INSUFFICIENT.value and self.n_eligible_cdms != 0:
            raise ValueError("insufficient-data events have no eligible CDMs")
        if self.latest_input_time_to_tca_days is not None and \
                self.latest_input_time_to_tca_days < MIN_TIME_TO_TCA_DAYS - ELIGIBILITY_EPS:
            raise ValueError("a forecast input is inside the 2-day cutoff")
        return self


class Capabilities(BaseModel):
    model_config = ConfigDict(extra="ignore")
    class_probabilities: bool = False
    class_probabilities_note: Optional[str] = None
    feature_contributions: bool = False
    feature_contributions_note: Optional[str] = None
    reason_codes: bool = False
    review_recommendation: bool = False
    snapshot_history: bool = False
    predicted_final_pc: bool = False
    predicted_final_miss_distance: bool = False
    uncertainty_intervals: bool = False
    prediction_sets: bool = False
    prediction_sets_note: Optional[str] = None
    single_cdm_inference: bool = False
    single_cdm_inference_note: Optional[str] = None


class DatasetInfo(BaseModel):
    model_config = ConfigDict(extra="ignore")
    label: str
    dataset_id: str
    has_outcomes: bool
    n_events: int
    note: Optional[str] = None
    n_rows_excluded_by_horizon: Optional[int] = None


class Manifest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    schema_version: str
    model: dict
    eligibility: dict
    capabilities: Capabilities
    datasets: dict[str, DatasetInfo]
    split: dict = Field(default_factory=dict)
    tier_rules: dict = Field(default_factory=dict)
    synthetic: bool = False

    @model_validator(mode="after")
    def _eligibility(self) -> "Manifest":
        d = float(self.eligibility.get("min_time_to_tca_days", -1))
        if abs(d - MIN_TIME_TO_TCA_DAYS) > 1e-9:
            raise ValueError(f"artifact eligibility rule ({d} days) differs from the challenge rule ({MIN_TIME_TO_TCA_DAYS} days)")
        return self

    @property
    def model_version(self) -> str:
        return str(self.model.get("model_version", "unknown"))
