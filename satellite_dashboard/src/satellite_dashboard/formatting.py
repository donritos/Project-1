"""Units, labels and colours. Source values stay in metres / log10 Pc; conversion happens only here."""
from __future__ import annotations

import math
from typing import Optional

from .config import RISK_FLOOR

# Status palette (fixed). Tier colour is always paired with a text label.
TIER_COLORS = {"Safe": "#0ca30c", "Elevated": "#fab219", "Critical": "#d03b3b"}
NO_TIER_COLOR = "#8a8985"
SERIES_1 = "#3987e5"   # categorical slot 1 (dark mode)
SERIES_2 = "#d95926"   # categorical slot 2 (dark mode)
INK = {"primary": "#f0efec", "secondary": "#c3c2b7", "muted": "#8a8985", "grid": "#2e2e2b", "axis": "#4a4a46"}
SURFACE = "#141413"

PRIORITY_HELP = {
    "Priority review": "Predicted final tier is Critical. Review first.",
    "Data review": "No usable eligible input. Check the data; there is no forecast.",
    "Escalation watch": "Not predicted Critical, but the probability of a Critical final tier is above the watch threshold.",
    "Analyst review": "The model is not confident enough to lower the priority, or disagrees with the latest observed tier.",
    "Monitor": "Predicted final tier is Elevated.",
    "Low-priority monitor": "Predicted Safe and currently Safe, but not confident enough for the lowest priority. Review after Monitor items.",
    "Lower review priority": "Predicted Safe with high confidence. Not a clearance decision.",
}

REASON_TEXT = {
    "FORECAST_CRITICAL": "Forecast: Critical final tier",
    "FORECAST_ELEVATED": "Forecast: Elevated final tier",
    "FORECAST_SAFE": "Forecast: Safe final tier",
    "CURRENT_TIER_CRITICAL": "Latest eligible CDM is already in the Critical tier",
    "CURRENT_TIER_ELEVATED": "Latest eligible CDM is in the Elevated tier",
    "CURRENT_TIER_NOT_SAFE": "Latest eligible CDM is not in the Safe tier",
    "RISK_RISING": "Risk has been rising across eligible CDMs",
    "MISS_SHRINKING": "Miss distance has been shrinking across eligible CDMs",
    "SIGMA_GROWING": "Radial position uncertainty has been growing",
    "HIGH_VOLATILITY": "Risk estimates have varied a lot between CDMs",
    "ESCALATION_RISK": "Probability of a Critical final tier is above the watch threshold",
    "DEESCALATION_CLAIM": "Model predicts a lower tier than the latest observed Critical tier (must be reviewed)",
    "MODEL_CARRY_FORWARD_DISAGREE": "Model and the latest observed tier differ by two tiers",
    "LOW_CONFIDENCE_SAFE": "Predicted Safe, but not confidently enough for lower review priority",
    "NO_HISTORY": "Only one eligible CDM: no trend information",
    "QUALITY_WARNING": "Input quality warnings present",
    "MISSING_TELEMETRY": "Some optional measurements are missing",
    "INVALID_INPUT": "Required geometry is missing or invalid",
    "AMBIGUOUS_PREDICTION_SET": "Prediction set contains non-adjacent tiers",
}

FLAG_TEXT = {
    "MISSING_REQUIRED_GEOMETRY": "Required geometry missing",
    "INVALID_MISS_DISTANCE": "Miss distance not positive",
    "RTN_NORM_MISMATCH": "RTN position does not match miss distance",
    "MISSING_SIGMA": "Radial sigma missing",
    "INVALID_SIGMA": "Radial sigma not positive",
    "SUSPICIOUS_SIGMA": "Radial sigma looks like a placeholder",
    "MISSING_MAX_RISK": "max_risk_estimate missing",
    "MISSING_SPACE_WEATHER": "F10 / AP missing",
    "MISSING_SPEED": "Relative speed missing",
    "INVALID_SPEED": "Relative speed not positive",
    "UNSEEN_OBJECT_TYPE": "Object type not seen in training",
    "MISSING_OBJECT_TYPE": "Object type missing",
    "OUT_OF_TRAINING_RANGE": "Input outside the training range",
    "TIED_TIME_TO_TCA": "Two CDMs share the same time to TCA",
    "NEGATIVE_TIME_TO_TCA": "Negative time to TCA",
    "DUPLICATE_ROW": "Duplicate CDM",
    "NO_ELIGIBLE_CDM": "No CDM at 2 or more days before TCA",
}

GROUP_TEXT = {"geometry": "Encounter geometry", "uncertainty": "Position uncertainty", "risk_trend": "Risk trend",
              "risk_level": "Risk level", "space_weather": "Space weather", "object_type": "Object type",
              "history": "Message history", "other": "Other"}


def is_missing(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def km(m: Optional[float], digits: int = 2) -> str:
    return "–" if is_missing(m) else f"{m / 1000:,.{digits}f} km"


def metres(m: Optional[float]) -> str:
    return "–" if is_missing(m) else f"{m:,.0f} m"


def days(d: Optional[float]) -> str:
    return "–" if is_missing(d) else f"{d:.2f} d"


def pct(p: Optional[float], digits: int = 0) -> str:
    return "–" if is_missing(p) else f"{100 * p:.{digits}f}%"


def log10_pc(x: Optional[float]) -> str:
    """Risk values are log10(Pc). -30 is the dataset placeholder for 'too small to compute'."""
    if is_missing(x):
        return "–"
    if x <= RISK_FLOOR + 1e-9:
        return "≈ 0 (placeholder −30)"
    return f"{x:.2f}  (Pc ≈ {10 ** x:.1e})"


def tier_or_status(tier: Optional[str], status: str) -> str:
    if isinstance(tier, str) and tier:
        return tier
    return status


def tier_badge(tier: Optional[str], text: Optional[str] = None) -> str:
    color = TIER_COLORS.get(tier or "", NO_TIER_COLOR)
    label = text or (tier if tier else "No forecast")
    return (f"<span style='display:inline-flex;align-items:center;gap:6px;padding:2px 10px;border-radius:999px;"
            f"border:1px solid {color};color:{INK['primary']};font-weight:600;font-size:0.9rem'>"
            f"<span style='width:9px;height:9px;border-radius:50%;background:{color};display:inline-block'></span>{label}</span>")


def flags_text(flags: list[str]) -> str:
    return "; ".join(FLAG_TEXT.get(f, f) for f in flags) if flags else "No issues"


def quality_status(status: str, flags: list[str]) -> str:
    if status != "Successful":
        return status
    return "Warnings" if flags else "OK"
