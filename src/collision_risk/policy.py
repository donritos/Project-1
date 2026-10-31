"""Official risk-tier rules and triage policy.

Tier rules (challenge definition; all distances in metres, risk in log10(Pc)):

1. Critical  : risk >= critical_risk (-4)  OR  miss < critical_distance_m (1000)
2. Elevated  : risk >= elevated_risk (-6)  OR  critical_distance_m <= miss <= elevated_distance_max_m
3. Safe      : otherwise (risk < -6 AND miss > 5000)

The most severe applicable rule wins. The Elevated distance band, described in the
challenge text as "between 1 and 5 km", is interpreted as INCLUSIVE at both ends
(1000 m and 5000 m both map to Elevated); 1000 m is therefore not Critical.

Invalid data is never classified Safe: a row whose risk is NaN gets a tier only if a
distance rule alone yields Critical or Elevated; otherwise the tier is None with
reason INVALID_INPUT. A NaN or negative miss distance always gives tier None.

Triage precedence (first match wins; reason codes still list every applicable
condition):

1. prediction_status == 'invalid' or tier None -> Human review - insufficient data
2. tier Critical                               -> Maneuver planning review
3. interval upper >= review_upper_threshold, or classifier disagreement
   (p_pc_critical >= classifier_disagreement_prob while tier is not Critical)
                                               -> Human review
4. tier Elevated                               -> Monitor
5. tier Safe, interval present, upper < auto_clear_upper_threshold and acceptable
   quality (status 'valid' or all flags allow-listed) -> Auto-clear candidate
6. everything else                             -> Human review

A missing interval never qualifies for auto-clear.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from collision_risk.contracts import (
    PredictionStatus,
    ReasonCode,
    RiskTier,
    TriageAction,
)

CRITICAL = RiskTier.CRITICAL.value
ELEVATED = RiskTier.ELEVATED.value
SAFE = RiskTier.SAFE.value

TIER_SEVERITY: dict[str, int] = {CRITICAL: 2, ELEVATED: 1, SAFE: 0}

# Stable ordering for reason codes in the output lists (official rules first).
_REASON_ORDER: dict[str, int] = {code.value: i for i, code in enumerate(ReasonCode)}

_MISSING_FLAG_PREFIX = "MISSING_"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _as_float_array(x: Any) -> np.ndarray:
    """Coerce scalars / lists / Series to a 1-D float64 array (None -> NaN)."""
    if isinstance(x, pd.Series):
        arr = pd.to_numeric(x, errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
    else:
        arr = np.asarray(x, dtype="object") if not isinstance(x, np.ndarray) else x
        if arr.dtype == object:
            arr = pd.to_numeric(pd.Series(arr.ravel()), errors="coerce").to_numpy(
                dtype="float64", na_value=np.nan
            )
        else:
            arr = np.asarray(arr, dtype="float64")
    return np.atleast_1d(arr).astype("float64", copy=False)


def _order_codes(codes: Iterable[str]) -> list[str]:
    """Deduplicate and sort reason codes in the ReasonCode enum order."""
    seen = dict.fromkeys(codes)
    return sorted(seen, key=lambda c: (_REASON_ORDER.get(c, len(_REASON_ORDER)), c))


def _tier_masks(
    risk: np.ndarray, miss: np.ndarray, thresholds: dict
) -> dict[str, np.ndarray]:
    """Vectorised evaluation of each official rule condition."""
    crit_risk = float(thresholds["critical_risk"])
    elev_risk = float(thresholds["elevated_risk"])
    crit_dist = float(thresholds["critical_distance_m"])
    elev_dist_max = float(thresholds["elevated_distance_max_m"])

    risk_ok = np.isfinite(risk)
    miss_ok = np.isfinite(miss) & (miss >= 0.0)

    with np.errstate(invalid="ignore"):
        pc_critical = risk_ok & (risk >= crit_risk)
        pc_elevated = risk_ok & (risk >= elev_risk) & ~pc_critical
        dist_critical = miss_ok & (miss < crit_dist)
        dist_elevated = miss_ok & (miss >= crit_dist) & (miss <= elev_dist_max)
        # Safe requires BOTH inputs valid and both rules to be in the safe zone.
        safe = risk_ok & miss_ok & (risk < elev_risk) & (miss > elev_dist_max)

    return {
        "pc_critical": pc_critical,
        "pc_elevated": pc_elevated,
        "dist_critical": dist_critical,
        "dist_elevated": dist_elevated,
        "safe": safe,
        "miss_ok": miss_ok,
        "risk_ok": risk_ok,
    }


def _tiers_from_masks(m: dict[str, np.ndarray]) -> np.ndarray:
    n = m["pc_critical"].shape[0]
    tier = np.full(n, None, dtype=object)
    # Miss distance must be valid for ANY tier assignment.
    critical = m["miss_ok"] & (m["pc_critical"] | m["dist_critical"])
    elevated = m["miss_ok"] & ~critical & (m["pc_elevated"] | m["dist_elevated"])
    safe = m["safe"] & ~critical & ~elevated
    tier[critical] = CRITICAL
    tier[elevated] = ELEVATED
    tier[safe] = SAFE
    return tier


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def assign_risk_tier(
    risk_pred: Any, miss_distance_m: Any, thresholds: dict
) -> tuple[np.ndarray, list[list[str]]]:
    """Apply the official challenge tier rules.

    Parameters
    ----------
    risk_pred : 1-D float array-like of predicted log10(Pc); NaN allowed.
    miss_distance_m : 1-D float array-like of miss distance in metres; NaN allowed.
    thresholds : dict with keys critical_risk, elevated_risk, critical_distance_m,
        elevated_distance_max_m (see configs/default.yaml -> thresholds).

    Returns
    -------
    (tier, reasons) where ``tier`` is an object array holding 'Critical',
    'Elevated', 'Safe' or None and ``reasons`` is a per-row list of reason codes
    produced by the official rules only (PC_*/DISTANCE_*/SAFE_RULE), plus
    INVALID_INPUT for rows that cannot be tiered.

    The Elevated distance band (1 to 5 km) is inclusive at both ends. Invalid
    inputs are never classified Safe.
    """
    risk = _as_float_array(risk_pred)
    miss = _as_float_array(miss_distance_m)
    if risk.shape != miss.shape:
        raise ValueError(
            f"risk_pred and miss_distance_m must have the same length "
            f"({risk.shape[0]} != {miss.shape[0]})"
        )

    m = _tier_masks(risk, miss, thresholds)
    tier = _tiers_from_masks(m)

    pc_c = ReasonCode.PC_CRITICAL.value
    pc_e = ReasonCode.PC_ELEVATED.value
    d_c = ReasonCode.DISTANCE_CRITICAL.value
    d_e = ReasonCode.DISTANCE_ELEVATED.value
    safe_rule = ReasonCode.SAFE_RULE.value
    invalid = ReasonCode.INVALID_INPUT.value

    pc_critical = m["pc_critical"]
    pc_elevated = m["pc_elevated"]
    dist_critical = m["dist_critical"]
    dist_elevated = m["dist_elevated"]

    reasons: list[list[str]] = []
    for i in range(tier.shape[0]):
        t = tier[i]
        if t is None:
            reasons.append([invalid])
            continue
        codes: list[str] = []
        if t == CRITICAL:
            if pc_critical[i]:
                codes.append(pc_c)
            if dist_critical[i]:
                codes.append(d_c)
        elif t == ELEVATED:
            if pc_elevated[i]:
                codes.append(pc_e)
            if dist_elevated[i]:
                codes.append(d_e)
        else:
            codes.append(safe_rule)
        reasons.append(codes)
    return tier, reasons


def tier_from_true_risk(risk_true: Any, miss_distance_m: Any, thresholds: dict) -> np.ndarray:
    """Official tier rules applied to ground-truth risk (evaluation labels).

    Identical logic to :func:`assign_risk_tier`; returns only the tier array
    (object dtype, None where the truth cannot be tiered).
    """
    tier, _ = assign_risk_tier(risk_true, miss_distance_m, thresholds)
    return tier


def _column_or_default(df: pd.DataFrame, name: str, default: Any) -> pd.Series:
    if name in df.columns:
        return df[name]
    return pd.Series([default] * len(df), index=df.index, dtype=object)


def _normalise_flags(value: Any) -> list[str]:
    """Return a clean list[str] of quality flags for one row."""
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
        return [str(v) for v in value if v is not None and str(v) != ""]
    return [str(value)]


def assign_triage(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Assign risk tiers and triage actions.

    Reads columns ``prediction_status``, ``risk_log10_pred``, ``miss_distance_m``,
    ``risk_log10_upper``, ``p_pc_critical`` (optional) and ``quality_flags``
    (optional). Adds/overwrites ``risk_tier`` (object, None allowed),
    ``triage_action`` (str) and ``reason_codes`` (list[str] per row).

    The input frame is not mutated; a copy with the same index order is returned.
    """
    out = df.copy()
    n = len(out)
    thresholds = cfg["thresholds"]
    triage_cfg = cfg["triage"]

    review_upper = float(triage_cfg["review_upper_threshold"])
    auto_clear_upper = float(triage_cfg["auto_clear_upper_threshold"])
    disagreement_prob = float(triage_cfg["classifier_disagreement_prob"])
    allowed_flags = set(str(f) for f in triage_cfg.get("auto_clear_allowed_flags", []) or [])

    status = (
        _column_or_default(out, "prediction_status", PredictionStatus.VALID.value)
        .astype(object)
        .where(lambda s: s.notna(), PredictionStatus.INVALID.value)
        .map(lambda s: str(s).lower())
        .to_numpy(dtype=object)
    )
    risk = _as_float_array(_column_or_default(out, "risk_log10_pred", np.nan))
    miss = _as_float_array(_column_or_default(out, "miss_distance_m", np.nan))
    upper = _as_float_array(_column_or_default(out, "risk_log10_upper", np.nan))
    p_crit = _as_float_array(_column_or_default(out, "p_pc_critical", np.nan))
    flags_col = _column_or_default(out, "quality_flags", None).tolist()

    is_invalid_status = status == PredictionStatus.INVALID.value
    is_valid_status = status == PredictionStatus.VALID.value

    # Official rules. Invalid predictions are forced to tier None.
    tier, rule_reasons = assign_risk_tier(risk, miss, thresholds)
    tier = tier.copy()
    tier[is_invalid_status] = None

    is_critical = tier == CRITICAL
    is_elevated = tier == ELEVATED
    is_safe = tier == SAFE
    tier_none = np.array([t is None for t in tier], dtype=bool)

    has_upper = np.isfinite(upper)
    with np.errstate(invalid="ignore"):
        upper_critical = has_upper & (upper >= review_upper)
        upper_elevated = has_upper & (upper >= auto_clear_upper)
        disagreement = np.isfinite(p_crit) & (p_crit >= disagreement_prob) & ~is_critical

    # Quality acceptability for auto-clear.
    flags_list: list[list[str]] = [_normalise_flags(f) for f in flags_col]
    flags_allowed = np.array(
        [all(f in allowed_flags for f in fl) for fl in flags_list], dtype=bool
    )
    has_missing_flag = np.array(
        [any(f.startswith(_MISSING_FLAG_PREFIX) for f in fl) for fl in flags_list], dtype=bool
    )
    quality_ok = is_valid_status | flags_allowed

    # Vectorised precedence (first match wins).
    insufficient = is_invalid_status | tier_none
    maneuver = ~insufficient & is_critical
    review_uncertain = ~insufficient & ~maneuver & (upper_critical | disagreement)
    monitor = ~insufficient & ~maneuver & ~review_uncertain & is_elevated
    auto_clear = (
        ~insufficient
        & ~maneuver
        & ~review_uncertain
        & ~monitor
        & is_safe
        & has_upper
        & (upper < auto_clear_upper)
        & quality_ok
    )

    action = np.full(n, TriageAction.HUMAN_REVIEW.value, dtype=object)
    action[auto_clear] = TriageAction.AUTO_CLEAR_CANDIDATE.value
    action[monitor] = TriageAction.MONITOR.value
    action[review_uncertain] = TriageAction.HUMAN_REVIEW.value
    action[maneuver] = TriageAction.MANEUVER_PLANNING_REVIEW.value
    action[insufficient] = TriageAction.HUMAN_REVIEW_INSUFFICIENT_DATA.value

    RC = ReasonCode
    reason_codes: list[list[str]] = []
    for i in range(n):
        codes: list[str] = list(rule_reasons[i])
        if insufficient[i]:
            codes.append(RC.INVALID_INPUT.value)
        if upper_critical[i]:
            codes.append(RC.UPPER_BOUND_CRITICAL.value)
        if disagreement[i]:
            codes.append(RC.MODEL_DISAGREEMENT.value)
        if is_safe[i]:
            # Document why a Safe row did not (or would not) auto-clear.
            if not has_upper[i]:
                codes.append(RC.INTERVAL_UNAVAILABLE.value)
            elif upper[i] >= auto_clear_upper:
                codes.append(RC.UPPER_BOUND_ELEVATED.value)
            if not quality_ok[i]:
                codes.append(RC.QUALITY_WARNING.value)
                if has_missing_flag[i]:
                    codes.append(RC.MISSING_TELEMETRY.value)
        reason_codes.append(_order_codes(codes))

    out["risk_tier"] = pd.Series(tier, index=out.index, dtype=object)
    out["triage_action"] = pd.Series(action, index=out.index, dtype=object).astype(str)
    out["reason_codes"] = pd.Series(reason_codes, index=out.index, dtype=object)
    return out


def apply_policy(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Convenience wrapper: compute tiers from the frame's columns and triage."""
    return assign_triage(df, cfg)


__all__ = [
    "TIER_SEVERITY",
    "assign_risk_tier",
    "tier_from_true_risk",
    "assign_triage",
    "apply_policy",
]
