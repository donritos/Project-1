"""Official risk-tier rules and event-level forecast triage policy.

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

Event-forecast triage (one row per event; see docs/forecast_schema.md). The model
forecasts the event's FINAL tier with calibrated probabilities p_safe / p_elevated /
p_critical and a frozen decision rule that already produced ``predicted_final_tier``.
Precedence (first match wins; reason codes still list every applicable condition):

1. prediction_status 'invalid', predicted tier None, or any NaN probability
                                            -> Human review - insufficient data
2. predicted Critical                       -> Maneuver planning review
3. not predicted Critical and p_critical >= t_watch
                                            -> Escalation watch
4. two-tier disagreement with the carry-forward tier, an ambiguous (non-adjacent)
   prediction set, or a de-escalation claim (current Critical, predicted lower) with
   p_critical < deescalation_p_critical    -> Human review
5. predicted Elevated                       -> Monitor
6. predicted Safe, p_safe >= t_clear, current tier Safe and acceptable quality
   (status 'valid' or every flag allow-listed)
                                            -> Auto-clear candidate
7. predicted Safe, p_safe < t_clear, current tier Safe and acceptable quality
                                            -> Low-priority monitor
8. everything else (predicted Safe with a non-Safe current tier or blocking quality
   flags)                                   -> Human review

Rule 7 exists because the auto-clear bar is tuned for zero final-Critical events
cleared and therefore sits very high; without it roughly half of all events, most of
them confidently predicted Safe, fell through to Human review. A low-priority monitor
item is still reviewed, but after Maneuver planning, Escalation watch, Human review and
Monitor items. A de-escalation claim is never auto-cleared or low-priority (rule 4 or
the current-tier condition of rules 6 and 7 always intercepts it). ``t_watch`` and ``t_clear`` are tuned on dev
out-of-fold forecasts by :func:`tune_policy_thresholds`.
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


# --------------------------------------------------------------------------- #
# Event-forecast triage
# --------------------------------------------------------------------------- #
_SEVERITY_NONE = -1

_THRESHOLD_GRID_STEP = 0.01


def _column_or_default(df: pd.DataFrame, name: str, default: Any) -> pd.Series:
    if name in df.columns:
        return df[name]
    return pd.Series([default] * len(df), index=df.index, dtype=object)


def _is_missing_scalar(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    try:
        return bool(pd.isna(value)) if np.isscalar(value) else False
    except (TypeError, ValueError):
        return False


def _normalise_flags(value: Any) -> list[str]:
    """Return a clean list[str] of quality flags for one row."""
    if _is_missing_scalar(value):
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple, set, frozenset, np.ndarray, pd.Series)):
        return [str(v) for v in value if v is not None and str(v) != ""]
    return [str(value)]


def _normalise_tier(value: Any) -> str | None:
    """Coerce a tier cell (str / RiskTier / None / NaN) to 'Critical'|'Elevated'|'Safe'|None."""
    if _is_missing_scalar(value):
        return None
    if isinstance(value, RiskTier):
        return value.value
    text = str(value).strip()
    if not text:
        return None
    for tier in (CRITICAL, ELEVATED, SAFE):
        if text.lower() == tier.lower():
            return tier
    return None


def _severity_array(values: Iterable[Any]) -> np.ndarray:
    """Map tier cells to integer severity (2/1/0); -1 for None / unrecognised."""
    return np.array(
        [TIER_SEVERITY.get(_normalise_tier(v), _SEVERITY_NONE) for v in values], dtype=np.int64
    )


def _status_array(df: pd.DataFrame) -> np.ndarray:
    """Lower-cased prediction_status; absent column -> 'valid', NaN -> 'invalid'."""
    status = _column_or_default(df, "prediction_status", PredictionStatus.VALID.value)
    out = np.empty(len(df), dtype=object)
    for i, s in enumerate(status.tolist()):
        if _is_missing_scalar(s):
            out[i] = PredictionStatus.INVALID.value
        elif isinstance(s, PredictionStatus):
            out[i] = s.value
        else:
            out[i] = str(s).strip().lower()
    return out


def _quality_masks(df: pd.DataFrame, cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (quality_ok, has_missing_flag, is_invalid_status) boolean arrays.

    Acceptable quality = prediction_status == 'valid' OR every quality flag is in
    cfg['triage']['auto_clear_allowed_flags'].
    """
    triage_cfg = cfg["triage"]
    allowed = set(str(f) for f in (triage_cfg.get("auto_clear_allowed_flags") or []))
    status = _status_array(df)
    flags = [_normalise_flags(f) for f in _column_or_default(df, "quality_flags", None).tolist()]
    flags_allowed = np.array([all(f in allowed for f in fl) for fl in flags], dtype=bool)
    has_missing = np.array(
        [any(f.startswith(_MISSING_FLAG_PREFIX) for f in fl) for fl in flags], dtype=bool
    )
    is_invalid = status == PredictionStatus.INVALID.value
    is_valid = status == PredictionStatus.VALID.value
    quality_ok = (is_valid | flags_allowed) & ~is_invalid
    return quality_ok, has_missing, is_invalid


def _prediction_set_ambiguous(value: Any) -> bool:
    """True when a conformal prediction set holds two non-adjacent classes (Safe and Critical)."""
    if _is_missing_scalar(value):
        return False
    if isinstance(value, str):
        members = [value]
    elif isinstance(value, (list, tuple, set, frozenset, np.ndarray, pd.Series)):
        members = list(value)
    else:
        return False
    tiers = {_normalise_tier(m) for m in members}
    return SAFE in tiers and CRITICAL in tiers


def _probability_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p_safe = _as_float_array(_column_or_default(df, "p_safe", np.nan))
    p_elev = _as_float_array(_column_or_default(df, "p_elevated", np.nan))
    p_crit = _as_float_array(_column_or_default(df, "p_critical", np.nan))
    return p_safe, p_elev, p_crit


def _trend_masks(df: pd.DataFrame, cfg: dict) -> dict[str, np.ndarray]:
    """Trend / volatility reason-code conditions (NaN never fires).

    Sign conventions (slopes are against time_to_tca, which decreases toward TCA):
    slope_log_miss >= miss_shrink_slope -> MISS_SHRINKING; slope_risk_or_mre <=
    risk_rise_slope -> RISK_RISING; slope_log_sigma <= sigma_grow_slope -> SIGMA_GROWING;
    volatility_mre >= mre_volatility or volatility_risk >= risk_volatility -> HIGH_VOLATILITY.
    """
    trends = (cfg.get("triage") or {}).get("trends") or {}
    n = len(df)

    def _col(name: str) -> np.ndarray:
        return _as_float_array(_column_or_default(df, name, np.nan))

    def _ge(values: np.ndarray, key: str) -> np.ndarray:
        if trends.get(key) is None:
            return np.zeros(n, dtype=bool)
        with np.errstate(invalid="ignore"):
            return np.isfinite(values) & (values >= float(trends[key]))

    def _le(values: np.ndarray, key: str) -> np.ndarray:
        if trends.get(key) is None:
            return np.zeros(n, dtype=bool)
        with np.errstate(invalid="ignore"):
            return np.isfinite(values) & (values <= float(trends[key]))

    return {
        "miss_shrinking": _ge(_col("slope_log_miss"), "miss_shrink_slope"),
        "risk_rising": _le(_col("slope_risk_or_mre"), "risk_rise_slope"),
        "sigma_growing": _le(_col("slope_log_sigma"), "sigma_grow_slope"),
        "high_volatility": _ge(_col("volatility_mre"), "mre_volatility")
        | _ge(_col("volatility_risk"), "risk_volatility"),
    }


def _policy_threshold(policy_params: dict, key: str) -> float:
    if policy_params is None or key not in policy_params:
        raise KeyError(f"policy_params must contain '{key}' (see tune_policy_thresholds)")
    value = float(policy_params[key])
    if not np.isfinite(value):
        raise ValueError(f"policy_params['{key}'] must be finite, got {value!r}")
    return value


def assign_triage(df: pd.DataFrame, cfg: dict, policy_params: dict) -> pd.DataFrame:
    """Event-level triage of final-tier forecasts.

    Parameters
    ----------
    df : event-forecast frame (docs/forecast_schema.md). Consumed columns:
        ``prediction_status``, ``quality_flags`` (list[str]; may be absent),
        ``predicted_final_tier`` (str/None), ``p_safe``, ``p_elevated``, ``p_critical``,
        ``current_tier`` (str/None), ``prediction_set`` (list[str]/None; may be absent),
        ``carry_forward_tier`` (str/None; falls back to ``current_tier`` when absent),
        ``slope_log_miss``, ``slope_risk_or_mre``, ``slope_log_sigma``, ``volatility_mre``,
        ``volatility_risk`` (floats, NaN allowed; may be absent), ``has_history`` (0/1; may
        be absent).
    cfg : project config; uses ``cfg['triage']`` (``deescalation_p_critical``,
        ``auto_clear_allowed_flags``, ``trends``).
    policy_params : ``{"t_watch": float, "t_clear": float}`` tuned on dev OOF forecasts
        (see :func:`tune_policy_thresholds`).

    Returns
    -------
    A copy of ``df`` (same index order) with ``triage_action`` (str) and
    ``reason_codes`` (list[str], deduplicated, in ReasonCode enum order). The input is
    not mutated.
    """
    t_watch = _policy_threshold(policy_params, "t_watch")
    t_clear = _policy_threshold(policy_params, "t_clear")
    triage_cfg = cfg["triage"]
    deesc_p = float(triage_cfg["deescalation_p_critical"])

    out = df.copy()
    n = len(out)

    quality_ok, has_missing_flag, is_invalid_status = _quality_masks(out, cfg)
    p_safe, p_elev, p_crit = _probability_arrays(out)
    probs_ok = np.isfinite(p_safe) & np.isfinite(p_elev) & np.isfinite(p_crit)

    pred_sev = _severity_array(_column_or_default(out, "predicted_final_tier", None).tolist())
    cur_sev = _severity_array(_column_or_default(out, "current_tier", None).tolist())
    if "carry_forward_tier" in out.columns:
        cf_sev = _severity_array(out["carry_forward_tier"].tolist())
        # Fall back to current_tier for rows where carry-forward is missing.
        cf_sev = np.where(cf_sev == _SEVERITY_NONE, cur_sev, cf_sev)
    else:
        cf_sev = cur_sev.copy()
    set_ambiguous = np.array(
        [_prediction_set_ambiguous(v) for v in _column_or_default(out, "prediction_set", None).tolist()],
        dtype=bool,
    )
    has_history = _as_float_array(_column_or_default(out, "has_history", np.nan))
    no_history = np.isfinite(has_history) & (has_history == 0)
    trends = _trend_masks(out, cfg)

    # ---- rule conditions (every applicable condition, independent of precedence) ----
    insufficient = is_invalid_status | (pred_sev == _SEVERITY_NONE) | ~probs_ok
    forecast_ok = ~insufficient
    pred_crit = forecast_ok & (pred_sev == TIER_SEVERITY[CRITICAL])
    pred_elev = forecast_ok & (pred_sev == TIER_SEVERITY[ELEVATED])
    pred_safe = forecast_ok & (pred_sev == TIER_SEVERITY[SAFE])

    with np.errstate(invalid="ignore"):
        escalation_risk = forecast_ok & ~pred_crit & (p_crit >= t_watch)
        cf_disagree = forecast_ok & (cf_sev != _SEVERITY_NONE) & (np.abs(pred_sev - cf_sev) == 2)
        ambiguous = forecast_ok & set_ambiguous
        deesc_claim = forecast_ok & (cur_sev == TIER_SEVERITY[CRITICAL]) & ~pred_crit
        deesc_review = deesc_claim & (p_crit < deesc_p)
        low_conf_safe = pred_safe & (p_safe < t_clear)
        p_safe_ok = pred_safe & (p_safe >= t_clear)
    cur_not_safe = pred_safe & (cur_sev != TIER_SEVERITY[SAFE])
    quality_blocks = pred_safe & ~quality_ok

    # ---- precedence (first match wins) ----
    maneuver = pred_crit
    watch = escalation_risk & ~maneuver
    review4 = forecast_ok & ~maneuver & ~watch & (cf_disagree | ambiguous | deesc_review)
    monitor = pred_elev & ~maneuver & ~watch & ~review4
    safe_eligible = pred_safe & ~watch & ~review4 & (cur_sev == TIER_SEVERITY[SAFE]) & quality_ok & ~deesc_claim
    auto_clear = safe_eligible & p_safe_ok
    low_priority = safe_eligible & low_conf_safe

    action = np.full(n, TriageAction.HUMAN_REVIEW.value, dtype=object)
    action[low_priority] = TriageAction.LOW_PRIORITY_MONITOR.value
    action[auto_clear] = TriageAction.AUTO_CLEAR_CANDIDATE.value
    action[monitor] = TriageAction.MONITOR.value
    action[watch] = TriageAction.ESCALATION_WATCH.value
    action[maneuver] = TriageAction.MANEUVER_PLANNING_REVIEW.value
    action[insufficient] = TriageAction.HUMAN_REVIEW_INSUFFICIENT_DATA.value

    # ---- reason codes ----
    RC = ReasonCode
    cur_crit = cur_sev == TIER_SEVERITY[CRITICAL]
    cur_elev = cur_sev == TIER_SEVERITY[ELEVATED]
    miss_shrinking = trends["miss_shrinking"]
    risk_rising = trends["risk_rising"]
    sigma_growing = trends["sigma_growing"]
    high_volatility = trends["high_volatility"]

    reason_codes: list[list[str]] = []
    for i in range(n):
        codes: list[str] = []
        if insufficient[i]:
            codes.append(RC.INVALID_INPUT.value)
        if pred_crit[i]:
            codes.append(RC.FORECAST_CRITICAL.value)
        if pred_elev[i]:
            codes.append(RC.FORECAST_ELEVATED.value)
        if pred_safe[i]:
            codes.append(RC.FORECAST_SAFE.value)
        if escalation_risk[i]:
            codes.append(RC.ESCALATION_RISK.value)
        if cf_disagree[i]:
            codes.append(RC.MODEL_CARRY_FORWARD_DISAGREE.value)
        if ambiguous[i]:
            codes.append(RC.AMBIGUOUS_PREDICTION_SET.value)
        if deesc_claim[i]:
            codes.append(RC.DEESCALATION_CLAIM.value)
        if low_conf_safe[i]:
            codes.append(RC.LOW_CONFIDENCE_SAFE.value)
        if cur_not_safe[i]:
            codes.append(RC.CURRENT_TIER_NOT_SAFE.value)
        if quality_blocks[i]:
            codes.append(RC.QUALITY_WARNING.value)
            if has_missing_flag[i]:
                codes.append(RC.MISSING_TELEMETRY.value)
        # Descriptive codes, independent of the action.
        if cur_crit[i]:
            codes.append(RC.CURRENT_TIER_CRITICAL.value)
        if cur_elev[i]:
            codes.append(RC.CURRENT_TIER_ELEVATED.value)
        if no_history[i]:
            codes.append(RC.NO_HISTORY.value)
        if miss_shrinking[i]:
            codes.append(RC.MISS_SHRINKING.value)
        if risk_rising[i]:
            codes.append(RC.RISK_RISING.value)
        if sigma_growing[i]:
            codes.append(RC.SIGMA_GROWING.value)
        if high_volatility[i]:
            codes.append(RC.HIGH_VOLATILITY.value)
        reason_codes.append(_order_codes(codes))

    out["triage_action"] = pd.Series(action, index=out.index, dtype=object).astype(str)
    out["reason_codes"] = pd.Series(reason_codes, index=out.index, dtype=object)
    return out


# --------------------------------------------------------------------------- #
# Threshold tuning (dev OOF only)
# --------------------------------------------------------------------------- #
def _threshold_grid(values: np.ndarray, step: float = _THRESHOLD_GRID_STEP) -> np.ndarray:
    """Fine grid: `step` increments on [0, 1] plus the exact finite probability values."""
    # Round the regular grid so 0.94 from arange and 0.94 from the data coincide.
    grid = np.round(np.arange(0.0, 1.0 + step / 2, step), 10)
    finite = values[np.isfinite(values)]
    return np.unique(np.concatenate([grid, finite]).astype("float64"))


def tune_policy_thresholds(oof: pd.DataFrame, cfg: dict) -> dict:
    """Tune ``t_watch`` and ``t_clear`` on dev out-of-fold forecasts.

    ``oof`` holds one row per event with ground truth ``final_tier`` plus the forecast
    columns (``predicted_final_tier``, ``p_safe``, ``p_critical``, ``current_tier``,
    ``prediction_status``, ``quality_flags``).

    t_watch : the smallest p_critical threshold such that the watch list (events NOT
        predicted Critical with p_critical >= t_watch) holds at most
        ``cfg['triage']['watch_max_share']`` of all events.
    t_clear : among auto-clear-eligible events (predicted Safe, current tier Safe,
        acceptable quality) the smallest p_safe threshold such that ZERO events whose
        final tier is Critical satisfy p_safe >= t_clear. If no threshold in [0, 1]
        achieves this, t_clear = 1.01 (nothing clears) and the result says so.

    Returns a JSON-serialisable dict
    ``{"t_watch", "t_clear", "watch": {...}, "auto_clear": {...}}``.
    """
    triage_cfg = cfg["triage"]
    watch_max_share = float(triage_cfg["watch_max_share"])

    n_events = int(len(oof))
    p_safe, _, p_crit = _probability_arrays(oof)
    pred_sev = _severity_array(_column_or_default(oof, "predicted_final_tier", None).tolist())
    cur_sev = _severity_array(_column_or_default(oof, "current_tier", None).tolist())
    final_sev = _severity_array(_column_or_default(oof, "final_tier", None).tolist())
    quality_ok, _, _ = _quality_masks(oof, cfg)

    final_crit = final_sev == TIER_SEVERITY[CRITICAL]
    pred_crit = pred_sev == TIER_SEVERITY[CRITICAL]
    pred_safe = pred_sev == TIER_SEVERITY[SAFE]

    # ---- t_watch ----
    watch_pool = ~pred_crit & np.isfinite(p_crit)
    pool_p = np.where(watch_pool, p_crit, np.nan)
    n_escalations_total = int(np.sum(~pred_crit & final_crit))

    t_watch = 1.01
    watch_mask = np.zeros(n_events, dtype=bool)
    if n_events > 0:
        for t in _threshold_grid(pool_p):
            with np.errstate(invalid="ignore"):
                mask = watch_pool & (pool_p >= t)
            if mask.sum() / n_events <= watch_max_share:
                t_watch = float(t)
                watch_mask = mask
                break
    n_watch = int(watch_mask.sum())
    n_caught = int(np.sum(watch_mask & final_crit))
    watch_info = {
        "watch_max_share": watch_max_share,
        "watch_share": float(n_watch / n_events) if n_events else 0.0,
        "n_watch": n_watch,
        "n_escalations_caught": n_caught,
        "n_escalations_total": n_escalations_total,
        "recall": float(n_caught / n_escalations_total) if n_escalations_total else 0.0,
        "n_events": n_events,
    }

    # ---- t_clear ----
    eligible = pred_safe & (cur_sev == TIER_SEVERITY[SAFE]) & quality_ok & np.isfinite(p_safe)
    elig_p = np.where(eligible, p_safe, np.nan)
    t_clear = 1.01
    zero_achievable = False
    for t in _threshold_grid(elig_p):
        with np.errstate(invalid="ignore"):
            cleared = eligible & (elig_p >= t)
        if not np.any(cleared & final_crit):
            t_clear = float(t)
            zero_achievable = True
            break
    with np.errstate(invalid="ignore"):
        cleared = eligible & (elig_p >= t_clear)
    n_auto_clear = int(cleared.sum())
    n_crit_cleared = int(np.sum(cleared & final_crit))
    auto_clear_info = {
        "n_eligible": int(eligible.sum()),
        "n_eligible_final_critical": int(np.sum(eligible & final_crit)),
        "auto_clear_share": float(n_auto_clear / n_events) if n_events else 0.0,
        "n_auto_clear": n_auto_clear,
        "n_final_critical_cleared": n_crit_cleared,
        "zero_missed_achievable": bool(zero_achievable),
        "note": (
            "zero final-Critical cleared at t_clear"
            if zero_achievable
            else "no p_safe threshold in [0, 1] clears zero final-Critical events; "
            "t_clear = 1.01 disables auto-clear"
        ),
        "n_events": n_events,
    }

    return {
        "t_watch": float(t_watch),
        "t_clear": float(t_clear),
        "watch": watch_info,
        "auto_clear": auto_clear_info,
    }


__all__ = [
    "TIER_SEVERITY",
    "assign_risk_tier",
    "tier_from_true_risk",
    "assign_triage",
    "tune_policy_thresholds",
]
