"""Local feature contributions for the event-level final-tier forecaster.

The model forecasts each event's *final* tier (Safe / Elevated / Critical)
with class probabilities. We explain the **Critical-class output** using
LightGBM's native ``pred_contrib=True`` (path-dependent TreeSHAP), so the
explanation is exact for the explained component: for every row

    baseline + sum(contributions) == booster.predict(X, raw_score=True)[row, k]

where ``k`` is the Critical class index (multiclass booster) or the single
binary head (ordinal model's P(Critical) head). Units are the booster's raw
score, i.e. log-odds-like softmax logits for multiclass and log-odds for a
binary booster. The reported probability is calibrated downstream and is
*not* decomposed here.

Wording is deliberately associational ("is associated with"): contributions
describe how the model's score moves with a feature, not why the conjunction
is risky. Causal wording ("cause", "because", "leads to") is never used.

Everything is vectorised with numpy along axis 1; the only per-row Python
work is the final dict / JSON / string formatting.
"""
from __future__ import annotations

import json
import math
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel

# --------------------------------------------------------------------------- #
# readable labels
# --------------------------------------------------------------------------- #
FEATURE_LABELS: dict[str, str] = {
    # base CDM fields
    "time_to_tca": "Time to TCA (days)",
    "miss_distance": "Miss distance (m)",
    "relative_speed": "Relative speed (m/s)",
    "relative_position_r": "Relative position, radial (m)",
    "relative_position_t": "Relative position, along-track (m)",
    "relative_position_n": "Relative position, cross-track (m)",
    "c_sigma_r": "Chaser radial sigma (m)",
    "t_sigma_r": "Target radial sigma (m)",
    "max_risk_estimate": "Max risk estimate (log10 Pc)",
    "F10": "Solar flux F10.7",
    "AP": "Geomagnetic index Ap",
    "c_object_type": "Object type",
    "risk": "Risk of this CDM (log10 Pc)",
    # derived
    "log_miss_distance": "Miss distance (log10 m)",
    "combined_sigma_r": "Combined radial sigma (m)",
    "log_combined_sigma_r": "Combined radial sigma (log10 m)",
    "radial_sep_ratio": "Radial separation / combined sigma",
    "sigma_ratio": "Chaser / target radial sigma ratio",
    "log_c_sigma_r": "Chaser radial sigma (log10 m)",
    "log_t_sigma_r": "Target radial sigma (log10 m)",
    "frac_r": "Radial fraction of miss distance",
    "frac_t": "Along-track fraction of miss distance",
    "frac_n": "Cross-track fraction of miss distance",
    "log_relative_speed": "Relative speed (log10 m/s)",
    "missing_c_sigma_r": "Chaser sigma missing (flag)",
    "missing_space_weather": "Space weather missing (flag)",
    "missing_max_risk": "Max risk estimate missing (flag)",
    "sigma_suspicious": "Suspicious sigma placeholder (flag)",
    "log_miss_minus_1km": "log10 miss distance minus log10(1000 m)",
    "log_miss_minus_5km": "log10 miss distance minus log10(5000 m)",
    "latest_risk_is_floor": "Latest risk is the -30 floor",
    # event history (per-CDM sequence features)
    "n_cdms_so_far": "CDMs seen so far",
    "days_since_first_cdm": "Days since first CDM",
    "delta_log_miss_distance": "Change in miss distance (log10 m) vs previous CDM",
    "delta_max_risk": "Change in max risk estimate vs previous CDM",
    "delta_log_combined_sigma": "Change in combined sigma (log10 m) vs previous CDM",
    "run_min_log_miss": "Running min miss distance (log10 m)",
    "run_max_max_risk": "Running max of max risk estimate (log10 Pc)",
    "run_min_max_risk": "Running min of max risk estimate (log10 Pc)",
    "mre_trend": "Max risk estimate trend (per day)",
    "has_history": "Event history available (flag)",
    # event-level forecast features
    "n_cdms_2d": "CDMs at >= 2 days so far",
    "span_days": "Days covered by CDMs so far",
    "d_log_miss_first": "Change in log10 miss distance since first CDM",
    "d_mre_first": "Change in max risk estimate since first CDM",
    "d_log_sigma_first": "Change in log10 combined sigma since first CDM",
    "d_risk_first": "Change in risk since first CDM",
    "slope_log_miss": "Trend of log10 miss distance (per day to TCA)",
    "slope_mre": "Trend of max risk estimate (per day to TCA)",
    "slope_log_sigma": "Trend of log10 combined sigma (per day to TCA)",
    "slope_risk": "Trend of risk (per day to TCA)",
    "volatility_mre": "Volatility of max risk estimate",
    "volatility_risk": "Volatility of risk",
    "n_up_risk": "Up-moves in risk",
    "n_down_risk": "Down-moves in risk",
    "n_up_mre": "Up-moves in max risk estimate",
    "n_down_mre": "Down-moves in max risk estimate",
    "tier_now_from_risk": "Current tier from risk (0 Safe, 1 Elevated, 2 Critical)",
    "tier_now_from_mre": "Current tier from max risk estimate",
    "margin_to_critical": "Margin to Critical (risk + 4)",
    "margin_to_elevated": "Margin to Elevated (risk + 6)",
}

GROUP_NAMES: tuple[str, ...] = (
    "geometry",
    "uncertainty",
    "risk_trend",
    "risk_level",
    "space_weather",
    "object_type",
    "history",
    "other",
)

FEATURE_GROUPS: dict[str, str] = {
    # geometry (including how the geometry evolves over the event)
    "time_to_tca": "geometry",
    "miss_distance": "geometry",
    "relative_speed": "geometry",
    "relative_position_r": "geometry",
    "relative_position_t": "geometry",
    "relative_position_n": "geometry",
    "log_miss_distance": "geometry",
    "frac_r": "geometry",
    "frac_t": "geometry",
    "frac_n": "geometry",
    "log_relative_speed": "geometry",
    "log_miss_minus_1km": "geometry",
    "log_miss_minus_5km": "geometry",
    "delta_log_miss_distance": "geometry",
    "run_min_log_miss": "geometry",
    "d_log_miss_first": "geometry",
    "slope_log_miss": "geometry",
    # covariance / uncertainty
    "c_sigma_r": "uncertainty",
    "t_sigma_r": "uncertainty",
    "combined_sigma_r": "uncertainty",
    "log_combined_sigma_r": "uncertainty",
    "radial_sep_ratio": "uncertainty",
    "sigma_ratio": "uncertainty",
    "log_c_sigma_r": "uncertainty",
    "log_t_sigma_r": "uncertainty",
    "missing_c_sigma_r": "uncertainty",
    "sigma_suspicious": "uncertainty",
    "delta_log_combined_sigma": "uncertainty",
    "d_log_sigma_first": "uncertainty",
    "slope_log_sigma": "uncertainty",
    # how the risk numbers have been moving
    "delta_max_risk": "risk_trend",
    "run_max_max_risk": "risk_trend",
    "run_min_max_risk": "risk_trend",
    "mre_trend": "risk_trend",
    "d_mre_first": "risk_trend",
    "d_risk_first": "risk_trend",
    "slope_mre": "risk_trend",
    "slope_risk": "risk_trend",
    "volatility_mre": "risk_trend",
    "volatility_risk": "risk_trend",
    "n_up_risk": "risk_trend",
    "n_down_risk": "risk_trend",
    "n_up_mre": "risk_trend",
    "n_down_mre": "risk_trend",
    # where the risk numbers currently sit
    "max_risk_estimate": "risk_level",
    "risk": "risk_level",
    "missing_max_risk": "risk_level",
    "latest_risk_is_floor": "risk_level",
    "tier_now_from_risk": "risk_level",
    "tier_now_from_mre": "risk_level",
    "margin_to_critical": "risk_level",
    "margin_to_elevated": "risk_level",
    # environment
    "F10": "space_weather",
    "AP": "space_weather",
    "missing_space_weather": "space_weather",
    # object
    "c_object_type": "object_type",
    # how much of the event we have seen
    "n_cdms_so_far": "history",
    "days_since_first_cdm": "history",
    "has_history": "history",
    "n_cdms_2d": "history",
    "span_days": "history",
}


def feature_label(feature: str) -> str:
    """Readable label for a feature name, falling back to the raw name."""
    return FEATURE_LABELS.get(feature, feature)


def feature_group(feature: str) -> str:
    """Feature group name, ``"other"`` for unknown features."""
    return FEATURE_GROUPS.get(feature, "other")


# --------------------------------------------------------------------------- #
# contracts for the forecast explanation (kept local to this module)
# --------------------------------------------------------------------------- #
class ForecastFeatureContribution(BaseModel):
    feature: str
    label: str
    value: Optional[float]
    contribution: float  # raw-score (log-odds) units of the explained output
    group: str


class ForecastExplanationRecord(BaseModel):
    """Local contributions to the Critical-class raw score for one forecast row.

    Explains a single model component (the booster's Critical logit), not the
    calibrated probability nor the downstream policy."""

    explanation_id: str
    dataset_id: str
    row_id: str
    component: str
    output_units: str
    baseline: float
    explained_output: float
    p_critical: Optional[float]
    predicted_tier: Optional[str]
    top_positive: list[ForecastFeatureContribution]
    top_negative: list[ForecastFeatureContribution]
    remaining_contribution: float
    group_contributions: dict[str, float]
    text: str
    model_version: str


_JSON_COLUMNS: tuple[str, ...] = ("top_positive", "top_negative", "group_contributions")

EXPLANATION_COLUMNS: list[str] = [
    "explanation_id",
    "dataset_id",
    "row_id",
    "component",
    "output_units",
    "baseline",
    "explained_output",
    "p_critical",
    "predicted_tier",
    "top_positive",
    "top_negative",
    "remaining_contribution",
    "group_contributions",
    "text",
    "model_version",
]

UNITS_MULTICLASS = "log-odds (Critical class raw score)"
UNITS_BINARY = "log-odds"


# --------------------------------------------------------------------------- #
# contributions
# --------------------------------------------------------------------------- #
def _booster_feature_names(booster: Any, X: pd.DataFrame) -> list[str]:
    names: list[str] | None = None
    if hasattr(booster, "feature_name"):
        try:
            names = [str(n) for n in booster.feature_name()]
        except Exception:  # pragma: no cover - defensive
            names = None
    if not names:
        names = [str(c) for c in X.columns]
    return names


def critical_class_contributions(
    booster: Any,
    X: pd.DataFrame,
    *,
    class_index: int | None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Per-feature contributions to the Critical-class raw score.

    Returns ``(contribs, bias, feature_names)`` where ``contribs`` has shape
    ``(n, F)`` (float64), ``bias`` has shape ``(n,)`` and ``feature_names`` is
    the booster's feature order. ``class_index`` is ``None`` for a binary
    booster, else the index of the Critical class in a multiclass booster.

    ``X`` is reordered to ``booster.feature_name()``; a ``ValueError`` is
    raised when features are missing or ``pred_contrib`` has an unexpected
    shape.
    """
    if class_index is not None and class_index < 0:
        raise ValueError("class_index must be None or a non-negative integer")

    feature_names = _booster_feature_names(booster, X)
    missing = [f for f in feature_names if f not in X.columns]
    if missing:
        raise ValueError(f"X is missing model features: {missing}")
    X = X.loc[:, feature_names]
    n_features = len(feature_names)
    block_width = n_features + 1

    if len(X) == 0:
        return (
            np.zeros((0, n_features), dtype="float64"),
            np.zeros(0, dtype="float64"),
            feature_names,
        )

    contrib = np.asarray(booster.predict(X, pred_contrib=True), dtype="float64")
    if contrib.ndim != 2:
        raise ValueError(
            f"pred_contrib returned an unexpected shape; expected 2-D, got {contrib.shape}"
        )
    if contrib.shape[0] != len(X):
        raise ValueError("pred_contrib row count does not match X")

    if class_index is None:
        if contrib.shape[1] != block_width:
            raise ValueError(
                f"pred_contrib returned {contrib.shape[1]} columns but a binary booster with "
                f"{n_features} features should return {block_width} (features + bias). "
                "For a multiclass booster pass class_index."
            )
        block = contrib
    else:
        if contrib.shape[1] % block_width != 0:
            raise ValueError(
                f"pred_contrib returned {contrib.shape[1]} columns, not a multiple of "
                f"{block_width} (features + bias) for {n_features} features"
            )
        n_classes = contrib.shape[1] // block_width
        if class_index >= n_classes:
            raise ValueError(
                f"class_index {class_index} out of range for a booster with {n_classes} classes"
            )
        start = class_index * block_width
        block = contrib[:, start : start + block_width]

    contribs = np.ascontiguousarray(block[:, :-1], dtype="float64")
    bias = np.ascontiguousarray(block[:, -1], dtype="float64")
    return contribs, bias, feature_names


# --------------------------------------------------------------------------- #
# feature values (vectorised)
# --------------------------------------------------------------------------- #
def _is_plain_numeric(dtype: Any) -> bool:
    return pd.api.types.is_numeric_dtype(dtype) and not isinstance(dtype, pd.CategoricalDtype)


def _value_matrix(X: pd.DataFrame, feature_names: list[str]) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Return ``(values (n,F) float64 with NaN for non-numeric/missing,
    {feature index: per-row category string array})`` for categorical/object columns."""
    n = len(X)
    values = np.full((n, len(feature_names)), np.nan, dtype="float64")
    cat_strings: dict[int, np.ndarray] = {}
    for j, f in enumerate(feature_names):
        col = X[f]
        if _is_plain_numeric(col.dtype):
            values[:, j] = pd.to_numeric(col, errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
        else:
            s = col.astype("object")
            isna = s.isna().to_numpy()
            arr = s.astype(str).to_numpy(dtype=object)
            arr[isna] = "missing"
            cat_strings[j] = arr
    return values, cat_strings


# --------------------------------------------------------------------------- #
# text rendering
# --------------------------------------------------------------------------- #
_UNITS_SENTENCE = "Contributions in log-odds; the probability is calibrated separately."


def _is_missing(x: Any) -> bool:
    if x is None:
        return True
    try:
        return bool(math.isnan(float(x)))
    except (TypeError, ValueError):
        return False


def _fmt_signed(x: Any, nd: int = 2) -> str:
    return "n/a" if _is_missing(x) else f"{float(x):+.{nd}f}"


def _fmt_plain(x: Any, nd: int = 2) -> str:
    return "n/a" if _is_missing(x) else f"{float(x):.{nd}f}"


def _driver_phrase(fc: dict) -> str:
    label = fc.get("label", fc.get("feature", "?"))
    value = fc.get("value")
    contribution = fc.get("contribution")
    if value is not None and not _is_missing(value):
        return f"{label} = {_fmt_plain(value, 3)} ({_fmt_signed(contribution)})"
    return f"{label} ({_fmt_signed(contribution)})"


def render_forecast_text(row_like: dict, max_chars: int = 600) -> str:
    """Deterministic, non-causal textual rendering of a forecast explanation.

    ``row_like`` is a dict with ``predicted_tier``, ``p_critical``,
    ``top_positive`` / ``top_negative`` (lists of dicts or JSON strings) and
    ``remaining_contribution``. Wording is associational: the drivers are
    features associated with a higher or lower Critical score, never reasons.
    """
    tier = row_like.get("predicted_tier")
    tier_s = "n/a" if tier is None or (isinstance(tier, float) and math.isnan(tier)) else str(tier)
    p = row_like.get("p_critical")
    head = f"Forecast {tier_s} (P = {_fmt_plain(p)})."

    pos = row_like.get("top_positive") or []
    neg = row_like.get("top_negative") or []
    if isinstance(pos, str):
        pos = json.loads(pos)
    if isinstance(neg, str):
        neg = json.loads(neg)

    parts: list[str] = [head]
    if pos:
        parts.append(
            "Drivers associated with a higher Critical score: "
            + "; ".join(_driver_phrase(fc) for fc in pos)
            + "."
        )
    if neg:
        parts.append(
            "Drivers associated with a lower Critical score: "
            + "; ".join(_driver_phrase(fc) for fc in neg)
            + "."
        )
    if not pos and not neg:
        parts.append("No feature is associated with a shift from the baseline.")
    remaining = row_like.get("remaining_contribution")
    if not _is_missing(remaining) and abs(float(remaining)) > 1e-9:
        parts.append(f"Other features: {_fmt_signed(remaining)}.")
    parts.append(_UNITS_SENTENCE)

    text = " ".join(parts)
    if len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    return text


# --------------------------------------------------------------------------- #
# main entry point
# --------------------------------------------------------------------------- #
def _float_or_none(x: float) -> float | None:
    return None if math.isnan(x) else x


def explain_forecasts(
    booster: Any,
    X: pd.DataFrame,
    *,
    class_index: int | None,
    row_ids: Sequence[str] | np.ndarray,
    dataset_id: str | Sequence[str],
    model_version: str,
    p_critical: np.ndarray | Sequence[float],
    predicted_tier: Sequence[str] | np.ndarray,
    top_k: int = 5,
    component: str = "lgbm_multiclass_critical_logit",
    explanation_id_prefix: str = "exp",
) -> pd.DataFrame:
    """Explain the Critical-class raw score for every row of ``X``.

    Returns one row per input row with columns :data:`EXPLANATION_COLUMNS`.
    ``top_positive`` / ``top_negative`` / ``group_contributions`` are JSON
    strings (parquet friendly); decode with :func:`frame_to_records`.

    Invariants (per row): ``baseline + sum(all contributions) == explained_output``,
    ``sum(top_positive) + sum(top_negative) + remaining_contribution ==
    explained_output - baseline`` and ``sum(group_contributions.values()) ==
    explained_output - baseline``.
    """
    n = len(X)
    row_ids_l = [str(r) for r in row_ids]
    if len(row_ids_l) != n:
        raise ValueError("row_ids must have one entry per row of X")
    if top_k < 0:
        raise ValueError("top_k must be >= 0")
    p_arr = np.asarray(p_critical, dtype="float64").ravel()
    if p_arr.shape[0] != n:
        raise ValueError("p_critical must have one entry per row of X")
    tier_l = [None if (t is None or (isinstance(t, float) and math.isnan(t))) else str(t) for t in predicted_tier]
    if len(tier_l) != n:
        raise ValueError("predicted_tier must have one entry per row of X")
    # One dataset id per row (a batch may mix datasets); a scalar applies to every row.
    if isinstance(dataset_id, str):
        dataset_l: list[str] = [dataset_id] * n
    else:
        dataset_l = [str(d) for d in dataset_id]
        if len(dataset_l) != n:
            raise ValueError("dataset_id must be a string or have one entry per row of X")

    contribs, bias, feature_names = critical_class_contributions(booster, X, class_index=class_index)
    output_units = UNITS_BINARY if class_index is None else UNITS_MULTICLASS
    if n == 0:
        return pd.DataFrame(columns=EXPLANATION_COLUMNS)

    X = X.loc[:, feature_names]
    n_features = len(feature_names)
    total = contribs.sum(axis=1)
    explained = bias + total

    # ---- top-k selection, vectorised along axis 1 ----------------------- #
    k = min(top_k, n_features)
    if k > 0:
        order_desc = np.argsort(-contribs, axis=1, kind="stable")[:, :k]
        pos_vals = np.take_along_axis(contribs, order_desc, axis=1)
        pos_mask = pos_vals > 0
        order_asc = np.argsort(contribs, axis=1, kind="stable")[:, :k]
        neg_vals = np.take_along_axis(contribs, order_asc, axis=1)
        neg_mask = neg_vals < 0
    else:
        order_desc = np.zeros((n, 0), dtype=np.intp)
        order_asc = np.zeros((n, 0), dtype=np.intp)
        pos_vals = neg_vals = np.zeros((n, 0), dtype="float64")
        pos_mask = neg_mask = np.zeros((n, 0), dtype=bool)
    shown = np.where(pos_mask, pos_vals, 0.0).sum(axis=1) + np.where(neg_mask, neg_vals, 0.0).sum(axis=1)
    remaining = total - shown

    # ---- group sums: one matmul ---------------------------------------- #
    groups = [feature_group(f) for f in feature_names]
    group_names = [g for g in GROUP_NAMES if g in set(groups)]
    g_index = {g: i for i, g in enumerate(group_names)}
    onehot = np.zeros((n_features, len(group_names)), dtype="float64")
    for j, g in enumerate(groups):
        onehot[j, g_index[g]] = 1.0
    group_sums = contribs @ onehot  # (n, G)

    # ---- feature values at the selected positions ----------------------- #
    values, cat_strings = _value_matrix(X, feature_names)
    pos_feat_vals = np.take_along_axis(values, order_desc, axis=1)
    neg_feat_vals = np.take_along_axis(values, order_asc, axis=1)

    labels = [feature_label(f) for f in feature_names]

    # python-level views (lists) for the formatting pass
    pos_idx_l = order_desc.tolist()
    neg_idx_l = order_asc.tolist()
    pos_vals_l = pos_vals.tolist()
    neg_vals_l = neg_vals.tolist()
    pos_mask_l = pos_mask.tolist()
    neg_mask_l = neg_mask.tolist()
    pos_fv_l = pos_feat_vals.tolist()
    neg_fv_l = neg_feat_vals.tolist()
    group_sums_l = group_sums.tolist()
    bias_l = bias.tolist()
    explained_l = explained.tolist()
    remaining_l = remaining.tolist()
    p_l = p_arr.tolist()

    def _entries(i: int, idx: list[int], vals: list[float], mask: list[bool], fv: list[float]) -> list[dict]:
        out: list[dict] = []
        for j, c, m, v in zip(idx, vals, mask, fv):
            if not m:
                break  # sorted, so the first failing entry ends the run
            if j in cat_strings:
                out.append(
                    {
                        "feature": feature_names[j],
                        "label": f"{labels[j]} = {cat_strings[j][i]}",
                        "value": None,
                        "contribution": c,
                        "group": groups[j],
                    }
                )
            else:
                out.append(
                    {
                        "feature": feature_names[j],
                        "label": labels[j],
                        "value": _float_or_none(v),
                        "contribution": c,
                        "group": groups[j],
                    }
                )
        return out

    top_pos = [_entries(i, pos_idx_l[i], pos_vals_l[i], pos_mask_l[i], pos_fv_l[i]) for i in range(n)]
    top_neg = [_entries(i, neg_idx_l[i], neg_vals_l[i], neg_mask_l[i], neg_fv_l[i]) for i in range(n)]
    group_dicts = [dict(zip(group_names, gs)) for gs in group_sums_l]
    p_out = [_float_or_none(p) for p in p_l]

    texts = [
        render_forecast_text(
            {
                "predicted_tier": tier_l[i],
                "p_critical": p_out[i],
                "top_positive": top_pos[i],
                "top_negative": top_neg[i],
                "remaining_contribution": remaining_l[i],
            }
        )
        for i in range(n)
    ]

    df = pd.DataFrame(
        {
            "explanation_id": [f"{explanation_id_prefix}:{r}" for r in row_ids_l],
            "dataset_id": dataset_l,
            "row_id": row_ids_l,
            "component": component,
            "output_units": output_units,
            "baseline": np.asarray(bias_l, dtype="float64"),
            "explained_output": np.asarray(explained_l, dtype="float64"),
            "p_critical": p_arr,
            "predicted_tier": pd.Series(tier_l, dtype="object"),
            "top_positive": [json.dumps(tp, ensure_ascii=False) for tp in top_pos],
            "top_negative": [json.dumps(tn, ensure_ascii=False) for tn in top_neg],
            "remaining_contribution": np.asarray(remaining_l, dtype="float64"),
            "group_contributions": [json.dumps(gd, ensure_ascii=False) for gd in group_dicts],
            "text": texts,
            "model_version": model_version,
        },
        columns=EXPLANATION_COLUMNS,
    )
    return df


# --------------------------------------------------------------------------- #
# frame <-> records
# --------------------------------------------------------------------------- #
def _none_if_na(v: Any) -> Any:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def frame_to_records(df: pd.DataFrame) -> list[dict]:
    """Decode the JSON columns of an :func:`explain_forecasts` frame into dicts.

    Missing scalars become ``None``. Used by ``service.get_explanation``; each
    dict validates against :class:`ForecastExplanationRecord`.
    """
    out: list[dict] = []
    for row in df.to_dict(orient="records"):
        rec: dict[str, Any] = {}
        for key, v in row.items():
            if key in _JSON_COLUMNS:
                raw = _none_if_na(v)
                if isinstance(raw, str):
                    rec[key] = json.loads(raw)
                elif raw is None:
                    rec[key] = {} if key == "group_contributions" else []
                else:
                    rec[key] = raw
            else:
                rec[key] = _none_if_na(v)
        out.append(rec)
    return out


__all__ = [
    "FEATURE_LABELS",
    "FEATURE_GROUPS",
    "GROUP_NAMES",
    "EXPLANATION_COLUMNS",
    "UNITS_MULTICLASS",
    "UNITS_BINARY",
    "ForecastFeatureContribution",
    "ForecastExplanationRecord",
    "feature_label",
    "feature_group",
    "critical_class_contributions",
    "explain_forecasts",
    "render_forecast_text",
    "frame_to_records",
]
