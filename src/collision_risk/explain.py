"""Local feature contributions for the LightGBM point regressor.

Uses LightGBM's native ``pred_contrib=True`` (path-dependent TreeSHAP) so the
explanation is exact for the explained component: for every row

    baseline + sum(contributions) == booster.predict(X)[row]

in the model's output units, which for the point regressor is ``log10(Pc)``.
A contribution of +1 therefore corresponds to a factor of 10 on Pc.

Only the *regressor* is explained. Post-processing applied downstream (target
floor, gating, tier rules) is reported through ``final_output`` and
``postprocessing_note`` but is not decomposed.
"""
from __future__ import annotations

import json
import math
from typing import Any, Sequence

import numpy as np
import pandas as pd

from collision_risk.contracts import ExplanationRecord, FeatureContribution

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
    # event history
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
}


def feature_label(feature: str) -> str:
    """Readable label for a feature name, falling back to the raw name."""
    return FEATURE_LABELS.get(feature, feature)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _is_numeric_dtype(dtype: Any) -> bool:
    return pd.api.types.is_numeric_dtype(dtype) and not isinstance(dtype, pd.CategoricalDtype)


def _feature_value_and_label(series: pd.Series, i: int, feature: str) -> tuple[float | None, str]:
    """Return (numeric value or None, label) for row ``i`` of a feature column.

    Numeric columns (including bool) yield a float, or ``None`` when NaN.
    Categorical / string / object columns yield ``None`` and the category is
    appended to the label, e.g. ``"Object type = DEBRIS"``.
    """
    label = feature_label(feature)
    raw = series.iloc[i]
    if _is_numeric_dtype(series.dtype):
        if raw is None or (isinstance(raw, float) and math.isnan(raw)) or pd.isna(raw):
            return None, label
        return float(raw), label
    # categorical / string / object
    if pd.isna(raw):
        return None, f"{label} = missing"
    return None, f"{label} = {raw}"


def _booster_feature_names(booster: Any, X: pd.DataFrame) -> list[str]:
    names = None
    if hasattr(booster, "feature_name"):
        try:
            names = list(booster.feature_name())
        except Exception:  # pragma: no cover - defensive
            names = None
    if not names:
        names = [str(c) for c in X.columns]
    return names


def _contrib_matrix(booster: Any, X: pd.DataFrame) -> np.ndarray:
    """Shape (n, n_features + 1); last column is the baseline."""
    contrib = np.asarray(booster.predict(X, pred_contrib=True), dtype="float64")
    if contrib.ndim != 2:
        raise ValueError(
            "pred_contrib returned an unexpected shape; expected a 2-D array for a "
            f"regression booster, got {contrib.shape}"
        )
    if contrib.shape[0] != len(X):
        raise ValueError("pred_contrib row count does not match X")
    return contrib


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def compute_contributions(
    booster: Any,
    X: pd.DataFrame,
    *,
    row_ids: Sequence[str],
    dataset_id: str,
    component: str,
    model_version: str,
    final_outputs: np.ndarray | Sequence[float] | None = None,
    postprocessing_note: str | None = None,
    top_k: int = 5,
    explanation_id_prefix: str = "exp",
) -> list[ExplanationRecord]:
    """Build one :class:`ExplanationRecord` per row of ``X``.

    ``booster`` is a fitted ``lightgbm.Booster`` (or any object exposing
    ``predict(X, pred_contrib=True)`` with LightGBM semantics). ``X`` must have
    exactly the model's feature columns, in the model's order.
    """
    if len(row_ids) != len(X):
        raise ValueError("row_ids must have one entry per row of X")
    if top_k < 0:
        raise ValueError("top_k must be >= 0")
    if final_outputs is not None:
        fo = np.asarray(final_outputs, dtype="float64").ravel()
        if fo.shape[0] != len(X):
            raise ValueError("final_outputs must have one entry per row of X")
    else:
        fo = None

    if len(X) == 0:
        return []

    feature_names = _booster_feature_names(booster, X)
    missing = [f for f in feature_names if f not in X.columns]
    if missing:
        raise ValueError(f"X is missing model features: {missing}")
    X = X.loc[:, feature_names]

    contrib = _contrib_matrix(booster, X)
    n_features = len(feature_names)
    if contrib.shape[1] != n_features + 1:
        raise ValueError(
            f"pred_contrib returned {contrib.shape[1]} columns but the booster has "
            f"{n_features} features (+1 baseline expected)"
        )

    baseline_col = contrib[:, -1]
    feat_contrib = contrib[:, :-1]
    explained = baseline_col + feat_contrib.sum(axis=1)

    columns = {f: X[f] for f in feature_names}
    records: list[ExplanationRecord] = []
    for i in range(len(X)):
        row_c = feat_contrib[i]
        pos_idx = [j for j in np.argsort(-row_c, kind="stable") if row_c[j] > 0][:top_k]
        neg_idx = [j for j in np.argsort(row_c, kind="stable") if row_c[j] < 0][:top_k]
        shown = set(pos_idx) | set(neg_idx)
        remaining = float(sum(row_c[j] for j in range(n_features) if j not in shown))

        def _fc(j: int) -> FeatureContribution:
            f = feature_names[j]
            value, label = _feature_value_and_label(columns[f], i, f)
            return FeatureContribution(
                feature=f, label=label, value=value, contribution=float(row_c[j])
            )

        final_output: float | None = None
        if fo is not None and not np.isnan(fo[i]):
            final_output = float(fo[i])

        rec = ExplanationRecord(
            explanation_id=f"{explanation_id_prefix}:{dataset_id}:{row_ids[i]}",
            dataset_id=dataset_id,
            row_id=str(row_ids[i]),
            component=component,
            output_units="log10(Pc)",
            baseline=float(baseline_col[i]),
            explained_output=float(explained[i]),
            final_output=final_output,
            postprocessing_note=postprocessing_note,
            top_positive=[_fc(j) for j in pos_idx],
            top_negative=[_fc(j) for j in neg_idx],
            remaining_contribution=remaining,
            text="",
            model_version=model_version,
        )
        rec.text = render_text(rec)
        records.append(rec)
    return records


def _fmt(x: float | None, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:+.{nd}f}" if nd else f"{x:.0f}"


def _fmt_plain(x: float | None, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def _driver_phrase(fc: FeatureContribution) -> str:
    if fc.value is not None:
        return f"{fc.label} = {_fmt_plain(fc.value, 3)} ({_fmt(fc.contribution)})"
    return f"{fc.label} ({_fmt(fc.contribution)})"


def render_text(record: ExplanationRecord, max_chars: int = 600) -> str:
    """Deterministic, non-causal textual rendering of an explanation.

    The wording is deliberately associational ("is associated with") -- the
    contributions describe how the model's output moves, not why the
    conjunction is risky.
    """
    units = record.output_units
    parts: list[str] = [
        f"Component {record.component}: baseline {_fmt_plain(record.baseline)} {units}, "
        f"model output {_fmt_plain(record.explained_output)} {units} "
        f"(+1 = factor of 10 in Pc)."
    ]
    if record.top_positive:
        parts.append(
            "Higher risk is associated with: "
            + "; ".join(_driver_phrase(fc) for fc in record.top_positive)
            + "."
        )
    if record.top_negative:
        parts.append(
            "Lower risk is associated with: "
            + "; ".join(_driver_phrase(fc) for fc in record.top_negative)
            + "."
        )
    if not record.top_positive and not record.top_negative:
        parts.append("No feature is associated with a shift from the baseline.")
    if abs(record.remaining_contribution) > 1e-9:
        parts.append(f"Other features: {_fmt(record.remaining_contribution)}.")
    if record.final_output is not None and not math.isclose(
        record.final_output, record.explained_output, rel_tol=1e-9, abs_tol=1e-9
    ):
        note = record.postprocessing_note or "post-processing applied"
        parts.append(f"Final output {_fmt_plain(record.final_output)} {units} ({note}).")

    text = " ".join(parts)
    if len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    return text


# --------------------------------------------------------------------------- #
# flat (parquet-friendly) representation
# --------------------------------------------------------------------------- #
_SCALAR_FIELDS: list[str] = [
    "explanation_id",
    "dataset_id",
    "row_id",
    "component",
    "output_units",
    "baseline",
    "explained_output",
    "final_output",
    "postprocessing_note",
    "remaining_contribution",
    "text",
    "model_version",
]
_LIST_FIELDS: list[str] = ["top_positive", "top_negative"]
EXPLANATION_COLUMNS: list[str] = _SCALAR_FIELDS + _LIST_FIELDS


def _dump_contribs(items: list[FeatureContribution]) -> str:
    return json.dumps([fc.model_dump() for fc in items], ensure_ascii=False)


def explanations_to_frame(records: list[ExplanationRecord]) -> pd.DataFrame:
    """Flatten records: scalar fields as columns, contribution lists as JSON."""
    rows: list[dict[str, Any]] = []
    for r in records:
        d = {f: getattr(r, f) for f in _SCALAR_FIELDS}
        d["top_positive"] = _dump_contribs(r.top_positive)
        d["top_negative"] = _dump_contribs(r.top_negative)
        rows.append(d)
    df = pd.DataFrame(rows, columns=EXPLANATION_COLUMNS)
    # keep nullable scalars as object-free floats / strings for parquet
    for c in ("baseline", "explained_output", "remaining_contribution", "final_output"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    return df


def _none_if_na(v: Any) -> Any:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def frame_to_explanations(df: pd.DataFrame) -> list[ExplanationRecord]:
    """Inverse of :func:`explanations_to_frame`."""
    out: list[ExplanationRecord] = []
    for row in df.to_dict(orient="records"):
        payload: dict[str, Any] = {}
        for f in _SCALAR_FIELDS:
            payload[f] = _none_if_na(row.get(f))
        for f in _LIST_FIELDS:
            raw = _none_if_na(row.get(f))
            items = json.loads(raw) if isinstance(raw, str) else (raw or [])
            payload[f] = [FeatureContribution.model_validate(it) for it in items]
        if payload.get("output_units") is None:
            payload["output_units"] = "log10(Pc)"
        out.append(ExplanationRecord.model_validate(payload))
    return out


__all__ = [
    "FEATURE_LABELS",
    "EXPLANATION_COLUMNS",
    "feature_label",
    "compute_contributions",
    "render_text",
    "explanations_to_frame",
    "frame_to_explanations",
]
