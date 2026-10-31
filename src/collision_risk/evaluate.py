"""Evaluation report for collision-risk triage predictions.

Consumes a prediction frame that already carries true and predicted tiers
(tier assignment lives in policy.py and is deliberately not imported here) and
produces a JSON-serialisable metrics dict across four views:

- row_level:           every CDM row
- latest_cdm:          one row per event, the CDM closest to TCA
- horizon_slice:       rows with time_to_tca_days >= cfg['split']['test_horizon_min_days']
- horizon_latest_cdm:  latest_cdm_view of the horizon slice

Conventions
- risk values are log10(Pc); the floor value (cfg['model']['target_floor'],
  default -30) marks "no meaningful probability".
- Any metric whose denominator is zero is reported as None and listed under the
  group's 'undefined' key. Every group dict always carries an 'undefined' list.
- Confusion matrices use labels ['Critical', 'Elevated', 'Safe', 'Invalid'];
  None / NaN tiers (true or predicted) map to 'Invalid'.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score

from .data import event_group_codes

TIER_LABELS: list[str] = ["Critical", "Elevated", "Safe"]
INVALID_LABEL = "Invalid"
CONFUSION_LABELS: list[str] = TIER_LABELS + [INVALID_LABEL]

AUTO_CLEAR_ACTION = "Auto-clear candidate"
MANEUVER_REVIEW_ACTION = "Maneuver planning review"
HUMAN_REVIEW_ACTION = "Human review"
HUMAN_REVIEW_INSUFFICIENT_ACTION = "Human review - insufficient data"
HUMAN_REVIEW_ACTIONS = (HUMAN_REVIEW_ACTION, HUMAN_REVIEW_INSUFFICIENT_ACTION)

DEFAULT_CRITICAL_RISK = -4.0
DEFAULT_ELEVATED_RISK = -6.0
DEFAULT_TARGET_FLOOR = -30.0
DEFAULT_COVERAGE = 0.90

BOOTSTRAP_METRICS = ("critical_recall", "critical_precision", "mae", "coverage")


# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #
def _is_missing(x: Any) -> bool:
    """True for None, NaN, pd.NA and the like."""
    if x is None:
        return True
    try:
        return bool(pd.isna(x))
    except (TypeError, ValueError):
        return False


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy / pandas scalars to python types and NaN -> None."""
    if obj is None:
        return None
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, (str, np.str_)):
        return str(obj)
    if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return str(obj)
    if obj is pd.NaT or obj is pd.NA:
        return None
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, pd.Series):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, pd.DataFrame):
        return [to_jsonable(r) for r in obj.to_dict(orient="records")]
    if hasattr(obj, "item"):
        try:
            return to_jsonable(obj.item())
        except (TypeError, ValueError):
            pass
    if hasattr(obj, "value") and not isinstance(obj, type):  # Enum members
        return to_jsonable(obj.value)
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return obj


def _safe_div(num: float, den: float) -> Optional[float]:
    if den is None or den == 0 or (isinstance(den, float) and math.isnan(den)):
        return None
    return float(num) / float(den)


def _f_beta(precision: Optional[float], recall: Optional[float], beta: float = 2.0) -> Optional[float]:
    """F-beta from precision/recall. Undefined when recall is undefined (no true
    positives in the truth); if precision is undefined (no predicted positives)
    recall is zero, so F-beta is zero (sklearn zero_division=0 convention)."""
    if recall is None:
        return None
    if precision is None:
        return 0.0
    b2 = beta * beta
    den = b2 * precision + recall
    if den == 0:
        return 0.0
    return float((1.0 + b2) * precision * recall / den)


def _finalise_group(group: dict, exclude: Iterable[str] = ()) -> dict:
    """Attach the 'undefined' list: every top-level scalar key that is None."""
    excl = set(exclude) | {"undefined"}
    undefined = [k for k, v in group.items() if k not in excl and v is None]
    group["undefined"] = undefined
    return group


def _tier_array(values: Sequence | pd.Series | np.ndarray) -> np.ndarray:
    """Normalise a tier column to an object array of strings in CONFUSION_LABELS."""
    out = []
    for v in list(values):
        if _is_missing(v):
            out.append(INVALID_LABEL)
            continue
        s = v.value if hasattr(v, "value") else v
        s = str(s)
        out.append(s if s in TIER_LABELS else INVALID_LABEL)
    return np.asarray(out, dtype=object)


def _get(cfg: dict, path: Sequence[str], default: Any) -> Any:
    cur: Any = cfg or {}
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _float_col(df: pd.DataFrame, col: str) -> np.ndarray:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    return np.full(len(df), np.nan, dtype=float)


def _str_col(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), None, dtype=object)
    out = []
    for v in df[col].tolist():
        if _is_missing(v):
            out.append(None)
        else:
            out.append(str(v.value if hasattr(v, "value") else v))
    return np.asarray(out, dtype=object)


# --------------------------------------------------------------------------- #
# Views
# --------------------------------------------------------------------------- #
def latest_cdm_view(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (dataset_id, event_id), or event_id if dataset_id is absent.

    Select the row with the smallest time_to_tca_days.

    Ties are broken by the last occurrence in input order. All columns are
    preserved; the original index is kept. Rows whose time_to_tca_days is NaN
    only win when every row of the event is NaN.
    """
    if len(df) == 0:
        return df.copy()
    order = np.arange(len(df))
    ttca = pd.to_numeric(df["time_to_tca_days"], errors="coerce").to_numpy(dtype=float)
    # NaN sorts last so valid rows are preferred; stable sort keeps input order
    # among ties, so taking the last row per event picks the last occurrence.
    key = pd.DataFrame({"_event": event_group_codes(df), "_ttca": ttca, "_pos": order})
    # Per event: the minimal time_to_tca (NaN only when every row is NaN), then
    # among rows at that minimum the largest input position (last occurrence).
    min_ttca = key.groupby("_event", sort=False)["_ttca"].transform("min")
    is_min = (key["_ttca"] == min_ttca) | (key["_ttca"].isna() & min_ttca.isna())
    picked = key[is_min].groupby("_event", sort=False)["_pos"].max()
    positions = np.sort(picked.to_numpy())
    return df.iloc[positions].copy()


# --------------------------------------------------------------------------- #
# Tier metrics
# --------------------------------------------------------------------------- #
def _binary_prf(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[Optional[float], Optional[float], Optional[float], int]:
    tp = int(np.sum(y_true & y_pred))
    n_true = int(np.sum(y_true))
    n_pred = int(np.sum(y_pred))
    recall = _safe_div(tp, n_true)
    precision = _safe_div(tp, n_pred)
    return recall, precision, _f_beta(precision, recall, 2.0), tp


def tier_metrics(y_true_tier, y_pred_tier) -> dict:
    """Tier classification metrics.

    None / NaN tiers are treated as the 'Invalid' class in the confusion matrix.
    Critical and Elevated-or-above (Critical or Elevated) recall / precision / F2
    are computed as binary problems; macro_f1 is sklearn f1_score over the three
    real tiers (zero_division=0).
    """
    yt = _tier_array(y_true_tier)
    yp = _tier_array(y_pred_tier)
    n = len(yt)

    crit_t = yt == "Critical"
    crit_p = yp == "Critical"
    elev_t = crit_t | (yt == "Elevated")
    elev_p = crit_p | (yp == "Elevated")

    c_rec, c_prec, c_f2, _ = _binary_prf(crit_t, crit_p)
    e_rec, e_prec, e_f2, _ = _binary_prf(elev_t, elev_p)

    # Confusion: rows = true, cols = predicted.
    idx = {lab: i for i, lab in enumerate(CONFUSION_LABELS)}
    matrix = np.zeros((len(CONFUSION_LABELS), len(CONFUSION_LABELS)), dtype=int)
    for t, p in zip(yt, yp):
        matrix[idx[t], idx[p]] += 1

    if n == 0 or not np.any(np.isin(yt, TIER_LABELS)):
        macro_f1 = None
    else:
        macro_f1 = float(
            f1_score(yt, yp, labels=TIER_LABELS, average="macro", zero_division=0)
        )

    group = {
        "critical_recall": c_rec,
        "critical_precision": c_prec,
        "critical_f2": c_f2,
        "elevated_or_above_recall": e_rec,
        "elevated_or_above_precision": e_prec,
        "elevated_or_above_f2": e_f2,
        "macro_f1": macro_f1,
        "confusion": {"labels": list(CONFUSION_LABELS), "matrix": matrix.tolist()},
    }
    return _finalise_group(group, exclude=("confusion",))


# --------------------------------------------------------------------------- #
# Per-view metric groups
# --------------------------------------------------------------------------- #
def _pc_only_metrics(risk_true: np.ndarray, risk_pred: np.ndarray, critical_risk: float) -> dict:
    """Critical defined ONLY by the risk threshold (no distance rule)."""
    yt = np.where(np.isnan(risk_true), False, risk_true >= critical_risk)
    yp = np.where(np.isnan(risk_pred), False, risk_pred >= critical_risk)
    rec, prec, f2, tp = _binary_prf(yt.astype(bool), yp.astype(bool))
    group = {
        "critical_recall": rec,
        "critical_precision": prec,
        "critical_f2": f2,
        "n_true_positive_rows": int(tp),
    }
    return _finalise_group(group, exclude=("n_true_positive_rows",))


def _mae_rmse(err: np.ndarray) -> tuple[Optional[float], Optional[float]]:
    if err.size == 0:
        return None, None
    return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err ** 2)))


def _regression_metrics(risk_true: np.ndarray, risk_pred: np.ndarray, floor: float, elevated_risk: float) -> dict:
    valid = ~np.isnan(risk_true) & ~np.isnan(risk_pred)
    err = risk_pred[valid] - risk_true[valid]
    rt = risk_true[valid]
    non_floor = rt > floor
    high = rt >= elevated_risk

    mae, rmse = _mae_rmse(err)
    mae_nf, rmse_nf = _mae_rmse(err[non_floor])
    mae_hr, rmse_hr = _mae_rmse(err[high])
    group = {
        "mae": mae,
        "rmse": rmse,
        "mae_non_floor": mae_nf,
        "rmse_non_floor": rmse_nf,
        "mae_high_risk": mae_hr,
        "rmse_high_risk": rmse_hr,
        "n_non_floor": int(np.sum(non_floor)),
        "n_high_risk": int(np.sum(high)),
    }
    return _finalise_group(group, exclude=("n_non_floor", "n_high_risk"))


def _interval_metrics(risk_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, nominal: float) -> dict:
    n = len(risk_true)
    has_int = ~np.isnan(lower) & ~np.isnan(upper)
    usable = has_int & ~np.isnan(risk_true)
    covered = (lower[usable] <= risk_true[usable]) & (risk_true[usable] <= upper[usable])
    coverage = _safe_div(int(np.sum(covered)), int(np.sum(usable)))
    width = (upper - lower)[has_int]
    median_width = float(np.median(width)) if width.size else None
    group = {
        "nominal_coverage": float(nominal) if nominal is not None else None,
        "coverage": coverage,
        "median_width": median_width,
        "n_with_interval": int(np.sum(has_int)),
        "fraction_with_interval": _safe_div(int(np.sum(has_int)), n),
    }
    return _finalise_group(group, exclude=("n_with_interval", "nominal_coverage"))


def _triage_metrics(actions: np.ndarray, tier_true: np.ndarray) -> dict:
    n = len(actions)
    counts: dict[str, int] = {}
    for a in actions:
        key = "None" if a is None else str(a)
        counts[key] = counts.get(key, 0) + 1
    counts = dict(sorted(counts.items()))

    is_auto = actions == AUTO_CLEAR_ACTION
    is_maneuver = actions == MANEUVER_REVIEW_ACTION
    is_human = np.isin(actions, HUMAN_REVIEW_ACTIONS)
    crit_true = tier_true == "Critical"
    n_crit = int(np.sum(crit_true))
    n_crit_auto = int(np.sum(crit_true & is_auto))
    group = {
        "action_counts": counts,
        "auto_clear_fraction": _safe_div(int(np.sum(is_auto)), n),
        "true_critical_routed_to_auto_clear_n": n_crit_auto,
        "true_critical_routed_to_auto_clear_fraction": _safe_div(n_crit_auto, n_crit),
        "maneuver_review_fraction": _safe_div(int(np.sum(is_maneuver)), n),
        "human_review_fraction": _safe_div(int(np.sum(is_human)), n),
    }
    return _finalise_group(group, exclude=("action_counts", "true_critical_routed_to_auto_clear_n"))


def _classifier_metrics(risk_true: np.ndarray, p_crit: Optional[np.ndarray], critical_risk: float) -> Optional[dict]:
    if p_crit is None:
        return None
    valid = ~np.isnan(p_crit) & ~np.isnan(risk_true)
    y = (risk_true[valid] >= critical_risk).astype(int)
    p = np.clip(p_crit[valid], 0.0, 1.0)
    n_pos = int(np.sum(y))
    pr_auc: Optional[float] = None
    brier: Optional[float] = None
    if y.size > 0:
        brier = float(brier_score_loss(y, p))
        if 0 < n_pos < y.size:
            pr_auc = float(average_precision_score(y, p))
        elif n_pos == y.size:
            pr_auc = 1.0  # all positive: every ranking is perfect
    group = {"pr_auc": pr_auc, "brier": brier, "n_pos": n_pos}
    return _finalise_group(group, exclude=("n_pos",))


def _bootstrap_metrics(
    event_id: np.ndarray,
    crit_t: np.ndarray,
    crit_p: np.ndarray,
    abs_err: np.ndarray,
    err_valid: np.ndarray,
    covered: np.ndarray,
    int_valid: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict:
    """Event-level bootstrap: resample events with replacement and recompute the
    point metrics with per-event multiplicities as weights. Returns 2.5 / 97.5
    percentiles; None bounds when a metric was undefined in (almost) every
    resample or when there is nothing to resample."""
    out: dict[str, dict] = {m: {"lower": None, "upper": None} for m in BOOTSTRAP_METRICS}
    n = len(event_id)
    if n == 0 or n_bootstrap <= 0:
        return _finalise_bootstrap(out)

    codes, uniques = pd.factorize(pd.Series(event_id), sort=False)
    n_events = len(uniques)
    rng = np.random.default_rng(seed)

    # Per-event sufficient statistics.
    tp = np.bincount(codes, weights=(crit_t & crit_p).astype(float), minlength=n_events)
    nt = np.bincount(codes, weights=crit_t.astype(float), minlength=n_events)
    npred = np.bincount(codes, weights=crit_p.astype(float), minlength=n_events)
    ae_sum = np.bincount(codes, weights=np.where(err_valid, abs_err, 0.0), minlength=n_events)
    ae_n = np.bincount(codes, weights=err_valid.astype(float), minlength=n_events)
    cov_sum = np.bincount(codes, weights=(covered & int_valid).astype(float), minlength=n_events)
    cov_n = np.bincount(codes, weights=int_valid.astype(float), minlength=n_events)

    samples = {m: [] for m in BOOTSTRAP_METRICS}
    for _ in range(int(n_bootstrap)):
        draw = rng.integers(0, n_events, size=n_events)
        w = np.bincount(draw, minlength=n_events).astype(float)
        d_nt = float(w @ nt)
        d_np = float(w @ npred)
        d_tp = float(w @ tp)
        if d_nt > 0:
            samples["critical_recall"].append(d_tp / d_nt)
        if d_np > 0:
            samples["critical_precision"].append(d_tp / d_np)
        d_aen = float(w @ ae_n)
        if d_aen > 0:
            samples["mae"].append(float(w @ ae_sum) / d_aen)
        d_cn = float(w @ cov_n)
        if d_cn > 0:
            samples["coverage"].append(float(w @ cov_sum) / d_cn)

    for m, vals in samples.items():
        if len(vals) >= 2:
            lo, hi = np.percentile(np.asarray(vals, dtype=float), [2.5, 97.5])
            out[m] = {"lower": float(lo), "upper": float(hi)}
    return _finalise_bootstrap(out)


def _finalise_bootstrap(out: dict) -> dict:
    out["undefined"] = [m for m in BOOTSTRAP_METRICS if out[m]["lower"] is None or out[m]["upper"] is None]
    return out


def _view_metrics(df: pd.DataFrame, cfg: dict, n_bootstrap: int, seed: int) -> dict:
    critical_risk = float(_get(cfg, ("thresholds", "critical_risk"), DEFAULT_CRITICAL_RISK))
    elevated_risk = float(_get(cfg, ("thresholds", "elevated_risk"), DEFAULT_ELEVATED_RISK))
    floor = float(_get(cfg, ("model", "target_floor"), DEFAULT_TARGET_FLOOR))
    nominal = _get(cfg, ("conformal", "coverage"), DEFAULT_COVERAGE)

    risk_true = _float_col(df, "risk_true")
    risk_pred = _float_col(df, "risk_log10_pred")
    lower = _float_col(df, "risk_log10_lower")
    upper = _float_col(df, "risk_log10_upper")
    tier_true = _tier_array(df["risk_tier_true"]) if "risk_tier_true" in df.columns else np.full(len(df), INVALID_LABEL, dtype=object)
    tier_pred = _tier_array(df["risk_tier"]) if "risk_tier" in df.columns else np.full(len(df), INVALID_LABEL, dtype=object)
    actions = _str_col(df, "triage_action")
    p_crit = _float_col(df, "p_pc_critical") if "p_pc_critical" in df.columns else None
    event_id = event_group_codes(df)

    crit_t = tier_true == "Critical"
    crit_p = tier_pred == "Critical"
    err_valid = ~np.isnan(risk_true) & ~np.isnan(risk_pred)
    abs_err = np.where(err_valid, np.abs(risk_pred - risk_true), 0.0)
    int_valid = ~np.isnan(lower) & ~np.isnan(upper) & ~np.isnan(risk_true)
    covered = int_valid & (lower <= risk_true) & (risk_true <= upper)

    return {
        "n_rows": int(len(df)),
        "n_events": int(pd.Series(event_id).nunique()) if len(df) else 0,
        "n_critical_true": int(np.sum(crit_t)),
        "tier": tier_metrics(tier_true, tier_pred),
        "pc_only": _pc_only_metrics(risk_true, risk_pred, critical_risk),
        "regression": _regression_metrics(risk_true, risk_pred, floor, elevated_risk),
        "interval": _interval_metrics(risk_true, lower, upper, nominal),
        "triage": _triage_metrics(actions, tier_true),
        "classifier": _classifier_metrics(risk_true, p_crit, critical_risk),
        "bootstrap": _bootstrap_metrics(
            event_id, crit_t, crit_p, abs_err, err_valid, covered, int_valid, n_bootstrap, seed
        ),
    }


# --------------------------------------------------------------------------- #
# Public report
# --------------------------------------------------------------------------- #
def evaluate_predictions(
    df: pd.DataFrame,
    cfg: dict,
    *,
    n_bootstrap: int = 200,
    seed: int = 0,
    scope: dict | None = None,
) -> dict:
    """Full JSON-serialisable metrics dict over the four standard views.

    See the module docstring for view definitions and conventions. `scope` is
    passed through verbatim (after conversion to JSON-safe types) so callers can
    label the result (model name, dataset id, ...).
    """
    df = df.reset_index(drop=True)
    floor = float(_get(cfg, ("model", "target_floor"), DEFAULT_TARGET_FLOOR))
    horizon_min = float(_get(cfg, ("split", "test_horizon_min_days"), 0.0))

    risk_true = _float_col(df, "risk_true")
    tier_true = _tier_array(df["risk_tier_true"]) if "risk_tier_true" in df.columns else np.full(len(df), INVALID_LABEL, dtype=object)
    status = _str_col(df, "prediction_status")
    ttca = _float_col(df, "time_to_tca_days")
    event_id = pd.Series(event_group_codes(df), index=df.index)

    crit_rows = tier_true == "Critical"
    counts = {
        "n_rows": int(len(df)),
        "n_events": int(event_id.nunique()) if len(df) else 0,
        "n_critical_rows": int(np.sum(crit_rows)),
        "n_critical_events": int(event_id[crit_rows].nunique()) if len(df) else 0,
        "n_elevated_rows": int(np.sum(tier_true == "Elevated")),
        "n_floor_rows": int(np.sum(~np.isnan(risk_true) & (risk_true <= floor))),
        "n_invalid_rows": int(np.sum(status == "invalid")),
    }

    horizon = df[ttca >= horizon_min] if len(df) else df
    result = {
        "scope": to_jsonable(scope) if scope else {},
        "counts": counts,
        "row_level": _view_metrics(df, cfg, n_bootstrap, seed),
        "latest_cdm": _view_metrics(latest_cdm_view(df), cfg, n_bootstrap, seed),
        "horizon_slice": _view_metrics(horizon, cfg, n_bootstrap, seed),
        "horizon_latest_cdm": _view_metrics(latest_cdm_view(horizon), cfg, n_bootstrap, seed),
    }
    return to_jsonable(result)


# --------------------------------------------------------------------------- #
# Model comparison and illustrative cases
# --------------------------------------------------------------------------- #
COMPARE_COLUMNS: list[tuple[str, tuple[str, ...]]] = [
    ("critical_recall", ("tier", "critical_recall")),
    ("critical_precision", ("tier", "critical_precision")),
    ("critical_f2", ("tier", "critical_f2")),
    ("pc_only_critical_recall", ("pc_only", "critical_recall")),
    ("pc_only_critical_precision", ("pc_only", "critical_precision")),
    ("macro_f1", ("tier", "macro_f1")),
    ("mae", ("regression", "mae")),
    ("mae_non_floor", ("regression", "mae_non_floor")),
    ("mae_high_risk", ("regression", "mae_high_risk")),
    ("coverage", ("interval", "coverage")),
    ("median_width", ("interval", "median_width")),
    ("auto_clear_fraction", ("triage", "auto_clear_fraction")),
    ("true_critical_routed_to_auto_clear_n", ("triage", "true_critical_routed_to_auto_clear_n")),
]


def compare_models(results: dict[str, dict], view: str = "latest_cdm") -> pd.DataFrame:
    """Side-by-side table of key metrics. Rows = model names (index), columns =
    key metrics pulled from results[name][view]; anything missing -> None."""
    rows = []
    for name, res in results.items():
        view_metrics = (res or {}).get(view) or {}
        row: dict[str, Any] = {"model": name}
        for col, path in COMPARE_COLUMNS:
            row[col] = _get(view_metrics, path, None)
        rows.append(row)
    cols = ["model"] + [c for c, _ in COMPARE_COLUMNS]
    out = pd.DataFrame(rows, columns=cols, dtype=object)
    out = out.set_index("model")
    return out.where(pd.notna(out), None)


def _rows_to_dicts(sub: pd.DataFrame, k: int) -> list[dict]:
    sub = sub.head(k)
    return [to_jsonable(rec) for rec in sub.to_dict(orient="records")]


def find_illustrative_cases(df: pd.DataFrame, k: int = 3) -> dict:
    """Pick up to k rows per failure category so the report always has examples.

    - missed_critical:      true Critical, predicted tier not Critical
                             (worst first: largest under-prediction of risk)
    - false_alarm_critical: predicted Critical, true tier not Critical
                             (worst first: largest over-prediction of risk)
    - uncertain:            triage_action == 'Human review', widest intervals first
    - invalid:              prediction_status == 'invalid'
    """
    df = df.reset_index(drop=True)
    n = len(df)
    tier_true = _tier_array(df["risk_tier_true"]) if "risk_tier_true" in df.columns else np.full(n, INVALID_LABEL, dtype=object)
    tier_pred = _tier_array(df["risk_tier"]) if "risk_tier" in df.columns else np.full(n, INVALID_LABEL, dtype=object)
    risk_true = _float_col(df, "risk_true")
    risk_pred = _float_col(df, "risk_log10_pred")
    lower = _float_col(df, "risk_log10_lower")
    upper = _float_col(df, "risk_log10_upper")
    actions = _str_col(df, "triage_action")
    status = _str_col(df, "prediction_status")

    err = risk_pred - risk_true  # positive = over-prediction
    err_sort = np.where(np.isnan(err), 0.0, err)

    missed_mask = (tier_true == "Critical") & (tier_pred != "Critical")
    missed = df[missed_mask].assign(_sort=err_sort[missed_mask]).sort_values("_sort", kind="mergesort")

    fa_mask = (tier_pred == "Critical") & (tier_true != "Critical")
    false_alarm = df[fa_mask].assign(_sort=-err_sort[fa_mask]).sort_values("_sort", kind="mergesort")

    width = upper - lower
    unc_mask = actions == HUMAN_REVIEW_ACTION
    uncertain = df[unc_mask].assign(_sort=np.where(np.isnan(width[unc_mask]), -np.inf, -width[unc_mask]))
    uncertain = uncertain.sort_values("_sort", kind="mergesort")

    invalid = df[status == "invalid"]

    def strip(sub: pd.DataFrame) -> pd.DataFrame:
        return sub.drop(columns=["_sort"], errors="ignore")

    return {
        "missed_critical": _rows_to_dicts(strip(missed), k),
        "false_alarm_critical": _rows_to_dicts(strip(false_alarm), k),
        "uncertain": _rows_to_dicts(strip(uncertain), k),
        "invalid": _rows_to_dicts(strip(invalid), k),
    }


__all__ = [
    "latest_cdm_view",
    "evaluate_predictions",
    "tier_metrics",
    "compare_models",
    "find_illustrative_cases",
    "to_jsonable",
    "CONFUSION_LABELS",
    "TIER_LABELS",
]
