"""Evaluation of final-tier forecasts for collision-risk triage.

The task: forecast each event's FINAL tier (Safe / Elevated / Critical) from the
CDMs available at >= 2 days before TCA. Evaluation is one row per event (the
`is_latest_snapshot` row of the frame described in docs/forecast_schema.md),
always reported on three subsets:

- all_usable:     every event
- trivial:        `is_trivial` is True (the final CDM was itself at >= 2 days, so the
                  answer is visible in the input)
- real_forecast:  `~is_trivial` (the headline subset)

Conventions
- Tier order everywhere is TIERS = [Safe, Elevated, Critical]; confusion matrices
  have true tiers on rows and predicted tiers on columns.
- A None / NaN predicted tier is never counted as Safe. It is counted as wrong and,
  when any exist, appears as an extra 'Invalid' column in the confusion matrix.
- Any metric with a zero denominator is None and listed under the enclosing
  group's 'undefined' key. No NaN ever leaves this module
  (json.dumps(..., allow_nan=False) passes on every result).
- Tier assignment and triage live in policy.py and are deliberately not imported.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .data import event_group_codes

TIERS: list[str] = ["Safe", "Elevated", "Critical"]
TIER_INDEX: dict[str, int] = {t: i for i, t in enumerate(TIERS)}
INVALID_LABEL = "Invalid"

AUTO_CLEAR_ACTION = "Auto-clear candidate"
MANEUVER_REVIEW_ACTION = "Maneuver planning review"
HUMAN_REVIEW_ACTION = "Human review"
HUMAN_REVIEW_INSUFFICIENT_ACTION = "Human review - insufficient data"
ESCALATION_WATCH_ACTION = "Escalation watch"
MONITOR_ACTION = "Monitor"
LOW_PRIORITY_MONITOR_ACTION = "Low-priority monitor"
HUMAN_REVIEW_ACTIONS = (HUMAN_REVIEW_ACTION, HUMAN_REVIEW_INSUFFICIENT_ACTION)
ACTION_ORDER = (MANEUVER_REVIEW_ACTION, ESCALATION_WATCH_ACTION, HUMAN_REVIEW_INSUFFICIENT_ACTION, HUMAN_REVIEW_ACTION,
                MONITOR_ACTION, LOW_PRIORITY_MONITOR_ACTION, AUTO_CLEAR_ACTION)

PROB_COLUMNS: list[str] = ["p_safe", "p_elevated", "p_critical"]
SUBSET_NAMES: tuple[str, ...] = ("all_usable", "trivial", "real_forecast")
BOOTSTRAP_METRICS: tuple[str, ...] = ("critical_recall", "critical_precision", "macro_f1", "auto_clear_share")
LEAD_TIME_EDGES: list[float] = [2.0, 3.0, 4.0, 5.0, math.inf]
LEAD_TIME_LABELS: list[str] = ["[2,3)", "[3,4)", "[4,5)", "[5,inf)"]
LEAD_TIME_OTHER = "<2_or_missing"
N_RELIABILITY_BINS = 10
LOG_LOSS_CLIP = 1e-12
DEFAULT_COVERAGE = 0.90


# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #
def _is_missing(x: Any) -> bool:
    """True for None, NaN, pd.NA and the like (never for lists/arrays)."""
    if x is None:
        return True
    if isinstance(x, (list, tuple, set, dict, np.ndarray)):
        return False
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
    """F-beta from precision / recall. Undefined when recall is undefined (no
    positives in the truth); when precision is undefined (nothing predicted
    positive) recall is zero, so F-beta is zero (sklearn zero_division=0)."""
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
    """Attach the 'undefined' list: every top-level key (not excluded) whose value is None."""
    excl = set(exclude) | {"undefined"}
    group["undefined"] = [k for k, v in group.items() if k not in excl and v is None]
    return group


def _get(cfg: Any, path: Sequence[str], default: Any) -> Any:
    cur: Any = cfg or {}
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _tier_array(values: Any) -> np.ndarray:
    """Normalise a tier sequence to an object array of TIERS entries or 'Invalid'."""
    if values is None:
        return np.asarray([], dtype=object)
    out = []
    for v in list(values):
        if _is_missing(v):
            out.append(INVALID_LABEL)
            continue
        s = v.value if hasattr(v, "value") and not isinstance(v, str) else v
        s = str(s)
        out.append(s if s in TIER_INDEX else INVALID_LABEL)
    return np.asarray(out, dtype=object)


def _tier_col(df: pd.DataFrame, col: str) -> np.ndarray:
    if col in df.columns:
        return _tier_array(df[col].tolist())
    return np.full(len(df), INVALID_LABEL, dtype=object)


def _str_col(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), None, dtype=object)
    out = []
    for v in df[col].tolist():
        if _is_missing(v):
            out.append(None)
        else:
            out.append(str(v.value if hasattr(v, "value") and not isinstance(v, str) else v))
    return np.asarray(out, dtype=object)


def _float_col(df: pd.DataFrame, col: str) -> np.ndarray:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    return np.full(len(df), np.nan, dtype=float)


def _bool_col(df: pd.DataFrame, col: str, default: bool = False) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), default, dtype=bool)
    return np.asarray([default if _is_missing(v) else bool(v) for v in df[col].tolist()], dtype=bool)


def _prob_matrix(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([_float_col(df, c) for c in PROB_COLUMNS]) if len(df) else np.zeros((0, 3))


def _class_counts(tiers: np.ndarray) -> dict[str, int]:
    return {t: int(np.sum(tiers == t)) for t in TIERS}


# --------------------------------------------------------------------------- #
# Views
# --------------------------------------------------------------------------- #
def latest_snapshot_view(df: pd.DataFrame) -> pd.DataFrame:
    """One row per event: the rows flagged `is_latest_snapshot`. When that column is
    absent, fall back to the row with the smallest `time_to_tca_days` per
    (dataset_id, event_id) (last occurrence wins ties; NaN only wins when every row
    of the event is NaN). Index is preserved."""
    if len(df) == 0:
        return df.copy()
    if "is_latest_snapshot" in df.columns:
        mask = _bool_col(df, "is_latest_snapshot")
        return df[mask].copy()
    order = np.arange(len(df))
    ttca = _float_col(df, "time_to_tca_days")
    key = pd.DataFrame({"_event": event_group_codes(df), "_ttca": ttca, "_pos": order})
    min_ttca = key.groupby("_event", sort=False)["_ttca"].transform("min")
    is_min = (key["_ttca"] == min_ttca) | (key["_ttca"].isna() & min_ttca.isna())
    picked = key[is_min].groupby("_event", sort=False)["_pos"].max()
    positions = np.sort(picked.to_numpy())
    return df.iloc[positions].copy()


# --------------------------------------------------------------------------- #
# Classification metrics
# --------------------------------------------------------------------------- #
def _binary_prf(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[Optional[float], Optional[float], Optional[float]]:
    tp = int(np.sum(y_true & y_pred))
    recall = _safe_div(tp, int(np.sum(y_true)))
    precision = _safe_div(tp, int(np.sum(y_pred)))
    return recall, precision, _f_beta(precision, recall, 2.0)


def classification_metrics(y_true, y_pred) -> dict:
    """Three-tier classification metrics.

    Rows whose truth is missing / not a tier are dropped (`n_dropped_invalid_truth`).
    Predictions that are None / NaN / not a tier count as wrong: when any exist the
    confusion matrix gets a 4th 'Invalid' column (never a Safe prediction).
    `macro_f1` averages F1 over the tiers present in truth or predictions
    (sklearn's default label set, zero_division=0)."""
    yt = _tier_array(y_true)
    yp = _tier_array(y_pred)
    keep = yt != INVALID_LABEL
    n_dropped = int(np.sum(~keep))
    yt, yp = yt[keep], yp[keep]
    n = int(len(yt))

    n_invalid_pred = int(np.sum(yp == INVALID_LABEL))
    pred_labels = list(TIERS) + ([INVALID_LABEL] if n_invalid_pred else [])
    col_index = {lab: i for i, lab in enumerate(pred_labels)}
    counts = np.zeros((len(TIERS), len(pred_labels)), dtype=int)
    for t, p in zip(yt, yp):
        counts[TIER_INDEX[t], col_index[p]] += 1
    row_sums = counts.sum(axis=1)
    row_norm = [
        [float(c) / float(s) for c in row] if s > 0 else None
        for row, s in zip(counts.tolist(), row_sums.tolist())
    ]

    per_class: dict[str, dict] = {}
    f1_present: list[float] = []
    for i, tier in enumerate(TIERS):
        tp = int(counts[i, i])
        support = int(row_sums[i])
        n_pred = int(counts[:, i].sum())
        prec = _safe_div(tp, n_pred)
        rec = _safe_div(tp, support)
        f1 = _f_beta(prec, rec, 1.0)
        per_class[tier] = {"precision": prec, "recall": rec, "f1": f1, "support": support, "n_predicted": n_pred}
        if support > 0 or n_pred > 0:
            f1_present.append(0.0 if f1 is None else f1)
    macro_f1 = float(np.mean(f1_present)) if (n > 0 and f1_present) else None
    accuracy = _safe_div(int(np.trace(counts[:, : len(TIERS)])), n)

    crit_t, crit_p = yt == "Critical", yp == "Critical"
    elev_t = crit_t | (yt == "Elevated")
    elev_p = crit_p | (yp == "Elevated")
    c_rec, c_prec, c_f2 = _binary_prf(crit_t, crit_p)
    e_rec, e_prec, e_f2 = _binary_prf(elev_t, elev_p)

    group = {
        "n": n,
        "n_dropped_invalid_truth": n_dropped,
        "n_invalid_pred": n_invalid_pred,
        "confusion": {
            "labels": list(TIERS),
            "pred_labels": pred_labels,
            "counts": counts.tolist(),
            "row_normalised": row_norm,
            "note": (
                "rows = true tier, columns = predicted tier"
                + ("; 'Invalid' column = missing predictions counted as wrong" if n_invalid_pred else "")
            ),
        },
        "per_class": per_class,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "critical_recall": c_rec,
        "critical_precision": c_prec,
        "critical_f2": c_f2,
        "elevated_or_above_recall": e_rec,
        "elevated_or_above_precision": e_prec,
        "elevated_or_above_f2": e_f2,
    }
    group = _finalise_group(group, exclude=("n", "n_dropped_invalid_truth", "n_invalid_pred", "confusion", "per_class"))
    for tier, stats in per_class.items():
        for k in ("precision", "recall", "f1"):
            if stats[k] is None:
                group["undefined"].append(f"per_class.{tier}.{k}")
    return group


# --------------------------------------------------------------------------- #
# Probability metrics
# --------------------------------------------------------------------------- #
def _empty_reliability() -> dict:
    edges = np.linspace(0.0, 1.0, N_RELIABILITY_BINS + 1)
    return {
        "bin_lower": edges[:-1].tolist(),
        "bin_upper": edges[1:].tolist(),
        "mean_pred": [None] * N_RELIABILITY_BINS,
        "frac_pos": [None] * N_RELIABILITY_BINS,
        "count": [0] * N_RELIABILITY_BINS,
        "ece": None,
    }


def _reliability(p: np.ndarray, y: np.ndarray) -> dict:
    """One-vs-rest reliability table over 10 equal-width bins (p = 1 falls in the last bin)."""
    out = _empty_reliability()
    n = len(p)
    if n == 0:
        return out
    edges = np.linspace(0.0, 1.0, N_RELIABILITY_BINS + 1)
    bins = np.clip(np.digitize(np.clip(p, 0.0, 1.0), edges[1:-1], right=False), 0, N_RELIABILITY_BINS - 1)
    count = np.bincount(bins, minlength=N_RELIABILITY_BINS).astype(float)
    sum_pred = np.bincount(bins, weights=p, minlength=N_RELIABILITY_BINS)
    sum_pos = np.bincount(bins, weights=y.astype(float), minlength=N_RELIABILITY_BINS)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_pred = np.where(count > 0, sum_pred / count, np.nan)
        frac_pos = np.where(count > 0, sum_pos / count, np.nan)
    nonempty = count > 0
    ece = float(np.sum(count[nonempty] / n * np.abs(mean_pred[nonempty] - frac_pos[nonempty])))
    out["mean_pred"] = [None if not nonempty[i] else float(mean_pred[i]) for i in range(N_RELIABILITY_BINS)]
    out["frac_pos"] = [None if not nonempty[i] else float(frac_pos[i]) for i in range(N_RELIABILITY_BINS)]
    out["count"] = [int(c) for c in count]
    out["ece"] = ece
    return out


def probability_metrics(P: np.ndarray, y_true) -> dict:
    """Multiclass probability quality. `P` is (n, 3) in TIERS order.

    Rows with any NaN probability (or an invalid truth) are excluded (`n_excluded`).
    brier = mean over rows of sum over classes (p - onehot)^2 (range 0..2);
    log_loss uses probabilities clipped at 1e-12; reliability is one-vs-rest per tier.
    Critical one-vs-rest ranking quality (PR-AUC, ROC-AUC) is included because the
    Critical class is rare (~7 %)."""
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[1] != len(TIERS):
        raise ValueError(f"P must have shape (n, {len(TIERS)}), got {P.shape}")
    yt = _tier_array(y_true)
    if len(yt) != P.shape[0]:
        raise ValueError("P and y_true have different lengths")
    valid = ~np.isnan(P).any(axis=1) & (yt != INVALID_LABEL)
    n_used = int(np.sum(valid))
    n_excluded = int(np.sum(~valid))

    group: dict[str, Any] = {
        "n_used": n_used,
        "n_excluded": n_excluded,
        "brier": None,
        "log_loss": None,
        "per_class_brier": {t: None for t in TIERS},
        "reliability": {t: _empty_reliability() for t in TIERS},
        "critical_pr_auc": None,
        "critical_roc_auc": None,
        "mean_max_prob": None,
    }
    if n_used == 0:
        return _finalise_group(group, exclude=("n_used", "n_excluded", "per_class_brier", "reliability"))

    Pv = P[valid]
    idx = np.asarray([TIER_INDEX[t] for t in yt[valid]], dtype=int)
    onehot = np.zeros_like(Pv)
    onehot[np.arange(n_used), idx] = 1.0
    sq = (Pv - onehot) ** 2
    group["brier"] = float(np.mean(sq.sum(axis=1)))
    group["per_class_brier"] = {t: float(np.mean(sq[:, i])) for i, t in enumerate(TIERS)}
    p_true = np.clip(Pv[np.arange(n_used), idx], LOG_LOSS_CLIP, 1.0)
    group["log_loss"] = float(-np.mean(np.log(p_true)))
    group["mean_max_prob"] = float(np.mean(Pv.max(axis=1)))
    group["reliability"] = {t: _reliability(Pv[:, i], onehot[:, i]) for i, t in enumerate(TIERS)}

    y_crit = onehot[:, TIER_INDEX["Critical"]].astype(int)
    p_crit = np.clip(Pv[:, TIER_INDEX["Critical"]], 0.0, 1.0)
    n_pos = int(y_crit.sum())
    if 0 < n_pos < n_used:
        group["critical_pr_auc"] = float(average_precision_score(y_crit, p_crit))
        group["critical_roc_auc"] = float(roc_auc_score(y_crit, p_crit))
    return _finalise_group(group, exclude=("n_used", "n_excluded", "per_class_brier", "reliability"))


# --------------------------------------------------------------------------- #
# Escalation / de-escalation and triage
# --------------------------------------------------------------------------- #
def escalation_metrics(df: pd.DataFrame) -> dict:
    """Escalation (currently not Critical, ends Critical) and de-escalation (currently
    Critical, ends lower) bookkeeping. A non-Critical event is 'flagged' when the
    forecast is Critical OR its triage action is 'Escalation watch'. Watch-list
    numbers are None when `triage_action` is absent."""
    cur = _tier_col(df, "current_tier")
    fin = _tier_col(df, "final_tier")
    pred = _tier_col(df, "predicted_final_tier")
    has_actions = "triage_action" in df.columns
    actions = _str_col(df, "triage_action")
    usable = (cur != INVALID_LABEL) & (fin != INVALID_LABEL)

    on_watch = actions == ESCALATION_WATCH_ACTION
    pred_crit = pred == "Critical"
    fin_crit = fin == "Critical"

    nc = usable & (cur != "Critical")
    flagged = nc & (pred_crit | on_watch)
    n_nc = int(np.sum(nc))
    n_end_crit = int(np.sum(nc & fin_crit))
    n_flagged = int(np.sum(flagged))
    n_caught = int(np.sum(flagged & fin_crit))
    n_pred_crit = int(np.sum(nc & pred_crit))
    n_pred_crit_hits = int(np.sum(nc & pred_crit & fin_crit))
    non_critical_now = {
        "n_total": n_nc,
        "n_end_critical": n_end_crit,
        "n_flagged": n_flagged,
        "n_caught": n_caught,
        "recall_of_escalations": _safe_div(n_caught, n_end_crit),
        "precision_of_flags": _safe_div(n_caught, n_flagged),
        "n_predicted_critical": n_pred_crit,
        "n_predicted_critical_hits": n_pred_crit_hits,
        "watch_list_size": int(np.sum(nc & on_watch)) if has_actions else None,
        "watch_list_hits": int(np.sum(nc & on_watch & fin_crit)) if has_actions else None,
        "watch_list_share": _safe_div(int(np.sum(nc & on_watch)), n_nc) if has_actions else None,
    }
    non_critical_now = _finalise_group(
        non_critical_now,
        exclude=("n_total", "n_end_critical", "n_flagged", "n_caught", "n_predicted_critical", "n_predicted_critical_hits"),
    )

    cc = usable & (cur == "Critical")
    pred_lower = cc & (pred != INVALID_LABEL) & ~pred_crit
    n_cc = int(np.sum(cc))
    n_deesc = int(np.sum(cc & ~fin_crit))
    n_pred_deesc = int(np.sum(pred_lower))
    n_correct = int(np.sum(pred_lower & ~fin_crit))
    n_false = int(np.sum(pred_lower & fin_crit))
    critical_now = {
        "n_total": n_cc,
        "n_deescalate": n_deesc,
        "n_predicted_deescalation": n_pred_deesc,
        "n_correct_deescalation": n_correct,
        "n_false_deescalation": n_false,
        "deescalation_recall": _safe_div(n_correct, n_deesc),
        "deescalation_precision": _safe_div(n_correct, n_pred_deesc),
        "false_deescalation_rate": _safe_div(n_false, n_cc - n_deesc),
    }
    critical_now = _finalise_group(
        critical_now,
        exclude=("n_total", "n_deescalate", "n_predicted_deescalation", "n_correct_deescalation", "n_false_deescalation"),
    )

    out = {
        "n_events": int(np.sum(usable)),
        "non_critical_now": non_critical_now,
        "critical_now": critical_now,
        "undefined": [f"non_critical_now.{k}" for k in non_critical_now["undefined"]]
        + [f"critical_now.{k}" for k in critical_now["undefined"]],
    }
    return out


def triage_metrics(df: pd.DataFrame) -> Optional[dict]:
    """Triage action mix and the safety counters. None when `triage_action` is absent."""
    if "triage_action" not in df.columns:
        return None
    actions = _str_col(df, "triage_action")
    fin = _tier_col(df, "final_tier")
    n = int(len(actions))
    counts: dict[str, int] = {}
    for a in actions:
        key = "None" if a is None else str(a)
        counts[key] = counts.get(key, 0) + 1
    counts = dict(sorted(counts.items()))
    shares = {k: (_safe_div(v, n) if n else None) for k, v in counts.items()}

    is_auto = actions == AUTO_CLEAR_ACTION
    is_watch = actions == ESCALATION_WATCH_ACTION
    is_maneuver = actions == MANEUVER_REVIEW_ACTION
    is_monitor = actions == MONITOR_ACTION
    is_low_priority = actions == LOW_PRIORITY_MONITOR_ACTION
    is_human = np.isin(actions, HUMAN_REVIEW_ACTIONS)
    fin_crit = fin == "Critical"
    n_crit = int(np.sum(fin_crit))
    n_crit_auto = int(np.sum(fin_crit & is_auto))
    # True final tier by triage action: shows every routing failure, not only Critical -> auto-clear
    # (e.g. final-Critical events parked in Monitor or Low-priority monitor).
    present = [a for a in ACTION_ORDER if a in counts] + [a for a in counts if a not in ACTION_ORDER]
    by_tier = {a: {t: int(np.sum((actions == a) & (fin == t))) for t in TIERS + [INVALID_LABEL]} for a in present}
    group = {
        "n_events": n,
        "action_counts": counts,
        "action_shares": shares,
        "action_by_final_tier": by_tier,
        "auto_clear_share": _safe_div(int(np.sum(is_auto)), n),
        "n_final_critical": n_crit,
        "n_final_critical_auto_cleared": n_crit_auto,
        "final_critical_auto_cleared_share": _safe_div(n_crit_auto, n_crit),
        "n_final_critical_in_watch": int(np.sum(fin_crit & is_watch)),
        "n_final_critical_in_maneuver_review": int(np.sum(fin_crit & is_maneuver)),
        "n_final_critical_in_monitor": int(np.sum(fin_crit & is_monitor)),
        "n_final_critical_in_low_priority_monitor": int(np.sum(fin_crit & is_low_priority)),
        "n_final_critical_in_human_review": int(np.sum(fin_crit & is_human)),
        "n_final_critical_not_urgent": int(np.sum(fin_crit & (is_monitor | is_low_priority | is_auto))),
        "maneuver_review_share": _safe_div(int(np.sum(is_maneuver)), n),
        "human_review_share": _safe_div(int(np.sum(is_human)), n),
        "watch_share": _safe_div(int(np.sum(is_watch)), n),
        "monitor_share": _safe_div(int(np.sum(is_monitor)), n),
        "low_priority_monitor_share": _safe_div(int(np.sum(is_low_priority)), n),
        "review_workload_share": _safe_div(int(np.sum(is_maneuver | is_watch | is_human)), n),
    }
    return _finalise_group(
        group,
        exclude=(
            "n_events", "action_counts", "action_shares", "action_by_final_tier", "n_final_critical",
            "n_final_critical_auto_cleared", "n_final_critical_in_watch", "n_final_critical_in_maneuver_review",
            "n_final_critical_in_monitor", "n_final_critical_in_low_priority_monitor", "n_final_critical_in_human_review",
            "n_final_critical_not_urgent",
        ),
    )


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #
def bootstrap_cis(df: pd.DataFrame, n_bootstrap: int = 1000, seed: int = 0) -> dict:
    """Event bootstrap (rows are events) of critical_recall, critical_precision,
    macro_f1 and auto_clear_share: 2.5 / 97.5 percentiles over resamples where the
    metric is defined. Vectorised: resample indices with default_rng(seed).choice,
    turn them into per-event multiplicities, and use matrix products."""
    out: dict[str, Any] = {m: {"lower": None, "upper": None, "n_valid_draws": 0} for m in BOOTSTRAP_METRICS}
    out["n_bootstrap"] = int(n_bootstrap)
    out["seed"] = int(seed)
    n = int(len(df))
    if n == 0 or n_bootstrap <= 0:
        return _finalise_bootstrap(out)

    yt = _tier_col(df, "final_tier")
    yp = _tier_col(df, "predicted_final_tier")
    T = np.column_stack([(yt == t) for t in TIERS]).astype(float)
    Pm = np.column_stack([(yp == t) for t in TIERS]).astype(float)
    TP = T * Pm
    has_actions = "triage_action" in df.columns
    auto = (_str_col(df, "triage_action") == AUTO_CLEAR_ACTION).astype(float)
    crit = TIER_INDEX["Critical"]

    rng = np.random.default_rng(seed)
    samples: dict[str, list[np.ndarray]] = {m: [] for m in BOOTSTRAP_METRICS}
    chunk = 200
    for start in range(0, int(n_bootstrap), chunk):
        b = min(chunk, int(n_bootstrap) - start)
        draws = rng.choice(n, size=(b, n), replace=True)
        flat = (draws + (np.arange(b) * n)[:, None]).ravel()
        W = np.bincount(flat, minlength=b * n).reshape(b, n).astype(float)
        nt = W @ T
        npred = W @ Pm
        tp = W @ TP
        with np.errstate(divide="ignore", invalid="ignore"):
            rec = np.where(nt > 0, tp / nt, np.nan)
            prec = np.where(npred > 0, tp / npred, np.nan)
            denom = prec + rec
            f1 = np.where(denom > 0, 2.0 * prec * rec / denom, 0.0)
        present = (nt > 0) | (npred > 0)
        f1 = np.where(np.isnan(f1), 0.0, f1)
        n_present = present.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            macro = np.where(n_present > 0, (f1 * present).sum(axis=1) / n_present, np.nan)
        samples["critical_recall"].append(rec[:, crit])
        samples["critical_precision"].append(prec[:, crit])
        samples["macro_f1"].append(macro)
        if has_actions:
            samples["auto_clear_share"].append((W @ auto) / float(n))

    for m in BOOTSTRAP_METRICS:
        if not samples[m]:
            continue
        vals = np.concatenate(samples[m])
        vals = vals[~np.isnan(vals)]
        out[m]["n_valid_draws"] = int(vals.size)
        if vals.size >= 2:
            lo, hi = np.percentile(vals, [2.5, 97.5])
            out[m]["lower"], out[m]["upper"] = float(lo), float(hi)
    return _finalise_bootstrap(out)


def _finalise_bootstrap(out: dict) -> dict:
    out["undefined"] = [m for m in BOOTSTRAP_METRICS if out[m]["lower"] is None or out[m]["upper"] is None]
    return out


# --------------------------------------------------------------------------- #
# Breakdowns
# --------------------------------------------------------------------------- #
def _group_stats(yt: np.ndarray, yp: np.ndarray) -> dict:
    cm = classification_metrics(yt, yp)
    group = {
        "n": int(len(yt)),
        "n_critical": int(np.sum(yt == "Critical")),
        "n_predicted_critical": int(np.sum(yp == "Critical")),
        "critical_recall": cm["critical_recall"],
        "critical_precision": cm["critical_precision"],
        "critical_f2": cm["critical_f2"],
        "macro_f1": cm["macro_f1"],
        "accuracy": cm["accuracy"],
    }
    return _finalise_group(group, exclude=("n", "n_critical", "n_predicted_critical"))


def breakdown_by(df: pd.DataFrame, column: str, min_rows: int = 20) -> dict:
    """Per-group classification stats. Missing group values become 'missing'; groups
    with fewer than `min_rows` rows are merged into 'other'. Empty dict when the
    column is absent or the frame is empty."""
    if column not in df.columns or len(df) == 0:
        return {}
    keys = _str_col(df, column)
    keys = np.asarray([("missing" if k is None else k) for k in keys], dtype=object)
    yt = _tier_col(df, "final_tier")
    yp = _tier_col(df, "predicted_final_tier")
    counts = pd.Series(keys).value_counts()
    small = set(counts[counts < int(min_rows)].index.tolist())
    groups = np.asarray([("other" if k in small else k) for k in keys], dtype=object)
    names = sorted(set(groups.tolist()), key=lambda g: (g == "other", g))
    out: dict[str, dict] = {}
    for g in names:
        mask = groups == g
        stats = _group_stats(yt[mask], yp[mask])
        if g == "other":
            stats["merged_groups"] = sorted(small)
        out[g] = stats
    return out


def _lead_time_bucket(ttca: np.ndarray) -> np.ndarray:
    out = np.full(len(ttca), LEAD_TIME_OTHER, dtype=object)
    for lo, hi, label in zip(LEAD_TIME_EDGES[:-1], LEAD_TIME_EDGES[1:], LEAD_TIME_LABELS):
        out[(ttca >= lo) & (ttca < hi)] = label
    return out


def lead_time_breakdown(df: pd.DataFrame) -> dict:
    """Stats per lead-time bucket of the latest snapshot's `time_to_tca_days`:
    [2,3), [3,4), [4,5), [5,inf); rows below 2 days or missing go to '<2_or_missing'.
    Buckets partition the rows (no merging)."""
    if len(df) == 0 or "time_to_tca_days" not in df.columns:
        return {}
    tmp = pd.DataFrame({"_lead_bucket": _lead_time_bucket(_float_col(df, "time_to_tca_days"))})
    for col in ("final_tier", "predicted_final_tier"):
        if col in df.columns:
            tmp[col] = df[col].tolist()
    raw = breakdown_by(tmp, "_lead_bucket", min_rows=0)
    ordered = {label: raw[label] for label in LEAD_TIME_LABELS + [LEAD_TIME_OTHER] if label in raw}
    return ordered


# --------------------------------------------------------------------------- #
# Conformal prediction sets (optional column)
# --------------------------------------------------------------------------- #
def _prediction_set_metrics(df: pd.DataFrame, nominal: float) -> Optional[dict]:
    if "prediction_set" not in df.columns:
        return None
    fin = _tier_col(df, "final_tier")
    sets = df["prediction_set"].tolist()
    pred = _tier_col(df, "predicted_final_tier")
    sizes, covered, truths, disagree = [], [], [], []
    for s, t, p in zip(sets, fin, pred):
        if _is_missing(s) or t == INVALID_LABEL:
            continue
        items = [str(x) for x in list(s)]
        sizes.append(len(items))
        covered.append(t in items)
        truths.append(t)
        # the threshold decision names a class the set does not contain
        disagree.append(p != INVALID_LABEL and p not in items)
    n_used = len(sizes)
    sizes_a, covered_a, truths_a = np.asarray(sizes), np.asarray(covered, dtype=bool), np.asarray(truths, dtype=object)
    # Class-conditional coverage: marginal 90 % coverage can hide near-zero coverage of the rare Critical class.
    by_class = {t: _safe_div(int(np.sum(covered_a & (truths_a == t))), int(np.sum(truths_a == t))) for t in TIERS}
    size_by_class = {t: (float(np.mean(sizes_a[truths_a == t])) if np.any(truths_a == t) else None) for t in TIERS}
    group = {
        "n_used": n_used,
        "coverage_nominal": float(nominal) if nominal is not None else None,
        "coverage_empirical": _safe_div(int(np.sum(covered_a)), n_used),
        "coverage_by_true_class": by_class,
        "avg_set_size": (float(np.mean(sizes_a)) if n_used else None),
        "avg_set_size_by_true_class": size_by_class,
        "singleton_share": _safe_div(int(np.sum(sizes_a == 1)), n_used) if n_used else None,
        "empty_share": _safe_div(int(np.sum(sizes_a == 0)), n_used) if n_used else None,
        "decision_outside_set_share": _safe_div(int(np.sum(disagree)), n_used) if n_used else None,
    }
    return _finalise_group(group, exclude=("n_used", "coverage_nominal", "coverage_by_true_class", "avg_set_size_by_true_class"))


# --------------------------------------------------------------------------- #
# Public report
# --------------------------------------------------------------------------- #
def _subset_report(sub: pd.DataFrame, cfg: dict, n_bootstrap: int, seed: int) -> dict:
    yt = _tier_col(sub, "final_tier")
    yp = _tier_col(sub, "predicted_final_tier")
    nominal = _get(cfg, ("conformal_sets", "coverage"), DEFAULT_COVERAGE)
    return {
        "n_events": int(len(sub)),
        "class_counts": _class_counts(yt),
        "predicted_class_counts": _class_counts(yp),
        "classification": classification_metrics(yt, yp),
        "probability": probability_metrics(_prob_matrix(sub), yt),
        "escalation": escalation_metrics(sub),
        "triage": triage_metrics(sub),
        "prediction_sets": _prediction_set_metrics(sub, nominal),
        "bootstrap": bootstrap_cis(sub, n_bootstrap=n_bootstrap, seed=seed),
    }


def evaluate_forecasts(
    df: pd.DataFrame,
    cfg: dict,
    *,
    n_bootstrap: int = 1000,
    seed: int = 0,
    scope: dict | None = None,
) -> dict:
    """Full JSON-serialisable evaluation of a latest-snapshot forecast frame.

    `df` must hold one row per event with `final_tier` (truth), `predicted_final_tier`,
    `p_safe` / `p_elevated` / `p_critical`, `current_tier`, `is_trivial`, and
    optionally `triage_action`, `object_type`, `time_to_tca_days`, `prediction_set`.
    If `is_latest_snapshot` is present, only those rows are used. When `is_trivial`
    is absent every event is treated as a real forecast (and a note is added).

    Returns {'scope', 'n_events', 'n_trivial', 'n_real_forecast', 'class_counts',
    'subsets': {all_usable | trivial | real_forecast: {...}},
    'by_object_type' / 'by_lead_time' / 'by_current_tier': {real_forecast: ..., all_usable: ...},
    'notes': [...]}. Every undefined metric is None and listed under the enclosing
    group's 'undefined'; json.dumps(result, allow_nan=False) always passes."""
    notes: list[str] = []
    if "is_latest_snapshot" in df.columns:
        df = latest_snapshot_view(df)
    df = df.reset_index(drop=True)
    n = int(len(df))
    if "is_trivial" in df.columns:
        trivial = _bool_col(df, "is_trivial", default=False)
    else:
        trivial = np.zeros(n, dtype=bool)
        notes.append("is_trivial column absent: every event treated as a real forecast")
    yt = _tier_col(df, "final_tier")

    subsets_df = {
        "all_usable": df,
        "trivial": df[trivial],
        "real_forecast": df[~trivial],
    }
    subsets = {name: _subset_report(sub, cfg, n_bootstrap, seed) for name, sub in subsets_df.items()}

    def both(fn, *args) -> dict:
        return {"real_forecast": fn(subsets_df["real_forecast"], *args), "all_usable": fn(subsets_df["all_usable"], *args)}

    result = {
        "scope": to_jsonable(scope) if scope else {},
        "n_events": n,
        "n_trivial": int(np.sum(trivial)),
        "n_real_forecast": int(np.sum(~trivial)),
        "class_counts": _class_counts(yt),
        "subsets": subsets,
        "by_object_type": both(breakdown_by, "object_type", 20),
        "by_lead_time": both(lead_time_breakdown),
        "by_current_tier": both(breakdown_by, "current_tier", 0),
        "notes": notes,
    }
    return to_jsonable(result)


# --------------------------------------------------------------------------- #
# Model comparison and illustrative cases
# --------------------------------------------------------------------------- #
COMPARE_COLUMNS: list[tuple[str, tuple[str, ...]]] = [
    ("n_events", ("n_events",)),
    ("critical_recall", ("classification", "critical_recall")),
    ("critical_precision", ("classification", "critical_precision")),
    ("critical_f2", ("classification", "critical_f2")),
    ("elevated_or_above_f2", ("classification", "elevated_or_above_f2")),
    ("macro_f1", ("classification", "macro_f1")),
    ("accuracy", ("classification", "accuracy")),
    ("brier", ("probability", "brier")),
    ("log_loss", ("probability", "log_loss")),
    ("auto_clear_share", ("triage", "auto_clear_share")),
    ("n_final_critical_auto_cleared", ("triage", "n_final_critical_auto_cleared")),
    ("low_priority_monitor_share", ("triage", "low_priority_monitor_share")),
    ("human_review_share", ("triage", "human_review_share")),
    ("n_final_critical_not_urgent", ("triage", "n_final_critical_not_urgent")),
    ("watch_share", ("triage", "watch_share")),
    ("escalation_recall", ("escalation", "non_critical_now", "recall_of_escalations")),
    ("escalation_flag_precision", ("escalation", "non_critical_now", "precision_of_flags")),
]


def compare_models(results: dict[str, dict], subset: str = "real_forecast") -> pd.DataFrame:
    """One row per model (index) with the key metrics of results[name]['subsets'][subset];
    anything missing is None."""
    rows = []
    for name, res in results.items():
        sub = (((res or {}).get("subsets") or {}).get(subset)) or {}
        row: dict[str, Any] = {"model": name}
        for col, path in COMPARE_COLUMNS:
            row[col] = _get(sub, path, None)
        rows.append(row)
    cols = ["model"] + [c for c, _ in COMPARE_COLUMNS]
    out = pd.DataFrame(rows, columns=cols, dtype=object).set_index("model")
    return out.where(pd.notna(out), None)


def _rows_to_dicts(sub: pd.DataFrame, k: int) -> list[dict]:
    sub = sub.drop(columns=["_sort", "_sort2"], errors="ignore").head(k)
    return [to_jsonable(rec) for rec in sub.to_dict(orient="records")]


def find_illustrative_cases(df: pd.DataFrame, k: int = 3) -> dict:
    """Up to k JSON-safe rows per category so the report always has examples.

    - missed_critical:      final Critical, not predicted Critical, not on the watch list
                             (lowest p_critical first)
    - caught_by_watch:      final Critical, not predicted Critical, triage 'Escalation watch'
    - false_alarm_critical: predicted Critical, final not Critical (highest p_critical first)
    - false_deescalation:   current Critical, predicted lower, final Critical
    - uncertain:            largest prediction_set (then smallest max probability), or
                             smallest max probability when no sets are available
    """
    df = df.reset_index(drop=True)
    fin = _tier_col(df, "final_tier")
    pred = _tier_col(df, "predicted_final_tier")
    cur = _tier_col(df, "current_tier")
    actions = _str_col(df, "triage_action")
    P = _prob_matrix(df)
    p_crit = P[:, TIER_INDEX["Critical"]] if len(df) else np.zeros(0)
    p_crit_sort = np.where(np.isnan(p_crit), 0.5, p_crit)
    max_p = np.nanmax(np.where(np.isnan(P), -np.inf, P), axis=1) if len(df) else np.zeros(0)
    max_p = np.where(np.isfinite(max_p), max_p, 2.0)  # rows without probabilities sort last

    fin_crit = fin == "Critical"
    pred_crit = pred == "Critical"
    on_watch = actions == ESCALATION_WATCH_ACTION

    missed_mask = fin_crit & ~pred_crit & ~on_watch
    missed = df[missed_mask].assign(_sort=p_crit_sort[missed_mask]).sort_values("_sort", kind="mergesort")

    watch_mask = fin_crit & ~pred_crit & on_watch
    caught = df[watch_mask].assign(_sort=p_crit_sort[watch_mask]).sort_values("_sort", kind="mergesort")

    fa_mask = pred_crit & ~fin_crit & (fin != INVALID_LABEL)
    false_alarm = df[fa_mask].assign(_sort=-p_crit_sort[fa_mask]).sort_values("_sort", kind="mergesort")

    fd_mask = (cur == "Critical") & (pred != INVALID_LABEL) & ~pred_crit & fin_crit
    false_deesc = df[fd_mask].assign(_sort=p_crit_sort[fd_mask]).sort_values("_sort", kind="mergesort")

    has_probs = ~np.isnan(P).any(axis=1) if len(df) else np.zeros(0, dtype=bool)
    if "prediction_set" in df.columns:
        sizes = np.asarray(
            [(-1 if _is_missing(s) else len(list(s))) for s in df["prediction_set"].tolist()], dtype=float
        )
        unc_mask = (sizes >= 0) | has_probs
        uncertain = (
            df[unc_mask]
            .assign(_sort=-sizes[unc_mask], _sort2=max_p[unc_mask])
            .sort_values(["_sort", "_sort2"], kind="mergesort")
        )
    else:
        uncertain = df[has_probs].assign(_sort=max_p[has_probs]).sort_values("_sort", kind="mergesort")

    return {
        "missed_critical": _rows_to_dicts(missed, k),
        "caught_by_watch": _rows_to_dicts(caught, k),
        "false_alarm_critical": _rows_to_dicts(false_alarm, k),
        "false_deescalation": _rows_to_dicts(false_deesc, k),
        "uncertain": _rows_to_dicts(uncertain, k),
    }


__all__ = [
    "TIERS",
    "INVALID_LABEL",
    "SUBSET_NAMES",
    "latest_snapshot_view",
    "classification_metrics",
    "probability_metrics",
    "escalation_metrics",
    "triage_metrics",
    "bootstrap_cis",
    "breakdown_by",
    "lead_time_breakdown",
    "evaluate_forecasts",
    "compare_models",
    "find_illustrative_cases",
    "to_jsonable",
]
