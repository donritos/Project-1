"""Event-level final-tier metrics for the held-out evaluation screen. Computed directly from the exported
predictions and the separately stored outcomes, one prediction per event."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..contracts import TIER_ORDER


def confusion(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    cm = pd.crosstab(pd.Categorical(y_true, TIER_ORDER), pd.Categorical(y_pred, TIER_ORDER), dropna=False)
    cm.index.name, cm.columns.name = "Observed final tier", "Predicted final tier"
    return cm.reindex(index=TIER_ORDER, columns=TIER_ORDER, fill_value=0)


def _prf(tp: int, fp: int, fn: int) -> tuple[Optional[float], Optional[float], Optional[float]]:
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f1 = 2 * p * r / (p + r) if p is not None and r is not None and p + r > 0 else None
    return p, r, f1


def f_beta(p: Optional[float], r: Optional[float], beta: float = 2.0) -> Optional[float]:
    if p is None or r is None:
        return None
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r) if (b2 * p + r) > 0 else 0.0


def tier_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    cm = confusion(y_true, y_pred).to_numpy()
    per = {}
    for i, t in enumerate(TIER_ORDER):
        tp = int(cm[i, i]); fp = int(cm[:, i].sum() - tp); fn = int(cm[i, :].sum() - tp)
        p, r, f1 = _prf(tp, fp, fn)
        per[t] = {"precision": p, "recall": r, "f1": f1, "support": int(cm[i, :].sum())}
    f1s = [per[t]["f1"] for t in TIER_ORDER if per[t]["f1"] is not None]
    crit = per["Critical"]
    # Elevated-or-above vs Safe
    ge_t = np.isin(np.asarray(y_true), ["Elevated", "Critical"]); ge_p = np.isin(np.asarray(y_pred), ["Elevated", "Critical"])
    pe, re, _ = _prf(int((ge_t & ge_p).sum()), int((~ge_t & ge_p).sum()), int((ge_t & ~ge_p).sum()))
    return {"n": int(cm.sum()), "macro_f1": float(np.mean(f1s)) if f1s else None, "accuracy": float(np.trace(cm) / cm.sum()) if cm.sum() else None,
            "critical_precision": crit["precision"], "critical_recall": crit["recall"],
            "critical_f2": f_beta(crit["precision"], crit["recall"]), "elevated_plus_f2": f_beta(pe, re),
            "per_class": per, "confusion": cm.tolist()}


def reliability(p: pd.Series, y: pd.Series, bins: int = 10) -> pd.DataFrame:
    p = pd.to_numeric(p, errors="coerce"); y = pd.Series(np.asarray(y, dtype=float), index=p.index)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    df = pd.DataFrame({"bin": idx, "p": p, "y": y}).dropna()
    g = df.groupby("bin").agg(mean_predicted=("p", "mean"), observed=("y", "mean"), events=("y", "size")).reset_index()
    g["bin_label"] = [f"{edges[b]:.1f}-{edges[b + 1]:.1f}" for b in g["bin"]]
    return g


def brier(p: pd.Series, y: pd.Series) -> Optional[float]:
    p = pd.to_numeric(p, errors="coerce"); m = p.notna()
    return float(np.mean((p[m] - np.asarray(y, dtype=float)[m.to_numpy()]) ** 2)) if m.any() else None


def evaluation_frame(events: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    """Join predictions to outcomes for the evaluation screen only."""
    o = outcomes.rename(columns={"final_tier": "observed_final_tier"})
    keep = [c for c in ["event_id", "observed_final_tier", "is_trivial", "final_time_to_tca_days"] if c in o.columns]
    return events.merge(o[keep], on="event_id", how="left", validate="one_to_one")
