"""Uncertainty calibration for the collision-risk engine.

Two independent concerns live here:

1. Split-conformal prediction intervals around the frozen point regressor
   (target ``risk = log10(Pc)``). CDMs within a conjunction event are strongly
   dependent, so the calibration set contributes ONE nonconformity score per
   event (the max absolute residual over that event's rows). The finite-sample
   quantile correction then operates on the number of events, not rows, and an
   interval is only ever produced when the calibration set is large enough.

2. Probability calibration for the binary "Pc critical" classifier score
   (Platt / sigmoid scaling and isotonic regression), plus the usual
   classification diagnostics (PR-AUC, Brier, reliability diagram).

Design rule: never fabricate a finite interval or metric when the data cannot
support it. Insufficient calibration yields NaN intervals and an explicit
status; undefined metrics are ``None`` with ``undefined: True``.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss

STATUS_OK = "ok"
STATUS_INSUFFICIENT = "insufficient_calibration"
STATUS_UNFITTED = "unfitted"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _as_float_array(x: Any) -> np.ndarray:
    """Coerce a sequence / Series / scalar to a 1-D float64 array (None -> NaN)."""
    if isinstance(x, pd.Series):
        x = x.to_numpy(dtype="float64", na_value=np.nan)
    arr = np.asarray(x, dtype="float64")
    return np.atleast_1d(arr)


# --------------------------------------------------------------------------- #
# split conformal
# --------------------------------------------------------------------------- #
def finite_sample_quantile(scores: np.ndarray, coverage: float) -> float | None:
    """Split-conformal finite-sample corrected quantile.

    Sort ``scores`` ascending, let ``n = len(scores)`` and
    ``k = ceil((n + 1) * coverage)``. If ``k > n`` the calibration set is too
    small to guarantee the requested coverage and ``None`` is returned.
    Otherwise the k-th smallest score (1-indexed) is returned.
    """
    if not (0.0 < coverage < 1.0):
        raise ValueError(f"coverage must be in (0, 1), got {coverage}")
    s = _as_float_array(scores)
    s = s[~np.isnan(s)]
    n = int(s.size)
    if n == 0:
        return None
    k = math.ceil((n + 1) * coverage)
    if k > n:
        return None
    s_sorted = np.sort(s)
    return float(s_sorted[k - 1])


class IsotonicCalibrator:
    """Monotone non-parametric calibration via ``IsotonicRegression``.

    Scores outside the fitted range are clipped to the boundary values
    (``out_of_bounds='clip'``). The fitted step function is stored as its
    breakpoints so it can be serialised without pickling.
    """

    def __init__(self) -> None:
        self.x_thresholds: np.ndarray | None = None
        self.y_thresholds: np.ndarray | None = None
        self.n: int = 0
        self.n_pos: int = 0

    @property
    def fitted(self) -> bool:
        return self.x_thresholds is not None and self.y_thresholds is not None

    def _model(self) -> IsotonicRegression:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(self.x_thresholds, self.y_thresholds)
        return iso

    def fit(self, scores, y) -> "IsotonicCalibrator":
        s = _as_float_array(scores)
        yy = np.asarray(y).astype(float).ravel()
        if s.shape != yy.shape:
            raise ValueError("scores and y must have the same shape")
        ok = ~np.isnan(s)
        s, yy = s[ok], yy[ok]
        self.n = int(s.size)
        self.n_pos = int(yy.sum())
        if self.n == 0:
            raise ValueError("IsotonicCalibrator needs at least one finite score")
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(s, yy)
        self.x_thresholds = np.asarray(iso.X_thresholds_, dtype="float64")
        self.y_thresholds = np.asarray(iso.y_thresholds_, dtype="float64")
        return self

    def predict_proba(self, scores) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("IsotonicCalibrator is not fitted")
        s = _as_float_array(scores)
        out = np.full(s.shape, np.nan, dtype="float64")
        ok = ~np.isnan(s)
        if ok.any():
            if self.x_thresholds.size == 1:
                out[ok] = float(self.y_thresholds[0])
            else:
                out[ok] = self._model().predict(s[ok])
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "isotonic",
            "x_thresholds": None if self.x_thresholds is None else self.x_thresholds.tolist(),
            "y_thresholds": None if self.y_thresholds is None else self.y_thresholds.tolist(),
            "n": self.n,
            "n_pos": self.n_pos,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "IsotonicCalibrator":
        obj = cls()
        xs, ys = d.get("x_thresholds"), d.get("y_thresholds")
        obj.x_thresholds = None if xs is None else np.asarray(xs, dtype="float64")
        obj.y_thresholds = None if ys is None else np.asarray(ys, dtype="float64")
        obj.n = int(d.get("n", 0))
        obj.n_pos = int(d.get("n_pos", 0))
        return obj


# --------------------------------------------------------------------------- #
# classifier diagnostics
# --------------------------------------------------------------------------- #


# =========================================================================== #
# Final-tier task: multiclass calibration, decision rule, conformal prediction sets
# =========================================================================== #
class MulticlassIsotonicCalibrator:
    """Per-class one-vs-rest isotonic calibration on out-of-fold probabilities, renormalised to sum to 1."""

    def __init__(self) -> None:
        self.calibrators: list[IsotonicCalibrator] = []

    def fit(self, P: np.ndarray, y_idx: np.ndarray) -> "MulticlassIsotonicCalibrator":
        P = np.asarray(P, float); y_idx = np.asarray(y_idx)
        ok = np.isfinite(P).all(axis=1)
        self.calibrators = []
        for k in range(P.shape[1]):
            self.calibrators.append(IsotonicCalibrator().fit(P[ok, k], (y_idx[ok] == k).astype(int)))
        return self

    def predict_proba(self, P: np.ndarray) -> np.ndarray:
        P = np.asarray(P, float)
        out = np.full_like(P, np.nan)
        ok = np.isfinite(P).all(axis=1)
        if ok.any():
            Q = np.column_stack([c.predict_proba(P[ok, k]) for k, c in enumerate(self.calibrators)])
            Q = np.clip(Q, 1e-6, 1.0)
            out[ok] = Q / Q.sum(axis=1, keepdims=True)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"type": "multiclass_isotonic", "per_class": [c.to_dict() for c in self.calibrators]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MulticlassIsotonicCalibrator":
        obj = cls()
        obj.calibrators = [IsotonicCalibrator.from_dict(x) for x in d["per_class"]]
        return obj


def apply_decision_rule(P: np.ndarray, t_crit: float, t_elev: float) -> np.ndarray:
    """Critical if P(C) >= t_crit; else Elevated if P(E)+P(C) >= t_elev; else Safe. NaN rows -> None."""
    P = np.asarray(P, float)
    out = np.full(len(P), None, dtype=object)
    ok = np.isfinite(P).all(axis=1)
    crit = ok & (P[:, 2] >= t_crit)
    elev = ok & ~crit & ((P[:, 1] + P[:, 2]) >= t_elev)
    safe = ok & ~crit & ~elev
    out[crit] = "Critical"; out[elev] = "Elevated"; out[safe] = "Safe"
    return out


def _f_beta(prec: float | None, rec: float | None, beta: float = 2.0) -> float | None:
    if prec is None or rec is None:
        return None
    b2 = beta * beta
    return (1 + b2) * prec * rec / (b2 * prec + rec) if (b2 * prec + rec) > 0 else 0.0


def tune_decision_thresholds(P: np.ndarray, y_idx: np.ndarray, *, grid: np.ndarray | None = None) -> dict[str, Any]:
    """t_crit maximises Critical F2; given t_crit, t_elev maximises Elevated-or-above F2 (on OOF rows)."""
    P = np.asarray(P, float); y_idx = np.asarray(y_idx)
    ok = np.isfinite(P).all(axis=1)
    P, y_idx = P[ok], y_idx[ok]
    grid = np.round(np.arange(0.05, 0.951, 0.01), 3) if grid is None else grid
    best_c, best_f2c = 0.5, -1.0
    true_c = y_idx == 2
    for t in grid:
        pc = P[:, 2] >= t
        tp = (pc & true_c).sum(); fp = (pc & ~true_c).sum(); fn = (~pc & true_c).sum()
        f2 = _f_beta(tp / (tp + fp) if tp + fp else None, tp / (tp + fn) if tp + fn else None)
        if f2 is not None and f2 > best_f2c:
            best_f2c, best_c = float(f2), float(t)
    best_e, best_f2e = 0.5, -1.0
    true_ge = y_idx >= 1
    for t in grid:
        pred = apply_decision_rule(P, best_c, float(t))
        ge = (pred == "Elevated") | (pred == "Critical")
        tp = (ge & true_ge).sum(); fp = (ge & ~true_ge).sum(); fn = (~ge & true_ge).sum()
        f2 = _f_beta(tp / (tp + fp) if tp + fp else None, tp / (tp + fn) if tp + fn else None)
        if f2 is not None and f2 > best_f2e:
            best_f2e, best_e = float(f2), float(t)
    return {"t_crit": best_c, "t_elev": best_e, "oof_critical_f2": best_f2c, "oof_elevated_or_above_f2": best_f2e}


class LACPredictionSets:
    """Split-conformal classification sets (least ambiguous set-valued classifier): score = 1 - p_true on
    calibration rows; q_hat = finite-sample quantile; set = {c : 1 - p_c <= q_hat}. Always non-empty (argmax kept)."""

    def __init__(self, coverage: float = 0.9, min_rows: int = 30) -> None:
        self.coverage = float(coverage); self.min_rows = int(min_rows)
        self.q_hat: float | None = None; self.status: str = STATUS_UNFITTED; self.n_rows: int = 0

    def fit(self, P: np.ndarray, y_idx: np.ndarray) -> "LACPredictionSets":
        P = np.asarray(P, float); y_idx = np.asarray(y_idx)
        ok = np.isfinite(P).all(axis=1)
        scores = 1.0 - P[ok, :][np.arange(ok.sum()), y_idx[ok]]
        self.n_rows = int(ok.sum())
        q = finite_sample_quantile(scores, self.coverage) if self.n_rows >= self.min_rows else None
        self.q_hat, self.status = (float(q), STATUS_OK) if q is not None else (None, STATUS_INSUFFICIENT)
        return self

    def predict_sets(self, P: np.ndarray, labels: list[str] = ("Safe", "Elevated", "Critical")) -> list[list[str] | None]:
        P = np.asarray(P, float)
        out: list[list[str] | None] = []
        for row in P:
            if self.status != STATUS_OK or not np.isfinite(row).all():
                out.append(None); continue
            inc = (1.0 - row) <= self.q_hat
            inc[int(row.argmax())] = True
            out.append([labels[i] for i in range(len(labels)) if inc[i]])
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"type": "lac", "coverage": self.coverage, "min_rows": self.min_rows, "q_hat": self.q_hat, "status": self.status, "n_rows": self.n_rows}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LACPredictionSets":
        obj = cls(coverage=float(d["coverage"]), min_rows=int(d.get("min_rows", 30)))
        obj.q_hat = d.get("q_hat"); obj.status = d.get("status", STATUS_UNFITTED); obj.n_rows = int(d.get("n_rows", 0))
        if obj.status == STATUS_OK and obj.q_hat is None:
            obj.status = STATUS_INSUFFICIENT
        return obj


def prediction_set_metrics(sets: list[list[str] | None], y_true: np.ndarray) -> dict[str, Any]:
    ok = [s is not None for s in sets]
    n = int(sum(ok))
    if n == 0:
        return {"coverage_empirical": None, "avg_set_size": None, "n": 0}
    cov = float(np.mean([y in s for s, y, o in zip(sets, y_true, ok) if o]))
    size = float(np.mean([len(s) for s, o in zip(sets, ok) if o]))
    return {"coverage_empirical": cov, "avg_set_size": size, "n": n}
