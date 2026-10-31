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


class SplitConformalCalibrator:
    """Symmetric split-conformal intervals with one score per event.

    Per-event nonconformity score = max |y_true - y_pred| over the event's rows
    (rows where either value is NaN are ignored). The interval half-width
    ``q_hat`` is the finite-sample corrected quantile of those event scores.
    """

    def __init__(self, coverage: float = 0.9, min_events: int = 30) -> None:
        if not (0.0 < coverage < 1.0):
            raise ValueError(f"coverage must be in (0, 1), got {coverage}")
        if min_events < 1:
            raise ValueError("min_events must be >= 1")
        self.coverage = float(coverage)
        self.min_events = int(min_events)
        self.status: str = STATUS_UNFITTED
        self.q_hat: float | None = None
        self.n_events: int = 0
        self.n_rows: int = 0

    # -- fitting ------------------------------------------------------------ #
    def fit(
        self,
        y_true: Sequence[float] | np.ndarray | pd.Series,
        y_pred: Sequence[float] | np.ndarray | pd.Series,
        event_ids: Sequence[Any] | np.ndarray | pd.Series,
    ) -> "SplitConformalCalibrator":
        yt = _as_float_array(y_true)
        yp = _as_float_array(y_pred)
        ev = np.asarray(event_ids)
        if not (yt.shape == yp.shape == ev.shape):
            raise ValueError(
                f"shape mismatch: y_true {yt.shape}, y_pred {yp.shape}, event_ids {ev.shape}"
            )

        valid = ~(np.isnan(yt) | np.isnan(yp))
        resid = np.abs(yt[valid] - yp[valid])
        ev_valid = ev[valid]
        self.n_rows = int(valid.sum())

        if self.n_rows == 0:
            self.n_events = 0
            self.q_hat = None
            self.status = STATUS_INSUFFICIENT
            return self

        per_event = (
            pd.Series(resid)
            .groupby(pd.Series(ev_valid).to_numpy(), sort=False)
            .max()
            .to_numpy(dtype="float64")
        )
        self.n_events = int(per_event.size)

        if self.n_events < self.min_events:
            self.q_hat = None
            self.status = STATUS_INSUFFICIENT
            return self

        q = finite_sample_quantile(per_event, self.coverage)
        if q is None:
            self.q_hat = None
            self.status = STATUS_INSUFFICIENT
        else:
            self.q_hat = float(q)
            self.status = STATUS_OK
        return self

    # -- prediction --------------------------------------------------------- #
    def predict_interval(
        self, y_pred: Sequence[float] | np.ndarray | pd.Series
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(lower, upper)`` arrays aligned with ``y_pred``.

        NaN in both bounds wherever the calibrator is not ``ok`` or the point
        prediction itself is NaN. A finite interval is never fabricated.
        """
        yp = _as_float_array(y_pred)
        lower = np.full(yp.shape, np.nan, dtype="float64")
        upper = np.full(yp.shape, np.nan, dtype="float64")
        if self.status != STATUS_OK or self.q_hat is None:
            return lower, upper
        ok = ~np.isnan(yp)
        lower[ok] = yp[ok] - self.q_hat
        upper[ok] = yp[ok] + self.q_hat
        return lower, upper

    # -- (de)serialisation -------------------------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "split_conformal",
            "coverage": self.coverage,
            "min_events": self.min_events,
            "status": self.status,
            "q_hat": None if self.q_hat is None else float(self.q_hat),
            "n_events": int(self.n_events),
            "n_rows": int(self.n_rows),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SplitConformalCalibrator":
        obj = cls(coverage=float(d["coverage"]), min_events=int(d.get("min_events", 30)))
        obj.status = str(d.get("status", STATUS_UNFITTED))
        q = d.get("q_hat")
        obj.q_hat = None if q is None else float(q)
        obj.n_events = int(d.get("n_events", 0))
        obj.n_rows = int(d.get("n_rows", 0))
        if obj.status == STATUS_OK and obj.q_hat is None:
            # Inconsistent payload: never allow an 'ok' calibrator without a width.
            obj.status = STATUS_INSUFFICIENT
        return obj

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"SplitConformalCalibrator(coverage={self.coverage}, status={self.status!r}, "
            f"q_hat={self.q_hat}, n_events={self.n_events}, n_rows={self.n_rows})"
        )


def empirical_coverage(
    y_true: Sequence[float] | np.ndarray | pd.Series,
    lower: Sequence[float] | np.ndarray | pd.Series,
    upper: Sequence[float] | np.ndarray | pd.Series,
) -> dict[str, Any]:
    """Observed coverage and width of intervals.

    Only rows where ``y_true``, ``lower`` and ``upper`` are all finite count.
    Returns ``coverage`` and ``median_width`` as ``None`` when no such rows exist.
    """
    yt = _as_float_array(y_true)
    lo = _as_float_array(lower)
    hi = _as_float_array(upper)
    if not (yt.shape == lo.shape == hi.shape):
        raise ValueError("y_true, lower and upper must have the same shape")
    has_interval = ~(np.isnan(lo) | np.isnan(hi))
    usable = has_interval & ~np.isnan(yt)
    n_usable = int(usable.sum())
    out: dict[str, Any] = {
        "coverage": None,
        "median_width": None,
        "n_rows": int(yt.size),
        "n_with_interval": int(has_interval.sum()),
    }
    if n_usable == 0:
        return out
    covered = (lo[usable] <= yt[usable]) & (yt[usable] <= hi[usable])
    out["coverage"] = float(covered.mean())
    out["median_width"] = float(np.median(hi[usable] - lo[usable]))
    return out


# --------------------------------------------------------------------------- #
# classifier calibration
# --------------------------------------------------------------------------- #
class SigmoidCalibrator:
    """Platt scaling: ``p = sigmoid(a * score + b)``.

    The logistic regression is fitted on the RAW classifier score (not its
    logit). Rationale: the score fed in is whatever the upstream classifier
    emits (a margin, a probability, or a boosted raw score); fitting on the raw
    value keeps the mapping monotone and avoids ``logit(0)``/``logit(1)``
    blow-ups when the upstream model is over-confident. ``C`` is set very large
    so regularisation has no practical effect.
    """

    def __init__(self, C: float = 1e6) -> None:
        self.C = float(C)
        self.a: float | None = None  # slope
        self.b: float | None = None  # intercept
        self.n: int = 0
        self.n_pos: int = 0

    @property
    def fitted(self) -> bool:
        return self.a is not None and self.b is not None

    def fit(self, scores, y) -> "SigmoidCalibrator":
        s = _as_float_array(scores)
        yy = np.asarray(y).astype(int).ravel()
        if s.shape != yy.shape:
            raise ValueError("scores and y must have the same shape")
        ok = ~np.isnan(s)
        s, yy = s[ok], yy[ok]
        self.n = int(s.size)
        self.n_pos = int(yy.sum())
        if self.n_pos == 0 or self.n_pos == self.n:
            raise ValueError("SigmoidCalibrator needs both classes present to fit")
        lr = LogisticRegression(C=self.C, solver="lbfgs", max_iter=10_000)
        lr.fit(s.reshape(-1, 1), yy)
        self.a = float(lr.coef_[0, 0])
        self.b = float(lr.intercept_[0])
        return self

    def predict_proba(self, scores) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("SigmoidCalibrator is not fitted")
        s = _as_float_array(scores)
        z = self.a * s + self.b
        # numerically stable sigmoid; NaN propagates as NaN
        with np.errstate(over="ignore", invalid="ignore"):
            p = np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))
        p = np.where(np.isnan(s), np.nan, p)
        return p.astype("float64")

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "sigmoid",
            "C": self.C,
            "a": self.a,
            "b": self.b,
            "n": self.n,
            "n_pos": self.n_pos,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SigmoidCalibrator":
        obj = cls(C=float(d.get("C", 1e6)))
        obj.a = None if d.get("a") is None else float(d["a"])
        obj.b = None if d.get("b") is None else float(d["b"])
        obj.n = int(d.get("n", 0))
        obj.n_pos = int(d.get("n_pos", 0))
        return obj


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
def classifier_metrics(y_true, p) -> dict[str, Any]:
    """PR-AUC and Brier score for a probabilistic binary classifier.

    Rows with NaN probability are dropped. When there are no positives (or no
    rows) the metrics are ``None`` and ``undefined`` is ``True`` -- never NaN.
    """
    yy = np.asarray(y_true).astype(float).ravel()
    pp = _as_float_array(p)
    if yy.shape != pp.shape:
        raise ValueError("y_true and p must have the same shape")
    ok = ~(np.isnan(pp) | np.isnan(yy))
    yy, pp = yy[ok].astype(int), pp[ok]
    n = int(yy.size)
    n_pos = int(yy.sum())
    out: dict[str, Any] = {
        "pr_auc": None,
        "brier": None,
        "n_pos": n_pos,
        "n": n,
        "undefined": True,
    }
    if n == 0 or n_pos == 0:
        return out
    out["brier"] = float(brier_score_loss(yy, np.clip(pp, 0.0, 1.0)))
    if n_pos == n:
        # all positives: PR-AUC is trivially 1 but carries no information
        out["pr_auc"] = 1.0
    else:
        out["pr_auc"] = float(average_precision_score(yy, pp))
    out["undefined"] = False
    return out


def reliability_diagram(y_true, p, n_bins: int = 10) -> dict[str, list]:
    """Equal-width reliability bins over [0, 1].

    Bin ``i`` covers ``[i/n_bins, (i+1)/n_bins)`` with the last bin closed on
    the right so that ``p == 1`` is included. Empty bins have ``None`` for
    ``mean_pred`` and ``frac_pos`` and ``count == 0``.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    yy = np.asarray(y_true).astype(float).ravel()
    pp = _as_float_array(p)
    if yy.shape != pp.shape:
        raise ValueError("y_true and p must have the same shape")
    ok = ~(np.isnan(pp) | np.isnan(yy))
    yy, pp = yy[ok], np.clip(pp[ok], 0.0, 1.0)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.floor(pp * n_bins).astype(int), 0, n_bins - 1)

    bin_lower: list[float] = []
    bin_upper: list[float] = []
    mean_pred: list[float | None] = []
    frac_pos: list[float | None] = []
    count: list[int] = []
    for i in range(n_bins):
        m = idx == i
        c = int(m.sum())
        bin_lower.append(float(edges[i]))
        bin_upper.append(float(edges[i + 1]))
        count.append(c)
        if c == 0:
            mean_pred.append(None)
            frac_pos.append(None)
        else:
            mean_pred.append(float(pp[m].mean()))
            frac_pos.append(float(yy[m].mean()))
    return {
        "bin_lower": bin_lower,
        "bin_upper": bin_upper,
        "mean_pred": mean_pred,
        "frac_pos": frac_pos,
        "count": count,
    }


__all__ = [
    "STATUS_OK",
    "STATUS_INSUFFICIENT",
    "STATUS_UNFITTED",
    "finite_sample_quantile",
    "SplitConformalCalibrator",
    "empirical_coverage",
    "SigmoidCalibrator",
    "IsotonicCalibrator",
    "classifier_metrics",
    "reliability_diagram",
]


# --------------------------------------------------------------------------- #
# Conformalized quantile regression (CQR) with one score per event
# --------------------------------------------------------------------------- #
class CQRCalibrator:
    """Conformalized quantile regression. Given lower/upper quantile predictions
    (already crossing-corrected), the per-row score is max(lower - y, y - upper); the
    per-event score is the max over the event's rows; ``q_hat`` is the finite-sample
    corrected quantile of event scores. Interval = [lower - q_hat, upper + q_hat].
    Widths adapt to the quantile heads; the correction is a constant shift."""

    def __init__(self, coverage: float = 0.9, min_events: int = 30) -> None:
        if not (0.0 < coverage < 1.0):
            raise ValueError(f"coverage must be in (0, 1), got {coverage}")
        self.coverage = float(coverage)
        self.min_events = int(min_events)
        self.status: str = STATUS_UNFITTED
        self.q_hat: float | None = None
        self.n_events: int = 0
        self.n_rows: int = 0

    def fit(self, y_true, lower_pred, upper_pred, event_ids) -> "CQRCalibrator":
        yt = _as_float_array(y_true)
        lo = _as_float_array(lower_pred)
        hi = _as_float_array(upper_pred)
        ev = np.asarray(event_ids)
        if not (yt.shape == lo.shape == hi.shape == ev.shape):
            raise ValueError("shape mismatch in CQRCalibrator.fit")
        valid = ~(np.isnan(yt) | np.isnan(lo) | np.isnan(hi))
        score = np.maximum(lo[valid] - yt[valid], yt[valid] - hi[valid])
        self.n_rows = int(valid.sum())
        if self.n_rows == 0:
            self.n_events, self.q_hat, self.status = 0, None, STATUS_INSUFFICIENT
            return self
        per_event = pd.Series(score).groupby(pd.Series(ev[valid]).to_numpy(), sort=False).max().to_numpy(dtype="float64")
        self.n_events = int(per_event.size)
        if self.n_events < self.min_events:
            self.q_hat, self.status = None, STATUS_INSUFFICIENT
            return self
        q = finite_sample_quantile(per_event, self.coverage)
        if q is None:
            self.q_hat, self.status = None, STATUS_INSUFFICIENT
        else:
            self.q_hat, self.status = float(q), STATUS_OK
        return self

    def predict_interval(self, lower_pred, upper_pred) -> tuple[np.ndarray, np.ndarray]:
        lo = _as_float_array(lower_pred)
        hi = _as_float_array(upper_pred)
        lower = np.full(lo.shape, np.nan, dtype="float64")
        upper = np.full(hi.shape, np.nan, dtype="float64")
        if self.status != STATUS_OK or self.q_hat is None:
            return lower, upper
        ok = ~(np.isnan(lo) | np.isnan(hi))
        lower[ok] = lo[ok] - self.q_hat
        upper[ok] = hi[ok] + self.q_hat
        return lower, upper

    def to_dict(self) -> dict[str, Any]:
        return {"type": "cqr", "coverage": self.coverage, "min_events": self.min_events, "status": self.status,
                "q_hat": None if self.q_hat is None else float(self.q_hat), "n_events": int(self.n_events), "n_rows": int(self.n_rows)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CQRCalibrator":
        obj = cls(coverage=float(d["coverage"]), min_events=int(d.get("min_events", 30)))
        obj.status = str(d.get("status", STATUS_UNFITTED))
        q = d.get("q_hat")
        obj.q_hat = None if q is None else float(q)
        obj.n_events = int(d.get("n_events", 0))
        obj.n_rows = int(d.get("n_rows", 0))
        if obj.status == STATUS_OK and obj.q_hat is None:
            obj.status = STATUS_INSUFFICIENT
        return obj


def fix_quantile_crossing(point: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ensure lower <= point <= upper elementwise (NaN preserved)."""
    lo = np.fmin(lower, point)
    hi = np.fmax(upper, point)
    return lo, hi
