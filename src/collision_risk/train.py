"""Final-tier forecasting: 80/20 event split, snapshot weights, LightGBM multiclass / ordinal
models, grouped out-of-fold search, and baselines. All splitting is by event, never by row."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split

from .features import CATEGORICAL, feature_matrix

def _hash_splits(events: dict[str, list[int]]) -> str:
    return hashlib.sha256(json.dumps(events, sort_keys=True).encode()).hexdigest()[:16]


def _lgb_dataset(X: pd.DataFrame, y: np.ndarray, w: np.ndarray | None = None, reference=None) -> lgb.Dataset:
    cats = [c for c in CATEGORICAL if c in X.columns]
    return lgb.Dataset(X, label=y, weight=w, categorical_feature=cats, reference=reference, free_raw_data=False)


# =========================================================================== #
# Final-tier forecasting task: 80/20 event split, weights, multiclass / ordinal models, baselines
# =========================================================================== #
TIERS = ["Safe", "Elevated", "Critical"]
TIER_INDEX = {t: i for i, t in enumerate(TIERS)}


def make_holdout_split(labels: pd.DataFrame, cfg: dict) -> dict[str, Any]:
    """80/20 event split over USABLE events, stratified on final_tier x is_trivial, fixed seed.
    Returns {'seed','holdout_fraction','events': {'dev': [...], 'holdout': [...]}, 'counts', 'split_hash'}."""
    s = cfg["split"]
    seed = int(s["seed"])
    frac = float(s["holdout"])
    lab = labels[labels["usable_for_training"]].copy()
    strata = lab["final_tier"].astype(str) + "|" + lab["is_trivial"].map({True: "trivial", False: "real"})
    vc = strata.value_counts()
    strata = strata.where(strata.map(vc) >= 2, lab["final_tier"].astype(str))
    dev, hold = train_test_split(lab, test_size=frac, random_state=seed, stratify=strata)
    events = {"dev": sorted(int(x) for x in dev["event_id"]), "holdout": sorted(int(x) for x in hold["event_id"])}
    assert not (set(events["dev"]) & set(events["holdout"]))
    counts = {}
    for name, sub in (("dev", dev), ("holdout", hold)):
        counts[name] = {"n_events": int(len(sub)), "class_counts": sub["final_tier"].value_counts().to_dict(),
                        "n_trivial": int(sub["is_trivial"].sum()), "n_real_forecast": int((~sub["is_trivial"]).sum()),
                        "critical_share": float((sub["final_tier"] == "Critical").mean())}
    return {"seed": seed, "holdout_fraction": frac, "events": events, "counts": counts, "split_hash": _hash_splits(events),
            "n_unusable_excluded": int((~labels["usable_for_training"]).sum())}


def snapshot_weights(df: pd.DataFrame, y: np.ndarray, cfg: dict) -> np.ndarray:
    """1 / n_snapshots_of_event (every event counts equally) x capped inverse class-frequency weight."""
    ev = df["dataset_id"].astype(str) + ":" + df["event_id"].astype(str)
    per_event = ev.map(ev.value_counts()).to_numpy(dtype=float)
    w = 1.0 / per_event
    y = np.asarray(y)
    n_events = ev.nunique()
    cap = float(cfg["model"].get("class_weight_cap", 10.0))
    # class frequency measured in events (sum of per-event weights), not rows
    ev_w = pd.Series(w).groupby(y).sum()
    cls_w = {k: min(cap, n_events / (len(ev_w) * v)) for k, v in ev_w.items()}
    return w * np.array([cls_w[k] for k in y], dtype=float)


def multiclass_params(cfg: dict, seed: int = 42) -> dict[str, Any]:
    return {"objective": "multiclass", "num_class": 3, "metric": "multi_logloss", "verbosity": -1, "seed": seed,
            "num_threads": int(cfg["model"].get("num_threads", 0)) or None, "feature_pre_filter": False}


def sample_class_param_grid(cfg: dict, n: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    m = cfg["model"]
    return [{
        "learning_rate": float(rng.uniform(*m["learning_rate"])),
        "num_leaves": int(rng.integers(m["num_leaves"][0], m["num_leaves"][1] + 1)),
        "min_child_samples": int(rng.integers(m["min_child_samples"][0], m["min_child_samples"][1] + 1)),
        "feature_fraction": float(rng.uniform(*m["feature_fraction"])),
        "lambda_l2": float(rng.choice(m["lambda_l2"])),
    } for _ in range(n)]


def fit_multiclass(X: pd.DataFrame, y_idx: np.ndarray, params: dict[str, Any], cfg: dict, *, w: np.ndarray | None = None,
                   X_val: pd.DataFrame | None = None, y_val: np.ndarray | None = None, w_val: np.ndarray | None = None,
                   num_rounds: int | None = None) -> tuple[lgb.Booster, int]:
    p = {**multiclass_params(cfg), **params}
    p = {k: v for k, v in p.items() if v is not None}
    dtrain = _lgb_dataset(X, y_idx.astype(float), w)
    valid, callbacks = None, None
    if X_val is not None:
        valid = [_lgb_dataset(X_val, y_val.astype(float), w_val, reference=dtrain)]
        callbacks = [lgb.early_stopping(int(cfg["model"]["early_stopping_rounds"]), verbose=False)]
    rounds = int(num_rounds or cfg["model"]["max_rounds"])
    b = lgb.train(p, dtrain, num_boost_round=rounds, valid_sets=valid, callbacks=callbacks)
    return b, int(b.best_iteration or rounds)


def fit_binary(X: pd.DataFrame, y_bin: np.ndarray, params: dict[str, Any], cfg: dict, *, w: np.ndarray | None = None,
               X_val: pd.DataFrame | None = None, y_val: np.ndarray | None = None, w_val: np.ndarray | None = None,
               num_rounds: int | None = None) -> tuple[lgb.Booster, int]:
    p = {"objective": "binary", "metric": "binary_logloss", "verbosity": -1, "seed": 42, "feature_pre_filter": False,
         "num_threads": int(cfg["model"].get("num_threads", 0)) or None, **params}
    p = {k: v for k, v in p.items() if v is not None}
    dtrain = _lgb_dataset(X, y_bin.astype(float), w)
    valid, callbacks = None, None
    if X_val is not None:
        valid = [_lgb_dataset(X_val, y_val.astype(float), w_val, reference=dtrain)]
        callbacks = [lgb.early_stopping(int(cfg["model"]["early_stopping_rounds"]), verbose=False)]
    rounds = int(num_rounds or cfg["model"]["max_rounds"])
    b = lgb.train(p, dtrain, num_boost_round=rounds, valid_sets=valid, callbacks=callbacks)
    return b, int(b.best_iteration or rounds)


def ordinal_probabilities(p_ge_elev: np.ndarray, p_crit: np.ndarray) -> np.ndarray:
    """Combine the two ordinal heads into (n,3) probabilities with P(C) <= P(>=E) enforced by clipping."""
    pc = np.minimum(np.asarray(p_crit, float), np.asarray(p_ge_elev, float))
    pe = np.asarray(p_ge_elev, float) - pc
    ps = 1.0 - np.asarray(p_ge_elev, float)
    P = np.column_stack([ps, pe, pc])
    P = np.clip(P, 0.0, 1.0)
    s = P.sum(axis=1, keepdims=True)
    return np.where(s > 0, P / s, np.full_like(P, 1 / 3))


def critical_f2_from_probs(P: np.ndarray, y_idx: np.ndarray, t_crit: float = 0.5) -> dict[str, float | None]:
    pred_c = P[:, 2] >= t_crit
    true_c = y_idx == 2
    tp = int((pred_c & true_c).sum()); fp = int((pred_c & ~true_c).sum()); fn = int((~pred_c & true_c).sum())
    rec = tp / (tp + fn) if tp + fn else None
    prec = tp / (tp + fp) if tp + fp else None
    f2 = (5 * prec * rec / (4 * prec + rec)) if (rec is not None and prec is not None and (4 * prec + rec) > 0) else (0.0 if tp + fn else None)
    return {"critical_recall": rec, "critical_precision": prec, "critical_f2": f2}


def macro_f1_from_probs(P: np.ndarray, y_idx: np.ndarray) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y_idx, P.argmax(axis=1), labels=[0, 1, 2], average="macro", zero_division=0))


def grouped_oof_multiclass(X: pd.DataFrame, y_idx: np.ndarray, groups: np.ndarray, w: np.ndarray, params: dict[str, Any], cfg: dict,
                           *, n_folds: int, fit_mask: np.ndarray | None = None) -> tuple[np.ndarray, list[int]]:
    """Out-of-fold (n,3) probabilities with GroupKFold by event. `fit_mask` restricts the rows used for
    FITTING (e.g. latest-snapshot-only training) while predictions are produced for every row."""
    gkf = GroupKFold(n_splits=n_folds)
    oof = np.full((len(y_idx), 3), np.nan)
    iters: list[int] = []
    fm = np.ones(len(y_idx), dtype=bool) if fit_mask is None else fit_mask
    for tr_idx, va_idx in gkf.split(X, y_idx, groups):
        tr_fit = tr_idx[fm[tr_idx]]
        va_fit = va_idx[fm[va_idx]]
        b, best = fit_multiclass(X.iloc[tr_fit], y_idx[tr_fit], params, cfg, w=w[tr_fit], X_val=X.iloc[va_fit], y_val=y_idx[va_fit], w_val=w[va_fit])
        oof[va_idx] = b.predict(X.iloc[va_idx], num_iteration=best)
        iters.append(best)
    return oof, iters


def grouped_oof_ordinal(X: pd.DataFrame, y_idx: np.ndarray, groups: np.ndarray, w: np.ndarray, params: dict[str, Any], cfg: dict,
                        *, n_folds: int, fit_mask: np.ndarray | None = None) -> tuple[np.ndarray, list[int], list[int]]:
    gkf = GroupKFold(n_splits=n_folds)
    oof = np.full((len(y_idx), 3), np.nan)
    it_e: list[int] = []; it_c: list[int] = []
    fm = np.ones(len(y_idx), dtype=bool) if fit_mask is None else fit_mask
    y_ge = (y_idx >= 1).astype(int); y_c = (y_idx == 2).astype(int)
    for tr_idx, va_idx in gkf.split(X, y_idx, groups):
        tr_fit = tr_idx[fm[tr_idx]]; va_fit = va_idx[fm[va_idx]]
        be, ie = fit_binary(X.iloc[tr_fit], y_ge[tr_fit], params, cfg, w=w[tr_fit], X_val=X.iloc[va_fit], y_val=y_ge[va_fit], w_val=w[va_fit])
        bc, ic = fit_binary(X.iloc[tr_fit], y_c[tr_fit], params, cfg, w=w[tr_fit], X_val=X.iloc[va_fit], y_val=y_c[va_fit], w_val=w[va_fit])
        oof[va_idx] = ordinal_probabilities(be.predict(X.iloc[va_idx], num_iteration=ie), bc.predict(X.iloc[va_idx], num_iteration=ic))
        it_e.append(ie); it_c.append(ic)
    return oof, it_e, it_c


def multiclass_search(X: pd.DataFrame, y_idx: np.ndarray, groups: np.ndarray, w: np.ndarray, cfg: dict, *, eval_mask: np.ndarray,
                      n_search: int | None = None, seed: int = 42, fit_mask: np.ndarray | None = None) -> dict[str, Any]:
    """Random search with grouped OOF. Selection: OOF Critical F2 on `eval_mask` rows (latest snapshot,
    real_forecast events) at the 0.5 threshold, tie-break macro-F1."""
    n_search = int(n_search or cfg["model"]["n_search"])
    grid = sample_class_param_grid(cfg, n_search, seed)
    results = []
    for i, params in enumerate(grid):
        t0 = time.time()
        oof, iters = grouped_oof_multiclass(X, y_idx, groups, w, params, cfg, n_folds=int(cfg["split"]["cv_folds"]), fit_mask=fit_mask)
        sc = critical_f2_from_probs(oof[eval_mask], y_idx[eval_mask])
        sc["macro_f1"] = macro_f1_from_probs(oof[eval_mask], y_idx[eval_mask])
        results.append({"index": i, "params": params, "best_iterations": iters, "mean_best_iteration": int(np.mean(iters)),
                        "scores": sc, "seconds": round(time.time() - t0, 1)})
    results_sorted = sorted(results, key=lambda r: (-(r["scores"]["critical_f2"] or 0.0), -r["scores"]["macro_f1"]))
    best = results_sorted[0]
    return {"results": results, "best_index": best["index"], "best": best, "n_folds": int(cfg["split"]["cv_folds"]),
            "n_configs": len(results),
            "selection_rule": "max OOF Critical F2 on real_forecast latest snapshots (t=0.5), tie-break macro-F1"}


# ---- baselines -------------------------------------------------------------- #
def carry_forward_probs(tier_now: np.ndarray) -> np.ndarray:
    """One-hot 'probabilities' for the naive carry-forward baseline (NaN tier -> uniform)."""
    P = np.full((len(tier_now), 3), np.nan)
    for i, t in enumerate(TIERS):
        P[:, i] = (tier_now == t).astype(float)
    bad = ~np.isin(tier_now, TIERS)
    P[bad] = 1 / 3
    return P


def majority_probs(n: int) -> np.ndarray:
    P = np.zeros((n, 3)); P[:, 0] = 1.0
    return P


@dataclass
class LogisticBaseline:
    """Multinomial logistic regression on a few standardised features (median-imputed)."""
    columns: list[str] = field(default_factory=lambda: ["log_miss_distance", "max_risk_estimate", "log_combined_sigma_r", "time_to_tca", "risk", "slope_log_miss", "slope_risk"])
    medians_: dict[str, float] = field(default_factory=dict)
    means_: list[float] = field(default_factory=list)
    scales_: list[float] = field(default_factory=list)
    coef_: list[list[float]] = field(default_factory=list)
    intercept_: list[float] = field(default_factory=list)

    def _prep(self, X: pd.DataFrame) -> np.ndarray:
        cols = [c for c in self.columns if c in X.columns]
        Z = X[cols].astype(float).copy()
        for c in cols:
            Z[c] = Z[c].fillna(self.medians_.get(c, 0.0))
        A = Z.to_numpy()
        if self.means_:
            A = (A - np.asarray(self.means_)) / np.asarray(self.scales_)
        return A

    def fit(self, X: pd.DataFrame, y_idx: np.ndarray, w: np.ndarray | None = None) -> "LogisticBaseline":
        from sklearn.linear_model import LogisticRegression
        self.columns = [c for c in self.columns if c in X.columns]
        self.medians_ = {c: float(X[c].median()) for c in self.columns}
        A = self._prep(X)
        self.means_ = A.mean(axis=0).tolist(); self.scales_ = (A.std(axis=0) + 1e-9).tolist()
        A = (A - np.asarray(self.means_)) / np.asarray(self.scales_)
        m = LogisticRegression(max_iter=2000, C=1.0).fit(A, y_idx, sample_weight=w)
        self.coef_ = m.coef_.tolist(); self.intercept_ = m.intercept_.tolist()
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        A = self._prep(X)
        z = A @ np.asarray(self.coef_).T + np.asarray(self.intercept_)
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def to_dict(self) -> dict:
        return {"type": "logistic", "columns": self.columns, "medians": self.medians_, "means": self.means_, "scales": self.scales_,
                "coef": self.coef_, "intercept": self.intercept_}

    @classmethod
    def from_dict(cls, d: dict) -> "LogisticBaseline":
        return cls(columns=d["columns"], medians_=d["medians"], means_=d["means"], scales_=d["scales"], coef_=d["coef"], intercept_=d["intercept"])
