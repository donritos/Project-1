"""Event-level splits, baselines, and LightGBM model development.

All splitting is by event (never by row). Stratification uses the maximum
observed training-label tier per event.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, train_test_split

from .features import CATEGORICAL, feature_matrix
from .policy import assign_risk_tier

SPLIT_NAMES = ["dev", "policy_cal", "conformal_cal", "eval"]


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def event_table(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """One row per event with the max-severity training-label tier (for stratification)."""
    tgt = cfg["target_column"]
    tiers, _ = assign_risk_tier(df[tgt].to_numpy(dtype=float), df["miss_distance"].to_numpy(dtype=float), cfg["thresholds"])
    sev = pd.Series(tiers, index=df.index).map({"Critical": 2, "Elevated": 1, "Safe": 0}).fillna(0)
    ev = pd.DataFrame({"event_id": df["event_id"], "sev": sev, "risk": df[tgt]})
    g = ev.groupby("event_id").agg(max_sev=("sev", "max"), n_rows=("risk", "size"), max_risk=("risk", "max")).reset_index()
    g["strata"] = g["max_sev"].map({2: "Critical", 1: "Elevated", 0: "Safe"})
    # Rows with the highest-severity Pc-only events are rare; mark them so stratification keeps them spread
    g.loc[g["max_risk"] >= cfg["thresholds"]["critical_risk"], "strata"] = "Critical_pc"
    return g


def make_event_splits(df: pd.DataFrame, cfg: dict) -> dict[str, Any]:
    """Stratified event partition: dev / policy_cal / conformal_cal / eval. Returns dict with event lists and counts."""
    s = cfg["split"]
    seed = int(s["seed"])
    ev = event_table(df, cfg)
    fracs = {k: float(s[k]) for k in SPLIT_NAMES}
    total = sum(fracs.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split fractions must sum to 1, got {total}")
    remaining = ev.copy()
    out: dict[str, list[int]] = {}
    remaining_frac = 1.0
    for name in SPLIT_NAMES[:-1]:
        take = fracs[name] / remaining_frac
        strat = remaining["strata"]
        # merge strata with <2 members into the next most severe for stratification
        vc = strat.value_counts()
        strat = strat.where(strat.map(vc) >= 2, "Elevated")
        keep, chosen = train_test_split(remaining, test_size=take, random_state=seed, stratify=strat)
        out[name] = sorted(int(x) for x in chosen["event_id"])
        remaining = keep
        remaining_frac -= fracs[name]
    out[SPLIT_NAMES[-1]] = sorted(int(x) for x in remaining["event_id"])
    # sanity: disjoint and complete
    all_ids = [e for v in out.values() for e in v]
    assert len(all_ids) == len(set(all_ids)) == ev["event_id"].nunique()
    counts = {}
    for name, ids in out.items():
        sub = ev[ev["event_id"].isin(ids)]
        counts[name] = {"n_events": int(len(sub)), "n_rows": int(sub["n_rows"].sum()),
                        "strata": {str(k): int(v) for k, v in sub["strata"].value_counts().items()}}
    return {"seed": seed, "fractions": fracs, "events": out, "counts": counts, "split_hash": _hash_splits(out)}


def _hash_splits(events: dict[str, list[int]]) -> str:
    return hashlib.sha256(json.dumps(events, sort_keys=True).encode()).hexdigest()[:16]


def attach_split(df: pd.DataFrame, splits: dict[str, Any]) -> pd.Series:
    m: dict[int, str] = {}
    for name, ids in splits["events"].items():
        for e in ids:
            m[int(e)] = name
    return df["event_id"].map(m)


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
@dataclass
class MedianBaseline:
    median: float = float("nan")

    def fit(self, y: np.ndarray) -> "MedianBaseline":
        self.median = float(np.nanmedian(y))
        return self

    def predict(self, n: int) -> np.ndarray:
        return np.full(n, self.median)

    def to_dict(self) -> dict:
        return {"type": "median", "median": self.median}


@dataclass
class LinearBaseline:
    """OLS on verified log-space max_risk_estimate and log10 miss distance (imputed with training medians)."""
    columns: list[str] = field(default_factory=lambda: ["max_risk_estimate", "log_miss_distance"])
    coef_: list[float] = field(default_factory=list)
    intercept_: float = 0.0
    medians_: dict[str, float] = field(default_factory=dict)

    def _prep(self, X: pd.DataFrame) -> np.ndarray:
        Z = X[self.columns].astype(float).copy()
        for c in self.columns:
            Z[c] = Z[c].fillna(self.medians_.get(c, 0.0))
        return Z.to_numpy()

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "LinearBaseline":
        self.medians_ = {c: float(X[c].median()) for c in self.columns}
        m = LinearRegression().fit(self._prep(X), y)
        self.coef_ = [float(c) for c in m.coef_]
        self.intercept_ = float(m.intercept_)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._prep(X) @ np.asarray(self.coef_) + self.intercept_

    def to_dict(self) -> dict:
        return {"type": "linear", "columns": self.columns, "coef": self.coef_, "intercept": self.intercept_, "medians": self.medians_}


# --------------------------------------------------------------------------- #
# LightGBM
# --------------------------------------------------------------------------- #
def postprocess(pred: np.ndarray, cfg: dict) -> np.ndarray:
    """Clip point predictions to the valid log10(Pc) range [floor, 0]. Part of the frozen pipeline."""
    floor = float(cfg["model"]["target_floor"])
    out = np.asarray(pred, dtype=float).copy()
    ok = np.isfinite(out)
    out[ok] = np.clip(out[ok], floor, 0.0)
    return out


def risk_weights(y: np.ndarray, cfg: dict) -> np.ndarray:
    """Capped risk-aware sample weights: 1 for floor/low-risk rows, rising linearly to the cap
    for rows between -8 and -4, so the rare high-risk region is not drowned out."""
    cap = float(cfg["model"].get("weight_cap", 5.0))
    w = 1.0 + (cap - 1.0) * np.clip((np.asarray(y, dtype=float) + 8.0) / 4.0, 0.0, 1.0)
    return w


def base_params(objective: str, cfg: dict, seed: int = 42) -> dict[str, Any]:
    return {
        "objective": "regression_l1" if objective == "l1" else "regression",
        "metric": "l1" if objective == "l1" else "l2",
        "verbosity": -1,
        "seed": seed,
        "num_threads": int(cfg["model"].get("num_threads", 0)) or None,
        "feature_pre_filter": False,
    }


def sample_param_grid(cfg: dict, n: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    m = cfg["model"]
    grid = []
    for i in range(n):
        grid.append({
            "objective": m["objective_candidates"][i % len(m["objective_candidates"])],
            "learning_rate": float(rng.uniform(*m["learning_rate"])),
            "num_leaves": int(rng.integers(m["num_leaves"][0], m["num_leaves"][1] + 1)),
            "min_child_samples": int(rng.integers(m["min_child_samples"][0], m["min_child_samples"][1] + 1)),
            "feature_fraction": float(rng.uniform(*m["feature_fraction"])),
            "lambda_l2": float(rng.choice(m["lambda_l2"])),
        })
    return grid


def _lgb_dataset(X: pd.DataFrame, y: np.ndarray, w: np.ndarray | None = None, reference=None) -> lgb.Dataset:
    cats = [c for c in CATEGORICAL if c in X.columns]
    return lgb.Dataset(X, label=y, weight=w, categorical_feature=cats, reference=reference, free_raw_data=False)


def fit_lgb(X: pd.DataFrame, y: np.ndarray, params: dict[str, Any], cfg: dict, *, w: np.ndarray | None = None,
            X_val: pd.DataFrame | None = None, y_val: np.ndarray | None = None, w_val: np.ndarray | None = None,
            num_rounds: int | None = None, monotone: dict[str, int] | None = None) -> tuple[lgb.Booster, int]:
    p = {**base_params(params["objective"], cfg), **{k: v for k, v in params.items() if k != "objective"}}
    p = {k: v for k, v in p.items() if v is not None}
    if monotone:
        p["monotone_constraints"] = [int(monotone.get(c, 0)) for c in X.columns]
        p["monotone_constraints_method"] = "advanced"
    dtrain = _lgb_dataset(X, y, w)
    callbacks = []
    valid_sets = []
    if X_val is not None:
        valid_sets = [_lgb_dataset(X_val, y_val, w_val, reference=dtrain)]
        callbacks.append(lgb.early_stopping(int(cfg["model"]["early_stopping_rounds"]), verbose=False))
    rounds = int(num_rounds or cfg["model"]["max_rounds"])
    booster = lgb.train(p, dtrain, num_boost_round=rounds, valid_sets=valid_sets or None, callbacks=callbacks or None)
    best = int(booster.best_iteration or rounds)
    return booster, best


def _oof_scores(y: np.ndarray, oof: np.ndarray, cfg: dict) -> dict[str, float | None]:
    th = cfg["thresholds"]
    floor = cfg["model"]["target_floor"]
    nf = y > floor + 1e-9
    hi = y >= th["elevated_risk"]
    ae = np.abs(y - oof)
    tp = ((oof >= th["critical_risk"]) & (y >= th["critical_risk"])).sum()
    fp = ((oof >= th["critical_risk"]) & (y < th["critical_risk"])).sum()
    fn = ((oof < th["critical_risk"]) & (y >= th["critical_risk"])).sum()
    rec = tp / (tp + fn) if (tp + fn) else None
    prec = tp / (tp + fp) if (tp + fp) else None
    f2 = (5 * prec * rec / (4 * prec + rec)) if (prec is not None and rec is not None and (4 * prec + rec) > 0) else (0.0 if (tp + fn) else None)
    return {
        "mae": float(ae.mean()), "mae_non_floor": float(ae[nf].mean()) if nf.any() else None,
        "mae_high_risk": float(ae[hi].mean()) if hi.any() else None,
        "critical_recall_pc_only": None if rec is None else float(rec),
        "critical_precision_pc_only": None if prec is None else float(prec),
        "critical_f2_pc_only": None if f2 is None else float(f2),
    }


def grouped_cv_search(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, cfg: dict, *, weights: np.ndarray | None = None,
                      n_search: int | None = None, seed: int = 42, monotone: dict[str, int] | None = None) -> dict[str, Any]:
    """Random search with GroupKFold inside the development partition. Selection combines
    regression quality on non-floor rows with operational Pc-only Critical F2 (rank average)."""
    m = cfg["model"]
    n_search = int(n_search or m["n_search"])
    grid = sample_param_grid(cfg, n_search, seed)
    gkf = GroupKFold(n_splits=int(m["cv_folds"]))
    folds = list(gkf.split(X, y, groups))
    results = []
    for i, params in enumerate(grid):
        t0 = time.time()
        oof = np.full(len(y), np.nan)
        best_iters = []
        for tr_idx, va_idx in folds:
            booster, best = fit_lgb(X.iloc[tr_idx], y[tr_idx], params, cfg, w=None if weights is None else weights[tr_idx],
                                    X_val=X.iloc[va_idx], y_val=y[va_idx], monotone=monotone)
            oof[va_idx] = booster.predict(X.iloc[va_idx], num_iteration=best)
            best_iters.append(best)
        oof = postprocess(oof, cfg)
        sc = _oof_scores(y, oof, cfg)
        results.append({"index": i, "params": params, "best_iterations": best_iters, "mean_best_iteration": int(np.mean(best_iters)),
                        "scores": sc, "seconds": round(time.time() - t0, 1)})
    # rank-average selection
    df = pd.DataFrame([{**r["scores"], "index": r["index"]} for r in results])
    df["rank_mae_nf"] = df["mae_non_floor"].rank(ascending=True)
    df["rank_f2"] = df["critical_f2_pc_only"].fillna(0.0).rank(ascending=False)
    df["selection_score"] = 0.5 * df["rank_mae_nf"] + 0.5 * df["rank_f2"]
    best_index = int(df.sort_values(["selection_score", "mae_non_floor"]).iloc[0]["index"])
    for r in results:
        r["selection_score"] = float(df.loc[df["index"] == r["index"], "selection_score"].iloc[0])
    return {"results": results, "best_index": best_index, "best": results[best_index], "n_folds": len(folds),
            "selection_rule": "0.5*rank(MAE on non-floor rows) + 0.5*rank(Pc-only Critical F2), lower is better"}


def train_classifier(X: pd.DataFrame, y_bin: np.ndarray, cfg: dict, *, X_val: pd.DataFrame | None = None, y_val: np.ndarray | None = None,
                     seed: int = 42) -> tuple[lgb.Booster, int]:
    """Binary LightGBM for P(risk >= -4). Scores are calibrated separately (SigmoidCalibrator)."""
    pos = max(int(y_bin.sum()), 1)
    neg = max(int(len(y_bin) - y_bin.sum()), 1)
    p = {"objective": "binary", "metric": "average_precision", "verbosity": -1, "seed": seed, "learning_rate": 0.03,
         "num_leaves": 15, "min_child_samples": 100, "feature_fraction": 0.9, "lambda_l2": 10.0,
         "scale_pos_weight": min(neg / pos, 50.0), "feature_pre_filter": False}
    dtrain = _lgb_dataset(X, y_bin.astype(float))
    callbacks, valid = [], None
    if X_val is not None:
        valid = [_lgb_dataset(X_val, y_val.astype(float), reference=dtrain)]
        callbacks.append(lgb.early_stopping(int(cfg["model"]["early_stopping_rounds"]), verbose=False))
    booster = lgb.train(p, dtrain, num_boost_round=int(cfg["model"]["max_rounds"]), valid_sets=valid, callbacks=callbacks or None)
    return booster, int(booster.best_iteration or cfg["model"]["max_rounds"])


def default_monotone(feature_cols: list[str]) -> dict[str, int]:
    """Experimental constraints on individual transformed columns (see spec caveats)."""
    cons = {"max_risk_estimate": 1, "log_miss_distance": -1, "miss_distance": -1}
    return {c: cons[c] for c in feature_cols if c in cons}


def fit_quantile_heads(X: pd.DataFrame, y: np.ndarray, params: dict[str, Any], cfg: dict, *, num_rounds: int,
                       alphas: tuple[float, ...] = (0.05, 0.95)) -> dict[float, lgb.Booster]:
    """Quantile LightGBM heads sharing the selected params (no monotone constraints, unweighted)."""
    heads: dict[float, lgb.Booster] = {}
    for a in alphas:
        p = {**base_params("l1", cfg), **{k: v for k, v in params.items() if k != "objective"},
             "objective": "quantile", "alpha": float(a), "metric": "quantile"}
        p = {k: v for k, v in p.items() if v is not None}
        heads[float(a)] = lgb.train(p, _lgb_dataset(X, y), num_boost_round=int(num_rounds))
    return heads
