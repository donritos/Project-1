"""Shared fixtures: a small synthetic CDM dataset that mimics the delivered files."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from collision_risk.config import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config()


def make_synthetic_cdms(n_events: int = 160, seed: int = 0, dataset_id: str = "synth", with_risk: bool = True) -> pd.DataFrame:
    """Events with 3-12 CDMs spread over 0-7 days, plausible telemetry, a -30 floor, rare high-risk
    events, extra forbidden columns (mission_id, mahalanobis_distance), and a few invalid rows.
    Risk and miss distance drift so that the final tier differs from the early tier for some events."""
    rng = np.random.default_rng(seed)
    rows = []
    for e in range(n_events):
        k = int(rng.integers(3, 13))
        times = np.sort(rng.uniform(0.0, 7.0, size=k))[::-1]
        miss0 = float(10 ** rng.uniform(2.3, 4.7))
        drift = float(rng.normal(0, 0.08))  # per-day log10 drift in miss distance
        mre = float(rng.uniform(-9.5, -2.0))
        high = rng.random() < 0.10
        obj = rng.choice(["DEBRIS", "UNKNOWN", "PAYLOAD", "ROCKET BODY"])
        mission = int(rng.integers(1, 6))
        for t in times:
            miss = max(10.0, miss0 * 10 ** (drift * (7.0 - t)) * float(np.exp(rng.normal(0, 0.05))))
            r = rng.normal(0, 300); frac = rng.uniform(0.3, 0.9)
            tt = np.sqrt(max(miss ** 2 - r ** 2, 1.0)) * frac
            nn = np.sqrt(max(miss ** 2 - r ** 2 - tt ** 2, 0.0))
            cs = float(10 ** rng.uniform(0.5, 3.0)); ts = float(10 ** rng.uniform(0.0, 1.2))
            if high:
                risk = float(np.clip(rng.normal(-4.5, 1.0) + 0.3 * (7.0 - t) * rng.normal(0, 0.3), -8, -1.5))
            else:
                risk = -30.0 if rng.random() < 0.45 else float(np.clip(rng.normal(-12, 4), -29, -6.5))
            row = {
                "event_id": e, "time_to_tca": float(t), "mission_id": mission,
                "max_risk_estimate": mre + float(rng.normal(0, 0.2)), "max_risk_scaling": float(rng.uniform(0, 10)),
                "miss_distance": miss, "relative_speed": float(rng.uniform(100, 15000)),
                "relative_position_r": r, "relative_position_t": tt * rng.choice([-1, 1]), "relative_position_n": nn * rng.choice([-1, 1]),
                "c_object_type": obj, "c_sigma_r": cs, "t_sigma_r": ts,
                "F10": float(rng.uniform(66, 120)) if rng.random() > 0.05 else np.nan, "AP": float(rng.integers(0, 40)),
                "mahalanobis_distance": float(rng.uniform(0, 10)),  # forbidden column
            }
            if with_risk:
                row["risk"] = risk
            rows.append(row)
    df = pd.DataFrame(rows)
    df.loc[3, "miss_distance"] = np.nan
    df.loc[7, "relative_position_t"] = np.nan
    return df


@pytest.fixture(scope="session")
def synthetic_raw() -> pd.DataFrame:
    return make_synthetic_cdms()


@pytest.fixture(scope="session")
def synthetic_paths(synthetic_raw, tmp_path_factory):
    """Synthetic train/test CSVs on disk (test = separate events, only CDMs at >= 2 days, no final outcome needed)."""
    d = tmp_path_factory.mktemp("synth_data")
    train_p = d / "train_data.csv"
    test_p = d / "test_data.csv"
    synthetic_raw.to_csv(train_p, index=False)
    test = make_synthetic_cdms(n_events=40, seed=7)
    test = test[test["time_to_tca"] >= 2.0]
    test.to_csv(test_p, index=False)
    return {"dir": d, "train": train_p, "test": test_p}


@pytest.fixture(scope="session")
def synthetic_cfg(cfg, synthetic_paths):
    """Config pointing at the synthetic files, with quick model settings and its own artifact dir."""
    import copy
    c = copy.deepcopy(cfg)
    d = synthetic_paths["dir"]
    c["paths"].update({"raw_train": str(synthetic_paths["train"]), "raw_test": str(synthetic_paths["test"]),
                       "event_labels": str(d / "missing_labels.xlsx"), "processed_dir": str(d / "processed"), "artifacts_dir": str(d / "artifacts")})
    c["model"].update({"n_search": 1, "max_rounds": 60, "early_stopping_rounds": 10, "min_child_samples": [5, 10], "num_leaves": [4, 8]})
    c["split"]["cv_folds"] = 3
    return c


@pytest.fixture(scope="session")
def synthetic_allowlisted(synthetic_paths, cfg):
    from collision_risk.data import load_raw
    df, meta = load_raw(synthetic_paths["train"], "synth", cfg)
    return df, meta
