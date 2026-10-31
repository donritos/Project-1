"""Shared fixtures: a small synthetic CDM dataset that mimics the delivered files."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from collision_risk.config import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config()


def make_synthetic_cdms(n_events: int = 120, seed: int = 0, dataset_id: str = "synth") -> pd.DataFrame:
    """Events with 3-12 CDMs, decreasing time_to_tca, plausible telemetry, a -30 floor,
    some rare high-risk events, extra forbidden columns, and a few invalid rows."""
    rng = np.random.default_rng(seed)
    rows = []
    for e in range(n_events):
        k = int(rng.integers(3, 13))
        times = np.sort(rng.uniform(0.0, 7.0, size=k))[::-1]
        miss0 = float(10 ** rng.uniform(2.0, 4.7))
        mre = float(rng.uniform(-9.5, -2.0))
        high = rng.random() < 0.08
        obj = rng.choice(["DEBRIS", "UNKNOWN", "PAYLOAD", "ROCKET BODY"])
        for j, t in enumerate(times):
            miss = max(10.0, miss0 * float(np.exp(rng.normal(0, 0.15))))
            r = rng.normal(0, 300); frac = rng.uniform(0.3, 0.9)
            tt = np.sqrt(max(miss ** 2 - r ** 2, 1.0)) * frac
            nn = np.sqrt(max(miss ** 2 - r ** 2 - tt ** 2, 0.0))
            cs = float(10 ** rng.uniform(0.5, 3.0)); ts = float(10 ** rng.uniform(0.0, 1.2))
            if high:
                risk = float(np.clip(rng.normal(-4.5, 1.0), -8, -1.5))
            else:
                risk = -30.0 if rng.random() < 0.45 else float(np.clip(rng.normal(-12, 4), -29, -6.5))
            rows.append({
                "event_id": e, "time_to_tca": float(t), "mission_id": int(rng.integers(1, 5)), "risk": risk,
                "max_risk_estimate": mre + float(rng.normal(0, 0.2)), "max_risk_scaling": float(rng.uniform(0, 10)),
                "miss_distance": miss, "relative_speed": float(rng.uniform(100, 15000)),
                "relative_position_r": r, "relative_position_t": tt * rng.choice([-1, 1]), "relative_position_n": nn * rng.choice([-1, 1]),
                "c_object_type": obj, "c_sigma_r": cs, "t_sigma_r": ts,
                "F10": float(rng.uniform(66, 120)) if rng.random() > 0.05 else np.nan, "AP": float(rng.integers(0, 40)),
                "mahalanobis_distance": float(rng.uniform(0, 10)),  # forbidden column
            })
    df = pd.DataFrame(rows)
    # a couple of invalid rows: missing miss_distance / NaN geometry
    df.loc[3, "miss_distance"] = np.nan
    df.loc[7, "relative_position_t"] = np.nan
    return df


@pytest.fixture(scope="session")
def synthetic_raw() -> pd.DataFrame:
    return make_synthetic_cdms()


@pytest.fixture(scope="session")
def synthetic_allowlisted(synthetic_raw, cfg, tmp_path_factory):
    from collision_risk.data import load_raw
    p = tmp_path_factory.mktemp("data") / "synth.csv"
    synthetic_raw.to_csv(p, index=False)
    df, meta = load_raw(p, "synth", cfg)
    return df, meta
