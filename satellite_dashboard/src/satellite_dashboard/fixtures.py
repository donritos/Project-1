"""Synthetic, contract-matching fixtures. They are written in the MODEL-NATIVE format so they flow through the
same adapter as real artifacts, and their manifest is flagged synthetic so the UI shows a persistent banner.
Nothing here is model output or a measured result."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_ID = "synthetic"
MODEL_VERSION = "synthetic-fixture-1"

# (event_id, scenario, predicted final tier, observed final tier, n eligible CDMs, n excluded CDMs)
SCENARIOS = [
    (101, "rising", "Critical", "Critical", 6, 3),
    (102, "rising", "Critical", "Elevated", 5, 2),      # false alarm
    (103, "single", "Critical", "Critical", 1, 0),      # one eligible CDM, no history
    (201, "steady_elev", "Elevated", "Elevated", 5, 3),
    (202, "steady_elev", "Elevated", "Critical", 4, 4),  # escalation the model only put on watch
    (203, "falling", "Elevated", "Safe", 6, 2),
    (301, "safe", "Safe", "Safe", 6, 3),
    (302, "safe", "Safe", "Safe", 3, 1),
    (303, "safe", "Safe", "Critical", 4, 2),             # missed Critical
    (401, "invalid", None, "Elevated", 3, 1),            # latest eligible CDM has invalid geometry
]
INELIGIBLE = [(501, 5)]                                   # every CDM is inside the 2-day cutoff

ACTIONS = {"Critical": "Maneuver planning review", "Elevated": "Monitor", "Safe": "Auto-clear candidate"}


def _probs(tier: str, step: float) -> tuple[float, float, float]:
    base = {"Critical": (0.05, 0.25, 0.70), "Elevated": (0.20, 0.65, 0.15), "Safe": (0.92, 0.07, 0.01)}[tier]
    s, e, c = base
    c = min(0.97, c * (0.55 + 0.45 * step)) if tier == "Critical" else c
    s = s if tier != "Safe" else min(0.995, s + 0.05 * step)
    rest = 1.0 - c
    tot = s + e
    return round(s / tot * rest, 4), round(e / tot * rest, 4), round(c, 4)


def build(out_dir: Path) -> Path:
    rng = np.random.default_rng(7)
    snaps, inputs, events, outcomes = [], [], [], []
    for ev_id, scen, pred, obs, n, n_excl in SCENARIOS:
        times = np.round(np.linspace(6.5, 2.2, n) if n > 1 else np.array([2.6]), 3)
        miss0 = {"rising": 4200, "single": 850, "steady_elev": 3200, "falling": 2600, "safe": 21000, "invalid": 2900}[scen]
        for i, t in enumerate(times):
            step = (i + 1) / n
            shrink = {"rising": 1 - 0.75 * step, "falling": 1 + 1.2 * step}.get(scen, 1 - 0.1 * step)
            miss = float(max(120.0, miss0 * shrink * (1 + rng.normal(0, 0.03))))
            u = rng.normal(size=3); u /= np.linalg.norm(u)
            r, tt, nn = (miss * u).round(1)
            risk = {"rising": -7.5 + 4.0 * step, "single": -4.2, "steady_elev": -5.2, "falling": -5.0 - 2.5 * step,
                    "safe": -30.0 if i % 2 else -12.0, "invalid": -5.5}[scen]
            invalid = scen == "invalid" and i == n - 1
            row_id = f"{DATASET_ID}:{ev_id}-{i}"
            inputs.append({"dataset_id": DATASET_ID, "event_id": ev_id, "row_id": row_id, "time_to_tca": float(t),
                           "miss_distance": np.nan if invalid else round(miss, 1), "relative_speed": float(round(rng.uniform(300, 14500), 1)),
                           "relative_position_r": np.nan if invalid else r, "relative_position_t": tt, "relative_position_n": nn,
                           "c_object_type": "DEBRIS" if ev_id % 2 else "UNKNOWN", "c_sigma_r": float(round(rng.uniform(20, 400), 1)),
                           "t_sigma_r": float(round(rng.uniform(5, 60), 1)), "max_risk_estimate": round(risk + 1.2, 3) if risk > -30 else -8.0,
                           "F10": float(round(rng.uniform(70, 160), 1)), "AP": float(round(rng.uniform(3, 40), 1)), "risk": round(risk, 3)})
            cur = "Critical" if (risk >= -4 or miss < 1000) else ("Elevated" if (risk >= -6 or miss <= 5000) else "Safe")
            tier_now = pred if pred else "Elevated"
            ps, pe, pcrit = _probs(tier_now, step)
            status = "invalid" if invalid else ("warning" if n == 1 else "valid")
            action = "Human review - insufficient data" if invalid else ACTIONS[tier_now]
            if ev_id == 202 and i == n - 1:
                action = "Escalation watch"
            if ev_id == 303 and i == n - 1:
                action = "Human review"
            codes = ([f"FORECAST_{tier_now.upper()}"] if not invalid else ["INVALID_INPUT"]) + (["NO_HISTORY"] if n == 1 else []) + \
                    (["MISS_SHRINKING", "RISK_RISING"] if scen == "rising" else [])
            snaps.append({"dataset_id": DATASET_ID, "event_id": ev_id, "row_id": row_id, "time_to_tca_days": float(t),
                          "miss_distance_m": np.nan if invalid else round(miss, 1), "n_cdms_2d": i + 1, "is_latest_snapshot": i == n - 1,
                          "current_tier": None if invalid else cur, "carry_forward_tier": None if invalid else cur,
                          "prediction_status": status, "quality_flags": ["MISSING_REQUIRED_GEOMETRY"] if invalid else [],
                          "p_safe": np.nan if invalid else ps, "p_elevated": np.nan if invalid else pe, "p_critical": np.nan if invalid else pcrit,
                          "predicted_final_tier": None if invalid else tier_now, "prediction_set": [], "triage_action": action,
                          "reason_codes": codes, "explanation_id": None, "object_type": "DEBRIS" if ev_id % 2 else "UNKNOWN",
                          "model_version": MODEL_VERSION})
        last = snaps[-1].copy()
        last["n_rows_excluded_by_horizon"] = n_excl
        events.append(last)
        outcomes.append({"dataset_id": DATASET_ID, "event_id": ev_id, "final_tier": obs, "final_risk_log10_pc": np.nan,
                         "final_miss_distance_m": np.nan, "final_time_to_tca_days": round(float(rng.uniform(0.05, 1.5)), 3),
                         "cdm_count": n + n_excl, "cdm_count_2d_or_more": n, "is_trivial": False})
    for ev_id, n_excl in INELIGIBLE:
        outcomes.append({"dataset_id": DATASET_ID, "event_id": ev_id, "final_tier": "Elevated", "final_risk_log10_pc": np.nan,
                         "final_miss_distance_m": np.nan, "final_time_to_tca_days": 0.4, "cdm_count": n_excl,
                         "cdm_count_2d_or_more": 0, "is_trivial": False})

    d = Path(out_dir) / "predictions" / DATASET_ID
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(snaps).to_parquet(d / "snapshots.parquet", index=False)
    pd.DataFrame(events).to_parquet(d / "events.parquet", index=False)
    pd.DataFrame(inputs).to_parquet(d / "inputs.parquet", index=False)
    pd.DataFrame(outcomes).to_parquet(d / "outcomes.parquet", index=False)
    pd.DataFrame([{"dataset_id": DATASET_ID, "event_id": e, "n_excluded_cdms": n} for e, n in INELIGIBLE]).to_parquet(
        d / "ineligible_events.parquet", index=False)
    manifest = {
        "schema_version": "dashboard-1.0", "synthetic": True,
        "model": {"model_version": MODEL_VERSION, "kind": "synthetic", "use_past_risk": True},
        "eligibility": {"min_time_to_tca_days": 2.0, "rule": "time_to_tca >= 2 days"},
        "tier_rules": {}, "split": {"method": "synthetic fixtures - no split"},
        "capabilities": {"class_probabilities": True, "class_probabilities_note": "Synthetic values, not model output.",
                         "feature_contributions": False, "feature_contributions_note": "Not provided by the synthetic fixtures.",
                         "reason_codes": True, "review_recommendation": True, "snapshot_history": True,
                         "uncertainty_intervals": False, "prediction_sets": False, "single_cdm_inference": False},
        "datasets": {DATASET_ID: {"label": "Synthetic demo data", "dataset_id": DATASET_ID, "has_outcomes": True,
                                  "n_events": len(SCENARIOS) + len(INELIGIBLE),
                                  "note": "Hand-made fixtures covering Critical, Elevated, Safe, invalid-input and insufficient-data cases."}},
    }
    (Path(out_dir) / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return Path(out_dir)
