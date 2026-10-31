"""Tests for the official tier rules and triage policy."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from collision_risk.config import load_config
from collision_risk.contracts import ReasonCode, RiskTier, TriageAction
from collision_risk.policy import (
    TIER_SEVERITY,
    apply_policy,
    assign_risk_tier,
    assign_triage,
    tier_from_true_risk,
)

CRIT = RiskTier.CRITICAL.value
ELEV = RiskTier.ELEVATED.value
SAFE = RiskTier.SAFE.value


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config()


@pytest.fixture(scope="module")
def thresholds(cfg) -> dict:
    return cfg["thresholds"]


def _tier(risk, miss, thresholds):
    tier, reasons = assign_risk_tier(np.array([risk]), np.array([miss]), thresholds)
    return tier[0], reasons[0]


def _row(**overrides) -> pd.DataFrame:
    base = dict(
        prediction_status="valid",
        risk_log10_pred=-8.0,
        miss_distance_m=20000.0,
        risk_log10_upper=-7.0,
        p_pc_critical=0.01,
        quality_flags=[],
    )
    base.update(overrides)
    return pd.DataFrame([base])


def _triage(cfg, **overrides) -> pd.Series:
    return assign_triage(_row(**overrides), cfg).iloc[0]


# --------------------------------------------------------------------------- #
# Tier rules: exact boundaries
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "risk, miss, expected",
    [
        (-4.0, 10000.0, CRIT),         # risk >= -4 inclusive
        (-4.000001, 10000.0, ELEV),    # just below critical risk
        (-6.0, 10000.0, ELEV),         # risk >= -6 inclusive
        (-6.000001, 10000.0, SAFE),    # just below elevated risk
        (-10.0, 999.999, CRIT),        # miss < 1000
        (-10.0, 1000.0, ELEV),         # 1000 m is Elevated (band inclusive)
        (-10.0, 5000.0, ELEV),         # 5000 m is Elevated (band inclusive)
        (-10.0, 5000.001, SAFE),       # just outside band
    ],
)
def test_tier_boundaries(risk, miss, expected, thresholds):
    tier, _ = _tier(risk, miss, thresholds)
    assert tier == expected


def test_tier_reason_codes_match_rule(thresholds):
    assert _tier(-4.0, 10000.0, thresholds)[1] == [ReasonCode.PC_CRITICAL.value]
    assert _tier(-10.0, 500.0, thresholds)[1] == [ReasonCode.DISTANCE_CRITICAL.value]
    assert _tier(-5.0, 10000.0, thresholds)[1] == [ReasonCode.PC_ELEVATED.value]
    assert _tier(-10.0, 3000.0, thresholds)[1] == [ReasonCode.DISTANCE_ELEVATED.value]
    assert _tier(-10.0, 10000.0, thresholds)[1] == [ReasonCode.SAFE_RULE.value]


def test_most_severe_rule_wins_with_both_codes(thresholds):
    # Both critical conditions fire.
    tier, codes = _tier(-3.0, 500.0, thresholds)
    assert tier == CRIT
    assert set(codes) == {ReasonCode.PC_CRITICAL.value, ReasonCode.DISTANCE_CRITICAL.value}

    # Critical risk with elevated-band distance -> Critical, only critical code.
    tier, codes = _tier(-3.0, 3000.0, thresholds)
    assert tier == CRIT
    assert codes == [ReasonCode.PC_CRITICAL.value]

    # Elevated risk with critical distance -> Critical via distance.
    tier, codes = _tier(-5.0, 500.0, thresholds)
    assert tier == CRIT
    assert codes == [ReasonCode.DISTANCE_CRITICAL.value]

    # Both elevated conditions fire.
    tier, codes = _tier(-5.0, 3000.0, thresholds)
    assert tier == ELEV
    assert set(codes) == {ReasonCode.PC_ELEVATED.value, ReasonCode.DISTANCE_ELEVATED.value}


def test_tier_severity_ordering():
    assert TIER_SEVERITY[CRIT] > TIER_SEVERITY[ELEV] > TIER_SEVERITY[SAFE]


# --------------------------------------------------------------------------- #
# Tier rules: NaN / invalid handling
# --------------------------------------------------------------------------- #
def test_nan_risk_with_close_miss_is_critical_via_distance(thresholds):
    tier, codes = _tier(np.nan, 500.0, thresholds)
    assert tier == CRIT
    assert codes == [ReasonCode.DISTANCE_CRITICAL.value]


def test_nan_risk_with_band_miss_is_elevated_via_distance(thresholds):
    tier, codes = _tier(np.nan, 3000.0, thresholds)
    assert tier == ELEV
    assert codes == [ReasonCode.DISTANCE_ELEVATED.value]


def test_nan_risk_with_far_miss_is_none_not_safe(thresholds):
    tier, codes = _tier(np.nan, 10000.0, thresholds)
    assert tier is None
    assert tier != SAFE
    assert codes == [ReasonCode.INVALID_INPUT.value]


@pytest.mark.parametrize("miss", [np.nan, -1.0, -1e-9])
def test_invalid_miss_distance_is_none(miss, thresholds):
    tier, codes = _tier(-10.0, miss, thresholds)
    assert tier is None
    assert codes == [ReasonCode.INVALID_INPUT.value]
    # Even a critical risk cannot be tiered without a valid miss distance.
    tier, _ = _tier(-2.0, miss, thresholds)
    assert tier is None


def test_assign_risk_tier_returns_object_array_and_lists(thresholds):
    tier, reasons = assign_risk_tier(
        np.array([-3.0, -5.0, -8.0, np.nan]),
        np.array([500.0, 3000.0, 10000.0, 10000.0]),
        thresholds,
    )
    assert tier.dtype == object
    assert list(tier) == [CRIT, ELEV, SAFE, None]
    assert isinstance(reasons, list) and len(reasons) == 4
    assert all(isinstance(r, list) for r in reasons)


def test_tier_from_true_risk_agrees_with_assign_risk_tier(thresholds):
    rng = np.random.default_rng(0)
    n = 5000
    risk = rng.uniform(-12.0, -1.0, n)
    miss = rng.uniform(0.0, 50000.0, n)
    # Inject boundaries and NaNs.
    risk[:4] = [-4.0, -6.0, np.nan, np.nan]
    miss[:4] = [10000.0, 10000.0, 500.0, 10000.0]
    miss[4:6] = [1000.0, 5000.0]
    risk[6] = -10.0
    miss[6] = np.nan

    expected, _ = assign_risk_tier(risk, miss, thresholds)
    got = tier_from_true_risk(risk, miss, thresholds)
    assert got.dtype == object
    assert list(got) == list(expected)


# --------------------------------------------------------------------------- #
# Triage policy
# --------------------------------------------------------------------------- #
def test_invalid_status_gives_insufficient_data_and_tier_none(cfg):
    row = _triage(
        cfg,
        prediction_status="invalid",
        risk_log10_pred=np.nan,
        miss_distance_m=np.nan,
        risk_log10_upper=np.nan,
        quality_flags=["MISSING_REQUIRED_GEOMETRY"],
    )
    assert row["risk_tier"] is None
    assert row["triage_action"] == TriageAction.HUMAN_REVIEW_INSUFFICIENT_DATA.value
    assert ReasonCode.INVALID_INPUT.value in row["reason_codes"]


def test_invalid_status_overrides_even_critical_geometry(cfg):
    # Even if the geometry alone looks Critical, an invalid prediction gets no tier.
    row = _triage(cfg, prediction_status="invalid", risk_log10_pred=-2.0, miss_distance_m=100.0)
    assert row["risk_tier"] is None
    assert row["triage_action"] == TriageAction.HUMAN_REVIEW_INSUFFICIENT_DATA.value


def test_nan_risk_far_miss_is_insufficient_data_not_safe(cfg):
    row = _triage(cfg, prediction_status="warning", risk_log10_pred=np.nan, risk_log10_upper=np.nan)
    assert row["risk_tier"] is None
    assert row["triage_action"] == TriageAction.HUMAN_REVIEW_INSUFFICIENT_DATA.value
    assert row["triage_action"] != TriageAction.AUTO_CLEAR_CANDIDATE.value


def test_critical_tier_gives_maneuver_planning_review(cfg):
    row = _triage(cfg, risk_log10_pred=-3.0, risk_log10_upper=-2.0, p_pc_critical=0.99)
    assert row["risk_tier"] == CRIT
    assert row["triage_action"] == TriageAction.MANEUVER_PLANNING_REVIEW.value
    # Critical rows do not get MODEL_DISAGREEMENT (classifier agrees).
    assert ReasonCode.MODEL_DISAGREEMENT.value not in row["reason_codes"]
    assert ReasonCode.PC_CRITICAL.value in row["reason_codes"]


def test_missing_interval_on_safe_row_is_human_review(cfg):
    row = _triage(cfg, risk_log10_upper=np.nan)
    assert row["risk_tier"] == SAFE
    assert row["triage_action"] == TriageAction.HUMAN_REVIEW.value
    assert ReasonCode.INTERVAL_UNAVAILABLE.value in row["reason_codes"]
    assert ReasonCode.SAFE_RULE.value in row["reason_codes"]


def test_upper_bound_critical_on_elevated_row_is_human_review(cfg):
    row = _triage(cfg, risk_log10_pred=-5.0, risk_log10_upper=-4.0)
    assert row["risk_tier"] == ELEV
    assert row["triage_action"] == TriageAction.HUMAN_REVIEW.value
    assert ReasonCode.UPPER_BOUND_CRITICAL.value in row["reason_codes"]
    assert ReasonCode.PC_ELEVATED.value in row["reason_codes"]


def test_elevated_row_with_benign_interval_is_monitor(cfg):
    row = _triage(cfg, risk_log10_pred=-5.0, risk_log10_upper=-4.5)
    assert row["risk_tier"] == ELEV
    assert row["triage_action"] == TriageAction.MONITOR.value


def test_classifier_disagreement_is_human_review(cfg):
    row = _triage(cfg, risk_log10_pred=-5.0, risk_log10_upper=-4.5, p_pc_critical=0.6)
    assert row["risk_tier"] == ELEV
    assert row["triage_action"] == TriageAction.HUMAN_REVIEW.value
    assert ReasonCode.MODEL_DISAGREEMENT.value in row["reason_codes"]

    # Also on a Safe row that would otherwise auto-clear.
    row = _triage(cfg, p_pc_critical=0.5)
    assert row["risk_tier"] == SAFE
    assert row["triage_action"] == TriageAction.HUMAN_REVIEW.value
    assert ReasonCode.MODEL_DISAGREEMENT.value in row["reason_codes"]


def test_safe_row_auto_clear_candidate(cfg):
    row = _triage(cfg, risk_log10_upper=-6.5)
    assert row["risk_tier"] == SAFE
    assert row["triage_action"] == TriageAction.AUTO_CLEAR_CANDIDATE.value
    assert row["reason_codes"] == [ReasonCode.SAFE_RULE.value]


def test_safe_row_upper_at_or_above_minus_six_is_human_review(cfg):
    row = _triage(cfg, risk_log10_upper=-6.0)
    assert row["risk_tier"] == SAFE
    assert row["triage_action"] == TriageAction.HUMAN_REVIEW.value
    assert ReasonCode.UPPER_BOUND_ELEVATED.value in row["reason_codes"]


def test_allowed_quality_flag_still_auto_clears(cfg):
    row = _triage(cfg, prediction_status="warning", quality_flags=["MISSING_SPACE_WEATHER"])
    assert row["triage_action"] == TriageAction.AUTO_CLEAR_CANDIDATE.value
    assert ReasonCode.QUALITY_WARNING.value not in row["reason_codes"]


def test_blocking_quality_flag_forces_human_review_with_codes(cfg):
    row = _triage(cfg, prediction_status="warning", quality_flags=["MISSING_SIGMA"])
    assert row["risk_tier"] == SAFE
    assert row["triage_action"] == TriageAction.HUMAN_REVIEW.value
    assert ReasonCode.QUALITY_WARNING.value in row["reason_codes"]
    assert ReasonCode.MISSING_TELEMETRY.value in row["reason_codes"]


def test_non_missing_blocking_flag_has_no_missing_telemetry(cfg):
    row = _triage(cfg, prediction_status="warning", quality_flags=["SUSPICIOUS_SIGMA"])
    assert row["triage_action"] == TriageAction.HUMAN_REVIEW.value
    assert ReasonCode.QUALITY_WARNING.value in row["reason_codes"]
    assert ReasonCode.MISSING_TELEMETRY.value not in row["reason_codes"]


def test_valid_status_with_flags_still_acceptable_quality(cfg):
    # Acceptable quality = status valid OR every flag allow-listed.
    row = _triage(cfg, prediction_status="valid", quality_flags=["MISSING_SIGMA"])
    assert row["triage_action"] == TriageAction.AUTO_CLEAR_CANDIDATE.value


def test_p_pc_critical_column_absent_is_tolerated(cfg):
    df = _row().drop(columns=["p_pc_critical"])
    out = assign_triage(df, cfg)
    assert out.iloc[0]["triage_action"] == TriageAction.AUTO_CLEAR_CANDIDATE.value


def test_reason_codes_deduplicated_and_stable(cfg):
    df = pd.concat([_row(risk_log10_upper=np.nan)] * 3, ignore_index=True)
    out = assign_triage(df, cfg)
    for codes in out["reason_codes"]:
        assert len(codes) == len(set(codes))
        assert codes == out["reason_codes"].iloc[0]


def test_input_not_mutated_and_order_preserved(cfg):
    df = pd.DataFrame(
        {
            "prediction_status": ["valid", "invalid", "warning", "valid", "valid"],
            "risk_log10_pred": [-8.0, np.nan, -5.0, -3.0, -8.0],
            "miss_distance_m": [20000.0, np.nan, 3000.0, 500.0, 20000.0],
            "risk_log10_upper": [-7.0, np.nan, -4.5, -2.0, np.nan],
            "p_pc_critical": [0.01, np.nan, 0.1, 0.9, 0.0],
            "quality_flags": [[], ["MISSING_REQUIRED_GEOMETRY"], [], [], ["MISSING_SIGMA"]],
        },
        index=[10, 3, 7, 1, 5],
    )
    snapshot = df.copy(deep=True)

    out = assign_triage(df, cfg)

    assert out is not df
    pd.testing.assert_frame_equal(df, snapshot)
    assert set(["risk_tier", "triage_action", "reason_codes"]).isdisjoint(df.columns)
    assert list(out.index) == list(df.index)
    assert list(out["risk_tier"]) == [SAFE, None, ELEV, CRIT, SAFE]
    assert list(out["triage_action"]) == [
        TriageAction.AUTO_CLEAR_CANDIDATE.value,
        TriageAction.HUMAN_REVIEW_INSUFFICIENT_DATA.value,
        TriageAction.MONITOR.value,
        TriageAction.MANEUVER_PLANNING_REVIEW.value,
        TriageAction.HUMAN_REVIEW.value,
    ]
    assert out["triage_action"].map(type).eq(str).all()
    assert out["reason_codes"].map(lambda c: isinstance(c, list)).all()


def test_apply_policy_matches_assign_triage(cfg):
    df = pd.concat(
        [_row(), _row(risk_log10_pred=-5.0), _row(prediction_status="invalid", risk_log10_pred=np.nan)],
        ignore_index=True,
    )
    pd.testing.assert_frame_equal(apply_policy(df, cfg), assign_triage(df, cfg))


def test_vectorised_bulk_is_consistent_with_per_row(cfg):
    rng = np.random.default_rng(1)
    n = 2000
    df = pd.DataFrame(
        {
            "prediction_status": rng.choice(["valid", "warning", "invalid"], n, p=[0.7, 0.2, 0.1]),
            "risk_log10_pred": rng.uniform(-12.0, -1.0, n),
            "miss_distance_m": rng.uniform(0.0, 30000.0, n),
            "risk_log10_upper": rng.uniform(-10.0, 0.0, n),
            "p_pc_critical": rng.uniform(0.0, 1.0, n),
            "quality_flags": [
                list(rng.choice(["MISSING_SIGMA", "MISSING_SPACE_WEATHER"], size=rng.integers(0, 3)))
                for _ in range(n)
            ],
        }
    )
    df.loc[df["prediction_status"] == "invalid", ["risk_log10_pred", "risk_log10_upper"]] = np.nan
    df.loc[rng.choice(n, 50, replace=False), "risk_log10_upper"] = np.nan

    bulk = assign_triage(df, cfg)
    for i in range(0, n, 97):
        single = assign_triage(df.iloc[[i]], cfg).iloc[0]
        assert bulk.iloc[i]["risk_tier"] == single["risk_tier"]
        assert bulk.iloc[i]["triage_action"] == single["triage_action"]
        assert bulk.iloc[i]["reason_codes"] == single["reason_codes"]

    # Invariants: invalid never Safe; missing interval never auto-clears.
    invalid = bulk["prediction_status"] == "invalid"
    assert (bulk.loc[invalid, "risk_tier"].isna()).all()
    no_interval = bulk["risk_log10_upper"].isna()
    assert not (bulk.loc[no_interval, "triage_action"] == TriageAction.AUTO_CLEAR_CANDIDATE.value).any()
