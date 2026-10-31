"""Tests for the official tier rules and the event-forecast triage policy."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from collision_risk.config import load_config
from collision_risk.contracts import ReasonCode, RiskTier, TriageAction
from collision_risk.policy import (
    TIER_SEVERITY,
    assign_risk_tier,
    assign_triage,
    tier_from_true_risk,
    tune_policy_thresholds,
)

CRIT = RiskTier.CRITICAL.value
ELEV = RiskTier.ELEVATED.value
SAFE = RiskTier.SAFE.value

RC = ReasonCode
TA = TriageAction

T_WATCH = 0.30
T_CLEAR = 0.80
PARAMS = {"t_watch": T_WATCH, "t_clear": T_CLEAR}


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config()


@pytest.fixture(scope="module")
def thresholds(cfg) -> dict:
    return cfg["thresholds"]


def _tier(risk, miss, thresholds):
    tier, reasons = assign_risk_tier(np.array([risk]), np.array([miss]), thresholds)
    return tier[0], reasons[0]


def _probs(p_safe: float, p_critical: float) -> dict:
    p_elev = 1.0 - p_safe - p_critical
    return dict(p_safe=p_safe, p_elevated=p_elev, p_critical=p_critical)


def _row(**overrides) -> pd.DataFrame:
    """A clean auto-clear-eligible event: predicted Safe, confident, current Safe, valid."""
    base = dict(
        prediction_status="valid",
        quality_flags=[],
        predicted_final_tier=SAFE,
        current_tier=SAFE,
        carry_forward_tier=SAFE,
        prediction_set=[SAFE],
        has_history=1,
        slope_log_miss=0.0,
        slope_risk_or_mre=0.0,
        slope_log_sigma=0.0,
        volatility_mre=0.0,
        volatility_risk=0.0,
        **_probs(0.95, 0.01),
    )
    base.update(overrides)
    return pd.DataFrame([base])


def _triage(cfg, params=PARAMS, **overrides) -> pd.Series:
    return assign_triage(_row(**overrides), cfg, params).iloc[0]


# --------------------------------------------------------------------------- #
# Tier rules: exact boundaries (official rules, unchanged)
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
    assert _tier(-4.0, 10000.0, thresholds)[1] == [RC.PC_CRITICAL.value]
    assert _tier(-10.0, 500.0, thresholds)[1] == [RC.DISTANCE_CRITICAL.value]
    assert _tier(-5.0, 10000.0, thresholds)[1] == [RC.PC_ELEVATED.value]
    assert _tier(-10.0, 3000.0, thresholds)[1] == [RC.DISTANCE_ELEVATED.value]
    assert _tier(-10.0, 10000.0, thresholds)[1] == [RC.SAFE_RULE.value]


def test_most_severe_rule_wins_with_both_codes(thresholds):
    # Both critical conditions fire.
    tier, codes = _tier(-3.0, 500.0, thresholds)
    assert tier == CRIT
    assert set(codes) == {RC.PC_CRITICAL.value, RC.DISTANCE_CRITICAL.value}

    # Critical risk with elevated-band distance -> Critical, only critical code.
    tier, codes = _tier(-3.0, 3000.0, thresholds)
    assert tier == CRIT
    assert codes == [RC.PC_CRITICAL.value]

    # Elevated risk with critical distance -> Critical via distance.
    tier, codes = _tier(-5.0, 500.0, thresholds)
    assert tier == CRIT
    assert codes == [RC.DISTANCE_CRITICAL.value]

    # Both elevated conditions fire.
    tier, codes = _tier(-5.0, 3000.0, thresholds)
    assert tier == ELEV
    assert set(codes) == {RC.PC_ELEVATED.value, RC.DISTANCE_ELEVATED.value}


def test_tier_severity_ordering():
    assert TIER_SEVERITY[CRIT] > TIER_SEVERITY[ELEV] > TIER_SEVERITY[SAFE]


# --------------------------------------------------------------------------- #
# Tier rules: NaN / invalid handling
# --------------------------------------------------------------------------- #
def test_nan_risk_with_close_miss_is_critical_via_distance(thresholds):
    tier, codes = _tier(np.nan, 500.0, thresholds)
    assert tier == CRIT
    assert codes == [RC.DISTANCE_CRITICAL.value]


def test_nan_risk_with_band_miss_is_elevated_via_distance(thresholds):
    tier, codes = _tier(np.nan, 3000.0, thresholds)
    assert tier == ELEV
    assert codes == [RC.DISTANCE_ELEVATED.value]


def test_nan_risk_with_far_miss_is_none_not_safe(thresholds):
    tier, codes = _tier(np.nan, 10000.0, thresholds)
    assert tier is None
    assert tier != SAFE
    assert codes == [RC.INVALID_INPUT.value]


@pytest.mark.parametrize("miss", [np.nan, -1.0, -1e-9])
def test_invalid_miss_distance_is_none(miss, thresholds):
    tier, codes = _tier(-10.0, miss, thresholds)
    assert tier is None
    assert codes == [RC.INVALID_INPUT.value]
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
# Event triage: rule 1 - insufficient data
# --------------------------------------------------------------------------- #
def test_invalid_status_is_insufficient_data_never_safe(cfg):
    row = _triage(
        cfg,
        prediction_status="invalid",
        predicted_final_tier=None,
        current_tier=None,
        carry_forward_tier=None,
        prediction_set=None,
        quality_flags=["MISSING_REQUIRED_GEOMETRY"],
        p_safe=np.nan,
        p_elevated=np.nan,
        p_critical=np.nan,
    )
    assert row["triage_action"] == TA.HUMAN_REVIEW_INSUFFICIENT_DATA.value
    assert RC.INVALID_INPUT.value in row["reason_codes"]
    assert RC.FORECAST_SAFE.value not in row["reason_codes"]


def test_invalid_status_overrides_a_confident_safe_forecast(cfg):
    # Even if the model emitted probabilities, 'invalid' status wins and never auto-clears.
    row = _triage(cfg, prediction_status="invalid")
    assert row["triage_action"] == TA.HUMAN_REVIEW_INSUFFICIENT_DATA.value
    assert row["triage_action"] != TA.AUTO_CLEAR_CANDIDATE.value
    assert RC.INVALID_INPUT.value in row["reason_codes"]


def test_invalid_status_overrides_a_critical_forecast(cfg):
    row = _triage(cfg, prediction_status="invalid", predicted_final_tier=CRIT, **_probs(0.05, 0.9))
    assert row["triage_action"] == TA.HUMAN_REVIEW_INSUFFICIENT_DATA.value
    assert RC.FORECAST_CRITICAL.value not in row["reason_codes"]


def test_missing_predicted_tier_is_insufficient_data(cfg):
    for missing in (None, np.nan):
        row = _triage(cfg, predicted_final_tier=missing)
        assert row["triage_action"] == TA.HUMAN_REVIEW_INSUFFICIENT_DATA.value
        assert RC.INVALID_INPUT.value in row["reason_codes"]


@pytest.mark.parametrize("col", ["p_safe", "p_elevated", "p_critical"])
def test_nan_probability_is_insufficient_data(cfg, col):
    row = _triage(cfg, **{col: np.nan})
    assert row["triage_action"] == TA.HUMAN_REVIEW_INSUFFICIENT_DATA.value
    assert RC.INVALID_INPUT.value in row["reason_codes"]


# --------------------------------------------------------------------------- #
# Event triage: rule 2 - predicted Critical
# --------------------------------------------------------------------------- #
def test_predicted_critical_is_maneuver_planning_review(cfg):
    row = _triage(cfg, predicted_final_tier=CRIT, current_tier=CRIT, carry_forward_tier=CRIT,
                  prediction_set=[CRIT], **_probs(0.02, 0.9))
    assert row["triage_action"] == TA.MANEUVER_PLANNING_REVIEW.value
    assert RC.FORECAST_CRITICAL.value in row["reason_codes"]
    assert RC.CURRENT_TIER_CRITICAL.value in row["reason_codes"]
    # Predicted Critical never gets the watch code.
    assert RC.ESCALATION_RISK.value not in row["reason_codes"]


def test_predicted_critical_beats_disagreement_and_watch(cfg):
    # Critical forecast with current Safe (two-tier disagreement) -> still maneuver review,
    # but the disagreement is documented.
    row = _triage(cfg, predicted_final_tier=CRIT, current_tier=SAFE, carry_forward_tier=SAFE,
                  prediction_set=[SAFE, CRIT], **_probs(0.3, 0.6))
    assert row["triage_action"] == TA.MANEUVER_PLANNING_REVIEW.value
    assert RC.FORECAST_CRITICAL.value in row["reason_codes"]
    assert RC.MODEL_CARRY_FORWARD_DISAGREE.value in row["reason_codes"]
    assert RC.AMBIGUOUS_PREDICTION_SET.value in row["reason_codes"]


# --------------------------------------------------------------------------- #
# Event triage: rule 3 - escalation watch
# --------------------------------------------------------------------------- #
def test_escalation_watch_fires_at_threshold_when_not_predicted_critical(cfg):
    for tier in (SAFE, ELEV):
        row = _triage(cfg, predicted_final_tier=tier, current_tier=tier, carry_forward_tier=tier,
                      prediction_set=[tier], **_probs(0.4, T_WATCH))
        assert row["triage_action"] == TA.ESCALATION_WATCH.value
        assert RC.ESCALATION_RISK.value in row["reason_codes"]


def test_escalation_watch_does_not_fire_below_threshold(cfg):
    row = _triage(cfg, predicted_final_tier=ELEV, current_tier=ELEV, carry_forward_tier=ELEV,
                  prediction_set=[ELEV], **_probs(0.3, T_WATCH - 1e-6))
    assert row["triage_action"] == TA.MONITOR.value
    assert RC.ESCALATION_RISK.value not in row["reason_codes"]


def test_escalation_watch_never_fires_for_predicted_critical(cfg):
    row = _triage(cfg, predicted_final_tier=CRIT, current_tier=CRIT, carry_forward_tier=CRIT,
                  prediction_set=[CRIT], **_probs(0.02, 0.98))
    assert row["triage_action"] == TA.MANEUVER_PLANNING_REVIEW.value
    assert RC.ESCALATION_RISK.value not in row["reason_codes"]


def test_escalation_watch_respects_policy_params(cfg):
    row = _triage(cfg, params={"t_watch": 0.10, "t_clear": T_CLEAR}, **_probs(0.8, 0.15))
    assert row["triage_action"] == TA.ESCALATION_WATCH.value
    row = _triage(cfg, params={"t_watch": 0.20, "t_clear": T_CLEAR}, **_probs(0.8, 0.15))
    assert row["triage_action"] == TA.AUTO_CLEAR_CANDIDATE.value


def test_escalation_watch_beats_human_review_conditions(cfg):
    # Predicted Safe, current Critical (two-tier disagreement + de-escalation claim) but
    # P(Critical) >= t_watch -> rule 3 fires first; all conditions documented.
    row = _triage(cfg, predicted_final_tier=SAFE, current_tier=CRIT, carry_forward_tier=CRIT,
                  prediction_set=[SAFE, CRIT], **_probs(0.5, 0.45))
    assert row["triage_action"] == TA.ESCALATION_WATCH.value
    codes = row["reason_codes"]
    for code in (RC.ESCALATION_RISK, RC.MODEL_CARRY_FORWARD_DISAGREE, RC.AMBIGUOUS_PREDICTION_SET,
                 RC.DEESCALATION_CLAIM, RC.CURRENT_TIER_CRITICAL, RC.FORECAST_SAFE):
        assert code.value in codes


# --------------------------------------------------------------------------- #
# Event triage: rule 4 - human review conditions
# --------------------------------------------------------------------------- #
def test_two_tier_disagreement_is_human_review(cfg):
    # Model Safe vs carry-forward Critical.
    row = _triage(cfg, predicted_final_tier=SAFE, current_tier=ELEV, carry_forward_tier=CRIT,
                  prediction_set=[SAFE], **_probs(0.9, 0.05))
    assert row["triage_action"] == TA.HUMAN_REVIEW.value
    assert RC.MODEL_CARRY_FORWARD_DISAGREE.value in row["reason_codes"]

    # Model Critical vs carry-forward Safe is handled by rule 2 (see above); the opposite
    # direction with an Elevated forecast is only one tier -> no disagreement code.
    row = _triage(cfg, predicted_final_tier=ELEV, current_tier=SAFE, carry_forward_tier=SAFE,
                  prediction_set=[ELEV], **_probs(0.3, 0.1))
    assert row["triage_action"] == TA.MONITOR.value
    assert RC.MODEL_CARRY_FORWARD_DISAGREE.value not in row["reason_codes"]


def test_one_tier_disagreement_does_not_force_human_review(cfg):
    # Safe forecast, carry-forward Elevated: one tier apart -> not rule 4. It still cannot
    # auto-clear because the current tier is not Safe (rule 6), which is the right code.
    row = _triage(cfg, predicted_final_tier=SAFE, current_tier=ELEV, carry_forward_tier=ELEV,
                  prediction_set=[SAFE], **_probs(0.9, 0.05))
    assert RC.MODEL_CARRY_FORWARD_DISAGREE.value not in row["reason_codes"]
    assert row["triage_action"] == TA.HUMAN_REVIEW.value
    assert RC.CURRENT_TIER_NOT_SAFE.value in row["reason_codes"]
    assert RC.CURRENT_TIER_ELEVATED.value in row["reason_codes"]

    # Elevated forecast with carry-forward Critical -> Monitor, no disagreement code.
    row = _triage(cfg, predicted_final_tier=ELEV, current_tier=ELEV, carry_forward_tier=CRIT,
                  prediction_set=[ELEV], **_probs(0.3, 0.1))
    assert row["triage_action"] == TA.MONITOR.value
    assert RC.MODEL_CARRY_FORWARD_DISAGREE.value not in row["reason_codes"]


def test_carry_forward_falls_back_to_current_tier_when_absent(cfg):
    df = _row(predicted_final_tier=SAFE, current_tier=CRIT, prediction_set=[SAFE],
              **_probs(0.9, 0.05)).drop(columns=["carry_forward_tier"])
    row = assign_triage(df, cfg, PARAMS).iloc[0]
    assert row["triage_action"] == TA.HUMAN_REVIEW.value
    assert RC.MODEL_CARRY_FORWARD_DISAGREE.value in row["reason_codes"]


def test_deescalation_claim_elevated_is_human_review(cfg):
    row = _triage(cfg, predicted_final_tier=ELEV, current_tier=CRIT, carry_forward_tier=CRIT,
                  prediction_set=[ELEV], **_probs(0.2, 0.2))
    assert row["triage_action"] == TA.HUMAN_REVIEW.value
    assert RC.DEESCALATION_CLAIM.value in row["reason_codes"]
    assert RC.CURRENT_TIER_CRITICAL.value in row["reason_codes"]
    assert RC.FORECAST_ELEVATED.value in row["reason_codes"]
    # One tier apart: no two-tier disagreement.
    assert RC.MODEL_CARRY_FORWARD_DISAGREE.value not in row["reason_codes"]


def test_deescalation_claim_to_safe_is_never_auto_cleared(cfg):
    row = _triage(cfg, predicted_final_tier=SAFE, current_tier=CRIT, carry_forward_tier=CRIT,
                  prediction_set=[SAFE], **_probs(0.99, 0.005))
    assert row["triage_action"] == TA.HUMAN_REVIEW.value
    assert row["triage_action"] != TA.AUTO_CLEAR_CANDIDATE.value
    assert RC.DEESCALATION_CLAIM.value in row["reason_codes"]
    assert RC.CURRENT_TIER_NOT_SAFE.value in row["reason_codes"]
    assert RC.FORECAST_SAFE.value in row["reason_codes"]


def test_deescalation_claim_with_high_p_critical_goes_to_watch(cfg):
    # Current Critical, predicted Elevated with P(Critical) above deescalation_p_critical:
    # the claim is documented and the escalation watch (rule 3) routes it.
    deesc_p = cfg["triage"]["deescalation_p_critical"]
    p_crit = max(deesc_p, T_WATCH)
    row = _triage(cfg, predicted_final_tier=ELEV, current_tier=CRIT, carry_forward_tier=CRIT,
                  prediction_set=[ELEV, CRIT], **_probs(0.05, p_crit))
    assert row["triage_action"] == TA.ESCALATION_WATCH.value
    assert RC.DEESCALATION_CLAIM.value in row["reason_codes"]


def test_ambiguous_prediction_set_is_human_review(cfg):
    row = _triage(cfg, predicted_final_tier=SAFE, prediction_set=[SAFE, CRIT], **_probs(0.9, 0.05))
    assert row["triage_action"] == TA.HUMAN_REVIEW.value
    assert RC.AMBIGUOUS_PREDICTION_SET.value in row["reason_codes"]

    # Order and extra classes do not matter.
    row = _triage(cfg, predicted_final_tier=SAFE, prediction_set=[CRIT, ELEV, SAFE], **_probs(0.9, 0.05))
    assert row["triage_action"] == TA.HUMAN_REVIEW.value
    assert RC.AMBIGUOUS_PREDICTION_SET.value in row["reason_codes"]


def test_adjacent_prediction_set_is_not_ambiguous(cfg):
    row = _triage(cfg, predicted_final_tier=ELEV, current_tier=ELEV, carry_forward_tier=ELEV,
                  prediction_set=[ELEV, CRIT], **_probs(0.2, 0.25))
    assert row["triage_action"] == TA.MONITOR.value
    assert RC.AMBIGUOUS_PREDICTION_SET.value not in row["reason_codes"]

    row = _triage(cfg, prediction_set=[SAFE, ELEV], **_probs(0.9, 0.02))
    assert row["triage_action"] == TA.AUTO_CLEAR_CANDIDATE.value
    assert RC.AMBIGUOUS_PREDICTION_SET.value not in row["reason_codes"]


def test_prediction_set_missing_or_absent_is_tolerated(cfg):
    for missing in (None, np.nan):
        row = _triage(cfg, prediction_set=missing)
        assert row["triage_action"] == TA.AUTO_CLEAR_CANDIDATE.value
    df = _row().drop(columns=["prediction_set"])
    assert assign_triage(df, cfg, PARAMS).iloc[0]["triage_action"] == TA.AUTO_CLEAR_CANDIDATE.value


# --------------------------------------------------------------------------- #
# Event triage: rule 5 - Monitor
# --------------------------------------------------------------------------- #
def test_predicted_elevated_is_monitor(cfg):
    row = _triage(cfg, predicted_final_tier=ELEV, current_tier=ELEV, carry_forward_tier=ELEV,
                  prediction_set=[ELEV], **_probs(0.3, 0.1))
    assert row["triage_action"] == TA.MONITOR.value
    assert RC.FORECAST_ELEVATED.value in row["reason_codes"]
    assert RC.CURRENT_TIER_ELEVATED.value in row["reason_codes"]


def test_predicted_elevated_with_current_safe_is_monitor(cfg):
    row = _triage(cfg, predicted_final_tier=ELEV, current_tier=SAFE, carry_forward_tier=SAFE,
                  prediction_set=[ELEV], **_probs(0.4, 0.05))
    assert row["triage_action"] == TA.MONITOR.value
    assert row["reason_codes"] == [RC.FORECAST_ELEVATED.value]


# --------------------------------------------------------------------------- #
# Event triage: rules 6/7 - auto-clear candidate and residual human review
# --------------------------------------------------------------------------- #
def test_auto_clear_candidate_when_all_conditions_hold(cfg):
    row = _triage(cfg)
    assert row["triage_action"] == TA.AUTO_CLEAR_CANDIDATE.value
    assert row["reason_codes"] == [RC.FORECAST_SAFE.value]


def test_auto_clear_threshold_is_inclusive(cfg):
    row = _triage(cfg, **_probs(T_CLEAR, 0.05))
    assert row["triage_action"] == TA.AUTO_CLEAR_CANDIDATE.value
    row = _triage(cfg, **_probs(T_CLEAR - 1e-6, 0.05))
    assert row["triage_action"] == TA.HUMAN_REVIEW.value
    assert RC.LOW_CONFIDENCE_SAFE.value in row["reason_codes"]
    assert RC.FORECAST_SAFE.value in row["reason_codes"]


def test_low_confidence_safe_blocks_auto_clear(cfg):
    row = _triage(cfg, **_probs(0.6, 0.1))
    assert row["triage_action"] == TA.HUMAN_REVIEW.value
    assert RC.LOW_CONFIDENCE_SAFE.value in row["reason_codes"]
    assert RC.CURRENT_TIER_NOT_SAFE.value not in row["reason_codes"]
    assert RC.QUALITY_WARNING.value not in row["reason_codes"]


def test_current_tier_not_safe_blocks_auto_clear(cfg):
    row = _triage(cfg, current_tier=ELEV, carry_forward_tier=ELEV)
    assert row["triage_action"] == TA.HUMAN_REVIEW.value
    assert RC.CURRENT_TIER_NOT_SAFE.value in row["reason_codes"]
    assert RC.CURRENT_TIER_ELEVATED.value in row["reason_codes"]
    assert RC.LOW_CONFIDENCE_SAFE.value not in row["reason_codes"]

    # Unknown current tier also blocks (cannot confirm it is Safe).
    row = _triage(cfg, current_tier=None, carry_forward_tier=None)
    assert row["triage_action"] == TA.HUMAN_REVIEW.value
    assert RC.CURRENT_TIER_NOT_SAFE.value in row["reason_codes"]


def test_blocking_quality_flag_forces_human_review_with_codes(cfg):
    row = _triage(cfg, prediction_status="warning", quality_flags=["MISSING_SIGMA"])
    assert row["triage_action"] == TA.HUMAN_REVIEW.value
    assert RC.QUALITY_WARNING.value in row["reason_codes"]
    assert RC.MISSING_TELEMETRY.value in row["reason_codes"]
    assert RC.FORECAST_SAFE.value in row["reason_codes"]


def test_non_missing_blocking_flag_has_no_missing_telemetry(cfg):
    row = _triage(cfg, prediction_status="warning", quality_flags=["SUSPICIOUS_SIGMA"])
    assert row["triage_action"] == TA.HUMAN_REVIEW.value
    assert RC.QUALITY_WARNING.value in row["reason_codes"]
    assert RC.MISSING_TELEMETRY.value not in row["reason_codes"]


def test_allowed_quality_flag_alone_still_auto_clears(cfg):
    row = _triage(cfg, prediction_status="warning", quality_flags=["MISSING_SPACE_WEATHER"])
    assert row["triage_action"] == TA.AUTO_CLEAR_CANDIDATE.value
    assert RC.QUALITY_WARNING.value not in row["reason_codes"]
    assert RC.MISSING_TELEMETRY.value not in row["reason_codes"]


def test_allowed_flag_mixed_with_blocking_flag_blocks(cfg):
    row = _triage(cfg, prediction_status="warning",
                  quality_flags=["MISSING_SPACE_WEATHER", "MISSING_SIGMA"])
    assert row["triage_action"] == TA.HUMAN_REVIEW.value
    assert RC.QUALITY_WARNING.value in row["reason_codes"]


def test_valid_status_with_flags_is_acceptable_quality(cfg):
    # Acceptable quality = status valid OR every flag allow-listed.
    row = _triage(cfg, prediction_status="valid", quality_flags=["MISSING_SIGMA"])
    assert row["triage_action"] == TA.AUTO_CLEAR_CANDIDATE.value


def test_all_blocking_conditions_are_reported_together(cfg):
    row = _triage(cfg, prediction_status="warning", quality_flags=["MISSING_SIGMA"],
                  current_tier=ELEV, carry_forward_tier=ELEV, **_probs(0.5, 0.1))
    assert row["triage_action"] == TA.HUMAN_REVIEW.value
    for code in (RC.FORECAST_SAFE, RC.LOW_CONFIDENCE_SAFE, RC.CURRENT_TIER_NOT_SAFE,
                 RC.QUALITY_WARNING, RC.MISSING_TELEMETRY, RC.CURRENT_TIER_ELEVATED):
        assert code.value in row["reason_codes"]


# --------------------------------------------------------------------------- #
# Descriptive codes: history and trends
# --------------------------------------------------------------------------- #
def test_no_history_code(cfg):
    row = _triage(cfg, has_history=0)
    assert RC.NO_HISTORY.value in row["reason_codes"]
    assert row["triage_action"] == TA.AUTO_CLEAR_CANDIDATE.value  # descriptive only
    row = _triage(cfg, has_history=1)
    assert RC.NO_HISTORY.value not in row["reason_codes"]
    df = _row().drop(columns=["has_history"])
    assert RC.NO_HISTORY.value not in assign_triage(df, cfg, PARAMS).iloc[0]["reason_codes"]


def test_trend_codes_follow_sign_conventions(cfg):
    tr = cfg["triage"]["trends"]

    # MISS_SHRINKING: slope_log_miss >= miss_shrink_slope (positive slope = shrinking).
    row = _triage(cfg, slope_log_miss=tr["miss_shrink_slope"])
    assert RC.MISS_SHRINKING.value in row["reason_codes"]
    row = _triage(cfg, slope_log_miss=tr["miss_shrink_slope"] - 1e-6)
    assert RC.MISS_SHRINKING.value not in row["reason_codes"]
    row = _triage(cfg, slope_log_miss=-abs(tr["miss_shrink_slope"]) * 10)
    assert RC.MISS_SHRINKING.value not in row["reason_codes"]

    # RISK_RISING: slope_risk_or_mre <= risk_rise_slope (negative slope = rising).
    row = _triage(cfg, slope_risk_or_mre=tr["risk_rise_slope"])
    assert RC.RISK_RISING.value in row["reason_codes"]
    row = _triage(cfg, slope_risk_or_mre=tr["risk_rise_slope"] + 1e-6)
    assert RC.RISK_RISING.value not in row["reason_codes"]
    row = _triage(cfg, slope_risk_or_mre=abs(tr["risk_rise_slope"]) * 10)
    assert RC.RISK_RISING.value not in row["reason_codes"]

    # SIGMA_GROWING: slope_log_sigma <= sigma_grow_slope.
    row = _triage(cfg, slope_log_sigma=tr["sigma_grow_slope"])
    assert RC.SIGMA_GROWING.value in row["reason_codes"]
    row = _triage(cfg, slope_log_sigma=tr["sigma_grow_slope"] + 1e-6)
    assert RC.SIGMA_GROWING.value not in row["reason_codes"]

    # HIGH_VOLATILITY: either volatility source.
    row = _triage(cfg, volatility_mre=tr["mre_volatility"])
    assert RC.HIGH_VOLATILITY.value in row["reason_codes"]
    row = _triage(cfg, volatility_risk=tr["risk_volatility"])
    assert RC.HIGH_VOLATILITY.value in row["reason_codes"]
    row = _triage(cfg, volatility_mre=tr["mre_volatility"] - 1e-6,
                  volatility_risk=tr["risk_volatility"] - 1e-6)
    assert RC.HIGH_VOLATILITY.value not in row["reason_codes"]


def test_trend_codes_never_fire_on_nan(cfg):
    row = _triage(cfg, slope_log_miss=np.nan, slope_risk_or_mre=np.nan, slope_log_sigma=np.nan,
                  volatility_mre=np.nan, volatility_risk=np.nan)
    trend_codes = {RC.MISS_SHRINKING.value, RC.RISK_RISING.value, RC.SIGMA_GROWING.value,
                   RC.HIGH_VOLATILITY.value}
    assert trend_codes.isdisjoint(row["reason_codes"])
    assert row["triage_action"] == TA.AUTO_CLEAR_CANDIDATE.value


def test_trend_columns_absent_are_tolerated(cfg):
    df = _row().drop(columns=["slope_log_miss", "slope_risk_or_mre", "slope_log_sigma",
                              "volatility_mre", "volatility_risk"])
    row = assign_triage(df, cfg, PARAMS).iloc[0]
    assert row["triage_action"] == TA.AUTO_CLEAR_CANDIDATE.value
    assert row["reason_codes"] == [RC.FORECAST_SAFE.value]


def test_trend_codes_are_descriptive_and_do_not_change_action(cfg):
    tr = cfg["triage"]["trends"]
    row = _triage(cfg, slope_log_miss=tr["miss_shrink_slope"] + 1, slope_risk_or_mre=tr["risk_rise_slope"] - 1,
                  volatility_mre=tr["mre_volatility"] + 1)
    assert row["triage_action"] == TA.AUTO_CLEAR_CANDIDATE.value
    for code in (RC.MISS_SHRINKING, RC.RISK_RISING, RC.HIGH_VOLATILITY):
        assert code.value in row["reason_codes"]


# --------------------------------------------------------------------------- #
# Output hygiene
# --------------------------------------------------------------------------- #
def test_reason_codes_deduplicated_and_in_enum_order(cfg):
    df = pd.concat(
        [_row(prediction_status="warning", quality_flags=["MISSING_SIGMA", "MISSING_SPEED"],
              current_tier=CRIT, carry_forward_tier=CRIT, **_probs(0.5, 0.1))] * 3,
        ignore_index=True,
    )
    out = assign_triage(df, cfg, PARAMS)
    order = {code.value: i for i, code in enumerate(ReasonCode)}
    for codes in out["reason_codes"]:
        assert len(codes) == len(set(codes))
        assert codes == sorted(codes, key=order.__getitem__)
        assert codes == out["reason_codes"].iloc[0]


def test_policy_params_are_required():
    df = _row()
    with pytest.raises(KeyError):
        assign_triage(df, load_config(), {"t_watch": 0.3})
    with pytest.raises(KeyError):
        assign_triage(df, load_config(), {"t_clear": 0.8})


def test_input_not_mutated_and_order_preserved(cfg):
    df = pd.DataFrame(
        {
            "prediction_status": ["valid", "invalid", "warning", "valid", "warning", "valid"],
            "quality_flags": [[], ["MISSING_REQUIRED_GEOMETRY"], [], [], ["MISSING_SIGMA"], []],
            "predicted_final_tier": [SAFE, None, ELEV, CRIT, SAFE, SAFE],
            "p_safe": [0.95, np.nan, 0.3, 0.02, 0.95, 0.5],
            "p_elevated": [0.04, np.nan, 0.6, 0.08, 0.04, 0.1],
            "p_critical": [0.01, np.nan, 0.1, 0.9, 0.01, 0.4],
            "current_tier": [SAFE, None, ELEV, CRIT, SAFE, SAFE],
            "carry_forward_tier": [SAFE, None, ELEV, CRIT, SAFE, SAFE],
            "prediction_set": [[SAFE], None, [ELEV], [CRIT], [SAFE], [SAFE, ELEV, CRIT]],
            "has_history": [1, 0, 1, 1, 1, 0],
            "slope_log_miss": [0.0, np.nan, 0.5, 0.0, 0.0, 0.0],
        },
        index=[10, 3, 7, 1, 5, 2],
    )
    snapshot = df.copy(deep=True)

    out = assign_triage(df, cfg, PARAMS)

    assert out is not df
    pd.testing.assert_frame_equal(df, snapshot)
    assert {"triage_action", "reason_codes"}.isdisjoint(df.columns)
    assert list(out.index) == list(df.index)
    assert list(out["triage_action"]) == [
        TA.AUTO_CLEAR_CANDIDATE.value,
        TA.HUMAN_REVIEW_INSUFFICIENT_DATA.value,
        TA.MONITOR.value,
        TA.MANEUVER_PLANNING_REVIEW.value,
        TA.HUMAN_REVIEW.value,
        TA.ESCALATION_WATCH.value,
    ]
    assert out["triage_action"].map(type).eq(str).all()
    assert out["reason_codes"].map(lambda c: isinstance(c, list)).all()
    assert RC.MISS_SHRINKING.value in out.loc[7, "reason_codes"]
    assert RC.NO_HISTORY.value in out.loc[2, "reason_codes"]


def test_minimal_columns_are_sufficient(cfg):
    # Only the required forecast columns; every optional column absent.
    df = pd.DataFrame(
        {
            "prediction_status": ["valid", "valid"],
            "predicted_final_tier": [SAFE, CRIT],
            "p_safe": [0.9, 0.1],
            "p_elevated": [0.08, 0.1],
            "p_critical": [0.02, 0.8],
            "current_tier": [SAFE, CRIT],
        }
    )
    out = assign_triage(df, cfg, PARAMS)
    assert list(out["triage_action"]) == [TA.AUTO_CLEAR_CANDIDATE.value, TA.MANEUVER_PLANNING_REVIEW.value]


def test_empty_frame(cfg):
    df = _row().iloc[0:0]
    out = assign_triage(df, cfg, PARAMS)
    assert len(out) == 0
    assert "triage_action" in out.columns and "reason_codes" in out.columns


def _synthetic_events(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    status = rng.choice(["valid", "warning", "invalid"], n, p=[0.75, 0.2, 0.05])
    p = rng.dirichlet([6.0, 2.0, 1.0], size=n)
    p[status == "invalid"] = np.nan
    pred = np.where(np.isnan(p[:, 0]), None, np.array([SAFE, ELEV, CRIT], dtype=object)[np.nan_to_num(p).argmax(1)])
    tiers = np.array([SAFE, ELEV, CRIT], dtype=object)
    current = rng.choice(tiers, n, p=[0.7, 0.2, 0.1])
    final = np.where(rng.uniform(size=n) < 0.8, current, rng.choice(tiers, n))
    flags = [
        list(rng.choice(["MISSING_SIGMA", "MISSING_SPACE_WEATHER"], size=rng.integers(0, 3)))
        if s == "warning" else ([] if s == "valid" else ["MISSING_REQUIRED_GEOMETRY"])
        for s in status
    ]
    sets = [None if t is None else sorted({t, rng.choice(tiers)}) for t in pred]
    return pd.DataFrame(
        {
            "prediction_status": status,
            "quality_flags": flags,
            "predicted_final_tier": pred,
            "p_safe": p[:, 0],
            "p_elevated": p[:, 1],
            "p_critical": p[:, 2],
            "current_tier": current,
            "carry_forward_tier": current,
            "prediction_set": sets,
            "has_history": rng.integers(0, 2, n),
            "slope_log_miss": rng.normal(0, 0.1, n),
            "slope_risk_or_mre": rng.normal(0, 0.2, n),
            "slope_log_sigma": rng.normal(0, 0.1, n),
            "volatility_mre": np.abs(rng.normal(0, 0.5, n)),
            "volatility_risk": np.where(rng.uniform(size=n) < 0.3, np.nan, np.abs(rng.normal(0, 1.5, n))),
            "final_tier": final,
        }
    )


def test_vectorised_bulk_is_consistent_with_per_row(cfg):
    df = _synthetic_events(1500, seed=1)
    bulk = assign_triage(df, cfg, PARAMS)
    for i in range(0, len(df), 53):
        single = assign_triage(df.iloc[[i]], cfg, PARAMS).iloc[0]
        assert bulk.iloc[i]["triage_action"] == single["triage_action"]
        assert bulk.iloc[i]["reason_codes"] == single["reason_codes"]

    # Invariants.
    invalid = bulk["prediction_status"] == "invalid"
    assert (bulk.loc[invalid, "triage_action"] == TA.HUMAN_REVIEW_INSUFFICIENT_DATA.value).all()
    auto = bulk["triage_action"] == TA.AUTO_CLEAR_CANDIDATE.value
    assert (bulk.loc[auto, "predicted_final_tier"] == SAFE).all()
    assert (bulk.loc[auto, "current_tier"] == SAFE).all()
    assert (bulk.loc[auto, "p_safe"] >= T_CLEAR).all()
    assert (bulk.loc[auto, "p_critical"] < T_WATCH).all()
    watch = bulk["triage_action"] == TA.ESCALATION_WATCH.value
    assert (bulk.loc[watch, "predicted_final_tier"] != CRIT).all()
    assert (bulk.loc[watch, "p_critical"] >= T_WATCH).all()
    maneuver = bulk["triage_action"] == TA.MANEUVER_PLANNING_REVIEW.value
    assert (bulk.loc[maneuver, "predicted_final_tier"] == CRIT).all()
    deesc = (bulk["current_tier"] == CRIT) & (bulk["predicted_final_tier"].isin([SAFE, ELEV]))
    assert not (bulk.loc[deesc, "triage_action"] == TA.AUTO_CLEAR_CANDIDATE.value).any()
    assert bulk.loc[deesc, "reason_codes"].map(lambda c: RC.DEESCALATION_CLAIM.value in c).all()


# --------------------------------------------------------------------------- #
# Threshold tuning
# --------------------------------------------------------------------------- #
def test_tune_policy_thresholds_on_synthetic_oof(cfg):
    oof = _synthetic_events(4000, seed=7)
    res = tune_policy_thresholds(oof, cfg)

    assert set(res) == {"t_watch", "t_clear", "watch", "auto_clear"}
    assert isinstance(res["t_watch"], float) and isinstance(res["t_clear"], float)

    # JSON-serialisable (python floats / ints / bools / strs only).
    json.dumps(res)

    watch = res["watch"]
    assert watch["watch_share"] <= cfg["triage"]["watch_max_share"] + 1e-12
    assert watch["n_watch"] == round(watch["watch_share"] * len(oof))
    assert 0 <= watch["n_escalations_caught"] <= watch["n_escalations_total"]
    assert 0.0 <= watch["recall"] <= 1.0

    ac = res["auto_clear"]
    assert ac["n_final_critical_cleared"] == 0
    assert 0.0 <= ac["auto_clear_share"] <= 1.0
    assert ac["n_auto_clear"] == round(ac["auto_clear_share"] * len(oof))

    # Applying the tuned thresholds reproduces the reported counts.
    out = assign_triage(oof, cfg, {"t_watch": res["t_watch"], "t_clear": res["t_clear"]})
    n_watch = int((out["triage_action"] == TA.ESCALATION_WATCH.value).sum())
    assert n_watch <= watch["n_watch"]  # rule 1 never contributes; rule 3 cannot exceed the pool
    cleared = out["triage_action"] == TA.AUTO_CLEAR_CANDIDATE.value
    assert not (out.loc[cleared, "final_tier"] == CRIT).any()


def test_tune_t_watch_is_smallest_threshold_within_budget(cfg):
    # 100 events, none predicted Critical, p_critical = 0.00..0.99; budget 5% -> 5 events.
    n = 100
    p_crit = np.arange(n) / n
    oof = pd.DataFrame(
        {
            "prediction_status": "valid",
            "quality_flags": [[] for _ in range(n)],
            "predicted_final_tier": SAFE,
            "p_safe": 1.0 - p_crit,
            "p_elevated": 0.0,
            "p_critical": p_crit,
            "current_tier": SAFE,
            "final_tier": np.where(p_crit >= 0.97, CRIT, SAFE),
        }
    )
    res = tune_policy_thresholds(oof, cfg)
    assert res["t_watch"] == pytest.approx(0.95)
    assert res["watch"]["n_watch"] == 5
    assert res["watch"]["watch_share"] == pytest.approx(0.05)
    assert res["watch"]["n_escalations_total"] == 3
    assert res["watch"]["n_escalations_caught"] == 3
    assert res["watch"]["recall"] == pytest.approx(1.0)


def test_tune_t_clear_excludes_every_final_critical(cfg):
    oof = pd.DataFrame(
        {
            "prediction_status": ["valid"] * 6,
            "quality_flags": [[] for _ in range(6)],
            "predicted_final_tier": [SAFE] * 6,
            "p_safe": [0.99, 0.95, 0.90, 0.85, 0.70, 0.60],
            "p_elevated": [0.005, 0.03, 0.05, 0.1, 0.2, 0.3],
            "p_critical": [0.005, 0.02, 0.05, 0.05, 0.1, 0.1],
            "current_tier": [SAFE] * 6,
            "final_tier": [SAFE, SAFE, CRIT, SAFE, SAFE, SAFE],
        }
    )
    res = tune_policy_thresholds(oof, cfg)
    assert res["t_clear"] > 0.90
    assert res["auto_clear"]["n_final_critical_cleared"] == 0
    assert res["auto_clear"]["n_auto_clear"] == 2
    assert res["auto_clear"]["auto_clear_share"] == pytest.approx(2 / 6)
    assert res["auto_clear"]["zero_missed_achievable"] is True

    # Ineligible events (current tier not Safe / blocking flag) never count, even if final Critical.
    oof2 = oof.copy()
    oof2.loc[2, "current_tier"] = ELEV
    res2 = tune_policy_thresholds(oof2, cfg)
    assert res2["t_clear"] <= 0.60
    assert res2["auto_clear"]["n_auto_clear"] == 5


def test_tune_t_clear_disables_auto_clear_when_zero_is_unreachable(cfg):
    oof = pd.DataFrame(
        {
            "prediction_status": ["valid"] * 3,
            "quality_flags": [[] for _ in range(3)],
            "predicted_final_tier": [SAFE] * 3,
            "p_safe": [1.0, 0.9, 0.8],
            "p_elevated": [0.0, 0.05, 0.1],
            "p_critical": [0.0, 0.05, 0.1],
            "current_tier": [SAFE] * 3,
            "final_tier": [CRIT, SAFE, SAFE],
        }
    )
    res = tune_policy_thresholds(oof, cfg)
    assert res["t_clear"] == pytest.approx(1.01)
    assert res["auto_clear"]["zero_missed_achievable"] is False
    assert res["auto_clear"]["n_auto_clear"] == 0
    assert res["auto_clear"]["n_final_critical_cleared"] == 0
    assert "1.01" in res["auto_clear"]["note"]
    json.dumps(res)

    # Nothing clears when applied.
    out = assign_triage(oof, cfg, {"t_watch": res["t_watch"], "t_clear": res["t_clear"]})
    assert not (out["triage_action"] == TA.AUTO_CLEAR_CANDIDATE.value).any()
