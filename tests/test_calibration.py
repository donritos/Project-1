import math

import numpy as np
import pytest

from collision_risk.calibration import (
    STATUS_INSUFFICIENT,
    STATUS_OK,
    IsotonicCalibrator,
    SigmoidCalibrator,
    SplitConformalCalibrator,
    classifier_metrics,
    empirical_coverage,
    finite_sample_quantile,
    reliability_diagram,
)


# --------------------------------------------------------------------------- #
# finite_sample_quantile
# --------------------------------------------------------------------------- #
def test_finite_sample_quantile_known_values():
    scores = np.arange(1, 11, dtype=float)  # 1..10
    # k = ceil(11 * 0.9) = 10 -> 10th smallest = 10
    assert finite_sample_quantile(scores, 0.9) == 10.0
    # k = ceil(11 * 0.95) = 11 > 10 -> None
    assert finite_sample_quantile(scores, 0.95) is None
    # k = ceil(11 * 0.5) = 6 -> 6
    assert finite_sample_quantile(scores, 0.5) == 6.0
    # order-independent
    rng = np.random.default_rng(0)
    assert finite_sample_quantile(rng.permutation(scores), 0.5) == 6.0


def test_finite_sample_quantile_edge_cases():
    assert finite_sample_quantile(np.array([]), 0.9) is None
    assert finite_sample_quantile(np.array([np.nan, 3.0, np.nan]), 0.5) == 3.0
    with pytest.raises(ValueError):
        finite_sample_quantile(np.array([1.0]), 1.0)


# --------------------------------------------------------------------------- #
# SplitConformalCalibrator
# --------------------------------------------------------------------------- #
def test_per_event_max_residual_is_used():
    # Event A: 50 rows with tiny residuals except one huge one.
    # Event B..: 39 events with single rows, residual 1.
    # Row-level quantile at 0.5 would be ~0.1 or 1; event-level: A's score is 100.
    y_true, y_pred, events = [], [], []
    for _ in range(49):
        y_true.append(0.0)
        y_pred.append(0.1)
        events.append(1)
    y_true.append(0.0)
    y_pred.append(100.0)
    events.append(1)
    for e in range(2, 41):  # 39 more events
        y_true.append(0.0)
        y_pred.append(1.0)
        events.append(e)

    cal = SplitConformalCalibrator(coverage=0.9, min_events=30).fit(y_true, y_pred, events)
    assert cal.status == STATUS_OK
    assert cal.n_events == 40
    assert cal.n_rows == 89
    # per-event scores: [100] + [1]*39 -> k = ceil(41*0.9) = 37 -> 37th smallest = 1
    assert cal.q_hat == 1.0

    # Compare against a coverage where the event-level max matters: coverage 0.99
    # k = ceil(41*0.99) = 41 > 40 -> insufficient; whereas at 0.975 -> k=40 -> 100
    cal2 = SplitConformalCalibrator(coverage=0.975, min_events=30).fit(y_true, y_pred, events)
    assert cal2.q_hat == 100.0

    # Row-level computation would have given a different answer at 0.975:
    # 89 rows -> k = ceil(90*0.975) = 88 -> 88th smallest of [0.1]*49+[1]*39+[100] = 1.0
    row_q = finite_sample_quantile(np.abs(np.array(y_true) - np.array(y_pred)), 0.975)
    assert row_q == 1.0
    assert cal2.q_hat != row_q


def test_nan_rows_are_ignored():
    y_true = [0.0, np.nan, 0.0, 0.0]
    y_pred = [0.5, 1.0, np.nan, 0.2]
    events = [1, 1, 2, 2]
    cal = SplitConformalCalibrator(coverage=0.5, min_events=1).fit(y_true, y_pred, events)
    assert cal.n_rows == 2
    assert cal.n_events == 2
    # scores [0.5, 0.2] -> k = ceil(3*0.5)=2 -> 0.5
    assert cal.status == STATUS_OK
    assert cal.q_hat == pytest.approx(0.5)


def test_insufficient_calibration_returns_nan_intervals():
    y_true = np.zeros(10)
    y_pred = np.linspace(-1, 1, 10)
    events = np.arange(10)
    cal = SplitConformalCalibrator(coverage=0.9, min_events=30).fit(y_true, y_pred, events)
    assert cal.status == STATUS_INSUFFICIENT
    assert cal.q_hat is None
    lo, hi = cal.predict_interval(np.array([-5.0, -6.0, np.nan]))
    assert np.all(np.isnan(lo)) and np.all(np.isnan(hi))

    # Enough events but quantile not achievable: 30 events at coverage 0.99
    # k = ceil(31 * 0.99) = 31 > 30 -> None
    cal2 = SplitConformalCalibrator(coverage=0.99, min_events=30).fit(
        np.zeros(30), np.ones(30), np.arange(30)
    )
    assert cal2.status == STATUS_INSUFFICIENT
    assert cal2.q_hat is None


def test_predict_interval_ok_and_nan_passthrough():
    rng = np.random.default_rng(1)
    n_events = 50
    y_pred = rng.normal(-6, 1, n_events)
    y_true = y_pred + rng.normal(0, 0.3, n_events)
    cal = SplitConformalCalibrator(coverage=0.9, min_events=30).fit(y_true, y_pred, np.arange(n_events))
    assert cal.status == STATUS_OK
    lo, hi = cal.predict_interval(np.array([-5.0, np.nan]))
    assert lo[0] == pytest.approx(-5.0 - cal.q_hat)
    assert hi[0] == pytest.approx(-5.0 + cal.q_hat)
    assert math.isnan(lo[1]) and math.isnan(hi[1])


def test_to_dict_from_dict_roundtrip_preserves_intervals():
    rng = np.random.default_rng(2)
    n = 200
    events = np.repeat(np.arange(40), 5)
    y_pred = rng.normal(-6, 1, n)
    y_true = y_pred + rng.normal(0, 0.5, n)
    cal = SplitConformalCalibrator(coverage=0.9, min_events=30).fit(y_true, y_pred, events)
    assert cal.status == STATUS_OK

    d = cal.to_dict()
    cal2 = SplitConformalCalibrator.from_dict(d)
    assert cal2.status == cal.status
    assert cal2.q_hat == cal.q_hat
    assert cal2.n_events == cal.n_events
    assert cal2.n_rows == cal.n_rows
    assert cal2.coverage == cal.coverage

    probe = np.array([-7.0, -5.5, np.nan, -3.0])
    lo1, hi1 = cal.predict_interval(probe)
    lo2, hi2 = cal2.predict_interval(probe)
    np.testing.assert_array_equal(lo1, lo2)
    np.testing.assert_array_equal(hi1, hi2)

    # insufficient roundtrip stays insufficient
    bad = SplitConformalCalibrator(coverage=0.9, min_events=30).fit(
        np.zeros(5), np.ones(5), np.arange(5)
    )
    bad2 = SplitConformalCalibrator.from_dict(bad.to_dict())
    assert bad2.status == STATUS_INSUFFICIENT
    lo, hi = bad2.predict_interval(np.array([-5.0]))
    assert np.isnan(lo).all() and np.isnan(hi).all()

    # Corrupted payload claiming 'ok' without q_hat is not trusted
    corrupt = SplitConformalCalibrator.from_dict({"coverage": 0.9, "status": "ok", "q_hat": None})
    assert corrupt.status == STATUS_INSUFFICIENT


# --------------------------------------------------------------------------- #
# empirical_coverage
# --------------------------------------------------------------------------- #
def test_empirical_coverage_known_counts():
    y_true = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.nan, 0.0])
    lower = np.array([-1.0, -1.0, -1.0, 0.5, -1.0, -1.0, -1.0, np.nan])
    upper = np.array([1.0, 1.0, 1.0, 1.5, -0.5, 0.0, 1.0, np.nan])
    # rows 0,1,2 covered; row 3 (0.5..1.5) not; row 4 (-1..-0.5) not; row 5 (-1..0) covered (inclusive)
    # row 6 y_true NaN -> excluded from coverage but has interval; row 7 no interval
    res = empirical_coverage(y_true, lower, upper)
    assert res["n_rows"] == 8
    assert res["n_with_interval"] == 7
    assert res["coverage"] == pytest.approx(4 / 6)
    assert res["median_width"] == pytest.approx(np.median([2.0, 2.0, 2.0, 1.0, 0.5, 1.0]))


def test_empirical_coverage_no_intervals():
    res = empirical_coverage(np.array([1.0, 2.0]), np.array([np.nan, np.nan]), np.array([np.nan, np.nan]))
    assert res["coverage"] is None
    assert res["median_width"] is None
    assert res["n_rows"] == 2
    assert res["n_with_interval"] == 0


# --------------------------------------------------------------------------- #
# Sigmoid / isotonic calibrators
# --------------------------------------------------------------------------- #
def _logistic_data(n=5000, seed=3, a=1.5, b=-0.5):
    rng = np.random.default_rng(seed)
    s = rng.normal(0, 2, n)
    p_true = 1 / (1 + np.exp(-(a * s + b)))
    y = (rng.uniform(size=n) < p_true).astype(int)
    return s, y, p_true


def test_sigmoid_calibrator_monotone_and_calibrated():
    s, y, p_true = _logistic_data()
    cal = SigmoidCalibrator().fit(s, y)
    grid = np.linspace(-6, 6, 50)
    p = cal.predict_proba(grid)
    assert p.shape == (50,)
    assert np.all(np.diff(p) >= 0)  # monotone increasing
    assert np.all((p >= 0) & (p <= 1))
    # recovered parameters near the generating ones
    assert cal.a == pytest.approx(1.5, abs=0.2)
    assert cal.b == pytest.approx(-0.5, abs=0.2)
    # roughly calibrated: mean abs error vs true prob is small
    p_hat = cal.predict_proba(s)
    assert np.mean(np.abs(p_hat - p_true)) < 0.03
    # NaN propagates
    assert math.isnan(cal.predict_proba(np.array([np.nan]))[0])


def test_sigmoid_calibrator_roundtrip():
    s, y, _ = _logistic_data(n=500)
    cal = SigmoidCalibrator().fit(s, y)
    cal2 = SigmoidCalibrator.from_dict(cal.to_dict())
    np.testing.assert_allclose(cal.predict_proba(s), cal2.predict_proba(s))


def test_sigmoid_requires_both_classes():
    with pytest.raises(ValueError):
        SigmoidCalibrator().fit(np.array([0.1, 0.2, 0.3]), np.array([0, 0, 0]))


def test_isotonic_calibrator_monotone_and_roundtrip():
    s, y, p_true = _logistic_data(n=3000, seed=4)
    cal = IsotonicCalibrator().fit(s, y)
    grid = np.linspace(-8, 8, 100)  # extends beyond training range -> clip
    p = cal.predict_proba(grid)
    assert np.all(np.diff(p) >= -1e-12)
    assert np.all((p >= 0) & (p <= 1))
    assert np.mean(np.abs(cal.predict_proba(s) - p_true)) < 0.05
    cal2 = IsotonicCalibrator.from_dict(cal.to_dict())
    np.testing.assert_allclose(cal.predict_proba(grid), cal2.predict_proba(grid))
    assert math.isnan(cal.predict_proba(np.array([np.nan]))[0])


# --------------------------------------------------------------------------- #
# classifier_metrics / reliability_diagram
# --------------------------------------------------------------------------- #
def test_classifier_metrics_undefined_without_positives():
    res = classifier_metrics(np.zeros(10, dtype=int), np.linspace(0, 1, 10))
    assert res["undefined"] is True
    assert res["pr_auc"] is None
    assert res["brier"] is None
    assert res["n_pos"] == 0
    assert res["n"] == 10

    empty = classifier_metrics(np.array([]), np.array([]))
    assert empty["undefined"] is True and empty["n"] == 0


def test_classifier_metrics_defined():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8])
    res = classifier_metrics(y, p)
    assert res["undefined"] is False
    assert res["n_pos"] == 2 and res["n"] == 4
    assert 0.0 <= res["pr_auc"] <= 1.0
    expected_brier = np.mean((p - y) ** 2)
    assert res["brier"] == pytest.approx(expected_brier)
    # perfect classifier -> PR-AUC 1
    perfect = classifier_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert perfect["pr_auc"] == pytest.approx(1.0)


def test_reliability_diagram_bins():
    y = np.array([0, 1, 1, 0, 1])
    p = np.array([0.05, 0.95, 0.96, 0.12, 1.0])
    rd = reliability_diagram(y, p, n_bins=10)
    assert len(rd["bin_lower"]) == 10
    assert rd["bin_lower"][0] == 0.0 and rd["bin_upper"][-1] == 1.0
    assert rd["count"] == [1, 1, 0, 0, 0, 0, 0, 0, 0, 3]
    assert rd["mean_pred"][0] == pytest.approx(0.05)
    assert rd["frac_pos"][0] == 0.0
    assert rd["mean_pred"][1] == pytest.approx(0.12)
    assert rd["frac_pos"][1] == 0.0
    assert rd["mean_pred"][-1] == pytest.approx(np.mean([0.95, 0.96, 1.0]))
    assert rd["frac_pos"][-1] == 1.0
    # empty bins -> None
    assert rd["mean_pred"][5] is None and rd["frac_pos"][5] is None
