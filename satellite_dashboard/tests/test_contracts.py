import json

import pytest

from conftest import needs_real
from satellite_dashboard.contracts import Manifest, PredictionRecord, PredictionStatus
from satellite_dashboard.formatting import km, log10_pc
from satellite_dashboard.services.model_adapter import record_dict

BASE = dict(dataset_id="d", event_id=1, prediction_id="p", model_version="m")


def test_successful_prediction_needs_a_tier():
    with pytest.raises(ValueError):
        PredictionRecord(**BASE, prediction_status=PredictionStatus.SUCCESS)


@pytest.mark.parametrize("status", [PredictionStatus.INVALID, PredictionStatus.INSUFFICIENT, PredictionStatus.FAILED])
def test_missing_predictions_never_carry_a_tier(status):
    with pytest.raises(ValueError):
        PredictionRecord(**BASE, prediction_status=status, predicted_final_tier="Safe")
    r = PredictionRecord(**BASE, prediction_status=status)
    assert r.predicted_final_tier is None and r.p_safe is None


def test_probabilities_must_be_complete_and_sum_to_one():
    with pytest.raises(ValueError):
        PredictionRecord(**BASE, prediction_status="Successful", predicted_final_tier="Safe", p_safe=0.9, p_elevated=0.05, p_critical=0.2)
    with pytest.raises(ValueError):
        PredictionRecord(**BASE, prediction_status="Successful", predicted_final_tier="Safe", p_safe=1.0)


def test_forecast_input_inside_cutoff_is_rejected():
    with pytest.raises(ValueError):
        PredictionRecord(**BASE, prediction_status="Successful", predicted_final_tier="Safe", latest_input_time_to_tca_days=1.99)
    PredictionRecord(**BASE, prediction_status="Successful", predicted_final_tier="Safe", latest_input_time_to_tca_days=2.0)


def test_manifest_with_another_eligibility_rule_is_rejected(tmp_path):
    m = {"schema_version": "x", "model": {}, "eligibility": {"min_time_to_tca_days": 1.0}, "capabilities": {}, "datasets": {}}
    with pytest.raises(ValueError):
        Manifest.model_validate(m)


def test_units_and_placeholder_display():
    assert km(1500.0) == "1.50 km"
    assert km(None) == "–"
    assert log10_pc(-30.0).startswith("≈ 0")
    assert "1.0e-05" in log10_pc(-5.0)


@needs_real
def test_every_real_record_validates(heldout, test_view, synthetic):
    for v in (heldout, test_view, synthetic):
        recs = [PredictionRecord(**record_dict(r)) for r in v.events.to_dict("records")]
        assert len(recs) == len(v.events)


@needs_real
def test_manifest_declares_missing_capabilities_honestly(heldout):
    caps = heldout.capabilities
    assert caps.class_probabilities and caps.feature_contributions
    assert not caps.uncertainty_intervals and not caps.predicted_final_pc
    man = json.loads((heldout.manifest and __import__("satellite_dashboard.config", fromlist=["x"]).ARTIFACTS_DIR / "manifest.json").read_text())
    for key in ("heldout", "test"):
        chk = man["datasets"][key]["reference_check"]
        assert chk["checked"] and chk["tier_agreement"] == 1.0 and chk["max_abs_diff_p_critical"] == 0.0
