"""Streamlit AppTest: every page renders for every dataset, banners are present, missing capabilities degrade gracefully,
and missing real artifacts raise an error instead of switching to fixtures."""
import pytest
from streamlit.testing.v1 import AppTest

from conftest import APP, HAS_REAL, needs_real
from satellite_dashboard.services.artifact_loader import ArtifactError
from satellite_dashboard.services.model_adapter import ModelAdapter

PAGES = ["overview", "event_detail", "model_evidence", "upload"]


def _run(module: str, dataset: str) -> AppTest:
    code = f'''
import importlib, sys
sys.path.insert(0, {str(APP / "src")!r})
sys.path.insert(1, {str(APP.parent / "src")!r})
from satellite_dashboard.components.status import inject_css
inject_css()
importlib.import_module("satellite_dashboard.views.{module}").render()
'''
    at = AppTest.from_string(code, default_timeout=180)
    at.session_state["dataset_key"] = dataset
    at.run()
    return at


DATASETS = ["synthetic"] + (["heldout", "test"] if HAS_REAL else [])


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("module", PAGES)
def test_every_page_renders(module, dataset):
    at = _run(module, dataset)
    assert not at.exception, [e.message for e in at.exception]


def test_synthetic_banner_is_always_shown():
    for module in ("overview", "event_detail", "model_evidence"):
        at = _run(module, "synthetic")
        assert any("Synthetic demo data" in m.value for m in at.markdown), module


def test_missing_capabilities_render_honest_placeholders():
    at = _run("event_detail", "synthetic")
    text = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption) + " ".join(i.value for i in at.info)
    assert "Uncertainty interval unavailable" in text
    assert "Feature contributions unavailable" in text


def test_test_set_never_reports_performance():
    if not HAS_REAL:
        pytest.skip("real artifacts not built")
    at = _run("model_evidence", "test")
    assert any("no test-set accuracy" in i.value for i in at.info)
    assert not at.dataframe


def test_missing_real_artifacts_raise_instead_of_using_fixtures(tmp_path):
    with pytest.raises(ArtifactError):
        ModelAdapter(artifacts_dir=tmp_path).load("heldout")


@needs_real
def test_evidence_metrics_match_the_model_report(heldout):
    import json
    from satellite_dashboard.config import ARTIFACTS_DIR
    from satellite_dashboard.services.metrics import evaluation_frame, tier_metrics
    ev = json.loads((ARTIFACTS_DIR / "evaluation.json").read_text())
    ref = ev["models"][ev["primary_model"]]["subsets"]["real_forecast"]["classification"]
    ef = evaluation_frame(heldout.events, heldout.outcomes)
    sub = ef[~ef["is_trivial"].astype(bool)]
    m = tier_metrics(sub["observed_final_tier"], sub["predicted_final_tier"])
    for k in ("critical_recall", "critical_precision", "critical_f2", "macro_f1", "accuracy"):
        assert abs(m[k] - ref[k]) < 1e-9, k
