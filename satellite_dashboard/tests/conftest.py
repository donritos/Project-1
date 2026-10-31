import os
import sys
import tempfile
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP / "src"))
# The model package is a declared dependency; fall back to the sibling checkout when running tests without installing it.
try:
    import collision_risk  # noqa: F401
except ImportError:
    sys.path.insert(0, str(APP.parent / "src"))

os.environ.setdefault("SATDASH_REVIEW_DB", str(Path(tempfile.mkdtemp()) / "reviews.sqlite"))

from satellite_dashboard.config import ARTIFACTS_DIR  # noqa: E402
from satellite_dashboard.services.model_adapter import ModelAdapter  # noqa: E402

HAS_REAL = (ARTIFACTS_DIR / "manifest.json").exists()
import importlib.util  # noqa: E402

HAS_MODEL = importlib.util.find_spec("collision_risk") is not None

needs_real = pytest.mark.skipif(not HAS_REAL, reason="real artifacts not built (scripts/build_artifacts.py)")
needs_model = pytest.mark.skipif(not (HAS_REAL and HAS_MODEL), reason="model package or bundle not available")


@pytest.fixture(scope="session")
def adapter():
    return ModelAdapter()


@pytest.fixture(scope="session")
def heldout(adapter):
    return adapter.load("heldout")


@pytest.fixture(scope="session")
def test_view(adapter):
    return adapter.load("test")


@pytest.fixture(scope="session")
def synthetic(adapter):
    return adapter.load("synthetic")
