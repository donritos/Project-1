"""Regenerate the synthetic demo fixtures in artifacts/demo/ (no model or data needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from satellite_dashboard.config import DEMO_DIR  # noqa: E402
from satellite_dashboard.fixtures import build  # noqa: E402

if __name__ == "__main__":
    print(f"wrote synthetic fixtures to {build(DEMO_DIR)}")
