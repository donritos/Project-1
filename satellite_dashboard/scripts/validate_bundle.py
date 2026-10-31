"""Check that the artifact bundle is complete and consistent before a demo.

    python scripts/validate_bundle.py

Exit code 0 when everything passes. Checks: manifest + eligibility rule, every dataset loads through the adapter and
every event validates against the prediction contract, forecast inputs are all >= 2 days, outcomes are separate,
and the bundled model loads with the same model version as the precomputed predictions."""
from __future__ import annotations

import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP / "src"))

from satellite_dashboard.config import MIN_TIME_TO_TCA_DAYS  # noqa: E402
from satellite_dashboard.contracts import PredictionRecord  # noqa: E402
from satellite_dashboard.services.model_adapter import ModelAdapter, record_dict  # noqa: E402


def main() -> int:
    a = ModelAdapter()
    failures = []
    for key in ("heldout", "test", "synthetic"):
        try:
            v = a.load(key)
            for r in v.events.to_dict("records"):
                PredictionRecord(**record_dict(r))
            assert (v.inputs["time_to_tca_days"] >= MIN_TIME_TO_TCA_DAYS).all(), "ineligible inputs present"
            assert not {"final_tier", "observed_final_tier"} & set(v.events.columns), "outcomes leaked into predictions"
            n_bad = int((v.events["prediction_status"] != "Successful").sum())
            print(f"OK   {key:9s} {len(v.events):5,} events · {len(v.snapshots):6,} snapshot forecasts · {n_bad} without a usable prediction "
                  f"· model {v.model_version}" + (" · SYNTHETIC" if v.kind == "synthetic" else ""))
        except Exception as exc:
            failures.append(key)
            print(f"FAIL {key:9s} {exc}")
    try:
        b = a.model_bundle()
        mv = a.manifest().model_version
        assert b.meta["model_version"] == mv, f"bundle {b.meta['model_version']} != manifest {mv}"
        print(f"OK   model     bundle {mv} loads (uploads available)")
    except ImportError:
        print("WARN model     collision-risk package not installed: precomputed views work, uploads will not")
    except Exception as exc:
        failures.append("model")
        print(f"FAIL model     {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
