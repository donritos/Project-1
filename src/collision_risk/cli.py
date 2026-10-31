"""Command-line entry points: audit, train, predict, evaluate, run-all."""
from __future__ import annotations

import argparse
import json
import sys

from .config import load_config


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="collision-risk", description="Satellite conjunction triage pipeline")
    p.add_argument("--config", default=None, help="Path to YAML config (default configs/default.yaml)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("audit", help="Audit raw files, cache allowlisted parquet, write artifacts/audit.json")
    t = sub.add_parser("train", help="80/20 event split, grouped OOF search, candidates, calibration, thresholds, holdout evaluation")
    t.add_argument("--quick", action="store_true", help="Small search/rounds for smoke testing")
    pr = sub.add_parser("predict", help="Forecast final tiers for a CSV of CDMs and export event/snapshot forecasts + submission.csv")
    pr.add_argument("--input", default=None, help="CSV path (default: configured test file)")
    pr.add_argument("--dataset-id", default=None)
    pr.add_argument("--out-dir", default=None)
    pr.add_argument("--bundle", default=None, help="Model bundle directory (default artifacts/model)")
    sub.add_parser("evaluate", help="Print the headline comparison table from artifacts/evaluation.json")
    ra = sub.add_parser("run-all", help="audit -> train -> predict")
    ra.add_argument("--quick", action="store_true")
    a = p.parse_args(argv)
    cfg = load_config(a.config)

    from . import pipeline
    if a.cmd == "audit":
        rep = pipeline.stage_audit(cfg)
        print(json.dumps(rep["checks"], indent=2))
    elif a.cmd == "train":
        ev = pipeline.stage_train(cfg, quick=a.quick)
        _print_table(ev)
    elif a.cmd == "predict":
        m = pipeline.stage_predict(cfg, input_path=a.input, dataset_id=a.dataset_id, out_dir=a.out_dir, bundle_path=a.bundle)
        print(json.dumps({k: m[k] for k in ("n_events", "n_snapshots", "predicted_class_distribution", "triage_counts", "batch_inference_seconds")}, indent=2))
    elif a.cmd == "evaluate":
        from .service import get_evaluation_report
        _print_table(get_evaluation_report())
    elif a.cmd == "run-all":
        pipeline.run_all(cfg, quick=a.quick)
    return 0


def _print_table(ev: dict) -> None:
    import pandas as pd
    pd.set_option("display.width", 220)
    print(f"primary model: {ev['primary_model']}  ({ev['model_version']})  thresholds={ev.get('decision_thresholds')}")
    print("
Holdout, real_forecast events (final CDM < 2 days; the answer is not visible in the input):")
    print(pd.DataFrame(ev["comparison_real_forecast"]).to_string(index=False))
    print("
Holdout, all usable events:")
    print(pd.DataFrame(ev["comparison_all_usable"]).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
