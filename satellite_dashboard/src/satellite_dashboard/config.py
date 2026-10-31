"""Paths and fixed rules. Every path is relative to the app folder (no machine-specific paths).

Environment overrides:
  SATDASH_ARTIFACTS   folder holding manifest.json, predictions/, model/ (default: <app>/artifacts)
  SATDASH_REVIEW_DB   SQLite file for analyst notes (default: <app>/data/local/reviews.sqlite)
"""
from __future__ import annotations

import os
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = Path(os.environ.get("SATDASH_ARTIFACTS", APP_ROOT / "artifacts"))
DEMO_DIR = APP_ROOT / "artifacts" / "demo"
REVIEW_DB = Path(os.environ.get("SATDASH_REVIEW_DB", APP_ROOT / "data" / "local" / "reviews.sqlite"))
SAMPLE_INPUTS_DIR = APP_ROOT / "data" / "sample_inputs"

# Challenge rule: only CDMs at least this many days before TCA may contribute to a forecast.
MIN_TIME_TO_TCA_DAYS = 2.0
ELIGIBILITY_EPS = 1e-12

# Official tier thresholds (metres, log10 Pc). Used only to label the latest OBSERVED tier and outcomes;
# forecasts come from the model.
CRITICAL_RISK = -4.0
ELEVATED_RISK = -6.0
CRITICAL_DISTANCE_M = 1000.0
ELEVATED_DISTANCE_MAX_M = 5000.0
RISK_FLOOR = -30.0  # placeholder for "too small to compute" (Pc ~ 0)

MAX_UPLOAD_MB = 25
MAX_UPLOAD_ROWS = 100_000

DATASET_KEYS = ("heldout", "test", "synthetic")
