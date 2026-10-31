"""Analyst review notes in a local SQLite file. Append-only: every save adds a row, so the history of an
event's reviews is kept. Reviews are keyed to the event AND the prediction they were made against, and never
change the prediction itself (acknowledging a Critical event leaves it Critical and in every count)."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from ..config import REVIEW_DB

REVIEW_STATUSES = ["Not reviewed", "Acknowledged", "Follow-up required", "Escalated to flight dynamics", "Closed - no further action"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS reviews (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id    TEXT    NOT NULL,
    event_id      INTEGER NOT NULL,
    prediction_id TEXT    NOT NULL,
    model_version TEXT    NOT NULL,
    predicted_final_tier TEXT,
    status        TEXT    NOT NULL,
    note          TEXT,
    follow_up     TEXT,
    reviewer      TEXT,
    created_at    TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_reviews_event ON reviews(dataset_id, event_id);
"""


class ReviewStore:
    def __init__(self, path: Path = REVIEW_DB) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def add(self, *, dataset_id: str, event_id: int, prediction_id: str, model_version: str, status: str,
            note: str = "", follow_up: str = "", reviewer: str = "", predicted_final_tier: Optional[str] = None) -> int:
        if status not in REVIEW_STATUSES:
            raise ValueError(f"unknown review status {status!r}")
        if not prediction_id:
            raise ValueError("a review must reference the prediction it was made against")
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO reviews (dataset_id, event_id, prediction_id, model_version, predicted_final_tier, status, note,"
                " follow_up, reviewer, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (dataset_id, int(event_id), prediction_id, model_version, predicted_final_tier, status, note.strip(),
                 follow_up.strip(), reviewer.strip(), datetime.now(timezone.utc).isoformat(timespec="seconds")))
            return int(cur.lastrowid)

    def history(self, dataset_id: str, event_id: int) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql_query("SELECT * FROM reviews WHERE dataset_id=? AND event_id=? ORDER BY id DESC", c,
                                     params=(dataset_id, int(event_id)))

    def latest_status(self, dataset_id: str) -> pd.DataFrame:
        """Latest review per (event, prediction) in a dataset."""
        with self._conn() as c:
            return pd.read_sql_query(
                "SELECT r.* FROM reviews r JOIN (SELECT dataset_id, event_id, prediction_id, MAX(id) AS mid FROM reviews"
                " WHERE dataset_id=? GROUP BY dataset_id, event_id, prediction_id) m ON r.id = m.mid", c, params=(dataset_id,))


def review_status_column(events: pd.DataFrame, latest: pd.DataFrame) -> pd.Series:
    """Analyst review status for each event's CURRENT prediction. A review of an earlier prediction of the same
    event is reported as such rather than carried over."""
    if latest.empty:
        return pd.Series("Not reviewed", index=events.index)
    cur = events[["event_id", "prediction_id"]].merge(latest[["event_id", "prediction_id", "status"]],
                                                     on=["event_id", "prediction_id"], how="left")["status"]
    earlier = set(latest["event_id"].astype(int))
    out = []
    for (ev, s) in zip(events["event_id"].astype(int), cur):
        if isinstance(s, str):
            out.append(s)
        elif ev in earlier:
            out.append("Reviewed earlier prediction")
        else:
            out.append("Not reviewed")
    return pd.Series(out, index=events.index)
