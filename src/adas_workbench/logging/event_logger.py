"""Event + frame-metric logging (CSV/JSON) — the pipeline's output boundary.

Every run writes (to the configured outputs/logs dir):
    - events.csv          one row per notable risk event (MEDIUM/HIGH)
    - frame_metrics.csv   per-frame stats (counts, max risk, ...)
    - scene_summary.json  run-level summary

events.csv columns (CLAUDE.md §A9):
    frame, timestamp, track_id, class, risk, bbox_area, lane_overlap, event

CSV rows are streamed to disk as they arrive (no unbounded in-memory buffer), so
long videos log fine. ROS 2 mapping: rosbag + CSV logger (keep CSVs for reports).
Implemented in Phase A9.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

EVENT_COLUMNS = [
    "frame",
    "timestamp",
    "track_id",
    "class",
    "risk",
    "bbox_area",
    "lane_overlap",
    "event",
]
FRAME_COLUMNS = [
    "frame",
    "timestamp",
    "detections",
    "tracks",
    "low",
    "medium",
    "high",
    "max_risk",
]


class EventLogger:
    """Streams risk events + frame metrics to CSV, then writes a JSON summary."""

    def __init__(self, out_dir: str | Path) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.out_dir / "events.csv"
        self.frame_metrics_path = self.out_dir / "frame_metrics.csv"
        self.summary_path = self.out_dir / "scene_summary.json"

        self._events_file = self.events_path.open("w", newline="", encoding="utf-8")
        self._frames_file = self.frame_metrics_path.open("w", newline="", encoding="utf-8")
        self._events_writer = csv.DictWriter(
            self._events_file, fieldnames=EVENT_COLUMNS, extrasaction="ignore"
        )
        self._frames_writer = csv.DictWriter(
            self._frames_file, fieldnames=FRAME_COLUMNS, extrasaction="ignore"
        )
        self._events_writer.writeheader()
        self._frames_writer.writeheader()

        self.event_count = 0
        self.frame_count = 0

    def log_event(self, row: dict[str, Any]) -> None:
        """Record one event row (keys = EVENT_COLUMNS; extras are ignored)."""
        self._events_writer.writerow(row)
        self.event_count += 1

    def log_frame(self, metrics: dict[str, Any]) -> None:
        """Record one frame's metrics (keys = FRAME_COLUMNS; extras are ignored)."""
        self._frames_writer.writerow(metrics)
        self.frame_count += 1

    def close(self, summary: dict[str, Any] | None = None) -> None:
        """Close the CSVs and write scene_summary.json (if a summary is given)."""
        if not self._events_file.closed:
            self._events_file.close()
        if not self._frames_file.closed:
            self._frames_file.close()
        if summary is not None:
            with self.summary_path.open("w", encoding="utf-8") as fh:
                json.dump(summary, fh, indent=2)
