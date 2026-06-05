"""Event + frame-metric logging (CSV/JSON) — the pipeline's output boundary.

Every run writes (to the configured outputs/logs dir):
    - events.csv          one row per risk event
    - frame_metrics.csv   per-frame stats (fps, counts, ...)
    - scene_summary.json  run-level summary

events.csv columns (CLAUDE.md §A9):
    frame, timestamp, track_id, class, risk, bbox_area, lane_overlap, event

ROS 2 mapping: rosbag + CSV logger (keep the CSV outputs for reports).
Implemented in Phase A9.
"""
from __future__ import annotations

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


class EventLogger:
    """Buffers per-frame metrics and risk events, then flushes them to disk."""

    def __init__(self, out_dir: str | Path) -> None:
        self.out_dir = Path(out_dir)
        # TODO(A9): mkdir(parents=True, exist_ok=True); open CSV buffers.
        raise NotImplementedError("EventLogger is implemented in Phase A9.")

    def log_event(self, row: dict[str, Any]) -> None:
        """Record one event row (keys = EVENT_COLUMNS)."""
        # TODO(A9)
        raise NotImplementedError

    def log_frame(self, metrics: dict[str, Any]) -> None:
        """Record one frame's metrics."""
        # TODO(A9)
        raise NotImplementedError

    def close(self, summary: dict[str, Any]) -> None:
        """Flush events.csv + frame_metrics.csv and write scene_summary.json."""
        # TODO(A9)
        raise NotImplementedError
