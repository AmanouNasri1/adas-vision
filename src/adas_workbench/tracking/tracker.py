"""Multi-object tracking (MVP: IoU-based id association).

Contract (CLAUDE.md §4):
    update(detections, frame_id) -> list[Track]
    Track = {
        "track_id":      int,
        "class_name":    str,
        "bbox":          [x1, y1, x2, y2],
        "age":           int,   # frames since the track was created
        "missed_frames": int,   # consecutive frames without a match
    }

Structured so a SORT/Kalman tracker can replace the matching step later.
ROS 2 mapping: tracking_node.py (publishes tracks as a custom msg / JSON).
Implemented in Phase A6.
"""
from __future__ import annotations

from typing import Any

Detection = dict[str, Any]
Track = dict[str, Any]


class Tracker:
    """Assigns persistent track ids by matching detections across frames.

    Per-instance state only (active tracks + the next id to assign); this is a
    node's internal state, not global state.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._next_id = int(config.get("tracker", {}).get("first_track_id", 1))
        self._tracks: list[Track] = []
        # TODO(A6): maintain active tracks with age / missed_frames bookkeeping.
        raise NotImplementedError("Tracker is implemented in Phase A6.")

    def update(self, detections: list[Detection], frame_id: int) -> list[Track]:
        """Match ``detections`` to existing tracks; return the live tracks."""
        # TODO(A6): IoU match -> update/create/age-out -> return tracks.
        raise NotImplementedError("Tracker.update is implemented in Phase A6.")
