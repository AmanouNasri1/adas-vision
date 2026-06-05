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

Greedy, class-aware IoU matching: each frame, detections are matched to existing
tracks by highest IoU (same class only). Matched tracks adopt the new box;
unmatched detections spawn new ids; tracks unseen for too long are dropped. The
association step is isolated in ``_associate`` so a SORT/Kalman tracker can drop
in later (predict -> associate -> update).
ROS 2 mapping: tracking_node.py (publishes tracks as a custom msg / JSON).
Implemented in Phase A6.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

Detection = dict[str, Any]
Track = dict[str, Any]


def _iou(box_a: list[float], box_b: list[float]) -> float:
    """Intersection-over-union of two xyxy boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


@dataclass
class _TrackState:
    """Internal mutable state for one tracked object."""

    track_id: int
    class_name: str
    bbox: list[float]
    age: int = 0
    missed_frames: int = 0
    hits: int = 1


class Tracker:
    """Assigns persistent track ids by matching detections across frames.

    Per-instance state only (active tracks + the next id to assign); this is a
    node's internal state, not global state.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        tcfg = config.get("tracker", {})
        self.iou_threshold = float(tcfg.get("iou_match_threshold", 0.3))
        self.max_missed_frames = int(tcfg.get("max_missed_frames", 30))
        self.min_hits = int(tcfg.get("min_hits", 3))
        self._next_id = int(tcfg.get("first_track_id", 1))
        self._tracks: list[_TrackState] = []

    def update(self, detections: list[Detection], frame_id: int) -> list[Track]:
        """Match ``detections`` to existing tracks; return live, confirmed tracks."""
        # Age every existing track by one frame.
        for t in self._tracks:
            t.age += 1

        matched, unmatched_tracks, unmatched_dets = self._associate(
            self._tracks, detections
        )

        # Matched: adopt the new box, reset the miss counter.
        for ti, di in matched:
            det = detections[di]
            track = self._tracks[ti]
            track.bbox = [float(v) for v in det["bbox_xyxy"]]
            track.class_name = det["class_name"]
            track.hits += 1
            track.missed_frames = 0

        # Unmatched existing tracks coast (missed++).
        for ti in unmatched_tracks:
            self._tracks[ti].missed_frames += 1

        # Unmatched detections become new tracks.
        for di in unmatched_dets:
            det = detections[di]
            self._tracks.append(
                _TrackState(
                    track_id=self._next_id,
                    class_name=det["class_name"],
                    bbox=[float(v) for v in det["bbox_xyxy"]],
                )
            )
            self._next_id += 1

        # Drop tracks unseen for too long.
        self._tracks = [
            t for t in self._tracks if t.missed_frames <= self.max_missed_frames
        ]

        # Return currently-visible, confirmed tracks (suppresses 1-frame noise).
        return [
            self._to_dict(t)
            for t in self._tracks
            if t.missed_frames == 0 and t.hits >= self.min_hits
        ]

    def _associate(
        self, tracks: list[_TrackState], detections: list[Detection]
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Greedy, class-aware IoU matching.

        Returns (matched_pairs, unmatched_track_indices, unmatched_det_indices).
        Isolated so a SORT/Kalman association can replace it without touching the
        bookkeeping in ``update``.
        """
        candidates: list[tuple[float, int, int]] = []
        for ti, t in enumerate(tracks):
            for di, d in enumerate(detections):
                if t.class_name != d["class_name"]:
                    continue  # only match within the same class
                iou = _iou(t.bbox, d["bbox_xyxy"])
                if iou >= self.iou_threshold:
                    candidates.append((iou, ti, di))

        candidates.sort(reverse=True)  # highest IoU first
        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        matched: list[tuple[int, int]] = []
        for _iou_val, ti, di in candidates:
            if ti in used_tracks or di in used_dets:
                continue
            matched.append((ti, di))
            used_tracks.add(ti)
            used_dets.add(di)

        unmatched_tracks = [ti for ti in range(len(tracks)) if ti not in used_tracks]
        unmatched_dets = [di for di in range(len(detections)) if di not in used_dets]
        return matched, unmatched_tracks, unmatched_dets

    @staticmethod
    def _to_dict(t: _TrackState) -> Track:
        return {
            "track_id": t.track_id,
            "class_name": t.class_name,
            "bbox": list(t.bbox),
            "age": t.age,
            "missed_frames": t.missed_frames,
        }
