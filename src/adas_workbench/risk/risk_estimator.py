"""Risk scoring — combines pseudo-distance, approach, lane overlap, class weight.

Contract (CLAUDE.md §4):
    score(tracks, lane_info, prev_state) -> list[RiskResult]
    RiskResult = {
        "track_id":   int,
        "risk_score": float,   # 0-100
        "risk_level": str,     # "LOW" | "MEDIUM" | "HIGH"
        ... plus extras used by the overlay + event log (class_name, bbox,
            ttc_proxy_s, lane_overlap, bbox_area)
    }

Score blends (per ``configs/risk.yaml`` weights): bbox closeness, bbox
growth/approach, ego-lane overlap, and class weight (person > bicycle > car >
traffic light), plus a pseudo-TTC term. Thresholds: 0-30 LOW, 31-70 MEDIUM,
71-100 HIGH.
ROS 2 mapping: risk_estimation_node.py (proxy -> depth-based TTC).
Implemented in Phase A8.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .distance_estimator import DistanceEstimator, clamp01

Track = dict[str, Any]
LaneInfo = dict[str, Any]
RiskResult = dict[str, Any]


class RiskEstimator:
    """Computes a 0-100 risk score and a LOW/MEDIUM/HIGH level per track."""

    def __init__(self, config: dict[str, Any], fps: float = 30.0) -> None:
        self.config = config
        levels = config.get("levels", {})
        self.low_max = float(levels.get("low_max", 30))
        self.medium_max = float(levels.get("medium_max", 70))

        self.class_weights = config.get("class_weights", {})
        self.default_weight = float(self.class_weights.get("default", 0.5))

        cues = config.get("cues", {})
        self.w_area = float(cues.get("bbox_area_weight", 0.35))
        self.w_growth = float(cues.get("bbox_growth_weight", 0.25))
        self.w_lane = float(cues.get("lane_overlap_weight", 0.25))
        self.w_ttc = float(cues.get("ttc_proxy_weight", 0.15))

        self.fps = float(fps)
        self._dist = DistanceEstimator(config)

    def score(
        self,
        tracks: list[Track],
        lane_info: LaneInfo,
        prev_state: dict[int, list[float]] | None = None,
    ) -> list[RiskResult]:
        """Return a RiskResult per track for the current frame."""
        prev_state = prev_state or {}
        lane_mask = lane_info.get("lane_mask") if lane_info else None
        if lane_mask is not None:
            frame_h, frame_w = lane_mask.shape[:2]
        else:
            frame_h = frame_w = None

        results: list[RiskResult] = []
        for trk in tracks:
            bbox = trk["bbox"]
            track_id = trk["track_id"]
            class_name = trk["class_name"]
            prev_bbox = prev_state.get(track_id)

            closeness = self._dist.closeness(bbox, frame_w, frame_h)
            growth = self._dist.growth(bbox, prev_bbox)
            lane_overlap = self._lane_overlap(bbox, lane_mask)
            ttc_risk = self._dist.ttc_risk(bbox, prev_bbox, self.fps)
            ttc_seconds = self._dist.ttc_proxy_seconds(bbox, prev_bbox, self.fps)

            raw = (
                self.w_area * closeness
                + self.w_growth * growth
                + self.w_lane * lane_overlap
                + self.w_ttc * ttc_risk
            )
            weight = float(self.class_weights.get(class_name, self.default_weight))
            score = clamp01(raw * weight) * 100.0

            results.append(
                {
                    "track_id": track_id,
                    "class_name": class_name,
                    "bbox": [float(v) for v in bbox],
                    "risk_score": round(score, 1),
                    "risk_level": self.level_for(score),
                    "ttc_proxy_s": None if ttc_seconds == float("inf") else round(ttc_seconds, 2),
                    "lane_overlap": round(lane_overlap, 3),
                    "bbox_area": round(self._dist.area(bbox), 1),
                }
            )
        return results

    def level_for(self, risk_score: float) -> str:
        """Map a 0-100 score to LOW / MEDIUM / HIGH using risk.yaml thresholds."""
        if risk_score <= self.low_max:
            return "LOW"
        if risk_score <= self.medium_max:
            return "MEDIUM"
        return "HIGH"

    @staticmethod
    def _lane_overlap(bbox: list[float], lane_mask: np.ndarray | None) -> float:
        """Lane membership in [0, 1] via the object's ground contact.

        Samples the bbox bottom edge (the object's footprint on the road) against
        the lane mask. This reflects "is this object standing in my lane?" far
        better than whole-box area overlap, since a vehicle's box extends upward
        out of the road-surface region and would otherwise dilute the score.
        """
        if lane_mask is None:
            return 0.0
        h, w = lane_mask.shape[:2]
        y_bottom = int(min(max(bbox[3], 0), h - 1))
        x1 = max(0, int(bbox[0]))
        x2 = min(w, int(bbox[2]))
        if x2 <= x1:
            x_mid = int(min(max((bbox[0] + bbox[2]) / 2.0, 0), w - 1))
            return 1.0 if lane_mask[y_bottom, x_mid] > 0 else 0.0
        strip = lane_mask[y_bottom, x1:x2]
        return float(np.count_nonzero(strip)) / float(x2 - x1)
