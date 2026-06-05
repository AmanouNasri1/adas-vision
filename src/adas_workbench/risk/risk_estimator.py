"""Risk scoring — combines pseudo-distance, approach, lane overlap, class weight.

Contract (CLAUDE.md §4):
    score(tracks, lane_info, prev_state) -> list[RiskResult]
    RiskResult = {
        "track_id":   int,
        "risk_score": float,   # 0-100
        "risk_level": str,     # "LOW" | "MEDIUM" | "HIGH"
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

Track = dict[str, Any]
LaneInfo = dict[str, Any]
RiskResult = dict[str, Any]


class RiskEstimator:
    """Computes a 0-100 risk score and a LOW/MEDIUM/HIGH level per track."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        # TODO(A8): pull levels, class_weights, and cue weights from config.
        raise NotImplementedError("RiskEstimator is implemented in Phase A8.")

    def score(
        self,
        tracks: list[Track],
        lane_info: LaneInfo,
        prev_state: dict[int, Track] | None = None,
    ) -> list[RiskResult]:
        """Return a RiskResult per track for the current frame."""
        # TODO(A8): blend cues -> 0-100 -> map to level via thresholds.
        raise NotImplementedError("RiskEstimator.score is implemented in Phase A8.")

    def level_for(self, risk_score: float) -> str:
        """Map a 0-100 score to LOW / MEDIUM / HIGH using risk.yaml thresholds."""
        # TODO(A8)
        raise NotImplementedError
