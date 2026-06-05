"""Lane / drivable-area estimation (classical CV for the MVP).

Contract (CLAUDE.md §4):
    estimate(frame) -> {
        "ego_lane_polygon": list[[x, y], ...] | None,
        "lane_mask":        np.ndarray | None,
    }

MVP pipeline: ROI crop -> colour/edge threshold -> Canny -> Hough lines ->
ego-lane polygon. A hook is left for a future lightweight segmentation model.
ROS 2 mapping: lane_drivable_node.py (publishes the lane mask image).
Implemented in Phase A7.
"""
from __future__ import annotations

from typing import Any

import numpy as np

LaneInfo = dict[str, Any]


class LaneDetector:
    """Estimates the ego-lane region from a single frame (no temporal state)."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        # TODO(A7): precompute ROI / thresholds from config.
        raise NotImplementedError("LaneDetector is implemented in Phase A7.")

    def estimate(self, frame: np.ndarray) -> LaneInfo:
        """Return the ego-lane polygon and optional lane mask for one frame."""
        # TODO(A7): classical CV pipeline; leave a segmentation-model hook.
        raise NotImplementedError("LaneDetector.estimate is implemented in Phase A7.")
