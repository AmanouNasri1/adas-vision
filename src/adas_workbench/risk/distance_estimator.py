"""Pseudo-distance & pseudo-TTC estimation (2-D bounding-box proxies).

WARNING: there is NO real depth here. "Distance" is inferred from bounding-box
geometry (area as a closeness proxy) and "TTC" from how fast the box grows
(bbox_height / bbox_height_change_rate). These are PROXIES only — never present
them as physical metres or seconds (CLAUDE.md hard rule; see
docs/limitations.md). Real depth-based distance/TTC arrives in Project B.

ROS 2 mapping: folded into risk_estimation_node.py, where the proxy is upgraded
to depth-based TTC.
Implemented in Phase A8.
"""
from __future__ import annotations

import math
from typing import Any

BBox = list[float]


def clamp01(value: float) -> float:
    """Clamp a value into [0, 1]."""
    return max(0.0, min(1.0, value))


class DistanceEstimator:
    """Turns bbox geometry + history into closeness and approach proxies."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        ttc = config.get("ttc_proxy", {})
        self.min_height_change = float(ttc.get("min_height_change", 0.5))  # px/frame
        self.clamp_seconds = float(ttc.get("clamp_seconds", 10.0))
        norm = config.get("normalize", {})
        self.area_full_frac = float(norm.get("area_full_frac", 0.15))
        self.growth_full_rate = float(norm.get("growth_full_rate", 0.06))

    @staticmethod
    def height(bbox: BBox) -> float:
        return max(1.0, float(bbox[3]) - float(bbox[1]))

    @staticmethod
    def area(bbox: BBox) -> float:
        w = max(0.0, float(bbox[2]) - float(bbox[0]))
        h = max(0.0, float(bbox[3]) - float(bbox[1]))
        return w * h

    def closeness(self, bbox: BBox, frame_w: int | None, frame_h: int | None) -> float:
        """Closeness proxy in [0, 1] from bbox area fraction (1 ~ very close)."""
        if not frame_w or not frame_h:
            return 0.0
        frac = self.area(bbox) / (float(frame_w) * float(frame_h))
        return clamp01(frac / self.area_full_frac)

    def growth(self, bbox: BBox, prev_bbox: BBox | None) -> float:
        """Approach proxy in [0, 1] from positive fractional height growth/frame."""
        if prev_bbox is None:
            return 0.0
        h_now = self.height(bbox)
        h_prev = self.height(prev_bbox)
        rate = (h_now - h_prev) / h_prev
        return clamp01(rate / self.growth_full_rate)

    def ttc_proxy_seconds(self, bbox: BBox, prev_bbox: BBox | None, fps: float) -> float:
        """Pseudo-TTC in seconds-equivalent (NOT real seconds).

        Returns ``inf`` when the object is not approaching (box not growing).
        """
        if prev_bbox is None or fps <= 0:
            return math.inf
        h_now = self.height(bbox)
        dh = h_now - self.height(prev_bbox)  # px/frame
        if dh < self.min_height_change:
            return math.inf  # not approaching (or receding)
        frames_to_collision = h_now / dh
        return frames_to_collision / fps

    def ttc_risk(self, bbox: BBox, prev_bbox: BBox | None, fps: float) -> float:
        """Convert the pseudo-TTC into a [0, 1] risk (sooner -> higher)."""
        ttc = self.ttc_proxy_seconds(bbox, prev_bbox, fps)
        if math.isinf(ttc):
            return 0.0
        return clamp01(1.0 - ttc / self.clamp_seconds)
