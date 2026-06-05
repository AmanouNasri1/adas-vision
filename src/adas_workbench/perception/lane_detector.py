"""Lane / drivable-area estimation (classical CV for the MVP).

Contract (CLAUDE.md §4):
    estimate(frame) -> {
        "ego_lane_polygon": list[[x, y], ...] | None,
        "lane_mask":        np.ndarray | None,
    }

Pipeline: ROI trapezoid -> grayscale -> blur -> Canny -> HoughLinesP -> split
segments into left/right by slope -> fit each side (x = m*y + b) -> ego-lane
quad. If clear lane lines are not found, it falls back to the ROI trapezoid as
the drivable-area estimate, so an overlay always renders. Stateless per frame.
A hook is left for a future lightweight segmentation model.
ROS 2 mapping: lane_drivable_node.py (publishes the lane mask image).
Implemented in Phase A7.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

LaneInfo = dict[str, Any]


class LaneDetector:
    """Estimates the ego-lane region from a single frame (no temporal state)."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        roi = cfg.get("roi", {})
        self.bottom_width = float(roi.get("bottom_width", 0.90))
        self.top_width = float(roi.get("top_width", 0.18))
        self.top_y = float(roi.get("top_y", 0.62))
        self.bottom_y = float(roi.get("bottom_y", 0.97))

        edges = cfg.get("edges", {})
        self.blur_kernel = int(edges.get("blur_kernel", 5)) | 1  # force odd
        self.canny_low = int(edges.get("canny_low", 60))
        self.canny_high = int(edges.get("canny_high", 150))

        hough = cfg.get("hough", {})
        self.rho = float(hough.get("rho", 2))
        self.theta = float(np.deg2rad(float(hough.get("theta_deg", 1))))
        self.threshold = int(hough.get("threshold", 25))
        self.min_line_length = int(hough.get("min_line_length", 18))
        self.max_line_gap = int(hough.get("max_line_gap", 120))

        lines = cfg.get("lines", {})
        self.min_abs_slope = float(lines.get("min_abs_slope", 0.5))
        self.max_abs_slope = float(lines.get("max_abs_slope", 6.0))

    def estimate(self, frame: np.ndarray) -> LaneInfo:
        """Return the ego-lane polygon and a filled lane mask for one frame."""
        h, w = frame.shape[:2]
        y_bottom = self.bottom_y * h
        y_top = self.top_y * h
        roi_poly = self._roi_polygon(w, h)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        edges = cv2.Canny(blur, self.canny_low, self.canny_high)
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, [roi_poly], 255)
        masked = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(
            masked,
            self.rho,
            self.theta,
            self.threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap,
        )
        left_pts, right_pts = self._collect_points(lines, w)
        left = self._fit_side(left_pts, y_bottom, y_top)
        right = self._fit_side(right_pts, y_bottom, y_top)

        polygon = self._lane_polygon(left, right, w)
        if polygon is None:
            # Graceful fallback: assume the ROI trapezoid as the drivable area.
            polygon = [[int(x), int(y)] for x, y in roi_poly.tolist()]

        lane_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(lane_mask, [np.array(polygon, dtype=np.int32)], 255)
        return {"ego_lane_polygon": polygon, "lane_mask": lane_mask}

    # --- internals -----------------------------------------------------------

    def _roi_polygon(self, w: int, h: int) -> np.ndarray:
        y_b = self.bottom_y * h
        y_t = self.top_y * h
        cx = w / 2.0
        bw = self.bottom_width * w / 2.0
        tw = self.top_width * w / 2.0
        return np.array(
            [[cx - bw, y_b], [cx - tw, y_t], [cx + tw, y_t], [cx + bw, y_b]],
            dtype=np.int32,
        )

    def _collect_points(
        self, lines: np.ndarray | None, w: int
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        left_pts: list[tuple[float, float]] = []
        right_pts: list[tuple[float, float]] = []
        if lines is None:
            return left_pts, right_pts
        for ln in lines:
            x1, y1, x2, y2 = ln[0]
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            a = abs(slope)
            if a < self.min_abs_slope or a > self.max_abs_slope:
                continue
            if slope < 0 and max(x1, x2) < w * 0.6:  # left lane line
                left_pts.extend([(x1, y1), (x2, y2)])
            elif slope > 0 and min(x1, x2) > w * 0.4:  # right lane line
                right_pts.extend([(x1, y1), (x2, y2)])
        return left_pts, right_pts

    def _fit_side(
        self,
        pts: list[tuple[float, float]],
        y_bottom: float,
        y_top: float,
    ) -> tuple[tuple[float, float], tuple[float, float]] | None:
        if len(pts) < 4:  # need a few points for a stable fit
            return None
        xs = np.array([p[0] for p in pts], dtype=float)
        ys = np.array([p[1] for p in pts], dtype=float)
        # Fit x = m*y + b (robust for near-vertical lane lines).
        m, b = np.polyfit(ys, xs, 1)
        return (m * y_bottom + b, y_bottom), (m * y_top + b, y_top)

    def _lane_polygon(
        self,
        left: tuple[tuple[float, float], tuple[float, float]] | None,
        right: tuple[tuple[float, float], tuple[float, float]] | None,
        w: int,
    ) -> list[list[int]] | None:
        if left is None or right is None:
            return None
        (lbx, lby), (ltx, lty) = left
        (rbx, rby), (rtx, rty) = right
        # Sanity: left edge must stay left of the right edge at both ends.
        if lbx >= rbx or ltx >= rtx:
            return None

        def clamp(v: float) -> int:
            return int(max(0, min(w - 1, v)))

        return [
            [clamp(lbx), int(lby)],
            [clamp(ltx), int(lty)],
            [clamp(rtx), int(rty)],
            [clamp(rbx), int(rby)],
        ]
