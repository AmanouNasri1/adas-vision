"""Lane / drivable-area estimation.

Two backends, selected by ``method`` in ``configs/lane.yaml``:

- ``"classical"`` (default): ROI + grayscale + Canny + HoughLinesP -> ego-lane
  quad, with a graceful fallback to the ROI trapezoid. No extra deps; fragile on
  faded markings / clutter (see docs/limitations.md).
- ``"learned"``: a fine-tuned YOLO segmentation model (e.g. trained on BDD100K
  drivable area — see ``training/colab_lane_training.md``) -> drivable-area mask.
  Falls back to classical for a frame if the model is missing or returns nothing.

Contract (unchanged across backends):
    estimate(frame) -> {
        "ego_lane_polygon": list[[x, y], ...] | None,
        "lane_mask":        np.ndarray | None,
    }

Stateless per frame. ROS 2 mapping: lane_drivable_node.py.
Implemented in Phase A7; learned backend added as the documented upgrade.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

LaneInfo = dict[str, Any]


class LaneDetector:
    """Estimates the ego-lane / drivable region from a single frame."""

    def __init__(self, config: dict[str, Any] | None = None, device: str = "cpu") -> None:
        cfg = config or {}
        self.method = str(cfg.get("method", "classical")).lower()
        self.device = device

        # --- classical backend params ---
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

        # --- learned backend params ---
        learned = cfg.get("learned", {})
        self.model_path = str(learned.get("model_path", "models/lane_seg.pt"))
        self.learned_imgsz = int(learned.get("imgsz", 640))
        self.learned_conf = float(learned.get("conf", 0.25))
        self._seg_model = self._load_seg_model() if self.method == "learned" else None

    # --- dispatch ------------------------------------------------------------

    def estimate(self, frame: np.ndarray) -> LaneInfo:
        """Return the ego-lane polygon + filled lane mask for one frame."""
        if self._seg_model is not None:
            info = self._estimate_learned(frame)
            if info is not None:
                return info
        return self._estimate_classical(frame)

    # --- learned backend -----------------------------------------------------

    def _load_seg_model(self):
        path = Path(self.model_path)
        if not path.is_file():
            print(
                f"[LaneDetector] learned model not found at '{path}'; "
                "using the classical backend (see training/colab_lane_training.md)."
            )
            return None
        try:
            from ultralytics import YOLO

            return YOLO(str(path))
        except Exception as exc:  # pragma: no cover - depends on weights/runtime
            print(f"[LaneDetector] could not load learned model ({exc}); using classical.")
            return None

    def _estimate_learned(self, frame: np.ndarray) -> LaneInfo | None:
        h, w = frame.shape[:2]
        dev = 0 if self.device == "cuda" else self.device
        results = self._seg_model.predict(
            frame, imgsz=self.learned_imgsz, conf=self.learned_conf, device=dev, verbose=False
        )
        if not results or results[0].masks is None:
            return None
        masks = results[0].masks.data
        if masks is None or len(masks) == 0:
            return None
        union = masks.cpu().numpy().max(axis=0)  # (mh, mw) in [0, 1]
        binary = (union > 0.5).astype(np.uint8) * 255
        lane_mask = cv2.resize(binary, (w, h), interpolation=cv2.INTER_NEAREST)
        polygon = self._mask_to_polygon(lane_mask)
        if polygon is None:
            return None
        return {"ego_lane_polygon": polygon, "lane_mask": lane_mask}

    @staticmethod
    def _mask_to_polygon(mask: np.ndarray) -> list[list[int]] | None:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 50:
            return None
        approx = cv2.approxPolyDP(largest, 0.01 * cv2.arcLength(largest, True), True)
        return [[int(p[0][0]), int(p[0][1])] for p in approx]

    # --- classical backend ---------------------------------------------------

    def _estimate_classical(self, frame: np.ndarray) -> LaneInfo:
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
            polygon = [[int(x), int(y)] for x, y in roi_poly.tolist()]

        lane_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(lane_mask, [np.array(polygon, dtype=np.int32)], 255)
        return {"ego_lane_polygon": polygon, "lane_mask": lane_mask}

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
            if slope < 0 and max(x1, x2) < w * 0.6:
                left_pts.extend([(x1, y1), (x2, y2)])
            elif slope > 0 and min(x1, x2) > w * 0.4:
                right_pts.extend([(x1, y1), (x2, y2)])
        return left_pts, right_pts

    def _fit_side(
        self,
        pts: list[tuple[float, float]],
        y_bottom: float,
        y_top: float,
    ) -> tuple[tuple[float, float], tuple[float, float]] | None:
        if len(pts) < 4:
            return None
        xs = np.array([p[0] for p in pts], dtype=float)
        ys = np.array([p[1] for p in pts], dtype=float)
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
