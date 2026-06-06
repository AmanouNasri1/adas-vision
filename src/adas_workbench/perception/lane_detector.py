"""Lane / drivable-area estimation.

Two backends, selected by ``method`` in ``configs/lane.yaml``:

- ``"classical"`` (default): ROI + (Canny edges fused with white/yellow colour
  masks) + HoughLinesP -> per-side line fit, **temporally smoothed (EMA)** with a
  short last-good-lane memory so the overlay is stable instead of flickering.
  Falls back to the ROI trapezoid only after the lane is lost for several frames.
- ``"learned"``: a fine-tuned YOLO segmentation model (see
  ``training/colab_lane_training.md``) -> drivable-area mask. Falls back to
  classical for a frame if the model is missing or returns nothing.

Contract (unchanged across backends):
    estimate(frame) -> {
        "ego_lane_polygon": list[[x, y], ...] | None,
        "lane_mask":        np.ndarray | None,
    }

The classical backend holds per-instance smoothing state (like the tracker) — a
node's internal state, not global. ROS 2 mapping: lane_drivable_node.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

LaneInfo = dict[str, Any]
Fit = tuple[float, float]  # (m, b) for x = m*y + b


class LaneDetector:
    """Estimates the ego-lane / drivable region from a frame stream."""

    def __init__(self, config: dict[str, Any] | None = None, device: str = "cpu") -> None:
        cfg = config or {}
        self.method = str(cfg.get("method", "classical")).lower()
        self.device = device
        self.process_scale = float(cfg.get("process_scale", 0.5))

        # --- classical: ROI ---
        roi = cfg.get("roi", {})
        self.bottom_width = float(roi.get("bottom_width", 0.90))
        self.top_width = float(roi.get("top_width", 0.18))
        self.top_y = float(roi.get("top_y", 0.62))
        self.bottom_y = float(roi.get("bottom_y", 0.97))

        # --- classical: edges ---
        edges = cfg.get("edges", {})
        self.blur_kernel = int(edges.get("blur_kernel", 5)) | 1
        self.canny_low = int(edges.get("canny_low", 60))
        self.canny_high = int(edges.get("canny_high", 150))

        # --- classical: Hough ---
        hough = cfg.get("hough", {})
        self.rho = float(hough.get("rho", 2))
        self.theta = float(np.deg2rad(float(hough.get("theta_deg", 1))))
        self.threshold = int(hough.get("threshold", 25))
        self.min_line_length = int(hough.get("min_line_length", 18))
        self.max_line_gap = int(hough.get("max_line_gap", 120))

        lines = cfg.get("lines", {})
        self.min_abs_slope = float(lines.get("min_abs_slope", 0.5))
        self.max_abs_slope = float(lines.get("max_abs_slope", 6.0))

        # --- classical: white/yellow colour masking ---
        color = cfg.get("color", {})
        self.use_color = bool(color.get("enabled", True))
        self.white_l_min = int(color.get("white_l_min", 200))
        self.yellow_min = tuple(color.get("yellow_hls_min", [15, 90, 80]))
        self.yellow_max = tuple(color.get("yellow_hls_max", [40, 255, 255]))

        # --- classical: temporal smoothing state ---
        sm = cfg.get("smoothing", {})
        self.smooth_alpha = float(sm.get("alpha", 0.3))
        self.max_misses = int(sm.get("max_misses", 12))
        self._left: Fit | None = None
        self._right: Fit | None = None
        self._left_miss = 0
        self._right_miss = 0

        # --- learned backend ---
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
        union = masks.cpu().numpy().max(axis=0)
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
        full_h, full_w = frame.shape[:2]
        s = self.process_scale
        if s != 1.0:
            work = cv2.resize(
                frame,
                (max(1, int(full_w * s)), max(1, int(full_h * s))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            work = frame
        h, w = work.shape[:2]
        y_bottom = self.bottom_y * h
        y_top = self.top_y * h
        roi_poly = self._roi_polygon(w, h)

        # Binary lane-feature mask: Canny edges fused with white/yellow paint.
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        feature = cv2.Canny(blur, self.canny_low, self.canny_high)
        if self.use_color:
            feature = cv2.bitwise_or(feature, self._color_mask(work))

        roi_mask = np.zeros_like(feature)
        cv2.fillPoly(roi_mask, [roi_poly], 255)
        masked = cv2.bitwise_and(feature, roi_mask)

        lines = cv2.HoughLinesP(
            masked,
            self.rho,
            self.theta,
            self.threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap,
        )
        left_pts, right_pts = self._collect_points(lines, w)

        # Per-side fit this frame, then exponentially smooth across frames.
        self._left, self._left_miss = self._update_side(
            self._left, self._left_miss, self._fit_params(left_pts)
        )
        self._right, self._right_miss = self._update_side(
            self._right, self._right_miss, self._fit_params(right_pts)
        )

        polygon = self._polygon_from_fits(self._left, self._right, y_bottom, y_top, w)
        if polygon is None:
            polygon = [[int(x), int(y)] for x, y in roi_poly.tolist()]
        if s != 1.0:  # scale the polygon back to full resolution
            inv = 1.0 / s
            polygon = [[int(round(x * inv)), int(round(y * inv))] for x, y in polygon]

        lane_mask = np.zeros((full_h, full_w), dtype=np.uint8)
        cv2.fillPoly(lane_mask, [np.array(polygon, dtype=np.int32)], 255)
        return {"ego_lane_polygon": polygon, "lane_mask": lane_mask}

    def _color_mask(self, frame: np.ndarray) -> np.ndarray:
        """Binary mask of white + yellow lane markings (HLS thresholds)."""
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        white = cv2.inRange(hls, (0, self.white_l_min, 0), (180, 255, 255))
        yellow = cv2.inRange(
            hls,
            np.array(self.yellow_min, dtype=np.uint8),
            np.array(self.yellow_max, dtype=np.uint8),
        )
        return cv2.bitwise_or(white, yellow)

    def _update_side(self, prev: Fit | None, miss: int, new: Fit | None) -> tuple[Fit | None, int]:
        """EMA-update a lane edge; keep the last edge for up to ``max_misses`` frames."""
        if new is not None:
            if prev is None:
                return new, 0
            a = self.smooth_alpha
            return (a * new[0] + (1 - a) * prev[0], a * new[1] + (1 - a) * prev[1]), 0
        miss += 1
        if miss > self.max_misses:
            return None, miss
        return prev, miss

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

    @staticmethod
    def _fit_params(pts: list[tuple[float, float]]) -> Fit | None:
        """Fit x = m*y + b (robust for near-vertical lane lines)."""
        if len(pts) < 4:
            return None
        xs = np.array([p[0] for p in pts], dtype=float)
        ys = np.array([p[1] for p in pts], dtype=float)
        m, b = np.polyfit(ys, xs, 1)
        return float(m), float(b)

    def _polygon_from_fits(
        self, left: Fit | None, right: Fit | None, y_bottom: float, y_top: float, w: int
    ) -> list[list[int]] | None:
        if left is None or right is None:
            return None
        lbx, ltx = left[0] * y_bottom + left[1], left[0] * y_top + left[1]
        rbx, rtx = right[0] * y_bottom + right[1], right[0] * y_top + right[1]
        if lbx >= rbx or ltx >= rtx:
            return None

        def clamp(v: float) -> int:
            return int(max(0, min(w - 1, v)))

        return [
            [clamp(lbx), int(y_bottom)],
            [clamp(ltx), int(y_top)],
            [clamp(rtx), int(y_top)],
            [clamp(rbx), int(y_bottom)],
        ]
