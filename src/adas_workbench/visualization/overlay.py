"""Frame annotation / overlay rendering.

Pure drawing helpers: take a frame + plain perception/risk data, return an
annotated frame. No file or model access. Risk colours follow the LOW/MEDIUM/
HIGH levels; any pseudo-distance/TTC text is always labelled as a proxy.
ROS 2 mapping: reused as-is (or wrapped as an annotated-image publisher).
Implemented across Phases A4-A8 (frame number first, then boxes/ids/lane/risk).
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

Track = dict[str, Any]
Detection = dict[str, Any]

_FONT = cv2.FONT_HERSHEY_SIMPLEX

# Stable BGR colour per ADAS class (boxes + label backgrounds).
_CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "person": (0, 200, 0),
    "bicycle": (0, 200, 200),
    "car": (255, 128, 0),
    "motorcycle": (0, 165, 255),
    "bus": (255, 0, 0),
    "truck": (180, 0, 180),
    "traffic light": (0, 255, 255),
    "stop sign": (0, 0, 255),
}
_DEFAULT_COLOR = (200, 200, 200)

# Lane / drivable-area overlay style.
_LANE_FILL = (0, 200, 0)
_LANE_ALPHA = 0.28

# Risk-level colours (BGR): green / amber / red.
_RISK_COLORS: dict[str, tuple[int, int, int]] = {
    "LOW": (0, 200, 0),
    "MEDIUM": (0, 165, 255),
    "HIGH": (0, 0, 255),
}


def _color_for(class_name: str) -> tuple[int, int, int]:
    return _CLASS_COLORS.get(class_name, _DEFAULT_COLOR)


def _id_color(track_id: int) -> tuple[int, int, int]:
    """Deterministic vivid BGR colour per track id (so an id keeps its colour)."""
    hue = (int(track_id) * 47) % 180  # spread ids around the hue wheel
    hsv = np.uint8([[[hue, 200, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(b), int(g), int(r)


def _put_label(
    frame: np.ndarray,
    text: str,
    org: tuple[int, int],
    scale: float = 0.8,
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    """Draw outlined text in place (black outline + coloured fill) for contrast."""
    cv2.putText(frame, text, org, _FONT, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, org, _FONT, scale, color, 2, cv2.LINE_AA)


def _draw_box_label(
    frame: np.ndarray,
    bbox: list[float],
    label: str,
    color: tuple[int, int, int],
) -> None:
    """Draw a coloured box with a filled label tab (shared by detections/tracks/risk)."""
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), baseline = cv2.getTextSize(label, _FONT, 0.5, 1)
    # Keep the label on-screen even when the box hugs the top edge.
    y_top = max(y1, th + baseline + 4)
    cv2.rectangle(frame, (x1, y_top - th - baseline - 4), (x1 + tw + 4, y_top), color, -1)
    cv2.putText(
        frame, label, (x1 + 2, y_top - baseline - 2), _FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA
    )


def draw_frame_number(frame: np.ndarray, frame_id: int) -> np.ndarray:
    """Overlay the frame number (the first overlay wired up, Phase A4)."""
    _put_label(frame, f"frame {frame_id}", (12, 34))
    return frame


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw class-labelled, confidence-annotated boxes (Phase A5)."""
    for det in detections:
        label = f'{det["class_name"]} {det["confidence"]:.2f}'
        _draw_box_label(frame, det["bbox_xyxy"], label, _color_for(det["class_name"]))
    return frame


def draw_tracks(frame: np.ndarray, tracks: list[Track]) -> np.ndarray:
    """Draw boxes with persistent track ids (Phase A6).

    Each id keeps a stable colour, so the same object visibly carries the same
    box colour + id across frames.
    """
    for trk in tracks:
        label = f'ID {trk["track_id"]} {trk["class_name"]}'
        _draw_box_label(frame, trk["bbox"], label, _id_color(trk["track_id"]))
    return frame


def draw_lane(frame: np.ndarray, lane_info: dict[str, Any]) -> np.ndarray:
    """Overlay the ego-lane polygon / drivable area (Phase A7).

    Translucent green fill + outline. Labelled as an estimate — classical CV,
    not ground truth.
    """
    poly = lane_info.get("ego_lane_polygon")
    if not poly:
        return frame
    pts = np.array(poly, dtype=np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], _LANE_FILL)
    cv2.addWeighted(overlay, _LANE_ALPHA, frame, 1.0 - _LANE_ALPHA, 0, frame)
    cv2.polylines(frame, [pts], isClosed=True, color=_LANE_FILL, thickness=2)
    _put_label(frame, "ego-lane (est.)", (12, 60), scale=0.6, color=(0, 255, 0))
    return frame


def draw_risk(frame: np.ndarray, risk_results: list[dict[str, Any]]) -> np.ndarray:
    """Colour-code tracks by risk level and label the proxy score (Phase A8).

    Boxes are green/amber/red for LOW/MEDIUM/HIGH. A footer keeps the demo
    honest: the score is a proxy from 2-D boxes, not real depth/TTC.
    """
    for res in risk_results:
        color = _RISK_COLORS.get(res["risk_level"], _DEFAULT_COLOR)
        label = (
            f'ID {res["track_id"]} {res["class_name"]} '
            f'{res["risk_level"]} {res["risk_score"]:.0f}'
        )
        _draw_box_label(frame, res["bbox"], label, color)
    _put_label(
        frame,
        "pseudo-risk: proxy from 2-D boxes, not real depth/TTC",
        (12, frame.shape[0] - 12),
        scale=0.45,
        color=(210, 210, 210),
    )
    return frame
