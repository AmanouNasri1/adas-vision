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


def draw_frame_number(frame: np.ndarray, frame_id: int) -> np.ndarray:
    """Overlay the frame number (the first overlay wired up, Phase A4)."""
    _put_label(frame, f"frame {frame_id}", (12, 34))
    return frame


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw class-labelled, confidence-annotated boxes (Phase A5)."""
    # TODO(A5)
    raise NotImplementedError


def draw_tracks(frame: np.ndarray, tracks: list[Track]) -> np.ndarray:
    """Draw boxes with persistent track ids (Phase A6)."""
    # TODO(A6)
    raise NotImplementedError


def draw_lane(frame: np.ndarray, lane_info: dict[str, Any]) -> np.ndarray:
    """Overlay the ego-lane polygon / drivable area (Phase A7)."""
    # TODO(A7)
    raise NotImplementedError


def draw_risk(frame: np.ndarray, risk_results: list[dict[str, Any]]) -> np.ndarray:
    """Colour-code tracks by risk level and label the proxy score (Phase A8)."""
    # TODO(A8)
    raise NotImplementedError
