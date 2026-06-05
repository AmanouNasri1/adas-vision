"""Object detection (Ultralytics YOLO).

Contract (CLAUDE.md §4):
    detect(frame, frame_id) -> list[Detection]
    Detection = {
        "frame_id":   int,
        "class_name": str,        # filtered to ADAS-relevant classes
        "confidence": float,      # 0-1
        "bbox_xyxy":  [x1, y1, x2, y2],
    }

ROS 2 mapping: detection_node.py (reuses this inference logic).
Implemented in Phase A5.
"""
from __future__ import annotations

from typing import Any

import numpy as np

Detection = dict[str, Any]


class Detector:
    """YOLO detector that emits plain detection dicts.

    Loads weights once (per ``configs/detector.yaml``) and filters predictions
    to the configured ADAS-relevant classes. Holds only the model — no frame
    history.
    """

    def __init__(self, config: dict[str, Any], device: str = "cpu") -> None:
        self.config = config
        self.device = device
        self._model = None
        # TODO(A5): load YOLO(weights) onto device; cache allowed-class id set.
        raise NotImplementedError("Detector is implemented in Phase A5.")

    def detect(self, frame: np.ndarray, frame_id: int) -> list[Detection]:
        """Run detection on one BGR frame; return filtered Detection dicts."""
        # TODO(A5): run model -> filter by allowed_classes + conf -> build dicts.
        raise NotImplementedError("Detector.detect is implemented in Phase A5.")
