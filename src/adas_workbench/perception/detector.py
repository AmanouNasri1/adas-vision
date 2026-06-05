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
from ultralytics import YOLO

Detection = dict[str, Any]


class Detector:
    """YOLO detector that emits plain detection dicts.

    Loads weights once (per ``configs/detector.yaml``) and filters predictions
    to the configured ADAS-relevant classes. Holds only the model — no frame
    history, no global state.
    """

    def __init__(self, config: dict[str, Any], device: str = "cpu") -> None:
        self.config = config
        model_cfg = config.get("model", {})
        inf_cfg = config.get("inference", {})

        self.weights = model_cfg.get("weights", "yolo11n.pt")
        self.imgsz = int(model_cfg.get("imgsz", 640))
        self.conf_threshold = float(inf_cfg.get("conf_threshold", 0.25))
        self.iou_threshold = float(inf_cfg.get("iou_threshold", 0.45))
        self.max_detections = int(inf_cfg.get("max_detections", 300))

        # Ultralytics expects 0 (first GPU) rather than the bare string "cuda".
        self.device: Any = 0 if device == "cuda" else device
        self.half = bool(model_cfg.get("half", False)) and device == "cuda"

        self._model = YOLO(self.weights)
        self._names: dict[int, str] = dict(self._model.names)

        # Resolve allowed class names -> model class ids (filter at inference).
        allowed_names = set(config.get("allowed_classes", []))
        if allowed_names:
            self._allowed_ids: list[int] | None = [
                i for i, n in self._names.items() if n in allowed_names
            ]
        else:
            self._allowed_ids = None  # keep every class

    def detect(self, frame: np.ndarray, frame_id: int) -> list[Detection]:
        """Run detection on one BGR frame; return filtered Detection dicts."""
        results = self._model.predict(
            frame,
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            max_det=self.max_detections,
            classes=self._allowed_ids,
            device=self.device,
            half=self.half,
            verbose=False,
        )
        detections: list[Detection] = []
        if not results:
            return detections
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, class_ids):
            detections.append(
                {
                    "frame_id": int(frame_id),
                    "class_name": self._names.get(int(cls_id), str(int(cls_id))),
                    "confidence": float(conf),
                    "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                }
            )
        return detections
