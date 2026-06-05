"""Video loading and frame iteration.

ROS 2 mapping: replaced by a camera node / Image subscriber (see
docs/ros2_porting_plan.md). Reads a local video file and yields
``(frame_id, frame)`` pairs, exposing FPS/size so the writer can match them.

Implemented in Phase A4.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

# Fallback FPS when a container does not report a usable frame rate.
_DEFAULT_FPS = 30.0


@dataclass
class VideoMeta:
    """Metadata describing an opened video source."""

    path: str
    fps: float
    width: int
    height: int
    frame_count: int


class VideoLoader:
    """Reads a driving video and yields frames in order.

    Plain data in (a file path), plain data out (BGR numpy frames + frame_id).
    Owns only the open capture handle, no pipeline state. Usable as a context
    manager so the capture is always released.
    """

    def __init__(self, source: str | Path) -> None:
        self.source = str(source)
        if not Path(self.source).is_file():
            raise FileNotFoundError(f"Input video not found: {self.source}")
        self._cap: cv2.VideoCapture | None = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise IOError(
                f"Could not open video (unsupported codec or corrupt file?): {self.source}"
            )
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        # FPS can be 0 or NaN for some files; fall back to a sane default.
        if not fps or fps != fps or fps <= 0:
            fps = _DEFAULT_FPS
        self.meta = VideoMeta(
            path=self.source,
            fps=float(fps),
            width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            frame_count=int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )

    def frames(self) -> Iterator[tuple[int, np.ndarray]]:
        """Yield ``(frame_id, frame_bgr)`` pairs, frame_id starting at 0."""
        if self._cap is None:
            raise RuntimeError("VideoLoader has been released.")
        frame_id = 0
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            yield frame_id, frame
            frame_id += 1

    def release(self) -> None:
        """Release the underlying capture handle (idempotent)."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "VideoLoader":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()
