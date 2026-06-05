"""Video loading and frame iteration.

ROS 2 mapping: replaced by a camera node / Image subscriber (see
docs/ros2_porting_plan.md). For now it reads a local video file and yields
``(frame_id, frame)`` pairs, exposing FPS/size so the writer can match them.

Implemented in Phase A4.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


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
    Owns only the open capture handle, no pipeline state.
    """

    def __init__(self, source: str | Path) -> None:
        self.source = str(source)
        self.meta: VideoMeta | None = None
        # TODO(A4): open cv2.VideoCapture, validate path, populate self.meta.
        raise NotImplementedError("VideoLoader is implemented in Phase A4.")

    def frames(self) -> Iterator[tuple[int, np.ndarray]]:
        """Yield ``(frame_id, frame_bgr)`` pairs, frame_id starting at 0."""
        # TODO(A4): loop cap.read(); yield until exhausted.
        raise NotImplementedError("VideoLoader.frames is implemented in Phase A4.")

    def release(self) -> None:
        """Release the underlying capture handle."""
        # TODO(A4)
        raise NotImplementedError
