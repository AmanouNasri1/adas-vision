"""Pseudo-distance & pseudo-TTC estimation (2-D bounding-box proxies).

WARNING: there is NO real depth here. "Distance" is inferred from bounding-box
geometry (height/area as a closeness proxy) and "TTC" from how fast the box
grows (bbox_height / bbox_height_change_rate). These are PROXIES only — never
present them as physical metres or seconds (CLAUDE.md hard rule; see
docs/limitations.md). Real depth-based distance/TTC arrives in Project B.

ROS 2 mapping: folded into risk_estimation_node.py, where the proxy is
upgraded to depth-based TTC.
Implemented in Phase A8.
"""
from __future__ import annotations

from typing import Any

Track = dict[str, Any]


class DistanceEstimator:
    """Turns bbox geometry + history into closeness and approach proxies."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        # TODO(A8): read ttc_proxy guardrails from config.
        raise NotImplementedError("DistanceEstimator is implemented in Phase A8.")

    def closeness(self, track: Track, frame_height: int) -> float:
        """Closeness proxy in [0, 1] from bbox area/height (1 ~ very close)."""
        # TODO(A8)
        raise NotImplementedError

    def ttc_proxy(self, track: Track, prev_track: Track | None) -> float:
        """Pseudo-TTC = bbox_height / bbox_height_change_rate (NOT real seconds)."""
        # TODO(A8)
        raise NotImplementedError
