"""Template-based scene explainer (no LLM).

Turns risk events / scene state into short, human-readable sentences for the
dashboard. Deliberately template-based (Phase A13): deterministic, dependency-
free, and honest. An LLM-backed explainer can later replace ``explain_event`` /
``explain_scene`` behind the same interface - keep these templates as the
offline fallback so the dashboard never depends on a network call.

Wording stays action-oriented; the dashboard carries the global "proxy, no real
depth" disclaimer, and sentences label the score as a proxy.
ROS 2 mapping: unchanged - pure function of plain event dicts.
Implemented in Phase A13.
"""
from __future__ import annotations

from typing import Any

Event = dict[str, Any]


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def _position(lane_overlap: float) -> str:
    if lane_overlap >= 0.6:
        return "in your lane"
    if lane_overlap >= 0.2:
        return "drifting into your lane"
    return "to the side"


def _high_action(class_name: str) -> str:
    if class_name == "person":
        return "Be ready to stop for the pedestrian."
    if class_name in ("bicycle", "motorcycle"):
        return f"Give the {class_name} space and cover the brake."
    if class_name in ("car", "truck", "bus"):
        return "Prepare to brake - closing distance."
    return "Prepare to slow down."


def explain_event(event: Event) -> str:
    """One readable sentence for a single risk event row (from events.csv)."""
    cls = str(event.get("class", "object"))
    tid = event.get("track_id", "?")
    risk = _as_float(event.get("risk"))
    lane = _as_float(event.get("lane_overlap"))
    kind = str(event.get("event", ""))
    position = _position(lane)

    if kind == "brake_warning" or risk > 70:
        return (
            f"Brake warning - {cls} #{tid} is close {position} "
            f"(proxy risk {risk:.0f}/100). {_high_action(cls)}"
        )
    return (
        f"Caution - {_article(cls)} {cls} (#{tid}) {position} at moderate "
        f"risk ({risk:.0f}/100). Keep monitoring."
    )


def explain_scene(recent_events: list[Event], last_frame: dict[str, Any]) -> str:
    """One-line headline describing the current scene's risk state."""
    high = int(_as_float(last_frame.get("high")))
    medium = int(_as_float(last_frame.get("medium")))
    tracks = int(_as_float(last_frame.get("tracks")))

    if high > 0:
        top = _top_event(recent_events, "brake_warning") or _top_event(recent_events)
        if top is not None:
            return explain_event(top)
        return f"{high} high-risk object(s) ahead - prepare to brake."
    if medium > 0:
        top = _top_event(recent_events)
        if top is not None:
            return explain_event(top)
        return f"{medium} object(s) at moderate risk in view - keep monitoring."
    if tracks > 0:
        return f"Clear - {tracks} tracked object(s) in view, none at elevated risk."
    return "No tracked objects in view."


def _top_event(events: list[Event], kind: str | None = None) -> Event | None:
    pool = [e for e in events if (kind is None or e.get("event") == kind)]
    if not pool:
        return None
    return max(pool, key=lambda e: _as_float(e.get("risk")))
