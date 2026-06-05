"""FastAPI dashboard backend.

Serves the live ADAS dashboard: latest annotated frame, FPS, object/track
counts, risk level, event timeline, summary metrics — reading the latest
pipeline state.

This is a Phase-A10 placeholder: the ``app`` object and a couple of routes exist
so the server is launchable now; the real panels are added in Phase A10.
ROS 2 mapping: the same backend, fed by ROS-derived state.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI(title="ADAS Vision Workbench", version="0.1.0")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Placeholder landing page (the real dashboard arrives in Phase A10)."""
    return (
        "<h1>ADAS Vision Workbench</h1>"
        "<p>Dashboard scaffold is live. Panels are implemented in Phase A10.</p>"
    )


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check."""
    return {"status": "ok"}


# TODO(A10): serve templates/index.html + static/, add /state and /frame
#            endpoints (latest frame, fps, counts, risk level, event timeline).
