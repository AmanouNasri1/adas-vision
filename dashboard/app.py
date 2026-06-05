"""FastAPI dashboard backend.

Serves the ADAS dashboard: latest annotated frame, FPS, object/track counts,
current risk level, event timeline, and the run summary. Reads the pipeline's
output artifacts (scene_summary.json, events.csv, frame_metrics.csv, and a
periodically-saved latest_frame.jpg) from the configured logs dir.

Run:  uvicorn dashboard.app:app --reload   (or: python apps/run_dashboard.py)
then open http://127.0.0.1:8000 .
ROS 2 mapping: same backend, fed by ROS-derived state.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adas_workbench.explain.scene_explainer import explain_event, explain_scene  # noqa: E402
from adas_workbench.utils.config import load_config, resolve_output_dir  # noqa: E402


def _resolve_data_dir() -> Path:
    """Logs/data dir: env override (ADAS_LOGS_DIR), else paths.yaml, else fallback."""
    env = os.environ.get("ADAS_LOGS_DIR")
    if env:
        return Path(env)
    paths_yaml = REPO_ROOT / "configs" / "paths.yaml"
    paths_cfg = load_config(paths_yaml) if paths_yaml.is_file() else None
    return resolve_output_dir(paths_cfg, "logs")


DATA_DIR = _resolve_data_dir()

app = FastAPI(title="ADAS Vision Workbench", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")


def _read_json(path: Path) -> dict:
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _read_csv_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    try:
        with path.open("r", newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
    except Exception:
        return []


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Serve the dashboard page (re-read each request so edits show live)."""
    return (HERE / "templates" / "index.html").read_text(encoding="utf-8")


@app.get("/api/state")
def api_state() -> JSONResponse:
    """Latest run summary + recent events (with explanations) + scene headline."""
    events = _read_csv_rows(DATA_DIR / "events.csv")
    frames = _read_csv_rows(DATA_DIR / "frame_metrics.csv")
    recent = list(reversed(events[-15:]))
    last_frame = frames[-1] if frames else {}
    for event in recent:
        event["explanation"] = explain_event(event)
    return JSONResponse(
        {
            "summary": _read_json(DATA_DIR / "scene_summary.json"),
            "recent_events": recent,
            "last_frame": last_frame,
            "headline": explain_scene(recent, last_frame),
            "has_frame": (DATA_DIR / "latest_frame.jpg").is_file(),
            "data_dir": str(DATA_DIR),
        }
    )


@app.get("/frame")
def frame() -> Response:
    """Serve the latest annotated frame snapshot (if the pipeline has run)."""
    img = DATA_DIR / "latest_frame.jpg"
    if img.is_file():
        return FileResponse(str(img), media_type="image/jpeg")
    return Response(status_code=404)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
