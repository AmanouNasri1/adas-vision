"""Launch the FastAPI dashboard (convenience wrapper around uvicorn).

Equivalent to:
    uvicorn dashboard.app:app --reload

Run:
    python apps/run_dashboard.py
then open http://127.0.0.1:8000 . Panels are fully wired up in Phase A10.
"""
from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="ADAS Vision Workbench — dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes.")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError as exc:  # clear message, not a stack trace
        raise SystemExit(
            "uvicorn is not installed. Run: pip install -r requirements.txt"
        ) from exc

    uvicorn.run("dashboard.app:app", host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
