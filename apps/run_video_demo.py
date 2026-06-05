"""Run the ADAS video demo: video -> detection -> tracking -> lane -> risk -> outputs.

Usage:
    python apps/run_video_demo.py --input data/sample_videos/test_drive.mp4
    python apps/run_video_demo.py --input <clip> --benchmark

The pipeline is wired up incrementally across Phases A4-A11. This entry point
parses arguments and locates the package; the per-stage logic lives in the
``src/adas_workbench`` modules.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# --- Make src/ importable when run as a plain script (no install needed) ---
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ADAS Vision Workbench — video demo")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input driving video (e.g. data/sample_videos/test_drive.mp4).",
    )
    parser.add_argument(
        "--config-dir",
        default="configs",
        help="Directory holding detector.yaml / tracker.yaml / risk.yaml.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output annotated-video path. Defaults to the configured outputs dir.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Collect timing/metrics and write reports/benchmark_report.md (Phase A11).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Fail clearly if the input is missing (CLAUDE.md §8: clear errors, not stack traces).
    if not Path(args.input).is_file():
        print(f"ERROR: input video not found: {args.input}", file=sys.stderr)
        return 2
    # TODO(A4+): build the pipeline — VideoLoader -> Detector -> Tracker ->
    #            LaneDetector -> RiskEstimator -> overlay -> writer + EventLogger.
    raise NotImplementedError("The demo pipeline is wired up starting in Phase A4.")


if __name__ == "__main__":
    raise SystemExit(main())
