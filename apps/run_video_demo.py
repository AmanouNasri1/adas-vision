"""Run the ADAS video demo: video -> detection -> tracking -> lane -> risk -> outputs.

Usage:
    python apps/run_video_demo.py --input data/sample_videos/test_drive.mp4
    python apps/run_video_demo.py --input <clip> --device cpu
    python apps/run_video_demo.py --input <clip> --benchmark

Phase A9 adds event + metric logging: every run writes events.csv,
frame_metrics.csv, and scene_summary.json to the configured logs dir (alongside
the annotated video). Pipeline per frame: detect -> track -> lane -> risk ->
log -> draw -> write. Output/log dirs come from configs/paths.yaml (external HDD
when mounted, else the in-repo outputs/ folder).
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
from tqdm import tqdm

# --- Make src/ importable when run as a plain script (no install needed) ---
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adas_workbench.input.video_loader import VideoLoader  # noqa: E402
from adas_workbench.logging.event_logger import EventLogger  # noqa: E402
from adas_workbench.perception.detector import Detector  # noqa: E402
from adas_workbench.perception.lane_detector import LaneDetector  # noqa: E402
from adas_workbench.risk.risk_estimator import RiskEstimator  # noqa: E402
from adas_workbench.tracking.tracker import Tracker  # noqa: E402
from adas_workbench.utils.config import (  # noqa: E402
    load_config,
    resolve_output_dir,
    select_device,
)
from adas_workbench.visualization.overlay import (  # noqa: E402
    draw_frame_number,
    draw_lane,
    draw_risk,
)


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
        help="Directory with detector.yaml / tracker.yaml / lane.yaml / risk.yaml / paths.yaml.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output annotated-video path. Defaults to the configured outputs dir.",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["auto", "cpu", "cuda"],
        help="Override compute device (default: from detector.yaml).",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Collect timing/metrics and write reports/benchmark_report.md (Phase A11).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"ERROR: input video not found: {input_path}", file=sys.stderr)
        return 2

    config_dir = Path(args.config_dir)
    detector_cfg = load_config(config_dir / "detector.yaml")
    tracker_cfg = load_config(config_dir / "tracker.yaml")
    risk_cfg = load_config(config_dir / "risk.yaml")
    lane_cfg = load_config(config_dir / "lane.yaml") if (config_dir / "lane.yaml").is_file() else {}
    paths_cfg = (
        load_config(config_dir / "paths.yaml") if (config_dir / "paths.yaml").is_file() else None
    )
    device = select_device(args.device or detector_cfg.get("model", {}).get("device", "auto"))

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = resolve_output_dir(paths_cfg, "videos") / f"{input_path.stem}_annotated.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logs_dir = resolve_output_dir(paths_cfg, "logs")

    print(f"Device: {device}")
    detector = Detector(detector_cfg, device=device)
    tracker = Tracker(tracker_cfg)
    lane_detector = LaneDetector(lane_cfg)
    logger = EventLogger(logs_dir)

    with VideoLoader(input_path) as loader:
        meta = loader.meta
        risk_estimator = RiskEstimator(risk_cfg, fps=meta.fps)
        print(
            f"Input : {meta.path}\n"
            f"        {meta.width}x{meta.height} @ {meta.fps:.2f} fps, "
            f"{meta.frame_count} frames (reported)\n"
            f"Logs  : {logs_dir}"
        )

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            meta.fps,
            (meta.width, meta.height),
        )
        if not writer.isOpened():
            print(f"ERROR: could not open VideoWriter for {output_path}.", file=sys.stderr)
            return 3

        start = time.perf_counter()
        processed = 0
        total_detections = 0
        seen_ids: set[int] = set()
        risk_hist = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
        max_score, max_score_frame = 0.0, -1
        prev_state: dict[int, list[float]] = {}
        total = meta.frame_count if meta.frame_count > 0 else None
        for frame_id, frame in tqdm(
            loader.frames(), total=total, unit="f", desc="Processing", disable=None
        ):
            timestamp = round(frame_id / meta.fps, 3)
            # --- analysis on the clean frame ---
            detections = detector.detect(frame, frame_id)
            tracks = tracker.update(detections, frame_id)
            lane_info = lane_detector.estimate(frame)
            risks = risk_estimator.score(tracks, lane_info, prev_state)

            # --- stats + logging ---
            total_detections += len(detections)
            seen_ids.update(t["track_id"] for t in tracks)
            frame_levels = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
            for r in risks:
                level = r["risk_level"]
                risk_hist[level] = risk_hist.get(level, 0) + 1
                frame_levels[level] = frame_levels.get(level, 0) + 1
                if r["risk_score"] > max_score:
                    max_score, max_score_frame = r["risk_score"], frame_id
                if level in ("MEDIUM", "HIGH"):
                    logger.log_event(
                        {
                            "frame": frame_id,
                            "timestamp": timestamp,
                            "track_id": r["track_id"],
                            "class": r["class_name"],
                            "risk": r["risk_score"],
                            "bbox_area": r["bbox_area"],
                            "lane_overlap": r["lane_overlap"],
                            "event": "brake_warning" if level == "HIGH" else "caution",
                        }
                    )
            logger.log_frame(
                {
                    "frame": frame_id,
                    "timestamp": timestamp,
                    "detections": len(detections),
                    "tracks": len(tracks),
                    "low": frame_levels["LOW"],
                    "medium": frame_levels["MEDIUM"],
                    "high": frame_levels["HIGH"],
                    "max_risk": max((r["risk_score"] for r in risks), default=0.0),
                }
            )

            # --- overlays (lane underneath, then risk boxes, then HUD text) ---
            draw_lane(frame, lane_info)
            draw_risk(frame, risks)
            draw_frame_number(frame, frame_id)
            writer.write(frame)

            prev_state = {t["track_id"]: t["bbox"] for t in tracks}
            processed += 1
        writer.release()
        elapsed = time.perf_counter() - start

    avg_fps = processed / elapsed if elapsed > 0 else 0.0
    summary = {
        "input_video": str(input_path),
        "device": device,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "frames_processed": processed,
        "duration_s": round(processed / meta.fps, 2) if meta.fps else None,
        "processing_avg_fps": round(avg_fps, 2),
        "total_detections": total_detections,
        "unique_tracks": len(seen_ids),
        "risk_track_frames": risk_hist,
        "events_logged": logger.event_count,
        "max_risk_score": max_score,
        "max_risk_frame": max_score_frame,
        "output_video": str(output_path),
    }
    logger.close(summary)

    print(
        f"\nDone.\n"
        f"Output: {output_path}\n"
        f"Logs  : events.csv, frame_metrics.csv, scene_summary.json -> {logs_dir}\n"
        f"Frames: {processed} | Detections: {total_detections} | "
        f"Unique tracks: {len(seen_ids)} | {elapsed:.1f}s (~{avg_fps:.1f} fps, {device})\n"
        f"Risk track-frames  LOW: {risk_hist['LOW']}  "
        f"MEDIUM: {risk_hist['MEDIUM']}  HIGH: {risk_hist['HIGH']}  "
        f"| events: {logger.event_count} | max score {max_score:.0f} @frame {max_score_frame}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
