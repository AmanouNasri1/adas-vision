# Architecture

The ADAS Vision Workbench is a linear perception pipeline. Each stage takes
**plain data in and returns plain data out** — no module reads files mid-pipeline
or owns global state — so every stage maps cleanly onto a future ROS 2 node
(see [`ros2_porting_plan.md`](ros2_porting_plan.md)).

## Data flow

```
Driving Video
     │
     ▼
┌──────────────┐  frames: (frame_id, BGR frame), FPS preserved
│ VideoLoader  │
└──────────────┘
     │
     ▼
┌──────────────┐  list[Detection] {frame_id, class_name, confidence, bbox_xyxy}
│  Detector    │  YOLO, filtered to ADAS classes
└──────────────┘
     │
     ▼
┌──────────────┐  list[Track] {track_id, class_name, bbox, age, missed_frames}
│  Tracker     │  persistent ids via IoU matching
└──────────────┘
     │
     ▼
┌──────────────┐  lane_info {ego_lane_polygon, lane_mask}
│ LaneDetector │  classical CV: ROI + Canny + Hough
└──────────────┘
     │
     ▼
┌───────────────┐ list[RiskResult] {track_id, risk_score (0-100), risk_level}
│ RiskEstimator │ pseudo-distance + pseudo-TTC PROXIES — never real depth
└───────────────┘
     │
     ▼
   Outputs
     ├── annotated video                                   (visualization/overlay.py)
     ├── events.csv / frame_metrics.csv / scene_summary.json (logging/event_logger.py)
     ├── dashboard                                          (dashboard/app.py)
     └── benchmark report                                   (reports/benchmark_report.md)
```

## Module responsibilities

| Module | Responsibility | Contract (in → out) |
| --- | --- | --- |
| `input/video_loader.py` | Read video, preserve FPS, yield frames | path → `(frame_id, frame)` |
| `perception/detector.py` | YOLO detection, filter to ADAS classes | `frame, frame_id` → `list[Detection]` |
| `tracking/tracker.py` | Assign persistent track ids | `detections, frame_id` → `list[Track]` |
| `perception/lane_detector.py` | Estimate ego-lane / drivable area | `frame` → `{ego_lane_polygon, lane_mask}` |
| `risk/distance_estimator.py` | Closeness + pseudo-TTC proxies | `track, prev` → floats |
| `risk/risk_estimator.py` | Blend cues → 0-100 risk + level | `tracks, lane_info, prev` → `list[RiskResult]` |
| `visualization/overlay.py` | Draw boxes / ids / lane / risk | `frame, data` → `frame` |
| `logging/event_logger.py` | Write events / metrics CSV + summary JSON | rows → files |
| `utils/config.py` | Load YAML config, pick CUDA/CPU device | path → dict; pref → device |

## Design rules

- **No magic numbers in logic.** All thresholds / weights live in `configs/*.yaml`
  and are loaded through `utils/config.py`.
- **Device safety.** `select_device` uses CUDA when available, else falls back to
  CPU — it never crashes if CUDA is absent.
- **Honesty.** Distance/TTC are bounding-box proxies, always labelled as such.
  Real depth-based metrics are deferred to Project B (CARLA). See
  [`limitations.md`](limitations.md).
