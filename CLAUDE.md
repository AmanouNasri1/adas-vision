CLAUDE.md — ADAS Vision Workbench (Windows)
This file gives Claude Code the full context, rules, and phased plan for building Project A: the Windows-based ADAS Vision Workbench. Read this entire file before writing code. Follow the phases in order. Do not skip ahead.

0. What we are building (and why)
A Windows-only computer-vision ADAS perception workbench that ingests a driving video and produces object detection, multi-object tracking, lane/drivable-area estimation, a pseudo-distance risk score, event logs, an annotated output video, a benchmark report, and a live FastAPI dashboard.
Strategic intent: This is a portfolio/thesis-facing asset to show CV + ML + ADAS competence to professors within ~1 month. It will later be ported to ROS 2 + CARLA (Project B), so every module must be designed to be ROS 2-portable (clean inputs/outputs, no hard coupling between modules).
Hard rules from the project owner — never violate these:

Do not split into many sub-projects. One coherent repo that evolves.
The first real milestone is a working demo video + a README a professor understands in 90 seconds, not perfect versions.
Do not pretend pseudo-distance / pseudo-TTC is real physical distance or TTC. Always label it as a proxy. Real depth-based distance only comes in Project B (CARLA).
Do not download large datasets (KITTI/BDD100K) until the MVP works on 2–3 short clips.
Do not spend time on React or fancy UI styling. Functional dashboard first.
Do not introduce Docker now.
No notebook-only code. Proper Python modules + runnable scripts.


1. Environment & machine constraints

OS: Windows. Shell is PowerShell (use Windows path separators and winget for installs; activate venv with .\.venv\Scripts\activate).
No internal 2 TB SSD yet. Storage strategy is mandatory:

LocationUseRuleC:\Projects\adas-vision-workbenchCode, virtual environment, configs, small sample videosFast internal storage. The .venv MUST live here.E:\datasets\drivingLarge datasets / downloaded driving videosExternal 2 TB HDDE:\adas_outputsLong output videos, logs, benchmark outputsKeeps C: clean

Never put the Python virtual environment on the external HDD (E:). Only large raw data and generated outputs go there.
GPU may or may not be present. Detect CUDA at runtime; fall back to CPU gracefully and log which device is used. Never crash if CUDA is unavailable.


2. Tech stack (pin loosely, verify against official docs)
Python 3.10, PyTorch, Ultralytics YOLO (YOLO11n / YOLOv8n), OpenCV, NumPy, Pandas, Matplotlib, PyYAML, tqdm, FastAPI + uvicorn[standard], plain HTML/CSS, FFmpeg.
Reference docs to verify install commands before running them (tooling changes):

PyTorch selector: https://pytorch.org/get-started/locally/
Ultralytics: https://docs.ultralytics.com/quickstart/
FastAPI: https://fastapi.tiangolo.com/

Do not chase perfect versions at the start.

3. Target repository structure
Create exactly this layout. Keep modules small and single-purpose so the ROS 2 port is painless.
adas-vision-workbench/
├── README.md
├── requirements.txt
├── .gitignore
├── CLAUDE.md
├── configs/
│   ├── detector.yaml
│   ├── tracker.yaml
│   └── risk.yaml
├── src/adas_workbench/
│   ├── input/video_loader.py
│   ├── perception/detector.py
│   ├── perception/lane_detector.py
│   ├── tracking/tracker.py
│   ├── risk/distance_estimator.py
│   ├── risk/risk_estimator.py
│   ├── visualization/overlay.py
│   ├── logging/event_logger.py
│   └── utils/config.py
├── apps/
│   ├── run_video_demo.py
│   └── run_dashboard.py
├── dashboard/
│   ├── app.py
│   ├── templates/index.html
│   └── static/style.css
├── data/sample_videos/
├── outputs/videos/
├── outputs/logs/
├── docs/
│   ├── architecture.md
│   ├── ros2_porting_plan.md
│   └── limitations.md
└── reports/benchmark_report.md
Note: large outputs are written to E:\adas_outputs per config; the outputs/ folder in the repo holds only small demo artifacts that go to GitHub (one short annotated clip, screenshots, GIF).

4. Architecture / data flow
Driving Video → Video Loader → Object Detection → Object Tracking
  → Lane / Drivable-Area → Risk Estimation → Outputs
       ├── annotated video
       ├── event log CSV (events.csv, frame_metrics.csv)
       ├── scene_summary.json
       ├── dashboard
       └── benchmark report
Design contract (critical for ROS 2 portability): Each module takes plain data in, returns plain data out. No module reads files or owns global state. Suggested per-frame contracts:

detector.detect(frame) -> list[Detection] where a detection is
{"frame_id": int, "class_name": str, "confidence": float, "bbox_xyxy": [x1,y1,x2,y2]}
tracker.update(detections, frame_id) -> list[Track] where a track is
{"track_id": int, "class_name": str, "bbox": [...], "age": int, "missed_frames": int}
lane_detector.estimate(frame) -> {"ego_lane_polygon": [...], "lane_mask": ndarray|None}
risk_estimator.score(tracks, lane_info, prev_state) -> list[{"track_id", "risk_score", "risk_level"}]

Keep configs in YAML and load through utils/config.py. No magic numbers inside logic files.

5. Phased execution plan
Work phase by phase. After each phase, run the stated Definition of Done, then make a git commit with a clear message. Do not start the next phase until DoD passes. Ask the user before any winget install or downloading models/data.
Phase A0 — Install tools
Install via PowerShell (Admin): Git, Python 3.10, VS Code, FFmpeg (winget install ...). Verify python --version, pip --version, git --version, ffmpeg -version.
DoD: Python, pip, Git, FFmpeg all run from PowerShell.
Phase A1 — Repo + virtual environment
Create C:\Projects\adas-vision-workbench, git init, create .venv on C:, activate it, upgrade pip/setuptools/wheel. Create .gitignore (ignore .venv/, __pycache__/, outputs/, *.pt, *.onnx, large media). Write an initial README skeleton.
DoD: prompt shows active .venv; initial commit pushed.
Phase A2 — Install ML/CV packages
Install PyTorch (use the CUDA index URL only if a CUDA GPU is present, else the CPU build), then ultralytics opencv-python numpy pandas matplotlib pyyaml tqdm fastapi "uvicorn[standard]". Verify torch import + torch.cuda.is_available(), and that a YOLO model loads. Write requirements.txt.
DoD: PyTorch imports, YOLO weights load, OpenCV imports.
Phase A3 — Repository structure
Create the full folder tree from section 3, with empty/stub module files and the three YAML configs. Add docs/architecture.md with the data-flow diagram.
DoD: structure exists; committed.
Phase A4 — Video input pipeline (no AI yet)
Implement input/video_loader.py and apps/run_video_demo.py. Read a driving video, preserve FPS, overlay the frame number, write an output video to the configured output dir. This validates the full I/O path before adding AI.
Run: python apps/run_video_demo.py --input data/sample_videos/test_drive.mp4
DoD: an annotated output video appears in the outputs dir.
Phase A5 — Object detection
Implement perception/detector.py with YOLO11n (or YOLOv8n). Filter to ADAS-relevant classes only: person, bicycle, car, motorcycle, bus, truck, traffic light, stop sign. Emit the detection dict contract from section 4. Wire into the demo so boxes/labels/confidence render.
DoD: output video shows boxes, class names, confidences.
Phase A6 — Object tracking
Implement tracking/tracker.py. Start with IoU-based matching to assign persistent track_ids; structure it so a SORT/Kalman upgrade can drop in later. Emit the track dict contract.
DoD: the same car/person keeps the same ID across consecutive frames.
Phase A7 — Lane / drivable-area module
Implement perception/lane_detector.py. MVP = OpenCV ROI + color/edge thresholding + Canny + Hough lines + a simple ego-lane polygon. Leave a hook for a future lightweight AI segmentation model.
DoD: output video shows a lane / drivable-area overlay.
Phase A8 — Risk estimation (pseudo-distance)
Implement risk/distance_estimator.py and risk/risk_estimator.py. No real depth — compute a risk score (0–100) from: bbox height/area (closeness proxy), bbox growth rate (approach cue), ego-lane overlap, class weight (person > bicycle > car > traffic light), and a TTC proxy = bbox_height / bbox_height_change_rate. Always label TTC as a proxy, never real TTC.
Thresholds: 0–30 LOW, 31–70 MEDIUM, 71–100 HIGH.
DoD: video/dashboard shows LOW / MEDIUM / HIGH.
Phase A9 — Event logging
Implement logging/event_logger.py. Every run writes:
events.csv, frame_metrics.csv, scene_summary.json (to the configured outputs/logs dir).
Event row columns: frame, timestamp, track_id, class, risk, bbox_area, lane_overlap, event (e.g. brake_warning).
DoD: each processed video produces the CSV + JSON logs.
Phase A10 — Web dashboard
Implement dashboard/app.py (FastAPI) + templates/index.html + static/style.css and apps/run_dashboard.py. Plain HTML/CSS only.
Panels: latest annotated frame, FPS, object count, tracked objects, risk level, event timeline, summary metrics.
Run: uvicorn dashboard.app:app --reload → open http://127.0.0.1:8000.
DoD: browser shows the latest processed frame and run metrics.
Phase A11 — Benchmark mode
Add --benchmark to apps/run_video_demo.py. Collect: average FPS, total detections, unique tracks, high-risk events, warning timestamps, processed frames. Write a results table into reports/benchmark_report.md.
DoD: reports/benchmark_report.md contains a results table.
Phase A12 — ONNX export & comparison (optional, post-MVP)
yolo export model=yolo11n.pt format=onnx imgsz=640. Add an ONNX Runtime inference path and a small PyTorch-vs-ONNX FPS table in the README.
DoD: README has a PyTorch vs ONNX performance table.
Phase A13 — LLM scene explainer (optional, last)
Template-based first (no API), turning a risk event JSON into a readable sentence; LLM API only later. Do not let this become a shallow wrapper.
DoD: dashboard shows human-readable event explanations.

6. Documentation deliverables (do alongside, not at the end)

docs/architecture.md — the data-flow diagram + module responsibilities.
docs/ros2_porting_plan.md — the mapping table below (this is the bridge to Project B):

Windows moduleFuture ROS 2 nodeNotevideo_loader.py/sim/front/rgb subscriber / camera nodeReplace frames with ROS Image msgsdetector.pydetection_node.pyReuse YOLO inference logictracker.pytracking_node.pyPublish tracks as custom msg / JSONlane_detector.pylane_drivable_node.pyPublish lane mask imagerisk_estimator.pyrisk_estimation_node.pyUpgrade pseudo-distance → depth-based TTCevent_logger.pyrosbag + CSV loggerKeep CSV outputs for reportsdashboard/app.pysame dashboard backendRead latest ROS-derived state

docs/limitations.md — honest limits: no real depth on Windows; TTC is a proxy; lane detection is classical CV; include failure cases.


7. Weekly schedule (target)
WeekGoalDeliverable1Setup + video + YOLO (A0–A5)Video input, YOLO detections, annotated video2Tracking + lane + risk (A6–A8)Track IDs, lane overlay, LOW/MED/HIGH risk3Dashboard + logs + GitHub polish (A9–A10)FastAPI dashboard, CSV/JSON logs, demo GIF, README4Benchmark + professor package (A11+)Benchmark table, report, thesis-extension section, CV bullets

8. Definition of "professor-ready" (gate before sharing)

README has a ≤90-second demo GIF/video at the top.
Installation is one clear command block that works on a fresh Windows machine.
Architecture diagram present.
Results/benchmark table present.
Limitations stated honestly (no real depth; TTC is approximate).
ROS 2 porting plan present.
One command runs the main demo. Config files exist. Missing video/model paths produce clear error messages, not stack traces.


9. Working agreement for Claude Code

Follow phases in order; commit after each passing DoD.
Prefer small, readable, well-documented functions over cleverness.
Keep every module decoupled and ROS 2-portable (plain data in/out).
Ask before installing tools, downloading models/datasets, or any irreversible action.
Never overstate capabilities (pseudo-distance ≠ real distance).
When in doubt, optimize for "a professor understands this in 90 seconds."