# ADAS Vision Workbench

> A Windows-based computer-vision **ADAS perception workbench**: it ingests a
> driving video and produces object detection, multi-object tracking,
> lane/drivable-area estimation, a **pseudo-distance risk score**, event logs,
> an annotated output video, a benchmark report, and a live FastAPI dashboard.

**Status:** 🚧 Under construction — **Phase A2 complete** (tooling + environment
ready; CUDA PyTorch verified on an RTX 3050). This README is an intentional
*skeleton*; sections fill in as the phased build progresses (see [Roadmap](#roadmap)).

> ⚠️ **Honesty note (read this):** distance and time-to-collision (TTC) here are
> **proxies** derived from 2-D bounding-box geometry — **not** real,
> depth-based measurements. Real depth-based distance/TTC is planned for the
> ROS 2 + CARLA port (Project B). This tool never claims true physical distance.

---

## Demo

_A ≤90-second annotated demo clip / GIF will live here once the MVP runs._

`(placeholder — added around Phase A4–A5)`

---

## Why this exists

A portfolio/thesis-facing asset demonstrating CV + ML + ADAS competence, designed
from day one to be **ROS 2-portable**: every module takes plain data in and
returns plain data out, with no hard coupling, so it can later move to
ROS 2 + CARLA (Project B).

---

## Architecture (data flow)

```
Driving Video → Video Loader → Object Detection → Object Tracking
  → Lane / Drivable-Area → Risk Estimation → Outputs
       ├── annotated video
       ├── event logs (events.csv, frame_metrics.csv)
       ├── scene_summary.json
       ├── dashboard
       └── benchmark report
```

Full module responsibilities: see [`docs/architecture.md`](docs/architecture.md)
_(added in Phase A3)_.

---

## Installation

Windows + PowerShell. First install the prerequisites — **Python 3.10, Git, FFmpeg**
(the Phase A0 `winget` commands are in [`CLAUDE.md`](CLAUDE.md)).

```powershell
# 1) Clone and enter the repo
git clone https://github.com/AmanouNasri1/adas-vision.git
cd adas-vision

# 2) Create and activate a virtual environment (kept on C:)
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3) Install dependencies (pulls the CUDA 12.4 PyTorch build, ~2.5 GB)
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> **CPU-only machine?** `requirements.txt` pins the CUDA build for an NVIDIA GPU.
> Swap the two `torch` lines for the CPU build from the
> [PyTorch selector](https://pytorch.org/get-started/locally/), then re-run step 3.

---

## Usage

_The main demo will run with a single command once the pipeline exists:_

```powershell
# (coming in Phase A4+)
# python apps/run_video_demo.py --input data/sample_videos/test_drive.mp4
```

---

## Outputs

| Artifact | Location | Notes |
| --- | --- | --- |
| Annotated video | `E:\adas_outputs\videos\` | Large run outputs go to the external HDD |
| Event log | `events.csv` | One row per risk event |
| Frame metrics | `frame_metrics.csv` | Per-frame stats |
| Scene summary | `scene_summary.json` | Run-level summary |
| Demo assets | `outputs/demo/` | Small clip / GIF / screenshots committed for GitHub |

---

## Benchmark

_A results table (avg FPS, total detections, unique tracks, high-risk events, …)
will live here once benchmark mode lands (Phase A11)._

---

## Limitations

- **No real depth.** Distance/TTC are 2-D bounding-box proxies, not physical
  measurements. (Real depth comes with CARLA in Project B.)
- **Lane detection is classical CV** (edges + Hough lines), not learned
  segmentation — expect failures in poor lighting, faded markings, sharp curves.
- Detection/tracking quality is bounded by a small YOLO model chosen for speed.

Full, honest write-up: [`docs/limitations.md`](docs/limitations.md) _(Phase A8+)_.

---

## ROS 2 porting plan

This workbench is the Windows precursor to a ROS 2 + CARLA system (Project B).
The module → ROS 2 node mapping lives in
[`docs/ros2_porting_plan.md`](docs/ros2_porting_plan.md) _(Phase A3)_.

---

## Roadmap

Built in strict phases (see [`CLAUDE.md`](CLAUDE.md) for the full plan):

| Phase | Milestone |
| --- | --- |
| A0 | ✅ Install tools (Git, Python 3.10, VS Code, FFmpeg) |
| A1 | ✅ Repo + virtual environment |
| A2 | ✅ Install ML/CV packages |
| **A3** | **Repository structure + docs ⬅ next** |
| A4 | Video input pipeline (no AI) |
| A5 | Object detection (YOLO) |
| A6 | Object tracking |
| A7 | Lane / drivable-area |
| A8 | Risk estimation (pseudo-distance) |
| A9 | Event logging |
| A10 | Web dashboard (FastAPI) |
| A11 | Benchmark mode |
| A12 | ONNX export & comparison (optional) |
| A13 | LLM scene explainer (optional) |

---

## License

_TBD._
