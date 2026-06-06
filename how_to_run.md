# How to Run & Verify — ADAS Vision Workbench

A practical, phase-by-phase guide. For **each phase** it lists the
**prerequisites**, **how to run it**, and **how to verify** its Definition of
Done (DoD).

> Platform: **Windows + PowerShell**. Run all commands from the **project root**
> with the virtual environment **activated** (`(.venv)` shown in the prompt),
> unless noted otherwise.

---

## Quick reference (TL;DR)

After the one-time setup (A0–A2) and dropping a clip at
`data\sample_videos\test_drive.mp4`:

```powershell
# 1) Full perception pipeline -> annotated video + logs (exercises A4–A9)
python apps\run_video_demo.py --input data\sample_videos\test_drive.mp4

# 2) Same, plus a benchmark report (A11)
python apps\run_video_demo.py --input data\sample_videos\test_drive.mp4 --benchmark

# 3) Live dashboard (A10 + A13) -> open http://127.0.0.1:8000
python apps\run_dashboard.py
```

Useful flags: `--device cpu` (force CPU), `--output <path.mp4>` (custom output),
`--config-dir <dir>` (alternate configs).

---

## Prerequisites at a glance

| Need | Why | Set up in |
| --- | --- | --- |
| Windows 10/11 + PowerShell | Target platform | — |
| Python **3.10**, Git, FFmpeg | Runtime + tooling | Phase A0 |
| Project cloned + `.venv` on C: | Isolated environment | Phase A1 |
| `pip install -r requirements.txt` | PyTorch/YOLO/OpenCV/FastAPI | Phase A2 |
| Short driving clip at `data\sample_videos\test_drive.mp4` | Pipeline input | before A4 |
| (Optional) NVIDIA GPU + driver | GPU speed-up; CPU fallback is automatic | — |
| (Optional) external drive `D:` | Large outputs go to `D:\adas_outputs` (else `.\outputs`) | `configs\paths.yaml` |

---

## Phase A0 — Install tools

**Prerequisites:** Windows with `winget` (App Installer). Run PowerShell **as Administrator**.

**Run:**
```powershell
winget install -e --id Git.Git                   --accept-package-agreements --accept-source-agreements
winget install -e --id Python.Python.3.10         --accept-package-agreements --accept-source-agreements
winget install -e --id Microsoft.VisualStudioCode --accept-package-agreements --accept-source-agreements
winget install -e --id Gyan.FFmpeg                --accept-package-agreements --accept-source-agreements
```
Then open a **new** PowerShell window (so PATH refreshes).

**Verify (DoD = all four run):**
```powershell
python --version    # Python 3.10.x   (or: py -3.10 --version)
pip --version
git --version
ffmpeg -version
```

---

## Phase A1 — Repo + virtual environment

**Prerequisites:** A0.

**Run:**
```powershell
git clone https://github.com/AmanouNasri1/adas-vision.git
cd adas-vision
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned   # one-time: allow venv activation
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

**Verify (DoD = active venv on C:):**
- Prompt shows `(.venv)`.
- `python -c "import sys; print(sys.executable)"` points inside `...\adas-vision\.venv\Scripts\python.exe`.

---

## Phase A2 — Install ML/CV dependencies

**Prerequisites:** A1 (venv active). The pin uses the **CUDA 12.4** PyTorch build;
CPU-only machines should swap the two `torch` lines for the CPU build from the
[PyTorch selector](https://pytorch.org/get-started/locally/), then re-install.

**Run:**
```powershell
python -m pip install -r requirements.txt
```

**Verify (DoD = torch imports, YOLO loads, OpenCV imports):**
```powershell
python -c "import torch, torchvision; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available())"
python -c "import cv2; print('OpenCV', cv2.__version__)"
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt'); print('YOLO OK')"
```
Expect `CUDA True` on a supported GPU; first YOLO load downloads `yolo11n.pt` (~5 MB).

---

## Phase A3 — Repository structure + configs

**Prerequisites:** A1 (repo present); A2 for the import check.

**Run:** nothing — the structure and `configs\*.yaml` ship with the repo.

**Verify (DoD = structure exists & imports cleanly):**
```powershell
.\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0,'src'); import adas_workbench; from adas_workbench.utils.config import load_config, select_device; print('package', adas_workbench.__version__, '| device', select_device('auto')); [print('config OK:', c, list(load_config('configs/'+c+'.yaml').keys())) for c in ('detector','tracker','lane','risk','paths')]"
```
Expect the package version, the selected device, and each config's top-level keys.

---

## Phase A4 — Video input pipeline

**Prerequisites:** A2 + a short clip at `data\sample_videos\test_drive.mp4`.

> The single demo command runs the **whole** pipeline (A4–A9). A4's specific job
> is reading the video (preserving FPS), overlaying the frame number, and writing
> the output video.

**Run:**
```powershell
python apps\run_video_demo.py --input data\sample_videos\test_drive.mp4
```

**Verify (DoD = an annotated output video appears):**
- Console prints the input resolution/FPS and `Output: ...`.
- An `*_annotated.mp4` appears in the outputs dir — `D:\adas_outputs\videos\`
  if `D:` is mounted, else `.\outputs\videos\`. Open it: it plays and shows
  `frame N` top-left.

---

## Phase A5 — Object detection (YOLO11n)

**Prerequisites:** A4 working.

**Run:** the same demo command.

**Verify (DoD = boxes + class names + confidences):**
- Console prints `Detections: <large number>` (e.g. ~4865).
- The output video shows colored boxes labelled `class 0.xx` (confidence), limited
  to the 8 ADAS classes.

---

## Phase A6 — Object tracking (IoU)

**Prerequisites:** A5.

**Run:** the same demo command.

**Verify (DoD = stable IDs across frames):**
- Console prints `Unique tracks: <N>` that is **much smaller** than total
  detections (e.g. 191 vs 4865).
- In the video, the same car/person keeps the same `ID n` (and box colour) across
  consecutive frames.

---

## Phase A7 — Lane / drivable-area

**Prerequisites:** A4.

**Run:** the same demo command.

**Verify (DoD = lane / drivable-area overlay):**
- The output video shows a translucent **green ego-lane region** labelled
  `ego-lane (est.)`. (Classical CV — it is an estimate; tune in `configs\lane.yaml`.)

---

## Phase A8 — Risk estimation (pseudo-distance)

**Prerequisites:** A6 + A7.

**Run:** the same demo command.

**Verify (DoD = LOW / MEDIUM / HIGH shown):**
- Console prints `Risk track-frames LOW/MEDIUM/HIGH` with all three present
  (e.g. `2453 / 1735 / 22`).
- Boxes are colour-coded **green (LOW) / amber (MEDIUM) / red (HIGH)**, with the
  footer `pseudo-risk: proxy ... not real depth/TTC`.
- Thresholds and cue weights live in `configs\risk.yaml` (0–30 LOW, 31–70 MEDIUM,
  71–100 HIGH).

---

## Phase A9 — Event logging

**Prerequisites:** A8.

**Run:** the same demo command.

**Verify (DoD = CSV + JSON logs produced):** in the logs dir
(`D:\adas_outputs\logs` or `.\outputs\logs`) three files appear:
```powershell
Get-Content D:\adas_outputs\logs\events.csv -TotalCount 4        # frame,timestamp,track_id,class,risk,bbox_area,lane_overlap,event
Get-Content D:\adas_outputs\logs\frame_metrics.csv -TotalCount 4 # per-frame counts + max_risk
Get-Content D:\adas_outputs\logs\scene_summary.json -Raw         # run summary
```

---

## Phase A10 — Web dashboard (FastAPI)

**Prerequisites:** run the demo at least once (so `scene_summary.json` and
`latest_frame.jpg` exist).

**Run:**
```powershell
python apps\run_dashboard.py
# equivalent: uvicorn dashboard.app:app --reload
```
Open **http://127.0.0.1:8000**.

**Verify (DoD = browser shows latest frame + metrics):**
- The page shows the latest annotated frame, metric cards (device, FPS, frames,
  detections, unique tracks, max risk), a risk-distribution bar, and the event
  timeline.
- Endpoints: `/` (page), `/api/state` (JSON), `/frame` (JPEG), `/health`.

---

## Phase A11 — Benchmark mode

**Prerequisites:** A8.

**Run:**
```powershell
python apps\run_video_demo.py --input data\sample_videos\test_drive.mp4 --benchmark
```

**Verify (DoD = results table in the report):** open
[`reports/benchmark_report.md`](reports/benchmark_report.md) — it contains a Run
configuration table, a Results table (avg FPS, detections, unique tracks, risk
counts, `brake_warning` events, max score), and Warning episodes.

---

## Phase A12 — ONNX export & comparison

**Prerequisites:** A2 **plus** `onnx`, `onnxruntime`, `onnxslim` (now pinned in
`requirements.txt`, so they install with the rest).

**Run:**
```powershell
python apps\benchmark_onnx.py             # exports yolo11n.onnx, benchmarks, writes the report
python apps\benchmark_onnx.py --runs 200  # more runs for a steadier average
```

**Verify (DoD = PyTorch-vs-ONNX table):** the console, the README, and
[`reports/onnx_benchmark.md`](reports/onnx_benchmark.md) show an FPS table for
PyTorch (CUDA), PyTorch (CPU), and ONNX Runtime (CPU).

**Use ONNX in the demo (optional):** set `weights: yolo11n.onnx` in
`configs\detector.yaml` — the detector loads it through Ultralytics' ONNX backend
(no code change needed).

---

## Phase A13 — Scene explainer (template-based)

**Prerequisites:** A9 (events exist) + A10 (dashboard).

**Run:** launch the dashboard (`python apps\run_dashboard.py`).

**Verify (DoD = human-readable explanations on the dashboard):**
- A plain-English **headline** appears under the frame, e.g.
  `Caution - a person (#381) in your lane at moderate risk (58/100). Keep monitoring.`
- The "Event explanations" panel lists recent events as sentences.

---

## Where things are written

| Output | Location | Notes |
| --- | --- | --- |
| Annotated video | `D:\adas_outputs\videos\` (or `.\outputs\videos\`) | Auto-falls back to in-repo `outputs/` if `D:` is absent |
| Logs (CSV/JSON) | `D:\adas_outputs\logs\` (or `.\outputs\logs\`) | `events.csv`, `frame_metrics.csv`, `scene_summary.json`, `latest_frame.jpg` |
| Benchmark report | `reports\benchmark_report.md` | In-repo (committed) |
| Demo assets | `outputs\demo\` | `demo.gif`, `highlight.png` (committed for the README) |

Output locations are configured in [`configs/paths.yaml`](configs/paths.yaml).

---

## Configuration files

| File | Controls |
| --- | --- |
| `configs\detector.yaml` | YOLO weights, image size, device, conf/IoU thresholds, ADAS class filter |
| `configs\tracker.yaml` | IoU match threshold, max missed frames, min hits |
| `configs\lane.yaml` | ROI trapezoid, Canny/Hough params, slope filters |
| `configs\risk.yaml` | Level thresholds, class weights, cue weights, normalization, pseudo-TTC |
| `configs\paths.yaml` | External vs in-repo output directories |

---

## Troubleshooting

- **`Activate.ps1` is blocked** → `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`.
- **`CUDA False` / runs on CPU** → expected without an NVIDIA GPU; it falls back
  automatically. For GPU, install the CUDA torch build (see A2) and check the driver.
- **`ERROR: input video not found`** → confirm the clip exists at the `--input` path.
- **Output not on `D:`** → `D:` isn't mounted; outputs fall back to `.\outputs\`.
  Edit `configs\paths.yaml` to change the drive/base.
- **Dashboard shows "no events yet"** → run the demo once first (it writes the logs
  the dashboard reads).
- **Port 8000 already in use** → `python apps\run_dashboard.py --port 8001`.
- **`ffprobe`/`ffmpeg` not found** → open a fresh terminal after A0 so PATH updates
  (OpenCV bundles its own FFmpeg, so the pipeline itself does not need system FFmpeg).
