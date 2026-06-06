# Training a learned lane / drivable-area model (Google Colab)

Goal: replace the fragile classical lane detector with a **fine-tuned YOLO
segmentation model** that outputs a drivable-area mask — a drop-in upgrade behind
`LaneDetector` (set `method: learned` in `configs/lane.yaml`).

## Assumptions

- **Target = drivable-area mask** (matches the existing `lane_mask` contract and
  the risk module's ground-contact lane check), *not* lane-line points.
- **Model = `yolo11n-seg`**, **fine-tuned** (transfer learning, not from scratch) —
  reuses this project's Ultralytics stack + ONNX export (Phase A12).
- **Compute = Colab free T4 (16 GB)** for the first runs; **Pro** only to scale up
  (full BDD100K or a larger `-seg` model).

> **Why fine-tune, not "from 0"?** Even on Colab, a nano model trained from random
> init needs far more data/epochs just to *match* a pretrained backbone — you'd
> get a worse model for more effort. Transfer learning reaches a good model in
> ~1–3 h. If your thesis specifically requires from-scratch, pass
> `pretrained=False` to `model.train(...)` and expect to train much longer.

## 1. Pick a dataset

| Option | Effort | Notes |
| --- | --- | --- |
| **Roboflow Universe** "drivable area" / "road segmentation" (export as YOLOv8/YOLO11 segmentation) | low | Already in YOLO format — fastest way to validate the whole pipeline |
| **BDD100K** drivable area | high | The canonical dataset (100k imgs, *direct* + *alternative* drivable); needs a mask→polygon conversion to YOLO-seg format |

**Recommended:** start with a small Roboflow set to get an end-to-end run working,
then scale to BDD100K for the real model.

## 2. Colab cells

```python
# Cell 1 — GPU + Google Drive (persist weights across disconnects)
!nvidia-smi
from google.colab import drive
drive.mount('/content/drive')
```

```python
# Cell 2 — install
!pip -q install ultralytics
```

```python
# Cell 3 — get the dataset (Roboflow example)
# !pip -q install roboflow
# from roboflow import Roboflow
# rf = Roboflow(api_key="YOUR_KEY")
# ds = rf.workspace("WS").project("PROJ").version(1).download("yolov8")
# -> this writes a data.yaml; point `data=` at it in Cell 4.
```

```python
# Cell 4 — fine-tune (checkpoints to Drive so a disconnect doesn't lose progress)
from ultralytics import YOLO
model = YOLO("yolo11n-seg.pt")                  # pretrained backbone
model.train(
    data="/content/<dataset>/data.yaml",        # <-- your dataset config
    epochs=50, imgsz=640, batch=16,
    project="/content/drive/MyDrive/adas_lane",
    name="yolo11n-seg-drivable",
    patience=20, save=True,
)
# Resume after a disconnect:
# YOLO("/content/drive/MyDrive/adas_lane/yolo11n-seg-drivable/weights/last.pt").train(resume=True)
```

```python
# Cell 5 — validate + export to ONNX
metrics = model.val()
print("mask mAP50-95:", metrics.seg.map)
model.export(format="onnx", imgsz=640)          # -> best.onnx next to best.pt
```

```python
# Cell 6 — download the trained weights
from google.colab import files
files.download("/content/drive/MyDrive/adas_lane/yolo11n-seg-drivable/weights/best.pt")
```

## 3. Integrate back into the workbench

1. Put `best.pt` (or `best.onnx`) at **`models/lane_seg.pt`** in this repo.
2. In **`configs/lane.yaml`** set `method: learned` (adjust `learned.model_path`
   if you used a different name).
3. Run the demo — `LaneDetector` now uses the learned drivable-area mask, falling
   back to classical only if the model is missing or predicts nothing:
   ```powershell
   python apps\run_video_demo.py --input data\sample_videos\test_drive.mp4
   ```

## Colab free-tier tips

- Sessions ~12 h with idle disconnects → **save to Drive** and use `resume=True`.
- Keep the dataset on Drive (or re-download each session) to avoid re-uploading.
- T4 runs `yolo11n-seg` at `batch=16, imgsz=640` comfortably; raise the batch on Pro / A100.
- **Domain gap:** BDD100K is mostly US / highway, so your urban (HK) clip is
  out-of-distribution — expect to fine-tune on a few local frames, or accept lower
  accuracy on that specific clip. (In Project B, CARLA gives ground-truth lanes for free.)
