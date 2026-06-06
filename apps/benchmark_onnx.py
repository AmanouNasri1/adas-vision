"""Benchmark YOLO11n: PyTorch vs ONNX Runtime inference speed (Phase A12).

Exports ``yolo11n.onnx`` (if missing), then times end-to-end ``predict()`` FPS
(preprocess + inference + NMS) for:
  - PyTorch on CUDA (if available)
  - PyTorch on CPU
  - ONNX Runtime on CPU
Prints a table and writes ``reports/onnx_benchmark.md``.

Usage:
    python apps/benchmark_onnx.py
    python apps/benchmark_onnx.py --weights yolo11n.pt --imgsz 640 --runs 100
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402


def _sample_frame(imgsz: int) -> np.ndarray:
    """A representative BGR frame: a real clip frame if available, else synthetic."""
    import cv2

    clip = REPO_ROOT / "data" / "sample_videos" / "test_drive.mp4"
    if clip.is_file():
        cap = cv2.VideoCapture(str(clip))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
        ok, frame = cap.read()
        cap.release()
        if ok:
            return frame
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(imgsz, imgsz, 3), dtype=np.uint8)


def _bench(model: YOLO, frame, imgsz: int, device, runs: int, warmup: int = 10) -> float:
    """Return end-to-end FPS over ``runs`` predicts (after a warmup)."""
    is_cuda = device == 0 or (isinstance(device, str) and device.startswith("cuda"))
    for _ in range(warmup):
        model.predict(frame, imgsz=imgsz, device=device, verbose=False)
    if is_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(runs):
        model.predict(frame, imgsz=imgsz, device=device, verbose=False)
    if is_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return runs / elapsed if elapsed > 0 else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="PyTorch vs ONNX Runtime FPS benchmark")
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()

    weights = Path(args.weights)
    onnx_path = weights.with_suffix(".onnx")
    frame = _sample_frame(args.imgsz)
    print(f"Benchmarking ({args.runs} runs, imgsz={args.imgsz}) ...")

    rows: list[tuple[str, str, float]] = []

    # Measure PyTorch-CUDA FIRST: the ONNX export runs on CPU and can leave the
    # CUDA context unusable, so the GPU run must happen before it. Guarded so a
    # CUDA hiccup degrades to a CPU-only comparison instead of aborting.
    try:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            gpu = torch.cuda.get_device_name(0)
            rows.append(
                ("PyTorch", f"CUDA ({gpu})", _bench(YOLO(str(weights)), frame, args.imgsz, 0, args.runs))
            )
    except Exception as exc:  # hardware-dependent
        print(f"[warn] CUDA benchmark skipped: {exc}")

    rows.append(("PyTorch", "CPU", _bench(YOLO(str(weights)), frame, args.imgsz, "cpu", args.runs)))

    if not onnx_path.is_file():
        print(f"Exporting {weights} -> {onnx_path} (imgsz={args.imgsz}) ...")
        YOLO(str(weights)).export(format="onnx", imgsz=args.imgsz)
    rows.append(
        ("ONNX Runtime", "CPU", _bench(YOLO(str(onnx_path)), frame, args.imgsz, "cpu", args.runs))
    )

    header = "| Backend | Device | FPS | ms/frame |"
    sep = "| --- | --- | --- | --- |"
    table = [header, sep] + [f"| {b} | {d} | {fps:.1f} | {1000 / fps:.1f} |" for b, d, fps in rows]
    print("\n" + "\n".join(table))

    report = REPO_ROOT / "reports" / "onnx_benchmark.md"
    lines = [
        "# ONNX vs PyTorch Benchmark",
        "",
        f"Model: `{weights.name}` exported to `{onnx_path.name}` — imgsz {args.imgsz}, "
        f"{args.runs} runs, end-to-end `predict()` (preprocess + inference + NMS).",
        "",
        *table,
        "",
        "_ONNX Runtime uses the CPU execution provider here; the PyTorch CUDA row is the "
        "deployment reference. Numbers are end-to-end per-frame, so Python/pre/post overhead "
        "is included (realistic FPS, not pure kernel time)._",
        "",
    ]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
