# ONNX vs PyTorch Benchmark

Model: `yolo11n.pt` exported to `yolo11n.onnx` — imgsz 640, 80 runs, end-to-end `predict()` (preprocess + inference + NMS).

| Backend | Device | FPS | ms/frame |
| --- | --- | --- | --- |
| PyTorch | CUDA (NVIDIA GeForce RTX 3050 Laptop GPU) | 71.4 | 14.0 |
| PyTorch | CPU | 22.0 | 45.5 |
| ONNX Runtime | CPU | 20.3 | 49.3 |

_ONNX Runtime uses the CPU execution provider here; the PyTorch CUDA row is the deployment reference. Numbers are end-to-end per-frame, so Python/pre/post overhead is included (realistic FPS, not pure kernel time)._
