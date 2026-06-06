# ONNX vs PyTorch Benchmark

Model: `yolo11n.pt` exported to `yolo11n.onnx` — imgsz 640, 100 runs, end-to-end `predict()` (preprocess + inference + NMS).

| Backend | Device | FPS | ms/frame |
| --- | --- | --- | --- |
| PyTorch | CUDA (NVIDIA GeForce RTX 3050 Laptop GPU) | 90.5 | 11.1 |
| PyTorch | CPU | 23.8 | 42.0 |
| ONNX Runtime | CPU | 23.2 | 43.2 |

_ONNX Runtime uses the CPU execution provider here; the PyTorch CUDA row is the deployment reference. Numbers are end-to-end per-frame, so Python/pre/post overhead is included (realistic FPS, not pure kernel time)._
