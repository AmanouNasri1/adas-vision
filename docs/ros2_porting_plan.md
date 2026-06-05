# ROS 2 Porting Plan (bridge to Project B)

This Windows workbench (Project A) is the precursor to a ROS 2 + CARLA system
(Project B). Because every module already takes plain data in / out with no
hidden state, the port is a node-wrapping exercise rather than a rewrite.

| Windows module | Future ROS 2 node | Notes |
| --- | --- | --- |
| `input/video_loader.py` | `/sim/front/rgb` subscriber / camera node | Replace file frames with ROS `Image` msgs |
| `perception/detector.py` | `detection_node.py` | Reuse the YOLO inference logic |
| `tracking/tracker.py` | `tracking_node.py` | Publish tracks as a custom msg / JSON |
| `perception/lane_detector.py` | `lane_drivable_node.py` | Publish the lane mask image |
| `risk/risk_estimator.py` | `risk_estimation_node.py` | Upgrade pseudo-distance → depth-based TTC |
| `logging/event_logger.py` | rosbag + CSV logger | Keep CSV outputs for reports |
| `dashboard/app.py` | same dashboard backend | Read the latest ROS-derived state |

## What changes vs. what stays

**Stays:** detection / tracking / risk / overlay logic, the config-driven design,
the CSV/JSON log formats, and the dashboard backend.

**Changes:** the frame source becomes a ROS topic; the pseudo-distance/TTC proxy
is replaced by **real depth-based TTC** (CARLA provides depth / ground truth);
inter-module data moves over ROS topics instead of in-process function calls.
