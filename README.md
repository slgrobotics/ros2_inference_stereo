Back to [Main Project Home](https://github.com/slgrobotics/articubot_one/wiki)

If you are looking for similar functionality on Jetson Nano B01 - see this [repo](https://github.com/slgrobotics/jetson_nano_b01/blob/main/README.md).

This project was inspired by excellent work by Eryk Pawełek - here is his [repo](https://github.com/erykpawelek/ros2_vision_playground).

## ROS2 Object Recognition and Stereo Vision on Raspberry Pi 5

A ROS2 node to perform object recognition and stereo-to-PointCloud2 publishing on a Raspberry Pi 5.

This package combines real-time stereo depth estimation with neural network–based object detection into a lightweight,
self-contained perception pipeline suitable for edge robotics applications.

<img alt="RPI5_stereo_RViz" src="https://github.com/user-attachments/assets/ba255693-3e80-4704-a0b7-ae2338170fba" />

### Hardware & Prerequisites

* **Hardware:**
  * Raspberry Pi 5 (8GB RAM)
  * Dual *Arducam 8MP IMX219 Camera [Module](https://www.amazon.com/dp/B09VSRH14M)* (CSI).
* **OS:** Ubuntu 24.04 LTS (Noble Numbat).
* **ROS Distro:** ROS 2 Jazzy Jalisco.

> **Important:**
> * Standard Ubuntu drivers do not support the CSI-connected cameras correctly out of the box. You MUST either:
>   - install pre-build binaries as described below
>   - or, build the Raspberry Pi fork of `libcamera` from [source](https://github.com/erykpawelek/libcamera_ros2_setup).
> * "Binocular" [cameras](https://www.amazon.com/IMX219-83-Stereo-Camera-Compatible-Applications/dp/B088RFT412) for Jetson Nano do not come with proper cables and are useless for Raspberry Pi 5

<img width="2301" height="746" alt="Screenshot from 2026-03-29 09-48-03" src="https://github.com/user-attachments/assets/1d8dd47f-b123-488f-8385-1a31b78f4167" />

<img width="1725" height="1508" alt="Screenshot from 2026-03-29 09-46-27" src="https://github.com/user-attachments/assets/0b5e6061-2988-43c8-8d3a-4a8217b6172a" />

### Features

**Stereo vision pipeline**
  * Rectified dual-camera input (e.g., CSI or USB cameras)
  * Disparity computation using OpenCV StereoSGBM
  * Sparse depth extraction on a configurable grid
  * Conversion to `sensor_msgs/PointCloud2` with per-point:
    * XYZ coordinates
    * confidence
    * image grid location
    * RGB color sampled from the source image

**Object detection (YOLO / Ultralytics)**
  * Supports TensorRT / CPU inference depending on platform
  * Efficient inference loop decoupled from stereo processing
  * Automatic handling of image resizing / letterboxing
  * Outputs `vision_msgs/Detection2DArray`

**ROS2-native integration**
  * Publishes:
    * `camera/image_raw`
    * `image_inference_detections`
    * `points` (`PointCloud2`)
  * Optional visualization node for overlaying detections on images
  * Consistent timestamping across image, detections, and point cloud

**Threaded execution model**
  * Separate callback groups for:
    * stereo / point cloud generation (high priority)
    * object detection (lower priority)
  * One-shot timer pattern to avoid callback pile-up under load

**Configurable behavior**
  * Detection filtering by label and confidence
  * Adjustable grid size for depth sampling
  * Tunable stereo and inference parameters
  * Optional mean-color sampling for point cloud coloring

### Design Goals

* Run fully on embedded hardware (Raspberry Pi 5 class devices)
* Keep ROS2 interfaces clean and standard-compliant
* Provide a modular structure (helpers for stereo, detection, ROS conversion)

### Typical Use Cases

* Mobile robot perception (Nav2 integration)
* Obstacle detection and local mapping
* Human / object awareness for behavior trees
* Lightweight edge AI “appliance” for distributed robotics systems

### Notes

* Object detection bounding boxes are reported in original image coordinates (Ultralytics handles internal resizing and remapping).
* Depth estimation and detection operate on the same image stream but are intentionally decoupled for performance.
* For best results, ensure proper stereo calibration and consistent camera synchronization.

---
## *Important:* Calibration is not optional

Stereo vision relies on properly calibrated cameras.

## Quick Start

### 1. Build

```bash
cd ~/rpi5_ws
colcon build --packages-select ros2_inference_stereo
source install/setup.bash
```

### 2. Run the Node

```bash
ros2 run ros2_inference_stereo inference_stereo_node
```

### 3. (Optional) Run Detection Visualizer

```bash
ros2 run ros2_inference_stereo detection_visualizer \
  --ros-args \
  -p image_topic:=/camera/image_raw \
  -p detection_topic:=/image_inference_detections \
  -p overlay_image_topic:=/image_inference_overlay
```

### 4. View Topics

```bash
# Raw camera image
ros2 run image_view image_view --ros-args -r image:=/camera/image_raw

# Overlay with detections
ros2 run image_view image_view --ros-args -r image:=/image_inference_overlay

# Point cloud (RViz2 recommended)
rviz2
```

## System Architecture

This package is designed as a **modular perception pipeline** with clear separation between acquisition, processing, and ROS interfaces.

### Data Flow

```
Stereo Cameras
    │
    ▼
Capture + Rectification
    │
    ├──► Disparity (StereoSGBM)
    │         │
    │         ▼
    │   Sparse Depth Grid
    │         │
    │         ▼
    │   PointCloud2 Publisher
    │
    └──► Object Detection (YOLO)
              │
              ▼
      Detection2DArray Publisher
              │
              ▼
      Detection Visualizer (optional)
```
<img width="1810" height="355" alt="Screenshot from 2026-03-29 09-42-23" src="https://github.com/user-attachments/assets/a8c5680e-5a5d-4927-8cc5-ff64dc22edea" />

## Timing and Synchronization

* All outputs (image, detections, point cloud) share the **same source image timestamp**
* Detection latency does **not affect synchronization**, since timestamps are preserved
* `ApproximateTimeSynchronizer` is used only for visualization alignment

## Topics

| Topic                         | Type                           | Description                     |
| ----------------------------- | ------------------------------ | ------------------------------- |
| `/camera/image_raw`           | `sensor_msgs/Image`            | Raw camera image                |
| `/image_inference_detections` | `vision_msgs/Detection2DArray` | YOLO detections                 |
| `/image_inference_overlay`    | `sensor_msgs/Image`            | Debug image with bounding boxes |
| `/points`                     | `sensor_msgs/PointCloud2`      | Sparse stereo point cloud       |

## Key Parameters

| Parameter                | Description                         |
| ------------------------ | ----------------------------------- |
| `min_confidence`         | Detection confidence threshold      |
| `objects_allowed`        | Optional whitelist of object labels |
| `grid_rows`, `grid_cols` | Resolution of depth sampling grid   |
| `time_slop`              | Sync tolerance for visualization    |
| `use_mean_color`         | Use average color per grid cell     |

## Design Notes

* **Stereo and detection pipelines are decoupled**
  → prevents slow inference from blocking depth estimation

* **One-shot timers instead of fixed-rate timers**
  → avoids backlog and keeps latency bounded

* **Threaded execution model**
  → prioritizes point cloud generation over detection

* **Ultralytics handles resizing internally**
  → bounding boxes are returned in original image coordinates

-------------------------

Back to [Main Project Home](https://github.com/slgrobotics/articubot_one/wiki)
