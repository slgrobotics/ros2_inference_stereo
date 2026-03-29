Back to [Main Project Home](https://github.com/slgrobotics/articubot_one/wiki)

If you are looking for similar functionality on Jetson Nano B01 - see this [repo](https://github.com/slgrobotics/jetson_nano_b01/blob/main/README.md).

This project was inspired by excellent work by Eryk Pawełek - here is his [repo](https://github.com/erykpawelek/ros2_vision_playground).

## ROS2 Object Recognition and Stereo Vision on Raspberry Pi 5

A ROS2 node to perform object recognition and stereo-to-PointCloud2 publishing on a Raspberry Pi 5.

This package combines real-time stereo depth estimation with neural network–based object detection into a lightweight,
self-contained perception pipeline suitable for edge robotics applications.

### Hardware & Prerequisites

* **Hardware:** Raspberry Pi 5 (8GB RAM) + Dual Arducam 8MP IMX219 Camera Module (CSI).
* **OS:** Ubuntu 24.04 LTS (Noble Numbat).
* **ROS Distro:** ROS 2 Jazzy Jalisco.

> **Important:** Standard Ubuntu drivers do not support the RPi Camera Module 3 correctly out of the box. You MUST build the Raspberry Pi fork of `libcamera` from source. 
> See my detailed guide here: **[Raspberry Pi 5 + Camera Module 3 Setup Guide](https://github.com/erykpawelek/libcamera_ros2_setup)**

### Features

* **Stereo vision pipeline**

  * Rectified dual-camera input (e.g., CSI or USB cameras)
  * Disparity computation using OpenCV StereoSGBM
  * Sparse depth extraction on a configurable grid
  * Conversion to `sensor_msgs/PointCloud2` with per-point:

    * XYZ coordinates
    * confidence
    * image grid location
    * RGB color sampled from the source image

* **Object detection (YOLO / Ultralytics)**

  * Supports TensorRT / CPU inference depending on platform
  * Efficient inference loop decoupled from stereo processing
  * Automatic handling of image resizing / letterboxing
  * Outputs `vision_msgs/Detection2DArray`

* **ROS2-native integration**

  * Publishes:

    * `camera/image_raw`
    * `image_inference_detections`
    * `points` (`PointCloud2`)
  * Optional visualization node for overlaying detections on images
  * Consistent timestamping across image, detections, and point cloud

* **Threaded execution model**

  * Separate callback groups for:

    * stereo / point cloud generation (high priority)
    * object detection (lower priority)
  * One-shot timer pattern to avoid callback pile-up under load

* **Configurable behavior**

  * Detection filtering by label and confidence
  * Adjustable grid size for depth sampling
  * Tunable stereo and inference parameters
  * Optional mean-color sampling for point cloud coloring

### Design Goals

* Run fully on embedded hardware (Raspberry Pi 5 class devices)
* Maintain deterministic behavior under CPU load
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

---

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

---

## Timing and Synchronization

* All outputs (image, detections, point cloud) share the **same source image timestamp**
* Detection latency does **not affect synchronization**, since timestamps are preserved
* `ApproximateTimeSynchronizer` is used only for visualization alignment

---

## Topics

| Topic                         | Type                           | Description                     |
| ----------------------------- | ------------------------------ | ------------------------------- |
| `/camera/image_raw`           | `sensor_msgs/Image`            | Raw camera image                |
| `/image_inference_detections` | `vision_msgs/Detection2DArray` | YOLO detections                 |
| `/image_inference_overlay`    | `sensor_msgs/Image`            | Debug image with bounding boxes |
| `/points`                     | `sensor_msgs/PointCloud2`      | Sparse stereo point cloud       |

---

## Key Parameters

| Parameter                | Description                         |
| ------------------------ | ----------------------------------- |
| `min_confidence`         | Detection confidence threshold      |
| `objects_allowed`        | Optional whitelist of object labels |
| `grid_rows`, `grid_cols` | Resolution of depth sampling grid   |
| `time_slop`              | Sync tolerance for visualization    |
| `use_mean_color`         | Use average color per grid cell     |

---

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
