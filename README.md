Back to [Main Project Home](https://github.com/slgrobotics/articubot_one/wiki)

If you are looking for similar functionality on Jetson Nano B01 - see this [repo](https://github.com/slgrobotics/jetson_nano_b01/blob/main/README.md).

This project was inspired by excellent work by Eryk Pawełek - here is his [repo](https://github.com/erykpawelek/ros2_vision_playground).

## ROS2 Object Recognition and Stereo Vision on Raspberry Pi 5

A ROS2 node to perform object recognition and stereo-to-PointCloud2 publishing on a Raspberry Pi 5.

This package combines real-time stereo depth estimation with neural network–based object detection into a lightweight,
self-contained perception pipeline suitable for edge robotics applications.

<img alt="RPI5_stereo_RViz" src="https://github.com/user-attachments/assets/ba255693-3e80-4704-a0b7-ae2338170fba" />

### Performance metrics

Here are the actual message rates:
```
ros2 topic hz /camera/image_raw
average rate: 0.969

ros2 topic hz /image_inference_detections
average rate: 0.992

ros2 topic hz /stereo/sparse_cloud
average rate: 1.974
```
All cores on RPi5 stay 80..90% busy, so there isn't much else the machine can do without an AI [accelerator](https://www.amazon.com/AI-Kit-Raspberry-Pi-Acceleration/dp/B0D8PF8WT4).

**Note:** there are two parameters that let processing loops sleep between cycles, higher values free CPUs:
```
'pointcloud_delay_sec': 0.02, # short "sleep" after pointcloud processing to free CPU
'detect_delay_sec': 0.02,     # short "sleep" after detections processing to free CPU
```

### Hardware & Prerequisites

* **Hardware:**
  * Raspberry Pi 5 (8GB) - please review this [guide](https://github.com/slgrobotics/articubot_one/wiki/Properly-feeding-your-Raspberry-Pi-5).
  * Dual *Arducam 8MP IMX219 Camera [Module](https://www.amazon.com/dp/B09VSRH14M)* (CSI).
* **OS:** Ubuntu 24.04 LTS (Noble Numbat). You need Desktop version for calibration.
* **ROS Distro:** ROS 2 Jazzy Jalisco.
* a 32 GB **SD card** is sufficient ("[high endurance](https://www.amazon.com/dp/B07P14QHB7)" type recommended).
* Connect Ethernet cable, keyboard and monitor while installing and configuring. Ensure SSH access.

> **Important:**
> * Standard Ubuntu drivers do not support the CSI-connected cameras correctly out of the box. You MUST either:
>   - install pre-build binaries as described below
>   - or, build the Raspberry Pi fork of `libcamera` from [source](https://github.com/erykpawelek/libcamera_ros2_setup).
> * "Binocular" [cameras](https://www.amazon.com/IMX219-83-Stereo-Camera-Compatible-Applications/dp/B088RFT412) for Jetson Nano do not come with proper cables and are useless for Raspberry Pi 5

Here is my DIY setup:

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

On the Raspberry Pi 5:

```bash
mkdir -p ~/inf_stereo_ws/src
cd ~/inf_stereo_ws/src
git clone https://github.com/slgrobotics/ros2_inference_stereo.git
```

The *[calib](https://github.com/slgrobotics/ros2_inference_stereo/tree/main/calib)* folder contains necessary scripts:
- `capture_stereo_pairs.py` allows you to collect a set of 50 *stereo pairs*, while positioning the checkerboard in every possible way
- `calib_file_generator.py` generates a calibration file (calib_820x616.npz)
- `disparity_viewer.py` lets you validate calibration before you go into ROS2
- other files are useful for debugging etc.

To make a large checkerboard I printed several 3x2 boards and glued them to a cardboard.

Make sure that the sizes of your squares and camera base are reflected in [config.py](https://github.com/slgrobotics/ros2_inference_stereo/blob/main/config/config.py)

**Note:** Camera *Field of View*
- the 105°(D) FOV specification means: 
  - Diagonal Measurement (D): The 105° angle is measured from one corner of the image to the opposite corner.
  - Effective Area: you can expect approximately 85°–90° horizontally and 60°–65° vertically.
- run `tests/print_sensor_modes.py` and look for `crop_limits: (0, 0, 3280, 2464)`
- to use full FOV use `RAW_*=1640x1232`  (half of IMX219 full resolution - 3280x2464)
- other modes may use only the central part of the sensor, and FOV will be narrower.
- `SCALE_BY = 2` ensures that processing pipelines operate on 820x616 resolution, sufficient for all needs and fast enough.

Calibration board generator:
- https://markhedleyjones.com/projects/calibration-checkerboard-collection
- https://markhedleyjones.com/media/projects/calibration-checkerboard-collection/Checkerboard-A4-30mm-8x6.pdf
- https://markhedleyjones.com/media/projects/calibration-checkerboard-collection/Checkerboard-A4-70mm-3x2.pdf

Here is how to use the checkerboard during calibration: https://youtu.be/G-Iw35VecI8

## Quick Start - ROS node on RPi5 and RViz on the workstation

### (on Raspberry Pi) Install prerequisites

Edit `sudo vi /boot/firmware/config.txt` to enable cameras, reboot:
```
[all]
camera_auto_detect=0
dtoverlay=imx219,cam0
dtoverlay=imx219,cam1
```

Follow this [guide](https://github.com/slgrobotics/robots_bringup/blob/main/Docs/Sensors/Camera.md#installation) to install *libcamera/picamera2* binaries.

Use these [scripts](https://github.com/slgrobotics/ros2_inference_stereo/tree/main/tests) to verify that the cameras are working.

**Important:** [calibrate](https://github.com/slgrobotics/ros2_inference_stereo?tab=readme-ov-file#important-calibration-is-not-optional) your cameras.

You need a ROS package and YOLO driver from Ultralytics (takes time to install):
```
sudo apt install ros-${ROS_DISTRO}-vision-msgs
python3 -m pip install ultralytics "numpy<2" --break-system-packages

```

### (On Raspberry Pi) Build and run

```bash
mkdir -p ~/inf_stereo_ws/src
cd ~/inf_stereo_ws/src
git clone https://github.com/slgrobotics/ros2_inference_stereo.git
cd ~/inf_stereo_ws

colcon build; source install/setup.bash; ros2 launch ros2_inference_stereo inference_stereo.launch.py
```

### (On the workstation) Run Detection Visualizer and RViz2

```bash
mkdir -p ~/inf_stereo_ws/src
cd ~/inf_stereo_ws/src
git clone https://github.com/slgrobotics/ros2_inference_stereo.git
cd ~/inf_stereo_ws

colcon build; source install/setup.bash; ros2 launch ros2_inference_stereo vis.launch.py
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

## Behavior trees connection - Perception Adapter

See this [guide](https://github.com/slgrobotics/ros2_jetson_nano_inference?tab=readme-ov-file#behavior-trees-connection---perception-adapter) for info.

See this [guide](https://github.com/slgrobotics/slg_bt_plugins) for information on Behavior Trees *plugins*.

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
