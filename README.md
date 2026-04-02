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
* **OS:** Ubuntu 24.04 LTS (Noble Numbat). The Desktop version is required for calibration. You can use this [guide](https://github.com/slgrobotics/robots_bringup/blob/main/Docs/Ubuntu-RPi/README.md) as a reference. 
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
## *Important:* Calibration Is Not Optional

Stereo vision relies on properly calibrated cameras.

On the Raspberry Pi 5:

```bash
mkdir -p ~/robot_ws/src
cd ~/robot_ws/src
git clone https://github.com/slgrobotics/ros2_inference_stereo.git
```

The *[calib](https://github.com/slgrobotics/ros2_inference_stereo/tree/main/calib)* folder contains the necessary scripts:
- `capture_stereo_pairs.py` — collects a set of ~50 stereo pairs while you move the checkerboard through a wide range of positions and orientations
- `calib_file_generator.py` — generates the calibration file (e.g., `calib_820x616.npz`)
- `disparity_viewer.py` — allows you to validate the calibration before integrating with ROS2
- other files — useful for debugging and experimentation

To create a large checkerboard, I printed several 3x2 boards and glued them onto cardboard.

Make sure that the checkerboard square size and camera baseline are correctly configured in [config.py](https://github.com/slgrobotics/ros2_inference_stereo/blob/main/config/config.py)

**Note:** Camera *Field of View*
- the 105°(D) FOV specification means: 
  - Diagonal measurement: the 105° angle is measured from one corner of the image to the opposite corner
  - Effective area: approximately 85°–90° horizontally and 60°–65° vertically
- run `tests/print_sensor_modes.py` and look for `crop_limits: (0, 0, 3280, 2464)`
- to use the full FOV set `RAW_*=1640x1232`  (half of the IMX219 full resolution: 3280x2464)
- other sensor modes may use only the central portion of the sensor, resulting in a narrower FOV.
- `SCALE_BY = 2` ensures that processing pipelines operate at 820x616 resolution, which is sufficient for most use cases and provides good performance.

Calibration board generator:
- https://markhedleyjones.com/projects/calibration-checkerboard-collection
- https://markhedleyjones.com/media/projects/calibration-checkerboard-collection/Checkerboard-A4-30mm-8x6.pdf
- https://markhedleyjones.com/media/projects/calibration-checkerboard-collection/Checkerboard-A4-70mm-3x2.pdf

Here is how to use the checkerboard during calibration: https://youtu.be/G-Iw35VecI8

## Image transport: Raw vs Compressed

Both `inference_stereo_node.py` and `detection_visualizer_node.py` provide a parameter that controls the type of image transport
```
'image_topic': "camera/image_raw/compressed",  # or "camera/image_raw", if WiFi traffic is not a concern.
```

The *"/compressed"* suffix enables JPEG compression - RPi5 CPU will take a small hit, while WiFi traffic will be significuntly lower:
```
nload wlan0
Avg: 14.4 MBit/s   - Raw
Avg: 635  KBit/s   - Compressed
```

This setting must be consistent on both sides - the publisher (RPi5) and subscriber (the workstation in this example).

The `/camera/image_inference_overlay` topic is always published uncompressed, as RViz2 and RQT do not support compressed image topics.

You can still view the compressed raw images published by the RPi5 on your workstation:
```
ros2 run image_view image_view --ros-args \
  -r image:=/camera/image_raw \
  -p image_transport:=compressed
```

## CameraInfo and Simulated LaserScan

The *CameraInfo* message is derived from stereo calibration data and is published by RPi5 node:
```
'camera_info_topic': "camera/camera_info"
```

Any *PointCloud2* message can be "sliced" into a *LaserScan* - see `launch/vis.launch.py` for usage example.

## Quick Start - ROS node on RPi5 and RViz2 on the workstation

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
mkdir -p ~/robot_ws/src
cd ~/robot_ws/src
git clone https://github.com/slgrobotics/ros2_inference_stereo.git
cd ~/robot_ws

colcon build; source install/setup.bash; ros2 launch ros2_inference_stereo inference_stereo.launch.py
```

### (On the workstation) Run Detection Visualizer and RViz2

```bash
sudo apt install ros-${ROS_DISTRO}-pointcloud-to-laserscan

mkdir -p ~/inf_stereo_ws/src
cd ~/inf_stereo_ws/src
git clone https://github.com/slgrobotics/ros2_inference_stereo.git
cd ~/inf_stereo_ws

colcon build; source install/setup.bash; ros2 launch ros2_inference_stereo vis.launch.py
```

## Create a Linux service for on-boot autostart

To turn your RPi5 into a headless "camera appliance" follow these steps:

1. Create and populate launch folder:
```
mkdir ~/launch
cd ~/launch
# place bootup_launch.sh here:
cp ~/robot_ws/src/ros2_inference_stereo/sys/bootup_launch.sh .
```

Try running the _bootup_launch.sh_ from the command line to see if anything fails.

2. Deploy service description file:
```
sudo cp ~/robot_ws/src/ros2_inference_stereo/sys/robot.service /etc/systemd/system/.
```

> **Note:** 
> Logs are stored in _/home/ros/.ros/log_ folder - these can grow if things go wrong.
> 
> You may want to edit parameters related to logging:
> ```
> # vi ~/robot_ws/src/ros2_inference_stereo/launch/inference_stereo.launch.py
> 'verbose': False,          # If true - print debug info.
> 'log_every_n_packets': 0,  # 0 for no log
> ```

3. Enable service:
```
sudo systemctl daemon-reload
sudo systemctl enable robot.service
sudo systemctl start robot.service
```
If all went well, the service will start automatically after you reboot the RPi, and all related nodes will show up on _rpt_ and _rpt_graph_

## Promptable Object recognition - YOLOE

Ultralytics provides a way to run other models, including a very recent *"yoloe-11s-seg-pf.pt"*.

**YOLOE** uses text prompts instead of fixed labels. So rather than being limited to, say, *“dog”* or *“car,”* you can tell YOLOE to look for a *“red mug”* or *“vintage camera”*, even if it hasn’t seen those exact things during training.

Here is a video tutorial by *Core Electronics Pty Ltd* (Australia) explaining how it works: https://youtu.be/yNPwsKa52zs  (thanks, Michael Wimble, for the find!)
Their text tutorial is [here](https://core-electronics.com.au/guides/raspberry-pi/custom-object-detection-models-without-training-yoloe-and-raspberry-pi/).

The model can be easily deployed by just changing `'model_path': 'models/yoloe-11s-seg-pf.pt'` in `launch/inference_stereo.launch.py`
Without optimization and AI Hat it is noticeably slower though (about 4 seconds per 820x616 frame).

## Faster/optimized model format - ONNX

You can use an optimized model format to significantly improve object detection performance.

First, install the prerequisites:
```
pip install --break-system-packages onnx onnxruntime onnxslim
```

If you have already used a `.pt` model, it will typically be located in either `~/robot_ws/models` or `~/launch/models`. Navigate to that directory and run:
```
yolo export model=yolo11n.pt format=onnx imgsz=640,832
  or
yolo export model=yoloe-11s-seg-pf.pt format=onnx imgsz=640,832
```

Be patient — there may be a short delay before Ultralytics starts printing progress to the terminal.

You can then update your launch file to use the exported model:
```
'model_path': 'models/yolo11n.onnx',
```

In practice, `yolo11n.onnx` runs about twice as fast as the `.pt` model, while `yoloe-11s-seg-pf.onnx` shows an improvement of roughly 1.4×.

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

| Topic                                | Type                           | Description                     |
| ------------------------------------ | ------------------------------ | ------------------------------- |
| `/camera/image_raw`                  | `sensor_msgs/Image`            | Raw camera image                |
| `/camera/camera_info`                | `sensor_msgs/CameraInfo`       | Camera calibration, FOV etc.    |
| `/image_inference_detections`        | `vision_msgs/Detection2DArray` | YOLO detections                 |
| `/camera/image_inference_overlay`    | `sensor_msgs/Image`            | Debug image with bounding boxes |
| `/points`                            | `sensor_msgs/PointCloud2`      | Sparse stereo point cloud       |

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
