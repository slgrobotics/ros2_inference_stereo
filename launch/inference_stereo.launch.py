import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

#
# See https://github.com/slgrobotics/ros2_inference_stereo
#
# Install dependencies:
#   sudo apt install ros-jazzy-vision-msgs
#   python3 -m pip install ultralytics "numpy<2" --break-system-packages
#
# Run it on Raspberry Pi with binocular cameras:
#   colcon build; source install/setup.bash; ros2 launch ros2_inference_stereo inference_stereo.launch.py
#

def generate_launch_description():
    # Get the package directory
    package_dir = get_package_share_directory('ros2_inference_stereo')

    # Path to stereo configuration NPZ file
    calib_file = os.path.join(package_dir, 'config', 'calib_820x616.npz')

    # Path to params YAML file (optional)
    params_file = os.path.join(package_dir, 'config', 'params.yaml')

    # disparity client node
    inference_stereo_node = Node(
        package='ros2_inference_stereo',
        executable='inference_stereo_node',
        name='inference_stereo_node',
        output='screen',
        parameters=[{
            'verbose': True,        # If true - print debug info.
            'log_every_n_packets': 10,    # 0 for no log
            'calibration_file': calib_file,

            # ONNX: see this guide: https://github.com/slgrobotics/ros2_inference_stereo/blob/main/README.md#fasteroptimized-model-format---onnx
            'model_path': 'models/yolo11n.pt',  # relative to where you launch, e.g. "~/robot_ws/models/*" or "~/launch/models/*"
            #'model_path': 'models/yolo11n.onnx',  # approx 2x faster than the .pt but may have slightly lower accuracy
            #'model_path': 'models/yoloe-11s-seg-pf.pt',    # prompt-free version of yoloe-11s-seg
            #'model_path': 'models/yoloe-11s-seg-pf.onnx',  # approx 1.4x faster than the .pt
            #'model_path': 'models/yoloe-11s-seg.pt',       # run this first to bring it into "models" directory, then run "launch/export_onnx.py"
            #'model_path': 'models/yoloe-11s-seg.onnx',     # when exported by "launch/export_onnx.py" - with custom dictionary

            'image_topic': "camera/image_raw/compressed",  # or "camera/image_raw", if WiFi traffic is not a concern.
            'jpeg_quality': 80,            # JPEG quality for compressed image output (1-100, higher is better quality and larger size)
            'camera_info_topic': "camera/camera_info",
            'depth_image_topic': "stereo/depth/image_rect_raw",
            'cloud_topic': "stereo/sparse_cloud",
            'detection_topic': 'image_inference_detections',
            'frame_id': "stereo_camera",
            'grid_size': 16,  # Grid size NxN for sparse sampling
            'close_cutout_factor': 1.0,
            'far_smoothing_factor': 1.0,
            'color_patch_fraction': 0.5,  # center patch size relative to cell
            'use_mean_color': True,
            'min_valid_disp': 1.0,
            'min_disp_confidence': 0.02,  # do not publish if stereo disparity confidence is below this threshold
            'pointcloud_delay_sec': 0.02, # short "sleep" after pointcloud processing to free CPU
            'detect_delay_sec': 0.02,     # short "sleep" after detections processing to free CPU
            'min_confidence': 0.6,        # object detection confidence threshold
            'objects_allowed': [''],  # Empty list means allow all detected objects.
            #'objects_allowed': ['person', 'cup', 'bottle', 'cell phone', 'banana', 'book', 'scissors', 'dog', 'cat'], # Not case sensitive.
        }]
        # parameters=[params_file]  # Load params from YAML instead
    )

    return LaunchDescription([
        inference_stereo_node,
    ])
