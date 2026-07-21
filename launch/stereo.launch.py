import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

#
# See https://github.com/slgrobotics/ros2_inference_stereo
#     https://github.com/slgrobotics/articubot_one/wiki/Visual-SLAM-with-RTAB%E2%80%90Map
#
# Install dependencies:
#   sudo apt install ros-jazzy-vision-msgs
#
# Run it on a Raspberry Pi with binocular cameras:
#   colcon build; source install/setup.bash; ros2 launch ros2_inference_stereo stereo.launch.py
#

def generate_launch_description():
    # Get the package directory
    package_dir = get_package_share_directory('ros2_inference_stereo')

    # Path to stereo configuration NPZ file
    calib_file = os.path.join(package_dir, 'config', 'calib_820x616.npz')

    # Path to params YAML file (optional)
    params_file = os.path.join(package_dir, 'config', 'params.yaml')

    # disparity client node compatible with RTAB-Map
    stereo_node = Node(
        package='ros2_inference_stereo',
        executable='stereo_node',
        name='stereo_node',
        output='screen',
        parameters=[{
            'calibration_file': calib_file,
            'image_topic': "camera/image_raw",
            'camera_info_topic': "camera/camera_info",
            'depth_image_topic': "stereo/depth/image_rect_raw",  # Empty string disables depth image publishing
            'frame_id': "stereo_camera",
            'max_depth_range_m': 5.0,  # cut-off range for detecting in depth image
            'close_cutout_factor': 1.0,
            'far_smoothing_factor': 1.0,
            'min_valid_disp': 1.0,
            'loop_delay_sec': 0.01, # short "sleep" after processing to free CPU
        }]
        # parameters=[params_file]  # Load params from YAML instead
    )

    return LaunchDescription([
        stereo_node,
    ])
