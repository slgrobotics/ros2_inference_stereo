import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

#
# See https://github.com/slgrobotics/ros2_inference_stereo
#
# Run it on Raspberry Pi with binocular cameras:
#   colcon build; source install/setup.bash; ros2 launch ros2_inference_stereo inference_stereo.launch.py
#

def generate_launch_description():
    # Get the package directory
    package_dir = get_package_share_directory('ros2_inference_stereo')

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
            'bind_ip': "0.0.0.0",
            'port': 5005,
            'image_topic': "camera/image_raw",
            'cloud_topic': "stereo/sparse_cloud",
            'detection_topic': 'image_inference_detections',
            'frame_id': "stereo_camera",
            'ticker_interval_sec': 0.1,   # 10 Hz UDP socket poll timer
            'socket_timeout_sec': 0.0,    # 0 for non-blocking
            'log_every_n_packets': 10,    # 0 for no log
            'color_patch_fraction': 0.5,  # center patch size relative to cell
            'use_mean_color': True,
            'tcp_host': "jetson.local",
            'tcp_port': 5006,
            'request_image_every_sec': 0.5,
            'jpeg_max_width': 320,
            'jpeg_max_height': 180,
            'jpeg_quality': 60,
            'tcp_timeout_sec': 5.0,
        }]
        # parameters=[params_file]  # Load params from YAML instead
    )

    return LaunchDescription([
        inference_stereo_node,
    ])
