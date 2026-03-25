import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

#
# See https://github.com/slgrobotics/jetson_nano_b01/blob/main/src/stereo/disparity_server.py
#
# colcon build; source install/setup.bash; ros2 launch ros2_inference_stereo ros2_disparity_client.launch.py
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

    # Visualize PointCloud2 in RViz2:
    rviz_config = os.path.join(package_dir, 'config', 'config.rviz')

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': False}],
        output='screen'
    )

    # static transform publisher for RViz2:
    tf_to_map = Node(package = "tf2_ros", 
                    executable = "static_transform_publisher",
                    arguments=[
                        '--x', '0.0',     # X translation in meters
                        '--y', '0.0',     # Y translation in meters
                        '--z', '1.0',     # Z translation in meters
                        '--roll', '0.0',  # Roll in radians
                        '--pitch', '0.0', # Pitch in radians
                        '--yaw', '0.0',   # Yaw in radians (e.g., 90 degrees)
                        '--frame-id', 'map', # Parent frame ID
                        '--child-frame-id', 'stereo_camera' # Child frame ID
                    ]
    )

    # Perception adapter node
    perception_adapter_node = Node(
        package='ros2_image_inference',
        executable='perception_adapter',
        name='perception_adapter',
        output='screen',
        parameters=[{
            "verbose": True,        # If true - print debug info about recognized and passed detections.
            'ticker_interval_sec': 0.1,
            'detection_topic': 'image_inference_detections',
            #'face_detected_sound': 'my_face.wav',
            'face_detected_text': 'I see you!',
            'min_confidence': 0.6,
            'face_cooldown_sec': 2.0,
            'gesture_cooldown_sec': 1.0,
            'camera_center_x': 320.0,  # Assuming 640x480 input images. Adjust if different.
            'target_label': 'person',  # The "face detected" logic is applied only to this object class.
            # Optional mapping of incoming class labels to "normalized class labels". Normally is not needed:
            'label_map_json': '{}',  # example: {"0":"person","1":"cat","cup_small":"cup"}',
            # Optional JSON string mapping object labels to "gestures". If not set, capitalized label will be used:
            'gesture_map_json': '{"bottle":"STOP", "cup":"OK", "banana":"LIKE", "cat":"MEOW", "dog":"WOOF"}',
            # Optional JSON string mapping gestures/commands to speech output. If not set, the gesture name will be used:
            'speech_map_json': '{"STOP":"Stop immediately!","OK":"good stuff!","scissors":"Careful, sharp!","FIRE_HYDRANT":"Fire hydrant - do not park!"}',
        }]
        # parameters=[params_file]  # Load params from YAML instead
    )

    detections_visualization_node = Node(
        package='detection_visualizer',
        executable='detection_visualizer',
        name='detection_visualizer',
        output='screen',
        remappings=[
            ('~/images', '/camera/image_raw'),
            ('~/detections', '/image_inference_detections'),
            ('~/dbg_images', '/image_inference_overlay'),
        ]
    )

    return LaunchDescription([
        inference_stereo_node,
        perception_adapter_node,
        detections_visualization_node,
        tf_to_map,
        rviz,
    ])
