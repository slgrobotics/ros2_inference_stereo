import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

#
# See https://github.com/slgrobotics/ros2_inference_stereo
# 
# make sure you have the following package in "src" directory:
#   git clone https://github.com/ros2/detection_visualizer.git
#
# Run it on the workstation with GUI:
#   colcon build; source install/setup.bash; ros2 launch ros2_inference_stereo vis.launch.py
#

def generate_launch_description():
    # Get the package directory
    package_dir = get_package_share_directory('ros2_inference_stereo')

    # Path to params YAML file (optional)
    params_file = os.path.join(package_dir, 'config', 'params.yaml')

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
        package='ros2_inference_stereo',
        executable='perception_adapter',
        name='perception_adapter',
        output='screen',
        parameters=[{
            "verbose": True,        # If true - print debug info about recognized and passed detections.
            'ticker_interval_sec': 0.1,
            'detection_topic': 'image_inference_detections',
            #'face_detected_sound': 'my_face.wav',
            'face_detected_text': 'I see you!',
            'min_confidence': 0.6,     # object detection confidence
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
        package='ros2_inference_stereo',
        #package='detection_visualizer',   # you can use ROS2 official https://github.com/ros2/detection_visualizer
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
        perception_adapter_node,
        detections_visualization_node,
        tf_to_map,
        rviz,
    ])
