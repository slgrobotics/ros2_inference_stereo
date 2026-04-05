#!/bin/bash

# ~/launch/bootup_launch.sh
# see https://github.com/slgrobotics/ros2_inference_stereo/blob/main/README.md#create-a-linux-service-for-on-boot-autostart

source /opt/ros/jazzy/setup.bash

cd /home/ros/robot_ws
colcon build
cd /home/ros/launch

source /home/ros/robot_ws/install/setup.bash

ros2 launch ros2_inference_stereo inference_stereo.launch.py
