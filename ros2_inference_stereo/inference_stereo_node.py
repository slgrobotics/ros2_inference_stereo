#!/usr/bin/env python3

# ========================================================
# ROS 2 stereo client node for sparse point cloud and preview image visualization.
#
# This node captures images from dual cameras and applies AI models for object recognition.
#
# This node also constructs a PointCloud2 message, and publishes it for downstream use
# (e.g., Nav2 obstacle processing and RViz2 visualization).
#
# It also publishes the latest rectified left preview image as a raw ROSimage,
# and derives approximate per-point RGB values from the corresponding grid cells.
#
# The resulting PointCloud2 contains:
# - geometry (x, y, z)
# - packed RGB color for visualization
# - confidence
# - grid row/column metadata
#
# Key features:
# - Captures images from dual cameras
# - Applies an inference model (e.g. yolo11n) to the image
# - publishes Detection2DArray message for perception_adapter.py or other ROS2 consumers
# - Performs disparity calculations to derive stereo depth
# - Publishes raw image for RViz2 and RQt
# - Calculates approximate point-cloud colorization from image grid cells
# - Publishes PointCloud2 topic usable for both debugging and navigation
#
# Intended use:
# - Object detection and classification for use by Behavior trees
# - Visualizing stereo-derived sparse point clouds in RViz2
# - Supplying point clouds to Nav2 / local costmap obstacle layers
# - Debugging perception alignment between cloud and camera image
#
# ========================================================

import threading

import struct
from typing import List, Tuple
import json
import numpy as np

import cv2
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import Header
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2

from ros2_inference_stereo.helper_picamera import CameraDriver, Picamera2Capture
from helpers_pointcloud import PointCloudHelper


class InferenceStereoNode(Node):
    def __init__(self) -> None:
        super().__init__("inference_stereo_node")

        self.declare_parameter("verbose", False)
        self.declare_parameter("cloud_topic", "stereo/sparse_cloud")
        self.declare_parameter("frame_id", "stereo_camera")
        self.declare_parameter("color_patch_fraction", 0.5)   # center patch size relative to cell
        self.declare_parameter("use_mean_color", True)
        self.declare_parameter("ticker_interval_sec", 0.1)  # 10 Hz UDP socket poll timer
        self.declare_parameter("log_every_n_packets", 10)   # 0 for no log

        self.declare_parameter("image_topic", "camera/image_raw")
        self.declare_parameter("request_image_every_sec", 0.5)
        self.declare_parameter("jpeg_max_width", 320)
        self.declare_parameter("jpeg_max_height", 180)
        self.declare_parameter("jpeg_quality", 60)

        self.verbose = bool(self.get_parameter("verbose").value)
        cloud_topic = str(self.get_parameter("cloud_topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.color_patch_fraction = float(self.get_parameter("color_patch_fraction").value)
        self.use_mean_color = bool(self.get_parameter("use_mean_color").value)
        ticker_interval_sec = float(self.get_parameter("ticker_interval_sec").value)
        self.log_every_n_packets = int(self.get_parameter("log_every_n_packets").value)

        image_topic = str(self.get_parameter("image_topic").value)
        self.request_image_every_sec = float(self.get_parameter("request_image_every_sec").value)
        self.jpeg_max_width = int(self.get_parameter("jpeg_max_width").value)
        self.jpeg_max_height = int(self.get_parameter("jpeg_max_height").value)
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)

        self.capL, self.capR = CameraDriver.open_stereo_cameras()

        self.pointcloud_helper = PointCloudHelper(self.use_mean_color, self.color_patch_fraction, self.frame_id)

        self.br = CvBridge()
        self.tcp_sock = None

        self.image_pub = self.create_publisher(Image, image_topic, 10)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.pub_cloud = self.create_publisher(PointCloud2, cloud_topic, qos)

        self.last_seq = -1
        self.packet_counter = 0

        self.timer = self.create_timer(ticker_interval_sec, self.main_loop)
        self.image_timer = self.create_timer(
            self.request_image_every_sec,
            self.image_publish_callback,
        )

        self.get_logger().info(
            f"publishing Image on '{image_topic}', PointCloud2 on '{cloud_topic}'"
        )

    def destroy_node(self):
        try:
            self.timer.cancel()
        except Exception:
            pass

        try:
            self.image_timer.cancel()
        except Exception:
            pass

        super().destroy_node()


    def image_publish_callback(self):

            # TODO: get latest frame here
            frame = left

            msg = self.br.cv2_to_imgmsg(frame, encoding="bgr8")

            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.frame_id

            self.image_pub.publish(msg)

            with self.latest_image_lock:
                self.latest_image = frame.copy()
                self.latest_image_stamp_ns = stamp_ns

            if self.verbose:
                self.get_logger().info(
                    f"Published raw image from TCP server: seq={seq}, shape={frame.shape[1]}x{frame.shape[0]}"
                )


    def main_loop(self) -> None:

        latest_msg = None

        while True:
            try:
                seq, stamp_ns, rows, cols, points = None

                okL, left = self.capL.read()
                okR, right = self.capR.read()

                self.pointcloud_helper.latest_image = left

            except Exception as exc:
                self.get_logger().error(f"Camera capture error: {exc}")
                return



            latest_msg = self.build_pointcloud2(seq, stamp_ns, rows, cols, points)

            self.packet_counter += 1
            if self.log_every_n_packets > 0 and (self.packet_counter % self.log_every_n_packets == 0):
                self.get_logger().info(
                    f"points={len(points)} grid={rows}x{cols}"
                )

            if latest_msg is not None:
                self.pub_cloud.publish(latest_msg)


def main(args=None):
    rclpy.init(args=args)
    node = InferenceStereoNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
