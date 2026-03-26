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
import time

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

from config.config import Stereo, Streamer
from ros2_inference_stereo.helper_picamera import CameraDriver, Picamera2Capture
from ros2_inference_stereo.helpers_pointcloud import PointCloudHelper
from ros2_inference_stereo.helpers_disparity import make_valid_disparity_mask, derive_sgbm_params, estimate_depth_cm_from_disparity, overlay_cell_distances, cam_to_ros, extract_sparse_points


class InferenceStereoNode(Node):
    def __init__(self) -> None:
        super().__init__("inference_stereo_node")

        self.declare_parameter("verbose", False)
        self.declare_parameter("calibration_file", "config/calib_820x616.npz")
        self.declare_parameter("cloud_topic", "stereo/sparse_cloud")
        self.declare_parameter("frame_id", "stereo_camera")
        self.declare_parameter("grid_size", 16)  # Grid size NxN for sparse sampling
        self.declare_parameter("close_cutout_factor", 1.0)
        self.declare_parameter("far_smoothing_factor", 1.0)
        self.declare_parameter("color_patch_fraction", 0.5)   # center patch size relative to cell
        self.declare_parameter("use_mean_color", True)
        self.declare_parameter("min_confidence", 0.02)
        self.declare_parameter("ticker_interval_sec", 0.1)  # 10 Hz UDP socket poll timer
        self.declare_parameter("log_every_n_packets", 10)   # 0 for no log

        self.declare_parameter("image_topic", "camera/image_raw")
        self.declare_parameter("request_image_every_sec", 0.5)
        self.declare_parameter("jpeg_max_width", 320)
        self.declare_parameter("jpeg_max_height", 180)
        self.declare_parameter("jpeg_quality", 60)

        self.verbose = bool(self.get_parameter("verbose").value)
        self.calibration_file = str(self.get_parameter("calibration_file").value)
        cloud_topic = str(self.get_parameter("cloud_topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.grid_size = int(self.get_parameter("grid_size").value)
        self.close_cutout_factor = float(self.get_parameter("close_cutout_factor").value)
        self.far_smoothing_factor = float(self.get_parameter("far_smoothing_factor").value)
        self.color_patch_fraction = float(self.get_parameter("color_patch_fraction").value)
        self.use_mean_color = bool(self.get_parameter("use_mean_color").value)
        self.min_confidence = float(self.get_parameter("min_confidence").value)
        ticker_interval_sec = float(self.get_parameter("ticker_interval_sec").value)
        self.log_every_n_packets = int(self.get_parameter("log_every_n_packets").value)

        image_topic = str(self.get_parameter("image_topic").value)
        self.request_image_every_sec = float(self.get_parameter("request_image_every_sec").value)
        self.jpeg_max_width = int(self.get_parameter("jpeg_max_width").value)
        self.jpeg_max_height = int(self.get_parameter("jpeg_max_height").value)
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)

        self.pointcloud_helper = PointCloudHelper(self.use_mean_color, self.color_patch_fraction, self.frame_id)

        # Load calibration NPZ:
        try:
            self.get_logger().info(
                f"Loading stereo calibration file: '{self.calibration_file}'"
            )
            calib = np.load(self.calibration_file)
        except FileNotFoundError:
            raise RuntimeError(f"Calibration file '{self.calibration_file}' not found")

        self.mapLx = calib["mapLx"]
        self.mapLy = calib["mapLy"]
        self.mapRx = calib["mapRx"]
        self.mapRy = calib["mapRy"]
        self.Q = calib["Q"]
        self.PL = calib["PL"]
        self.T = calib["T"]

        self.width = int(calib["image_width"])
        self.height = int(calib["image_height"])

        self.focal_px = float(self.PL[0, 0])
        self.baseline_m = float(np.linalg.norm(self.T))

        # min_disp = minimum disparity the matcher will search
        # the algorithm searches disparities in: [min_disp, min_disp + num_disp]
        #min_disp = 1    # min_disp = 0: full range, includes far; min_disp > 0: ignore far, focus near
        #block_size = 9  # matching window size (odd number); larger = smoother, less detail

        # smaller num_disp means the nearest measurable depth moves farther away
        # larger num_disp means the matcher can represent closer objects
        #num_disp = 16 * 6  # closest objects cutoff at 0.9 meters
        #num_disp = 16 * 8  # closest objects cutoff at 0.5 meters

        self.min_disp, self.num_disp, self.block_size = derive_sgbm_params(
            self.close_cutout_factor,
            self.far_smoothing_factor
        )

        self.get_logger().info(f"min_disp   : {self.min_disp}")
        self.get_logger().info(f"num_disp   : {self.num_disp}")
        self.get_logger().info(f"block_size : {self.block_size}")

        self.stereo = cv2.StereoSGBM_create(
            minDisparity=self.min_disp,
            numDisparities=self.num_disp,
            blockSize=self.block_size,
            P1=8 * 1 * self.block_size * self.block_size,
            P2=32 * 1 * self.block_size * self.block_size,
            disp12MaxDiff=1,
            uniquenessRatio=5,
            speckleWindowSize=50,
            speckleRange=2,
            preFilterCap=31,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

        self.grid_rows = self.grid_cols = self.grid_size

        # Open cameras:
        self.capL, self.capR = CameraDriver.open_stereo_cameras()

        self.br = CvBridge()
        self.tcp_sock = None

        self.image_pub = self.create_publisher(Image, image_topic, 10)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.pub_cloud = self.create_publisher(PointCloud2, cloud_topic, qos)

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

            # get latest frame:
            frame, img_stamp_ns = self.pointcloud_helper.get_latest_image_copy_with_stamp()

            msg = self.br.cv2_to_imgmsg(frame, encoding="bgr8")

            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.frame_id

            self.image_pub.publish(msg)

            with self.latest_image_lock:
                self.latest_image = frame.copy()
                self.latest_image_stamp_ns = img_stamp_ns

            if self.verbose:
                self.get_logger().info(
                    f"Published raw image from TCP server: seq={self.packet_counter}, shape={frame.shape[1]}x{frame.shape[0]}"
                )


    def main_loop(self) -> None:

        latest_msg = None
        now = time.time()
        stamp_ns = int(now * 1e9)
        last_time = now
        fps_filtered = 0.0

        while True:

            points = None

            try:
                okL, left = self.capL.read()
                okR, right = self.capR.read()

                now = time.time()
                stamp_ns = int(now * 1e9)

                self.pointcloud_helper.update_latest_image(left, stamp_ns)

            except Exception as exc:
                self.get_logger().error(f"Camera capture error: {exc}")
                return

            if not okL or not okR:
                self.get_logger().error("bad camera read")
                continue

            left_rect = cv2.remap(left, self.mapLx, self.mapLy, cv2.INTER_LINEAR)
            right_rect = cv2.remap(right, self.mapRx, self.mapRy, cv2.INTER_LINEAR)

            left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

            disparity = self.stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

            invalid_left_cols = self.num_disp
            invalid_right_cols = self.block_size // 2

            valid_mask = make_valid_disparity_mask(
                disparity,
                min_valid_disp=Stereo.MIN_VALID_DISP,
                invalid_left_cols=invalid_left_cols,
                invalid_right_cols=invalid_right_cols,
            )

            points_3d = cv2.reprojectImageTo3D(disparity, self.Q, handleMissingValues=False)

            points = extract_sparse_points(
                disparity,
                points_3d,
                valid_mask=valid_mask,
                rows=self.grid_rows,
                cols=self.grid_cols,
                max_range_m=Streamer.MAX_RANGE_M,
                min_confidence=self.min_confidence,
            )

            # packet = pack_packet(seq, grid_rows, grid_cols, points, timestamp_ns)
            # sock.sendto(packet, (udp_ip, udp_port))
            # frame_buffer.update(left_rect, seq, timestamp_ns)

            dt = now - last_time
            last_time = now
            fps_now = 1.0 / dt if dt > 0 else 0.0
            fps_filtered = 0.9 * fps_filtered + 0.1 * fps_now if fps_filtered > 0 else fps_now

            latest_msg = self.pointcloud_helper.build_pointcloud2(self.packet_counter, stamp_ns, self.grid_rows, self.grid_cols, points)

            self.packet_counter += 1
            if self.log_every_n_packets > 0 and (self.packet_counter % self.log_every_n_packets == 0):
                self.get_logger().info(
                    f"seq={self.packet_counter}  grid={self.grid_rows}x{self.grid_cols}  num_points={len(points)}  fps={fps_filtered:.2f}"
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
