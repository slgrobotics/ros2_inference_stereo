#!/usr/bin/env python3

# =========================================================================
#
# This ROS2 node captures images from dual cameras and computes stereo disparity information.
# It publishes the rectified left Image for visualization, CameraInfo and depth Image.
#
# Intended use:
#  input streams for RTAB-Map, SLAM, or other ROS2 processing nodes.
#
# See https://github.com/slgrobotics/articubot_one/wiki/Visual-SLAM-with-RTAB%E2%80%90Map
#
# =========================================================================

import time

import numpy as np

import cv2
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from sensor_msgs.msg import Image, CameraInfo

from config.config import Camera  # Important: review and adjust camera settings in config/config.py to match your hardware and calibration.

from ros2_inference_stereo.helpers import CameraInfoHelper, CameraDriver
from ros2_inference_stereo.helpers.disparity import (make_valid_disparity_mask, derive_sgbm_params, extract_sparse_points, build_depth_image)

class InferenceStereoNode(Node):
    def __init__(self) -> None:
        super().__init__("stereo_node")

        self.get_logger().info("stereo_node started")

        self.declare_parameter("verbose", False)
        self.declare_parameter("calibration_file", "config/calib_820x616.npz")
        self.declare_parameter("image_topic", "camera/image_raw")  # or "camera/image_raw/compressed"
        self.declare_parameter("camera_info_topic", "camera/camera_info")  # must be consistent with vis.launch and RViz2 config if you use RViz2 for visualization
        self.declare_parameter("depth_image_topic", "stereo/depth/image_rect_raw")  # Empty string disables depth image publishing
        self.declare_parameter("frame_id", "stereo_camera")
        self.declare_parameter("max_depth_range_m", 5.0) # cut-off range for detecting in depth image
        self.declare_parameter("close_cutout_factor", 1.0)
        self.declare_parameter("far_smoothing_factor", 1.0)
        self.declare_parameter("min_valid_disp", 1.0)
        self.declare_parameter("loop_delay_sec", 0.01)      # short "sleep" after detections processing to free CPU

        self.verbose = bool(self.get_parameter("verbose").value)
        calibration_file = str(self.get_parameter("calibration_file").value)
        image_topic = str(self.get_parameter("image_topic").value)
        camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        depth_image_topic = str(self.get_parameter("depth_image_topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.max_depth_range_m = float(self.get_parameter("max_depth_range_m").value)
        self.close_cutout_factor = float(self.get_parameter("close_cutout_factor").value)
        self.far_smoothing_factor = float(self.get_parameter("far_smoothing_factor").value)
        self.min_valid_disp = float(self.get_parameter("min_valid_disp").value)
        self.loop_delay_sec = float(self.get_parameter("loop_delay_sec").value)

        self.load_calibration(calibration_file)

        self.min_disp, self.num_disp, self.block_size = derive_sgbm_params(
            self.close_cutout_factor,
            self.far_smoothing_factor
        )

        self.get_logger().info(f"min_disp   : {self.min_disp}")
        self.get_logger().info(f"num_disp   : {self.num_disp}")
        self.get_logger().info(f"block_size : {self.block_size}")

        # By default, OpenCV operations within a Python script might run single-threaded.
        # Force OpenCV to utilize multiple CPU cores for its internal C++ math operations.
        cv2.setUseOptimized(True)
        cv2.setNumThreads(4) # Adjust based on how many CPU cores your platform has

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

        # Open cameras:
        self.capL, self.capR = CameraDriver.open_stereo_cameras()

        self.br = CvBridge()

        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.image_pub = self.create_publisher(Image, image_topic, image_qos)

        camera_info_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.camera_info_pub = self.create_publisher(CameraInfo, camera_info_topic, camera_info_qos)

        depth_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.depth_image_pub = self.create_publisher(Image, depth_image_topic, depth_qos)

        # groups are mutually exclusive, not reentrant, so the same callback never overlaps with itself.
        self.stereo_group = MutuallyExclusiveCallbackGroup()

        self.loop_timer = self.create_timer(
            self.loop_delay_sec,
            self.stereo_publish_callback,
            callback_group=self.stereo_group,
            autostart=False,
        )

        self.loop_timer.reset()

        enabled_outputs = []
        enabled_outputs.append(f"CameraInfo on '{camera_info_topic}'")
        enabled_outputs.append(f"raw Image on '{image_topic}'")
        enabled_outputs.append(f"depth Image on '{depth_image_topic}'")
        self.get_logger().info("Publishing: " + ", ".join(enabled_outputs))

        self.camera_info_counter = 0

    def destroy_node(self):
        try:
            self.loop_timer.cancel()
        except Exception:
            pass

        try:
            self.capL.release()
        except Exception:
            pass

        try:
            self.capR.release()
        except Exception:
            pass

        super().destroy_node()

    def load_calibration(self, calibration_file):
        # Load calibration NPZ:
        try:
            self.get_logger().info(
                f"Loading stereo calibration file: '{calibration_file}'"
            )
            calib = np.load(calibration_file)

            self.camera_info_helper = CameraInfoHelper(calib)

        except FileNotFoundError:
            raise RuntimeError(f"Calibration file '{calibration_file}' not found")

        self.mapLx = calib["mapLx"]
        self.mapLy = calib["mapLy"]
        self.mapRx = calib["mapRx"]
        self.mapRy = calib["mapRy"]
        self.Q = calib["Q"]
        #self.PL = calib["PL"]
        #self.T = calib["T"]
        
    def stereo_publish_callback(self) -> None:

        self.loop_timer.cancel()
        try:

            t0 = time.perf_counter()

            try:
                okL, left = self.capL.read()
                okR, right = self.capR.read()

                time_ros = self.get_clock().now().to_msg()

            except Exception as exc:
                self.get_logger().error(f"Camera capture error: {exc}")
                return

            if not okL or not okR:
                self.get_logger().error("bad camera read")
                return

            t1 = time.perf_counter()

            left_rect = cv2.remap(left, self.mapLx, self.mapLy, cv2.INTER_LINEAR)
            right_rect = cv2.remap(right, self.mapRx, self.mapRy, cv2.INTER_LINEAR)

            image_msg = self.br.cv2_to_imgmsg(left_rect, encoding="bgr8")

            # that's when the image was captured:
            image_msg.header.stamp = time_ros
            image_msg.header.frame_id = self.frame_id

            # compute disparity and 3D points:

            left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

            disparity = self.stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

            invalid_left_cols = self.num_disp
            invalid_right_cols = self.block_size // 2

            valid_mask = make_valid_disparity_mask(
                disparity,
                min_valid_disp=self.min_valid_disp,
                invalid_left_cols=invalid_left_cols,
                invalid_right_cols=invalid_right_cols,
            )

            points_3d = cv2.reprojectImageTo3D(disparity, self.Q, handleMissingValues=False)

            t2 = time.perf_counter()

            depth_image = build_depth_image(
                disparity,
                points_3d,
                valid_mask,
                max_range_m=self.max_depth_range_m,
            )

            t3 = time.perf_counter()

            dt_capture_ms = (t1 - t0) * 1000.0
            dt_projecting_ms = (t2 - t1) * 1000.0
            dt_depth_calc_ms = (t3 - t2) * 1000.0

            if self.verbose:
                self.get_logger().info(f"Capture time: {dt_capture_ms:.2f} ms, Projecting time: {dt_projecting_ms:.2f} ms, Depth calc time: {dt_depth_calc_ms:.2f} ms")

            self.camera_info_counter += 1

            # publish CameraInfo every 5 frames, and also on the first 5 frames to ensure subscribers get it quickly
            if self.camera_info_counter < 5 or self.camera_info_counter % 5 == 0:
                img_h, img_w = left.shape[:2]
                cam_info = self.camera_info_helper.build_scaled_camera_info(
                    img_w,
                    img_h,
                    image_msg.header.frame_id,
                    image_msg.header.stamp,
                )
                self.camera_info_pub.publish(cam_info)

            self.image_pub.publish(image_msg)

            if depth_image is not None:
                depth_msg = self.br.cv2_to_imgmsg(depth_image.astype(np.float32), encoding="32FC1")
                depth_msg.header.stamp = time_ros
                depth_msg.header.frame_id = self.frame_id
                self.depth_image_pub.publish(depth_msg)

        finally:
            self.loop_timer.reset()


def main(args=None):
    rclpy.init(args=args)
    node = InferenceStereoNode()
    executor = MultiThreadedExecutor(num_threads=2)

    try:
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()

