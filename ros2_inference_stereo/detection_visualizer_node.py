# Copyright 2019 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#
# Note: this is a verbatim copy of https://github.com/ros2/detection_visualizer/blob/master/detection_visualizer/__init__.py
#       with minor local modifications (mostly logging).
#       All credits and original copyright belong to the original authors.

import sys
import math

import cv2
import cv_bridge
import message_filters
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray


import sys
import math

import cv2
import cv_bridge
import message_filters
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from vision_msgs.msg import Detection2DArray


class DetectionVisualizerNode(Node):

    def __init__(self):
        super().__init__('detection_visualizer')

        self.declare_parameter("verbose", False)
        self.declare_parameter("image_topic", "camera/image_raw")  # or "camera/image_raw/compressed"
        self.declare_parameter("detection_topic", "image_inference_detections")
        self.declare_parameter("overlay_image_topic", "image_inference_overlay")
        self.declare_parameter("time_slop", 0.01)

        self.verbose = bool(self.get_parameter("verbose").value)
        self.image_topic = str(self.get_parameter("image_topic").value)
        self.detection_topic = str(self.get_parameter("detection_topic").value)
        self.overlay_image_topic = str(self.get_parameter("overlay_image_topic").value)
        self.time_slop = float(self.get_parameter("time_slop").value)

        self._bridge = cv_bridge.CvBridge()
        self._use_compressed = self.image_topic.endswith("/compressed")

        output_image_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            depth=1)

        self._image_pub = self.create_publisher(
            Image, self.overlay_image_topic, output_image_qos
        )

        # The two incoming messages on a single callback require synchronization.
        # "self.time_slop" defines tolerance to header timestamps:
        image_msg_type = CompressedImage if self._use_compressed else Image
        self._image_sub = message_filters.Subscriber(self, image_msg_type, self.image_topic)
        self._detections_sub = message_filters.Subscriber(self, Detection2DArray, self.detection_topic)

        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            (self._image_sub, self._detections_sub), queue_size=5, slop=self.time_slop)

        self._synchronizer.registerCallback(self.on_detections)

        image_mode = "CompressedImage" if self._use_compressed else "Image"
        self.get_logger().info(
            f"detection_visualizer started: '{self.image_topic}' ({image_mode}) + "
            f"'{self.detection_topic}' --> '{self.overlay_image_topic}' "
            f"time_slop: {self.time_slop}  verbose: {self.verbose}"
        )

    def _to_cv_image(self, image_msg):
        if self._use_compressed:
            return self._bridge.compressed_imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        return self._bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

    def _output_encoding(self, image_msg):
        if self._use_compressed:
            return 'bgr8'
        return image_msg.encoding if image_msg.encoding else 'bgr8'

    def on_detections(self, image_msg, detections_msg):
        cv_image = self._to_cv_image(image_msg)

        if self.verbose:
            self.get_logger().info(f"Received: {len(detections_msg.detections)} detections")

        # Draw boxes on image
        for detection in detections_msg.detections:
            max_class = None
            max_score = 0.0
            for result in detection.results:
                hypothesis = result.hypothesis
                if hypothesis.score > max_score:
                    max_score = hypothesis.score
                    max_class = hypothesis.class_id
            if max_class is None:
                if self.verbose:
                    self.get_logger().warning("Failed to find class with highest score")
                continue

            if self.verbose:
                self.get_logger().info(f"IP: processing class_id={max_class}  score={max_score}")

            cx = detection.bbox.center.position.x
            cy = detection.bbox.center.position.y
            sx = detection.bbox.size_x
            sy = detection.bbox.size_y

            min_pt = (round(cx - sx / 2.0), round(cy - sy / 2.0))
            max_pt = (round(cx + sx / 2.0), round(cy + sy / 2.0))
            color = (0, 255, 0)
            thickness = 1

            if detection.bbox.center.theta == 0.0:
                cv2.rectangle(cv_image, min_pt, max_pt, color, thickness)
            else:
                rotation = -math.degrees(detection.bbox.center.theta)
                box = cv2.boxPoints(((cx, cy), (sx, sy), rotation))
                box = box.astype(int)
                cv2.drawContours(cv_image, [box], 0, color, thickness)

            label = f"{max_class} {max_score:.3f}"

            x = max(0, min_pt[0] + 5)
            y = max(20, max_pt[1] - 5)
            pos = (x, y)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(cv_image, label, pos, font, 0.75, color, 1, cv2.LINE_AA)

        detection_image_msg = self._bridge.cv2_to_imgmsg(
            cv_image, encoding=self._output_encoding(image_msg)
        )
        detection_image_msg.header = image_msg.header

        if self.verbose:
            self.get_logger().info("IP: Publishing detection_image_msg")

        self._image_pub.publish(detection_image_msg)


def main(args=None):
    rclpy.init(args=args)
    node = DetectionVisualizerNode()
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
