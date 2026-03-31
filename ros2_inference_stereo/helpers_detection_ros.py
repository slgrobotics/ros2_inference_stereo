"""
Helpers for converting model detections into ROS2 Detection2DArray messages.
"""

from typing import Iterable, Optional, Set

from std_msgs.msg import Header
from vision_msgs.msg import (
    Point2D,
    Pose2D,
    BoundingBox2D,
    Detection2D,
    Detection2DArray,
    ObjectHypothesis,
    ObjectHypothesisWithPose,
)

from ros2_inference_stereo.helpers_inference import DetectionResult


class DetectionRosHelper:
    def __init__(
        self,
        min_confidence: float = 0.25,
        allowed_labels: Optional[Iterable[str]] = None,
    ):
        self.min_confidence = float(min_confidence)
        self.allowed_labels: Set[str] = {
            str(s).strip().lower()
            for s in (allowed_labels or [])
            if str(s).strip()
        }

    def set_allowed_labels(self, allowed_labels: Optional[Iterable[str]]) -> None:
        self.allowed_labels = {
            str(s).strip().lower()
            for s in (allowed_labels or [])
            if str(s).strip()
        }

    def _passes_filters(self, det: DetectionResult) -> bool:
        if float(det.confidence) < self.min_confidence:
            return False

        if self.allowed_labels and det.label.lower() not in self.allowed_labels:
            return False

        return True

    def build_detection_array_msg(
        self,
        detections: Iterable[DetectionResult],
        header: Header,
    ) -> Detection2DArray:

        msg = Detection2DArray()

        for det in detections:
            if not self._passes_filters(det):
                continue

            #x1, y1, x2, y2 = det.bbox_xyxy
            cx, cy, w, h = det.bbox_xywh

            # if self.verbose:
            #     self.get_logger().info(
            #         f"Publishing detection: label={det.label}, confidence={confidence:.3f}, "
            #         f"bbox_xyxy=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}), "
            #         f"bbox_xywh=({cx:.0f}, {cy:.0f}, {w:.0f}, {h:.0f})"
            #     )

            detection = Detection2D()
            detection.header = header

            center = Pose2D()
            center.position = Point2D(x=float(cx), y=float(cy))
            center.theta = 0.0

            bbox = BoundingBox2D()
            bbox.center = center
            bbox.size_x = float(w)
            bbox.size_y = float(h)
            detection.bbox = bbox

            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis = ObjectHypothesis()
            hypothesis.hypothesis.class_id = det.label  # str(det.class_id)   "label" is easier for downstream use than "class_id", and we have it available
            hypothesis.hypothesis.score = float(det.confidence)

            detection.results.append(hypothesis)
            msg.detections.append(detection)

        # Normally we want to publish message with empty detections, so downstream nodes can get
        # the updated header and know that inference was performed on a new frame, even if nothing
        # was detected with confidence above the threshold.
        # If you really want to skip publishing in case of no detections, uncomment the following:
        #
        # if len(msg.detections) == 0:
        #     return None

        msg.header = header
        return msg
