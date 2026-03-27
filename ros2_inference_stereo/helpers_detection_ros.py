#!/usr/bin/env python3

"""
Helpers for converting model detections into ROS2 Detection2DArray messages.
"""

from typing import Iterable, Optional, Set, List

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

from ultralytics import DetectionResult

class DetectionRosHelper:
    def __init__(
        self,
        frame_id: str,
        min_confidence: float = 0.25,
        allowed_labels: Optional[Iterable[str]] = None,
    ):
        self.frame_id = frame_id
        self.min_confidence = float(min_confidence)
        self.allowed_labels: Set[str] = {
            s.strip().lower() for s in (allowed_labels or []) if str(s).strip()
        }

    def set_allowed_labels(self, allowed_labels: Optional[Iterable[str]]) -> None:
        self.allowed_labels = {
            s.strip().lower() for s in (allowed_labels or []) if str(s).strip()
        }

    def _build_header(self, stamp, source_frame_id: Optional[str] = None) -> Header:
        header = Header()
        header.stamp = stamp
        header.frame_id = source_frame_id or self.frame_id
        return header

    def _passes_filters(self, det) -> bool:
        if float(det.confidence) < self.min_confidence:
            return False

        if self.allowed_labels and det.label.lower() not in self.allowed_labels:
            return False

        return True

    def _build_detection_msg(self, det, header: Header) -> Detection2D:
        cx, cy, w, h = det.bbox_xywh

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
        hypothesis.hypothesis.class_id = det.label
        hypothesis.hypothesis.score = float(det.confidence)

        detection.results.append(hypothesis)
        return detection

    def build_detection_array_msg(
        self,
        detections: List[DetectionResult],
        stamp,
        source_frame_id: Optional[str] = None,
    ) -> Detection2DArray:
        """
        Convert a list of DetectionResult-like objects into Detection2DArray.

        Expected detection object fields:
          - label: str
          - confidence: float
          - bbox_xywh: (cx, cy, w, h)
        """
        header = self._build_header(stamp, source_frame_id)

        msg = Detection2DArray()
        msg.header = header

        for det in detections:
            if not self._passes_filters(det):
                continue

            detection_msg = self._build_detection_msg(det, header)
            msg.detections.append(detection_msg)

        return msg
