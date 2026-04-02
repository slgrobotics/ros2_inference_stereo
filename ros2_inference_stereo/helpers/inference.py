#!/usr/bin/env python3

from dataclasses import dataclass
from typing import List, Tuple

import math
import cv2
from ultralytics import YOLO


@dataclass
class DetectionResult:
    class_id: int
    label: str
    confidence: float
    bbox_xyxy: Tuple[float, float, float, float]
    bbox_xywh: Tuple[float, float, float, float]


class ObjectDetector:
    def __init__(
        self,
        model_path: str,
        imgsz: Tuple[int, int] = (640, 832),  # (height, width) both must be divisible by 32
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cpu",
    ):
        self.model_path = model_path
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device

        self.model = YOLO(model_path)


    def infer(self, frame_bgr) -> List[DetectionResult]:
        results = self.model.predict(
            source=frame_bgr,
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        detections: List[DetectionResult] = []

        if not results:
            return detections

        result = results[0]
        names = result.names
        boxes = result.boxes

        if boxes is None:
            return detections

        for box in boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            cx, cy, w, h = [float(v) for v in box.xywh[0].tolist()]
            label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)

            detections.append(
                DetectionResult(
                    class_id=cls_id,
                    label=label,
                    confidence=conf,
                    bbox_xyxy=(x1, y1, x2, y2),
                    bbox_xywh=(cx, cy, w, h),
                )
            )

        return detections

    @staticmethod
    def compute_detection_size(height, width, stride=32) -> Tuple[int, int]:  # (height, width)
        new_h = int(math.ceil(height / stride) * stride)
        new_w = int(math.ceil(width / stride) * stride)
        return (new_h, new_w)


    def draw_detections(self, frame_bgr, detections: List[DetectionResult]):
        out = frame_bgr.copy()

        for det in detections:
            x1, y1, x2, y2 = [int(round(v)) for v in det.bbox_xyxy]
            text = f"{det.label} {det.confidence:.2f}"

            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                out,
                text,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        return out
