#!/usr/bin/env python3

import time
import os
import sys

from typing import Tuple

import cv2
import numpy as np
from picamera2 import Picamera2

# Add ../config to Python path so we can import config.py
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "config"))
sys.path.append(CONFIG_DIR)

from config import Camera


class Picamera2Capture:
    def __init__(self, picam: Picamera2, sensor_id: int):
        self.picam = picam
        self.sensor_id = sensor_id
        self.started = False

    def start(self) -> None:
        if not self.started:
            self.picam.start()
            self.started = True

    def read(self):
        """
        OpenCV-like API:
        returns (ok, frame_bgr)
        """
        try:
            frame = self.picam.capture_array()
            if frame is None:
                return False, None

            # Picamera2 often returns RGB; convert to BGR for OpenCV compatibility.
            if frame.ndim == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif frame.ndim == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

            return True, frame
        except Exception:
            return False, None

    def grab(self) -> bool:
        """
        Approximate OpenCV grab(): capture and discard one frame.
        """
        ok, _ = self.read()
        return ok

    def release(self) -> None:
        try:
            if self.started:
                self.picam.stop()
                self.started = False
        finally:
            try:
                self.picam.close()
            except Exception:
                pass


class CameraDriver:
    @staticmethod
    def open_camera(sensor_id: int, width: int, height: int, fps: int):
        picam = Picamera2(sensor_id)

        config = picam.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"},
            controls={"FrameDurationLimits": (int(1e6 / fps), int(1e6 / fps))},
        )

        picam.configure(config)

        cap = Picamera2Capture(picam, sensor_id)
        cap.start()
        return cap

    @staticmethod
    def open_stereo_cameras(
        width: int = Camera.WIDTH,
        height: int = Camera.HEIGHT,
        fps: int = Camera.FPS,
        left_id: int = Camera.LEFT,
        right_id: int = Camera.RIGHT,
        startup_delay_sec: float = 2.0,
    ) -> Tuple[Picamera2Capture, Picamera2Capture]:
        cap_l = CameraDriver.open_camera(left_id, width, height, fps)
        cap_r = CameraDriver.open_camera(right_id, width, height, fps)

        # Let auto exposure / white balance settle.
        time.sleep(startup_delay_sec)

        return cap_l, cap_r
    