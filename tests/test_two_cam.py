#!/usr/bin/env python3

import os
import sys
import time

from picamera2 import Picamera2

# Add ../config to Python path so we can import config.py
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "config"))
sys.path.append(CONFIG_DIR)

from config import Camera


def main():
    picam0 = None
    picam1 = None

    try:
        print(f"Initializing Camera {Camera.LEFT}...")
        picam0 = Picamera2(Camera.LEFT)

        print(f"Initializing Camera {Camera.RIGHT}...")
        picam1 = Picamera2(Camera.RIGHT)

        print(
            f"Configuring still capture: "
            f"{Camera.WIDTH}x{Camera.HEIGHT} @ {Camera.FPS} fps request"
        )

        config0 = picam0.create_still_configuration(
            main={"size": (Camera.WIDTH, Camera.HEIGHT), "format": "RGB888"},
            controls={"FrameDurationLimits": (int(1e6 / Camera.FPS), int(1e6 / Camera.FPS))},
        )
        config1 = picam1.create_still_configuration(
            main={"size": (Camera.WIDTH, Camera.HEIGHT), "format": "RGB888"},
            controls={"FrameDurationLimits": (int(1e6 / Camera.FPS), int(1e6 / Camera.FPS))},
        )

        picam0.configure(config0)
        picam1.configure(config1)

        print("Starting cameras...")
        picam0.start()
        picam1.start()

        print("Waiting for auto-exposure / auto-white-balance to settle...")
        time.sleep(2.0)

        print("Capturing stereo pair...")
        picam0.capture_file("stereo_left.jpg")
        picam1.capture_file("stereo_right.jpg")

        print("Success! Check files: 'stereo_left.jpg' and 'stereo_right.jpg'")

    except Exception as e:
        print(f"Failed to initialize or capture from cameras: {e}")

    finally:
        for cam in (picam0, picam1):
            if cam is not None:
                try:
                    cam.stop()
                except Exception:
                    pass
                try:
                    cam.close()
                except Exception:
                    pass


if __name__ == "__main__":
    main()
