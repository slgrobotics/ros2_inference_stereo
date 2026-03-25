#!/usr/bin/env python3

import time
from picamera2 import Picamera2

# 1. Initialize using plain integer indices
# On Pi 5, these correspond to the two CSI ports
try:
    print("Initializing Camera 0...")
    picam0 = Picamera2(0) 
    
    print("Initializing Camera 1...")
    picam1 = Picamera2(1)

    # 2. Configure for still capture
    picam0.configure("main")
    picam1.configure("main")

    # 3. Start the cameras
    print("Starting cameras...")
    picam0.start()
    picam1.start()

    # Wait for exposure to stabilize
    time.sleep(2)

    # 4. Capture images
    print("Capturing stereo pair...")
    picam0.capture_file("stereo_left.jpg")
    picam1.capture_file("stereo_right.jpg")

    # 5. Clean up
    picam0.stop()
    picam1.stop()
    picam0.close()
    picam1.close()
    print("Success! Check files: 'stereo_left.jpg' and 'stereo_right.jpg'")

except Exception as e:
    print(f"Failed to initialize cameras: {e}")

