#!/usr/bin/env python3

# =======================================================================================
#
# Note: to use full FOV use 1640x1232  (or full IMX219 res 3280x2464)
#       run print_sensor_modes.py and look for modes with "crop_limits: (0, 0, 3280, 2464)"
#
# =======================================================================================

from picamera2 import Picamera2

picam = Picamera2(0)

print("\n=== Available sensor modes ===\n")

for i, mode in enumerate(picam.sensor_modes):
    print(f"[{i}]")
    for k, v in mode.items():
        print(f"  {k}: {v}")
    print()

picam.close()
