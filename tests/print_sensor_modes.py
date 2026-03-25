#!/usr/bin/env python3

from picamera2 import Picamera2

picam = Picamera2(0)

print("\n=== Available sensor modes ===\n")

for i, mode in enumerate(picam.sensor_modes):
    print(f"[{i}]")
    for k, v in mode.items():
        print(f"  {k}: {v}")
    print()

picam.close()
