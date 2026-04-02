#!/usr/bin/env python3

# =============================================================================================
#
# This code was authored by Jaryd, Core Electronics Ltd, Australia and is included here for your convenience.
# All copy rights and credits belong to the original author(s).
# This code has not been modified, verified or tested by us (slgrobotics), and may not work as expected.
# There is no warranty or support for this code, and we are not liable for any issues that may arise from using it.
# Please refer to the original source for the most up-to-date version.
#
# See https://youtu.be/yNPwsKa52zs
#     https://core-electronics.com.au/guides/raspberry-pi/custom-object-detection-models-without-training-yoloe-and-raspberry-pi/
#
# =============================================================================================


import cv2
from picamera2 import Picamera2
import time

# ========== CONFIGURATION ==========
# Set the output file name heres
filename = f"image.jpg"

# Set up the camera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (800, 800)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

print("Reference Photo Capture Tool")
print("Position your object in the frame and press SPACE to capture")
print("Press ESC to exit")
print()

while True:
    # Capture frame
    frame = picam2.capture_array()
    
    # Add instructions on screen
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "SPACE: Take Photo | ESC: Exit", (10, 30), 
                font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Display the frame
    cv2.imshow("Capture Reference Photo", frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    # Capture photo on spacebar
    if key == ord(' '):
        # Save photo
        cv2.imwrite(filename, frame)
        print(f"✅ Photo saved: {filename}")
        
        # Show confirmation
        confirm_frame = frame.copy()
        cv2.putText(confirm_frame, f"Saved: {filename}", (10, 70), 
                    font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Capture Reference Photo", confirm_frame)
        cv2.waitKey(2000)
    
    # Exit on ESC
    elif key == 27:
        break

cv2.destroyAllWindows()