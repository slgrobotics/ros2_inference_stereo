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
from ultralytics import YOLO

# Set up the camera with Picam
picam2 = Picamera2()
picam2.preview_configuration.main.size = (800, 800)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# Load YOLOE prompt-free model
model = YOLO("yoloe-11s-seg-pf.pt")

while True:
    # Capture a frame from the camera
    frame = picam2.capture_array()
    
    # Run YOLOE model on the captured fram
    results = model.predict(frame)
    
    # Output the visual detection data
    annotated_frame = results[0].plot(boxes=True, masks=False)
    
    # Get inference time
    inference_time = results[0].speed['inference']
    fps = 1000 / inference_time  # Convert to milliseconds
    text = f'FPS: {fps:.1f}'
    
    # Define font and position
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    text_x = annotated_frame.shape[1] - text_size[0] - 10  # 10 pixels from the right
    text_y = text_size[1] + 10  # 10 pixels from the top
    
    # Draw the text on the annotated frame
    cv2.putText(annotated_frame, text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Display the resulting frame
    cv2.imshow("Camera", annotated_frame)
    
    # Exit the program if q is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# Close all windows
cv2.destroyAllWindows()
