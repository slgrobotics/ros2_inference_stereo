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

TARGET_OBJECT = "hand"        # What object to look for (e.g., "person", "bottle", "cup")
TARGET_COUNT = 1              # How many of that object should trigger the action
CONFIDENCE_THRESHOLD = 0.2    # Minimum confidence score (0.0 to 1.0)

picam2 = Picamera2()
picam2.preview_configuration.main.size = (800, 800)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

model = YOLO("yoloe-11s-seg.onnx")
# Get the class names that the model can detect
class_names = model.names  # Dictionary mapping class IDs to names

print(f"Looking for {TARGET_COUNT} {TARGET_OBJECT}(s)")
print(f"Minimum confidence: {CONFIDENCE_THRESHOLD}")
print("Press 'q' to quit")

while True:
    # Capture a frame from the camera
    frame = picam2.capture_array()
    
    # Run YOLO model on the captured frame
    results = model.predict(frame)
    
    # Count objects of the target type
    object_count = 0
    confident_objects = []  # Store info about confident detections
    
    if results[0].boxes is not None:  # Check if any objects were detected
        # Get the detected class IDs and confidence scores
        detected_classes = results[0].boxes.cls.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores
        
        # Count how many match our target object AND meet confidence threshold
        for i, class_id in enumerate(detected_classes):
            class_name = class_names[int(class_id)]
            confidence = confidences[i]
            
            if class_name.lower() == TARGET_OBJECT.lower() and confidence >= CONFIDENCE_THRESHOLD:
                object_count += 1
                confident_objects.append({
                    'class_name': class_name,
                    'confidence': confidence
                })
    
    # ACTION TRIGGER - This is where you add your custom code
    if object_count >= TARGET_COUNT:
        print(f"Target number of objects detected! ({object_count} confident {TARGET_OBJECT}(s))")
        for i, obj in enumerate(confident_objects):
            print(f"  {TARGET_OBJECT} #{i+1}: {obj['confidence']:.3f} confidence")
        # ADD YOUR CUSTOM ACTION HERE
        # Examples:
        # - Send a notification
        # - Control a servo motor
        # - Log data to a file
        # - Trigger an LED
        # - etc.
    
    # Create annotated frame with detection boxes
    annotated_frame = results[0].plot(boxes=True, masks=False)
    
    # Calculate and display FPS
    inference_time = results[0].speed['inference']
    fps = 1000 / inference_time
    
    # Display object count and status
    status_text = f"Looking for: {TARGET_COUNT} {TARGET_OBJECT}(s) | Found: {object_count} | FPS: {fps:.1f} | Min Conf: {CONFIDENCE_THRESHOLD}"
    
    # Add status text to frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(annotated_frame, status_text, (10, 30), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Highlight when target is reached
    if object_count >= TARGET_COUNT:
        cv2.putText(annotated_frame, "TARGET REACHED!", (10, 70), font, 1, (0, 255, 0), 3, cv2.LINE_AA)
        
        # Show confidence scores of detected objects
        for i, obj in enumerate(confident_objects):
            conf_text = f"{TARGET_OBJECT} #{i+1}: {obj['confidence']:.2f}"
            cv2.putText(annotated_frame, conf_text, (10, 110 + i*30), font, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    
    # Display the frame
    cv2.imshow("Object Counter", annotated_frame)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()