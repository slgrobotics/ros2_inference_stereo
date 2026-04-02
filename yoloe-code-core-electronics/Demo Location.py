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

TARGET_OBJECT = "hand"  # What object to look for (e.g., "person", "bottle", "cup")
CONFIDENCE_THRESHOLD = 0.2  # Minimum confidence score (0.0 to 1.0)

picam2 = Picamera2()
picam2.preview_configuration.main.size = (800, 800)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

model = YOLO("yoloe-11l-seg.onnx")
class_names = model.names

print(f"Tracking location of: {TARGET_OBJECT}")
print(f"Minimum confidence: {CONFIDENCE_THRESHOLD}")
print("Press 'q' to quit")

while True:
    # Capture a frame from the camera
    frame = picam2.capture_array()
    frame_height, frame_width = frame.shape[:2]
    
    # Run YOLO model on the captured frame
    results = model.predict(frame)
    
    # Get object locations
    object_locations = []
    
    if results[0].boxes is not None:
        # Get bounding boxes, class IDs, and confidence scores
        boxes = results[0].boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2 format
        detected_classes = results[0].boxes.cls.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores
        
        # Process each detected object
        for i, class_id in enumerate(detected_classes):
            class_name = class_names[int(class_id)]
            confidence = confidences[i]
            
            # Check if this is our target object AND meets confidence threshold
            if class_name.lower() == TARGET_OBJECT.lower() and confidence >= CONFIDENCE_THRESHOLD:
                # Get bounding box coordinates
                x1, y1, x2, y2 = boxes[i]
                
                # Calculate center point
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Convert to relative coordinates (0.0 to 1.0)
                relative_x = center_x / frame_width
                relative_y = center_y / frame_height
                
                # Store the location with confidence
                object_locations.append({
                    'x': relative_x,
                    'y': relative_y,
                    'pixel_x': int(center_x),
                    'pixel_y': int(center_y),
                    'confidence': confidence
                })
    
    # ACTION TRIGGER - This is where you add your custom code
    for i, location in enumerate(object_locations):
        confidence = location['confidence']
        print(f"{TARGET_OBJECT} #{i+1} at relative position: ({location['x']:.3f}, {location['y']:.3f}) - Confidence: {confidence:.3f}")
        
        # Example: Detect object in bottom right corner with high confidence
        if location['x'] > 0.5 and location['y'] > 0.5:
            print(f"HIGH CONFIDENCE {TARGET_OBJECT} detected in top right corner! (Confidence: {confidence:.3f})")
            # ADD YOUR CUSTOM ACTION HERE
            # - Trigger an alarm
            # - Move camera to center on object
            # - Take a photo
            # - Send notification
            # - etc.
    
    # Create annotated frame with detection boxes
    annotated_frame = results[0].plot(boxes=True, masks=False)
    
    # Draw center points and coordinates on detected objects
    for location in object_locations:
        px, py = location['pixel_x'], location['pixel_y']
        rx, ry = location['x'], location['y']
        confidence = location['confidence']
        
        # Draw center point
        cv2.circle(annotated_frame, (px, py), 5, (0, 255, 255), -1)
        
        # Draw relative coordinates and confidence
        coord_text = f"({rx:.2f}, {ry:.2f}) {confidence:.2f}"
        cv2.putText(annotated_frame, coord_text, (px + 10, py - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    
    # Calculate and display FPS
    inference_time = results[0].speed['inference']
    fps = 1000 / inference_time
    
    # Display status
    status_text = f"Tracking: {TARGET_OBJECT} | Found: {len(object_locations)} | FPS: {fps:.1f} | Min Conf: {CONFIDENCE_THRESHOLD}"
    cv2.putText(annotated_frame, status_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Display the frame
    cv2.imshow("Object Location Tracker", annotated_frame)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()