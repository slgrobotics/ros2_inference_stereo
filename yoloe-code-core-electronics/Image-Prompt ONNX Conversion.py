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


from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
import numpy as np

# ========== CONFIGURATION ==========
# Add as many objects as you want:
training_data = [
    {
        "image": "image1.jpg", 
        "box": [134, 182, 550, 510],
    },
    {
        "image": "image2.jpg",
        "box": [158, 51, 346, 312],
    }
]
# ===================================

model = YOLOE("yoloe-11s-seg.pt")

# Collect all bboxes and class IDs
all_bboxes = []
all_class_ids = []

for i, data in enumerate(training_data):
    all_bboxes.append(data["box"])
    all_class_ids.append(i)  # Class ID will be 0, 1, 2, etc.

visual_prompts = {
    'bboxes': np.array(all_bboxes),
    'cls': np.array(all_class_ids)
}

# Train with all objects at once
model.predict(
    training_data[0]["image"],
    refer_image=training_data[0]["image"], 
    visual_prompts=visual_prompts,
    predictor=YOLOEVPSegPredictor,
    conf=0.1
)

model.export(format="onnx", imgsz=640)

print("Training complete!")
print("Object mapping:")
for i, data in enumerate(training_data):
    print(f"  ID {i}: {data['image']}")