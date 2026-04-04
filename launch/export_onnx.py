#!/usr/bin/env python3

# =============================================================================================
#
# see https://github.com/slgrobotics/ros2_inference_stereo/blob/main/README.md#promptable-models
# run this script where your models are - ~/robot_ws/models or ~/launch/models
#
# This script converts a YOLOE segmentation model into an ONNX model with a fixed, prompt-defined vocabulary.
#
# It loads `yoloe-11s-seg.pt`, encodes the object prompts listed in `CLASS_NAMES`, binds them to the model,
#  and exports the result as an ONNX file for `640x832` input images.
#
# The goal is to create a deployment-ready model tailored to a specific set of objects and animals.
#
# Derived from https://github.com/slgrobotics/ros2_inference_stereo/blob/main/yoloe-code-core-electronics/Text-Prompt%20ONNX%20Conversion.py
#
# =============================================================================================

from pathlib import Path
from ultralytics import YOLOE

MODEL_PATH = "yoloe-11s-seg.pt"
CLASS_NAMES = [
    "person",
    "blue cup",
    "orange cup",
    "white cup",
    "cup",
    "yellow ball",
    "blue ball",
    "red ball",
    "ball",
    "human hand",
    "tv",
    "mobile phone",
    "animal",
    "tiger",
    "squirrel",
    "a raccoon with a black mask face",
    "a small domestic cat",
    "a medium sized dog",
]
IMG_SIZE = (640, 832)  # (height, width), both divisible by 32


def main() -> None:
    model_path = Path(MODEL_PATH).expanduser()

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"...loading model: {model_path}")

    # Load model
    model = YOLOE(str(model_path))

    # Build and set prompt embeddings
    text_pe = model.get_text_pe(CLASS_NAMES)
    model.set_classes(CLASS_NAMES, text_pe)

    # Export ONNX with fixed input size
    model.export(
        format="onnx",
        imgsz=IMG_SIZE,
    )

    print(f"Export complete: {model_path.with_suffix('.onnx')}")


if __name__ == "__main__":
    main()
