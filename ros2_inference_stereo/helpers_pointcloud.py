
import threading

import struct
from typing import List, Tuple
import numpy as np

from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2

class PointCloudHelper:

    def __init__(self, use_mean_color, color_patch_fraction, frame_id):

        self.use_mean_color = use_mean_color
        self.color_patch_fraction = color_patch_fraction
        self.frame_id = frame_id

        self.latest_image = None
        self.latest_image_stamp_ns = 0
        self.latest_image_lock = threading.Lock()

        self.fields = [
            PointField(name="x", offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name="confidence", offset=16, datatype=PointField.FLOAT32, count=1),
            PointField(name="row", offset=20, datatype=PointField.UINT16, count=1),
            PointField(name="col", offset=22, datatype=PointField.UINT16, count=1),
        ]


    # RGB component helpers:

    def pack_rgb_float(self, r: int, g: int, b: int) -> float:
        rgb_uint32 = (int(r) << 16) | (int(g) << 8) | int(b)
        return struct.unpack("f", struct.pack("I", rgb_uint32))[0]

    def get_latest_image_copy(self):
        with self.latest_image_lock:
            if self.latest_image is None:
                return None
            return self.latest_image.copy()

    def sample_cell_rgb(self, img: np.ndarray, row: int, col: int, rows: int, cols: int) -> float:
        h, w = img.shape[:2]

        x0 = int(col * w / cols)
        x1 = int((col + 1) * w / cols)
        y0 = int(row * h / rows)
        y1 = int((row + 1) * h / rows)

        cell_w = max(1, x1 - x0)
        cell_h = max(1, y1 - y0)

        frac = max(0.05, min(1.0, self.color_patch_fraction))
        patch_w = max(1, int(round(cell_w * frac)))
        patch_h = max(1, int(round(cell_h * frac)))

        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2

        px0 = max(x0, cx - patch_w // 2)
        px1 = min(x1, px0 + patch_w)
        py0 = max(y0, cy - patch_h // 2)
        py1 = min(y1, py0 + patch_h)

        patch = img[py0:py1, px0:px1]
        if patch.size == 0:
            return self.pack_rgb_float(255, 0, 255)  # magenta for unavailable color

        if self.use_mean_color:
            mean_bgr = patch.reshape(-1, 3).mean(axis=0)
            b, g, r = [int(round(v)) for v in mean_bgr]
        else:
            b, g, r = [int(v) for v in patch[patch.shape[0] // 2, patch.shape[1] // 2]]

        return self.pack_rgb_float(r, g, b)


    def build_pointcloud2(
        self,
        seq: int,
        stamp_ns: int,
        rows: int,
        cols: int,
        points: List[Tuple[float, float, float, float, int, int]],
    ) -> PointCloud2:
        header = Header()
        header.frame_id = self.frame_id
        header.stamp.sec = int(stamp_ns // 1_000_000_000)
        header.stamp.nanosec = int(stamp_ns % 1_000_000_000)

        img = self.get_latest_image_copy()

        colored_points = []
        for x, y, z, confidence, row, col in points:
            if img is not None:
                rgb = self.sample_cell_rgb(img, row, col, rows, cols)
            else:
                rgb = self.pack_rgb_float(255, 255, 255)

            colored_points.append((x, y, z, rgb, confidence, row, col))

        msg = point_cloud2.create_cloud(header, self.fields, colored_points)
        msg.is_dense = False
        return msg

