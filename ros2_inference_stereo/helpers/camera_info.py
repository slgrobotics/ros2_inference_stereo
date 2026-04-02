import numpy as np
from sensor_msgs.msg import CameraInfo
from builtin_interfaces.msg import Time

class CameraInfoHelper:
    def __init__(self, calib):
        self.calib = calib
        self.camera_info_template = self._load_camera_info_template()

    def _load_camera_info_template(self) -> CameraInfo:
        required = ["K1", "D1", "image_width", "image_height"]
        for key in required:
            if key not in self.calib:
                raise KeyError(f"Calibration is missing required key '{key}'")

        k1 = self.calib["K1"]
        d1 = self.calib["D1"]

        width = int(self.calib["image_width"])
        height = int(self.calib["image_height"])

        info = CameraInfo()
        info.width = width
        info.height = height
        info.distortion_model = "plumb_bob"
        info.d = d1.ravel().astype(float).tolist()

        info.k = [
            float(k1[0, 0]), float(k1[0, 1]), float(k1[0, 2]),
            float(k1[1, 0]), float(k1[1, 1]), float(k1[1, 2]),
            float(k1[2, 0]), float(k1[2, 1]), float(k1[2, 2]),
        ]

        if "RL" in self.calib and "PL" in self.calib:
            rl = self.calib["RL"]
            pl = self.calib["PL"]

            info.r = [
                float(rl[0, 0]), float(rl[0, 1]), float(rl[0, 2]),
                float(rl[1, 0]), float(rl[1, 1]), float(rl[1, 2]),
                float(rl[2, 0]), float(rl[2, 1]), float(rl[2, 2]),
            ]

            info.p = [
                float(pl[0, 0]), float(pl[0, 1]), float(pl[0, 2]), float(pl[0, 3]),
                float(pl[1, 0]), float(pl[1, 1]), float(pl[1, 2]), float(pl[1, 3]),
                float(pl[2, 0]), float(pl[2, 1]), float(pl[2, 2]), float(pl[2, 3]),
            ]
        else:
            info.r = [
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0,
            ]
            info.p = [
                float(k1[0, 0]), float(k1[0, 1]), float(k1[0, 2]), 0.0,
                float(k1[1, 0]), float(k1[1, 1]), float(k1[1, 2]), 0.0,
                float(k1[2, 0]), float(k1[2, 1]), float(k1[2, 2]), 0.0,
            ]

        return info

    def build_scaled_camera_info(
        self,
        image_width: int,
        image_height: int,
        frame_id: str | None = None,
        stamp: Time | None = None,
    ) -> CameraInfo:
        template = self.camera_info_template

        cam_info = CameraInfo()
        cam_info.width = image_width
        cam_info.height = image_height
        cam_info.distortion_model = template.distortion_model
        cam_info.d = list(template.d)
        cam_info.r = list(template.r)

        if frame_id is not None:
            cam_info.header.frame_id = frame_id
        if stamp is not None:
            cam_info.header.stamp = stamp

        scale_x = float(image_width) / float(template.width)
        scale_y = float(image_height) / float(template.height)

        k = list(template.k)
        p = list(template.p)

        # Scale K
        k[0] *= scale_x   # fx
        k[2] *= scale_x   # cx
        k[4] *= scale_y   # fy
        k[5] *= scale_y   # cy

        # Scale P
        p[0] *= scale_x   # fx'
        p[2] *= scale_x   # cx'
        p[3] *= scale_x   # Tx
        p[5] *= scale_y   # fy'
        p[6] *= scale_y   # cy'

        cam_info.k = k
        cam_info.p = p

        return cam_info
    