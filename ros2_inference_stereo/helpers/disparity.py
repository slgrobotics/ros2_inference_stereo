#!/usr/bin/env python3

# =====================================================
# Using synchronized frames from two CSI cameras, applies stereo
# rectification using precomputed calibration, computes a disparity map via
# StereoSGBM, and converts it into a sparse 3D point representation.
#
# The image is divided into a fixed grid (e.g., 10x10 cells). For each cell,
# a representative 3D point is selected based on high-percentile disparity
# (closest obstacle), filtered for validity and range, and converted into a
# ROS-compatible coordinate frame.
#
# Key characteristics:
# - Sparse, obstacle-focused point cloud (one point per grid cell)
# - Designed for constrained platforms (Raspberry Pi)
#
# Intended use:
# - Lightweight stereo depth server feeding external ROS 2 processing
# - Obstacle detection and navigation experiments
# =====================================================

import cv2
import numpy as np

def make_valid_disparity_mask(disparity, min_valid_disp, invalid_left_cols, invalid_right_cols=0):
    valid = np.isfinite(disparity) & (disparity > min_valid_disp)

    if invalid_left_cols > 0:
        valid[:, :invalid_left_cols] = False

    if invalid_right_cols > 0:
        valid[:, -invalid_right_cols:] = False

    return valid


def derive_sgbm_params(
    close_cutout_factor: float = 1.0,
    far_smoothing_factor: float = 1.0,
):
    # ==============================================
    # Human-friendly mapping to StereoSGBM parameters.
    #
    # close_cutout_factor:
    #     Larger -> keep closer objects, larger disparity search range.
    #     Smaller -> near cutoff moves farther away.
    #
    # far_smoothing_factor:
    #     Larger -> smoother disparity, less detail.
    #     Smaller -> sharper detail, more noise.
    # ==============================================

    # Clamp to sane ranges
    close_cutout_factor = max(0.5, min(2.0, close_cutout_factor))
    far_smoothing_factor = max(0.5, min(2.0, far_smoothing_factor))

    # Baseline working settings
    base_min_disp = 1
    base_num_disp = 16 * 8
    base_block_size = 9

    # Near-range preservation:
    # Larger factor -> larger num_disp
    num_disp_steps = round(8 * close_cutout_factor)   # around 4..16
    num_disp_steps = max(4, min(16, num_disp_steps))
    num_disp = 16 * num_disp_steps

    # Slightly bias min_disp upward when focusing on near field
    min_disp = max(0, int(round(base_min_disp * close_cutout_factor)))

    # Smoothing/detail tradeoff:
    # 0.5 -> block 5
    # 1.0 -> block 9
    # 2.0 -> block 15
    block_size = int(round(9 * far_smoothing_factor))
    if block_size % 2 == 0:
        block_size += 1
    block_size = max(5, min(15, block_size))

    return min_disp, num_disp, block_size


def estimate_depth_cm_from_disparity(disparity_px, focal_px, baseline_m):
    if disparity_px <= 0:
        return None
    z_m = (focal_px * baseline_m) / disparity_px
    return z_m * 100.0

# A generic way of building depth image, using float precision
# This code contains nested Python for loops that process pixels individually.
# This is incredibly slow and is the root cause of the 0.4 Hz bottleneck.
def build_depth_image_float(disparity, points_3d, valid_mask, max_range_m=5.0):
    """Build a per-pixel depth image in meters from stereo reprojection data."""

    if disparity is None or points_3d is None or valid_mask is None:
        return None

    h, w = disparity.shape[:2]

    if points_3d.shape[:2] != (h, w):
        return None

    depth_image = np.full((h, w), np.nan, dtype=np.float32)

    for y in range(h):
        for x in range(w):
            if not valid_mask[y, x]:
                continue

            if not np.isfinite(disparity[y, x]) or disparity[y, x] <= 0.0:
                continue

            xyz = points_3d[y, x]
            x_cam, y_cam, z_cam = float(xyz[0]), float(xyz[1]), float(xyz[2])

            if not np.isfinite(x_cam) or not np.isfinite(y_cam) or not np.isfinite(z_cam):
                continue
            if z_cam <= 0.0 or z_cam > max_range_m:
                continue

            depth_image[y, x] = z_cam

    return depth_image

# ROS and RTAB-Map standard depth images natively prefer 16-bit integer formats (TYPE_16UC1)
#  where the pixel value represents the distance in millimeters.
#  Invalid pixels are set to 0 instead of np.nan. 
# This conversion drastically reduces network bandwidth, stops the RVL compression warning,
#  and optimizes RTAB-Map's processing pipelines.
# Here is the optimized, fully vectorized version of the function that outputs
#  a standard 16-bit millimeter depth image instantly using NumPy
def build_depth_image_uint16(disparity, points_3d, valid_mask, max_range_m=5.0):
    """Build a standard 16-bit integer (mm) depth image using fast vectorization."""

    if disparity is None or points_3d is None or valid_mask is None:
        return None

    h, w = disparity.shape[:2]

    if points_3d.shape[:2] != (h, w):
        return None

    # 1. Extract the Z-channel (depth) natively using slicing
    z_cam = points_3d[:, :, 2]

    # 2. Build a comprehensive mask combining all constraints at once
    combined_mask = (
        valid_mask & 
        np.isfinite(disparity) & (disparity > 0.0) &
        np.isfinite(points_3d[:, :, 0]) & np.isfinite(points_3d[:, :, 1]) & np.isfinite(z_cam) &
        (z_cam > 0.0) & (z_cam <= max_range_m)
    )

    # 3. Initialize the standard 16-bit unsigned integer depth image with 0 (invalid)
    depth_image_uint16 = np.zeros((h, w), dtype=np.uint16)

    # 4. Convert valid meters to millimeters (multiply by 1000) and cast to uint16
    depth_image_uint16[combined_mask] = (z_cam[combined_mask] * 1000.0).astype(np.uint16)

    return depth_image_uint16


def overlay_cell_distances(
    img,
    disparity,
    valid_mask,
    focal_px,
    baseline_m,
    rows=10,
    cols=10,
    max_depth_cm=999,
):
    out = draw_overlay_grid(img, rows=rows, cols=cols, color=(255, 255, 255), thickness=1)
    h, w = disparity.shape[:2]

    for r in range(rows):
        y0 = int(r * h / rows)
        y1 = int((r + 1) * h / rows)

        for c in range(cols):
            x0 = int(c * w / cols)
            x1 = int((c + 1) * w / cols)

            cell = disparity[y0:y1, x0:x1]
            cell_valid = valid_mask[y0:y1, x0:x1]

            if not np.any(cell_valid):
                text = "--"
            else:
                closest_disp = float(np.percentile(cell[cell_valid], 95))
                depth_cm = estimate_depth_cm_from_disparity(
                    closest_disp, focal_px, baseline_m
                )

                if depth_cm is None or not np.isfinite(depth_cm):
                    text = "--"
                else:
                    depth_cm = min(depth_cm, max_depth_cm)
                    text = f"{int(round(depth_cm))}"

            cx = x0 + (x1 - x0) // 2
            cy = y0 + (y1 - y0) // 2

            cv2.putText(
                out, text, (cx - 18, cy + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA
            )
            cv2.putText(
                out, text, (cx - 18, cy + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
            )

    return out


def cam_to_ros(x_cam, y_cam, z_cam):
    # =====================================================
    # OpenCV stereo camera coordinates:
    #   x right, y down, z forward
    #
    # ROS-like convention:
    #   x forward, y left, z up
    # =====================================================
    x_ros = z_cam
    y_ros = -x_cam
    z_ros = -y_cam
    return x_ros, y_ros, z_ros


def extract_sparse_points(
    disparity,
    points_3d,
    valid_mask,
    rows,
    cols,
    max_range_m,
    min_disp_confidence,
):
    # =====================================================
    # One representative point per cell.
    #
    # Strategy:
    # - valid disparity mask in cell
    # - pick 95th percentile disparity
    # - choose actual pixel nearest that target disparity
    # - emit XYZ + confidence + grid row/col
    # =====================================================
    
    h, w = disparity.shape[:2]
    points = []

    for r in range(rows):
        y0 = int(r * h / rows)
        y1 = int((r + 1) * h / rows)

        for c in range(cols):
            x0 = int(c * w / cols)
            x1 = int((c + 1) * w / cols)

            cell_disp = disparity[y0:y1, x0:x1]
            cell_valid = valid_mask[y0:y1, x0:x1]

            valid_fraction = float(np.count_nonzero(cell_valid)) / float(cell_disp.size)
            if valid_fraction < min_disp_confidence:
                continue

            valid_values = cell_disp[cell_valid]
            target_disp = float(np.percentile(valid_values, 95))

            ys, xs = np.where(cell_valid)
            disp_candidates = cell_disp[ys, xs]
            best_idx = int(np.argmin(np.abs(disp_candidates - target_disp)))

            py = y0 + int(ys[best_idx])
            px = x0 + int(xs[best_idx])

            xyz = points_3d[py, px]
            x_cam, y_cam, z_cam = float(xyz[0]), float(xyz[1]), float(xyz[2])

            if not np.isfinite(x_cam) or not np.isfinite(y_cam) or not np.isfinite(z_cam):
                continue
            if z_cam <= 0.0 or z_cam > max_range_m:
                continue

            x_ros, y_ros, z_ros = cam_to_ros(x_cam, y_cam, z_cam)
            points.append((x_ros, y_ros, z_ros, valid_fraction, r, c))

    return points


