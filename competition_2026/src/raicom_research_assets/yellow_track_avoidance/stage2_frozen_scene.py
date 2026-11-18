"""Pure helpers for the frozen Stage-2 corridor map.

The Stage-2 lidar capture already reduces the scene to three physical wall
segments.  This module rasterizes those segments once for costmap/AMCL use and
serializes the same geometry for the goal and compact-TEB publishers.
"""

import math

import numpy as np


SCENE_POINT_KEYS = (
    "left_wall_start",
    "left_wall_end",
    "right_wall_start",
    "right_wall_end",
    "terminal_right",
    "terminal_left",
    "terminal_center",
    "outer_corner",
)


def geometry_payload(geometry, frame_id="map"):
    """Return a JSON-safe representation of one frozen corridor geometry."""
    payload = {
        "frame_id": str(frame_id).lstrip("/"),
        "heading_rad": float(geometry.heading_rad),
        "measured_width_m": float(geometry.measured_width_m),
        "terminal_support_voxels": int(geometry.terminal_support_voxels),
        "terminal_rms_m": float(geometry.terminal_rms_m),
        "terminal_observed_span_m": float(geometry.terminal_observed_span_m),
    }
    for key in SCENE_POINT_KEYS:
        value = np.asarray(getattr(geometry, key), dtype=np.float64)
        payload[key] = [float(value[0]), float(value[1])]
    return payload


def wall_segments(payload):
    """Return right, left and terminal wall segments from a scene payload."""
    return (
        (payload["right_wall_start"], payload["right_wall_end"]),
        (payload["left_wall_start"], payload["left_wall_end"]),
        (payload["terminal_right"], payload["terminal_left"]),
    )


def goal_from_payload(payload, vehicle_front_offset_m=0.17, front_clearance_m=0.10):
    """Compute the base center that leaves the requested front-wall clearance."""
    heading = float(payload["heading_rad"])
    tangent = np.asarray((math.cos(heading), math.sin(heading)), dtype=np.float64)
    terminal_center = np.asarray(payload["terminal_center"], dtype=np.float64)
    setback = float(vehicle_front_offset_m) + float(front_clearance_m)
    return terminal_center - setback * tangent, heading


def rasterize_segments(
    segments,
    resolution_m=0.01,
    width_m=5.0,
    height_m=4.0,
    origin_x_m=-0.5,
    origin_y_m=-2.0,
    wall_thickness_m=0.01,
):
    """Rasterize finite metric line segments into a free/occupied int8 grid."""
    resolution = float(resolution_m)
    if resolution <= 0.0:
        raise ValueError("resolution_m must be positive")
    width = int(round(float(width_m) / resolution))
    height = int(round(float(height_m) / resolution))
    if width <= 0 or height <= 0:
        raise ValueError("map dimensions must be positive")
    grid = np.zeros((height, width), dtype=np.int8)
    radius_cells = max(0, int(math.ceil(0.5 * float(wall_thickness_m) / resolution)))

    for first, second in segments:
        start = np.asarray(first, dtype=np.float64)
        finish = np.asarray(second, dtype=np.float64)
        length = float(np.linalg.norm(finish - start))
        sample_count = max(2, int(math.ceil(length / (0.4 * resolution))) + 1)
        for point in np.linspace(start, finish, sample_count):
            cell_x = int(math.floor((point[0] - float(origin_x_m)) / resolution))
            cell_y = int(math.floor((point[1] - float(origin_y_m)) / resolution))
            for offset_y in range(-radius_cells, radius_cells + 1):
                row = cell_y + offset_y
                if row < 0 or row >= height:
                    continue
                for offset_x in range(-radius_cells, radius_cells + 1):
                    column = cell_x + offset_x
                    if 0 <= column < width:
                        grid[row, column] = 100
    return grid


def rasterize_scene(payload, **options):
    """Rasterize the three frozen corridor walls in ``payload``."""
    return rasterize_segments(wall_segments(payload), **options)
