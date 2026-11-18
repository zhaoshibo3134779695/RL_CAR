"""Static-map reference route generation and footprint validation."""

import math
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import yaml


def load_yaml(path: str) -> Tuple[Dict, Path]:
    resolved = Path(path).expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping: {}".format(resolved))
    return data, resolved


def smoothstep(value):
    clipped = np.clip(value, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def build_reference_path(route: Dict) -> np.ndarray:
    start_x = float(route["start"]["x_m"])
    finish_x = float(route["narrow_passage"]["finish_x_m"])
    spacing = float(route["sampling"]["spacing_m"])
    if not (start_x > finish_x and spacing > 0.0):
        raise ValueError("right-to-left route requires start_x > finish_x and positive spacing")
    count = int(math.ceil((start_x - finish_x) / spacing)) + 1
    x = np.linspace(start_x, finish_x, count, dtype=np.float64)
    alignment = route["alignment"]
    shift_start = float(alignment["start_x_m"])
    shift_end = float(alignment["end_x_m"])
    if not start_x >= shift_start > shift_end >= finish_x:
        raise ValueError("alignment x coordinates are not ordered for right-to-left travel")
    wide_y = float(alignment["wide_center_y_m"])
    narrow_y = float(alignment["narrow_center_y_m"])
    progress = (shift_start - x) / (shift_start - shift_end)
    y = wide_y + (narrow_y - wide_y) * smoothstep(progress)
    return np.column_stack([x, y])


def path_headings(path_m: np.ndarray) -> np.ndarray:
    if len(path_m) < 2:
        return np.zeros(len(path_m), dtype=np.float64)
    return np.unwrap(
        np.arctan2(np.gradient(path_m[:, 1]), np.gradient(path_m[:, 0]))
    )


def local_to_map(points_m: np.ndarray, pose, forward_origin_offset_m=0.0) -> np.ndarray:
    """Transform local [forward, left] points into ROS map coordinates."""
    points = np.asarray(points_m, dtype=np.float64).reshape(-1, 2).copy()
    if len(points) == 0:
        return points
    points[:, 0] += float(forward_origin_offset_m)
    x, y, yaw = [float(value) for value in pose]
    cosine, sine = math.cos(yaw), math.sin(yaw)
    rotation = np.asarray([[cosine, -sine], [sine, cosine]])
    return points.dot(rotation.T) + np.asarray([x, y], dtype=np.float64)


def avoidance_enabled(route: Dict, pose) -> bool:
    return bool(route["avoidance"]["enabled"]) and float(pose[0]) > float(
        route["avoidance"]["handoff_x_m"]
    )


def remaining_reference_path(reference_path_m: np.ndarray, pose) -> np.ndarray:
    """Return the right-to-left reference path beginning nearest the robot."""
    path = np.asarray(reference_path_m, dtype=np.float64).reshape(-1, 2)
    if len(path) == 0:
        return path
    point = np.asarray(pose[:2], dtype=np.float64)
    start = int(np.argmin(np.linalg.norm(path - point, axis=1)))
    return path[start:].copy()


def fuse_candidate_with_reference(
    route: Dict,
    reference_path_m: np.ndarray,
    pose,
    center_candidate_m: np.ndarray,
    selected_candidate_m: np.ndarray,
):
    """Apply the local avoidance offset to the map reference route.

    The local planner's absolute centreline is camera-centric. Subtracting its
    centre candidate leaves only the avoidance displacement, which is applied
    along the normal of the static map route. This guarantees a continuous
    return to the mapped narrow-passage centreline.
    """
    remaining = remaining_reference_path(reference_path_m, pose)
    if not avoidance_enabled(route, pose):
        return remaining, "fixed_narrow_route"
    center = np.asarray(center_candidate_m, dtype=np.float64).reshape(-1, 2)
    selected = np.asarray(selected_candidate_m, dtype=np.float64).reshape(-1, 2)
    if len(center) < 2 or len(selected) < 2:
        return None, "no_safe_local_candidate"
    selected_y = np.interp(center[:, 0], selected[:, 0], selected[:, 1])
    lateral_offset = selected_y - center[:, 1]
    if len(remaining) < 2:
        return None, "map_route_finished"
    distances = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(remaining, axis=0), axis=1))]
    )
    offsets = np.interp(
        distances,
        center[:, 0],
        lateral_offset,
        left=0.0,
        right=0.0,
    )
    headings = path_headings(remaining)
    normals = np.column_stack([-np.sin(headings), np.cos(headings)])
    fused = remaining + offsets[:, None] * normals
    return fused, "local_avoidance_fused"


def load_metric_map(route: Dict, route_path: Path):
    map_path = (route_path.parent / str(route["map_file"])).resolve()
    map_cfg, map_path = load_yaml(str(map_path))
    image_path = (map_path.parent / str(map_cfg["image"])).resolve()
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("could not read occupancy image: {}".format(image_path))
    resolution = float(map_cfg["resolution"])
    origin = [float(v) for v in map_cfg["origin"]]
    if abs(origin[2]) > 1e-9:
        raise ValueError("rotated map origins are not supported")
    return image, resolution, origin, map_cfg, map_path


def metric_to_pixel(points_m: np.ndarray, image, resolution: float, origin) -> np.ndarray:
    points = np.asarray(points_m, dtype=np.float64).reshape(-1, 2)
    px = (points[:, 0] - origin[0]) / resolution
    py = image.shape[0] - 1.0 - (points[:, 1] - origin[1]) / resolution
    return np.column_stack([px, py])


def _footprint_polygon(center, yaw, length_m, width_m, margin_m):
    half_length = 0.5 * float(length_m) + float(margin_m)
    half_width = 0.5 * float(width_m) + float(margin_m)
    local = np.asarray(
        [
            [-half_length, -half_width],
            [half_length, -half_width],
            [half_length, half_width],
            [-half_length, half_width],
        ],
        dtype=np.float64,
    )
    cosine, sine = math.cos(float(yaw)), math.sin(float(yaw))
    rotation = np.asarray([[cosine, -sine], [sine, cosine]])
    return local.dot(rotation.T) + np.asarray(center, dtype=np.float64)


def validate_vehicle_path(
    occupancy: np.ndarray,
    resolution: float,
    origin,
    path_m: np.ndarray,
    length_m: float,
    width_m: float,
    margin_m: float,
) -> Dict:
    occupied = occupancy < 128
    headings = path_headings(path_m)
    collision_indices = []
    for index, (center, yaw) in enumerate(zip(path_m, headings)):
        footprint = _footprint_polygon(center, yaw, length_m, width_m, margin_m)
        pixels = metric_to_pixel(footprint, occupancy, resolution, origin)
        polygon = np.rint(pixels).astype(np.int32)
        mask = np.zeros(occupancy.shape, dtype=np.uint8)
        cv2.fillConvexPoly(mask, polygon, 1)
        if bool(np.any(occupied & (mask > 0))):
            collision_indices.append(index)

    free = (~occupied).astype(np.uint8)
    clearance = cv2.distanceTransform(
        free, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    ) * resolution
    center_pixels = np.rint(
        metric_to_pixel(path_m, occupancy, resolution, origin)
    ).astype(np.int32)
    valid = (
        (center_pixels[:, 0] >= 0)
        & (center_pixels[:, 0] < occupancy.shape[1])
        & (center_pixels[:, 1] >= 0)
        & (center_pixels[:, 1] < occupancy.shape[0])
    )
    samples = np.zeros(len(path_m), dtype=np.float64)
    samples[valid] = clearance[
        center_pixels[valid, 1], center_pixels[valid, 0]
    ]
    return {
        "length_m": float(length_m),
        "width_m": float(width_m),
        "margin_m": float(margin_m),
        "collision_free": len(collision_indices) == 0 and bool(np.all(valid)),
        "collision_pose_count": len(collision_indices),
        "first_collision_index": None if not collision_indices else collision_indices[0],
        "minimum_centerline_wall_clearance_m": float(np.min(samples)),
    }


def validate_route(route_file: str):
    route, route_path = load_yaml(route_file)
    occupancy, resolution, origin, map_cfg, map_path = load_metric_map(
        route, route_path
    )
    profiles_path = (
        route_path.parent / str(route["vehicle_profiles_file"])
    ).resolve()
    profiles, _ = load_yaml(str(profiles_path))
    path_m = build_reference_path(route)
    headings = path_headings(path_m)
    margin = float(route["safety"]["map_boundary_margin_m"])
    vehicle_results = {
        name: validate_vehicle_path(
            occupancy,
            resolution,
            origin,
            path_m,
            float(profile["length_m"]),
            float(profile["width_m"]),
            margin,
        )
        for name, profile in profiles["profiles"].items()
    }
    handoff_x = float(route["avoidance"]["handoff_x_m"])
    segment_lengths = np.linalg.norm(np.diff(path_m, axis=0), axis=1)
    report = {
        "route_name": route["route_name"],
        "map_yaml": str(map_path),
        "map_size_m": [
            occupancy.shape[1] * resolution,
            occupancy.shape[0] * resolution,
        ],
        "map_resolution_m_per_pixel": resolution,
        "direction": route["travel_direction"],
        "path_point_count": len(path_m),
        "path_length_m": float(np.sum(segment_lengths)),
        "start_m": path_m[0].tolist(),
        "handoff_m": [handoff_x, float(route["narrow_passage"]["center_y_m"])],
        "finish_m": path_m[-1].tolist(),
        "avoidance_enabled_before_handoff": True,
        "avoidance_disabled_at_and_after_handoff": True,
        "maximum_heading_change_from_west_deg": float(
            np.max(np.abs(np.rad2deg(headings) - 180.0))
        ),
        "vehicle_validation": vehicle_results,
        "route_valid_for_all_recorded_vehicles": all(
            result["collision_free"] for result in vehicle_results.values()
        ),
    }
    return route, route_path, occupancy, resolution, origin, path_m, headings, report


def draw_route_preview(
    route: Dict,
    occupancy: np.ndarray,
    resolution: float,
    origin,
    path_m: np.ndarray,
) -> np.ndarray:
    canvas = cv2.cvtColor(occupancy, cv2.COLOR_GRAY2BGR)
    handoff_x = float(route["avoidance"]["handoff_x_m"])
    avoidance = path_m[:, 0] > handoff_x
    for mask, color in ((avoidance, (0, 80, 255)), (~avoidance, (0, 180, 0))):
        points = metric_to_pixel(path_m[mask], occupancy, resolution, origin)
        if len(points) >= 2:
            cv2.polylines(
                canvas, [np.rint(points).astype(np.int32)], False, color, 4, cv2.LINE_AA
            )
    labels = [
        (path_m[0], "START", (255, 0, 0)),
        (np.asarray([handoff_x, route["narrow_passage"]["center_y_m"]]), "NO DISC AFTER", (0, 180, 255)),
        (path_m[-1], "FINISH", (0, 150, 0)),
    ]
    for point, label, color in labels:
        pixel = metric_to_pixel([point], occupancy, resolution, origin)[0]
        center = tuple(np.rint(pixel).astype(int))
        cv2.circle(canvas, center, 6, color, -1, cv2.LINE_AA)
        cv2.putText(
            canvas, label, (center[0] + 8, center[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
        )
    return canvas
