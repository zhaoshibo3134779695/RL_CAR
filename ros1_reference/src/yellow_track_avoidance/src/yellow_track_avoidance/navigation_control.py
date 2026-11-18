"""ROS-independent path tracking helpers for map navigation."""

import math
from typing import Dict, Tuple

import numpy as np


def wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def nearest_path_index(path_m: np.ndarray, position_m) -> int:
    path = np.asarray(path_m, dtype=np.float64).reshape(-1, 2)
    if len(path) == 0:
        raise ValueError("path must not be empty")
    position = np.asarray(position_m, dtype=np.float64).reshape(2)
    return int(np.argmin(np.sum((path - position) ** 2, axis=1)))


def lookahead_target(
    path_m: np.ndarray, position_m, lookahead_m: float
) -> Tuple[np.ndarray, float, int]:
    """Return an interpolated lookahead point, its heading, and start index."""
    path = np.asarray(path_m, dtype=np.float64).reshape(-1, 2)
    if len(path) < 2:
        raise ValueError("path must contain at least two points")
    start = nearest_path_index(path, position_m)
    remaining = max(0.0, float(lookahead_m))
    current = path[start]
    for index in range(start, len(path) - 1):
        following = path[index + 1]
        delta = following - current
        length = float(np.linalg.norm(delta))
        if length > 1e-9 and remaining <= length:
            ratio = remaining / length
            target = current + ratio * delta
            heading = math.atan2(delta[1], delta[0])
            return target, heading, start
        remaining -= length
        current = following
    delta = path[-1] - path[-2]
    heading = math.atan2(delta[1], delta[0])
    return path[-1].copy(), heading, start


def compute_navigation_command(
    path_m: np.ndarray,
    pose,
    lookahead_m: float,
    goal_tolerance_m: float,
    goal_slowdown_distance_m: float,
    position_gain: float,
    heading_gain: float,
    maximum_linear_speed_mps: float,
    maximum_angular_speed_radps: float,
    yaw_slow_threshold_rad: float,
    yaw_stop_threshold_rad: float,
) -> Dict:
    """Compute a holonomic body-frame command for a map-frame path."""
    path = np.asarray(path_m, dtype=np.float64).reshape(-1, 2)
    if len(path) < 2 or not np.all(np.isfinite(path)):
        raise ValueError("path must contain at least two finite points")
    x, y, yaw = [float(value) for value in pose]
    goal_distance = float(np.linalg.norm(path[-1] - np.asarray([x, y])))
    if goal_distance <= float(goal_tolerance_m):
        return {
            "linear_x": 0.0,
            "linear_y": 0.0,
            "angular_z": 0.0,
            "goal_reached": True,
            "goal_distance_m": goal_distance,
            "yaw_error_rad": 0.0,
            "nearest_index": nearest_path_index(path, (x, y)),
            "target_m": path[-1].tolist(),
        }

    target, target_heading, nearest = lookahead_target(
        path, (x, y), lookahead_m
    )
    delta = target - np.asarray([x, y], dtype=np.float64)
    cosine, sine = math.cos(yaw), math.sin(yaw)
    local_x = cosine * delta[0] + sine * delta[1]
    local_y = -sine * delta[0] + cosine * delta[1]
    linear = float(position_gain) * np.asarray([local_x, local_y])

    maximum_linear = max(0.0, float(maximum_linear_speed_mps))
    norm = float(np.linalg.norm(linear))
    if norm > maximum_linear and norm > 1e-9:
        linear *= maximum_linear / norm
    slowdown_distance = max(float(goal_tolerance_m), float(goal_slowdown_distance_m))
    if goal_distance < slowdown_distance:
        scale = max(0.15, goal_distance / slowdown_distance)
        linear *= scale

    yaw_error = wrap_angle(target_heading - yaw)
    absolute_yaw_error = abs(yaw_error)
    slow_threshold = max(0.0, float(yaw_slow_threshold_rad))
    stop_threshold = max(slow_threshold + 1e-6, float(yaw_stop_threshold_rad))
    if absolute_yaw_error >= stop_threshold:
        linear[:] = 0.0
    elif absolute_yaw_error > slow_threshold:
        linear *= (stop_threshold - absolute_yaw_error) / (
            stop_threshold - slow_threshold
        )

    maximum_angular = max(0.0, float(maximum_angular_speed_radps))
    angular = float(np.clip(float(heading_gain) * yaw_error, -maximum_angular, maximum_angular))
    return {
        "linear_x": float(linear[0]),
        "linear_y": float(linear[1]),
        "angular_z": angular,
        "goal_reached": False,
        "goal_distance_m": goal_distance,
        "yaw_error_rad": yaw_error,
        "nearest_index": nearest,
        "target_m": target.tolist(),
    }
