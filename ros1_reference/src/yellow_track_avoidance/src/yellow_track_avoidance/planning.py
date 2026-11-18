"""Metric safe-corridor construction and smooth candidate-path planning."""

import math
from typing import List, Optional

import numpy as np

from .model import BoundaryResult, PathCandidate, PlanningResult, SafeCorridor, TrackedDisk


def smoothstep(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def path_hits_circle(points_m: np.ndarray, center_m, radius_m: float) -> bool:
    points = np.asarray(points_m, dtype=np.float64).reshape(-1, 2)
    if points.size == 0:
        return False
    return polyline_clearance_to_point(points, center_m) <= float(radius_m)


def polyline_clearance_to_point(points_m: np.ndarray, center_m) -> float:
    """Return the exact minimum distance from a sampled polyline to a point.

    Checking only vertices can miss a disk whose centre lies between two path
    samples.  Segment projection keeps collision checking independent of the
    configured path sample spacing.
    """
    points = np.asarray(points_m, dtype=np.float64).reshape(-1, 2)
    if len(points) == 0:
        return float("inf")
    center = np.asarray(center_m, dtype=np.float64).reshape(2)
    if len(points) == 1:
        return float(np.linalg.norm(points[0] - center))
    starts = points[:-1]
    segments = points[1:] - starts
    lengths_squared = np.sum(segments * segments, axis=1)
    projection = np.sum((center - starts) * segments, axis=1) / np.maximum(
        lengths_squared, 1e-12
    )
    projection = np.clip(projection, 0.0, 1.0)
    closest = starts + projection[:, None] * segments
    return float(np.min(np.linalg.norm(closest - center, axis=1)))


def build_safe_corridor(boundary: BoundaryResult, transformer, config) -> SafeCorridor:
    observed = (
        (boundary.observed_min_row_px >= 0)
        & (boundary.rows_px >= int(boundary.observed_min_row_px))
        & (boundary.rows_px <= int(boundary.observed_max_row_px))
        & np.isfinite(boundary.left_px)
        & np.isfinite(boundary.right_px)
        & np.isfinite(boundary.center_px)
    )
    if int(np.count_nonzero(observed)) < 2:
        return SafeCorridor(
            np.array([]), np.array([]), np.array([]), np.array([]),
            False, 0.0, "boundaries_unavailable",
        )
    pixels_left = np.column_stack([boundary.left_px[observed], boundary.rows_px[observed]])
    pixels_right = np.column_stack([boundary.right_px[observed], boundary.rows_px[observed]])
    pixels_center = np.column_stack([boundary.center_px[observed], boundary.rows_px[observed]])
    if not np.all(np.isfinite(pixels_left)) or not np.all(np.isfinite(pixels_right)):
        return SafeCorridor(np.array([]), np.array([]), np.array([]), np.array([]), False, 0.0, "boundaries_unavailable")

    left = transformer.pixel_to_metric(pixels_left)
    right = transformer.pixel_to_metric(pixels_right)
    center = transformer.pixel_to_metric(pixels_center)
    physical = config["physical"]
    safety = config["safety"]
    inset = (
        0.5 * float(physical["robot_width_m"])
        + float(safety["boundary_margin_m"])
        + 0.5 * float(physical["yellow_line_width_m"])
    )
    safe_left = left[:, 1] - inset
    safe_right = right[:, 1] + inset
    width = safe_left - safe_right
    observed_center_width = left[:, 1] - right[:, 1]
    expected_center_width = float(physical["track_gray_width_m"]) + float(physical["yellow_line_width_m"])
    measured = float(np.nanmedian(observed_center_width))
    tolerance = float(safety["track_width_tolerance_m"])
    if abs(measured - expected_center_width) > tolerance:
        return SafeCorridor(center[:, 0], safe_left, safe_right, center[:, 1], False, float(np.nanmin(width)), "track_width_inconsistent")
    minimum_width = float(np.nanmin(width))
    if minimum_width < float(safety["minimum_corridor_width_m"]):
        return SafeCorridor(center[:, 0], safe_left, safe_right, center[:, 1], False, minimum_width, "robot_does_not_fit_corridor")
    return SafeCorridor(center[:, 0], safe_left, safe_right, center[:, 1], True, minimum_width, "")


class LocalPathPlanner:
    def __init__(self, config):
        self.config = config
        self.cfg = config["planner"]
        self.previous_path: Optional[np.ndarray] = None

    def expanded_radius(self) -> float:
        physical = self.config["physical"]
        return (
            0.5 * float(physical["disk_diameter_m"])
            + 0.5 * float(physical["robot_width_m"])
            + float(self.config["safety"]["obstacle_margin_m"])
        )

    @staticmethod
    def _curvature(points: np.ndarray):
        if len(points) < 4:
            return np.zeros(len(points), dtype=np.float64)
        x, y = points[:, 0], points[:, 1]
        dx, dy = np.gradient(x), np.gradient(y)
        ddx, ddy = np.gradient(dx), np.gradient(dy)
        denominator = np.maximum((dx * dx + dy * dy) ** 1.5, 1e-6)
        return (dx * ddy - dy * ddx) / denominator

    def _offset_profile(self, forward: np.ndarray, offset: float, obstacles: List[TrackedDisk]) -> np.ndarray:
        if abs(offset) < 1e-9:
            return np.zeros_like(forward)
        relevant_x = [disk.center_m[0] for disk in obstacles if 0.0 < disk.center_m[0] <= forward[-1]]
        minimum_transition = float(self.cfg["minimum_transition_m"])
        if relevant_x:
            first = min(relevant_x)
            last = max(relevant_x)
            rise_end = max(minimum_transition, first - float(self.cfg["pre_obstacle_clearance_m"]))
            fall_start = min(forward[-1] - minimum_transition, last + float(self.cfg["post_obstacle_clearance_m"]))
            fall_start = max(rise_end, fall_start)
        else:
            rise_end = max(minimum_transition, 0.35 * forward[-1])
            fall_start = max(rise_end, 0.65 * forward[-1])
        profile = np.full_like(forward, offset, dtype=np.float64)
        profile[forward < rise_end] = offset * smoothstep(forward[forward < rise_end] / max(rise_end, 1e-6))
        after = forward > fall_start
        profile[after] = offset * (1.0 - smoothstep((forward[after] - fall_start) / max(forward[-1] - fall_start, 1e-6)))
        return profile

    def _evaluate(self, name: str, points: np.ndarray, offset: float, corridor: SafeCorridor, obstacles: List[TrackedDisk]) -> PathCandidate:
        forward = points[:, 0]
        lateral = points[:, 1]
        order = np.argsort(corridor.forward_m)
        corridor_x = corridor.forward_m[order]
        safe_left = np.interp(forward, corridor_x, corridor.safe_left_m[order])
        safe_right = np.interp(forward, corridor_x, corridor.safe_right_m[order])
        boundary_clearance = np.minimum(safe_left - lateral, lateral - safe_right)
        minimum_boundary = float(np.min(boundary_clearance))
        reject_reason = "" if minimum_boundary >= 0.0 else "yellow_boundary_collision"

        expanded = self.expanded_radius() + float(self.cfg["collision_sample_padding_m"])
        minimum_obstacle = float("inf")
        for disk in obstacles:
            clearance = polyline_clearance_to_point(points, disk.center_m) - expanded
            minimum_obstacle = min(minimum_obstacle, clearance)
        if minimum_obstacle < 0.0:
            reject_reason = "disk_collision"

        curvature = self._curvature(points)
        maximum_curvature = float(np.max(np.abs(curvature))) if curvature.size else 0.0
        if maximum_curvature > float(self.cfg["maximum_curvature"]):
            reject_reason = "curvature_limit"
        collision_free = not reject_reason

        length = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
        smoothness = float(np.mean(np.abs(np.diff(curvature)))) if curvature.size > 1 else 0.0
        previous_cost = 0.0
        if self.previous_path is not None and len(self.previous_path) > 1:
            previous_y = np.interp(forward, self.previous_path[:, 0], self.previous_path[:, 1])
            previous_cost = float(np.mean((lateral - previous_y) ** 2))
        weights = self.cfg["score_weights"]
        obstacle_term = 0.0 if math.isinf(minimum_obstacle) else 1.0 / max(0.01, minimum_obstacle + 0.01)
        boundary_term = 1.0 / max(0.01, minimum_boundary + 0.01)
        costs = {
            "obstacle_clearance": obstacle_term,
            "boundary_clearance": boundary_term,
            "lateral_offset": abs(offset),
            "path_length": length,
            "curvature": float(np.mean(np.abs(curvature))) if curvature.size else 0.0,
            "smoothness": smoothness,
            "previous_path": previous_cost,
            "return_to_center": abs(float(lateral[-1])),
        }
        score = sum(float(weights[key]) * value for key, value in costs.items())
        if not collision_free:
            score = float("inf")
        return PathCandidate(
            name=name, points_m=points, offset_m=offset,
            direction=1 if offset > 0.0 else (-1 if offset < 0.0 else 0),
            collision_free=collision_free, score=score, costs=costs,
            minimum_boundary_clearance_m=minimum_boundary,
            minimum_obstacle_clearance_m=minimum_obstacle,
            max_curvature=maximum_curvature, reject_reason=reject_reason,
        )

    def plan(self, corridor: SafeCorridor, obstacles: List[TrackedDisk], locked_direction: int = 0) -> PlanningResult:
        if not corridor.valid or corridor.forward_m.size < 2:
            return PlanningResult([], None, False, 0.0, 0.0, corridor.reason or "invalid_corridor")
        horizon = min(float(self.cfg["horizon_m"]), float(np.max(corridor.forward_m)))
        spacing = float(self.cfg["sample_spacing_m"])
        if horizon <= spacing:
            return PlanningResult([], None, False, 0.0, 0.0, "planning_horizon_too_short")
        forward = np.arange(0.0, horizon + 0.5 * spacing, spacing)
        order = np.argsort(corridor.forward_m)
        center = np.interp(forward, corridor.forward_m[order], corridor.center_m[order])

        expanded = self.expanded_radius()
        relevant = [disk for disk in obstacles if 0.0 <= disk.center_m[0] <= horizon + expanded]
        left_space = float("inf")
        right_space = float("inf")
        for disk in relevant:
            left_limit = float(np.interp(disk.center_m[0], corridor.forward_m[order], corridor.safe_left_m[order]))
            right_limit = float(np.interp(disk.center_m[0], corridor.forward_m[order], corridor.safe_right_m[order]))
            left_space = min(left_space, left_limit - (disk.center_m[1] + expanded))
            right_space = min(right_space, (disk.center_m[1] - expanded) - right_limit)
        if not relevant:
            left_space = right_space = 0.5 * corridor.minimum_width_m

        candidates: List[PathCandidate] = []
        for offset in [float(value) for value in self.cfg["candidate_offsets_m"]]:
            direction = 1 if offset > 0.0 else (-1 if offset < 0.0 else 0)
            if locked_direction and direction not in (0, locked_direction):
                continue
            lateral = center + self._offset_profile(forward, offset, relevant)
            points = np.column_stack([forward, lateral])
            label = "center" if direction == 0 else ("left" if direction > 0 else "right") + "_{:.2f}".format(abs(offset))
            candidates.append(self._evaluate(label, points, offset, corridor, relevant))

        valid = [candidate for candidate in candidates if candidate.collision_free]
        selected = min(valid, key=lambda item: item.score) if valid else None
        if selected is not None:
            self.previous_path = selected.points_m.copy()
        reason = "" if selected is not None else "no_collision_free_candidate"
        return PlanningResult(candidates, selected, selected is not None, left_space, right_space, reason)
