"""Geometry-constrained lidar and yellow-wall matching against a static map."""

from dataclasses import dataclass
import math
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class FieldPoseMatch:
    x_m: float
    y_m: float
    yaw_rad: float
    score: float
    lidar_score: float
    wall_score: float
    ray_penalty: float
    alternative_score: float
    score_margin: float


class OccupancyDistanceField:
    def __init__(
        self,
        image: np.ndarray,
        resolution_m: float,
        origin_xy: Tuple[float, float] = (0.0, 0.0),
        occupied_pixel_threshold: int = 90,
    ):
        grayscale = np.asarray(image)
        if grayscale.ndim != 2 or grayscale.size == 0:
            raise ValueError("map image must be a non-empty grayscale array")
        if float(resolution_m) <= 0.0:
            raise ValueError("map resolution must be positive")
        self.image = grayscale.astype(np.uint8)
        self.resolution_m = float(resolution_m)
        self.origin_x = float(origin_xy[0])
        self.origin_y = float(origin_xy[1])
        self.height, self.width = self.image.shape
        self.occupied = self.image <= int(occupied_pixel_threshold)
        self.distance_m = cv2.distanceTransform(
            (~self.occupied).astype(np.uint8),
            cv2.DIST_L2,
            5,
        ) * self.resolution_m

    @classmethod
    def from_pgm(
        cls,
        image_path: str,
        resolution_m: float,
        origin_xy: Tuple[float, float] = (0.0, 0.0),
        occupied_pixel_threshold: int = 90,
    ):
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("could not read map image: {}".format(image_path))
        return cls(
            image,
            resolution_m,
            origin_xy,
            occupied_pixel_threshold,
        )

    def sample(self, points_m: np.ndarray):
        points = np.asarray(points_m, dtype=np.float64)
        columns = np.rint(
            (points[..., 0] - self.origin_x) / self.resolution_m
        ).astype(np.int32)
        rows = self.height - 1 - np.rint(
            (points[..., 1] - self.origin_y) / self.resolution_m
        ).astype(np.int32)
        valid = (
            (columns >= 0)
            & (columns < self.width)
            & (rows >= 0)
            & (rows < self.height)
        )
        values = np.full(columns.shape, 0.40, dtype=np.float32)
        values[valid] = self.distance_m[rows[valid], columns[valid]]
        return values, valid

    def free_positions(
        self,
        step_m: float,
        clearance_m: float,
        bounds_m: Optional[Tuple[float, float, float, float]] = None,
    ) -> np.ndarray:
        step = float(step_m)
        if step <= 0.0:
            raise ValueError("candidate position step must be positive")
        minimum_x = self.origin_x + 2.0 * step
        maximum_x = (
            self.origin_x
            + (self.width - 1) * self.resolution_m
            - 2.0 * step
        )
        minimum_y = self.origin_y + 2.0 * step
        maximum_y = (
            self.origin_y
            + (self.height - 1) * self.resolution_m
            - 2.0 * step
        )
        if bounds_m is not None:
            if len(bounds_m) != 4:
                raise ValueError("position bounds must be [xmin, ymin, xmax, ymax]")
            bound_min_x, bound_min_y, bound_max_x, bound_max_y = map(
                float, bounds_m
            )
            if bound_min_x >= bound_max_x or bound_min_y >= bound_max_y:
                raise ValueError("position bounds must have positive area")
            minimum_x = max(minimum_x, bound_min_x)
            minimum_y = max(minimum_y, bound_min_y)
            maximum_x = min(maximum_x, bound_max_x)
            maximum_y = min(maximum_y, bound_max_y)
        if minimum_x > maximum_x or minimum_y > maximum_y:
            return np.empty((0, 2), dtype=np.float64)
        x_values = np.arange(minimum_x, maximum_x + 0.5 * step, step)
        y_values = np.arange(minimum_y, maximum_y + 0.5 * step, step)
        xx, yy = np.meshgrid(x_values, y_values)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        distances, valid = self.sample(positions)
        return positions[valid & (distances >= float(clearance_m))]


def _rotation(yaw_rad: float) -> np.ndarray:
    cosine = math.cos(float(yaw_rad))
    sine = math.sin(float(yaw_rad))
    return np.asarray([[cosine, -sine], [sine, cosine]])


def normalize_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(float(angle_rad)), math.cos(float(angle_rad)))


def _pose_score(
    distance_field: OccupancyDistanceField,
    lidar_points_base_m: np.ndarray,
    wall_points_base_m: np.ndarray,
    ray_points_base_m: np.ndarray,
    x_m: float,
    y_m: float,
    yaw_rad: float,
    lidar_weight: float,
    wall_weight: float,
    ray_weight: float,
    wall_target_distance_m: float,
):
    rotation = _rotation(yaw_rad)
    translation = np.asarray([x_m, y_m], dtype=np.float64)
    lidar_global = translation + lidar_points_base_m @ rotation.T
    wall_global = translation + wall_points_base_m @ rotation.T
    rays_global = translation + ray_points_base_m @ rotation.T
    lidar_distance, lidar_valid = distance_field.sample(lidar_global)
    wall_distance, wall_valid = distance_field.sample(wall_global)
    ray_distance, ray_valid = distance_field.sample(rays_global)
    lidar_score = float(
        np.mean(np.minimum(lidar_distance, 0.25))
        + 0.20 * (1.0 - np.mean(lidar_valid))
    )
    wall_error = np.abs(
        wall_distance - float(wall_target_distance_m)
    )
    wall_score = float(
        np.mean(np.minimum(wall_error, 0.20))
        + 0.30 * (1.0 - np.mean(wall_valid))
    )
    ray_penalty = float(
        np.mean((ray_distance < 0.012) | (~ray_valid))
    )
    score = (
        float(lidar_weight) * lidar_score
        + float(wall_weight) * wall_score
        + float(ray_weight) * ray_penalty
    )
    return score, lidar_score, wall_score, ray_penalty


def search_field_pose(
    distance_field: OccupancyDistanceField,
    lidar_points_base_m: np.ndarray,
    wall_points_base_m: np.ndarray,
    sensor_origin_base_m: Iterable[float] = (0.0, 0.0),
    coarse_position_step_m: float = 0.05,
    coarse_yaw_step_deg: float = 5.0,
    refine_position_radius_m: float = 0.08,
    refine_position_step_m: float = 0.01,
    refine_yaw_radius_deg: float = 5.0,
    refine_yaw_step_deg: float = 1.0,
    robot_clearance_m: float = 0.16,
    lidar_weight: float = 1.0,
    wall_weight: float = 3.5,
    ray_weight: float = 0.12,
    maximum_lidar_points: int = 160,
    maximum_wall_points: int = 24,
    position_bounds_m: Optional[Tuple[float, float, float, float]] = None,
    wall_target_distance_m: float = 0.0,
) -> FieldPoseMatch:
    lidar = np.asarray(
        lidar_points_base_m, dtype=np.float64
    ).reshape(-1, 2)
    wall = np.asarray(
        wall_points_base_m, dtype=np.float64
    ).reshape(-1, 2)
    if len(lidar) < 30:
        raise ValueError("at least 30 finite lidar endpoints are required")
    if len(wall) < 4:
        raise ValueError("at least four right-wall points are required")
    if float(wall_target_distance_m) < 0.0:
        raise ValueError("yellow-to-wall distance must be non-negative")
    if float(lidar_weight) <= 0.0:
        raise ValueError("lidar weight must be positive")
    if len(lidar) > int(maximum_lidar_points):
        indices = np.linspace(
            0, len(lidar) - 1, int(maximum_lidar_points)
        ).astype(np.int32)
        lidar = lidar[indices]
    if len(wall) > int(maximum_wall_points):
        indices = np.linspace(
            0, len(wall) - 1, int(maximum_wall_points)
        ).astype(np.int32)
        wall = wall[indices]
    sensor_origin = np.asarray(
        list(sensor_origin_base_m), dtype=np.float64
    ).reshape(2)
    ray_points = sensor_origin + 0.55 * (lidar - sensor_origin)
    positions = distance_field.free_positions(
        coarse_position_step_m,
        robot_clearance_m,
        position_bounds_m,
    )
    if len(positions) == 0:
        raise ValueError("static map contains no coarse free-pose candidates")

    coarse_results: List[Tuple[float, ...]] = []
    yaw_values_deg = np.arange(
        -180.0,
        180.0,
        float(coarse_yaw_step_deg),
    )
    for yaw_deg in yaw_values_deg:
        yaw = math.radians(float(yaw_deg))
        rotation = _rotation(yaw)
        rotated_lidar = lidar @ rotation.T
        rotated_wall = wall @ rotation.T
        rotated_rays = ray_points @ rotation.T
        for start in range(0, len(positions), 700):
            batch = positions[start:start + 700]
            lidar_global = (
                batch[:, None, :] + rotated_lidar[None, :, :]
            )
            wall_global = (
                batch[:, None, :] + rotated_wall[None, :, :]
            )
            rays_global = (
                batch[:, None, :] + rotated_rays[None, :, :]
            )
            lidar_distance, lidar_valid = distance_field.sample(
                lidar_global
            )
            wall_distance, wall_valid = distance_field.sample(wall_global)
            ray_distance, ray_valid = distance_field.sample(rays_global)
            lidar_scores = (
                np.mean(np.minimum(lidar_distance, 0.25), axis=1)
                + 0.20 * (1.0 - np.mean(lidar_valid, axis=1))
            )
            wall_errors = np.abs(
                wall_distance - float(wall_target_distance_m)
            )
            wall_scores = (
                np.mean(np.minimum(wall_errors, 0.20), axis=1)
                + 0.30 * (1.0 - np.mean(wall_valid, axis=1))
            )
            ray_penalties = np.mean(
                (ray_distance < 0.012) | (~ray_valid), axis=1
            )
            scores = (
                float(lidar_weight) * lidar_scores
                + float(wall_weight) * wall_scores
                + float(ray_weight) * ray_penalties
            )
            take = min(3, len(scores))
            best_indices = np.argpartition(scores, take - 1)[:take]
            for index in best_indices:
                coarse_results.append(
                    (
                        float(scores[index]),
                        float(batch[index, 0]),
                        float(batch[index, 1]),
                        yaw,
                        float(lidar_scores[index]),
                        float(wall_scores[index]),
                        float(ray_penalties[index]),
                    )
                )
    coarse_results.sort(key=lambda item: item[0])
    seeds: List[Tuple[float, ...]] = []
    for candidate in coarse_results:
        distinct = all(
            (
                math.hypot(
                    candidate[1] - existing[1],
                    candidate[2] - existing[2],
                )
                > 0.12
            )
            or (
                abs(
                    normalize_angle(candidate[3] - existing[3])
                )
                > math.radians(8.0)
            )
            for existing in seeds
        )
        if distinct:
            seeds.append(candidate)
        if len(seeds) >= 8:
            break

    refined: List[Tuple[float, ...]] = []
    for seed in seeds:
        x_values = np.arange(
            seed[1] - float(refine_position_radius_m),
            seed[1] + float(refine_position_radius_m) + 0.5
            * float(refine_position_step_m),
            float(refine_position_step_m),
        )
        y_values = np.arange(
            seed[2] - float(refine_position_radius_m),
            seed[2] + float(refine_position_radius_m) + 0.5
            * float(refine_position_step_m),
            float(refine_position_step_m),
        )
        yaw_values = np.arange(
            seed[3] - math.radians(float(refine_yaw_radius_deg)),
            seed[3] + math.radians(float(refine_yaw_radius_deg))
            + 0.5 * math.radians(float(refine_yaw_step_deg)),
            math.radians(float(refine_yaw_step_deg)),
        )
        for x_m in x_values:
            for y_m in y_values:
                if position_bounds_m is not None and not (
                    float(position_bounds_m[0]) <= x_m
                    <= float(position_bounds_m[2])
                    and float(position_bounds_m[1]) <= y_m
                    <= float(position_bounds_m[3])
                ):
                    continue
                position_distance, position_valid = distance_field.sample(
                    np.asarray([[x_m, y_m]])
                )
                if (
                    not bool(position_valid[0])
                    or position_distance[0] < float(robot_clearance_m)
                ):
                    continue
                for yaw_rad in yaw_values:
                    components = _pose_score(
                        distance_field,
                        lidar,
                        wall,
                        ray_points,
                        x_m,
                        y_m,
                        yaw_rad,
                        lidar_weight,
                        wall_weight,
                        ray_weight,
                        wall_target_distance_m,
                    )
                    refined.append(
                        (
                            components[0],
                            float(x_m),
                            float(y_m),
                            normalize_angle(float(yaw_rad)),
                            components[1],
                            components[2],
                            components[3],
                        )
                    )
    if not refined:
        raise ValueError("pose refinement produced no valid candidate")
    refined.sort(key=lambda item: item[0])
    best = refined[0]
    alternative: Optional[Tuple[float, ...]] = None
    for candidate in refined[1:]:
        if (
            math.hypot(
                candidate[1] - best[1],
                candidate[2] - best[2],
            )
            >= 0.20
            or abs(normalize_angle(candidate[3] - best[3]))
            >= math.radians(12.0)
        ):
            alternative = candidate
            break
    alternative_score = (
        float("inf") if alternative is None else float(alternative[0])
    )
    return FieldPoseMatch(
        x_m=best[1],
        y_m=best[2],
        yaw_rad=best[3],
        score=best[0],
        lidar_score=best[4],
        wall_score=best[5],
        ray_penalty=best[6],
        alternative_score=alternative_score,
        score_margin=alternative_score - best[0],
    )
