"""Stage-2 narrow-corridor geometry from a stationary 2-D lidar capture.

The Stage-2 start pose is already inside/aligned with the narrow passage.  Two
longitudinal wall surfaces determine its centerline and heading.  The
perpendicular wall at the far, outside corner determines the end cross-section.
The navigation goal is a base-link pose whose *front edge* remains a requested
distance before that end wall.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ForwardWallFit:
    anchor: np.ndarray
    tangent: np.ndarray
    left_normal: np.ndarray
    inliers: np.ndarray
    rms_m: float
    minimum_s_m: float
    maximum_s_m: float


@dataclass(frozen=True)
class Stage2TerminalGeometry:
    right_wall: ForwardWallFit
    left_wall_start: np.ndarray
    left_wall_end: np.ndarray
    right_wall_start: np.ndarray
    right_wall_end: np.ndarray
    terminal_right: np.ndarray
    terminal_left: np.ndarray
    terminal_center: np.ndarray
    outer_corner: np.ndarray
    goal_base_center: np.ndarray
    goal_front_center: np.ndarray
    heading_rad: float
    measured_width_m: float
    terminal_support_voxels: int
    terminal_rms_m: float
    terminal_observed_span_m: float


def voxel_downsample(points, resolution_m=0.015):
    """Average all points falling in the same square metric voxel."""
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) == 0:
        raise ValueError("point array must have shape Nx2 and be non-empty")
    keys = np.floor(values / float(resolution_m)).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    return np.column_stack(
        (
            np.bincount(inverse, weights=values[:, 0]) / counts,
            np.bincount(inverse, weights=values[:, 1]) / counts,
        )
    )


def fit_right_wall(
    points,
    minimum_x_m=0.20,
    maximum_x_m=3.50,
    maximum_y_m=-0.10,
    ransac_threshold_m=0.025,
    minimum_inliers=18,
    minimum_sample_span_m=0.55,
    minimum_forward_component=0.88,
    iterations=5000,
    random_seed=20260812,
):
    """Fit the vehicle-right longitudinal wall with RANSAC and TLS."""
    values = np.asarray(points, dtype=np.float64)
    candidates = values[
        (values[:, 0] >= float(minimum_x_m))
        & (values[:, 0] <= float(maximum_x_m))
        & (values[:, 1] <= float(maximum_y_m))
    ]
    if len(candidates) < int(minimum_inliers):
        raise ValueError("insufficient points for the Stage-2 right wall")

    generator = np.random.default_rng(int(random_seed))
    best = None
    for _ in range(int(iterations)):
        first, second = generator.choice(len(candidates), 2, replace=False)
        vector = candidates[second] - candidates[first]
        span = float(np.linalg.norm(vector))
        if span < float(minimum_sample_span_m):
            continue
        tangent = vector / span
        if tangent[0] < 0.0:
            tangent = -tangent
        if tangent[0] < float(minimum_forward_component):
            continue
        normal = np.asarray((-tangent[1], tangent[0]), dtype=np.float64)
        residual = np.abs((candidates - candidates[first]) @ normal)
        selected = residual <= float(ransac_threshold_m)
        count = int(np.count_nonzero(selected))
        if count < int(minimum_inliers):
            continue
        longitudinal = (candidates[selected] - candidates[first]) @ tangent
        score = count + 20.0 * float(np.ptp(longitudinal))
        if best is None or score > best[0]:
            best = score, selected
    if best is None:
        raise ValueError("unable to fit the Stage-2 right wall")

    inliers = candidates[best[1]]
    anchor = np.mean(inliers, axis=0)
    _, _, vectors = np.linalg.svd(inliers - anchor, full_matrices=False)
    tangent = vectors[0]
    if tangent[0] < 0.0:
        tangent = -tangent
    left_normal = np.asarray((-tangent[1], tangent[0]), dtype=np.float64)
    if left_normal[1] < 0.0:
        left_normal = -left_normal
    lateral = (inliers - anchor) @ left_normal
    longitudinal = (inliers - anchor) @ tangent
    return ForwardWallFit(
        anchor=anchor,
        tangent=tangent,
        left_normal=left_normal,
        inliers=inliers,
        rms_m=float(np.sqrt(np.mean(lateral * lateral))),
        minimum_s_m=float(np.min(longitudinal)),
        maximum_s_m=float(np.max(longitudinal)),
    )


def _longitudinal_clusters(indices, longitudinal, cluster_gap_m):
    order = indices[np.argsort(longitudinal[indices])]
    clusters = []
    current = [int(order[0])]
    for previous, following in zip(order[:-1], order[1:]):
        if longitudinal[following] - longitudinal[previous] <= float(cluster_gap_m):
            current.append(int(following))
        else:
            clusters.append(np.asarray(current, dtype=np.int64))
            current = [int(following)]
    clusters.append(np.asarray(current, dtype=np.int64))
    return clusters


def build_stage2_geometry(
    points,
    expected_width_m=0.485,
    width_tolerance_m=0.10,
    vehicle_front_offset_m=0.17,
    desired_front_clearance_m=0.10,
    left_wall_band_m=0.10,
    left_wall_inlier_threshold_m=0.030,
    terminal_side_exclusion_m=0.05,
    terminal_minimum_forward_m=1.0,
    terminal_cluster_gap_m=0.08,
    terminal_minimum_support=8,
    terminal_minimum_span_m=0.22,
    **right_wall_options
):
    """Return the Stage-2 end-wall and goal geometry in the capture frame."""
    values = np.asarray(points, dtype=np.float64)
    wall = fit_right_wall(values, **right_wall_options)
    relative = values - wall.anchor
    longitudinal = relative @ wall.tangent
    lateral = relative @ wall.left_normal

    expected = float(expected_width_m)
    left_band = (
        (values[:, 0] >= float(right_wall_options.get("minimum_x_m", 0.20)))
        & (values[:, 0] <= float(right_wall_options.get("maximum_x_m", 3.50)))
        & (np.abs(lateral - expected) <= float(left_wall_band_m))
    )
    left_candidates = np.flatnonzero(left_band)
    if len(left_candidates) < 12:
        raise ValueError("insufficient points for the Stage-2 left wall")
    measured_width = float(np.median(lateral[left_candidates]))
    left_inliers = left_candidates[
        np.abs(lateral[left_candidates] - measured_width)
        <= float(left_wall_inlier_threshold_m)
    ]
    if len(left_inliers) < 10:
        raise ValueError("Stage-2 left wall lacks stable parallel support")
    measured_width = float(np.median(lateral[left_inliers]))
    if abs(measured_width - expected) > float(width_tolerance_m):
        raise ValueError(
            "measured Stage-2 width %.3f m differs from expected %.3f m"
            % (measured_width, expected)
        )

    interior = (
        (values[:, 0] >= float(terminal_minimum_forward_m))
        & (lateral >= float(terminal_side_exclusion_m))
        & (lateral <= measured_width - float(terminal_side_exclusion_m))
    )
    terminal_indices = np.flatnonzero(interior)
    if not len(terminal_indices):
        raise ValueError("no Stage-2 terminal-wall points were observed")
    clusters = _longitudinal_clusters(
        terminal_indices, longitudinal, terminal_cluster_gap_m
    )
    eligible = []
    for cluster in clusters:
        support = len(cluster)
        span = float(np.ptp(lateral[cluster])) if support > 1 else 0.0
        rms = float(np.std(longitudinal[cluster])) if support > 1 else math.inf
        if support < int(terminal_minimum_support) or span < float(
            terminal_minimum_span_m
        ):
            continue
        # Prefer a wide perpendicular return, then support and forward range.
        score = 6.0 * span + 0.03 * support + 0.05 * float(
            np.median(longitudinal[cluster])
        )
        eligible.append((score, cluster, span, rms))
    if not eligible:
        raise ValueError("the Stage-2 terminal outside wall was not captured")
    _, terminal_cluster, observed_span, terminal_rms = max(
        eligible, key=lambda item: item[0]
    )
    terminal_s = float(np.median(longitudinal[terminal_cluster]))

    right_s_min = max(wall.minimum_s_m, float(np.percentile(longitudinal[left_inliers], 2)))
    left_s_max = min(terminal_s, float(np.percentile(longitudinal[left_inliers], 98)))
    right_start = wall.anchor + right_s_min * wall.tangent
    right_end = wall.anchor + terminal_s * wall.tangent
    left_start = right_start + measured_width * wall.left_normal
    left_end = wall.anchor + left_s_max * wall.tangent + measured_width * wall.left_normal
    terminal_right = right_end
    terminal_left = terminal_right + measured_width * wall.left_normal
    terminal_center = 0.5 * (terminal_right + terminal_left)

    setback = float(vehicle_front_offset_m) + float(desired_front_clearance_m)
    goal = terminal_center - setback * wall.tangent
    front = goal + float(vehicle_front_offset_m) * wall.tangent
    return Stage2TerminalGeometry(
        right_wall=wall,
        left_wall_start=left_start,
        left_wall_end=left_end,
        right_wall_start=right_start,
        right_wall_end=right_end,
        terminal_right=terminal_right,
        terminal_left=terminal_left,
        terminal_center=terminal_center,
        outer_corner=terminal_right,
        goal_base_center=goal,
        goal_front_center=front,
        heading_rad=math.atan2(wall.tangent[1], wall.tangent[0]),
        measured_width_m=measured_width,
        terminal_support_voxels=int(len(terminal_cluster)),
        terminal_rms_m=terminal_rms,
        terminal_observed_span_m=observed_span,
    )
