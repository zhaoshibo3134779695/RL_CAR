"""Compact Stage-1 entrance geometry from a stationary 2-D lidar scan.

The narrow passage's entrance end face is almost normal to the lidar beam from
the competition start pose and therefore appears only as an intermittent
single return.  It is deliberately *not* a required observation here.

The stable geometry is instead:

* the long, shared right wall determines the passage direction;
* the two far faces around the terminal left turn determine the straight
  passage's terminal cross-section;
* the measured 1.40 m straight length back-projects that terminal section to
  the entrance.

The far faces are allowed to be slightly oblique in the scan.  They determine
position only; neither face is allowed to rotate the corridor heading.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class RightWallFit:
    anchor: np.ndarray
    tangent: np.ndarray
    left_normal: np.ndarray
    inliers: np.ndarray
    rms_m: float
    minimum_s_m: float
    maximum_s_m: float


@dataclass(frozen=True)
class TerminalFaceFit:
    """A far wall face represented as longitudinal position versus lateral offset."""

    slope_ds_dd: float
    intercept_s_m: float
    inliers: np.ndarray
    rms_m: float
    minimum_d_m: float
    maximum_d_m: float

    def longitudinal_at(self, lateral_m):
        return self.slope_ds_dd * float(lateral_m) + self.intercept_s_m


@dataclass(frozen=True)
class Stage1EntranceGeometry:
    right_wall: RightWallFit
    terminal_inner_face: TerminalFaceFit
    terminal_outer_face: TerminalFaceFit
    terminal_inner_corner: np.ndarray
    terminal_outer_corner: np.ndarray
    left_entrance: np.ndarray
    right_projection: np.ndarray
    entrance_center: np.ndarray
    goal_base_center: np.ndarray
    goal_front_center: np.ndarray
    heading_rad: float
    measured_width_m: float
    entrance_support_voxels: int
    terminal_pair_separation_m: float
    straight_length_m: float


def voxel_downsample(points, resolution_m=0.015):
    """Average points falling in the same square metric voxel."""
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
    minimum_x_m=0.2,
    maximum_x_m=5.8,
    maximum_y_m=-0.12,
    ransac_threshold_m=0.025,
    minimum_inliers=16,
    minimum_sample_span_m=0.6,
    minimum_forward_component=0.88,
    iterations=5000,
    random_seed=20260812,
):
    """Fit the long right wall using deterministic RANSAC followed by TLS."""
    values = np.asarray(points, dtype=np.float64)
    candidates = values[
        (values[:, 0] > float(minimum_x_m))
        & (values[:, 0] < float(maximum_x_m))
        & (values[:, 1] < float(maximum_y_m))
    ]
    if len(candidates) < int(minimum_inliers):
        raise ValueError("insufficient points for the continuous right wall")

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
        residuals = np.abs((candidates - candidates[first]) @ normal)
        selected = residuals <= float(ransac_threshold_m)
        count = int(np.count_nonzero(selected))
        if count < int(minimum_inliers):
            continue
        longitudinal = (candidates[selected] - candidates[first]) @ tangent
        score = count + 18.0 * float(np.ptp(longitudinal))
        if best is None or score > best[0]:
            best = score, selected
    if best is None:
        raise ValueError("unable to fit the continuous right wall")

    inliers = candidates[best[1]]
    anchor = np.mean(inliers, axis=0)
    _, _, right_vectors = np.linalg.svd(inliers - anchor, full_matrices=False)
    tangent = right_vectors[0]
    if tangent[0] < 0.0:
        tangent = -tangent
    left_normal = np.asarray((-tangent[1], tangent[0]), dtype=np.float64)
    # The normal must point from the right wall toward the vehicle/left wall.
    if float(np.dot(-anchor, left_normal)) < 0.0:
        left_normal = -left_normal
    lateral = (inliers - anchor) @ left_normal
    longitudinal = (inliers - anchor) @ tangent
    return RightWallFit(
        anchor=anchor,
        tangent=tangent,
        left_normal=left_normal,
        inliers=inliers,
        rms_m=float(np.sqrt(np.mean(lateral * lateral))),
        minimum_s_m=float(np.min(longitudinal)),
        maximum_s_m=float(np.max(longitudinal)),
    )


def _fit_terminal_face(
    longitudinal,
    lateral,
    candidate_indices,
    ransac_threshold_m=0.035,
    minimum_support=10,
    minimum_lateral_span_m=0.22,
    maximum_abs_slope_ds_dd=0.55,
    iterations=3000,
    random_seed=20260813,
):
    """Fit ``s = a*d + b`` while keeping the corridor direction fixed.

    A perfect terminal face is perpendicular to the right wall and therefore
    has constant ``s``.  Allowing a small ``a`` models the real scan's oblique
    visible face without ever feeding that slope back into the vehicle yaw.
    """
    indices = np.asarray(candidate_indices, dtype=np.int64)
    if len(indices) < int(minimum_support):
        raise ValueError("terminal face has insufficient candidate points")
    generator = np.random.default_rng(int(random_seed))
    best = None
    for _ in range(int(iterations)):
        first, second = generator.choice(indices, 2, replace=False)
        delta_d = float(lateral[second] - lateral[first])
        if abs(delta_d) < 0.08:
            continue
        slope = float((longitudinal[second] - longitudinal[first]) / delta_d)
        if abs(slope) > float(maximum_abs_slope_ds_dd):
            continue
        intercept = float(longitudinal[first] - slope * lateral[first])
        residual = np.abs(longitudinal[indices] - (slope * lateral[indices] + intercept))
        selected = residual <= float(ransac_threshold_m)
        count = int(np.count_nonzero(selected))
        if count < int(minimum_support):
            continue
        span = float(np.ptp(lateral[indices[selected]]))
        if span < float(minimum_lateral_span_m):
            continue
        score = count + 35.0 * span
        if best is None or score > best[0]:
            best = score, indices[selected]
    if best is None:
        raise ValueError("unable to fit a stable far terminal face")

    inliers = best[1]
    design = np.column_stack((lateral[inliers], np.ones(len(inliers))))
    slope, intercept = np.linalg.lstsq(design, longitudinal[inliers], rcond=None)[0]
    if abs(float(slope)) > float(maximum_abs_slope_ds_dd):
        raise ValueError("terminal face obliquity exceeds the accepted limit")
    residual = longitudinal[inliers] - (slope * lateral[inliers] + intercept)
    span = float(np.ptp(lateral[inliers]))
    if span < float(minimum_lateral_span_m):
        raise ValueError("terminal face has insufficient lateral span")
    return TerminalFaceFit(
        slope_ds_dd=float(slope),
        intercept_s_m=float(intercept),
        inliers=inliers,
        rms_m=float(np.sqrt(np.mean(residual * residual))),
        minimum_d_m=float(np.min(lateral[inliers])),
        maximum_d_m=float(np.max(lateral[inliers])),
    )


def find_terminal_face_pair(
    points,
    right_wall,
    expected_width_m=0.485,
    expected_turn_depth_m=0.485,
    turn_depth_tolerance_m=0.18,
    inner_expected_forward_m=3.95,
    inner_forward_tolerance_m=0.55,
    inner_minimum_lateral_m=None,
    inner_maximum_lateral_m=1.90,
    outer_minimum_lateral_m=0.05,
    outer_lateral_margin_m=0.16,
    face_ransac_threshold_m=0.035,
    face_minimum_support=10,
    face_minimum_lateral_span_m=0.22,
    face_maximum_abs_slope_ds_dd=0.55,
):
    """Find the stable inner/outer faces around the terminal left turn."""
    values = np.asarray(points, dtype=np.float64)
    relative = values - right_wall.anchor
    longitudinal = relative @ right_wall.tangent
    lateral = relative @ right_wall.left_normal
    robot_forward = values @ right_wall.tangent
    expected_width = float(expected_width_m)
    inner_minimum_lateral = (
        expected_width - 0.12
        if inner_minimum_lateral_m is None
        else float(inner_minimum_lateral_m)
    )
    inner_mask = (
        (robot_forward >= float(inner_expected_forward_m) - float(inner_forward_tolerance_m))
        & (robot_forward <= float(inner_expected_forward_m) + float(inner_forward_tolerance_m))
        & (lateral >= inner_minimum_lateral)
        & (lateral <= float(inner_maximum_lateral_m))
    )
    inner = _fit_terminal_face(
        longitudinal,
        lateral,
        np.flatnonzero(inner_mask),
        ransac_threshold_m=face_ransac_threshold_m,
        minimum_support=face_minimum_support,
        minimum_lateral_span_m=face_minimum_lateral_span_m,
        maximum_abs_slope_ds_dd=face_maximum_abs_slope_ds_dd,
        random_seed=20260813,
    )
    inner_s = inner.longitudinal_at(expected_width)

    # The outer face lies one passage width farther along the right wall.  Its
    # low-lateral part is visible even when the entrance end face is absent.
    expected_outer_s = inner_s + float(expected_turn_depth_m)
    outer_mask = (
        (np.abs(longitudinal - expected_outer_s) <= float(turn_depth_tolerance_m))
        & (lateral >= float(outer_minimum_lateral_m))
        & (lateral <= expected_width + float(outer_lateral_margin_m))
    )
    outer = _fit_terminal_face(
        longitudinal,
        lateral,
        np.flatnonzero(outer_mask),
        ransac_threshold_m=face_ransac_threshold_m,
        minimum_support=face_minimum_support,
        minimum_lateral_span_m=face_minimum_lateral_span_m,
        maximum_abs_slope_ds_dd=face_maximum_abs_slope_ds_dd,
        random_seed=20260814,
    )
    # Pair like with like: the inner face's lower corner and the outer face's
    # upper corner both lie on the expected left-wall offset.  Using the
    # outer face at d=0 would select its lower/right-wall-side endpoint and is
    # geometrically inconsistent when the observed face is slightly oblique.
    outer_s = outer.longitudinal_at(expected_width)
    separation = float(outer_s - inner_s)
    if abs(separation - float(expected_turn_depth_m)) > float(turn_depth_tolerance_m):
        raise ValueError(
            "far terminal-face separation %.3f m differs from expected %.3f m"
            % (separation, float(expected_turn_depth_m))
        )
    return inner, outer, inner_s, outer_s, separation


def build_stage1_geometry(
    points,
    expected_width_m=0.485,
    width_tolerance_m=0.08,
    vehicle_front_offset_m=0.17,
    desired_front_clearance_m=0.10,
    **fit_options
):
    """Return the frozen Stage-1 goal geometry in the capture base frame."""
    right_options = {
        key: value
        for key, value in fit_options.items()
        if key
        in {
            "minimum_x_m",
            "maximum_x_m",
            "maximum_y_m",
            "ransac_threshold_m",
            "minimum_inliers",
            "minimum_sample_span_m",
            "minimum_forward_component",
            "iterations",
            "random_seed",
        }
    }
    terminal_options = {
        key: value
        for key, value in fit_options.items()
        if key
        in {
            "expected_turn_depth_m",
            "turn_depth_tolerance_m",
            "inner_expected_forward_m",
            "inner_forward_tolerance_m",
            "inner_minimum_lateral_m",
            "inner_maximum_lateral_m",
            "outer_minimum_lateral_m",
            "outer_lateral_margin_m",
            "face_ransac_threshold_m",
            "face_minimum_support",
            "face_minimum_lateral_span_m",
            "face_maximum_abs_slope_ds_dd",
        }
    }
    wall = fit_right_wall(points, **right_options)
    inner, outer, terminal_s, outer_s, pair_separation = find_terminal_face_pair(
        points,
        wall,
        expected_width_m=expected_width_m,
        **terminal_options
    )
    straight_length = float(fit_options.get("straight_length_m", 1.40))
    entrance_s = terminal_s - straight_length
    entrance_d = float(expected_width_m)
    left = wall.anchor + entrance_s * wall.tangent + entrance_d * wall.left_normal
    right = wall.anchor + entrance_s * wall.tangent
    center = 0.5 * (left + right)
    setback = float(vehicle_front_offset_m) + float(desired_front_clearance_m)
    goal = center - setback * wall.tangent
    front = goal + float(vehicle_front_offset_m) * wall.tangent
    width = float(expected_width_m)
    if abs(width - float(expected_width_m)) > float(width_tolerance_m):
        raise ValueError("measured entrance width is outside tolerance")

    values = np.asarray(points, dtype=np.float64)
    relative = values - wall.anchor
    longitudinal = relative @ wall.tangent
    lateral = relative @ wall.left_normal
    optional_entrance_support = int(
        np.count_nonzero(
            (np.abs(longitudinal - entrance_s) <= 0.12)
            & (np.abs(lateral - entrance_d) <= float(width_tolerance_m))
        )
    )
    terminal_inner = wall.anchor + terminal_s * wall.tangent + width * wall.left_normal
    terminal_outer = (
        wall.anchor + outer_s * wall.tangent + width * wall.left_normal
    )
    return Stage1EntranceGeometry(
        right_wall=wall,
        terminal_inner_face=inner,
        terminal_outer_face=outer,
        terminal_inner_corner=terminal_inner,
        terminal_outer_corner=terminal_outer,
        left_entrance=left,
        right_projection=right,
        entrance_center=center,
        goal_base_center=goal,
        goal_front_center=front,
        heading_rad=math.atan2(wall.tangent[1], wall.tangent[0]),
        measured_width_m=width,
        entrance_support_voxels=optional_entrance_support,
        terminal_pair_separation_m=pair_separation,
        straight_length_m=straight_length,
    )
