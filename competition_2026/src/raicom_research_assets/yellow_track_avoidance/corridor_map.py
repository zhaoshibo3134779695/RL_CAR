"""Geometry helpers for the odom-fixed competition corridor map."""

from dataclasses import dataclass
import math
from typing import Sequence, Tuple

import numpy as np


Point2 = Tuple[float, float]


@dataclass(frozen=True)
class CorridorFit:
    tangent: Point2
    left_normal: Point2
    anchor: Point2
    residual_m: float


@dataclass(frozen=True)
class CorridorSegments:
    left: Tuple[Point2, Point2]
    right: Tuple[Point2, Point2]
    center: Tuple[Point2, Point2]


@dataclass(frozen=True)
class CorridorStopGoal:
    reference: Point2
    front_center: Point2
    base_center: Point2
    yaw: float


def blend_heading_reference(
    fit: CorridorFit,
    heading_rad: float,
    reference_weight: float = 0.65,
    maximum_disagreement_rad: float = math.radians(3.0),
    maximum_correction_rad: float = math.radians(1.0),
) -> Tuple[CorridorFit, float, bool]:
    """Use a parallel wall as a bounded reference, never as sole authority.

    The extended visual line remains the primary measurement and retains its
    lateral anchor.  A plausible lidar wall may pull its heading part-way;
    a large disagreement is rejected, and even an accepted correction is
    capped so one poor wall fit cannot rotate the complete corridor.
    """
    if not math.isfinite(float(heading_rad)):
        raise ValueError("heading reference must be finite")
    weight = float(reference_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("reference weight must lie in [0, 1]")
    visual_heading = math.atan2(fit.tangent[1], fit.tangent[0])
    difference = math.atan2(
        math.sin(float(heading_rad) - visual_heading),
        math.cos(float(heading_rad) - visual_heading),
    )
    accepted = abs(difference) <= float(maximum_disagreement_rad)
    if accepted:
        correction = max(
            -float(maximum_correction_rad),
            min(float(maximum_correction_rad), weight * difference),
        )
    else:
        correction = 0.0
    fused_heading = visual_heading + correction
    tangent = (math.cos(fused_heading), math.sin(fused_heading))
    normal = (-tangent[1], tangent[0])
    return (
        CorridorFit(
            tangent=tangent,
            left_normal=normal,
            anchor=fit.anchor,
            residual_m=fit.residual_m,
        ),
        correction,
        accepted,
    )


def fit_left_boundary(
    points: Sequence[Point2],
    forward_hint: Point2,
    left_hint: Point2,
) -> CorridorFit:
    """Fit one straight left boundary while preserving vehicle orientation."""
    values = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    values = values[np.all(np.isfinite(values), axis=1)]
    if len(values) < 2:
        raise ValueError("at least two finite boundary points are required")

    center = np.median(values, axis=0)
    centered = values - center
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    tangent = axes[0]
    forward = np.asarray(forward_hint, dtype=np.float64)
    left = np.asarray(left_hint, dtype=np.float64)
    if np.linalg.norm(forward) <= 1e-9 or np.linalg.norm(left) <= 1e-9:
        raise ValueError("orientation hints must be non-zero")
    if float(np.dot(tangent, forward)) < 0.0:
        tangent = -tangent
    tangent /= np.linalg.norm(tangent)
    normal = np.asarray((-tangent[1], tangent[0]), dtype=np.float64)
    if float(np.dot(normal, left)) < 0.0:
        # A valid rigid transform keeps vehicle-left on the tangent's left.
        # This fallback protects tests and malformed external TF sources.
        normal = -normal
    residuals = centered @ normal
    residual = float(np.sqrt(np.mean(np.square(residuals))))
    return CorridorFit(
        tangent=(float(tangent[0]), float(tangent[1])),
        left_normal=(float(normal[0]), float(normal[1])),
        anchor=(float(center[0]), float(center[1])),
        residual_m=residual,
    )


def build_corridor_segments(
    fit: CorridorFit,
    robot_xy: Point2,
    center_separation_m: float,
    vehicle_front_offset_m: float,
    corridor_length_m: float,
) -> CorridorSegments:
    """Build a front-bumper-referenced corridor and infer its right boundary."""
    if center_separation_m <= 0.0:
        raise ValueError("center separation must be positive")
    if vehicle_front_offset_m < 0.0 or corridor_length_m <= 0.0:
        raise ValueError("invalid front offset or corridor length")

    tangent = np.asarray(fit.tangent, dtype=np.float64)
    normal = np.asarray(fit.left_normal, dtype=np.float64)
    robot = np.asarray(robot_xy, dtype=np.float64)
    anchor = np.asarray(fit.anchor, dtype=np.float64)
    # Project base_link onto the fitted left line. The physical segment begins
    # at the vehicle's front edge and has the measured field length, so neither
    # GlobalPlanner nor DWA sees an invented continuation past the bend.
    line_origin = robot + normal * float(np.dot(anchor - robot, normal))
    left_start = line_origin + tangent * float(vehicle_front_offset_m)
    left_end = left_start + tangent * float(corridor_length_m)
    right_shift = normal * float(center_separation_m)
    right_start = left_start - right_shift
    right_end = left_end - right_shift
    center_shift = 0.5 * right_shift
    center_start = left_start - center_shift
    center_end = left_end - center_shift

    def pair(first, second):
        return (
            (float(first[0]), float(first[1])),
            (float(second[0]), float(second[1])),
        )

    return CorridorSegments(
        left=pair(left_start, left_end),
        right=pair(right_start, right_end),
        center=pair(center_start, center_end),
    )


def extend_line_segment(segment, before_m=0.0, after_m=0.0):
    """Extend a finite segment without changing its line or direction."""
    values = (before_m, after_m)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("segment extensions must be finite")
    if before_m < 0.0 or after_m < 0.0:
        raise ValueError("segment extensions must be non-negative")

    start = np.asarray(segment[0], dtype=np.float64)
    end = np.asarray(segment[1], dtype=np.float64)
    delta = end - start
    length = float(np.linalg.norm(delta))
    if length <= 1e-9:
        raise ValueError("line segment must have non-zero length")
    tangent = delta / length
    extended_start = start - tangent * float(before_m)
    extended_end = end + tangent * float(after_m)
    return (
        (float(extended_start[0]), float(extended_start[1])),
        (float(extended_end[0]), float(extended_end[1])),
    )


def build_corridor_stop_goal(
    segments: CorridorSegments,
    left_offset_m: float,
    forward_offset_m: float,
    vehicle_front_offset_m: float,
) -> CorridorStopGoal:
    """Place the vehicle front center relative to the far-right endpoint."""
    values = (left_offset_m, forward_offset_m, vehicle_front_offset_m)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("goal offsets must be finite")
    if vehicle_front_offset_m < 0.0:
        raise ValueError("vehicle front offset must be non-negative")

    reference = np.asarray(segments.right[1], dtype=np.float64)
    corridor_delta = np.asarray(segments.right[1], dtype=np.float64) - np.asarray(
        segments.right[0], dtype=np.float64
    )
    corridor_length = float(np.linalg.norm(corridor_delta))
    if corridor_length <= 1e-9:
        raise ValueError("right boundary segment must have non-zero length")
    tangent = corridor_delta / corridor_length
    left_normal = np.asarray((-tangent[1], tangent[0]), dtype=np.float64)

    front_center = (
        reference
        + tangent * float(forward_offset_m)
        + left_normal * float(left_offset_m)
    )
    base_center = front_center - tangent * float(vehicle_front_offset_m)
    return CorridorStopGoal(
        reference=(float(reference[0]), float(reference[1])),
        front_center=(float(front_center[0]), float(front_center[1])),
        base_center=(float(base_center[0]), float(base_center[1])),
        yaw=math.atan2(float(tangent[1]), float(tangent[0])),
    )


def sample_line_strip(
    segment: Tuple[Point2, Point2],
    width_m: float,
    resolution_m: float,
) -> np.ndarray:
    """Rasterize a finite line strip into XYZ points for costmap_2d."""
    if width_m <= 0.0 or resolution_m <= 0.0:
        raise ValueError("line width and resolution must be positive")
    start = np.asarray(segment[0], dtype=np.float64)
    end = np.asarray(segment[1], dtype=np.float64)
    delta = end - start
    length = float(np.linalg.norm(delta))
    if length <= 1e-9:
        raise ValueError("line segment must have non-zero length")
    tangent = delta / length
    normal = np.asarray((-tangent[1], tangent[0]), dtype=np.float64)
    along_count = max(2, int(math.ceil(length / resolution_m)) + 1)
    across_count = max(2, int(math.ceil(width_m / resolution_m)) + 1)
    along = np.linspace(0.0, length, along_count)
    across = np.linspace(-0.5 * width_m, 0.5 * width_m, across_count)
    points = (
        start[None, None, :]
        + along[:, None, None] * tangent[None, None, :]
        + across[None, :, None] * normal[None, None, :]
    ).reshape(-1, 2)
    return np.column_stack((points, np.zeros(len(points), dtype=np.float64)))


def angular_difference(first: Point2, second: Point2) -> float:
    """Unsigned angle between two unit direction vectors."""
    dot = max(-1.0, min(1.0, first[0] * second[0] + first[1] * second[1]))
    cross = first[0] * second[1] - first[1] * second[0]
    return abs(math.atan2(cross, dot))
