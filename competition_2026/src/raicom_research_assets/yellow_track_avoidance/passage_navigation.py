"""Deterministic local-passage planning and control without ROS dependencies.

The decision layer classifies geometry rather than detection count:

* normal navigation when no fresh disc intersects the route corridor;
* a smooth holonomic bypass for one or more non-gate obstacles;
* perpendicular-bisector control for a true close two-disc gate;
* a latched stop when no physically valid passage exists.

All positions are expressed in one fixed world frame.  This module deliberately
contains no ROS imports so classification, planning and state transitions can
be tested before hardware is enabled.
"""

from dataclasses import dataclass
import bisect
import math
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .disc_passage import (
    PHASE_TOMBSTONE,
    circle_to_rectangle_clearance,
    cumulative_lengths,
    project_point_to_polyline,
)
from .narrow_gate import (
    DiscTrack,
    GateFrame,
    GatePairDiagnostic,
    NarrowGateConfig,
    RobotPose,
    gate_errors,
    projected_vehicle_width,
    select_gate_candidate,
    wrap_angle,
)


SCENE_NORMAL = "normal"
SCENE_BYPASS = "single_disc"
SCENE_MULTI = "multi_obstacle"
SCENE_GATE = "narrow_gate"
SCENE_BLOCKED = "blocked"

STATE_NORMAL = "normal_nav"
STATE_SINGLE_ARMED = "single_disc_armed"
STATE_MULTI_ARMED = "multi_obstacle_armed"
STATE_GATE_ARMED = "gate_armed"
STATE_TAKEOVER_BRAKE = "takeover_brake"
STATE_SINGLE_AVOID = "single_disc_avoid"
STATE_MULTI_AVOID = "multi_obstacle_avoid"
STATE_GATE_ALIGN = "gate_align"
STATE_GATE_CROSS = "gate_cross"
STATE_GATE_EXIT = "gate_exit"
STATE_EXIT_RESUME = "exit_and_resume"
STATE_RECOVERABLE_HOLD = "recoverable_hold"
STATE_DEGRADED_HOLD = "degraded_hold"
STATE_FAILED_STOP = "failed_stop"


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(float(upper), max(float(lower), float(value)))


def point_in_polygon(
    point: Tuple[float, float],
    polygon: Sequence[Tuple[float, float]],
) -> bool:
    """Return True for points inside or on the edge of a simple polygon."""
    if len(polygon) < 3:
        return False
    px, py = float(point[0]), float(point[1])
    inside = False
    for index, first in enumerate(polygon):
        second = polygon[(index + 1) % len(polygon)]
        x1, y1 = float(first[0]), float(first[1])
        x2, y2 = float(second[0]), float(second[1])
        cross = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)
        if abs(cross) <= 1e-9 and (
            min(x1, x2) - 1e-9 <= px <= max(x1, x2) + 1e-9
            and min(y1, y2) - 1e-9 <= py <= max(y1, y2) + 1e-9
        ):
            return True
        if (y1 > py) != (y2 > py):
            crossing_x = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if px < crossing_x:
                inside = not inside
    return inside


def odom_corridor_cache_valid(
    now_sec: float,
    source_sec: Optional[float],
    cached_robot_xy: Optional[Tuple[float, float]],
    robot: Optional[RobotPose],
    maximum_age_sec: float,
    maximum_propagation_sec: float,
    maximum_distance_m: float,
) -> bool:
    """Bound stale vision by both source age and odometric displacement."""
    if source_sec is None or cached_robot_xy is None or robot is None:
        return False
    age_sec = max(0.0, float(now_sec) - float(source_sec))
    if age_sec > min(float(maximum_age_sec), float(maximum_propagation_sec)):
        return False
    return math.hypot(
        float(robot.x) - float(cached_robot_xy[0]),
        float(robot.y) - float(cached_robot_xy[1]),
    ) <= float(maximum_distance_m)


def _dot(first: Tuple[float, float], second: Tuple[float, float]) -> float:
    return first[0] * second[0] + first[1] * second[1]


def _sub(
    first: Tuple[float, float], second: Tuple[float, float]
) -> Tuple[float, float]:
    return first[0] - second[0], first[1] - second[1]


def _norm(vector: Tuple[float, float]) -> float:
    return math.hypot(vector[0], vector[1])


def _normalize(vector: Tuple[float, float]) -> Tuple[float, float]:
    length = _norm(vector)
    if length <= 1e-9:
        return 1.0, 0.0
    return vector[0] / length, vector[1] / length


def _smooth5(value: float) -> float:
    tau = _clamp(value, 0.0, 1.0)
    return tau ** 3 * (10.0 + tau * (-15.0 + 6.0 * tau))


def _interpolate(
    x_values: Sequence[float],
    y_values: Sequence[float],
    query: float,
) -> Optional[float]:
    if not x_values or len(x_values) != len(y_values):
        return None
    if query < x_values[0] or query > x_values[-1]:
        return None
    index = bisect.bisect_right(x_values, query) - 1
    index = min(max(index, 0), len(x_values) - 1)
    if index >= len(x_values) - 1:
        return float(y_values[-1])
    denominator = float(x_values[index + 1]) - float(x_values[index])
    if denominator <= 1e-9:
        return float(y_values[index])
    fraction = (float(query) - float(x_values[index])) / denominator
    return float(y_values[index]) + fraction * (
        float(y_values[index + 1]) - float(y_values[index])
    )


def _point_at_s(
    path: Sequence[Tuple[float, float]],
    cumulative_s: Sequence[float],
    progress_m: float,
) -> Tuple[float, float]:
    if len(path) < 2 or len(cumulative_s) != len(path):
        raise ValueError("reference path requires at least two points")
    progress = _clamp(progress_m, cumulative_s[0], cumulative_s[-1])
    index = bisect.bisect_right(cumulative_s, progress) - 1
    index = min(max(index, 0), len(path) - 2)
    length = cumulative_s[index + 1] - cumulative_s[index]
    fraction = 0.0 if length <= 1e-9 else (
        progress - cumulative_s[index]
    ) / length
    first = path[index]
    second = path[index + 1]
    return (
        first[0] + fraction * (second[0] - first[0]),
        first[1] + fraction * (second[1] - first[1]),
    )


def _path_tangent_at_s(
    path: Sequence[Tuple[float, float]],
    cumulative_s: Sequence[float],
    progress_m: float,
) -> Tuple[float, float]:
    progress = _clamp(progress_m, cumulative_s[0], cumulative_s[-1])
    index = bisect.bisect_right(cumulative_s, progress) - 1
    index = min(max(index, 0), len(path) - 2)
    return _normalize(_sub(path[index + 1], path[index]))


def _signed_lateral(
    point: Tuple[float, float],
    path: Sequence[Tuple[float, float]],
    cumulative_s: Sequence[float],
    maximum_distance_m: float = math.inf,
) -> Tuple[Optional[float], object]:
    projection = project_point_to_polyline(
        point,
        path,
        cumulative_s,
        max_distance_m=maximum_distance_m,
    )
    if not projection.valid:
        return None, projection
    path_point = _point_at_s(path, cumulative_s, projection.s)
    normal = (-projection.tangent_y, projection.tangent_x)
    return _dot(_sub(point, path_point), normal), projection


def route_tangent_at_point(
    point: Tuple[float, float],
    path: Sequence[Tuple[float, float]],
    maximum_distance_m: float,
) -> Optional[Tuple[float, float]]:
    """Return the SBPL path tangent nearest a fixed-frame point."""
    if len(path) < 2:
        return None
    projection = project_point_to_polyline(
        point,
        path,
        cumulative_lengths(path),
        max_distance_m=float(maximum_distance_m),
    )
    if not projection.valid:
        return None
    return projection.tangent_x, projection.tangent_y


@dataclass(frozen=True)
class BoundaryLines:
    left_world: Tuple[Tuple[float, float], ...] = ()
    right_world: Tuple[Tuple[float, float], ...] = ()
    yellow_line_width_m: float = 0.030
    quality: float = 0.0
    fresh: bool = False

    @property
    def valid(self) -> bool:
        return (
            self.fresh
            and self.quality > 0.0
            and len(self.left_world) >= 2
            and len(self.right_world) >= 2
        )


def infer_parallel_boundary(
    points: Sequence[Tuple[float, float]],
    lateral_offset_m: float,
) -> Tuple[Tuple[float, float], ...]:
    """Shift a local-frame line; positive offset is toward vehicle-left."""
    source = tuple((float(x), float(y)) for x, y in points)
    if len(source) < 2:
        return ()
    direction = _sub(source[-1], source[0])
    length = _norm(direction)
    if length <= 1e-9:
        return ()
    tangent = direction[0] / length, direction[1] / length
    left_normal = -tangent[1], tangent[0]
    if left_normal[1] < 0.0:
        left_normal = -left_normal[0], -left_normal[1]
    shift = (
        left_normal[0] * float(lateral_offset_m),
        left_normal[1] * float(lateral_offset_m),
    )
    return tuple(
        (point[0] + shift[0], point[1] + shift[1])
        for point in source
    )


@dataclass
class PassageConfig:
    vehicle_half_length_m: float = 0.165
    vehicle_half_width_m: float = 0.150
    disc_radius_m: float = 0.050

    decision_horizon_m: float = 1.20
    decision_min_forward_m: float = 0.10
    bypass_takeover_distance_m: float = 1.05
    gate_takeover_distance_m: float = 0.70
    track_fresh_sec: float = 2.00
    gate_visual_update_max_age_sec: float = 0.60
    locked_track_drift_stop_m: float = 0.030
    scene_confirm_sec: float = 0.50
    scene_release_sec: float = 0.80
    scene_position_variation_max_m: float = 0.015
    scene_progress_variation_max_m: float = 0.030
    scene_angle_variation_max_rad: float = math.radians(1.5)
    route_projection_max_distance_m: float = 0.40

    line_clearance_margin_m: float = 0.030
    disc_clearance_margin_m: float = 0.030
    geometry_uncertainty_m: float = 0.005
    minimum_residual_clearance_m: float = 0.020
    minimum_predicted_gate_clearance_m: float = 0.020
    emergency_direct_clearance_m: float = 0.020
    emergency_line_clearance_m: float = 0.010
    boundary_start_extrapolation_m: float = 0.45
    boundary_end_extrapolation_m: float = 0.30
    enable_gate_special_mode: bool = True
    claim_all_road_discs: bool = False
    force_between_disc_pair: bool = False
    competition_two_disc_mode: bool = False

    swept_prediction_horizon_sec: float = 0.35
    swept_translation_step_m: float = 0.005
    swept_rotation_step_rad: float = math.radians(1.0)

    side_choice_min_advantage_m: float = 0.040
    side_switch_required_advantage_m: float = 0.060
    side_invalid_confirm_sec: float = 0.30
    minimum_shift_start_distance_m: float = 0.40
    late_detection_stop_distance_m: float = 0.30

    single_shift_length_m: float = 0.45
    single_hold_after_disc_m: float = 0.30
    single_merge_length_m: float = 0.45
    passage_sample_step_m: float = 0.05
    passage_lateral_grid_step_m: float = 0.01
    maximum_lateral_slope: float = 0.60

    bypass_speed_mps: float = 0.050
    gate_approach_speed_mps: float = 0.080
    gate_speed_mps: float = 0.030
    gate_exit_speed_mps: float = 0.050
    lateral_gain: float = 1.00
    yaw_gain: float = 1.50
    maximum_lateral_speed_mps: float = 0.025
    maximum_angular_speed_radps: float = 0.10
    stop_lateral_error_m: float = 0.080
    stop_yaw_error_rad: float = math.radians(15.0)
    completion_lateral_tolerance_m: float = 0.025

    gate_align_distance_m: float = 0.50
    gate_align_lateral_tolerance_m: float = 0.005
    gate_align_yaw_tolerance_rad: float = math.radians(1.5)
    gate_align_hold_sec: float = 0.30
    gate_max_align_speed_mps: float = 0.020
    gate_lateral_abort_m: float = 0.15
    gate_yaw_abort_rad: float = math.radians(45.0)
    gate_exit_distance_m: float = 0.35
    gate_exit_clearance_margin_m: float = 0.05

    handoff_linear_speed_max_mps: float = 0.010
    handoff_angular_speed_max_radps: float = 0.030
    handoff_stop_hold_sec: float = 0.25
    degraded_hold_sec: float = 0.40
    exit_stop_hold_sec: float = 0.30
    takeover_ack_timeout_sec: float = 0.50
    special_control_max_sec: float = 20.0
    special_no_progress_timeout_sec: float = 1.20
    special_min_progress_m: float = 0.015
    special_min_yaw_progress_rad: float = math.radians(0.5)
    handoff_stop_timeout_sec: float = 1.50
    approach_align_timeout_sec: float = 10.0
    cross_exit_timeout_sec: float = 22.0
    competition_overall_timeout_sec: float = 30.0
    special_retry_cooldown_sec: float = 2.0
    special_max_attempts: int = 2
    competition_finish_priority: bool = False

    acceleration_x_mps2: float = 0.10
    acceleration_y_mps2: float = 0.08
    acceleration_yaw_radps2: float = 0.25

    gate_special_mode_max_gap_m: float = 0.45
    gate_release_gap_m: float = 0.48
    gate_longitudinal_mismatch_max_m: float = 0.25
    gate_midpoint_path_distance_max_m: float = 0.12
    gate_normal_angle_tolerance_rad: float = math.radians(30.0)

    def validate(self) -> None:
        positive = {
            name: value
            for name, value in vars(self).items()
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
            )
        }
        invalid = [
            name
            for name, value in positive.items()
            if name not in ("side_choice_min_advantage_m",)
            and float(value) <= 0.0
        ]
        if invalid:
            raise ValueError(
                "passage parameters must be positive: "
                + ", ".join(sorted(invalid))
            )
        if self.special_max_attempts < 1:
            raise ValueError("special attempt count must be positive")
        if self.gate_release_gap_m <= self.gate_special_mode_max_gap_m:
            raise ValueError("gate release gap must exceed trigger gap")
        if self.bypass_takeover_distance_m > self.decision_horizon_m:
            raise ValueError("bypass takeover must be inside decision horizon")
        if self.gate_takeover_distance_m > self.decision_horizon_m:
            raise ValueError("gate takeover must be inside decision horizon")
        if (
            self.passage_lateral_grid_step_m
            > self.passage_sample_step_m
            * self.maximum_lateral_slope
        ):
            raise ValueError(
                "lateral grid is coarser than one reachable sample step"
            )
        if (
            self.late_detection_stop_distance_m
            >= self.minimum_shift_start_distance_m
        ):
            raise ValueError(
                "late stop distance must be below minimum shift distance"
            )

    def gate_config(self) -> NarrowGateConfig:
        return NarrowGateConfig(
            disc_radius_m=self.disc_radius_m,
            vehicle_half_length_m=self.vehicle_half_length_m,
            vehicle_half_width_m=self.vehicle_half_width_m,
            trigger_edge_gap_m=self.gate_special_mode_max_gap_m,
            release_edge_gap_m=self.gate_release_gap_m,
            pair_confirm_duration_sec=self.scene_confirm_sec,
            pair_midpoint_variation_max_m=0.015,
            pair_distance_variation_max_m=0.015,
            pair_angle_variation_max_rad=math.radians(1.5),
            maximum_candidate_track_age_sec=self.track_fresh_sec,
            midpoint_path_distance_max_m=(
                self.gate_midpoint_path_distance_max_m
            ),
            pair_along_path_mismatch_max_m=(
                self.gate_longitudinal_mismatch_max_m
            ),
            crossing_angle_deviation_max_rad=(
                self.gate_normal_angle_tolerance_rad
            ),
            gate_ahead_max_m=self.decision_horizon_m,
            gate_ahead_min_m=self.decision_min_forward_m,
            prealign_distance_m=self.gate_align_distance_m,
            latest_takeover_progress_m=-self.decision_min_forward_m,
            align_lateral_tolerance_m=(
                self.gate_align_lateral_tolerance_m
            ),
            align_yaw_tolerance_rad=self.gate_align_yaw_tolerance_rad,
            align_hold_duration_sec=self.gate_align_hold_sec,
            alignment_lateral_abort_m=self.gate_lateral_abort_m,
            alignment_yaw_abort_rad=self.gate_yaw_abort_rad,
            pass_speed_mps=self.gate_speed_mps,
            exit_speed_mps=self.gate_exit_speed_mps,
            lateral_gain=self.lateral_gain,
            yaw_gain=self.yaw_gain,
            prealign_longitudinal_gain=0.8,
            maximum_lateral_speed_mps=self.maximum_lateral_speed_mps,
            maximum_align_longitudinal_speed_mps=(
                self.gate_approach_speed_mps
                if self.competition_two_disc_mode
                else self.gate_max_align_speed_mps
            ),
            maximum_angular_speed_radps=(
                self.maximum_angular_speed_radps
            ),
            minimum_physical_clearance_m=max(
                0.001,
                self.minimum_predicted_gate_clearance_m
                - self.geometry_uncertainty_m,
            ),
            geometry_uncertainty_m=self.geometry_uncertainty_m,
            emergency_direct_clearance_m=(
                self.emergency_direct_clearance_m
            ),
            exit_clearance_margin_m=self.gate_exit_clearance_margin_m,
            exit_distance_m=self.gate_exit_distance_m,
            stop_hold_duration_sec=self.exit_stop_hold_sec,
            acceleration_x_mps2=self.acceleration_x_mps2,
            acceleration_y_mps2=self.acceleration_y_mps2,
            acceleration_yaw_radps2=self.acceleration_yaw_radps2,
        )


@dataclass(frozen=True)
class ProjectedDisc:
    track: DiscTrack
    path_s_m: float
    lateral_m: float
    ahead_m: float


@dataclass(frozen=True)
class BypassPlan:
    scene: str
    obstacle_ids: Tuple[int, ...]
    reference_s_m: Tuple[float, ...]
    reference_d_m: Tuple[float, ...]
    selected_side: int
    left_residual_m: float
    right_residual_m: float
    minimum_channel_width_m: float
    start_s_m: float
    end_s_m: float

    def lateral_at(self, progress_m: float) -> float:
        value = _interpolate(
            self.reference_s_m, self.reference_d_m, progress_m
        )
        if value is None:
            return (
                self.reference_d_m[0]
                if progress_m < self.reference_s_m[0]
                else self.reference_d_m[-1]
            )
        return value

    def slope_at(self, progress_m: float) -> float:
        if len(self.reference_s_m) < 2:
            return 0.0
        index = bisect.bisect_right(
            self.reference_s_m, progress_m
        ) - 1
        index = min(max(index, 0), len(self.reference_s_m) - 2)
        delta_s = (
            self.reference_s_m[index + 1]
            - self.reference_s_m[index]
        )
        if delta_s <= 1e-9:
            return 0.0
        return (
            self.reference_d_m[index + 1]
            - self.reference_d_m[index]
        ) / delta_s


def _plan_has_exact_disc_clearance(
    plan: BypassPlan,
    obstacles: Sequence[ProjectedDisc],
    path: Sequence[Tuple[float, float]],
    cumulative_s: Sequence[float],
    config: PassageConfig,
) -> bool:
    required = (
        config.disc_clearance_margin_m
        + config.geometry_uncertainty_m
    )
    for progress, lateral in zip(
        plan.reference_s_m, plan.reference_d_m
    ):
        centerline = _point_at_s(path, cumulative_s, progress)
        tangent = _path_tangent_at_s(path, cumulative_s, progress)
        normal = -tangent[1], tangent[0]
        center = (
            centerline[0] + lateral * normal[0],
            centerline[1] + lateral * normal[1],
        )
        for obstacle in obstacles:
            delta = _sub(obstacle.track.fixed_xy, center)
            longitudinal = _dot(delta, tangent)
            cross_track = _dot(delta, normal)
            clearance = circle_to_rectangle_clearance(
                longitudinal,
                cross_track,
                config.vehicle_half_length_m,
                config.vehicle_half_width_m,
                config.disc_radius_m,
            )
            if clearance + 1e-9 < required:
                return False
    return True


@dataclass(frozen=True)
class SceneAssessment:
    scene: str
    key: Tuple[object, ...]
    reason: str
    nearest_ahead_m: float = math.inf
    affecting_ids: Tuple[int, ...] = ()
    bypass_plan: Optional[BypassPlan] = None
    gate_frame: Optional[GateFrame] = None
    projected_discs: Tuple[ProjectedDisc, ...] = ()
    admission_safety: Optional["SweptSafety"] = None
    gate_pair_diagnostics: Tuple[GatePairDiagnostic, ...] = ()


def passage_attempt_key(
    logical_goal_epoch: int,
    assessment: Optional[SceneAssessment],
) -> Optional[Tuple[object, ...]]:
    """Identify one geometric passage attempt within one external goal."""
    if assessment is None or not assessment.key:
        return None
    return (int(logical_goal_epoch),) + tuple(assessment.key)


def register_passage_attempt(
    attempted: Set[Tuple[object, ...]],
    key: Optional[Tuple[object, ...]],
) -> bool:
    """Register the first attempt; return True for a repeated takeover."""
    if key is None:
        return False
    if key in attempted:
        return True
    attempted.add(key)
    return False


@dataclass(frozen=True)
class PassageErrors:
    lateral_m: float
    yaw_rad: float
    progress_m: float
    reference_lateral_m: float
    minimum_direct_clearance_m: float


@dataclass(frozen=True)
class PassageDecision:
    state: str
    scene: str
    authority_requested: bool
    command_body: Tuple[float, float, float]
    reason: str
    cancel_move_base: bool = False
    resume_move_base: bool = False
    assessment: Optional[SceneAssessment] = None
    bypass_plan: Optional[BypassPlan] = None
    gate_frame: Optional[GateFrame] = None
    errors: Optional[PassageErrors] = None
    gate_diagnostics: Optional[object] = None
    scene_stable_sec: float = 0.0
    swept_safety: Optional["SweptSafety"] = None


def _boundary_samples(
    points: Sequence[Tuple[float, float]],
    path: Sequence[Tuple[float, float]],
    cumulative_s: Sequence[float],
) -> Tuple[Tuple[float, float], ...]:
    values = []
    for point in points:
        lateral, projection = _signed_lateral(
            point, path, cumulative_s, maximum_distance_m=1.0
        )
        if lateral is None:
            continue
        values.append((projection.s, lateral))
    values.sort(key=lambda item: item[0])
    deduplicated = []
    for progress, lateral in values:
        if deduplicated and abs(progress - deduplicated[-1][0]) < 1e-4:
            deduplicated[-1] = (progress, lateral)
        else:
            deduplicated.append((progress, lateral))
    return tuple(deduplicated)


class BoundaryEnvelope:
    def __init__(
        self,
        boundary: BoundaryLines,
        path: Sequence[Tuple[float, float]],
        cumulative_s: Sequence[float],
    ):
        self.boundary = boundary
        self.left = _boundary_samples(
            boundary.left_world, path, cumulative_s
        )
        self.right = _boundary_samples(
            boundary.right_world, path, cumulative_s
        )

    @property
    def valid(self) -> bool:
        return (
            self.boundary.valid
            and len(self.left) >= 2
            and len(self.right) >= 2
        )

    def limits_at(self, progress_m: float) -> Optional[Tuple[float, float]]:
        if not self.valid:
            return None
        query = float(progress_m)
        first_s = max(self.left[0][0], self.right[0][0])
        last_s = min(self.left[-1][0], self.right[-1][0])
        if query < first_s:
            query = first_s
        elif query > last_s:
            query = last_s
        left = _interpolate(
            tuple(item[0] for item in self.left),
            tuple(item[1] for item in self.left),
            query,
        )
        right = _interpolate(
            tuple(item[0] for item in self.right),
            tuple(item[1] for item in self.right),
            query,
        )
        if left is None or right is None or left <= right:
            return None
        return left, right


@dataclass(frozen=True)
class SweptSafety:
    safe: bool
    reason: str
    minimum_disc_clearance_m: float = math.inf
    minimum_line_clearance_m: float = math.inf
    sample_count: int = 0


def robot_boundary_clearance(
    robot: RobotPose,
    path: Sequence[Tuple[float, float]],
    boundary: BoundaryLines,
    config: PassageConfig,
) -> Optional[float]:
    """Signed clearance from the full rectangle to both yellow inner edges."""
    if len(path) < 2 or not boundary.valid:
        return None
    cumulative_s = cumulative_lengths(path)
    center_d, projection = _signed_lateral(
        robot.xy,
        path,
        cumulative_s,
        maximum_distance_m=config.route_projection_max_distance_m,
    )
    if center_d is None:
        return None
    envelope = BoundaryEnvelope(boundary, path, cumulative_s)
    if not envelope.valid:
        return None
    tangent = _path_tangent_at_s(path, cumulative_s, projection.s)
    path_yaw = math.atan2(tangent[1], tangent[0])
    relative_yaw = wrap_angle(robot.yaw - path_yaw)
    cosine = math.cos(relative_yaw)
    sine = math.sin(relative_yaw)
    half_line = 0.5 * boundary.yellow_line_width_m
    clearances = []
    for longitudinal in (
        -config.vehicle_half_length_m,
        config.vehicle_half_length_m,
    ):
        for lateral in (
            -config.vehicle_half_width_m,
            config.vehicle_half_width_m,
        ):
            progress = projection.s + longitudinal * cosine - lateral * sine
            corner_d = center_d + longitudinal * sine + lateral * cosine
            limits = envelope.limits_at(progress)
            if limits is None:
                return None
            left_line, right_line = limits
            clearances.extend(
                (
                    left_line - half_line - corner_d,
                    corner_d - (right_line + half_line),
                )
            )
    return min(clearances) if clearances else None


def swept_command_safety(
    robot: RobotPose,
    command_body: Tuple[float, float, float],
    path: Sequence[Tuple[float, float]],
    boundary: BoundaryLines,
    tracks: Sequence[DiscTrack],
    config: PassageConfig,
) -> SweptSafety:
    """Predict a short body-twist sweep and reject line or disc contact."""
    if len(path) < 2:
        return SweptSafety(False, "swept_route_unavailable")
    if not boundary.valid:
        return SweptSafety(False, "swept_boundary_unavailable")
    vx, vy, angular = (float(value) for value in command_body)
    horizon = config.swept_prediction_horizon_sec
    steps = max(
        1,
        int(
            math.ceil(
                math.hypot(vx, vy)
                * horizon
                / config.swept_translation_step_m
            )
        ),
        int(
            math.ceil(
                abs(angular)
                * horizon
                / config.swept_rotation_step_rad
            )
        ),
    )
    dt = horizon / float(steps)
    pose = RobotPose(robot.x, robot.y, robot.yaw)
    minimum_disc = math.inf
    minimum_line = math.inf
    active_tracks = tuple(
        track
        for track in tracks
        if (
            track.confirmed
            and track.phase != PHASE_TOMBSTONE
            and track.last_seen_age_sec <= config.track_fresh_sec
        )
    )
    for index in range(steps + 1):
        line_clearance = robot_boundary_clearance(
            pose, path, boundary, config
        )
        if line_clearance is None:
            return SweptSafety(
                False,
                "swept_boundary_unavailable",
                minimum_disc,
                minimum_line,
                index + 1,
            )
        minimum_line = min(minimum_line, line_clearance)
        if minimum_line + 1e-9 < config.emergency_line_clearance_m:
            return SweptSafety(
                False,
                "swept_yellow_clearance",
                minimum_disc,
                minimum_line,
                index + 1,
            )
        cosine = math.cos(pose.yaw)
        sine = math.sin(pose.yaw)
        for track in active_tracks:
            delta = _sub(track.fixed_xy, pose.xy)
            base_x = cosine * delta[0] + sine * delta[1]
            base_y = -sine * delta[0] + cosine * delta[1]
            clearance = circle_to_rectangle_clearance(
                base_x,
                base_y,
                config.vehicle_half_length_m,
                config.vehicle_half_width_m,
                config.disc_radius_m,
            )
            minimum_disc = min(minimum_disc, clearance)
        if minimum_disc + 1e-9 < config.emergency_direct_clearance_m:
            return SweptSafety(
                False,
                "swept_disc_clearance",
                minimum_disc,
                minimum_line,
                index + 1,
            )
        if index == steps:
            break
        world_x = cosine * vx - sine * vy
        world_y = sine * vx + cosine * vy
        pose = RobotPose(
            pose.x + world_x * dt,
            pose.y + world_y * dt,
            wrap_angle(pose.yaw + angular * dt),
        )
    return SweptSafety(
        True,
        "swept_safe",
        minimum_disc,
        minimum_line,
        steps + 1,
    )


def gate_passage_admission_safety(
    robot: RobotPose,
    frame: GateFrame,
    path: Sequence[Tuple[float, float]],
    boundary: BoundaryLines,
    tracks: Sequence[DiscTrack],
    config: PassageConfig,
) -> SweptSafety:
    """Check the full rectangle from the current pose through gate exit."""
    if len(path) < 2:
        return SweptSafety(False, "admission_route_unavailable")
    if not boundary.valid:
        return SweptSafety(False, "admission_boundary_unavailable")
    entry = (
        frame.midpoint_xy[0]
        - config.gate_align_distance_m * frame.travel_xy[0],
        frame.midpoint_xy[1]
        - config.gate_align_distance_m * frame.travel_xy[1],
    )
    exit_xy = (
        frame.midpoint_xy[0]
        + config.gate_exit_distance_m * frame.travel_xy[0],
        frame.midpoint_xy[1]
        + config.gate_exit_distance_m * frame.travel_xy[1],
    )
    waypoints = (
        (robot.xy, robot.yaw),
        (entry, frame.yaw_ref),
        (frame.midpoint_xy, frame.yaw_ref),
        (exit_xy, frame.yaw_ref),
    )
    active_tracks = tuple(
        track
        for track in tracks
        if track.memory_id in frame.pair_ids
    )
    minimum_disc = math.inf
    minimum_line = math.inf
    sample_count = 0
    for segment_index in range(len(waypoints) - 1):
        (start_xy, start_yaw) = waypoints[segment_index]
        (end_xy, end_yaw) = waypoints[segment_index + 1]
        distance = _norm(_sub(end_xy, start_xy))
        yaw_delta = wrap_angle(end_yaw - start_yaw)
        steps = max(
            1,
            int(math.ceil(distance / config.swept_translation_step_m)),
            int(
                math.ceil(
                    abs(yaw_delta) / config.swept_rotation_step_rad
                )
            ),
        )
        for index in range(steps + 1):
            if segment_index > 0 and index == 0:
                continue
            fraction = index / float(steps)
            pose = RobotPose(
                start_xy[0] + fraction * (end_xy[0] - start_xy[0]),
                start_xy[1] + fraction * (end_xy[1] - start_xy[1]),
                wrap_angle(start_yaw + fraction * yaw_delta),
            )
            sample_count += 1
            line_clearance = robot_boundary_clearance(
                pose, path, boundary, config
            )
            if line_clearance is None:
                return SweptSafety(
                    False,
                    "admission_boundary_unavailable",
                    minimum_disc,
                    minimum_line,
                    sample_count,
                )
            minimum_line = min(minimum_line, line_clearance)
            if minimum_line + 1e-9 < config.emergency_line_clearance_m:
                return SweptSafety(
                    False,
                    "admission_yellow_clearance",
                    minimum_disc,
                    minimum_line,
                    sample_count,
                )
            cosine = math.cos(pose.yaw)
            sine = math.sin(pose.yaw)
            for track in active_tracks:
                delta = _sub(track.fixed_xy, pose.xy)
                base_x = cosine * delta[0] + sine * delta[1]
                base_y = -sine * delta[0] + cosine * delta[1]
                clearance = circle_to_rectangle_clearance(
                    base_x,
                    base_y,
                    config.vehicle_half_length_m,
                    config.vehicle_half_width_m,
                    config.disc_radius_m,
                )
                minimum_disc = min(minimum_disc, clearance)
            if (
                minimum_disc + 1e-9
                < config.minimum_predicted_gate_clearance_m
            ):
                return SweptSafety(
                    False,
                    "admission_disc_clearance",
                    minimum_disc,
                    minimum_line,
                    sample_count,
                )
    return SweptSafety(
        True,
        "admission_safe",
        minimum_disc,
        minimum_line,
        sample_count,
    )


def project_discs(
    tracks: Sequence[DiscTrack],
    path: Sequence[Tuple[float, float]],
    robot: RobotPose,
    config: PassageConfig,
) -> Tuple[ProjectedDisc, ...]:
    cumulative_s = cumulative_lengths(path)
    if len(cumulative_s) < 2:
        return ()
    robot_lateral, robot_projection = _signed_lateral(
        robot.xy,
        path,
        cumulative_s,
        maximum_distance_m=config.route_projection_max_distance_m,
    )
    if robot_lateral is None:
        return ()
    values = []
    for track in tracks:
        if (
            not track.confirmed
            or track.phase == PHASE_TOMBSTONE
            or track.last_seen_age_sec > config.track_fresh_sec
        ):
            continue
        lateral, projection = _signed_lateral(
            track.fixed_xy,
            path,
            cumulative_s,
            maximum_distance_m=1.0,
        )
        if lateral is None:
            continue
        ahead = projection.s - robot_projection.s
        if (
            config.decision_min_forward_m
            <= ahead
            <= config.decision_horizon_m
        ):
            values.append(
                ProjectedDisc(track, projection.s, lateral, ahead)
            )
    return tuple(sorted(values, key=lambda item: item.ahead_m))


def _single_side_geometry(
    obstacle: ProjectedDisc,
    envelope: BoundaryEnvelope,
    config: PassageConfig,
) -> Optional[Tuple[float, float, float, float]]:
    limits = envelope.limits_at(obstacle.path_s_m)
    if limits is None:
        return None
    left_line, right_line = limits
    half_line = 0.5 * envelope.boundary.yellow_line_width_m
    left_physical_gap = (
        left_line - half_line
        - (obstacle.lateral_m + config.disc_radius_m)
    )
    right_physical_gap = (
        obstacle.lateral_m - config.disc_radius_m
        - (right_line + half_line)
    )
    fixed_allowance = (
        2.0 * config.vehicle_half_width_m
        + config.line_clearance_margin_m
        + config.disc_clearance_margin_m
        + config.geometry_uncertainty_m
    )
    left_residual = left_physical_gap - fixed_allowance
    right_residual = right_physical_gap - fixed_allowance

    left_lower = (
        obstacle.lateral_m
        + config.disc_radius_m
        + config.vehicle_half_width_m
        + config.disc_clearance_margin_m
        + config.geometry_uncertainty_m
    )
    left_upper = (
        left_line
        - half_line
        - config.vehicle_half_width_m
        - config.line_clearance_margin_m
    )
    right_lower = (
        right_line
        + half_line
        + config.vehicle_half_width_m
        + config.line_clearance_margin_m
    )
    right_upper = (
        obstacle.lateral_m
        - config.disc_radius_m
        - config.vehicle_half_width_m
        - config.disc_clearance_margin_m
        - config.geometry_uncertainty_m
    )
    left_target = 0.5 * (left_lower + left_upper)
    right_target = 0.5 * (right_lower + right_upper)
    return left_residual, right_residual, left_target, right_target


def _boundary_query_valid(
    envelope: BoundaryEnvelope,
    progress_m: float,
    config: PassageConfig,
) -> bool:
    if not envelope.valid:
        return False
    first_s = max(envelope.left[0][0], envelope.right[0][0])
    last_s = min(envelope.left[-1][0], envelope.right[-1][0])
    return (
        first_s - config.boundary_start_extrapolation_m
        <= progress_m
        <= last_s + config.boundary_end_extrapolation_m
    )


def _quintic_single_plan(
    robot_s: float,
    robot_d: float,
    obstacle: ProjectedDisc,
    selected_side: int,
    target_d: float,
    left_residual: float,
    right_residual: float,
    config: PassageConfig,
) -> Optional[BypassPlan]:
    distance = obstacle.path_s_m - robot_s
    if distance < config.late_detection_stop_distance_m:
        return None
    obstacle_entry = (
        obstacle.path_s_m
        - config.vehicle_half_length_m
        - config.disc_radius_m
        - config.disc_clearance_margin_m
    )
    shift_end = obstacle_entry
    available = shift_end - robot_s
    if available < config.minimum_shift_start_distance_m:
        return None
    shift_start = max(robot_s, shift_end - config.single_shift_length_m)
    hold_end = (
        obstacle.path_s_m
        + config.vehicle_half_length_m
        + config.disc_radius_m
        + config.disc_clearance_margin_m
        + config.single_hold_after_disc_m
    )
    merge_end = hold_end + config.single_merge_length_m
    spacing = config.passage_sample_step_m
    count = max(2, int(math.ceil((merge_end - robot_s) / spacing)) + 1)
    samples_s = [
        robot_s + (merge_end - robot_s) * index / (count - 1)
        for index in range(count)
    ]
    samples_d = []
    for progress in samples_s:
        if progress <= shift_start:
            lateral = robot_d
        elif progress < shift_end:
            fraction = (progress - shift_start) / max(
                shift_end - shift_start, 1e-6
            )
            lateral = robot_d + (target_d - robot_d) * _smooth5(fraction)
        elif progress <= hold_end:
            lateral = target_d
        else:
            fraction = (progress - hold_end) / max(
                merge_end - hold_end, 1e-6
            )
            lateral = target_d * (1.0 - _smooth5(fraction))
        samples_d.append(lateral)
    return BypassPlan(
        scene=SCENE_BYPASS,
        obstacle_ids=(obstacle.track.memory_id,),
        reference_s_m=tuple(samples_s),
        reference_d_m=tuple(samples_d),
        selected_side=selected_side,
        left_residual_m=left_residual,
        right_residual_m=right_residual,
        minimum_channel_width_m=max(
            left_residual if selected_side > 0 else right_residual, 0.0
        ),
        start_s_m=robot_s,
        end_s_m=merge_end,
    )


def build_single_bypass_plan(
    obstacle: ProjectedDisc,
    path: Sequence[Tuple[float, float]],
    robot: RobotPose,
    boundary: BoundaryLines,
    config: PassageConfig,
    preferred_side: int = 0,
) -> Tuple[Optional[BypassPlan], str]:
    cumulative_s = cumulative_lengths(path)
    robot_d, robot_projection = _signed_lateral(
        robot.xy,
        path,
        cumulative_s,
        maximum_distance_m=config.route_projection_max_distance_m,
    )
    if robot_d is None:
        return None, "robot_off_reference_route"
    envelope = BoundaryEnvelope(boundary, path, cumulative_s)
    geometry = _single_side_geometry(obstacle, envelope, config)
    if geometry is None:
        return None, "yellow_boundaries_unavailable"
    left_residual, right_residual, left_target, right_target = geometry
    minimum = config.minimum_residual_clearance_m
    left_valid = left_residual >= minimum
    right_valid = right_residual >= minimum
    if not left_valid and not right_valid:
        return None, "single_disc_no_safe_side"
    if preferred_side > 0 and left_valid:
        side = (
            -1
            if (
                right_valid
                and right_residual - left_residual
                >= config.side_switch_required_advantage_m
            )
            else 1
        )
    elif preferred_side < 0 and right_valid:
        side = (
            1
            if (
                left_valid
                and left_residual - right_residual
                >= config.side_switch_required_advantage_m
            )
            else -1
        )
    elif left_valid and not right_valid:
        side = 1
    elif right_valid and not left_valid:
        side = -1
    elif (
        left_residual - right_residual
        >= config.side_choice_min_advantage_m
    ):
        side = 1
    elif (
        right_residual - left_residual
        >= config.side_choice_min_advantage_m
    ):
        side = -1
    else:
        # Equal-space cases choose the smaller lateral displacement.  The
        # caller may supply the previous successful side as preferred_side.
        side = (
            1
            if abs(left_target - robot_d)
            <= abs(right_target - robot_d)
            else -1
        )
    plan_failure_reason = "single_disc_route_generation_failed"
    distance = obstacle.path_s_m - robot_projection.s
    obstacle_entry = (
        obstacle.path_s_m
        - config.vehicle_half_length_m
        - config.disc_radius_m
        - config.disc_clearance_margin_m
    )
    available_shift = obstacle_entry - robot_projection.s
    if distance < config.late_detection_stop_distance_m:
        plan_failure_reason = "single_disc_detected_too_late"
    elif available_shift < config.minimum_shift_start_distance_m:
        plan_failure_reason = "single_disc_insufficient_shift_distance"

    plan = _quintic_single_plan(
        robot_projection.s,
        robot_d,
        obstacle,
        side,
        left_target if side > 0 else right_target,
        left_residual,
        right_residual,
        config,
    )
    if plan is not None:
        half_line = 0.5 * boundary.yellow_line_width_m
        for progress, lateral in zip(
            plan.reference_s_m, plan.reference_d_m
        ):
            if not _boundary_query_valid(envelope, progress, config):
                plan = None
                plan_failure_reason = "single_disc_boundary_uncovered"
                break
            limits = envelope.limits_at(progress)
            if limits is None:
                plan = None
                plan_failure_reason = "single_disc_boundary_invalid"
                break
            left_line, right_line = limits
            lower = (
                right_line
                + half_line
                + config.vehicle_half_width_m
                + config.line_clearance_margin_m
            )
            upper = (
                left_line
                - half_line
                - config.vehicle_half_width_m
                - config.line_clearance_margin_m
            )
            if not lower <= lateral <= upper:
                plan = None
                plan_failure_reason = (
                    "single_disc_yellow_line_clearance_insufficient"
                )
                break
    if (
        plan is not None
        and not _plan_has_exact_disc_clearance(
            plan,
            (obstacle,),
            path,
            cumulative_s,
            config,
        )
    ):
        plan = None
        plan_failure_reason = "single_disc_disc_clearance_insufficient"
    return (
        (plan, "")
        if plan is not None
        else (None, plan_failure_reason)
    )


def _subtract_interval(
    intervals: Sequence[Tuple[float, float]],
    occupied: Tuple[float, float],
) -> List[Tuple[float, float]]:
    output = []
    for lower, upper in intervals:
        if occupied[1] <= lower or occupied[0] >= upper:
            output.append((lower, upper))
            continue
        if occupied[0] > lower:
            output.append((lower, min(upper, occupied[0])))
        if occupied[1] < upper:
            output.append((max(lower, occupied[1]), upper))
    return output


@dataclass
class _ReachableNode:
    value: float
    lower: float
    upper: float
    parent: Optional[int]
    cost: float


def _interval_lateral_samples(
    lower: float,
    upper: float,
    spacing: float,
) -> Tuple[float, ...]:
    if upper < lower:
        return ()
    values = [lower, upper, 0.5 * (lower + upper)]
    first_index = int(math.ceil(lower / spacing))
    last_index = int(math.floor(upper / spacing))
    values.extend(
        index * spacing
        for index in range(first_index, last_index + 1)
    )
    if lower <= 0.0 <= upper:
        values.append(0.0)
    return tuple(
        sorted(
            {
                round(_clamp(value, lower, upper), 6)
                for value in values
            }
        )
    )


def build_multi_channel_plan(
    obstacles: Sequence[ProjectedDisc],
    path: Sequence[Tuple[float, float]],
    robot: RobotPose,
    boundary: BoundaryLines,
    config: PassageConfig,
    preferred_side: int = 0,
    force_between_pair: bool = False,
    diagnostics: Optional[Dict[str, object]] = None,
) -> Tuple[Optional[BypassPlan], str]:
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics["planner"] = "free_interval_dynamic_programming"
        diagnostics["force_between_pair"] = bool(force_between_pair)

    def fail(reason: str) -> Tuple[Optional[BypassPlan], str]:
        if diagnostics is not None:
            diagnostics["failure_reason"] = str(reason)
        return None, reason

    cumulative_s = cumulative_lengths(path)
    robot_d, robot_projection = _signed_lateral(
        robot.xy,
        path,
        cumulative_s,
        maximum_distance_m=config.route_projection_max_distance_m,
    )
    if robot_d is None:
        return fail("robot_off_reference_route")
    if not obstacles:
        return fail("multi_obstacles_missing")
    nearest = min(item.ahead_m for item in obstacles)
    if nearest < config.late_detection_stop_distance_m:
        return fail("multi_obstacle_detected_too_late")
    envelope = BoundaryEnvelope(boundary, path, cumulative_s)
    if not envelope.valid:
        return fail("yellow_boundaries_unavailable")

    end_s = min(
        cumulative_s[-1],
        max(item.path_s_m for item in obstacles)
        + config.vehicle_half_length_m
        + config.disc_radius_m
        + config.disc_clearance_margin_m
        + config.single_hold_after_disc_m
        + config.single_merge_length_m,
    )
    spacing = config.passage_sample_step_m
    count = max(
        2,
        int(math.ceil((end_s - robot_projection.s) / spacing)) + 1,
    )
    samples_s = [
        robot_projection.s
        + (end_s - robot_projection.s) * index / (count - 1)
        for index in range(count)
    ]
    free_by_sample: List[List[Tuple[float, float]]] = []
    half_line = 0.5 * boundary.yellow_line_width_m
    obstacle_half_width = (
        config.vehicle_half_width_m
        + config.disc_radius_m
        + config.disc_clearance_margin_m
        + config.geometry_uncertainty_m
    )
    longitudinal_before = (
        config.vehicle_half_length_m
        + config.disc_radius_m
        + config.disc_clearance_margin_m
    )
    longitudinal_after = (
        longitudinal_before + config.single_hold_after_disc_m
    )
    between_pair = (
        tuple(sorted(obstacles, key=lambda item: item.lateral_m))
        if force_between_pair and len(obstacles) == 2
        else ()
    )
    for progress in samples_s:
        if not _boundary_query_valid(envelope, progress, config):
            return fail("yellow_boundary_range_too_short")
        limits = envelope.limits_at(progress)
        if limits is None:
            return fail("yellow_boundary_range_too_short")
        left_line, right_line = limits
        intervals = [
            (
                right_line
                + half_line
                + config.vehicle_half_width_m
                + config.line_clearance_margin_m,
                left_line
                - half_line
                - config.vehicle_half_width_m
                - config.line_clearance_margin_m,
            )
        ]
        for obstacle in obstacles:
            if (
                obstacle.path_s_m - longitudinal_before
                <= progress
                <= obstacle.path_s_m + longitudinal_after
            ):
                intervals = _subtract_interval(
                    intervals,
                    (
                        obstacle.lateral_m - obstacle_half_width,
                        obstacle.lateral_m + obstacle_half_width,
                    ),
                )
                if between_pair:
                    if obstacle is between_pair[0]:
                        inside_edge = (
                            obstacle.lateral_m + obstacle_half_width
                        )
                        intervals = [
                            (max(lower, inside_edge), upper)
                            for lower, upper in intervals
                            if upper >= inside_edge
                        ]
                    else:
                        inside_edge = (
                            obstacle.lateral_m - obstacle_half_width
                        )
                        intervals = [
                            (lower, min(upper, inside_edge))
                            for lower, upper in intervals
                            if lower <= inside_edge
                        ]
        intervals = [
            item
            for item in intervals
            if item[1] - item[0]
            >= config.minimum_residual_clearance_m
        ]
        if not intervals:
            if diagnostics is not None:
                diagnostics["sample_s_m"] = tuple(samples_s)
                diagnostics["free_intervals_m"] = tuple(
                    tuple(value) for value in free_by_sample
                ) + ((),)
                diagnostics["blocked_at_s_m"] = float(progress)
            return fail("multi_obstacle_no_free_interval")
        residual_half = 0.5 * config.minimum_residual_clearance_m
        free_by_sample.append(
            [
                (
                    lower + residual_half,
                    upper - residual_half,
                )
                for lower, upper in intervals
            ]
        )

    if diagnostics is not None:
        diagnostics["sample_s_m"] = tuple(samples_s)
        diagnostics["free_intervals_m"] = tuple(
            tuple(value) for value in free_by_sample
        )

    layers: List[List[_ReachableNode]] = []
    initial_nodes = []
    for lower, upper in free_by_sample[0]:
        if lower <= robot_d <= upper:
            initial_nodes.append(
                _ReachableNode(
                    robot_d,
                    lower,
                    upper,
                    None,
                    abs(robot_d),
                )
            )
    if not initial_nodes:
        return fail("robot_outside_boundary_channel")
    layers.append(initial_nodes)

    delta_s = (end_s - robot_projection.s) / max(count - 1, 1)
    maximum_step = config.maximum_lateral_slope * delta_s
    for sample_index in range(1, len(samples_s)):
        next_nodes = []
        for free_lower, free_upper in free_by_sample[sample_index]:
            for desired in _interval_lateral_samples(
                free_lower,
                free_upper,
                config.passage_lateral_grid_step_m,
            ):
                parents = [
                    (parent_index, parent)
                    for parent_index, parent in enumerate(layers[-1])
                    if abs(desired - parent.value)
                    <= maximum_step + 1e-6
                ]
                if not parents:
                    continue
                centered_clearance = min(
                    desired - free_lower,
                    free_upper - desired,
                )
                side_penalty = (
                    0.05
                    if (
                        preferred_side
                        and desired * preferred_side < -1e-6
                    )
                    else 0.0
                )
                parent_index, parent = min(
                    parents,
                    key=lambda item: (
                        item[1].cost
                        + 0.15
                        * abs(desired - item[1].value)
                    ),
                )
                cost = (
                    parent.cost
                    + 0.15 * abs(desired - parent.value)
                    + 0.003 * abs(desired)
                    - 0.02 * max(0.0, centered_clearance)
                    + side_penalty
                )
                next_nodes.append(
                    _ReachableNode(
                        desired,
                        free_lower,
                        free_upper,
                        parent_index,
                        cost,
                    )
                )
        if not next_nodes:
            if diagnostics is not None:
                diagnostics["candidate_lateral_m"] = tuple(
                    tuple(node.value for node in layer)
                    for layer in layers
                ) + ((),)
                diagnostics["blocked_at_s_m"] = float(
                    samples_s[sample_index]
                )
            return fail("multi_channel_lateral_rate_limit")
        layers.append(next_nodes)

    if diagnostics is not None:
        diagnostics["candidate_lateral_m"] = tuple(
            tuple(node.value for node in layer) for layer in layers
        )

    final_index = min(
        range(len(layers[-1])),
        key=lambda index: (
            layers[-1][index].cost
            + abs(layers[-1][index].value)
        ),
    )
    values = [0.0] * len(layers)
    selected_widths = [0.0] * len(layers)
    for layer_index in range(len(layers) - 1, -1, -1):
        node = layers[layer_index][final_index]
        values[layer_index] = node.value
        selected_widths[layer_index] = node.upper - node.lower
        if node.parent is None:
            break
        final_index = node.parent

    primary = obstacles[0]
    primary_index = min(
        range(len(samples_s)),
        key=lambda index: abs(samples_s[index] - primary.path_s_m),
    )
    side = 1 if values[primary_index] >= primary.lateral_m else -1
    geometry = _single_side_geometry(primary, envelope, config)
    left_residual = -math.inf if geometry is None else geometry[0]
    right_residual = -math.inf if geometry is None else geometry[1]
    minimum_width = (
        min(selected_widths)
        + config.minimum_residual_clearance_m
    )
    plan = BypassPlan(
        scene=SCENE_MULTI,
        obstacle_ids=tuple(
            sorted(item.track.memory_id for item in obstacles)
        ),
        reference_s_m=tuple(samples_s),
        reference_d_m=tuple(values),
        selected_side=side,
        left_residual_m=left_residual,
        right_residual_m=right_residual,
        minimum_channel_width_m=minimum_width,
        start_s_m=robot_projection.s,
        end_s_m=end_s,
    )
    if not _plan_has_exact_disc_clearance(
        plan,
        obstacles,
        path,
        cumulative_s,
        config,
    ):
        return fail("multi_channel_exact_footprint_collision")
    if diagnostics is not None:
        diagnostics["selected_lateral_m"] = tuple(values)
        diagnostics["selected_channel_width_m"] = tuple(
            width + config.minimum_residual_clearance_m
            for width in selected_widths
        )
        diagnostics["failure_reason"] = ""
    return plan, ""


def assess_competition_pair(
    tracks: Sequence[DiscTrack],
    path: Sequence[Tuple[float, float]],
    robot: RobotPose,
    boundary: BoundaryLines,
    config: PassageConfig,
) -> SceneAssessment:
    """Admit the best stable left/right pair used in competition."""
    if len(path) < 2:
        return SceneAssessment(
            SCENE_NORMAL, ("normal",), "competition_route_unavailable"
        )
    usable = tuple(
        track
        for track in tracks
        if (
            track.confirmed
            and track.phase != PHASE_TOMBSTONE
            and track.last_seen_age_sec <= config.track_fresh_sec
        )
    )
    projected = project_discs(usable, path, robot, config)
    if len(usable) < 2:
        return SceneAssessment(
            SCENE_NORMAL,
            ("competition_pair_wait", len(usable)),
            "competition_waiting_for_exact_pair",
            projected_discs=projected,
        )
    # A restarted image tracker can leave a duplicate world-fixed memory.
    # Do not let one extra confirmed track disable the whole competition
    # controller. select_gate_candidate still requires the chosen pair to
    # straddle the active path, match its normal, fit the metric gap and lie
    # within the takeover horizon; admission below checks the full sweep.
    pair_diagnostics = []
    candidate = select_gate_candidate(
        usable,
        path,
        robot,
        config.gate_config(),
        maximum_edge_gap_m=config.gate_special_mode_max_gap_m,
        pair_diagnostics=pair_diagnostics,
    )
    if candidate is None:
        return SceneAssessment(
            SCENE_NORMAL,
            ("competition_pair_rejected",)
            + tuple(sorted(track.memory_id for track in usable)),
            "competition_pair_geometry_rejected",
            projected_discs=projected,
            gate_pair_diagnostics=tuple(pair_diagnostics),
        )
    frame = candidate.frame
    ideal_clearance = 0.5 * (
        frame.edge_gap_m - 2.0 * config.vehicle_half_width_m
    )
    if ideal_clearance < config.minimum_predicted_gate_clearance_m:
        return SceneAssessment(
            SCENE_BLOCKED,
            ("competition_pair_too_narrow",) + frame.pair_ids,
            "gate_too_narrow",
            candidate.path_progress_ahead_m,
            frame.pair_ids,
            gate_frame=frame,
            projected_discs=projected,
            gate_pair_diagnostics=tuple(pair_diagnostics),
        )
    admission = gate_passage_admission_safety(
        robot, frame, path, boundary, usable, config
    )
    if not admission.safe:
        return SceneAssessment(
            SCENE_BLOCKED,
            ("competition_pair_unsafe",) + frame.pair_ids,
            admission.reason,
            candidate.path_progress_ahead_m,
            frame.pair_ids,
            gate_frame=frame,
            projected_discs=projected,
            admission_safety=admission,
            gate_pair_diagnostics=tuple(pair_diagnostics),
        )
    return SceneAssessment(
        SCENE_GATE,
        ("competition_pair",) + frame.pair_ids,
        "competition_two_disc_gate",
        candidate.path_progress_ahead_m,
        frame.pair_ids,
        gate_frame=frame,
        projected_discs=projected,
        admission_safety=admission,
        gate_pair_diagnostics=tuple(pair_diagnostics),
    )


def assess_scene(
    tracks: Sequence[DiscTrack],
    path: Sequence[Tuple[float, float]],
    robot: RobotPose,
    boundary: BoundaryLines,
    config: PassageConfig,
    preferred_side: int = 0,
    planner_diagnostics: Optional[Dict[str, object]] = None,
) -> SceneAssessment:
    if config.competition_two_disc_mode:
        return assess_competition_pair(
            tracks, path, robot, boundary, config
        )
    if len(path) < 2:
        return SceneAssessment(
            SCENE_BLOCKED, ("blocked",), "reference_route_unavailable"
        )
    projected = project_discs(tracks, path, robot, config)
    if not projected:
        return SceneAssessment(
            SCENE_NORMAL, ("normal",), "route_corridor_clear"
        )

    gate_candidate = None
    if config.enable_gate_special_mode:
        gate_candidate = select_gate_candidate(
            tracks,
            path,
            robot,
            config.gate_config(),
            maximum_edge_gap_m=config.gate_special_mode_max_gap_m,
        )
    if gate_candidate is not None:
        if (
            gate_candidate.path_progress_ahead_m
            < config.late_detection_stop_distance_m
        ):
            return SceneAssessment(
                SCENE_BLOCKED,
                ("gate_detected_too_late",)
                + gate_candidate.frame.pair_ids,
                "gate_detected_too_late",
                gate_candidate.path_progress_ahead_m,
                gate_candidate.frame.pair_ids,
                gate_frame=gate_candidate.frame,
                projected_discs=projected,
            )
        ideal_clearance = 0.5 * (
            gate_candidate.frame.edge_gap_m
            - 2.0 * config.vehicle_half_width_m
        )
        if ideal_clearance < config.minimum_predicted_gate_clearance_m:
            return SceneAssessment(
                SCENE_BLOCKED,
                ("gate_too_narrow",) + gate_candidate.frame.pair_ids,
                "gate_too_narrow",
                gate_candidate.path_progress_ahead_m,
                gate_candidate.frame.pair_ids,
                gate_frame=gate_candidate.frame,
                projected_discs=projected,
            )
        return SceneAssessment(
            SCENE_GATE,
            ("gate",) + gate_candidate.frame.pair_ids,
            "true_narrow_gate",
            gate_candidate.path_progress_ahead_m,
            gate_candidate.frame.pair_ids,
            gate_frame=gate_candidate.frame,
            projected_discs=projected,
        )

    cumulative_s = cumulative_lengths(path)
    _, robot_projection = _signed_lateral(
        robot.xy,
        path,
        cumulative_s,
        maximum_distance_m=config.route_projection_max_distance_m,
    )
    if not robot_projection.valid:
        return SceneAssessment(
            SCENE_BLOCKED,
            ("blocked",),
            "robot_off_reference_route",
            projected_discs=projected,
        )
    tangent_yaw = math.atan2(
        robot_projection.tangent_y, robot_projection.tangent_x
    )
    width = projected_vehicle_width(
        wrap_angle(robot.yaw - tangent_yaw),
        config.vehicle_half_length_m,
        config.vehicle_half_width_m,
    )
    blocking_threshold = (
        0.5 * width
        + config.disc_radius_m
        + config.disc_clearance_margin_m
        + config.geometry_uncertainty_m
    )
    affecting = tuple(
        item for item in projected
        if abs(item.lateral_m) < blocking_threshold
    )
    envelope = BoundaryEnvelope(boundary, path, cumulative_s)
    road_context = []
    for item in projected:
        limits = envelope.limits_at(item.path_s_m)
        if limits is None:
            continue
        left_line, right_line = limits
        half_line = 0.5 * boundary.yellow_line_width_m
        if (
            item.lateral_m + config.disc_radius_m
            >= right_line + half_line
            and item.lateral_m - config.disc_radius_m
            <= left_line - half_line
        ):
            road_context.append(item)
    forced_pair_context = (
        config.force_between_disc_pair and len(road_context) == 2
    )
    claimed_road_context = (
        config.claim_all_road_discs and bool(road_context)
    )
    if (
        not affecting
        and not forced_pair_context
        and not claimed_road_context
    ):
        return SceneAssessment(
            SCENE_NORMAL,
            ("normal",),
            "discs_outside_route_corridor",
            projected_discs=projected,
        )
    context_obstacles = tuple(road_context) or affecting

    # A second visible disc can constrain the side selected around the one
    # intersecting the nominal centerline.  Count-independent topology means
    # it must stay in the free-space model even when it is not itself on the
    # original path.
    if len(context_obstacles) == 1 and len(affecting) == 1:
        plan, reason = build_single_bypass_plan(
            affecting[0],
            path,
            robot,
            boundary,
            config,
            preferred_side=preferred_side,
        )
        if plan is None:
            return SceneAssessment(
                SCENE_BLOCKED,
                ("blocked", affecting[0].track.memory_id),
                reason,
                affecting[0].ahead_m,
                (affecting[0].track.memory_id,),
                projected_discs=projected,
            )
        return SceneAssessment(
            SCENE_BYPASS,
            (
                "single",
                affecting[0].track.memory_id,
                plan.selected_side,
            ),
            "single_disc_bypass",
            affecting[0].ahead_m,
            plan.obstacle_ids,
            bypass_plan=plan,
            projected_discs=projected,
        )

    plan, reason = build_multi_channel_plan(
        context_obstacles,
        path,
        robot,
        boundary,
        config,
        preferred_side=preferred_side,
        force_between_pair=forced_pair_context,
        diagnostics=planner_diagnostics,
    )
    if plan is None:
        return SceneAssessment(
            SCENE_BLOCKED,
            ("blocked_multi",)
            + tuple(item.track.memory_id for item in context_obstacles),
            reason,
            min(item.ahead_m for item in context_obstacles),
            tuple(item.track.memory_id for item in context_obstacles),
            projected_discs=projected,
        )
    return SceneAssessment(
        SCENE_MULTI,
        ("multi",) + plan.obstacle_ids + (plan.selected_side,),
        (
            "forced_between_disc_pair"
            if forced_pair_context
            else "multi_obstacle_channel"
        ),
        min(item.ahead_m for item in context_obstacles),
        plan.obstacle_ids,
        bypass_plan=plan,
        projected_discs=projected,
    )


class PassageStateMachine:
    """Unified move_base handoff and special-passage controller."""

    def __init__(self, config: Optional[PassageConfig] = None):
        self.config = config or PassageConfig()
        self.config.validate()
        self.state = STATE_NORMAL
        self.scene = SCENE_NORMAL
        self.candidate_key: Optional[Tuple[object, ...]] = None
        self.candidate_since_sec: Optional[float] = None
        self.candidate_missing_since_sec: Optional[float] = None
        self.candidate_assessment: Optional[SceneAssessment] = None
        self.latest_assessment: Optional[SceneAssessment] = None
        self.locked_path: Tuple[Tuple[float, float], ...] = ()
        self.locked_bypass_plan: Optional[BypassPlan] = None
        self.locked_gate_frame: Optional[GateFrame] = None
        self.locked_obstacles: Tuple[DiscTrack, ...] = ()
        self.pending_mode = ""
        self.handoff_since_sec: Optional[float] = None
        self.align_since_sec: Optional[float] = None
        self.exit_since_sec: Optional[float] = None
        self.degraded_since_sec: Optional[float] = None
        self.degraded_return_state = ""
        self.failure_reason = ""
        self.failure_recoverable = False
        self.failure_recovery_since_sec: Optional[float] = None
        self.failure_recovery_assessment: Optional[SceneAssessment] = None
        self.last_command = (0.0, 0.0, 0.0)
        self.last_update_sec: Optional[float] = None
        self.last_successful_side = 0
        self.special_started_sec: Optional[float] = None
        self.special_progress_xy: Optional[Tuple[float, float]] = None
        self.special_progress_sec: Optional[float] = None
        self.alignment_best_translation_error_m: Optional[float] = None
        self.alignment_best_yaw_error_rad: Optional[float] = None
        self.alignment_progress_sec: Optional[float] = None
        self.stage_started_sec: Optional[float] = None
        self.special_attempt_count = 0
        self.special_retry_not_before_sec: Optional[float] = None
        self.pair_consumed = False
        self.pair_disabled = False
        self.gate_frame_live_updated = False
        self.gate_frame_live_update_count = 0
        self.gate_frame_live_last_update_sec: Optional[float] = None

    @property
    def authority_requested(self) -> bool:
        return self.state in (
            STATE_TAKEOVER_BRAKE,
            STATE_SINGLE_AVOID,
            STATE_MULTI_AVOID,
            STATE_GATE_ALIGN,
            STATE_GATE_CROSS,
            STATE_GATE_EXIT,
            STATE_EXIT_RESUME,
            STATE_RECOVERABLE_HOLD,
            STATE_DEGRADED_HOLD,
            STATE_FAILED_STOP,
        )

    def reset(self, clear_competition_latch: bool = False) -> None:
        previous_side = self.last_successful_side
        attempt_count = self.special_attempt_count
        retry_not_before = self.special_retry_not_before_sec
        pair_consumed = self.pair_consumed
        pair_disabled = self.pair_disabled
        live_update_count = self.gate_frame_live_update_count
        live_last_update_sec = self.gate_frame_live_last_update_sec
        self.__init__(self.config)
        self.last_successful_side = previous_side
        self.gate_frame_live_update_count = live_update_count
        self.gate_frame_live_last_update_sec = live_last_update_sec
        if not clear_competition_latch:
            self.special_attempt_count = attempt_count
            self.special_retry_not_before_sec = retry_not_before
            self.pair_consumed = pair_consumed
            self.pair_disabled = pair_disabled

    def fail(
        self, reason: str, recoverable_scene_failure: bool = False
    ) -> None:
        self.state = (
            STATE_RECOVERABLE_HOLD
            if recoverable_scene_failure
            else STATE_FAILED_STOP
        )
        self.failure_reason = str(reason)
        self.failure_recoverable = bool(recoverable_scene_failure)
        self.failure_recovery_since_sec = None
        self.failure_recovery_assessment = None
        self.last_command = (0.0, 0.0, 0.0)
        self.last_update_sec = None

    def _scene_failure_can_clear(
        self,
        assessment: SceneAssessment,
        tracks: Sequence[DiscTrack],
    ) -> bool:
        if assessment.scene in (SCENE_BYPASS, SCENE_MULTI, SCENE_GATE):
            return True
        if (
            assessment.scene == SCENE_NORMAL
            and assessment.reason == "discs_outside_route_corridor"
        ):
            return True
        if (
            assessment.scene != SCENE_NORMAL
            or assessment.reason != "route_corridor_clear"
        ):
            return False

        # An empty projection may mean either a genuinely changed safe route
        # or temporary perception loss. Only accept it when every obstacle
        # that caused the failure is still being observed freshly.
        failed_ids = {
            track.memory_id for track in self.locked_obstacles
        }
        if not failed_ids:
            return False
        fresh_ids = {
            track.memory_id
            for track in tracks
            if (
                track.confirmed
                and track.phase != PHASE_TOMBSTONE
                and track.last_seen_age_sec
                <= self.config.track_fresh_sec
            )
        }
        return failed_ids.issubset(fresh_ids)

    def _update_recoverable_scene_failure(
        self,
        now: float,
        robot: Optional[RobotPose],
        tracks: Sequence[DiscTrack],
        path: Sequence[Tuple[float, float]],
        boundary: BoundaryLines,
        measured_linear_speed_mps: float,
        measured_angular_speed_radps: float,
        critical_inputs_fresh: bool,
        navigation_active: bool,
    ) -> PassageDecision:
        if not self.failure_recoverable:
            return self._decision(now, reason=self.failure_reason)
        stationary = (
            abs(float(measured_linear_speed_mps))
            <= self.config.handoff_linear_speed_max_mps
            and abs(float(measured_angular_speed_radps))
            <= self.config.handoff_angular_speed_max_radps
        )
        if (
            robot is None
            or not critical_inputs_fresh
            or len(path) < 2
            or not stationary
        ):
            self.failure_recovery_since_sec = None
            self.failure_recovery_assessment = None
            return self._decision(now, reason=self.failure_reason)

        assessment = assess_scene(
            tracks,
            path,
            robot,
            boundary,
            self.config,
            preferred_side=self.last_successful_side,
        )
        self.latest_assessment = assessment
        if not self._scene_failure_can_clear(assessment, tracks):
            self.failure_recovery_since_sec = None
            self.failure_recovery_assessment = None
            return self._decision(now, reason=self.failure_reason)

        if not self._same_candidate_geometry(
            self.failure_recovery_assessment, assessment
        ):
            self.failure_recovery_since_sec = now
            self.failure_recovery_assessment = assessment
            return self._decision(
                now, reason="failed_scene_recovery_confirming"
            )
        self.failure_recovery_assessment = assessment
        stable_sec = max(
            0.0, now - float(self.failure_recovery_since_sec)
        )
        safe_plan_ready = assessment.scene in (
            SCENE_BYPASS,
            SCENE_MULTI,
            SCENE_GATE,
        )
        required_stable_sec = (
            self.config.scene_confirm_sec
            if safe_plan_ready
            else self.config.scene_release_sec
        )
        if stable_sec < required_stable_sec:
            return self._decision(
                now, reason="failed_scene_recovery_confirming"
            )

        if safe_plan_ready:
            self._lock_assessment(
                now, robot, assessment, path, tracks
            )
            return self._decision(
                now, reason="safe_local_plan_reacquired"
            )

        should_resume = not navigation_active
        self.reset()
        self.latest_assessment = assessment
        return self._decision(
            now,
            reason=(
                "recoverable_hold_cleared_resuming_goal"
                if should_resume
                else "recoverable_hold_cleared_navigation_continues"
            ),
            resume=should_resume,
        )

    def _stable_sec(self, now_sec: float) -> float:
        if self.candidate_since_sec is None:
            return 0.0
        return max(0.0, now_sec - self.candidate_since_sec)

    def _same_candidate_geometry(
        self,
        previous: Optional[SceneAssessment],
        current: SceneAssessment,
    ) -> bool:
        if (
            previous is None
            or previous.scene != current.scene
            or previous.key != current.key
        ):
            return False
        if previous.gate_frame is not None or current.gate_frame is not None:
            if previous.gate_frame is None or current.gate_frame is None:
                return False
            first = previous.gate_frame
            second = current.gate_frame
            midpoint_delta = _norm(
                _sub(first.midpoint_xy, second.midpoint_xy)
            )
            gap_delta = abs(first.edge_gap_m - second.edge_gap_m)
            angle_delta = abs(wrap_angle(first.yaw_ref - second.yaw_ref))
            return (
                midpoint_delta
                <= self.config.scene_position_variation_max_m
                and gap_delta
                <= self.config.scene_position_variation_max_m
                and angle_delta
                <= self.config.scene_angle_variation_max_rad
            )
        if (
            previous.bypass_plan is not None
            and current.bypass_plan is not None
        ):
            first = previous.bypass_plan
            second = current.bypass_plan
            if (
                first.obstacle_ids != second.obstacle_ids
                or first.selected_side != second.selected_side
            ):
                return False
            common_start = max(first.start_s_m, second.start_s_m)
            common_end = min(first.end_s_m, second.end_s_m)
            if common_end < common_start:
                return False
            reference_delta = max(
                abs(
                    first.lateral_at(
                        common_start
                        + (common_end - common_start) * index / 4.0
                    )
                    - second.lateral_at(
                        common_start
                        + (common_end - common_start) * index / 4.0
                    )
                )
                for index in range(5)
            )
            return (
                abs(first.end_s_m - second.end_s_m)
                <= self.config.scene_progress_variation_max_m
                and abs(
                    first.minimum_channel_width_m
                    - second.minimum_channel_width_m
                )
                <= self.config.scene_position_variation_max_m
                and reference_delta
                <= self.config.scene_position_variation_max_m
            )
        return True

    def _decision(
        self,
        now_sec: float,
        command=(0.0, 0.0, 0.0),
        reason="",
        cancel=False,
        resume=False,
        errors=None,
        gate_diagnostics=None,
        swept_safety=None,
    ) -> PassageDecision:
        return PassageDecision(
            state=self.state,
            scene=self.scene,
            authority_requested=self.authority_requested,
            command_body=command,
            reason=reason,
            cancel_move_base=cancel,
            resume_move_base=resume,
            assessment=self.latest_assessment,
            bypass_plan=(
                self.locked_bypass_plan
                if self.locked_bypass_plan is not None
                else (
                    None
                    if self.latest_assessment is None
                    else self.latest_assessment.bypass_plan
                )
            ),
            gate_frame=(
                self.locked_gate_frame
                if self.locked_gate_frame is not None
                else (
                    None
                    if self.latest_assessment is None
                    else self.latest_assessment.gate_frame
                )
            ),
            errors=errors,
            gate_diagnostics=gate_diagnostics,
            scene_stable_sec=self._stable_sec(now_sec),
            swept_safety=(
                swept_safety
                if swept_safety is not None
                else (
                    None
                    if self.latest_assessment is None
                    else self.latest_assessment.admission_safety
                )
            ),
        )

    def _competition_fallback(
        self,
        now_sec: float,
        failure_reason: str,
        errors=None,
        gate_diagnostics=None,
        swept_safety=None,
    ) -> PassageDecision:
        """Release special authority and retry the saved move_base goal."""
        if self.config.competition_two_disc_mode:
            self.special_retry_not_before_sec = (
                float(now_sec) + self.config.special_retry_cooldown_sec
            )
            if self.special_attempt_count >= self.config.special_max_attempts:
                self.pair_disabled = True
        self.reset()
        return self._decision(
            now_sec,
            reason="competition_resume_after:" + str(failure_reason),
            resume=True,
            errors=errors,
            gate_diagnostics=gate_diagnostics,
            swept_safety=swept_safety,
        )

    def _rate_limit(
        self,
        target: Tuple[float, float, float],
        now_sec: float,
    ) -> Tuple[float, float, float]:
        if self.last_update_sec is None:
            self.last_update_sec = now_sec
            self.last_command = (0.0, 0.0, 0.0)
        dt = min(0.20, max(0.001, now_sec - self.last_update_sec))
        limits = (
            self.config.acceleration_x_mps2 * dt,
            self.config.acceleration_y_mps2 * dt,
            self.config.acceleration_yaw_radps2 * dt,
        )
        limited = tuple(
            old + _clamp(new - old, -limit, limit)
            for old, new, limit in zip(
                self.last_command, target, limits
            )
        )
        self.last_command = limited
        self.last_update_sec = now_sec
        return limited

    @staticmethod
    def _world_to_body(
        world: Tuple[float, float], yaw: float
    ) -> Tuple[float, float]:
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        return (
            cosine * world[0] + sine * world[1],
            -sine * world[0] + cosine * world[1],
        )

    def _direct_clearance(
        self, robot: RobotPose
    ) -> float:
        if not self.locked_obstacles:
            return math.inf
        cosine = math.cos(robot.yaw)
        sine = math.sin(robot.yaw)
        values = []
        for track in self.locked_obstacles:
            delta = _sub(track.fixed_xy, robot.xy)
            base_x = cosine * delta[0] + sine * delta[1]
            base_y = -sine * delta[0] + cosine * delta[1]
            values.append(
                circle_to_rectangle_clearance(
                    base_x,
                    base_y,
                    self.config.vehicle_half_length_m,
                    self.config.vehicle_half_width_m,
                    self.config.disc_radius_m,
                )
            )
        return min(values)

    def _swept_safety(
        self,
        robot: RobotPose,
        command: Tuple[float, float, float],
        boundary: BoundaryLines,
    ) -> SweptSafety:
        return swept_command_safety(
            robot=robot,
            command_body=command,
            path=self.locked_path,
            boundary=boundary,
            tracks=self.locked_obstacles,
            config=self.config,
        )

    def _bypass_control(
        self, robot: RobotPose, now_sec: float
    ) -> Tuple[Tuple[float, float, float], PassageErrors]:
        path = self.locked_path
        plan = self.locked_bypass_plan
        cumulative_s = cumulative_lengths(path)
        current_d, projection = _signed_lateral(
            robot.xy,
            path,
            cumulative_s,
            maximum_distance_m=self.config.route_projection_max_distance_m,
        )
        if current_d is None or plan is None:
            raise ValueError("locked bypass reference is unavailable")
        reference_d = plan.lateral_at(projection.s)
        slope = _clamp(
            plan.slope_at(projection.s),
            -self.config.maximum_lateral_slope,
            self.config.maximum_lateral_slope,
        )
        tangent = (projection.tangent_x, projection.tangent_y)
        normal = (-tangent[1], tangent[0])
        yaw_ref = math.atan2(tangent[1], tangent[0])
        yaw_error = wrap_angle(yaw_ref - robot.yaw)
        lateral_error = current_d - reference_d
        forward = self.config.bypass_speed_mps
        if (
            abs(lateral_error) > 0.025
            or abs(yaw_error) > math.radians(5.0)
        ):
            forward *= 0.45
        feedforward = forward * slope
        lateral_speed = _clamp(
            feedforward - self.config.lateral_gain * lateral_error,
            -self.config.maximum_lateral_speed_mps,
            self.config.maximum_lateral_speed_mps,
        )
        world = (
            forward * tangent[0] + lateral_speed * normal[0],
            forward * tangent[1] + lateral_speed * normal[1],
        )
        body = self._world_to_body(world, robot.yaw)
        angular = _clamp(
            self.config.yaw_gain * yaw_error,
            -self.config.maximum_angular_speed_radps,
            self.config.maximum_angular_speed_radps,
        )
        errors = PassageErrors(
            lateral_m=lateral_error,
            yaw_rad=yaw_error,
            progress_m=projection.s,
            reference_lateral_m=reference_d,
            minimum_direct_clearance_m=self._direct_clearance(robot),
        )
        return self._rate_limit(
            (body[0], body[1], angular), now_sec
        ), errors

    def _gate_control(
        self,
        robot: RobotPose,
        now_sec: float,
        forward_speed: float,
        align: bool,
    ):
        frame = self.locked_gate_frame
        diagnostics = gate_errors(
            robot, frame, self.config.gate_config()
        )
        lateral_speed = _clamp(
            -self.config.lateral_gain * diagnostics.lateral_m,
            -self.config.maximum_lateral_speed_mps,
            self.config.maximum_lateral_speed_mps,
        )
        if align:
            desired = -self.config.gate_align_distance_m
            maximum_approach_speed = (
                self.config.gate_approach_speed_mps
                if self.config.competition_two_disc_mode
                else self.config.gate_max_align_speed_mps
            )
            forward_speed = _clamp(
                0.8 * (desired - diagnostics.progress_m),
                -maximum_approach_speed,
                maximum_approach_speed,
            )
        elif (
            abs(diagnostics.lateral_m)
            > self.config.gate_align_lateral_tolerance_m
            or abs(diagnostics.yaw_rad)
            > self.config.gate_align_yaw_tolerance_rad
        ):
            # A 40 cm clear gap leaves 35 mm per side for the 33 cm body.
            # Do not spend that margin while translating forward: pause and
            # use the holonomic/yaw axes to recover the aligned pose first.
            forward_speed = 0.0
        world = (
            forward_speed * frame.travel_xy[0]
            + lateral_speed * frame.normal_xy[0],
            forward_speed * frame.travel_xy[1]
            + lateral_speed * frame.normal_xy[1],
        )
        body = self._world_to_body(world, robot.yaw)
        angular = _clamp(
            self.config.yaw_gain * diagnostics.yaw_rad,
            -self.config.maximum_angular_speed_radps,
            self.config.maximum_angular_speed_radps,
        )
        errors = PassageErrors(
            lateral_m=diagnostics.lateral_m,
            yaw_rad=diagnostics.yaw_rad,
            progress_m=diagnostics.progress_m,
            reference_lateral_m=0.0,
            minimum_direct_clearance_m=(
                diagnostics.minimum_direct_clearance_m
            ),
        )
        return (
            self._rate_limit((body[0], body[1], angular), now_sec),
            errors,
            diagnostics,
        )

    def _refresh_gate_frame_from_visual_tracks(
        self,
        now_sec: float,
        tracks: Sequence[DiscTrack],
    ) -> bool:
        """Refresh the locked gate with the same freshly observed disc pair."""
        frame = self.locked_gate_frame
        if frame is None:
            return False
        by_id = {
            track.memory_id: track
            for track in tracks
            if (
                track.confirmed
                and track.phase != PHASE_TOMBSTONE
                and track.last_seen_age_sec
                <= self.config.gate_visual_update_max_age_sec
            )
        }
        if not all(memory_id in by_id for memory_id in frame.pair_ids):
            return False
        first = by_id[frame.pair_ids[0]]
        second = by_id[frame.pair_ids[1]]
        difference = _sub(second.fixed_xy, first.fixed_xy)
        center_distance = _norm(difference)
        if center_distance <= 2.0 * self.config.disc_radius_m:
            return False
        edge_gap = center_distance - 2.0 * self.config.disc_radius_m
        if edge_gap > self.config.gate_release_gap_m:
            return False

        normal = (
            difference[0] / center_distance,
            difference[1] / center_distance,
        )
        if _dot(normal, frame.normal_xy) < 0.0:
            normal = -normal[0], -normal[1]
        travel = -normal[1], normal[0]
        if _dot(travel, frame.travel_xy) < 0.0:
            travel = -travel[0], -travel[1]
        midpoint = (
            0.5 * (first.fixed_xy[0] + second.fixed_xy[0]),
            0.5 * (first.fixed_xy[1] + second.fixed_xy[1]),
        )
        self.locked_gate_frame = GateFrame(
            pair_ids=frame.pair_ids,
            disc_first_xy=first.fixed_xy,
            disc_second_xy=second.fixed_xy,
            midpoint_xy=midpoint,
            normal_xy=normal,
            travel_xy=travel,
            yaw_ref=math.atan2(travel[1], travel[0]),
            center_distance_m=center_distance,
            edge_gap_m=edge_gap,
            midpoint_path_distance_m=frame.midpoint_path_distance_m,
            along_path_mismatch_m=frame.along_path_mismatch_m,
            crossing_angle_deviation_rad=(
                frame.crossing_angle_deviation_rad
            ),
        )
        self.locked_obstacles = first, second
        self.gate_frame_live_updated = True
        self.gate_frame_live_update_count += 1
        self.gate_frame_live_last_update_sec = float(now_sec)
        return True

    def _lock_assessment(
        self,
        now_sec: float,
        robot: RobotPose,
        assessment: SceneAssessment,
        path: Sequence[Tuple[float, float]],
        tracks: Sequence[DiscTrack],
    ) -> None:
        self.latest_assessment = assessment
        self.locked_path = tuple(path)
        self.locked_bypass_plan = assessment.bypass_plan
        self.locked_gate_frame = assessment.gate_frame
        wanted = set(assessment.affecting_ids)
        self.locked_obstacles = tuple(
            track for track in tracks if track.memory_id in wanted
        )
        self.scene = assessment.scene
        self.pending_mode = assessment.scene
        self.state = STATE_TAKEOVER_BRAKE
        self.handoff_since_sec = None
        self.last_command = (0.0, 0.0, 0.0)
        self.last_update_sec = None
        self.special_started_sec = float(now_sec)
        self.special_progress_xy = (float(robot.x), float(robot.y))
        self.special_progress_sec = float(now_sec)
        self.alignment_progress_sec = float(now_sec)
        self.stage_started_sec = float(now_sec)
        self.failure_reason = ""
        self.failure_recoverable = False
        self.failure_recovery_since_sec = None
        self.failure_recovery_assessment = None
        if self.config.competition_two_disc_mode:
            self.special_attempt_count += 1

    def update(
        self,
        now_sec: float,
        robot: Optional[RobotPose],
        tracks: Sequence[DiscTrack],
        path: Sequence[Tuple[float, float]],
        boundary: BoundaryLines,
        navigation_active: bool,
        measured_linear_speed_mps: float,
        measured_angular_speed_radps: float,
        critical_inputs_fresh: bool = True,
        mission_active: Optional[bool] = None,
        live_gate_tracks: Sequence[DiscTrack] = (),
    ) -> PassageDecision:
        now = float(now_sec)
        self.gate_frame_live_updated = False
        if (
            self.config.competition_two_disc_mode
            and self.state in (
                STATE_TAKEOVER_BRAKE,
                STATE_GATE_ALIGN,
                STATE_GATE_CROSS,
                STATE_GATE_EXIT,
            )
        ):
            self._refresh_gate_frame_from_visual_tracks(
                now, live_gate_tracks
            )
        mission_requested = (
            bool(navigation_active)
            if mission_active is None
            else bool(mission_active)
        )
        if self.state in (STATE_RECOVERABLE_HOLD, STATE_FAILED_STOP):
            return self._update_recoverable_scene_failure(
                now=now,
                robot=robot,
                tracks=tracks,
                path=path,
                boundary=boundary,
                measured_linear_speed_mps=measured_linear_speed_mps,
                measured_angular_speed_radps=measured_angular_speed_radps,
                critical_inputs_fresh=critical_inputs_fresh,
                navigation_active=navigation_active,
            )

        if self.authority_requested and (
            robot is None or not critical_inputs_fresh
        ):
            if self.state != STATE_DEGRADED_HOLD:
                self.degraded_return_state = self.state
                self.state = STATE_DEGRADED_HOLD
                self.degraded_since_sec = now
                self.last_command = (0.0, 0.0, 0.0)
                self.last_update_sec = None
            elif (
                self.degraded_since_sec is not None
                and now - self.degraded_since_sec
                > self.config.degraded_hold_sec
            ):
                if self.config.competition_finish_priority:
                    return self._competition_fallback(
                        now, "critical_input_timeout"
                    )
                self.fail("critical_input_timeout")
            return self._decision(now, reason="degraded_input_hold")

        if self.state == STATE_DEGRADED_HOLD:
            self.state = self.degraded_return_state
            self.degraded_since_sec = None
            self.last_command = (0.0, 0.0, 0.0)
            self.last_update_sec = None

        if self.authority_requested and robot is not None:
            overall_timeout = (
                self.config.competition_overall_timeout_sec
                if self.config.competition_two_disc_mode
                else self.config.special_control_max_sec
            )
            if (
                self.special_started_sec is not None
                and now - self.special_started_sec
                >= overall_timeout
            ):
                if self.config.competition_finish_priority:
                    return self._competition_fallback(
                        now, "special_control_timeout"
                    )
                self.fail("special_control_timeout")
                return self._decision(now, reason=self.failure_reason)
            if (
                self.config.competition_two_disc_mode
                and self.stage_started_sec is not None
            ):
                stage_elapsed = now - self.stage_started_sec
                stage_timeout = None
                stage_reason = ""
                if self.state == STATE_TAKEOVER_BRAKE:
                    stage_timeout = self.config.handoff_stop_timeout_sec
                    stage_reason = "handoff_stop_timeout"
                elif self.state == STATE_GATE_ALIGN:
                    stage_timeout = self.config.approach_align_timeout_sec
                    stage_reason = "approach_align_timeout"
                elif self.state in (STATE_GATE_CROSS, STATE_GATE_EXIT):
                    stage_timeout = self.config.cross_exit_timeout_sec
                    stage_reason = "cross_exit_timeout"
                if (
                    stage_timeout is not None
                    and stage_elapsed >= stage_timeout
                ):
                    if self.config.competition_finish_priority:
                        return self._competition_fallback(now, stage_reason)
                    self.fail(stage_reason)
                    return self._decision(now, reason=self.failure_reason)
            current_xy = float(robot.x), float(robot.y)
            if self.special_progress_xy is None:
                self.special_progress_xy = current_xy
                self.special_progress_sec = now
            elif (
                math.hypot(
                    current_xy[0] - self.special_progress_xy[0],
                    current_xy[1] - self.special_progress_xy[1],
                )
                >= self.config.special_min_progress_m
            ):
                self.special_progress_xy = current_xy
                self.special_progress_sec = now
            elif (
                self.special_progress_sec is not None
                and now - self.special_progress_sec
                >= self.config.special_no_progress_timeout_sec
                and (
                    not self.config.competition_two_disc_mode
                    or self.state in (STATE_GATE_CROSS, STATE_GATE_EXIT)
                )
            ):
                if self.config.competition_finish_priority:
                    return self._competition_fallback(
                        now, "special_no_progress_timeout"
                    )
                self.fail("special_no_progress_timeout")
                return self._decision(now, reason=self.failure_reason)

        if self.state in (
            STATE_NORMAL,
            STATE_SINGLE_ARMED,
            STATE_MULTI_ARMED,
            STATE_GATE_ARMED,
        ):
            previous_assessment = self.latest_assessment
            competition_inhibited_reason = ""
            if self.config.competition_two_disc_mode:
                if self.pair_consumed:
                    competition_inhibited_reason = "competition_pair_consumed"
                elif self.pair_disabled:
                    competition_inhibited_reason = "competition_pair_disabled"
                elif (
                    self.special_retry_not_before_sec is not None
                    and now < self.special_retry_not_before_sec
                ):
                    competition_inhibited_reason = "competition_retry_cooldown"
            if competition_inhibited_reason:
                assessment = SceneAssessment(
                    SCENE_NORMAL,
                    ("normal",),
                    competition_inhibited_reason,
                )
            elif (
                robot is None
                or not mission_requested
                or not critical_inputs_fresh
                or len(path) < 2
            ):
                assessment = SceneAssessment(
                    SCENE_NORMAL, ("normal",), "navigation_inactive"
                )
            else:
                assessment = assess_scene(
                    tracks,
                    path,
                    robot,
                    boundary,
                    self.config,
                    preferred_side=(
                        self.latest_assessment.bypass_plan.selected_side
                        if (
                            self.latest_assessment is not None
                            and self.latest_assessment.bypass_plan is not None
                            and self.state
                            in (STATE_SINGLE_ARMED, STATE_MULTI_ARMED)
                        )
                        else self.last_successful_side
                    ),
                )
            self.latest_assessment = assessment
            if assessment.scene == SCENE_NORMAL:
                if self.state == STATE_NORMAL:
                    self.candidate_key = None
                    self.candidate_since_sec = None
                    self.candidate_assessment = None
                    return self._decision(
                        now, reason=assessment.reason
                    )
                if self.candidate_missing_since_sec is None:
                    self.candidate_missing_since_sec = now
                if (
                    now - self.candidate_missing_since_sec
                    >= self.config.scene_release_sec
                ):
                    self.state = STATE_NORMAL
                    self.scene = SCENE_NORMAL
                    self.candidate_key = None
                    self.candidate_since_sec = None
                    self.candidate_assessment = None
                return self._decision(
                    now, reason="armed_scene_release_hysteresis"
                )
            self.candidate_missing_since_sec = None
            if not self._same_candidate_geometry(
                self.candidate_assessment, assessment
            ):
                self.candidate_key = assessment.key
                self.candidate_since_sec = now
                self.candidate_assessment = assessment
            else:
                self.candidate_assessment = assessment
            stable = self._stable_sec(now)
            required_confirm_sec = self.config.scene_confirm_sec
            if (
                previous_assessment is not None
                and previous_assessment.bypass_plan is not None
                and assessment.bypass_plan is not None
                and previous_assessment.bypass_plan.selected_side
                != assessment.bypass_plan.selected_side
            ):
                required_confirm_sec = max(
                    required_confirm_sec,
                    self.config.side_invalid_confirm_sec,
                )
            armed_state = {
                SCENE_BYPASS: STATE_SINGLE_ARMED,
                SCENE_MULTI: STATE_MULTI_ARMED,
                SCENE_GATE: STATE_GATE_ARMED,
                SCENE_BLOCKED: (
                    STATE_GATE_ARMED
                    if assessment.gate_frame is not None
                    else STATE_SINGLE_ARMED
                ),
            }[assessment.scene]
            if stable >= required_confirm_sec:
                self.state = (
                    STATE_NORMAL
                    if self.config.competition_two_disc_mode
                    else armed_state
                )
                self.scene = assessment.scene
            trigger_distance = (
                self.config.gate_takeover_distance_m
                if assessment.gate_frame is not None
                else self.config.bypass_takeover_distance_m
            )
            if (
                stable >= required_confirm_sec
                and assessment.nearest_ahead_m <= trigger_distance
            ):
                if assessment.scene == SCENE_BLOCKED:
                    if self.config.competition_finish_priority:
                        self.reset()
                        self.latest_assessment = assessment
                        return self._decision(
                            now,
                            reason=(
                                "competition_continue_blocked_scene:"
                                + assessment.reason
                            ),
                        )
                    self.scene = SCENE_BLOCKED
                    self.locked_path = tuple(path)
                    self.locked_gate_frame = assessment.gate_frame
                    self.locked_bypass_plan = assessment.bypass_plan
                    wanted = set(assessment.affecting_ids)
                    self.locked_obstacles = tuple(
                        track
                        for track in tracks
                        if track.memory_id in wanted
                    )
                    self.fail(
                        assessment.reason,
                        recoverable_scene_failure=True,
                    )
                    return self._decision(
                        now,
                        reason="safe_hold_no_local_plan:" + self.failure_reason,
                        cancel=True,
                    )
                self._lock_assessment(now, robot, assessment, path, tracks)
                return self._decision(
                    now, reason="special_mode_takeover", cancel=True
                )
            return self._decision(
                now,
                reason=(
                    "scene_confirming"
                    if stable < required_confirm_sec
                    else "special_mode_armed"
                ),
            )

        if robot is None:
            self.fail("robot_pose_missing")
            return self._decision(now, reason=self.failure_reason)

        if self.state == STATE_TAKEOVER_BRAKE:
            kinematically_stopped = (
                abs(measured_linear_speed_mps)
                <= self.config.handoff_linear_speed_max_mps
                and abs(measured_angular_speed_radps)
                <= self.config.handoff_angular_speed_max_radps
            )
            cancel_ack_timed_out = (
                self.config.competition_finish_priority
                and self.special_started_sec is not None
                and now - self.special_started_sec
                >= self.config.takeover_ack_timeout_sec
            )
            stopped = kinematically_stopped and (
                not navigation_active or cancel_ack_timed_out
            )
            if stopped:
                if self.handoff_since_sec is None:
                    self.handoff_since_sec = now
                elif (
                    now - self.handoff_since_sec
                    >= self.config.handoff_stop_hold_sec
                ):
                    if self.pending_mode == SCENE_GATE:
                        self.state = STATE_GATE_ALIGN
                        self.alignment_best_translation_error_m = None
                        self.alignment_best_yaw_error_rad = None
                        self.alignment_progress_sec = now
                    elif self.pending_mode == SCENE_MULTI:
                        self.state = STATE_MULTI_AVOID
                    else:
                        self.state = STATE_SINGLE_AVOID
                    self.last_command = (0.0, 0.0, 0.0)
                    self.last_update_sec = None
                    self.stage_started_sec = now
            else:
                self.handoff_since_sec = None
            return self._decision(now, reason="waiting_for_handoff_stop")

        if self.state in (STATE_SINGLE_AVOID, STATE_MULTI_AVOID):
            try:
                command, errors = self._bypass_control(robot, now)
            except ValueError as error:
                if self.config.competition_finish_priority:
                    return self._competition_fallback(now, str(error))
                self.fail(str(error))
                return self._decision(now, reason=self.failure_reason)
            failure_reason = ""
            if (
                errors.minimum_direct_clearance_m
                < self.config.emergency_direct_clearance_m
            ):
                failure_reason = "direct_disc_clearance"
            elif abs(errors.lateral_m) > self.config.stop_lateral_error_m:
                failure_reason = "bypass_lateral_error_limit"
            elif abs(errors.yaw_rad) > self.config.stop_yaw_error_rad:
                failure_reason = "bypass_yaw_error_limit"
            swept_safety = self._swept_safety(
                robot, command, boundary
            )
            if not swept_safety.safe and not failure_reason:
                failure_reason = swept_safety.reason
            if failure_reason:
                if self.config.competition_finish_priority:
                    return self._competition_fallback(
                        now,
                        failure_reason,
                        errors=errors,
                        swept_safety=swept_safety,
                    )
                self.fail(failure_reason)
                return self._decision(
                    now,
                    reason=self.failure_reason,
                    errors=errors,
                    swept_safety=swept_safety,
                )
            if (
                errors.progress_m
                >= self.locked_bypass_plan.end_s_m
                and abs(errors.lateral_m)
                <= self.config.completion_lateral_tolerance_m
            ):
                self.last_successful_side = (
                    self.locked_bypass_plan.selected_side
                )
                self.state = STATE_EXIT_RESUME
                self.exit_since_sec = now
                self.last_command = (0.0, 0.0, 0.0)
                self.last_update_sec = None
                return self._decision(
                    now,
                    reason="bypass_complete_stopping",
                    errors=errors,
                    swept_safety=swept_safety,
                )
            return self._decision(
                now,
                command=command,
                reason="following_bypass",
                errors=errors,
                swept_safety=swept_safety,
            )

        if self.state == STATE_GATE_ALIGN:
            command, errors, diagnostics = self._gate_control(
                robot, now, 0.0, align=True
            )
            if self.config.competition_two_disc_mode:
                translation_error = math.hypot(
                    errors.lateral_m,
                    errors.progress_m + self.config.gate_align_distance_m,
                )
                yaw_error = abs(errors.yaw_rad)
                translation_improved = (
                    self.alignment_best_translation_error_m is None
                    or translation_error
                    <= self.alignment_best_translation_error_m
                    - self.config.special_min_progress_m
                )
                yaw_improved = (
                    self.alignment_best_yaw_error_rad is None
                    or yaw_error
                    <= self.alignment_best_yaw_error_rad
                    - self.config.special_min_yaw_progress_rad
                )
                if translation_improved or yaw_improved:
                    self.alignment_best_translation_error_m = min(
                        translation_error,
                        (
                            translation_error
                            if self.alignment_best_translation_error_m is None
                            else self.alignment_best_translation_error_m
                        ),
                    )
                    self.alignment_best_yaw_error_rad = min(
                        yaw_error,
                        (
                            yaw_error
                            if self.alignment_best_yaw_error_rad is None
                            else self.alignment_best_yaw_error_rad
                        ),
                    )
                    self.alignment_progress_sec = now
                elif (
                    self.alignment_progress_sec is not None
                    and now - self.alignment_progress_sec
                    >= self.config.special_no_progress_timeout_sec
                ):
                    if self.config.competition_finish_priority:
                        return self._competition_fallback(
                            now, "approach_align_no_progress"
                        )
                    self.fail("approach_align_no_progress")
                    return self._decision(now, reason=self.failure_reason)
            ideal_clearance = 0.5 * (
                self.locked_gate_frame.edge_gap_m
                - 2.0 * self.config.vehicle_half_width_m
            )
            failure_reason = ""
            if (
                diagnostics.minimum_direct_clearance_m
                < self.config.emergency_direct_clearance_m
            ):
                failure_reason = "direct_disc_clearance"
            elif abs(errors.lateral_m) > self.config.gate_lateral_abort_m:
                failure_reason = "gate_alignment_lateral_abort"
            elif abs(errors.yaw_rad) > self.config.gate_yaw_abort_rad:
                failure_reason = "gate_alignment_yaw_abort"
            elif (
                ideal_clearance
                < self.config.minimum_predicted_gate_clearance_m
            ):
                failure_reason = "gate_too_narrow"
            swept_safety = self._swept_safety(
                robot, command, boundary
            )
            if not swept_safety.safe and not failure_reason:
                failure_reason = swept_safety.reason
            if failure_reason:
                if self.config.competition_finish_priority:
                    return self._competition_fallback(
                        now,
                        failure_reason,
                        errors=errors,
                        gate_diagnostics=diagnostics,
                        swept_safety=swept_safety,
                    )
                self.fail(failure_reason)
                return self._decision(
                    now,
                    reason=self.failure_reason,
                    errors=errors,
                    gate_diagnostics=diagnostics,
                    swept_safety=swept_safety,
                )
            aligned = (
                abs(errors.lateral_m)
                <= self.config.gate_align_lateral_tolerance_m
                and abs(errors.yaw_rad)
                <= self.config.gate_align_yaw_tolerance_rad
                and diagnostics.predicted_clearance_m
                >= self.config.minimum_predicted_gate_clearance_m
            )
            if aligned:
                if self.align_since_sec is None:
                    self.align_since_sec = now
                elif (
                    now - self.align_since_sec
                    >= self.config.gate_align_hold_sec
                ):
                    self.state = STATE_GATE_CROSS
                    self.stage_started_sec = now
            else:
                self.align_since_sec = None
            return self._decision(
                now,
                command=command,
                reason="aligning_gate",
                errors=errors,
                gate_diagnostics=diagnostics,
                swept_safety=swept_safety,
            )

        if self.state in (STATE_GATE_CROSS, STATE_GATE_EXIT):
            speed = (
                self.config.gate_speed_mps
                if self.state == STATE_GATE_CROSS
                else self.config.gate_exit_speed_mps
            )
            command, errors, diagnostics = self._gate_control(
                robot, now, speed, align=False
            )
            failure_reason = ""
            if (
                diagnostics.minimum_direct_clearance_m
                < self.config.emergency_direct_clearance_m
            ):
                failure_reason = "direct_disc_clearance"
            elif abs(errors.lateral_m) > 0.030:
                failure_reason = "gate_lateral_error_limit"
            elif abs(errors.yaw_rad) > math.radians(6.0):
                failure_reason = "gate_yaw_error_limit"
            elif (
                self.state == STATE_GATE_CROSS
                and diagnostics.predicted_clearance_m
                < self.config.minimum_predicted_gate_clearance_m
            ):
                failure_reason = "predicted_gate_clearance"
            swept_safety = self._swept_safety(
                robot, command, boundary
            )
            if not swept_safety.safe and not failure_reason:
                failure_reason = swept_safety.reason
            if failure_reason:
                if self.config.competition_finish_priority:
                    return self._competition_fallback(
                        now,
                        failure_reason,
                        errors=errors,
                        gate_diagnostics=diagnostics,
                        swept_safety=swept_safety,
                    )
                self.fail(failure_reason)
                return self._decision(
                    now,
                    reason=self.failure_reason,
                    errors=errors,
                    gate_diagnostics=diagnostics,
                    swept_safety=swept_safety,
                )
            exit_start = (
                self.config.vehicle_half_length_m
                + self.config.disc_radius_m
                + self.config.gate_exit_clearance_margin_m
            )
            if (
                self.state == STATE_GATE_CROSS
                and errors.progress_m >= exit_start
            ):
                self.state = STATE_GATE_EXIT
            if errors.progress_m >= self.config.gate_exit_distance_m:
                self.state = STATE_EXIT_RESUME
                self.exit_since_sec = now
                self.stage_started_sec = now
                self.last_command = (0.0, 0.0, 0.0)
                self.last_update_sec = None
                return self._decision(
                    now,
                    reason="gate_complete_stopping",
                    errors=errors,
                    gate_diagnostics=diagnostics,
                    swept_safety=swept_safety,
                )
            return self._decision(
                now,
                command=command,
                reason=(
                    "crossing_gate"
                    if self.state == STATE_GATE_CROSS
                    else "clearing_gate_exit"
                ),
                errors=errors,
                gate_diagnostics=diagnostics,
                swept_safety=swept_safety,
            )

        if self.state == STATE_EXIT_RESUME:
            if (
                self.exit_since_sec is not None
                and now - self.exit_since_sec
                >= self.config.exit_stop_hold_sec
            ):
                if self.config.competition_two_disc_mode:
                    self.pair_consumed = True
                self.reset()
                return self._decision(
                    now, reason="passage_complete", resume=True
                )
            return self._decision(now, reason="exit_stop_hold")

        self.fail("unknown_passage_state")
        return self._decision(now, reason=self.failure_reason)
