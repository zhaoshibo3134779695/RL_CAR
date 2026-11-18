"""Geometry and control state machine for passing between a close disc pair.

The normal SBPL/TEB navigation stack remains authoritative unless two stable
disc tracks form a narrow gate across the active global path.  Once the robot
reaches the pre-alignment point, this module freezes the gate geometry and
generates a short, low-speed holonomic crossing command.

This file deliberately has no ROS imports so the safety-critical geometry and
state transitions can be exercised with deterministic unit tests.
"""

from dataclasses import dataclass
import math
from typing import Dict, Optional, Sequence, Tuple

from .disc_passage import (
    PHASE_TOMBSTONE,
    circle_to_rectangle_clearance,
    cumulative_lengths,
    project_point_to_polyline,
)


STATE_NORMAL = "normal_teb"
STATE_ARMED = "gate_armed"
STATE_ALIGN = "align"
STATE_CROSSING = "crossing"
STATE_EXIT_CLEAR = "exit_clear"
STATE_STOP_RELEASE = "stop_and_release"
STATE_FAILED_STOP = "failed_stop"


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(float(angle)), math.cos(float(angle)))


def _dot(first: Tuple[float, float], second: Tuple[float, float]) -> float:
    return first[0] * second[0] + first[1] * second[1]


def _subtract(
    first: Tuple[float, float], second: Tuple[float, float]
) -> Tuple[float, float]:
    return first[0] - second[0], first[1] - second[1]


def _norm(vector: Tuple[float, float]) -> float:
    return math.hypot(vector[0], vector[1])


def _normalized(vector: Tuple[float, float]) -> Tuple[float, float]:
    length = _norm(vector)
    if length <= 1e-9:
        raise ValueError("cannot normalize a zero-length vector")
    return vector[0] / length, vector[1] / length


@dataclass(frozen=True)
class DiscTrack:
    memory_id: int
    fixed_xy: Tuple[float, float]
    confirmed: bool = True
    phase: str = "ahead"
    last_seen_age_sec: float = 0.0
    estimated_diameter_m: Optional[float] = None
    confidence: Optional[float] = None


@dataclass(frozen=True)
class RobotPose:
    x: float
    y: float
    yaw: float

    @property
    def xy(self) -> Tuple[float, float]:
        return float(self.x), float(self.y)


@dataclass
class NarrowGateConfig:
    disc_radius_m: float = 0.050
    vehicle_half_length_m: float = 0.165
    vehicle_half_width_m: float = 0.150

    # The trigger is based on physical edge-to-edge gap, not center distance.
    trigger_edge_gap_m: float = 0.450
    release_edge_gap_m: float = 0.480
    pair_confirm_duration_sec: float = 0.60
    pair_midpoint_variation_max_m: float = 0.015
    pair_distance_variation_max_m: float = 0.015
    pair_angle_variation_max_rad: float = math.radians(1.5)
    maximum_candidate_track_age_sec: float = 0.40
    midpoint_path_distance_max_m: float = 0.15
    pair_along_path_mismatch_max_m: float = 0.10
    crossing_angle_deviation_max_rad: float = math.radians(20.0)
    minimum_opposite_side_offset_m: float = 0.02
    gate_ahead_max_m: float = 1.20
    gate_ahead_min_m: float = 0.10

    prealign_distance_m: float = 0.50
    latest_takeover_progress_m: float = -0.10
    align_lateral_tolerance_m: float = 0.008
    align_yaw_tolerance_rad: float = math.radians(1.5)
    align_hold_duration_sec: float = 0.30
    alignment_lateral_abort_m: float = 0.15
    alignment_yaw_abort_rad: float = math.radians(45.0)

    pass_speed_mps: float = 0.040
    exit_speed_mps: float = 0.030
    lateral_gain: float = 0.80
    yaw_gain: float = 1.50
    prealign_longitudinal_gain: float = 0.80
    maximum_lateral_speed_mps: float = 0.015
    maximum_align_longitudinal_speed_mps: float = 0.015
    maximum_angular_speed_radps: float = 0.080
    slow_lateral_error_m: float = 0.012
    slow_yaw_error_rad: float = math.radians(2.0)
    stop_lateral_error_m: float = 0.030
    stop_yaw_error_rad: float = math.radians(6.0)

    # Physical clearance is kept separate from TEB's soft obstacle distance.
    minimum_physical_clearance_m: float = 0.008
    geometry_uncertainty_m: float = 0.005
    emergency_direct_clearance_m: float = 0.005
    exit_clearance_margin_m: float = 0.050
    exit_distance_m: float = 0.350
    stop_hold_duration_sec: float = 0.30

    acceleration_x_mps2: float = 0.10
    acceleration_y_mps2: float = 0.08
    acceleration_yaw_radps2: float = 0.25

    def validate(self) -> None:
        positive = {
            "disc_radius_m": self.disc_radius_m,
            "vehicle_half_length_m": self.vehicle_half_length_m,
            "vehicle_half_width_m": self.vehicle_half_width_m,
            "trigger_edge_gap_m": self.trigger_edge_gap_m,
            "release_edge_gap_m": self.release_edge_gap_m,
            "pair_confirm_duration_sec": self.pair_confirm_duration_sec,
            "pair_midpoint_variation_max_m":
                self.pair_midpoint_variation_max_m,
            "pair_distance_variation_max_m":
                self.pair_distance_variation_max_m,
            "pair_angle_variation_max_rad":
                self.pair_angle_variation_max_rad,
            "maximum_candidate_track_age_sec":
                self.maximum_candidate_track_age_sec,
            "midpoint_path_distance_max_m":
                self.midpoint_path_distance_max_m,
            "pair_along_path_mismatch_max_m":
                self.pair_along_path_mismatch_max_m,
            "crossing_angle_deviation_max_rad":
                self.crossing_angle_deviation_max_rad,
            "gate_ahead_max_m": self.gate_ahead_max_m,
            "gate_ahead_min_m": self.gate_ahead_min_m,
            "prealign_distance_m": self.prealign_distance_m,
            "align_lateral_tolerance_m": self.align_lateral_tolerance_m,
            "align_yaw_tolerance_rad": self.align_yaw_tolerance_rad,
            "align_hold_duration_sec": self.align_hold_duration_sec,
            "alignment_lateral_abort_m":
                self.alignment_lateral_abort_m,
            "alignment_yaw_abort_rad": self.alignment_yaw_abort_rad,
            "pass_speed_mps": self.pass_speed_mps,
            "exit_speed_mps": self.exit_speed_mps,
            "maximum_lateral_speed_mps": self.maximum_lateral_speed_mps,
            "maximum_align_longitudinal_speed_mps":
                self.maximum_align_longitudinal_speed_mps,
            "maximum_angular_speed_radps":
                self.maximum_angular_speed_radps,
            "stop_lateral_error_m": self.stop_lateral_error_m,
            "stop_yaw_error_rad": self.stop_yaw_error_rad,
            "minimum_physical_clearance_m":
                self.minimum_physical_clearance_m,
            "exit_distance_m": self.exit_distance_m,
            "stop_hold_duration_sec": self.stop_hold_duration_sec,
            "acceleration_x_mps2": self.acceleration_x_mps2,
            "acceleration_y_mps2": self.acceleration_y_mps2,
            "acceleration_yaw_radps2": self.acceleration_yaw_radps2,
        }
        invalid = [name for name, value in positive.items() if value <= 0.0]
        if invalid:
            raise ValueError(
                "narrow gate parameters must be positive: "
                + ", ".join(sorted(invalid))
            )
        if self.release_edge_gap_m <= self.trigger_edge_gap_m:
            raise ValueError(
                "release_edge_gap_m must exceed trigger_edge_gap_m"
            )
        if self.gate_ahead_max_m <= self.gate_ahead_min_m:
            raise ValueError("gate ahead range is empty")
        if self.prealign_distance_m <= self.gate_ahead_min_m:
            raise ValueError("prealign distance is too small")
        if not (
            -self.prealign_distance_m
            < self.latest_takeover_progress_m
            < 0.0
        ):
            raise ValueError("latest takeover progress must be before the gate")
        if self.exit_distance_m <= (
            self.vehicle_half_length_m
            + self.disc_radius_m
            + self.exit_clearance_margin_m
        ):
            raise ValueError("exit distance does not clear the robot rear")


@dataclass(frozen=True)
class GateFrame:
    pair_ids: Tuple[int, int]
    disc_first_xy: Tuple[float, float]
    disc_second_xy: Tuple[float, float]
    midpoint_xy: Tuple[float, float]
    normal_xy: Tuple[float, float]
    travel_xy: Tuple[float, float]
    yaw_ref: float
    center_distance_m: float
    edge_gap_m: float
    midpoint_path_distance_m: float
    along_path_mismatch_m: float
    crossing_angle_deviation_rad: float


@dataclass(frozen=True)
class GateCandidate:
    frame: GateFrame
    robot_progress_m: float
    path_progress_ahead_m: float


@dataclass(frozen=True)
class GatePairDiagnostic:
    """Complete, read-only explanation of one gate-pair decision."""

    pair_ids: Tuple[int, int]
    first_xy: Tuple[float, float]
    second_xy: Tuple[float, float]
    first_diameter_m: Optional[float]
    second_diameter_m: Optional[float]
    first_confidence: Optional[float]
    second_confidence: Optional[float]
    first_age_sec: float
    second_age_sec: float
    center_distance_m: float
    edge_gap_m: float
    first_route_s_m: Optional[float]
    first_route_d_m: Optional[float]
    second_route_s_m: Optional[float]
    second_route_d_m: Optional[float]
    midpoint_xy: Tuple[float, float]
    midpoint_route_s_m: Optional[float]
    midpoint_route_offset_m: Optional[float]
    longitudinal_mismatch_m: Optional[float]
    pair_angle_rad: float
    route_angle_rad: Optional[float]
    angle_error_rad: Optional[float]
    path_progress_ahead_m: Optional[float]
    robot_progress_m: Optional[float]
    predicted_left_clearance_m: float
    predicted_right_clearance_m: float
    predicted_min_clearance_m: float
    checks: Tuple[Dict[str, object], ...]
    reason_codes: Tuple[str, ...]
    accepted: bool


@dataclass(frozen=True)
class GateErrors:
    lateral_m: float
    yaw_rad: float
    progress_m: float
    projected_width_m: float
    predicted_clearance_m: float
    direct_clearance_first_m: float
    direct_clearance_second_m: float

    @property
    def minimum_direct_clearance_m(self) -> float:
        return min(
            self.direct_clearance_first_m,
            self.direct_clearance_second_m,
        )


@dataclass(frozen=True)
class GateDecision:
    state: str
    authority_requested: bool
    command_body: Tuple[float, float, float]
    cancel_move_base: bool = False
    resume_move_base: bool = False
    reason: str = ""
    frame: Optional[GateFrame] = None
    errors: Optional[GateErrors] = None
    candidate_stable_sec: float = 0.0


def _path_point_from_projection(
    path: Sequence[Tuple[float, float]],
    projection,
) -> Tuple[float, float]:
    index = projection.segment_index
    first = path[index]
    tangent = projection.tangent_x, projection.tangent_y
    segment_s = projection.s
    cumulative = cumulative_lengths(path)
    offset = segment_s - cumulative[index]
    return (
        float(first[0]) + offset * tangent[0],
        float(first[1]) + offset * tangent[1],
    )


def select_gate_candidate(
    tracks: Sequence[DiscTrack],
    path: Sequence[Tuple[float, float]],
    robot: RobotPose,
    config: NarrowGateConfig,
    maximum_edge_gap_m: Optional[float] = None,
    pair_diagnostics: Optional[list] = None,
) -> Optional[GateCandidate]:
    """Return the best close pair and optionally explain every pair check."""

    if len(path) < 2:
        return None
    cumulative = cumulative_lengths(path)
    robot_projection = project_point_to_polyline(
        robot.xy,
        path,
        cumulative,
        max_distance_m=config.midpoint_path_distance_max_m * 2.0,
    )
    if not robot_projection.valid:
        return None

    usable = [
        track
        for track in tracks
        if (
            track.confirmed
            and track.phase != PHASE_TOMBSTONE
            and track.last_seen_age_sec
            <= config.maximum_candidate_track_age_sec
        )
    ]
    gap_limit = (
        config.trigger_edge_gap_m
        if maximum_edge_gap_m is None
        else float(maximum_edge_gap_m)
    )
    candidates = []
    maximum_angle_dot = math.sin(
        config.crossing_angle_deviation_max_rad
    )

    def route_coordinates(point):
        projection = project_point_to_polyline(
            point,
            path,
            cumulative,
            max_distance_m=max(
                10.0, config.midpoint_path_distance_max_m * 10.0
            ),
        )
        if not projection.valid:
            return None, None, None
        path_point = _path_point_from_projection(path, projection)
        tangent = projection.tangent_x, projection.tangent_y
        normal = -tangent[1], tangent[0]
        lateral = _dot(_subtract(point, path_point), normal)
        return projection, float(projection.s), float(lateral)

    for first_index, first in enumerate(usable[:-1]):
        for second in usable[first_index + 1:]:
            difference = _subtract(second.fixed_xy, first.fixed_xy)
            center_distance = _norm(difference)
            edge_gap = center_distance - 2.0 * config.disc_radius_m
            midpoint = (
                0.5 * (first.fixed_xy[0] + second.fixed_xy[0]),
                0.5 * (first.fixed_xy[1] + second.fixed_xy[1]),
            )
            first_projection, first_s, first_d = route_coordinates(
                first.fixed_xy
            )
            second_projection, second_s, second_d = route_coordinates(
                second.fixed_xy
            )
            midpoint_projection, midpoint_s, midpoint_d = route_coordinates(
                midpoint
            )

            checks = []
            blocking_reasons = []

            def add_check(
                code, actual, operator, threshold, passed, blocking=True
            ):
                checks.append(
                    {
                        "code": str(code),
                        "actual": actual,
                        "operator": str(operator),
                        "threshold": threshold,
                        "passed": bool(passed),
                        "blocking": bool(blocking),
                    }
                )
                if blocking and not passed:
                    blocking_reasons.append(str(code))

            non_overlapping = center_distance > 2.0 * config.disc_radius_m
            add_check(
                "CENTER_DISTANCE_TOO_SMALL",
                center_distance,
                ">",
                2.0 * config.disc_radius_m,
                non_overlapping,
            )
            add_check(
                "EDGE_GAP_TOO_LARGE",
                edge_gap,
                "<=",
                gap_limit,
                edge_gap <= gap_limit,
            )

            midpoint_valid = midpoint_projection is not None
            midpoint_distance = (
                None
                if midpoint_projection is None
                else float(midpoint_projection.distance)
            )
            add_check(
                "MIDPOINT_PROJECTION_INVALID",
                midpoint_valid,
                "==",
                True,
                midpoint_valid,
            )
            add_check(
                "MIDPOINT_TOO_FAR_FROM_ROUTE",
                midpoint_distance,
                "<=",
                config.midpoint_path_distance_max_m,
                midpoint_distance is not None
                and midpoint_distance
                <= config.midpoint_path_distance_max_m,
            )

            path_tangent = None
            path_point = None
            disc_line = None
            crossing_angle_deviation = None
            along_mismatch = None
            first_side = first_d
            second_side = second_d
            travel = None
            robot_progress = None
            ahead = None
            route_angle = None
            if midpoint_projection is not None:
                path_tangent = (
                    midpoint_projection.tangent_x,
                    midpoint_projection.tangent_y,
                )
                path_point = _path_point_from_projection(
                    path, midpoint_projection
                )
                route_angle = math.atan2(
                    path_tangent[1], path_tangent[0]
                )
            if non_overlapping and path_tangent is not None:
                disc_line = _normalized(difference)
                angle_dot = abs(_dot(disc_line, path_tangent))
                crossing_angle_deviation = math.asin(
                    min(1.0, max(0.0, angle_dot))
                )
                along_mismatch = abs(_dot(difference, path_tangent))
                path_normal = (-path_tangent[1], path_tangent[0])
                first_side = _dot(
                    _subtract(first.fixed_xy, path_point), path_normal
                )
                second_side = _dot(
                    _subtract(second.fixed_xy, path_point), path_normal
                )
                travel_a = (-disc_line[1], disc_line[0])
                travel = (
                    travel_a
                    if _dot(travel_a, path_tangent) >= 0.0
                    else (-travel_a[0], -travel_a[1])
                )
                robot_progress = _dot(
                    _subtract(robot.xy, midpoint), travel
                )
                ahead = midpoint_projection.s - robot_projection.s

            add_check(
                "PAIR_ANGLE_MISMATCH",
                crossing_angle_deviation,
                "<=",
                config.crossing_angle_deviation_max_rad,
                crossing_angle_deviation is not None
                and abs(_dot(disc_line, path_tangent))
                <= maximum_angle_dot,
            )
            add_check(
                "LONGITUDINAL_MISMATCH",
                along_mismatch,
                "<=",
                config.pair_along_path_mismatch_max_m,
                along_mismatch is not None
                and along_mismatch
                <= config.pair_along_path_mismatch_max_m,
            )
            opposite_sides = (
                first_side is not None
                and second_side is not None
                and first_side * second_side < 0.0
            )
            add_check(
                "NOT_STRADDLING_ROUTE",
                {
                    "first_d_m": first_side,
                    "second_d_m": second_side,
                },
                "opposite_signs",
                True,
                opposite_sides,
            )
            minimum_side_offset = (
                None
                if first_side is None or second_side is None
                else min(abs(first_side), abs(second_side))
            )
            add_check(
                "OPPOSITE_SIDE_OFFSET_TOO_SMALL",
                minimum_side_offset,
                ">=",
                config.minimum_opposite_side_offset_m,
                minimum_side_offset is not None
                and minimum_side_offset
                >= config.minimum_opposite_side_offset_m,
            )
            add_check(
                "GATE_AHEAD_OUT_OF_RANGE",
                ahead,
                "within",
                [config.gate_ahead_min_m, config.gate_ahead_max_m],
                ahead is not None
                and config.gate_ahead_min_m
                <= ahead
                <= config.gate_ahead_max_m,
            )
            add_check(
                "ROBOT_NOT_BEFORE_GATE",
                robot_progress,
                "<",
                -config.gate_ahead_min_m,
                robot_progress is not None
                and robot_progress < -config.gate_ahead_min_m,
            )

            centered_clearance = 0.5 * (
                edge_gap - 2.0 * config.vehicle_half_width_m
            )
            diagnostic_clearance_threshold = (
                config.minimum_physical_clearance_m
                + config.geometry_uncertainty_m
            )
            add_check(
                "PREDICTED_CLEARANCE_BELOW_MINIMUM",
                centered_clearance,
                ">=",
                diagnostic_clearance_threshold,
                centered_clearance >= diagnostic_clearance_threshold,
                blocking=False,
            )

            accepted = not blocking_reasons
            diagnostic = GatePairDiagnostic(
                pair_ids=tuple(
                    sorted((int(first.memory_id), int(second.memory_id)))
                ),
                first_xy=first.fixed_xy,
                second_xy=second.fixed_xy,
                first_diameter_m=first.estimated_diameter_m,
                second_diameter_m=second.estimated_diameter_m,
                first_confidence=first.confidence,
                second_confidence=second.confidence,
                first_age_sec=float(first.last_seen_age_sec),
                second_age_sec=float(second.last_seen_age_sec),
                center_distance_m=center_distance,
                edge_gap_m=edge_gap,
                first_route_s_m=first_s,
                first_route_d_m=first_d,
                second_route_s_m=second_s,
                second_route_d_m=second_d,
                midpoint_xy=midpoint,
                midpoint_route_s_m=midpoint_s,
                midpoint_route_offset_m=midpoint_d,
                longitudinal_mismatch_m=along_mismatch,
                pair_angle_rad=math.atan2(difference[1], difference[0]),
                route_angle_rad=route_angle,
                angle_error_rad=crossing_angle_deviation,
                path_progress_ahead_m=ahead,
                robot_progress_m=robot_progress,
                predicted_left_clearance_m=centered_clearance,
                predicted_right_clearance_m=centered_clearance,
                predicted_min_clearance_m=centered_clearance,
                checks=tuple(checks),
                reason_codes=tuple(blocking_reasons),
                accepted=accepted,
            )
            if pair_diagnostics is not None:
                pair_diagnostics.append(diagnostic)
            if not accepted:
                continue

            frame = GateFrame(
                pair_ids=diagnostic.pair_ids,
                disc_first_xy=first.fixed_xy,
                disc_second_xy=second.fixed_xy,
                midpoint_xy=midpoint,
                normal_xy=disc_line,
                travel_xy=travel,
                yaw_ref=math.atan2(travel[1], travel[0]),
                center_distance_m=center_distance,
                edge_gap_m=edge_gap,
                midpoint_path_distance_m=midpoint_projection.distance,
                along_path_mismatch_m=along_mismatch,
                crossing_angle_deviation_rad=crossing_angle_deviation,
            )
            score = (
                ahead,
                midpoint_projection.distance,
                crossing_angle_deviation,
                edge_gap,
            )
            candidates.append(
                (score, GateCandidate(frame, robot_progress, ahead))
            )

    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def projected_vehicle_width(
    yaw_error_rad: float,
    half_length_m: float,
    half_width_m: float,
) -> float:
    return 2.0 * (
        float(half_width_m) * abs(math.cos(float(yaw_error_rad)))
        + float(half_length_m) * abs(math.sin(float(yaw_error_rad)))
    )


def gate_errors(
    robot: RobotPose,
    frame: GateFrame,
    config: NarrowGateConfig,
) -> GateErrors:
    relative = _subtract(robot.xy, frame.midpoint_xy)
    lateral = _dot(relative, frame.normal_xy)
    progress = _dot(relative, frame.travel_xy)
    yaw_error = wrap_angle(frame.yaw_ref - robot.yaw)
    width = projected_vehicle_width(
        yaw_error,
        config.vehicle_half_length_m,
        config.vehicle_half_width_m,
    )
    predicted_clearance = (
        0.5 * (frame.edge_gap_m - width) - abs(lateral)
    )

    cosine = math.cos(robot.yaw)
    sine = math.sin(robot.yaw)

    def clearance(disc_xy):
        delta_x = disc_xy[0] - robot.x
        delta_y = disc_xy[1] - robot.y
        base_x = cosine * delta_x + sine * delta_y
        base_y = -sine * delta_x + cosine * delta_y
        return circle_to_rectangle_clearance(
            base_x,
            base_y,
            config.vehicle_half_length_m,
            config.vehicle_half_width_m,
            config.disc_radius_m,
        )

    return GateErrors(
        lateral_m=lateral,
        yaw_rad=yaw_error,
        progress_m=progress,
        projected_width_m=width,
        predicted_clearance_m=predicted_clearance,
        direct_clearance_first_m=clearance(frame.disc_first_xy),
        direct_clearance_second_m=clearance(frame.disc_second_xy),
    )


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, float(value)))


class NarrowGateStateMachine:
    """Latch a validated gate and generate a bounded crossing command."""

    def __init__(self, config: Optional[NarrowGateConfig] = None):
        self.config = config or NarrowGateConfig()
        self.config.validate()
        self.state = STATE_NORMAL
        self.candidate_pair_ids: Optional[Tuple[int, int]] = None
        self.candidate_since_sec: Optional[float] = None
        self.candidate_reference_frame: Optional[GateFrame] = None
        self.latest_candidate: Optional[GateCandidate] = None
        self.locked_frame: Optional[GateFrame] = None
        self.align_since_sec: Optional[float] = None
        self.stop_since_sec: Optional[float] = None
        self.failure_reason = ""
        self.last_command = (0.0, 0.0, 0.0)
        self.last_update_sec: Optional[float] = None
        self.debug: Dict[str, object] = {}

    @property
    def authority_requested(self) -> bool:
        return self.state in (
            STATE_ALIGN,
            STATE_CROSSING,
            STATE_EXIT_CLEAR,
            STATE_STOP_RELEASE,
            STATE_FAILED_STOP,
        )

    def reset(self) -> None:
        self.state = STATE_NORMAL
        self.candidate_pair_ids = None
        self.candidate_since_sec = None
        self.candidate_reference_frame = None
        self.latest_candidate = None
        self.locked_frame = None
        self.align_since_sec = None
        self.stop_since_sec = None
        self.failure_reason = ""
        self.last_command = (0.0, 0.0, 0.0)
        self.last_update_sec = None
        self.debug = {}

    def fail(self, reason: str) -> None:
        self.state = STATE_FAILED_STOP
        self.failure_reason = str(reason)
        self.align_since_sec = None
        self.stop_since_sec = None
        self.last_command = (0.0, 0.0, 0.0)

    def _candidate_stable_sec(self, now_sec: float) -> float:
        if self.candidate_since_sec is None:
            return 0.0
        return max(0.0, float(now_sec) - self.candidate_since_sec)

    def _required_predicted_clearance(self) -> float:
        return (
            self.config.minimum_physical_clearance_m
            + self.config.geometry_uncertainty_m
        )

    def _candidate_geometry_stable(self, frame: GateFrame) -> bool:
        reference = self.candidate_reference_frame
        if reference is None:
            return False
        midpoint_shift = _norm(
            _subtract(frame.midpoint_xy, reference.midpoint_xy)
        )
        distance_shift = abs(
            frame.center_distance_m - reference.center_distance_m
        )
        angle_shift = abs(wrap_angle(frame.yaw_ref - reference.yaw_ref))
        return (
            midpoint_shift
            <= self.config.pair_midpoint_variation_max_m
            and distance_shift
            <= self.config.pair_distance_variation_max_m
            and angle_shift
            <= self.config.pair_angle_variation_max_rad
        )

    def _safety_reason(self, errors: GateErrors) -> Optional[str]:
        if (
            errors.minimum_direct_clearance_m
            < self.config.emergency_direct_clearance_m
        ):
            return "direct_disc_clearance"
        if self.state == STATE_ALIGN:
            if (
                abs(errors.lateral_m)
                > self.config.alignment_lateral_abort_m
            ):
                return "alignment_lateral_abort"
            if abs(errors.yaw_rad) > self.config.alignment_yaw_abort_rad:
                return "alignment_yaw_abort"
            ideal_clearance = 0.5 * (
                self.locked_frame.edge_gap_m
                - 2.0 * self.config.vehicle_half_width_m
            )
            if ideal_clearance < self._required_predicted_clearance():
                return "gate_too_narrow"
            return None
        if abs(errors.lateral_m) > self.config.stop_lateral_error_m:
            return "lateral_error_limit"
        if abs(errors.yaw_rad) > self.config.stop_yaw_error_rad:
            return "yaw_error_limit"
        if (
            self.state == STATE_CROSSING
            and errors.predicted_clearance_m
            < self._required_predicted_clearance()
        ):
            return "predicted_gate_clearance"
        return None

    def _rate_limit(
        self,
        target: Tuple[float, float, float],
        now_sec: float,
    ) -> Tuple[float, float, float]:
        if self.last_update_sec is None:
            self.last_update_sec = float(now_sec)
            self.last_command = target
            return target
        dt = min(0.20, max(0.001, float(now_sec) - self.last_update_sec))
        limits = (
            self.config.acceleration_x_mps2 * dt,
            self.config.acceleration_y_mps2 * dt,
            self.config.acceleration_yaw_radps2 * dt,
        )
        limited = tuple(
            previous + _clamp(desired - previous, -limit, limit)
            for previous, desired, limit in zip(
                self.last_command, target, limits
            )
        )
        self.last_update_sec = float(now_sec)
        self.last_command = limited
        return limited

    @staticmethod
    def _world_to_body(
        world_velocity: Tuple[float, float],
        yaw: float,
    ) -> Tuple[float, float]:
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        return (
            cosine * world_velocity[0] + sine * world_velocity[1],
            -sine * world_velocity[0] + cosine * world_velocity[1],
        )

    def _control(
        self,
        robot: RobotPose,
        errors: GateErrors,
        forward_speed: float,
        align_longitudinal: bool,
        now_sec: float,
    ) -> Tuple[float, float, float]:
        frame = self.locked_frame
        if frame is None:
            return 0.0, 0.0, 0.0
        lateral_speed = _clamp(
            -self.config.lateral_gain * errors.lateral_m,
            -self.config.maximum_lateral_speed_mps,
            self.config.maximum_lateral_speed_mps,
        )
        if align_longitudinal:
            desired_progress = -self.config.prealign_distance_m
            forward_speed = _clamp(
                self.config.prealign_longitudinal_gain
                * (desired_progress - errors.progress_m),
                -self.config.maximum_align_longitudinal_speed_mps,
                self.config.maximum_align_longitudinal_speed_mps,
            )
        elif (
            abs(errors.lateral_m) > self.config.slow_lateral_error_m
            or abs(errors.yaw_rad) > self.config.slow_yaw_error_rad
        ):
            forward_speed *= 0.35

        world_velocity = (
            forward_speed * frame.travel_xy[0]
            + lateral_speed * frame.normal_xy[0],
            forward_speed * frame.travel_xy[1]
            + lateral_speed * frame.normal_xy[1],
        )
        body_x, body_y = self._world_to_body(world_velocity, robot.yaw)
        angular = _clamp(
            self.config.yaw_gain * errors.yaw_rad,
            -self.config.maximum_angular_speed_radps,
            self.config.maximum_angular_speed_radps,
        )
        return self._rate_limit((body_x, body_y, angular), now_sec)

    def _decision(
        self,
        command=(0.0, 0.0, 0.0),
        cancel=False,
        resume=False,
        reason="",
        errors=None,
        now_sec=0.0,
    ) -> GateDecision:
        return GateDecision(
            state=self.state,
            authority_requested=self.authority_requested,
            command_body=command,
            cancel_move_base=cancel,
            resume_move_base=resume,
            reason=reason,
            frame=self.locked_frame or (
                None
                if self.latest_candidate is None
                else self.latest_candidate.frame
            ),
            errors=errors,
            candidate_stable_sec=self._candidate_stable_sec(now_sec),
        )

    def update(
        self,
        now_sec: float,
        robot: Optional[RobotPose],
        tracks: Sequence[DiscTrack],
        path: Sequence[Tuple[float, float]],
        navigation_active: bool,
        inputs_fresh: bool = True,
    ) -> GateDecision:
        now = float(now_sec)
        if self.state == STATE_FAILED_STOP:
            return self._decision(
                reason=self.failure_reason,
                now_sec=now,
            )

        if self.state in (STATE_NORMAL, STATE_ARMED):
            maximum_gap = (
                self.config.release_edge_gap_m
                if self.state == STATE_ARMED
                else self.config.trigger_edge_gap_m
            )
            candidate = (
                None
                if robot is None or not navigation_active or not inputs_fresh
                else select_gate_candidate(
                    tracks,
                    path,
                    robot,
                    self.config,
                    maximum_edge_gap_m=maximum_gap,
                )
            )
            if candidate is None:
                self.state = STATE_NORMAL
                self.candidate_pair_ids = None
                self.candidate_since_sec = None
                self.candidate_reference_frame = None
                self.latest_candidate = None
                self.last_command = (0.0, 0.0, 0.0)
                self.last_update_sec = now
                return self._decision(
                    reason="normal_navigation",
                    now_sec=now,
                )

            if (
                candidate.frame.pair_ids != self.candidate_pair_ids
                or not self._candidate_geometry_stable(candidate.frame)
            ):
                self.candidate_pair_ids = candidate.frame.pair_ids
                self.candidate_since_sec = now
                self.candidate_reference_frame = candidate.frame
            self.latest_candidate = candidate
            self.state = STATE_ARMED
            stable = self._candidate_stable_sec(now)

            if (
                stable >= self.config.pair_confirm_duration_sec
                and candidate.robot_progress_m
                >= -self.config.prealign_distance_m
            ):
                if (
                    candidate.robot_progress_m
                    > self.config.latest_takeover_progress_m
                ):
                    self.locked_frame = candidate.frame
                    self.fail("gate_reached_before_safe_takeover")
                    return self._decision(
                        cancel=True,
                        reason=self.failure_reason,
                        now_sec=now,
                    )
                self.locked_frame = candidate.frame
                self.state = STATE_ALIGN
                self.align_since_sec = None
                self.last_command = (0.0, 0.0, 0.0)
                self.last_update_sec = now
                return self._decision(
                    cancel=True,
                    reason="gate_takeover",
                    now_sec=now,
                )

            return self._decision(
                reason="gate_candidate_confirming",
                now_sec=now,
            )

        if robot is None or not inputs_fresh:
            self.fail("gate_input_timeout")
            return self._decision(
                cancel=True,
                reason=self.failure_reason,
                now_sec=now,
            )
        if self.locked_frame is None:
            self.fail("locked_gate_missing")
            return self._decision(
                cancel=True,
                reason=self.failure_reason,
                now_sec=now,
            )

        errors = gate_errors(robot, self.locked_frame, self.config)
        unsafe = self._safety_reason(errors)
        if unsafe is not None:
            self.fail(unsafe)
            return self._decision(
                cancel=True,
                reason=self.failure_reason,
                errors=errors,
                now_sec=now,
            )

        if self.state == STATE_ALIGN:
            aligned = (
                abs(errors.lateral_m)
                <= self.config.align_lateral_tolerance_m
                and abs(errors.yaw_rad)
                <= self.config.align_yaw_tolerance_rad
                and errors.predicted_clearance_m
                >= self._required_predicted_clearance()
            )
            if aligned:
                if self.align_since_sec is None:
                    self.align_since_sec = now
                elif (
                    now - self.align_since_sec
                    >= self.config.align_hold_duration_sec
                ):
                    self.state = STATE_CROSSING
            else:
                self.align_since_sec = None
            command = self._control(
                robot,
                errors,
                forward_speed=0.0,
                align_longitudinal=True,
                now_sec=now,
            )
            return self._decision(
                command=command,
                reason="aligning_gate",
                errors=errors,
                now_sec=now,
            )

        if self.state == STATE_CROSSING:
            exit_start = (
                self.config.vehicle_half_length_m
                + self.config.disc_radius_m
                + self.config.exit_clearance_margin_m
            )
            if errors.progress_m >= exit_start:
                self.state = STATE_EXIT_CLEAR
            command = self._control(
                robot,
                errors,
                forward_speed=self.config.pass_speed_mps,
                align_longitudinal=False,
                now_sec=now,
            )
            return self._decision(
                command=command,
                reason="crossing_gate",
                errors=errors,
                now_sec=now,
            )

        if self.state == STATE_EXIT_CLEAR:
            if errors.progress_m >= self.config.exit_distance_m:
                self.state = STATE_STOP_RELEASE
                self.stop_since_sec = now
                self.last_command = (0.0, 0.0, 0.0)
                self.last_update_sec = now
                return self._decision(
                    reason="gate_exit_stopping",
                    errors=errors,
                    now_sec=now,
                )
            command = self._control(
                robot,
                errors,
                forward_speed=self.config.exit_speed_mps,
                align_longitudinal=False,
                now_sec=now,
            )
            return self._decision(
                command=command,
                reason="clearing_gate_exit",
                errors=errors,
                now_sec=now,
            )

        if self.state == STATE_STOP_RELEASE:
            if (
                self.stop_since_sec is not None
                and now - self.stop_since_sec
                >= self.config.stop_hold_duration_sec
            ):
                self.reset()
                self.last_update_sec = now
                return self._decision(
                    resume=True,
                    reason="gate_complete",
                    now_sec=now,
                )
            return self._decision(
                reason="gate_exit_stop_hold",
                errors=errors,
                now_sec=now,
            )

        self.fail("unknown_gate_state")
        return self._decision(
            cancel=True,
            reason=self.failure_reason,
            errors=errors,
            now_sec=now,
        )
