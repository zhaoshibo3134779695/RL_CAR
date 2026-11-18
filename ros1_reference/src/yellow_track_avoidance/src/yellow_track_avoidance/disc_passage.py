"""Path-relative lifecycle logic for world-fixed disc obstacles."""

from dataclasses import dataclass, field
import bisect
import math
from typing import Dict, List, Optional, Sequence, Tuple


PHASE_UNCONFIRMED = "unconfirmed"
PHASE_AHEAD = "ahead"
PHASE_ABREAST = "abreast"
PHASE_PASSED_CANDIDATE = "passed_candidate"
PHASE_TOMBSTONE = "tombstone"


@dataclass(frozen=True)
class Projection:
    valid: bool
    s: float = 0.0
    segment_index: int = 0
    tangent_x: float = 1.0
    tangent_y: float = 0.0
    distance: float = math.inf


@dataclass
class PassageConfig:
    vehicle_half_length_m: float = 0.165
    vehicle_half_width_m: float = 0.150
    disc_radius_m: float = 0.050
    progress_margin_m: float = 0.050
    clearance_m: float = 0.030
    confirm_duration_sec: float = 1.0
    confirm_frames: int = 8
    minimum_advance_m: float = 0.10
    projection_max_distance_m: float = 0.35
    projection_back_segments: int = 5
    projection_forward_segments: int = 20
    maximum_progress_jump_m: float = 0.10
    path_back_length_m: float = 0.60
    path_forward_length_m: float = 1.20
    path_resample_m: float = 0.03
    candidate_backtrack_reset_m: float = 0.03
    tombstone_reactivate_backtrack_m: float = 0.20
    abreast_extra_m: float = 0.15


@dataclass
class PassageState:
    phase: str = PHASE_UNCONFIRMED
    ever_ahead: bool = False
    navigation_epoch: int = 0
    encounter_path: Tuple[Tuple[float, float], ...] = ()
    encounter_cumulative_s: Tuple[float, ...] = ()
    disc_projection: Optional[Projection] = None
    last_robot_segment_index: Optional[int] = None
    last_robot_s: Optional[float] = None
    candidate_start_sec: Optional[float] = None
    candidate_start_s: Optional[float] = None
    candidate_frames: int = 0
    passed_sec: Optional[float] = None
    debug: Dict[str, object] = field(default_factory=dict)

    def activate(self, navigation_epoch: int) -> None:
        self.phase = PHASE_AHEAD
        self.ever_ahead = False
        self.navigation_epoch = int(navigation_epoch)
        self.encounter_path = ()
        self.encounter_cumulative_s = ()
        self.disc_projection = None
        self.last_robot_segment_index = None
        self.last_robot_s = None
        self.passed_sec = None
        self.debug = {}
        reset_candidate(self)


@dataclass(frozen=True)
class PassageUpdate:
    entered_tombstone: bool = False
    projection_valid: bool = False


def cumulative_lengths(
    path: Sequence[Tuple[float, float]],
) -> Tuple[float, ...]:
    if not path:
        return ()
    values: List[float] = [0.0]
    for first, second in zip(path[:-1], path[1:]):
        values.append(
            values[-1]
            + math.hypot(
                float(second[0]) - float(first[0]),
                float(second[1]) - float(first[1]),
            )
        )
    return tuple(values)


def project_point_to_polyline(
    point: Tuple[float, float],
    path: Sequence[Tuple[float, float]],
    cumulative_s: Sequence[float],
    start_segment: Optional[int] = None,
    back_segments: int = 5,
    forward_segments: int = 20,
    max_distance_m: float = math.inf,
) -> Projection:
    if len(path) < 2 or len(cumulative_s) != len(path):
        return Projection(valid=False)

    if start_segment is None:
        first_index = 0
        last_index = len(path) - 2
    else:
        first_index = max(0, int(start_segment) - int(back_segments))
        last_index = min(
            len(path) - 2,
            int(start_segment) + int(forward_segments),
        )

    query_x = float(point[0])
    query_y = float(point[1])
    best = Projection(valid=False)
    for index in range(first_index, last_index + 1):
        first_x, first_y = path[index]
        second_x, second_y = path[index + 1]
        delta_x = float(second_x) - float(first_x)
        delta_y = float(second_y) - float(first_y)
        length_squared = delta_x * delta_x + delta_y * delta_y
        if length_squared <= 1e-12:
            continue

        fraction = (
            (query_x - float(first_x)) * delta_x
            + (query_y - float(first_y)) * delta_y
        ) / length_squared
        fraction = min(1.0, max(0.0, fraction))
        projected_x = float(first_x) + fraction * delta_x
        projected_y = float(first_y) + fraction * delta_y
        distance = math.hypot(
            query_x - projected_x, query_y - projected_y
        )
        if best.valid and distance >= best.distance:
            continue
        segment_length = math.sqrt(length_squared)
        best = Projection(
            valid=True,
            s=float(cumulative_s[index]) + fraction * segment_length,
            segment_index=index,
            tangent_x=delta_x / segment_length,
            tangent_y=delta_y / segment_length,
            distance=distance,
        )

    if not best.valid or best.distance > float(max_distance_m):
        return Projection(valid=False)
    return best


def _point_at_progress(
    path: Sequence[Tuple[float, float]],
    cumulative_s: Sequence[float],
    progress_m: float,
) -> Tuple[float, float]:
    progress = min(
        max(float(progress_m), float(cumulative_s[0])),
        float(cumulative_s[-1]),
    )
    index = bisect.bisect_right(cumulative_s, progress) - 1
    index = min(max(index, 0), len(path) - 2)
    segment_start = float(cumulative_s[index])
    segment_end = float(cumulative_s[index + 1])
    denominator = segment_end - segment_start
    fraction = (
        0.0
        if denominator <= 1e-12
        else (progress - segment_start) / denominator
    )
    first = path[index]
    second = path[index + 1]
    return (
        float(first[0])
        + fraction * (float(second[0]) - float(first[0])),
        float(first[1])
        + fraction * (float(second[1]) - float(first[1])),
    )


def resample_path_window(
    path: Sequence[Tuple[float, float]],
    cumulative_s: Sequence[float],
    center_s: float,
    back_length_m: float,
    forward_length_m: float,
    spacing_m: float,
) -> Tuple[Tuple[float, float], ...]:
    if len(path) < 2 or len(cumulative_s) != len(path):
        return ()
    start_s = max(0.0, float(center_s) - float(back_length_m))
    end_s = min(
        float(cumulative_s[-1]),
        float(center_s) + float(forward_length_m),
    )
    if end_s - start_s <= 1e-9:
        return ()
    spacing = max(0.005, float(spacing_m))
    samples = [start_s]
    next_s = start_s + spacing
    while next_s < end_s:
        samples.append(next_s)
        next_s += spacing
    samples.append(end_s)
    return tuple(
        _point_at_progress(path, cumulative_s, value)
        for value in samples
    )


def robot_half_extent_along_path(
    robot_yaw: float,
    tangent_x: float,
    tangent_y: float,
    half_length_m: float,
    half_width_m: float,
) -> float:
    cosine = math.cos(float(robot_yaw))
    sine = math.sin(float(robot_yaw))
    tangent_body_x = cosine * tangent_x + sine * tangent_y
    tangent_body_y = -sine * tangent_x + cosine * tangent_y
    return (
        float(half_length_m) * abs(tangent_body_x)
        + float(half_width_m) * abs(tangent_body_y)
    )


def circle_to_rectangle_clearance(
    disc_x_in_base: float,
    disc_y_in_base: float,
    half_length_m: float,
    half_width_m: float,
    disc_radius_m: float,
) -> float:
    delta_x = max(
        abs(float(disc_x_in_base)) - float(half_length_m), 0.0
    )
    delta_y = max(
        abs(float(disc_y_in_base)) - float(half_width_m), 0.0
    )
    return math.hypot(delta_x, delta_y) - float(disc_radius_m)


def reset_candidate(state: PassageState) -> None:
    state.candidate_start_sec = None
    state.candidate_start_s = None
    state.candidate_frames = 0
    if state.phase == PHASE_PASSED_CANDIDATE:
        state.phase = PHASE_ABREAST


def _freeze_encounter_path(
    state: PassageState,
    disc_fixed_xy: Tuple[float, float],
    latest_path: Sequence[Tuple[float, float]],
    latest_cumulative_s: Sequence[float],
    disc_projection: Projection,
    config: PassageConfig,
) -> bool:
    frozen_path = resample_path_window(
        latest_path,
        latest_cumulative_s,
        disc_projection.s,
        config.path_back_length_m,
        config.path_forward_length_m,
        config.path_resample_m,
    )
    frozen_cumulative = cumulative_lengths(frozen_path)
    frozen_disc_projection = project_point_to_polyline(
        disc_fixed_xy,
        frozen_path,
        frozen_cumulative,
        max_distance_m=config.projection_max_distance_m,
    )
    if not frozen_disc_projection.valid:
        return False
    state.encounter_path = frozen_path
    state.encounter_cumulative_s = frozen_cumulative
    state.disc_projection = frozen_disc_projection
    state.last_robot_segment_index = None
    state.last_robot_s = None
    state.phase = PHASE_ABREAST
    reset_candidate(state)
    return True


def update_passage(
    state: PassageState,
    now_sec: float,
    disc_fixed_xy: Tuple[float, float],
    disc_base_xy: Tuple[float, float],
    robot_fixed_xy: Tuple[float, float],
    robot_yaw: float,
    latest_path: Sequence[Tuple[float, float]],
    latest_cumulative_s: Sequence[float],
    config: PassageConfig,
) -> PassageUpdate:
    if state.phase in (PHASE_UNCONFIRMED, PHASE_TOMBSTONE):
        return PassageUpdate()

    if not state.encounter_path:
        disc_projection = project_point_to_polyline(
            disc_fixed_xy,
            latest_path,
            latest_cumulative_s,
            max_distance_m=config.projection_max_distance_m,
        )
        robot_projection = project_point_to_polyline(
            robot_fixed_xy,
            latest_path,
            latest_cumulative_s,
            max_distance_m=config.projection_max_distance_m,
        )
        if not disc_projection.valid or not robot_projection.valid:
            state.debug = {"projection_valid": False}
            reset_candidate(state)
            return PassageUpdate()

        robot_extent = robot_half_extent_along_path(
            robot_yaw,
            disc_projection.tangent_x,
            disc_projection.tangent_y,
            config.vehicle_half_length_m,
            config.vehicle_half_width_m,
        )
        route_delta_s = robot_projection.s - disc_projection.s
        if -route_delta_s > robot_extent + config.disc_radius_m:
            state.ever_ahead = True
        clearance = circle_to_rectangle_clearance(
            disc_base_xy[0],
            disc_base_xy[1],
            config.vehicle_half_length_m,
            config.vehicle_half_width_m,
            config.disc_radius_m,
        )
        abreast_band = (
            robot_extent
            + config.disc_radius_m
            + config.abreast_extra_m
        )
        state.debug = {
            "projection_valid": True,
            "s_robot": robot_projection.s,
            "s_disc": disc_projection.s,
            "delta_s": route_delta_s,
            "robot_path_half_extent": robot_extent,
            "footprint_clearance": clearance,
        }
        if (
            abs(route_delta_s) > abreast_band
            and clearance > config.abreast_extra_m
        ):
            return PassageUpdate(projection_valid=True)
        if not _freeze_encounter_path(
            state,
            disc_fixed_xy,
            latest_path,
            latest_cumulative_s,
            disc_projection,
            config,
        ):
            state.debug["projection_valid"] = False
            return PassageUpdate()

    disc_projection = state.disc_projection
    if disc_projection is None:
        reset_candidate(state)
        return PassageUpdate()
    robot_projection = project_point_to_polyline(
        robot_fixed_xy,
        state.encounter_path,
        state.encounter_cumulative_s,
        start_segment=state.last_robot_segment_index,
        back_segments=config.projection_back_segments,
        forward_segments=config.projection_forward_segments,
        max_distance_m=config.projection_max_distance_m,
    )
    if not robot_projection.valid:
        state.debug = {"projection_valid": False}
        reset_candidate(state)
        return PassageUpdate()

    previous_robot_s = state.last_robot_s
    state.last_robot_segment_index = robot_projection.segment_index
    state.last_robot_s = robot_projection.s
    if (
        previous_robot_s is not None
        and abs(robot_projection.s - previous_robot_s)
        > config.maximum_progress_jump_m
    ):
        state.debug = {
            "projection_valid": False,
            "projection_jump_m": robot_projection.s - previous_robot_s,
        }
        reset_candidate(state)
        return PassageUpdate()
    if (
        previous_robot_s is not None
        and robot_projection.s
        < previous_robot_s - config.candidate_backtrack_reset_m
    ):
        reset_candidate(state)

    robot_extent = robot_half_extent_along_path(
        robot_yaw,
        disc_projection.tangent_x,
        disc_projection.tangent_y,
        config.vehicle_half_length_m,
        config.vehicle_half_width_m,
    )
    route_delta_s = robot_projection.s - disc_projection.s
    if -route_delta_s > robot_extent + config.disc_radius_m:
        state.ever_ahead = True
    clearance = circle_to_rectangle_clearance(
        disc_base_xy[0],
        disc_base_xy[1],
        config.vehicle_half_length_m,
        config.vehicle_half_width_m,
        config.disc_radius_m,
    )
    required_delta_s = (
        robot_extent
        + config.disc_radius_m
        + config.progress_margin_m
    )
    candidate_ok = (
        state.ever_ahead
        and route_delta_s >= required_delta_s
        and clearance >= config.clearance_m
    )

    candidate_duration = 0.0
    candidate_advance = 0.0
    if not candidate_ok:
        reset_candidate(state)
    elif state.phase != PHASE_PASSED_CANDIDATE:
        state.phase = PHASE_PASSED_CANDIDATE
        state.candidate_start_sec = float(now_sec)
        state.candidate_start_s = robot_projection.s
        state.candidate_frames = 1
    else:
        state.candidate_frames += 1
        candidate_duration = float(now_sec) - float(
            state.candidate_start_sec
        )
        candidate_advance = robot_projection.s - float(
            state.candidate_start_s
        )
        if (
            candidate_duration >= config.confirm_duration_sec
            and state.candidate_frames >= config.confirm_frames
            and candidate_advance >= config.minimum_advance_m
        ):
            state.phase = PHASE_TOMBSTONE
            state.passed_sec = float(now_sec)
            state.debug = {
                "projection_valid": True,
                "s_robot": robot_projection.s,
                "s_disc": disc_projection.s,
                "delta_s": route_delta_s,
                "required_delta_s": required_delta_s,
                "robot_path_half_extent": robot_extent,
                "footprint_clearance": clearance,
                "candidate_duration": candidate_duration,
                "candidate_frames": state.candidate_frames,
                "candidate_advance": candidate_advance,
            }
            state.candidate_start_sec = None
            state.candidate_start_s = None
            state.candidate_frames = 0
            return PassageUpdate(
                entered_tombstone=True, projection_valid=True
            )

    state.debug = {
        "projection_valid": True,
        "s_robot": robot_projection.s,
        "s_disc": disc_projection.s,
        "delta_s": route_delta_s,
        "required_delta_s": required_delta_s,
        "robot_path_half_extent": robot_extent,
        "footprint_clearance": clearance,
        "candidate_duration": candidate_duration,
        "candidate_frames": state.candidate_frames,
        "candidate_advance": candidate_advance,
    }
    return PassageUpdate(projection_valid=True)


def tombstone_should_reactivate(
    state: PassageState,
    robot_fixed_xy: Tuple[float, float],
    config: PassageConfig,
) -> bool:
    if (
        state.phase != PHASE_TOMBSTONE
        or not state.encounter_path
        or state.disc_projection is None
    ):
        return False
    projection = project_point_to_polyline(
        robot_fixed_xy,
        state.encounter_path,
        state.encounter_cumulative_s,
        # A tombstone can remain downstream for many seconds before the
        # robot reverses. Search the complete short encounter path here;
        # limiting the search around the last downstream segment would make
        # the configured upstream reactivation distance unreachable.
        start_segment=None,
        max_distance_m=config.projection_max_distance_m,
    )
    return (
        projection.valid
        and projection.s
        <= state.disc_projection.s
        - config.tombstone_reactivate_backtrack_m
    )
