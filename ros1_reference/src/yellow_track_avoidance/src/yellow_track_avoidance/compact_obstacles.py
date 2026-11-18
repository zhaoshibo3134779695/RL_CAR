"""Build the semantic obstacles used by the compact TEB profile.

This module deliberately has no ROS dependency.  It validates that the locked
corridor and the disc bridge describe the same complete mission scene before
returning two yellow-line inner edges and two frozen circular discs.  Stage 1
can additionally close the narrower disc-to-line gap with one finite line per
disc.  This makes the global costmap and TEB agree on which side of a disc is
traversable without adding another controller.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CompactObstacle:
    obstacle_id: int
    kind: str
    points: tuple
    radius_m: float = 0.0


@dataclass(frozen=True)
class CompactObstacleScene:
    scene_generation: int
    frame_id: str
    obstacles: tuple
    passage_decisions: tuple = ()


@dataclass(frozen=True)
class PassageDecision:
    memory_id: int
    center: tuple
    longitudinal_m: float
    left_gap_m: float
    right_gap_m: float
    required_gap_m: float
    pass_side: str
    blocked_side: str
    closure: tuple = ()


class CompactSceneError(ValueError):
    """The downstream world model is not yet complete or self-consistent."""


def select_teb_obstacles(scene, include_passage_closures=True):
    """Return physical obstacles plus optional global-topology closures."""

    if include_passage_closures:
        return tuple(scene.obstacles)
    closure_ids = {
        3000 + decision.memory_id
        for decision in scene.passage_decisions
        if decision.closure
    }
    return tuple(
        obstacle
        for obstacle in scene.obstacles
        if obstacle.obstacle_id not in closure_ids
    )


def sample_filled_disc_points(scene, resolution_m=0.005):
    """Rasterize every circular obstacle as a filled metric point cloud.

    A virtual LaserScan contains only the first visible surface of a disc. It
    is useful for local sensing, but it is not a complete collision model after
    the robot passes the obstacle. GlobalPlanner and SBPL profiles can consume
    this filled representation so footprint checks see the complete guarded
    disc without adding semantic passage closures.
    """

    try:
        resolution = float(resolution_m)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CompactSceneError("disc cloud resolution is invalid") from exc
    if not math.isfinite(resolution) or resolution <= 0.0:
        raise CompactSceneError("disc cloud resolution must be positive")

    points = []
    for obstacle in scene.obstacles:
        if obstacle.kind != "circle":
            continue
        if len(obstacle.points) != 1 or obstacle.radius_m <= 0.0:
            raise CompactSceneError("circular obstacle geometry is invalid")
        center_x, center_y = obstacle.points[0]
        radius = float(obstacle.radius_m)
        steps = int(math.ceil(radius / resolution))
        radius_squared = radius * radius + 1e-12
        for x_index in range(-steps, steps + 1):
            offset_x = x_index * resolution
            for y_index in range(-steps, steps + 1):
                offset_y = y_index * resolution
                if offset_x * offset_x + offset_y * offset_y <= radius_squared:
                    points.append(
                        (center_x + offset_x, center_y + offset_y, 0.0)
                    )
    return tuple(points)


def _finite_point(value, label):
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise CompactSceneError("{} is not a 2-D point".format(label))
    try:
        point = float(value[0]), float(value[1])
    except (TypeError, ValueError, OverflowError) as exc:
        raise CompactSceneError("{} is not numeric".format(label)) from exc
    if not all(math.isfinite(item) for item in point):
        raise CompactSceneError("{} is not finite".format(label))
    return point


def _segment(payload, name):
    values = payload.get(name)
    if not isinstance(values, list) or len(values) != 2:
        raise CompactSceneError("{} must contain exactly two endpoints".format(name))
    return (
        _finite_point(values[0], "{}[0]".format(name)),
        _finite_point(values[1], "{}[1]".format(name)),
    )


def _inner_edges(left, right, half_width_m):
    left_inner = []
    right_inner = []
    for index, (left_point, right_point) in enumerate(zip(left, right)):
        across_x = right_point[0] - left_point[0]
        across_y = right_point[1] - left_point[1]
        separation = math.hypot(across_x, across_y)
        if separation <= 2.0 * half_width_m:
            raise CompactSceneError(
                "boundary separation at endpoint {} is invalid".format(index)
            )
        unit_x = across_x / separation
        unit_y = across_y / separation
        left_inner.append(
            (
                left_point[0] + half_width_m * unit_x,
                left_point[1] + half_width_m * unit_y,
            )
        )
        right_inner.append(
            (
                right_point[0] - half_width_m * unit_x,
                right_point[1] - half_width_m * unit_y,
            )
        )
    return tuple(left_inner), tuple(right_inner)


def _unit(vector, label):
    length = math.hypot(vector[0], vector[1])
    if length <= 1e-9:
        raise CompactSceneError("{} has zero length".format(label))
    return vector[0] / length, vector[1] / length


def _passage_decisions(
    left_inner,
    right_inner,
    circles,
    vehicle_width_m,
    total_clearance_m,
    side_tie_tolerance_m,
):
    """Choose the larger physical side-gap and close the narrower one.

    Gap values are edge-to-edge distances in the locked corridor frame.  The
    yellow boundary inputs are already their physical inner edges, so line
    width is never counted as free space here.
    """

    values = (vehicle_width_m, total_clearance_m, side_tie_tolerance_m)
    if not all(math.isfinite(float(value)) for value in values):
        raise CompactSceneError("passage geometry parameters must be finite")
    if vehicle_width_m <= 0.0 or total_clearance_m < 0.0:
        raise CompactSceneError("vehicle width or passage clearance is invalid")
    if side_tie_tolerance_m < 0.0:
        raise CompactSceneError("side tie tolerance is invalid")

    left_delta = (
        left_inner[1][0] - left_inner[0][0],
        left_inner[1][1] - left_inner[0][1],
    )
    right_delta = (
        right_inner[1][0] - right_inner[0][0],
        right_inner[1][1] - right_inner[0][1],
    )
    left_tangent = _unit(left_delta, "left inner boundary")
    right_tangent = _unit(right_delta, "right inner boundary")
    if left_tangent[0] * right_tangent[0] + left_tangent[1] * right_tangent[1] < 0.0:
        right_tangent = -right_tangent[0], -right_tangent[1]
    tangent = _unit(
        (
            left_tangent[0] + right_tangent[0],
            left_tangent[1] + right_tangent[1],
        ),
        "corridor tangent",
    )
    left_start = left_inner[0]
    right_start = right_inner[0]
    across = _unit(
        (
            right_start[0] - left_start[0],
            right_start[1] - left_start[1],
        ),
        "corridor inner width",
    )
    required_gap = float(vehicle_width_m) + float(total_clearance_m)
    corridor_length = 0.5 * (
        math.hypot(*left_delta) + math.hypot(*right_delta)
    )

    decisions = []
    for circle in circles:
        center = circle.points[0]
        relative = center[0] - left_start[0], center[1] - left_start[1]
        longitudinal = relative[0] * tangent[0] + relative[1] * tangent[1]
        clamped = min(corridor_length, max(0.0, longitudinal))
        left_reference = (
            left_start[0] + clamped * tangent[0],
            left_start[1] + clamped * tangent[1],
        )
        right_reference = (
            right_start[0] + clamped * tangent[0],
            right_start[1] + clamped * tangent[1],
        )
        width_vector = (
            right_reference[0] - left_reference[0],
            right_reference[1] - left_reference[1],
        )
        inner_width = width_vector[0] * across[0] + width_vector[1] * across[1]
        lateral = (
            (center[0] - left_reference[0]) * across[0]
            + (center[1] - left_reference[1]) * across[1]
        )
        left_gap = lateral - circle.radius_m
        right_gap = inner_width - lateral - circle.radius_m
        if left_gap < -circle.radius_m or right_gap < -circle.radius_m:
            raise CompactSceneError(
                "disc {} lies outside the locked corridor".format(
                    circle.obstacle_id - 2000
                )
            )

        memory_id = circle.obstacle_id - 2000
        left_feasible = left_gap >= required_gap
        right_feasible = right_gap >= required_gap
        # A closure is only needed when exactly one side satisfies the complete
        # vehicle/clearance requirement.  If both sides are feasible, leave
        # both topology choices open so GlobalPlanner can account for the
        # second disc and the remaining route.  If neither side reaches the
        # nominal recommendation, competition policy still forbids a software
        # stop: force the wider side unless the two measurements are tied.
        if left_feasible and right_feasible:
            pass_side = "either"
            blocked_side = "none"
            closure = ()
        elif left_feasible:
            pass_side = "left"
            blocked_side = "right"
            closure = (center, right_reference)
        elif right_feasible:
            pass_side = "right"
            blocked_side = "left"
            closure = (left_reference, center)
        elif abs(left_gap - right_gap) <= side_tie_tolerance_m:
            pass_side = "either"
            blocked_side = "none"
            closure = ()
        elif left_gap > right_gap:
            pass_side = "left"
            blocked_side = "right"
            closure = (center, right_reference)
        else:
            pass_side = "right"
            blocked_side = "left"
            closure = (left_reference, center)
        decisions.append(
            PassageDecision(
                memory_id=memory_id,
                center=center,
                longitudinal_m=float(longitudinal),
                left_gap_m=float(left_gap),
                right_gap_m=float(right_gap),
                required_gap_m=required_gap,
                pass_side=pass_side,
                blocked_side=blocked_side,
                closure=closure,
            )
        )
    return tuple(sorted(decisions, key=lambda item: item.longitudinal_m))


def build_compact_obstacle_scene(
    boundary_payload,
    disc_status,
    disc_radius_m=0.05,
    disc_collision_radius_m=None,
    yellow_line_half_width_m=0.015,
    enable_passage_closures=False,
    vehicle_width_m=0.33,
    passage_total_clearance_m=0.02,
    side_tie_tolerance_m=0.01,
    minimum_disc_count=2,
):
    """Return the locked corridor/disc model and optional gap closures.

    ``CompactSceneError`` is raised until both downstream latches are complete
    for the same explicit mission generation.  This prevents a transient
    one-disc or old-corridor scene from reaching TEB.
    """

    if not isinstance(boundary_payload, dict) or not isinstance(disc_status, dict):
        raise CompactSceneError("corridor and disc status are both required")
    try:
        corridor_generation = int(boundary_payload["scene_generation"])
        disc_generation = int(disc_status["scene_generation"])
        radius = float(disc_radius_m)
        collision_radius = float(
            radius
            if disc_collision_radius_m is None
            else disc_collision_radius_m
        )
        half_width = float(yellow_line_half_width_m)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise CompactSceneError("invalid scene generation or geometry parameter") from exc
    if corridor_generation != disc_generation:
        raise CompactSceneError("corridor/disc scene generation mismatch")
    if (
        radius <= 0.0
        or collision_radius < radius
        or not math.isfinite(collision_radius)
        or half_width < 0.0
    ):
        raise CompactSceneError("disc radius and yellow half-width are invalid")
    if not bool(boundary_payload.get("boundary_pair_valid", False)):
        raise CompactSceneError("corridor boundary pair is not locked")
    if not bool(disc_status.get("confirmed_positions_frozen", False)):
        raise CompactSceneError("disc positions are not mission-frozen")
    if not bool(disc_status.get("costmap_writes_enabled", False)):
        raise CompactSceneError("disc costmap writes are disabled")
    if not bool(disc_status.get("fixed_transform_available", False)):
        raise CompactSceneError("disc fixed-frame transform is unavailable")
    try:
        minimum_count = max(1, int(minimum_disc_count))
    except (TypeError, ValueError, OverflowError) as exc:
        raise CompactSceneError("minimum disc count is invalid") from exc

    tracks = disc_status.get("active_tracks")
    if not isinstance(tracks, list) or len(tracks) < minimum_count:
        raise CompactSceneError(
            "active_tracks must contain at least {} discs".format(minimum_count)
        )
    track_count = len(tracks)
    for field in ("memory_count", "active_memory_count", "disc_count", "costmap_disc_count"):
        try:
            value = int(disc_status.get(field, -1))
        except (TypeError, ValueError, OverflowError) as exc:
            raise CompactSceneError("{} is invalid".format(field)) from exc
        if value != track_count:
            raise CompactSceneError(
                "{} must equal active track count {}".format(field, track_count)
            )

    frame_id = str(boundary_payload.get("frame_id", "")).lstrip("/")
    disc_frame = str(disc_status.get("fixed_frame", "")).lstrip("/")
    if not frame_id or frame_id != disc_frame:
        raise CompactSceneError("corridor and discs do not share a fixed frame")

    left = _segment(boundary_payload, "left_base_m")
    right = _segment(boundary_payload, "right_base_m")
    left_inner, right_inner = _inner_edges(left, right, half_width)

    physical_circles = []
    collision_circles = []
    seen_ids = set()
    for index, track in enumerate(tracks):
        if not isinstance(track, dict) or not bool(track.get("confirmed", False)):
            raise CompactSceneError("active disc {} is not confirmed".format(index))
        try:
            memory_id = int(track["memory_id"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise CompactSceneError("active disc id is invalid") from exc
        if memory_id in seen_ids:
            raise CompactSceneError("active disc ids are not unique")
        seen_ids.add(memory_id)
        center = _finite_point(track.get("fixed_xy"), "active disc fixed_xy")
        physical_circles.append(
            CompactObstacle(
                obstacle_id=2000 + memory_id,
                kind="circle",
                points=(center,),
                radius_m=radius,
            )
        )
        collision_circles.append(
            CompactObstacle(
                obstacle_id=2000 + memory_id,
                kind="circle",
                points=(center,),
                radius_m=collision_radius,
            )
        )

    base_obstacles = (
        CompactObstacle(1001, "line", left_inner),
        CompactObstacle(1002, "line", right_inner),
    ) + tuple(sorted(collision_circles, key=lambda item: item.obstacle_id))
    decisions = ()
    closures = ()
    if enable_passage_closures:
        decisions = _passage_decisions(
            left_inner,
            right_inner,
            physical_circles,
            vehicle_width_m=vehicle_width_m,
            total_clearance_m=passage_total_clearance_m,
            side_tie_tolerance_m=side_tie_tolerance_m,
        )
        closures = tuple(
            CompactObstacle(
                obstacle_id=3000 + decision.memory_id,
                kind="line",
                points=decision.closure,
            )
            for decision in decisions
            if decision.closure
        )
    return CompactObstacleScene(
        scene_generation=corridor_generation,
        frame_id=frame_id,
        obstacles=base_obstacles + closures,
        passage_decisions=decisions,
    )
