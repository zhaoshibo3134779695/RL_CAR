"""Vision-referenced terminal alignment and stop control.

This module has no ROS dependencies so the terminal state machine and its
footprint-clearance checks can be validated without commanding the vehicle.
"""

from dataclasses import dataclass
import math
from typing import Optional, Sequence, Tuple


STATE_NORMAL = "normal_nav"
STATE_PREALIGN = "terminal_prealign"
STATE_FINAL_CREEP = "terminal_final_creep"
STATE_DWA_FALLBACK = "terminal_dwa_fallback"
STATE_LATCHED = "terminal_stop_latched"


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(float(upper), max(float(lower), float(value)))


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(float(angle)), math.cos(float(angle)))


def goal_frame_errors(
    robot_xy: Tuple[float, float],
    goal_xy: Tuple[float, float],
    goal_yaw: float,
) -> Tuple[float, float, float]:
    """Return Euclidean, forward-to-goal and goal-frame lateral errors."""
    delta_x = float(goal_xy[0]) - float(robot_xy[0])
    delta_y = float(goal_xy[1]) - float(robot_xy[1])
    cosine = math.cos(float(goal_yaw))
    sine = math.sin(float(goal_yaw))
    remaining_m = delta_x * cosine + delta_y * sine
    lateral_error_m = -delta_x * sine + delta_y * cosine
    return math.hypot(delta_x, delta_y), remaining_m, lateral_error_m


def fit_left_boundary_line(
    points: Sequence[Tuple[float, float]],
) -> Tuple[float, float]:
    """Fit ``left = slope * forward + intercept`` in base_link.

    The returned distance is the signed perpendicular distance from the robot
    centre to the left yellow centreline.  Heading is the yellow-line angle in
    base_link, positive counter-clockwise.
    """
    finite = [
        (float(point[0]), float(point[1]))
        for point in points
        if len(point) >= 2
        and math.isfinite(float(point[0]))
        and math.isfinite(float(point[1]))
    ]
    if len(finite) < 2:
        raise ValueError("left boundary fit requires at least two points")
    count = float(len(finite))
    mean_x = sum(point[0] for point in finite) / count
    mean_y = sum(point[1] for point in finite) / count
    denominator = sum((point[0] - mean_x) ** 2 for point in finite)
    if denominator <= 1e-9:
        raise ValueError("left boundary forward span is too small")
    slope = sum(
        (point[0] - mean_x) * (point[1] - mean_y)
        for point in finite
    ) / denominator
    intercept = mean_y - slope * mean_x
    normalizer = math.sqrt(1.0 + slope * slope)
    return intercept / normalizer, math.atan(slope)


def terminal_sweep_clearances(
    left_distance_m: float,
    yaw_error_rad: float,
    vehicle_length_m: float,
    vehicle_width_m: float,
    corridor_width_m: float,
    boundary_margin_m: float,
) -> Tuple[float, float]:
    """Return left/right clearance for the rotated rectangular footprint."""
    projected_half_width = (
        0.5 * float(vehicle_width_m) * abs(math.cos(yaw_error_rad))
        + 0.5 * float(vehicle_length_m) * abs(math.sin(yaw_error_rad))
    )
    left_clearance = (
        float(left_distance_m)
        - float(boundary_margin_m)
        - projected_half_width
    )
    right_clearance = (
        float(corridor_width_m)
        - float(left_distance_m)
        - float(boundary_margin_m)
        - projected_half_width
    )
    return left_clearance, right_clearance


@dataclass(frozen=True)
class TerminalLineObservation:
    stamp_sec: float
    left_distance_m: float
    yaw_error_rad: float
    odom_yaw_rad: float = 0.0


@dataclass(frozen=True)
class TerminalConfig:
    takeover_remaining_m: float = 0.12
    required_line_frames: int = 3
    line_propagation_sec: float = 0.25
    line_wait_sec: float = 0.80
    propagated_command_scale: float = 0.65

    left_reference_m: float = 0.67
    corridor_width_m: float = 0.91
    vehicle_length_m: float = 0.34
    vehicle_width_m: float = 0.33
    boundary_margin_m: float = 0.015
    minimum_sweep_clearance_m: float = 0.020
    clearance_translation_mps: float = 0.008

    prealign_lateral_gain: float = 1.20
    prealign_yaw_gain: float = 1.60
    prealign_max_lateral_mps: float = 0.020
    prealign_max_angular_radps: float = 0.080
    aligned_yaw_rad: float = math.radians(1.5)
    aligned_lateral_m: float = 0.008
    aligned_linear_speed_mps: float = 0.015
    aligned_angular_speed_radps: float = 0.030
    aligned_stable_sec: float = 0.40

    final_lateral_gain: float = 0.80
    final_yaw_gain: float = 1.00
    final_max_lateral_mps: float = 0.012
    final_max_angular_radps: float = 0.050
    realign_yaw_rad: float = math.radians(2.5)
    realign_lateral_m: float = 0.015
    realign_min_remaining_m: float = 0.030

    stop_min_remaining_m: float = -0.005
    stop_max_remaining_m: float = 0.015
    stop_lateral_m: float = 0.008
    stop_yaw_rad: float = math.radians(1.5)
    stop_linear_speed_mps: float = 0.010
    stop_angular_speed_radps: float = 0.025
    stop_stable_sec: float = 0.40
    overrun_m: float = 0.025

    line_loss_straight_speed_mps: float = 0.010
    line_loss_max_advance_m: float = 0.050


@dataclass(frozen=True)
class TerminalDecision:
    state: str
    owns_output: bool
    command_body: Tuple[float, float, float]
    latch_reason: Optional[str]
    remaining_m: Optional[float]
    goal_lateral_m: Optional[float]
    visual_lateral_error_m: Optional[float]
    visual_yaw_error_rad: Optional[float]
    line_mode: str
    valid_line_frames: int
    left_clearance_m: Optional[float]
    right_clearance_m: Optional[float]
    stable_sec: float


class TerminalVisualController:
    """Four-state terminal controller with an independent overrun guard."""

    def __init__(self, config: TerminalConfig):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.state = STATE_NORMAL
        self._line_frames = 0
        self._last_counted_stamp = None
        self._aligned_since = None
        self._stop_since = None
        self._aligned_once = False
        self._line_loss_start_remaining = None
        self._latch_reason = None

    def note_invalid_line_frame(self) -> None:
        """Break the consecutive-frame count but retain short propagation."""
        self._line_frames = 0

    def hold(
        self,
        line_mode: str,
        remaining_m: Optional[float],
        goal_lateral_m: Optional[float],
    ) -> TerminalDecision:
        """Keep an active terminal takeover stopped while inputs are stale."""
        owns_output = self.state in (STATE_PREALIGN, STATE_FINAL_CREEP)
        self._aligned_since = None
        self._stop_since = None
        return self._decision(
            owns_output,
            (0.0, 0.0, 0.0),
            remaining_m,
            goal_lateral_m,
            str(line_mode),
        )

    def _decision(
        self,
        owns_output: bool,
        command: Tuple[float, float, float],
        remaining_m: Optional[float],
        goal_lateral_m: Optional[float],
        line_mode: str,
        lateral_error_m: Optional[float] = None,
        yaw_error_rad: Optional[float] = None,
        left_clearance_m: Optional[float] = None,
        right_clearance_m: Optional[float] = None,
        stable_sec: float = 0.0,
    ) -> TerminalDecision:
        return TerminalDecision(
            state=self.state,
            owns_output=bool(owns_output),
            command_body=tuple(float(value) for value in command),
            latch_reason=self._latch_reason,
            remaining_m=remaining_m,
            goal_lateral_m=goal_lateral_m,
            visual_lateral_error_m=lateral_error_m,
            visual_yaw_error_rad=yaw_error_rad,
            line_mode=line_mode,
            valid_line_frames=self._line_frames,
            left_clearance_m=left_clearance_m,
            right_clearance_m=right_clearance_m,
            stable_sec=max(0.0, float(stable_sec)),
        )

    def _latch(
        self,
        reason: str,
        remaining_m: Optional[float],
        goal_lateral_m: Optional[float],
    ) -> TerminalDecision:
        self.state = STATE_LATCHED
        self._latch_reason = str(reason)
        return self._decision(
            True,
            (0.0, 0.0, 0.0),
            remaining_m,
            goal_lateral_m,
            "latched",
        )

    def _line_errors(
        self,
        now_sec: float,
        odom_yaw_rad: float,
        line: Optional[TerminalLineObservation],
    ):
        if line is None:
            self._line_frames = 0
            return "lost", None, None, None, None, 0.0
        age_sec = max(0.0, float(now_sec) - float(line.stamp_sec))
        if age_sec <= self.config.line_propagation_sec:
            new_frame = self._last_counted_stamp != float(line.stamp_sec)
            if new_frame:
                self._line_frames += 1
                self._last_counted_stamp = float(line.stamp_sec)
            yaw_delta = _wrap_angle(
                float(odom_yaw_rad) - float(line.odom_yaw_rad)
            )
            yaw_error = _wrap_angle(float(line.yaw_error_rad) - yaw_delta)
            lateral_error = (
                float(line.left_distance_m)
                - self.config.left_reference_m
            )
            left_clearance, right_clearance = terminal_sweep_clearances(
                left_distance_m=float(line.left_distance_m),
                yaw_error_rad=yaw_error,
                vehicle_length_m=self.config.vehicle_length_m,
                vehicle_width_m=self.config.vehicle_width_m,
                corridor_width_m=self.config.corridor_width_m,
                boundary_margin_m=self.config.boundary_margin_m,
            )
            return (
                "live" if new_frame else "short_odom_propagation",
                lateral_error,
                yaw_error,
                left_clearance,
                right_clearance,
                1.0 if new_frame else self.config.propagated_command_scale,
            )
        self._line_frames = 0
        if age_sec <= self.config.line_wait_sec:
            return "waiting_for_live_line", None, None, None, None, 0.0
        return "lost", None, None, None, None, 0.0

    def _alignment_command(
        self,
        lateral_error_m: float,
        yaw_error_rad: float,
        left_clearance_m: float,
        right_clearance_m: float,
        final: bool,
        scale: float,
    ) -> Tuple[float, float, float]:
        if final:
            lateral_gain = self.config.final_lateral_gain
            yaw_gain = self.config.final_yaw_gain
            maximum_lateral = self.config.final_max_lateral_mps
            maximum_angular = self.config.final_max_angular_radps
        else:
            lateral_gain = self.config.prealign_lateral_gain
            yaw_gain = self.config.prealign_yaw_gain
            maximum_lateral = self.config.prealign_max_lateral_mps
            maximum_angular = self.config.prealign_max_angular_radps
        lateral = _clamp(
            lateral_gain * lateral_error_m,
            -maximum_lateral,
            maximum_lateral,
        )
        angular = _clamp(
            yaw_gain * yaw_error_rad,
            -maximum_angular,
            maximum_angular,
        )
        if min(left_clearance_m, right_clearance_m) < (
            self.config.minimum_sweep_clearance_m
        ):
            angular = 0.0
            if left_clearance_m < right_clearance_m:
                lateral = min(
                    lateral, -self.config.clearance_translation_mps
                )
            else:
                lateral = max(
                    lateral, self.config.clearance_translation_mps
                )
        return 0.0, lateral * scale, angular * scale

    def update(
        self,
        *,
        now_sec: float,
        armed: bool,
        takeover_allowed: bool,
        remaining_m: Optional[float],
        goal_lateral_m: Optional[float],
        odom_yaw_rad: float,
        linear_speed_mps: float,
        angular_speed_radps: float,
        line: Optional[TerminalLineObservation],
    ) -> TerminalDecision:
        if not armed or remaining_m is None:
            self.reset()
            return self._decision(
                False,
                (0.0, 0.0, 0.0),
                remaining_m,
                goal_lateral_m,
                "disabled",
            )
        if self.state == STATE_LATCHED:
            return self._decision(
                True,
                (0.0, 0.0, 0.0),
                remaining_m,
                goal_lateral_m,
                "latched",
            )
        if float(remaining_m) <= -self.config.overrun_m:
            return self._latch(
                "terminal_overrun_guard", remaining_m, goal_lateral_m
            )

        (
            line_mode,
            lateral_error,
            yaw_error,
            left_clearance,
            right_clearance,
            command_scale,
        ) = self._line_errors(now_sec, odom_yaw_rad, line)

        if self.state == STATE_NORMAL:
            if (
                takeover_allowed
                and float(remaining_m)
                <= self.config.takeover_remaining_m
                and self._line_frames >= self.config.required_line_frames
                and lateral_error is not None
            ):
                self.state = STATE_PREALIGN
                self._aligned_since = None
            else:
                return self._decision(
                    False,
                    (0.0, 0.0, 0.0),
                    remaining_m,
                    goal_lateral_m,
                    line_mode,
                    lateral_error,
                    yaw_error,
                    left_clearance,
                    right_clearance,
                )

        if self.state == STATE_DWA_FALLBACK:
            return self._decision(
                False,
                (0.0, 0.0, 0.0),
                remaining_m,
                goal_lateral_m,
                line_mode,
                lateral_error,
                yaw_error,
                left_clearance,
                right_clearance,
            )

        if lateral_error is None or yaw_error is None:
            self._aligned_since = None
            self._stop_since = None
            if line_mode == "waiting_for_live_line":
                return self._decision(
                    True,
                    (0.0, 0.0, 0.0),
                    remaining_m,
                    goal_lateral_m,
                    line_mode,
                )
            if self.state == STATE_FINAL_CREEP and self._aligned_once:
                if float(remaining_m) <= 0.0:
                    return self._latch(
                        "terminal_line_loss_goal_plane",
                        remaining_m,
                        goal_lateral_m,
                    )
                if self._line_loss_start_remaining is None:
                    self._line_loss_start_remaining = float(remaining_m)
                advanced = (
                    self._line_loss_start_remaining - float(remaining_m)
                )
                if advanced <= self.config.line_loss_max_advance_m:
                    return self._decision(
                        True,
                        (
                            self.config.line_loss_straight_speed_mps,
                            0.0,
                            0.0,
                        ),
                        remaining_m,
                        goal_lateral_m,
                        "lost_aligned_straight",
                    )
            self.state = STATE_DWA_FALLBACK
            return self._decision(
                False,
                (0.0, 0.0, 0.0),
                remaining_m,
                goal_lateral_m,
                "lost_dwa_fallback",
            )
        self._line_loss_start_remaining = None

        if self.state == STATE_PREALIGN:
            aligned = (
                abs(yaw_error) <= self.config.aligned_yaw_rad
                and abs(lateral_error)
                <= self.config.aligned_lateral_m
                and float(linear_speed_mps)
                <= self.config.aligned_linear_speed_mps
                and abs(float(angular_speed_radps))
                <= self.config.aligned_angular_speed_radps
            )
            if aligned:
                if self._aligned_since is None:
                    self._aligned_since = float(now_sec)
                stable_sec = float(now_sec) - self._aligned_since
            else:
                self._aligned_since = None
                stable_sec = 0.0
            command = self._alignment_command(
                lateral_error,
                yaw_error,
                left_clearance,
                right_clearance,
                False,
                command_scale,
            )
            if aligned and stable_sec >= self.config.aligned_stable_sec:
                self.state = STATE_FINAL_CREEP
                self._aligned_once = True
                self._stop_since = None
            return self._decision(
                True,
                command,
                remaining_m,
                goal_lateral_m,
                line_mode,
                lateral_error,
                yaw_error,
                left_clearance,
                right_clearance,
                stable_sec,
            )

        if (
            float(remaining_m) > self.config.realign_min_remaining_m
            and (
                abs(yaw_error) > self.config.realign_yaw_rad
                or abs(lateral_error) > self.config.realign_lateral_m
            )
        ):
            self.state = STATE_PREALIGN
            self._aligned_since = None
            self._stop_since = None
            command = self._alignment_command(
                lateral_error,
                yaw_error,
                left_clearance,
                right_clearance,
                False,
                command_scale,
            )
            return self._decision(
                True,
                command,
                remaining_m,
                goal_lateral_m,
                line_mode,
                lateral_error,
                yaw_error,
                left_clearance,
                right_clearance,
            )

        stop_ready = (
            self.config.stop_min_remaining_m
            <= float(remaining_m)
            <= self.config.stop_max_remaining_m
            and abs(lateral_error) <= self.config.stop_lateral_m
            and abs(yaw_error) <= self.config.stop_yaw_rad
            and float(linear_speed_mps)
            <= self.config.stop_linear_speed_mps
            and abs(float(angular_speed_radps))
            <= self.config.stop_angular_speed_radps
        )
        if stop_ready:
            if self._stop_since is None:
                self._stop_since = float(now_sec)
            stable_sec = float(now_sec) - self._stop_since
            if stable_sec >= self.config.stop_stable_sec:
                return self._latch(
                    "terminal_precision_reached",
                    remaining_m,
                    goal_lateral_m,
                )
        else:
            self._stop_since = None
            stable_sec = 0.0

        if float(remaining_m) > 0.05:
            forward = 0.025
        elif float(remaining_m) > 0.02:
            forward = 0.015
        elif float(remaining_m) >= self.config.stop_min_remaining_m:
            forward = 0.009
        else:
            forward = 0.0
        if min(left_clearance, right_clearance) < (
            self.config.minimum_sweep_clearance_m
        ):
            forward = 0.0
        _, lateral, angular = self._alignment_command(
            lateral_error,
            yaw_error,
            left_clearance,
            right_clearance,
            True,
            command_scale,
        )
        return self._decision(
            True,
            (forward * command_scale, lateral, angular),
            remaining_m,
            goal_lateral_m,
            line_mode,
            lateral_error,
            yaw_error,
            left_clearance,
            right_clearance,
            stable_sec,
        )
