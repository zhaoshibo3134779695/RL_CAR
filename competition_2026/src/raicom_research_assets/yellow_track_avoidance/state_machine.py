"""Debounced obstacle-avoidance safety state machine."""

from typing import List

from .model import MotionState, PlanningResult, StateDecision, TrackedDisk


class AvoidanceStateMachine:
    def __init__(self, config):
        self.config = config
        self.cfg = config["state_machine"]
        self.safety = config["safety"]
        self.state = MotionState.STOP
        self.stop_reason = "startup_not_ready"
        self.stop_latched = bool(self.safety["stop_is_latched"])
        self.reset_requested = False
        self.locked_direction = 0
        self.state_entered_sec = 0.0
        self.obstacle_frames = 0
        self.clear_frames = 0
        self.return_frames = 0
        self.safe_recovery_frames = 0

    def force_stop(self, reason: str, now_sec: float) -> StateDecision:
        changed = self.state != MotionState.STOP or self.stop_reason != reason
        self.state = MotionState.STOP
        self.stop_reason = reason
        self.state_entered_sec = now_sec
        return StateDecision(self.state, self.stop_reason, self.locked_direction, changed)

    def request_reset(self) -> None:
        self.reset_requested = True

    def _transition(self, state: MotionState, now_sec: float, reason: str = "") -> bool:
        changed = self.state != state
        if changed:
            self.state = state
            self.state_entered_sec = now_sec
            self.clear_frames = 0
            self.return_frames = 0
        self.stop_reason = reason if state == MotionState.STOP else ""
        return changed

    def update(
        self,
        now_sec: float,
        calibration_ok: bool,
        boundary_ok: bool,
        boundary_loss_frames: int,
        planning: PlanningResult,
        obstacles: List[TrackedDisk],
        processing_ok: bool = True,
    ) -> StateDecision:
        if not processing_ok:
            return self.force_stop("image_processing_error", now_sec)
        if not calibration_ok:
            return self.force_stop("calibration_incomplete", now_sec)
        if boundary_loss_frames >= int(self.config["boundary"]["both_side_stop_frames"]) or not boundary_ok:
            return self.force_stop("yellow_boundaries_lost", now_sec)
        if not planning.safe:
            return self.force_stop(planning.reason or "no_safe_path", now_sec)

        forward_offset = float(
            self.config["physical"].get(
                "local_forward_origin_offset_m", 0.0
            )
        )
        forward_obstacles = [
            disk
            for disk in obstacles
            if disk.center_m[0] + forward_offset >= 0.0
        ]
        nearest = min(
            (
                disk.center_m[0] + forward_offset
                for disk in forward_obstacles
            ),
            default=float("inf"),
        )
        expanded_radius = (
            0.5 * float(self.config["physical"]["disk_diameter_m"])
            + 0.5 * float(self.config["physical"]["robot_width_m"])
            + float(self.safety["obstacle_margin_m"])
        )
        nearest_clearance = nearest - expanded_radius
        if nearest_clearance <= float(self.safety["emergency_obstacle_distance_m"]):
            return self.force_stop("obstacle_too_close", now_sec)

        safe_now = calibration_ok and boundary_ok and planning.safe
        if self.state == MotionState.STOP:
            self.safe_recovery_frames = self.safe_recovery_frames + 1 if safe_now else 0
            can_recover = self.safe_recovery_frames >= int(self.safety["stop_recovery_frames"])
            if can_recover and (not self.stop_latched or self.reset_requested):
                self.reset_requested = False
                self.safe_recovery_frames = 0
                self.locked_direction = 0
                changed = self._transition(MotionState.NORMAL, now_sec)
                return StateDecision(self.state, self.stop_reason, self.locked_direction, changed)
            return StateDecision(self.state, self.stop_reason, self.locked_direction, False)

        obstacle_present = bool(forward_obstacles) and nearest <= float(self.cfg["approach_distance_m"])
        self.obstacle_frames = self.obstacle_frames + 1 if obstacle_present else 0
        self.clear_frames = 0 if obstacle_present else self.clear_frames + 1
        changed = False

        if self.state == MotionState.NORMAL:
            if self.obstacle_frames >= int(self.cfg["approach_confirm_frames"]):
                if planning.selected is None:
                    return self.force_stop("no_safe_approach_path", now_sec)
                self.locked_direction = planning.selected.direction
                changed = self._transition(MotionState.APPROACH, now_sec)
        elif self.state == MotionState.APPROACH:
            if not obstacle_present:
                self.locked_direction = 0
                changed = self._transition(MotionState.NORMAL, now_sec)
            elif nearest <= float(self.cfg["avoid_start_distance_m"]):
                self.return_frames += 1
                if self.return_frames >= int(self.cfg["avoid_confirm_frames"]):
                    changed = self._transition(MotionState.AVOID, now_sec)
        elif self.state == MotionState.AVOID:
            minimum_time_done = now_sec - self.state_entered_sec >= float(self.cfg["minimum_avoid_time_sec"])
            if minimum_time_done and self.clear_frames >= int(self.cfg["clear_confirm_frames"]):
                changed = self._transition(MotionState.RETURN, now_sec)
        elif self.state == MotionState.RETURN:
            if obstacle_present:
                changed = self._transition(MotionState.AVOID, now_sec)
            else:
                selected_offset = 0.0 if planning.selected is None else abs(planning.selected.offset_m)
                if selected_offset <= float(self.cfg["center_tolerance_m"]):
                    self.return_frames += 1
                else:
                    self.return_frames = 0
                timed_out = now_sec - self.state_entered_sec >= float(self.cfg["return_timeout_sec"])
                if self.return_frames >= int(self.cfg["return_confirm_frames"]) or timed_out:
                    self.locked_direction = 0
                    changed = self._transition(MotionState.NORMAL, now_sec)
        return StateDecision(self.state, self.stop_reason, self.locked_direction, changed)
