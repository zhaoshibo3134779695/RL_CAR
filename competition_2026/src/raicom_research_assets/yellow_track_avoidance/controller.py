"""Low-speed pure-pursuit controller with acceleration and speed limits."""

import math

import numpy as np

from .model import ControlCommand, MotionState, PlanningResult


class PathController:
    def __init__(self, config):
        self.cfg = config["control"]
        self.previous = ControlCommand()
        self.previous_stamp = None

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    def reset(self) -> None:
        self.previous = ControlCommand()
        self.previous_stamp = None

    def _state_speed(self, state: MotionState) -> float:
        if state == MotionState.APPROACH:
            return float(self.cfg["approach_speed_mps"])
        if state == MotionState.AVOID:
            return float(self.cfg["avoid_speed_mps"])
        if state == MotionState.RETURN:
            return float(self.cfg["return_speed_mps"])
        return float(self.cfg["max_linear_speed_mps"])

    def command(self, planning: PlanningResult, state: MotionState, stamp_sec: float) -> ControlCommand:
        if state == MotionState.STOP or not planning.safe or planning.selected is None:
            self.reset()
            return ControlCommand()
        points = planning.selected.points_m
        if points.size == 0:
            self.reset()
            return ControlCommand()
        lookahead = float(self.cfg["lookahead_distance_m"])
        target_index = int(np.searchsorted(points[:, 0], lookahead, side="left"))
        target_index = min(max(0, target_index), len(points) - 1)
        target_x, target_y = float(points[target_index, 0]), float(points[target_index, 1])
        distance_sq = max(1e-6, target_x * target_x + target_y * target_y)
        curvature = 2.0 * target_y / distance_sq
        heading_error = math.atan2(target_y, max(1e-6, target_x))
        speed = self._state_speed(state)
        speed /= 1.0 + float(self.cfg["curvature_slowdown_gain"]) * abs(curvature)
        speed = self._clamp(speed, 0.0, float(self.cfg["max_linear_speed_mps"]))
        angular = speed * curvature + float(self.cfg["heading_gain"]) * heading_error
        angular = self._clamp(
            angular,
            -float(self.cfg["max_angular_speed_radps"]),
            float(self.cfg["max_angular_speed_radps"]),
        )

        if self.previous_stamp is not None:
            dt = max(0.0, min(0.5, stamp_sec - self.previous_stamp))
            dv = float(self.cfg["max_linear_accel_mps2"]) * dt
            dw = float(self.cfg["max_angular_accel_radps2"]) * dt
            speed = self._clamp(speed, self.previous.linear_x - dv, self.previous.linear_x + dv)
            angular = self._clamp(angular, self.previous.angular_z - dw, self.previous.angular_z + dw)
        command = ControlCommand(linear_x=speed, angular_z=angular)
        self.previous = command
        self.previous_stamp = stamp_sec
        return command
