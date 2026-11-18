#!/usr/bin/env python3

import math
import unittest

from yellow_track_avoidance.terminal_control import (
    STATE_FINAL_CREEP,
    STATE_LATCHED,
    STATE_NORMAL,
    STATE_PREALIGN,
    TerminalConfig,
    TerminalLineObservation,
    TerminalVisualController,
    fit_left_boundary_line,
    goal_frame_errors,
    terminal_sweep_clearances,
)


def update_controller(
    controller,
    now_sec,
    remaining_m=0.10,
    line=None,
    linear_speed_mps=0.0,
    angular_speed_radps=0.0,
):
    return controller.update(
        now_sec=now_sec,
        armed=True,
        takeover_allowed=True,
        remaining_m=remaining_m,
        goal_lateral_m=0.0,
        odom_yaw_rad=0.0,
        linear_speed_mps=linear_speed_mps,
        angular_speed_radps=angular_speed_radps,
        line=line,
    )


def line(stamp_sec, distance_m=0.67, yaw_deg=0.0):
    return TerminalLineObservation(
        stamp_sec=stamp_sec,
        left_distance_m=distance_m,
        yaw_error_rad=math.radians(yaw_deg),
    )


class TerminalControlTests(unittest.TestCase):
    def test_goal_frame_errors_keep_components_separate(self):
        distance, remaining, lateral = goal_frame_errors(
            robot_xy=(1.90, -0.02),
            goal_xy=(2.0, 0.0),
            goal_yaw=0.0,
        )
        self.assertAlmostEqual(distance, math.hypot(0.10, 0.02))
        self.assertAlmostEqual(remaining, 0.10)
        self.assertAlmostEqual(lateral, 0.02)

    def test_left_boundary_fit_recovers_distance_and_heading(self):
        slope = math.tan(math.radians(5.0))
        points = [
            (x, 0.67 + slope * x)
            for x in (0.35, 0.60, 0.90, 1.20)
        ]
        distance, yaw = fit_left_boundary_line(points)
        self.assertAlmostEqual(
            distance, 0.67 * math.cos(math.radians(5.0)), places=6
        )
        self.assertAlmostEqual(math.degrees(yaw), 5.0, places=6)

    def test_sweep_clearance_blocks_rotation_near_right_boundary(self):
        left_clearance, right_clearance = terminal_sweep_clearances(
            left_distance_m=0.67,
            yaw_error_rad=math.radians(18.0),
            vehicle_length_m=0.34,
            vehicle_width_m=0.33,
            corridor_width_m=0.91,
            boundary_margin_m=0.015,
        )
        self.assertGreater(left_clearance, 0.40)
        self.assertLess(right_clearance, 0.02)

    def test_requires_three_distinct_live_frames_before_takeover(self):
        controller = TerminalVisualController(TerminalConfig())
        first = update_controller(controller, 10.00, line=line(9.99))
        repeated = update_controller(controller, 10.05, line=line(9.99))
        second = update_controller(controller, 10.10, line=line(10.09))
        third = update_controller(controller, 10.20, line=line(10.19))

        self.assertEqual(first.state, STATE_NORMAL)
        self.assertEqual(repeated.state, STATE_NORMAL)
        self.assertEqual(second.state, STATE_NORMAL)
        self.assertEqual(third.state, STATE_PREALIGN)
        self.assertTrue(third.owns_output)

    def test_prealign_moves_away_before_rotating(self):
        controller = TerminalVisualController(TerminalConfig())
        controller.state = STATE_PREALIGN
        decision = update_controller(
            controller,
            20.0,
            line=line(19.99, distance_m=0.67, yaw_deg=18.0),
        )
        self.assertAlmostEqual(decision.command_body[2], 0.0)
        self.assertGreater(decision.command_body[1], 0.0)
        self.assertLess(decision.right_clearance_m, 0.02)

    def test_visual_error_signs_match_base_link_commands(self):
        controller = TerminalVisualController(TerminalConfig())
        controller.state = STATE_PREALIGN
        decision = update_controller(
            controller,
            25.0,
            line=line(24.99, distance_m=0.68, yaw_deg=2.0),
        )
        self.assertGreater(decision.command_body[1], 0.0)
        self.assertGreater(decision.command_body[2], 0.0)

    def test_alignment_must_be_stable_before_final_creep(self):
        controller = TerminalVisualController(TerminalConfig())
        controller.state = STATE_PREALIGN

        initial = update_controller(controller, 30.0, line=line(29.99))
        not_yet = update_controller(controller, 30.39, line=line(30.38))
        stable = update_controller(controller, 30.41, line=line(30.40))

        self.assertEqual(initial.state, STATE_PREALIGN)
        self.assertEqual(not_yet.state, STATE_PREALIGN)
        self.assertEqual(stable.state, STATE_FINAL_CREEP)

    def test_precision_stop_requires_alignment_and_stability(self):
        controller = TerminalVisualController(TerminalConfig())
        controller.state = STATE_FINAL_CREEP

        initial = update_controller(
            controller, 40.0, remaining_m=0.01, line=line(39.99)
        )
        not_yet = update_controller(
            controller, 40.39, remaining_m=0.007, line=line(40.38)
        )
        latched = update_controller(
            controller, 40.41, remaining_m=0.006, line=line(40.40)
        )

        self.assertEqual(initial.state, STATE_FINAL_CREEP)
        self.assertEqual(not_yet.state, STATE_FINAL_CREEP)
        self.assertEqual(latched.state, STATE_LATCHED)
        self.assertEqual(latched.latch_reason, "terminal_precision_reached")
        self.assertEqual(latched.command_body, (0.0, 0.0, 0.0))

    def test_two_centimetres_alone_never_triggers_latch(self):
        controller = TerminalVisualController(TerminalConfig())
        controller.state = STATE_FINAL_CREEP
        misaligned = line(49.99, distance_m=0.64, yaw_deg=4.0)
        decision = update_controller(
            controller, 50.0, remaining_m=0.01, line=misaligned
        )
        self.assertEqual(decision.state, STATE_FINAL_CREEP)
        self.assertIsNone(decision.latch_reason)

    def test_overrun_guard_latches_without_visual_alignment(self):
        controller = TerminalVisualController(TerminalConfig())
        decision = update_controller(
            controller, 60.0, remaining_m=-0.026, line=None
        )
        self.assertEqual(decision.state, STATE_LATCHED)
        self.assertEqual(decision.latch_reason, "terminal_overrun_guard")
        self.assertEqual(decision.command_body, (0.0, 0.0, 0.0))

    def test_aligned_line_loss_stops_at_goal_plane(self):
        controller = TerminalVisualController(TerminalConfig())
        controller.state = STATE_FINAL_CREEP
        controller._aligned_once = True
        decision = update_controller(
            controller, 70.0, remaining_m=-0.001, line=None
        )
        self.assertEqual(decision.state, STATE_LATCHED)
        self.assertEqual(
            decision.latch_reason, "terminal_line_loss_goal_plane"
        )


if __name__ == "__main__":
    unittest.main()
