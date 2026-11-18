#!/usr/bin/env python3

import importlib.util
from pathlib import Path
from unittest import mock
import unittest

from yellow_track_avoidance.narrow_gate import RobotPose
from yellow_track_avoidance.passage_navigation import (
    BoundaryLines,
    PassageConfig,
    PassageDecision,
    SCENE_NORMAL,
    STATE_NORMAL,
)


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "passage_supervisor.py"
)
SPEC = importlib.util.spec_from_file_location(
    "passage_supervisor_recovery_under_test", str(SCRIPT_PATH)
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
PassageSupervisorNode = MODULE.PassageSupervisorNode


class FinishRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.node = object.__new__(PassageSupervisorNode)
        self.node.finish_recovery_enabled = True
        self.node.config = PassageConfig(
            competition_finish_priority=True,
            competition_two_disc_mode=True,
        )
        self.node.robot_pose = RobotPose(0.0, 0.0, 0.0)
        self.node.robot_linear_speed = 0.0
        self.node.robot_angular_speed = 0.0
        self.node.stationary_linear_limit = 0.01
        self.node.stationary_angular_limit = 0.03
        self.node.navigation_active = True
        self.node.navigation_failure_receipt_sec = None
        self.node.navigation_no_progress_sec = 4.0
        self.node.navigation_progress_distance_m = 0.05
        self.node.recovery_short_goal_distance_m = 0.40
        self.node.recovery_creep_distance_m = 0.40
        self.node.recovery_creep_speed_mps = 0.04
        self.node.finish_recovery_stage = 0
        self.node.finish_recovery_action = "idle"
        self.node.navigation_progress_xy = (0.0, 0.0)
        self.node.navigation_progress_sec = 0.0
        self.node.recovery_goal_in_flight = False
        self.node.recovery_short_goal_active = False
        self.node.recovery_resume_original_pending = False
        self.node.recovery_creep_active = False
        self.node.recovery_creep_handoff_complete = False
        self.node.recovery_creep_start_xy = None
        self.node.recovery_creep_travel_xy = None
        self.node.recovery_creep_requested_sec = None
        self.node.recovery_count = 0
        self.node.recovery_lidar_receipt_sec = 12.4
        self.node.recovery_lidar_timeout_sec = 0.30
        self.node.recovery_lidar_forward_min_m = 1.0
        self.node.recovery_lidar_stop_distance_m = 0.30
        self.node._clear_costmaps_and_retry = mock.Mock(return_value=True)
        self.node._publish_recovery_short_goal = mock.Mock(return_value=True)
        self.node._publish_supervisor_cancel = mock.Mock()
        self.node.command_pub = mock.Mock()
        self.path = ((0.0, 0.0), (2.0, 0.0))
        self.boundary = BoundaryLines(
            left_world=((0.0, 0.455), (2.0, 0.455)),
            right_world=((0.0, -0.455), (2.0, -0.455)),
            yellow_line_width_m=0.030,
            quality=1.0,
            fresh=True,
        )
        self.decision = PassageDecision(
            state=STATE_NORMAL,
            scene=SCENE_NORMAL,
            authority_requested=False,
            command_body=(0.0, 0.0, 0.0),
            reason="normal",
        )

    def update(self, now):
        return self.node._finish_recovery_override(
            now_sec=now,
            decision=self.decision,
            robot_fresh=True,
            boundary=self.boundary,
            boundary_fresh=True,
            path=self.path,
            route_valid=True,
        )

    def test_escalates_clear_short_goal_then_corridor_creep(self):
        first = self.update(4.1)
        self.assertEqual(first.state, STATE_NORMAL)
        self.node._clear_costmaps_and_retry.assert_called_once()
        self.assertEqual(self.node.finish_recovery_stage, 1)

        second = self.update(8.2)
        self.assertEqual(second.state, STATE_NORMAL)
        self.node._publish_recovery_short_goal.assert_called_once()
        self.assertEqual(self.node.finish_recovery_stage, 2)

        handoff = self.update(12.3)
        self.assertEqual(handoff.state, "finish_recovery_creep")
        self.assertTrue(handoff.authority_requested)
        self.assertEqual(handoff.command_body, (0.0, 0.0, 0.0))
        self.node._publish_supervisor_cancel.assert_called_once()
        self.assertTrue(self.node.recovery_creep_active)

        self.node.navigation_active = False
        moving = self.update(12.4)
        self.assertEqual(moving.state, "finish_recovery_creep")
        self.assertTrue(moving.swept_safety.safe)
        self.assertGreater(moving.command_body[0], 0.0)
        self.assertTrue(self.node.recovery_creep_handoff_complete)

        # Once the one-time stop handoff has completed, actual creep motion
        # must not send the next control cycle back to handoff_stop.
        self.node.robot_linear_speed = 0.04
        self.node.recovery_lidar_receipt_sec = 12.5
        continuing = self.update(12.5)
        self.assertEqual(continuing.state, "finish_recovery_creep")
        self.assertGreater(continuing.command_body[0], 0.0)

    def test_corridor_creep_stops_for_lidar_wall(self):
        self.node.recovery_creep_active = True
        self.node.recovery_creep_start_xy = (0.0, 0.0)
        self.node.recovery_creep_travel_xy = (1.0, 0.0)
        self.node.recovery_creep_requested_sec = 1.0
        self.node.navigation_active = False
        self.node.recovery_lidar_receipt_sec = 1.1
        self.node.recovery_lidar_forward_min_m = 0.20
        result = self.update(1.1)
        self.assertEqual(result.state, "finish_recovery_hard_stop")
        self.assertEqual(result.command_body, (0.0, 0.0, 0.0))
        self.assertIn("recovery_lidar_wall_clearance", result.reason)

    def test_progress_resets_escalation_before_short_goal(self):
        self.node.finish_recovery_stage = 1
        self.node.robot_pose = RobotPose(0.06, 0.0, 0.0)
        result = self.update(1.0)
        self.assertEqual(result.state, STATE_NORMAL)
        self.assertEqual(self.node.finish_recovery_stage, 0)
        self.node._publish_recovery_short_goal.assert_not_called()


if __name__ == "__main__":
    unittest.main()
