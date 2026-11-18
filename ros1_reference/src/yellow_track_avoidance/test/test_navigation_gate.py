#!/usr/bin/env python3

import importlib.util
import json
import math
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest import mock


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "navigation_safety_gate.py"
)
SPEC = importlib.util.spec_from_file_location(
    "navigation_safety_gate_under_test", str(SCRIPT_PATH)
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
NavigationSafetyGate = MODULE.NavigationSafetyGate
TerminalConfig = MODULE.TerminalConfig
TerminalVisualController = MODULE.TerminalVisualController
terminal_takeover_allowed = MODULE.terminal_takeover_allowed
zone_navigation_safety = MODULE.zone_navigation_safety


class NavigationSafetyGateTests(unittest.TestCase):
    def test_active_gate_keeps_authority_before_pair_is_consumed(self):
        allowed = terminal_takeover_allowed(
            auto_localization_active=False,
            special_authority_latched=True,
            special_status={
                "state": "gate_cross",
                "competition_pair_consumed": False,
            },
        )
        self.assertFalse(allowed)

    def test_terminal_can_replace_stale_finish_recovery_authority(self):
        allowed = terminal_takeover_allowed(
            auto_localization_active=False,
            special_authority_latched=True,
            special_status={
                "state": "finish_recovery_creep",
                "competition_pair_consumed": True,
            },
        )
        self.assertTrue(allowed)

    def test_terminal_boundary_callback_uses_live_base_link_line(self):
        gate = object.__new__(NavigationSafetyGate)
        gate.lock = threading.RLock()
        gate.terminal_line_minimum_quality = 0.50
        gate.terminal_robot = ("odom", 0.0, 0.0, 0.20, 0.0, 0.0)
        gate.terminal_line = None
        gate.terminal_controller = TerminalVisualController(TerminalConfig())
        slope = math.tan(math.radians(3.0))
        message = SimpleNamespace(
            data=json.dumps(
                {
                    "stamp": 10.0,
                    "frame_id": "base_link",
                    "left_detected": True,
                    "quality": 0.90,
                    "left_base_m": [
                        [0.4, 0.67 + 0.4 * slope],
                        [1.0, 0.67 + 1.0 * slope],
                    ],
                }
            )
        )
        gate._terminal_boundary_callback(message)
        self.assertIsNotNone(gate.terminal_line)
        self.assertAlmostEqual(gate.terminal_line.odom_yaw_rad, 0.20)
        self.assertAlmostEqual(
            math.degrees(gate.terminal_line.yaw_error_rad), 3.0
        )

    def test_terminal_control_uses_directional_overrun_guard(self):
        gate = object.__new__(NavigationSafetyGate)
        gate.terminal_goal_stop_enabled = True
        gate.enabled = True
        gate.terminal_goal_armed = True
        gate.terminal_goal = ("odom", 2.0, 0.0, 0.0)
        gate.terminal_robot = ("odom", 2.026, 0.0, 0.0, 0.0, 0.0)
        gate.terminal_robot_receipt_sec = 10.0
        gate.terminal_goal_odom_timeout = 0.30
        gate.terminal_goal_metrics = {}
        gate.terminal_line = None
        gate.terminal_controller = TerminalVisualController(TerminalConfig())
        gate.terminal_decision = None
        decision = gate._terminal_control_locked(
            10.1, takeover_allowed=False
        )
        self.assertEqual(decision.latch_reason, "terminal_overrun_guard")
        self.assertAlmostEqual(
            gate.terminal_goal_metrics["remaining_m"], -0.026
        )

    @staticmethod
    def minimal_gate():
        gate = object.__new__(NavigationSafetyGate)
        gate._output_conflicts = lambda force=False: []
        gate.enforce_perception_safety = False
        gate.enforce_boundary_in_activation_zone = False
        gate.disk_receipt_sec = None
        gate.disk_timeout = 0.50
        gate.minimum_disk_clearance = None
        gate.emergency_clearance = 0.005
        gate.enforce_disc_emergency_clearance = True
        return gate

    def test_competition_mode_still_requires_fresh_disc_status(self):
        gate = self.minimal_gate()
        ready, reason = gate._ready(10.0)
        self.assertFalse(ready)
        self.assertEqual(reason, "disc_status_timeout")

        gate.disk_receipt_sec = 9.8
        ready, reason = gate._ready(10.0)
        self.assertTrue(ready)
        self.assertEqual(reason, "ready_competition_mode")

    def test_competition_mode_still_enforces_emergency_clearance(self):
        gate = self.minimal_gate()
        gate.disk_receipt_sec = 9.8
        gate.minimum_disk_clearance = 0.004
        ready, reason = gate._ready(10.0)
        self.assertFalse(ready)
        self.assertEqual(reason, "disc_inside_emergency_clearance")

    def test_finish_priority_can_treat_disc_contact_as_penalty(self):
        gate = self.minimal_gate()
        gate.disk_receipt_sec = 9.8
        gate.minimum_disk_clearance = -0.050
        gate.enforce_disc_emergency_clearance = False
        ready, reason = gate._ready(10.0)
        self.assertTrue(ready)
        self.assertEqual(reason, "ready_competition_mode")

    def test_finish_priority_does_not_lock_on_disc_stream_timeout(self):
        gate = self.minimal_gate()
        gate.enforce_disc_emergency_clearance = False
        ready, reason = gate._ready(10.0)
        self.assertTrue(ready)
        self.assertEqual(reason, "ready_competition_mode")

    def test_disc_behind_robot_is_included_in_clearance(self):
        gate = object.__new__(NavigationSafetyGate)
        gate.lock = threading.RLock()
        gate.vehicle_half_length = 0.165
        gate.vehicle_half_width = 0.150
        gate.disk_radius = 0.050
        gate.forward_offset = 0.39
        message = SimpleNamespace(
            data=json.dumps(
                {
                    "raw": [
                        {
                            "track_id": 4,
                            "center_base_m": [-0.10, 0.0],
                        }
                    ]
                }
            )
        )
        fake_time = SimpleNamespace(to_sec=lambda: 10.0)
        with mock.patch.object(
            MODULE.rospy.Time, "now", return_value=fake_time
        ):
            gate._disk_callback(message)
        self.assertAlmostEqual(gate.minimum_disk_distance, 0.10)
        self.assertLess(gate.minimum_disk_clearance, 0.0)
        self.assertEqual(gate.disk_receipt_sec, 10.0)

    def test_disc_far_behind_uses_rear_rectangle_edge(self):
        gate = object.__new__(NavigationSafetyGate)
        gate.lock = threading.RLock()
        gate.vehicle_half_length = 0.165
        gate.vehicle_half_width = 0.150
        gate.disk_radius = 0.050
        gate.forward_offset = 0.39
        message = SimpleNamespace(
            data=json.dumps(
                {
                    "raw": [
                        {
                            "track_id": 5,
                            "center_base_m": [-1.0, 0.0],
                        }
                    ]
                }
            )
        )
        fake_time = SimpleNamespace(to_sec=lambda: 10.0)
        with mock.patch.object(
            MODULE.rospy.Time, "now", return_value=fake_time
        ):
            gate._disk_callback(message)
        self.assertAlmostEqual(gate.minimum_disk_clearance, 0.785)

    def test_passage_authority_is_latched_from_status(self):
        gate = object.__new__(NavigationSafetyGate)
        gate.lock = threading.RLock()
        gate.special_status = None
        gate.special_status_receipt_sec = None
        gate.special_authority_latched = False
        fake_time = SimpleNamespace(to_sec=lambda: 10.0)
        message = SimpleNamespace(
            data=json.dumps(
                {
                    "authority_requested": True,
                    "state": "single_disc_avoid",
                    "safe_to_move": True,
                }
            )
        )
        with mock.patch.object(
            MODULE.rospy.Time, "now", return_value=fake_time
        ):
            gate._special_status_callback(message)
        self.assertTrue(gate.special_authority_latched)
        self.assertEqual(gate.special_status_receipt_sec, 10.0)

    def test_passage_explicit_release_clears_authority(self):
        gate = object.__new__(NavigationSafetyGate)
        gate.lock = threading.RLock()
        gate.special_status = None
        gate.special_status_receipt_sec = None
        gate.special_authority_latched = True
        fake_time = SimpleNamespace(to_sec=lambda: 11.0)
        message = SimpleNamespace(
            data=json.dumps(
                {
                    "authority_requested": False,
                    "state": "normal_nav",
                    "safe_to_move": True,
                }
            )
        )
        with mock.patch.object(
            MODULE.rospy.Time, "now", return_value=fake_time
        ):
            gate._special_status_callback(message)
        self.assertFalse(gate.special_authority_latched)

    def test_zone_guard_allows_move_base_outside_configured_zone(self):
        safe, reason = zone_navigation_safety(
            {
                "activation_zone": {
                    "enabled": True,
                    "pose_available": True,
                    "active": False,
                },
                "boundary_fresh": False,
            },
            status_fresh=True,
            command_fresh=True,
            special_authority_latched=False,
        )
        self.assertTrue(safe)
        self.assertEqual(reason, "outside_activation_zone")

    def test_zone_guard_fails_closed_when_boundary_is_stale(self):
        safe, reason = zone_navigation_safety(
            {
                "activation_zone": {
                    "enabled": True,
                    "pose_available": True,
                    "active": True,
                },
                "boundary_fresh": False,
                "normal_command_safety": {"safe": True},
            },
            status_fresh=True,
            command_fresh=True,
            special_authority_latched=False,
        )
        self.assertFalse(safe)
        self.assertEqual(reason, "zone_boundary_stale")

    def test_zone_guard_rejects_unsafe_swept_footprint(self):
        safe, reason = zone_navigation_safety(
            {
                "activation_zone": {
                    "enabled": True,
                    "pose_available": True,
                    "active": True,
                },
                "boundary_fresh": True,
                "normal_command_safety": {
                    "safe": False,
                    "reason": "swept_yellow_clearance",
                },
            },
            status_fresh=True,
            command_fresh=True,
            special_authority_latched=False,
        )
        self.assertFalse(safe)
        self.assertEqual(
            reason, "zone_swept_guard:swept_yellow_clearance"
        )

    def test_zone_guard_accepts_fresh_safe_candidate(self):
        safe, reason = zone_navigation_safety(
            {
                "activation_zone": {
                    "enabled": True,
                    "pose_available": True,
                    "active": True,
                },
                "boundary_fresh": True,
                "normal_command_safety": {
                    "safe": True,
                    "reason": "swept_safe",
                },
            },
            status_fresh=True,
            command_fresh=True,
            special_authority_latched=False,
        )
        self.assertTrue(safe)
        self.assertEqual(reason, "zone_swept_safe")

    def test_auto_localization_compatibility_service_is_stationary(self):
        gate = object.__new__(NavigationSafetyGate)
        gate.lock = threading.RLock()
        gate.enabled = True
        gate.latest_command = object()
        gate.command_receipt_sec = 9.0
        gate.auto_localization_active = False
        gate.auto_localization_state = "idle"
        gate.auto_localization_message = ""
        gate.auto_localization_start_sec = None
        gate.auto_localization_elapsed_sec = 0.0
        gate.field_localization_service_name = "/field/localize"
        gate.output_pub = mock.Mock()
        gate.goal_cancel_pub = mock.Mock()
        fake_time = SimpleNamespace(to_sec=lambda: 10.0)
        result = SimpleNamespace(success=True, message="stationary match ok")
        proxy = mock.Mock(return_value=result)
        with mock.patch.object(
            MODULE.rospy.Time, "now", return_value=fake_time
        ), mock.patch.object(
            MODULE.rospy, "wait_for_service"
        ), mock.patch.object(
            MODULE.rospy, "ServiceProxy", return_value=proxy
        ):
            response = gate._start_auto_localization(None)

        self.assertTrue(response.success)
        self.assertFalse(gate.enabled)
        self.assertFalse(gate.auto_localization_active)
        self.assertEqual(gate.auto_localization_state, "localized")
        self.assertIsNone(gate.latest_command)
        self.assertEqual(proxy.call_count, 1)
        for call in gate.output_pub.publish.call_args_list:
            command = call.args[0]
            self.assertEqual(command.linear.x, 0.0)
            self.assertEqual(command.linear.y, 0.0)
            self.assertEqual(command.angular.z, 0.0)


if __name__ == "__main__":
    unittest.main()
