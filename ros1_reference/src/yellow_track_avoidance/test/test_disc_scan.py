#!/usr/bin/env python3
"""Regression tests for the bridge's full-circle scan and RViz output."""

import importlib.util
import math
import os
import unittest
from unittest import mock

import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import Marker

from yellow_track_avoidance.disc_memory import StableDiscMemory


PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRIDGE_PATH = os.path.join(PACKAGE_DIR, "scripts", "disc_laserscan_bridge.py")
SPEC = importlib.util.spec_from_file_location("disc_laserscan_bridge", BRIDGE_PATH)
BRIDGE_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BRIDGE_MODULE)


class DiscScanTest(unittest.TestCase):
    def setUp(self):
        self.bridge = BRIDGE_MODULE.DiscLaserScanBridge.__new__(
            BRIDGE_MODULE.DiscLaserScanBridge
        )
        self.bridge.frame_id = "base_link"
        self.bridge.angle_min = -math.pi
        self.bridge.angle_increment = math.radians(1.0)
        self.bridge.bin_count = 361
        self.bridge.range_min = 0.08
        self.bridge.range_max = 1.70
        self.bridge.publish_rate = 10.0
        self.bridge.disk_radius = 0.05
        self.bridge.costmap_disk_radius = 0.05
        self.bridge.marker_persistence = 0.30
        self.bridge.previous_marker_ids = set()
        self.bridge.require_navigation_gate_for_scene_reset = True

    @staticmethod
    def rear_disc():
        return {
            "memory_id": 7,
            "forward": -0.50,
            "left": 0.0,
            "phase": "ahead",
        }

    def test_rear_disc_is_drawn_across_both_scan_seam_bins(self):
        scan, count = self.bridge._scan(
            rospy.Time(0), [self.rear_disc()]
        )
        finite_indices = [
            index
            for index, value in enumerate(scan.ranges)
            if math.isfinite(value)
        ]
        self.assertEqual(count, 1)
        self.assertIn(0, finite_indices)
        self.assertIn(360, finite_indices)
        self.assertAlmostEqual(scan.ranges[0], 0.45, places=9)
        self.assertAlmostEqual(scan.ranges[360], 0.45, places=9)
        self.assertFalse(
            any(math.isfinite(scan.ranges[index]) for index in range(170, 191))
        )

    def test_collision_radius_is_used_only_for_costmap_scan(self):
        self.bridge.costmap_disk_radius = 0.08
        scan, count = self.bridge._scan(rospy.Time(0), [self.rear_disc()])
        self.assertEqual(count, 1)
        self.assertAlmostEqual(scan.ranges[0], 0.42, places=9)
        self.assertEqual(self.bridge.disk_radius, 0.05)

    def test_rear_marker_remains_visible_then_is_explicitly_deleted(self):
        visible = self.bridge._markers(
            rospy.Time(0), [self.rear_disc()]
        )
        additions = [
            marker
            for marker in visible.markers
            if marker.action == Marker.ADD
        ]
        self.assertEqual(len(additions), 2)
        self.assertEqual({marker.id for marker in additions}, {7})
        self.assertTrue(all(marker.pose.position.x < 0.0 for marker in additions))

        cleared = self.bridge._markers(rospy.Time(0), [])
        deletions = [
            marker
            for marker in cleared.markers
            if marker.action == Marker.DELETE
        ]
        self.assertEqual(len(deletions), 2)
        self.assertEqual({marker.id for marker in deletions}, {7})

    def test_timestamped_perception_transform_never_uses_latest(self):
        self.bridge.tf_buffer = mock.Mock()
        self.bridge.tf_buffer.lookup_transform.side_effect = (
            tf2_ros.ExtrapolationException("outside cache")
        )

        with self.assertRaises(tf2_ros.ExtrapolationException):
            self.bridge._lookup_transform(
                "odom",
                "base_link",
                rospy.Time.from_sec(10.0),
                allow_latest=False,
            )

        self.assertEqual(
            self.bridge.tf_buffer.lookup_transform.call_count,
            1,
        )

    def test_scene_reset_requires_locked_gate_zero_commands_and_stationary_odom(self):
        self.bridge.reset_state_timeout = 0.75
        self.bridge.navigation_gate_receipt_sec = 10.0
        self.bridge.output_command_receipt_sec = 10.0
        self.bridge.odom_receipt_sec = 10.0
        self.bridge.navigation_gate_enabled = False
        self.bridge.navigation_gate_allowed = False
        self.bridge.navigation_active = False
        self.bridge.output_command = (0.0, 0.0, 0.0)
        self.bridge.odom_linear_speed = 0.0
        self.bridge.odom_angular_speed = 0.0
        self.bridge.reset_linear_speed_limit = 0.005
        self.bridge.reset_angular_speed_limit = 0.01

        self.assertEqual(self.bridge._scene_reset_block_reason(10.1), "")
        self.bridge.navigation_gate_enabled = True
        self.assertEqual(
            self.bridge._scene_reset_block_reason(10.1),
            "navigation_gate_not_locked",
        )
        self.bridge.navigation_gate_enabled = False
        self.bridge.output_command = (0.006, 0.0, 0.0)
        self.assertEqual(
            self.bridge._scene_reset_block_reason(10.1),
            "output_command_linear_nonzero",
        )
        self.bridge.output_command = (0.0, 0.0, 0.0)
        self.bridge.navigation_active = True
        self.assertEqual(
            self.bridge._scene_reset_block_reason(10.1),
            "navigation_goal_active",
        )

    def test_scene_reset_rejects_stale_safety_state(self):
        self.bridge.reset_state_timeout = 0.75
        self.bridge.navigation_gate_receipt_sec = 8.0
        self.bridge.output_command_receipt_sec = 10.0
        self.bridge.odom_receipt_sec = 10.0
        self.assertEqual(
            self.bridge._scene_reset_block_reason(10.1),
            "navigation_gate_status_stale",
        )

    def test_minimal_profile_reset_does_not_require_absent_gate(self):
        self.bridge.require_navigation_gate_for_scene_reset = False
        self.bridge.reset_state_timeout = 0.75
        self.bridge.navigation_gate_receipt_sec = None
        self.bridge.output_command_receipt_sec = 10.0
        self.bridge.odom_receipt_sec = 10.0
        self.bridge.navigation_gate_enabled = True
        self.bridge.navigation_gate_allowed = True
        self.bridge.navigation_active = False
        self.bridge.output_command = (0.0, 0.0, 0.0)
        self.bridge.odom_linear_speed = 0.0
        self.bridge.odom_angular_speed = 0.0
        self.bridge.reset_linear_speed_limit = 0.005
        self.bridge.reset_angular_speed_limit = 0.01
        self.assertEqual(self.bridge._scene_reset_block_reason(10.1), "")

    def test_mission_memory_marks_only_confirmed_and_never_ages_it(self):
        transform = TransformStamped()
        transform.transform.rotation.w = 1.0
        self.bridge._lookup_transform = mock.Mock(
            return_value=transform
        )
        self.bridge.fixed_frame = "odom"
        self.bridge.lock_confirmed_discs_for_mission = True
        self.bridge.competition_two_disc_mode = False
        self.bridge.navigation_active = False
        self.bridge.latest_global_path_receipt_sec = None
        self.bridge.latest_global_path = ()
        self.bridge.latest_global_path_cumulative = ()
        self.bridge.path_timeout = 5.0
        self.bridge.tombstone_timeout = 20.0
        self.bridge.timeout_removal_count = 0
        self.bridge.tombstone_gc_count = 0
        self.bridge.last_removal_reason = None
        self.bridge.memory = StableDiscMemory(
            1.0,
            0.8,
            0.12,
            0.25,
            freeze_confirmed_position=True,
        )
        confirmed = self.bridge.memory.observe(
            4, (1.0, 0.2), 1.0, True
        )
        unconfirmed = self.bridge.memory.observe(
            9, (1.2, -0.2), 9.5, False
        )

        centers = self.bridge._current_centers(
            rospy.Time.from_sec(10.0)
        )

        self.assertEqual(
            [center["memory_id"] for center in centers],
            [confirmed.memory_id],
        )
        self.assertIn(
            unconfirmed.memory_id,
            [entry.memory_id for entry in self.bridge.memory.entries()],
        )
        self.assertGreater(centers[0]["last_seen_age_sec"], 8.0)


if __name__ == "__main__":
    unittest.main()
