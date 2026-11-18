#!/usr/bin/env python3

import unittest

from yellow_track_avoidance.stage1_auto import (
    command_topology_reason,
    goal_is_ready,
)


class Stage1AutoTest(unittest.TestCase):
    def test_goal_requires_world_freeze_and_nonempty_plan(self):
        payload = {
            "state": "goal_ready",
            "world_ready": True,
            "world_stable": True,
            "goal_frozen": True,
            "goal_sent": False,
            "preview_plan_pose_count": 800,
            "last_error": None,
        }
        self.assertTrue(goal_is_ready(payload))
        payload["preview_plan_pose_count"] = 0
        self.assertFalse(goal_is_ready(payload))
        payload["preview_plan_pose_count"] = 800
        payload["world_stable"] = False
        self.assertFalse(goal_is_ready(payload))

    def test_motion_requires_move_base_and_chassis(self):
        self.assertEqual(
            command_topology_reason(
                ["/move_base"],
                ["/yellow_track_disc_costmap_bridge", "/cmd_vel_listener"],
                True,
            ),
            "ready",
        )
        self.assertEqual(
            command_topology_reason(
                ["/move_base"], ["/yellow_track_disc_costmap_bridge"], True
            ),
            "chassis_subscriber_missing",
        )
        self.assertEqual(
            command_topology_reason(
                ["/move_base", "/other"], ["/cmd_vel_listener"], True
            ),
            "cmd_vel_publishers_not_move_base_only",
        )

    def test_dry_run_rejects_live_chassis(self):
        self.assertEqual(
            command_topology_reason(["/move_base"], [], False), "ready"
        )
        self.assertEqual(
            command_topology_reason(
                ["/move_base"], ["/cmd_vel_listener"], False
            ),
            "chassis_present_in_dry_run",
        )


if __name__ == "__main__":
    unittest.main()
