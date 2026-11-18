import unittest

from yellow_track_avoidance.stage2_auto import (
    command_topology_reason,
    goal_is_ready,
    scene_is_ready,
)


class Stage2AutoTests(unittest.TestCase):
    def test_scene_requires_frozen_three_primitive_map(self):
        payload = {
            "mode": "FROZEN",
            "map_frozen": True,
            "scan_updates_costmap": False,
            "compact_obstacle_count": 3,
            "last_error": None,
        }
        self.assertTrue(scene_is_ready(payload))
        payload["compact_obstacle_count"] = 2
        self.assertFalse(scene_is_ready(payload))
        payload["compact_obstacle_count"] = 3
        payload["scan_updates_costmap"] = True
        self.assertFalse(scene_is_ready(payload))

    def test_goal_requires_frozen_nonempty_preview(self):
        payload = {
            "state": "goal_ready",
            "goal_frozen": True,
            "goal_sent": False,
            "preview_plan_pose_count": 100,
            "last_error": None,
        }
        self.assertTrue(goal_is_ready(payload))
        payload["preview_plan_pose_count"] = 0
        self.assertFalse(goal_is_ready(payload))
        payload["preview_plan_pose_count"] = 100
        payload["goal_sent"] = True
        self.assertFalse(goal_is_ready(payload))

    def test_command_topology(self):
        self.assertEqual(
            command_topology_reason(
                ["/move_base"], ["/cmd_vel_listener"], True
            ),
            "ready",
        )
        self.assertEqual(
            command_topology_reason(["/move_base"], [], True),
            "chassis_subscriber_missing",
        )
        self.assertEqual(
            command_topology_reason(["/move_base"], [], False),
            "ready",
        )
        self.assertEqual(
            command_topology_reason(
                ["/move_base", "/another_controller"],
                ["/cmd_vel_listener"],
                True,
            ),
            "cmd_vel_publishers_not_move_base_only",
        )


if __name__ == "__main__":
    unittest.main()
