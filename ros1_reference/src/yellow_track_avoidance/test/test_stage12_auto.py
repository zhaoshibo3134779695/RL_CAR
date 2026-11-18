import unittest

from yellow_track_avoidance.stage12_auto import (
    build_stage_command,
    runner_outcome,
)


class Stage12AutoTests(unittest.TestCase):
    def test_runner_outcome_only_accepts_explicit_success(self):
        self.assertEqual(runner_outcome({"state": "SUCCEEDED"}), "success")
        self.assertEqual(runner_outcome({"state": "FAILED"}), "failure")
        self.assertIsNone(runner_outcome({"state": "ACTIVE"}))
        self.assertIsNone(runner_outcome(None))

    def test_stage_command_is_live_and_explicit(self):
        command = build_stage_command(
            "competition_stage1_auto.launch",
            90.0,
            120.0,
            start_rviz=True,
            start_perception_view=True,
        )
        self.assertIn("enable_motion:=true", command)
        self.assertIn("start_rviz:=true", command)
        self.assertIn("start_perception_view:=true", command)
        self.assertIn("startup_timeout_sec:=90.0", command)
        self.assertIn("navigation_timeout_sec:=120.0", command)

    def test_stage2_command_uses_formal_launch_defaults(self):
        command = build_stage_command(
            "competition_stage2_auto.launch",
            45.0,
            90.0,
        )
        self.assertFalse(
            any(item.startswith("start_perception_view:=") for item in command)
        )
        self.assertFalse(any(item.startswith("front_clearance_m:=") for item in command))
        self.assertFalse(
            any(item.startswith("stage2_laser_lateral_offset_m:=") for item in command)
        )


if __name__ == "__main__":
    unittest.main()
