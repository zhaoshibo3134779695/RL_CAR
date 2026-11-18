#!/usr/bin/env python3

import importlib.util
from pathlib import Path
import unittest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "offline_passage_scenarios.py"
)
SPEC = importlib.util.spec_from_file_location(
    "offline_passage_scenarios_under_test", str(SCRIPT_PATH)
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class OfflinePassageScenarioTests(unittest.TestCase):
    def test_locked_four_scenario_acceptance(self):
        report = MODULE.build_report()
        self.assertTrue(report["locked_offline_only"])
        self.assertFalse(report["official_image_metric_scale_available"])
        self.assertTrue(report["passed"])
        self.assertEqual(len(report["scenarios"]), 4)
        self.assertEqual(
            [item["selection"] for item in report["scenarios"]],
            ["BETWEEN", "S_CURVE", "RIGHT", "S_CURVE"],
        )
        self.assertTrue(
            all(
                item["passage_authority_after_confirmation"]
                for item in report["scenarios"]
            )
        )

    def test_formal_profile_has_no_gate_state(self):
        report = MODULE.build_report()
        profile = report["formal_profile"]
        self.assertFalse(profile["competition_two_disc_mode"])
        self.assertFalse(profile["enable_gate_special_mode"])
        self.assertTrue(profile["claim_all_road_discs"])
        self.assertFalse(profile["force_between_disc_pair"])
        self.assertFalse(profile["competition_finish_priority"])
        self.assertFalse(profile["enable_costmap_writes"])
        self.assertEqual(
            report["formal_gate_states_now_unreachable"],
            ["gate_armed", "gate_align", "gate_cross", "gate_exit"],
        )


if __name__ == "__main__":
    unittest.main()
