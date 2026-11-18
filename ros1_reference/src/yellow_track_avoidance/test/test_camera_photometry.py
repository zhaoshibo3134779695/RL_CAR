#!/usr/bin/env python3

import importlib.util
import os
import unittest


PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_PATH = os.path.join(PACKAGE_DIR, "scripts", "v4l2_camera_controls.py")
SPEC = importlib.util.spec_from_file_location("v4l2_camera_controls", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CameraPhotometryTest(unittest.TestCase):
    def test_v4l2_control_output_is_parsed(self):
        output = """\
auto_exposure: 1
exposure_time_absolute: 37
gain: 4
white_balance_temperature: 6120
privacy: disabled
"""
        self.assertEqual(
            MODULE.parse_control_values(output),
            {
                "auto_exposure": 1,
                "exposure_time_absolute": 37,
                "gain": 4,
                "white_balance_temperature": 6120,
            },
        )

    def test_exposure_sequence_is_numeric_unique_and_ascending(self):
        self.assertEqual(
            MODULE.parse_exposures("50,20,40,20,bad"),
            [20, 40, 50],
        )

    def test_complete_metric_scene_passes(self):
        payload = {
            "left_line_detected": True,
            "left_line_quality": 0.91,
            "disc_count": 2,
            "detections": [
                {
                    "confidence": 0.95,
                    "metric_projection": {"base_m": [0.44, -0.09]},
                },
                {
                    "confidence": 0.82,
                    "metric_projection": {"base_m": [1.03, 0.27]},
                },
            ],
        }
        metrics = MODULE.detection_metrics(payload, 2, 0.70, 0.68)
        self.assertTrue(metrics["accepted"])
        self.assertEqual(metrics["metric_disc_count"], 2)

    def test_image_only_candidate_cannot_finish_metric_commissioning(self):
        payload = {
            "left_line_detected": True,
            "left_line_quality": 0.91,
            "disc_count": 1,
            "detections": [
                {
                    "confidence": 0.95,
                    "metric_projection": {"base_m": [0.44, -0.09]},
                }
            ],
            "candidates": [
                {"accepted": True, "confidence": 0.99, "metric_projection": None}
            ],
        }
        metrics = MODULE.detection_metrics(payload, 2, 0.70, 0.68)
        self.assertFalse(metrics["accepted"])


if __name__ == "__main__":
    unittest.main()
