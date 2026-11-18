#!/usr/bin/env python3

import math
import unittest

import numpy as np

from yellow_track_avoidance.corridor_map import (
    CorridorFit,
    angular_difference,
    build_corridor_segments,
    build_corridor_stop_goal,
    extend_line_segment,
    fit_left_boundary,
    sample_line_strip,
)


class CorridorMapTest(unittest.TestCase):
    def test_measured_left_line_builds_metric_right_boundary(self):
        points = [(0.75, 0.423), (1.00, 0.438), (1.25, 0.453)]
        fit = fit_left_boundary(points, (1.0, 0.0), (0.0, 1.0))
        segments = build_corridor_segments(
            fit,
            robot_xy=(0.0, 0.0),
            center_separation_m=0.91,
            vehicle_front_offset_m=0.17,
            corridor_length_m=1.85,
        )
        for left, right in zip(segments.left, segments.right):
            separation = math.hypot(left[0] - right[0], left[1] - right[1])
            self.assertAlmostEqual(separation, 0.91, places=9)
        self.assertAlmostEqual(
            math.dist(segments.left[0], segments.left[1]), 1.85, places=9
        )
        start_progress = (
            segments.left[0][0] * fit.tangent[0]
            + segments.left[0][1] * fit.tangent[1]
        )
        end_progress = (
            segments.left[1][0] * fit.tangent[0]
            + segments.left[1][1] * fit.tangent[1]
        )
        self.assertAlmostEqual(start_progress, 0.17, places=9)
        self.assertAlmostEqual(end_progress, 2.02, places=9)
        self.assertLess(fit.residual_m, 1e-9)

    def test_fit_orientation_follows_vehicle_forward_and_left(self):
        yaw = math.radians(35.0)
        forward = math.cos(yaw), math.sin(yaw)
        left = -math.sin(yaw), math.cos(yaw)
        points = [
            (
                2.0 + forward[0] * distance + left[0] * 0.42,
                -1.0 + forward[1] * distance + left[1] * 0.42,
            )
            for distance in (0.7, 0.9, 1.1, 1.3)
        ]
        fit = fit_left_boundary(list(reversed(points)), forward, left)
        self.assertLess(angular_difference(fit.tangent, forward), 1e-9)
        self.assertGreater(
            fit.left_normal[0] * left[0] + fit.left_normal[1] * left[1], 0.999
        )

    def test_line_strip_contains_full_physical_width(self):
        points = sample_line_strip(((0.0, 0.4), (1.0, 0.4)), 0.03, 0.01)
        self.assertGreaterEqual(len(points), 404)
        self.assertAlmostEqual(float(np.min(points[:, 1])), 0.385, places=9)
        self.assertAlmostEqual(float(np.max(points[:, 1])), 0.415, places=9)
        self.assertTrue(np.all(points[:, 2] == 0.0))

    def test_costmap_segment_extensions_preserve_direction(self):
        segment = extend_line_segment(
            ((0.17, 0.4), (2.02, 0.4)), before_m=0.55, after_m=0.75
        )
        self.assertEqual(segment, ((-0.38, 0.4), (2.77, 0.4)))
        self.assertAlmostEqual(math.dist(*segment), 3.15, places=9)

    def test_costmap_segment_extensions_reject_negative_values(self):
        with self.assertRaises(ValueError):
            extend_line_segment(((0.0, 0.0), (1.0, 0.0)), before_m=-0.01)

    def test_stop_goal_tracks_far_right_endpoint_and_front_edge(self):
        segments = build_corridor_segments(
            CorridorFit(
                tangent=(1.0, 0.0),
                left_normal=(0.0, 1.0),
                anchor=(0.0, 0.45),
                residual_m=0.0,
            ),
            robot_xy=(0.0, 0.0),
            center_separation_m=0.91,
            vehicle_front_offset_m=0.17,
            corridor_length_m=1.85,
        )
        goal = build_corridor_stop_goal(
            segments,
            left_offset_m=0.24,
            forward_offset_m=0.54,
            vehicle_front_offset_m=0.17,
        )
        for actual, expected in zip(goal.reference, (2.02, -0.46)):
            self.assertAlmostEqual(actual, expected, places=9)
        for actual, expected in zip(goal.front_center, (2.56, -0.22)):
            self.assertAlmostEqual(actual, expected, places=9)
        for actual, expected in zip(goal.base_center, (2.39, -0.22)):
            self.assertAlmostEqual(actual, expected, places=9)
        self.assertAlmostEqual(goal.yaw, 0.0, places=9)


if __name__ == "__main__":
    unittest.main()
