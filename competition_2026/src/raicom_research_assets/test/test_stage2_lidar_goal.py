#!/usr/bin/env python3

import unittest

import numpy as np

from yellow_track_avoidance.stage2_lidar_goal import (
    build_stage2_geometry,
    voxel_downsample,
)


class Stage2LidarGoalTest(unittest.TestCase):
    def _scene(self, angle_deg=0.0):
        generator = np.random.default_rng(812)
        angle = np.deg2rad(angle_deg)
        tangent = np.asarray((np.cos(angle), np.sin(angle)))
        normal = np.asarray((-np.sin(angle), np.cos(angle)))
        right_origin = -0.33 * normal
        s = np.linspace(0.30, 2.28, 300)
        right = right_origin + s[:, None] * tangent
        left = right_origin + s[:, None] * tangent + 0.485 * normal
        lateral = np.linspace(0.0, 0.485, 90)
        terminal = right_origin + 2.28 * tangent + lateral[:, None] * normal
        noise = generator.normal(0.0, 0.002, (len(right) + len(left) + len(terminal), 2))
        walls = np.vstack((right, left, terminal)) + noise
        clutter = np.column_stack(
            (generator.uniform(0.3, 2.8, 100), generator.uniform(-1.5, 1.6, 100))
        )
        return voxel_downsample(np.vstack((walls, clutter)), 0.01), tangent, normal

    def test_terminal_wall_defines_ten_centimeter_front_clearance(self):
        points, tangent, normal = self._scene()
        result = build_stage2_geometry(points)
        self.assertAlmostEqual(result.measured_width_m, 0.485, delta=0.025)
        self.assertAlmostEqual(result.terminal_center[0], 2.28, delta=0.025)
        self.assertAlmostEqual(result.goal_base_center[0], 2.01, delta=0.025)
        clearance = np.dot(
            result.terminal_center - result.goal_front_center, tangent
        )
        self.assertAlmostEqual(clearance, 0.10, delta=0.012)
        self.assertGreater(result.terminal_support_voxels, 20)

    def test_rotated_corridor_preserves_heading_and_goal(self):
        points, tangent, _ = self._scene(angle_deg=7.0)
        result = build_stage2_geometry(points)
        self.assertAlmostEqual(result.heading_rad, np.deg2rad(7.0), delta=0.02)
        clearance = np.dot(
            result.terminal_center - result.goal_front_center, tangent
        )
        self.assertAlmostEqual(clearance, 0.10, delta=0.015)


if __name__ == "__main__":
    unittest.main()
