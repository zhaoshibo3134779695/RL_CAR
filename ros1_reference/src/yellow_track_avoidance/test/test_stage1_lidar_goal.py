#!/usr/bin/env python3

import unittest

import numpy as np

from yellow_track_avoidance.stage1_lidar_goal import (
    build_stage1_geometry,
    voxel_downsample,
)


def _scene(angle_rad=0.0, inner_face_slope=0.0, include_entrance=False):
    generator = np.random.default_rng(12)
    tangent = np.asarray((np.cos(angle_rad), np.sin(angle_rad)))
    normal = np.asarray((-np.sin(angle_rad), np.cos(angle_rad)))
    right_anchor = -0.46 * normal
    right_s = np.linspace(0.3, 5.2, 420)
    right = (
        right_anchor
        + right_s[:, None] * tangent
        + generator.normal(0.0, 0.003, len(right_s))[:, None] * normal
    )

    # The inner terminal face marks the end of the 1.40 m straight at 3.95 m.
    # The outer face lies one passage width farther forward.  Both faces may
    # appear slightly oblique, but neither is allowed to rotate the heading.
    inner_d = np.linspace(0.485, 1.45, 90)
    inner_s = 3.95 + inner_face_slope * (inner_d - 0.485)
    inner = (
        right_anchor
        + inner_s[:, None] * tangent
        + inner_d[:, None] * normal
    )
    outer_d = np.linspace(0.06, 0.55, 55)
    outer_s = 4.435 - 0.025 * outer_d
    outer = (
        right_anchor
        + outer_s[:, None] * tangent
        + outer_d[:, None] * normal
    )
    clutter = np.column_stack(
        (generator.uniform(0.4, 5.0, 70), generator.uniform(-1.5, 1.8, 70))
    )
    points = [right, inner, outer, clutter]
    if include_entrance:
        entrance = right_anchor + 2.55 * tangent + 0.485 * normal
        points.append(entrance[None, :])
    return voxel_downsample(np.vstack(points), 0.01)


class Stage1LidarGoalTest(unittest.TestCase):
    def test_far_terminal_pair_defines_goal_without_entrance_return(self):
        result = build_stage1_geometry(_scene(), straight_length_m=1.40)
        self.assertAlmostEqual(result.measured_width_m, 0.485, delta=0.005)
        self.assertAlmostEqual(result.entrance_center[0], 2.55, delta=0.04)
        self.assertAlmostEqual(result.goal_base_center[0], 2.28, delta=0.05)
        self.assertAlmostEqual(result.goal_front_center[0], 2.45, delta=0.05)
        self.assertAlmostEqual(result.heading_rad, 0.0, delta=0.02)
        self.assertAlmostEqual(result.terminal_pair_separation_m, 0.485, delta=0.04)
        self.assertEqual(result.entrance_support_voxels, 0)

    def test_goal_heading_follows_rotated_right_wall(self):
        angle = np.deg2rad(8.0)
        result = build_stage1_geometry(
            _scene(angle_rad=angle, inner_face_slope=0.18),
            straight_length_m=1.40,
        )
        self.assertAlmostEqual(result.heading_rad, angle, delta=0.02)
        self.assertAlmostEqual(result.terminal_pair_separation_m, 0.485, delta=0.05)

    def test_optional_entrance_return_does_not_change_terminal_solution(self):
        absent = build_stage1_geometry(_scene(include_entrance=False))
        present = build_stage1_geometry(_scene(include_entrance=True))
        np.testing.assert_allclose(
            absent.goal_base_center, present.goal_base_center, atol=0.012
        )
        self.assertGreaterEqual(present.entrance_support_voxels, 1)


if __name__ == "__main__":
    unittest.main()
