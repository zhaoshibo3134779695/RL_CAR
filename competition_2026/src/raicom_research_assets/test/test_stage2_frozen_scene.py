import math
import unittest

import numpy as np

from yellow_track_avoidance.stage2_frozen_scene import (
    geometry_payload,
    goal_from_payload,
    rasterize_scene,
    wall_segments,
)
from yellow_track_avoidance.stage2_lidar_goal import build_stage2_geometry


def synthetic_corridor():
    x_values = np.linspace(0.2, 2.3, 220)
    right = np.column_stack((x_values, np.full_like(x_values, -0.25)))
    left = np.column_stack((x_values, np.full_like(x_values, 0.235)))
    y_values = np.linspace(-0.25, 0.235, 70)
    terminal = np.column_stack((np.full_like(y_values, 2.3), y_values))
    return np.vstack((right, left, terminal))


class FrozenStage2SceneTest(unittest.TestCase):
    def setUp(self):
        self.geometry = build_stage2_geometry(synthetic_corridor())
        self.payload = geometry_payload(self.geometry)

    def test_payload_contains_exactly_three_wall_segments(self):
        segments = wall_segments(self.payload)
        self.assertEqual(len(segments), 3)
        self.assertAlmostEqual(
            self.payload["measured_width_m"], 0.485, delta=0.002
        )

    def test_goal_keeps_front_clearance(self):
        goal, heading = goal_from_payload(
            self.payload, vehicle_front_offset_m=0.17, front_clearance_m=0.10
        )
        tangent = np.asarray((math.cos(heading), math.sin(heading)))
        front = goal + 0.17 * tangent
        terminal = np.asarray(self.payload["terminal_center"])
        self.assertAlmostEqual(float((terminal - front) @ tangent), 0.10, places=6)

    def test_rasterized_map_has_three_thin_obstacle_bands(self):
        grid = rasterize_scene(
            self.payload,
            resolution_m=0.01,
            width_m=4.0,
            height_m=2.0,
            origin_x_m=-0.5,
            origin_y_m=-1.0,
            wall_thickness_m=0.01,
        )
        self.assertEqual(grid.shape, (200, 400))
        self.assertGreater(int(np.count_nonzero(grid == 100)), 500)
        self.assertLess(int(np.count_nonzero(grid == 100)), 2500)
        self.assertFalse(np.any((grid != 0) & (grid != 100)))


if __name__ == "__main__":
    unittest.main()
