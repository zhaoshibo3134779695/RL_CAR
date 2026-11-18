#!/usr/bin/env python3

import unittest

import numpy as np

from yellow_track_avoidance.field_localization import (
    OccupancyDistanceField,
    _pose_score,
)


class FieldLocalizationTests(unittest.TestCase):
    def test_candidate_positions_are_limited_to_known_segment(self):
        field = OccupancyDistanceField(
            np.full((120, 120), 255, dtype=np.uint8),
            resolution_m=0.10,
        )
        positions = field.free_positions(
            0.20,
            0.0,
            bounds_m=(2.0, 3.0, 4.0, 5.0),
        )
        self.assertGreater(len(positions), 0)
        self.assertGreaterEqual(float(np.min(positions[:, 0])), 2.0)
        self.assertLessEqual(float(np.max(positions[:, 0])), 4.0 + 1e-9)
        self.assertGreaterEqual(float(np.min(positions[:, 1])), 3.0)
        self.assertLessEqual(float(np.max(positions[:, 1])), 5.0 + 1e-9)

    def test_yellow_score_models_nonzero_wall_offset(self):
        image = np.full((120, 120), 255, dtype=np.uint8)
        image[69:71, :] = 0
        field = OccupancyDistanceField(image, resolution_m=0.05)
        lidar = np.column_stack(
            [np.linspace(0.5, 4.5, 40), np.full(40, 2.45)]
        )
        yellow = np.column_stack(
            [np.linspace(1.0, 3.0, 8), np.full(8, 2.20)]
        )
        rays = lidar * 0.55

        offset_score = _pose_score(
            field,
            lidar,
            yellow,
            rays,
            0.0,
            0.0,
            0.0,
            lidar_weight=0.2,
            wall_weight=4.0,
            ray_weight=0.0,
            wall_target_distance_m=0.25,
        )
        wall_score = _pose_score(
            field,
            lidar,
            yellow,
            rays,
            0.0,
            0.0,
            0.0,
            lidar_weight=0.2,
            wall_weight=4.0,
            ray_weight=0.0,
            wall_target_distance_m=0.0,
        )
        self.assertLess(offset_score[2], 0.04)
        self.assertGreater(wall_score[2], offset_score[2] + 0.10)


if __name__ == "__main__":
    unittest.main()
