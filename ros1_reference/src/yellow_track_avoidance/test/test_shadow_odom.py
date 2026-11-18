#!/usr/bin/env python3

import math
import unittest

from yellow_track_avoidance.shadow_odom import (
    PlanarState,
    integrate_body_twist,
    normalized_angle,
)


class ShadowOdomTest(unittest.TestCase):
    def test_forward_at_zero_yaw(self):
        result = integrate_body_twist(PlanarState(), 0.1, 0.0, 0.0, 2.0)
        self.assertAlmostEqual(result.x_m, 0.2)
        self.assertAlmostEqual(result.y_m, 0.0)

    def test_body_strafe_rotates_into_world(self):
        state = PlanarState(yaw_rad=0.5 * math.pi)
        result = integrate_body_twist(state, 0.0, 0.1, 0.0, 1.0)
        self.assertAlmostEqual(result.x_m, -0.1)
        self.assertAlmostEqual(result.y_m, 0.0, places=7)

    def test_yaw_is_wrapped(self):
        result = integrate_body_twist(
            PlanarState(yaw_rad=math.pi - 0.1), 0.0, 0.0, 0.2, 1.0
        )
        self.assertAlmostEqual(result.yaw_rad, -math.pi + 0.1)
        self.assertGreaterEqual(result.yaw_rad, -math.pi)
        self.assertLess(result.yaw_rad, math.pi)

    def test_negative_dt_does_not_rewind(self):
        state = PlanarState(1.0, 2.0, 0.3)
        self.assertEqual(integrate_body_twist(state, 1.0, 1.0, 1.0, -1.0), state)


if __name__ == "__main__":
    unittest.main()
