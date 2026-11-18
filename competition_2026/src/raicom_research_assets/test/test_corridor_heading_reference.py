#!/usr/bin/env python3

import math
import unittest

from yellow_track_avoidance.corridor_map import (
    CorridorFit,
    blend_heading_reference,
)


class CorridorHeadingReferenceTest(unittest.TestCase):
    def test_preserves_visual_anchor_and_uses_lidar_heading(self):
        visual = CorridorFit(
            tangent=(1.0, 0.0),
            left_normal=(0.0, 1.0),
            anchor=(1.2, 0.45),
            residual_m=0.004,
        )
        corrected, correction, accepted = blend_heading_reference(
            visual,
            math.radians(2.0),
            reference_weight=0.5,
            maximum_correction_rad=math.radians(3.0),
        )
        self.assertTrue(accepted)
        self.assertAlmostEqual(correction, math.radians(1.0))
        self.assertEqual(corrected.anchor, visual.anchor)
        self.assertEqual(corrected.residual_m, visual.residual_m)
        self.assertAlmostEqual(
            math.atan2(corrected.tangent[1], corrected.tangent[0]),
            math.radians(1.0),
        )
        self.assertAlmostEqual(corrected.left_normal[0], -corrected.tangent[1])
        self.assertAlmostEqual(corrected.left_normal[1], corrected.tangent[0])

    def test_rejects_nonfinite_heading(self):
        visual = CorridorFit((1.0, 0.0), (0.0, 1.0), (0.0, 0.0), 0.0)
        with self.assertRaises(ValueError):
            blend_heading_reference(visual, float("nan"))

    def test_large_disagreement_is_rejected_without_rotating_visual_fit(self):
        visual = CorridorFit((1.0, 0.0), (0.0, 1.0), (0.0, 0.0), 0.0)
        corrected, correction, accepted = blend_heading_reference(
            visual, math.radians(8.0)
        )
        self.assertFalse(accepted)
        self.assertEqual(correction, 0.0)
        self.assertAlmostEqual(corrected.tangent[0], 1.0)
        self.assertAlmostEqual(corrected.tangent[1], 0.0)


if __name__ == "__main__":
    unittest.main()
