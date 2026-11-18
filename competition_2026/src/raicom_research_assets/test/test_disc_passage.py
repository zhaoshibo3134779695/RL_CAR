#!/usr/bin/env python3
"""ROS-independent tests for conservative disc lifecycle handling."""

import math
import unittest

from yellow_track_avoidance.disc_memory import StableDiscMemory
from yellow_track_avoidance.disc_passage import (
    PHASE_ABREAST,
    PHASE_AHEAD,
    PHASE_PASSED_CANDIDATE,
    PHASE_TOMBSTONE,
    PassageConfig,
    PassageState,
    circle_to_rectangle_clearance,
    cumulative_lengths,
    tombstone_should_reactivate,
    update_passage,
)


class DiscPassageTest(unittest.TestCase):
    def setUp(self):
        self.path = tuple((index * 0.05, 0.0) for index in range(61))
        self.cumulative = cumulative_lengths(self.path)
        self.disc = (1.0, 0.0)
        self.config = PassageConfig()

    def update(self, state, time_sec, robot_x, robot_y=0.0, yaw=0.0):
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        delta_x = self.disc[0] - robot_x
        delta_y = self.disc[1] - robot_y
        disc_base = (
            cosine * delta_x + sine * delta_y,
            -sine * delta_x + cosine * delta_y,
        )
        return update_passage(
            state=state,
            now_sec=time_sec,
            disc_fixed_xy=self.disc,
            disc_base_xy=disc_base,
            robot_fixed_xy=(robot_x, robot_y),
            robot_yaw=yaw,
            latest_path=self.path,
            latest_cumulative_s=self.cumulative,
            config=self.config,
        )

    def activated_state(self):
        state = PassageState()
        state.activate(1)
        return state

    def enter_candidate(self, state, lateral_y=0.0):
        self.update(state, 0.0, 0.0, lateral_y)
        self.update(state, 0.5, 0.80, lateral_y)
        positions = [
            0.85,
            0.90,
            0.95,
            1.00,
            1.05,
            1.10,
            1.15,
            1.20,
            1.25,
            1.27,
        ]
        result = None
        for index, position in enumerate(positions, start=1):
            result = self.update(
                state,
                0.5 + index * 0.1,
                position,
                lateral_y,
            )
        self.assertFalse(result.entered_tombstone)
        self.assertEqual(state.phase, PHASE_PASSED_CANDIDATE)
        return 1.5, 1.27

    def test_rotation_without_translation_never_passes_disc(self):
        state = self.activated_state()
        for index in range(30):
            result = self.update(
                state,
                index * 0.1,
                0.0,
                yaw=index * 0.2,
            )
            self.assertFalse(result.entered_tombstone)
        self.assertEqual(state.phase, PHASE_AHEAD)
        self.assertTrue(state.ever_ahead)

    def test_lateral_pass_requires_hold_frames_and_extra_advance(self):
        state = self.activated_state()
        start_time, start_x = self.enter_candidate(
            state, lateral_y=0.30
        )
        entered = False
        for index in range(1, 13):
            result = self.update(
                state,
                start_time + index * 0.1,
                start_x + index * 0.011,
                0.30,
            )
            entered = entered or result.entered_tombstone
        self.assertTrue(entered)
        self.assertEqual(state.phase, PHASE_TOMBSTONE)

    def test_stopping_in_candidate_does_not_remove_obstacle(self):
        state = self.activated_state()
        start_time, start_x = self.enter_candidate(state)
        for index in range(1, 31):
            result = self.update(
                state, start_time + index * 0.1, start_x
            )
            self.assertFalse(result.entered_tombstone)
        self.assertEqual(state.phase, PHASE_PASSED_CANDIDATE)
        self.assertAlmostEqual(
            state.debug["candidate_advance"], 0.0, places=9
        )

    def test_candidate_backtrack_resets_confirmation(self):
        state = self.activated_state()
        start_time, _ = self.enter_candidate(state)
        self.update(state, start_time + 0.1, 1.30)
        result = self.update(state, start_time + 0.2, 1.25)
        self.assertFalse(result.entered_tombstone)
        self.assertEqual(state.phase, PHASE_ABREAST)
        self.assertEqual(state.candidate_frames, 0)
        self.assertIsNone(state.candidate_start_sec)

    def test_tombstone_reactivates_only_after_upstream_return(self):
        state = self.activated_state()
        start_time, start_x = self.enter_candidate(state)
        for index in range(1, 13):
            self.update(
                state,
                start_time + index * 0.1,
                start_x + index * 0.011,
            )
        self.assertEqual(state.phase, PHASE_TOMBSTONE)
        self.assertFalse(
            tombstone_should_reactivate(
                state, (1.45, 0.0), self.config
            )
        )
        self.assertTrue(
            tombstone_should_reactivate(
                state, (0.70, 0.0), self.config
            )
        )

    def test_circle_clearance_uses_real_rectangle_geometry(self):
        self.assertAlmostEqual(
            circle_to_rectangle_clearance(
                0.265, 0.0, 0.165, 0.150, 0.050
            ),
            0.050,
            places=9,
        )
        self.assertLess(
            circle_to_rectangle_clearance(
                0.18, 0.0, 0.165, 0.150, 0.050
            ),
            0.0,
        )

    def test_tombstone_match_suppresses_new_nearby_id(self):
        memory = StableDiscMemory(30.0, 0.8, 0.12, 0.25, 0.15)
        passed = memory.observe(4, self.disc, 1.0, True)
        passed.passage.phase = PHASE_TOMBSTONE
        passed.passage.passed_sec = 2.0
        matched = memory.observe(9, (1.04, 0.01), 3.0, True)
        self.assertEqual(matched.memory_id, passed.memory_id)
        self.assertEqual(len(memory), 1)
        self.assertEqual(matched.fixed_xy, self.disc)

    def test_reused_track_id_cannot_suppress_far_new_disc(self):
        memory = StableDiscMemory(30.0, 0.8, 0.12, 0.25, 0.15)
        passed = memory.observe(4, self.disc, 1.0, True)
        passed.passage.phase = PHASE_TOMBSTONE
        passed.passage.passed_sec = 2.0
        new_disc = memory.observe(4, (2.0, 0.0), 3.0, True)
        self.assertNotEqual(new_disc.memory_id, passed.memory_id)
        self.assertEqual(len(memory), 2)


if __name__ == "__main__":
    unittest.main()
