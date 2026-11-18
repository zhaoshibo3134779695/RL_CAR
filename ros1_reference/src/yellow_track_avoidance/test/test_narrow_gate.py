#!/usr/bin/env python3

import math
import unittest

from yellow_track_avoidance.narrow_gate import (
    DiscTrack,
    NarrowGateConfig,
    NarrowGateStateMachine,
    RobotPose,
    STATE_ALIGN,
    STATE_ARMED,
    STATE_CROSSING,
    STATE_EXIT_CLEAR,
    STATE_FAILED_STOP,
    STATE_NORMAL,
    STATE_STOP_RELEASE,
    gate_errors,
    projected_vehicle_width,
    select_gate_candidate,
)


class NarrowGateGeometryTests(unittest.TestCase):
    def setUp(self):
        self.config = NarrowGateConfig()
        self.path = [(0.0, 0.0), (2.0, 0.0)]
        self.robot = RobotPose(0.0, 0.0, 0.0)

    @staticmethod
    def gate_tracks(center_distance=0.462, x=1.0):
        half = 0.5 * center_distance
        return [
            DiscTrack(11, (x, -half)),
            DiscTrack(12, (x, half)),
        ]

    def test_close_pair_straddling_path_is_selected(self):
        candidate = select_gate_candidate(
            self.gate_tracks(), self.path, self.robot, self.config
        )
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.frame.pair_ids, (11, 12))
        self.assertAlmostEqual(candidate.frame.edge_gap_m, 0.362, places=6)
        self.assertAlmostEqual(candidate.frame.midpoint_xy[1], 0.0)
        self.assertGreater(candidate.frame.travel_xy[0], 0.99)
        self.assertAlmostEqual(candidate.robot_progress_m, -1.0)

    def test_wide_pair_keeps_normal_navigation(self):
        candidate = select_gate_candidate(
            self.gate_tracks(center_distance=0.56),
            self.path,
            self.robot,
            self.config,
        )
        self.assertIsNone(candidate)

    def test_close_pair_on_same_side_of_path_is_rejected(self):
        tracks = [
            DiscTrack(1, (1.0, 0.15)),
            DiscTrack(2, (1.0, 0.50)),
        ]
        self.assertIsNone(
            select_gate_candidate(
                tracks, self.path, self.robot, self.config
            )
        )

    def test_rejection_diagnostics_report_each_failed_rule(self):
        tracks = [
            DiscTrack(
                1,
                (1.0, 0.15),
                estimated_diameter_m=0.064,
                confidence=0.91,
            ),
            DiscTrack(
                2,
                (1.0, 0.50),
                estimated_diameter_m=0.071,
                confidence=0.87,
            ),
        ]
        diagnostics = []
        candidate = select_gate_candidate(
            tracks,
            self.path,
            self.robot,
            self.config,
            pair_diagnostics=diagnostics,
        )
        self.assertIsNone(candidate)
        self.assertEqual(len(diagnostics), 1)
        diagnostic = diagnostics[0]
        self.assertEqual(diagnostic.pair_ids, (1, 2))
        self.assertEqual(diagnostic.first_diameter_m, 0.064)
        self.assertEqual(diagnostic.second_confidence, 0.87)
        self.assertIn("NOT_STRADDLING_ROUTE", diagnostic.reason_codes)
        checks = {item["code"]: item for item in diagnostic.checks}
        self.assertFalse(checks["NOT_STRADDLING_ROUTE"]["passed"])
        self.assertTrue(checks["LONGITUDINAL_MISMATCH"]["passed"])
        self.assertIn("actual", checks["MIDPOINT_TOO_FAR_FROM_ROUTE"])
        self.assertIn("threshold", checks["MIDPOINT_TOO_FAR_FROM_ROUTE"])

    def test_accepted_pair_still_reports_nonblocking_clearance(self):
        diagnostics = []
        candidate = select_gate_candidate(
            self.gate_tracks(),
            self.path,
            self.robot,
            self.config,
            pair_diagnostics=diagnostics,
        )
        self.assertIsNotNone(candidate)
        self.assertEqual(len(diagnostics), 1)
        self.assertTrue(diagnostics[0].accepted)
        clearance = next(
            item
            for item in diagnostics[0].checks
            if item["code"] == "PREDICTED_CLEARANCE_BELOW_MINIMUM"
        )
        self.assertFalse(clearance["blocking"])

    def test_pair_parallel_to_path_is_rejected(self):
        tracks = [
            DiscTrack(1, (0.80, 0.0)),
            DiscTrack(2, (1.20, 0.0)),
        ]
        self.assertIsNone(
            select_gate_candidate(
                tracks, self.path, self.robot, self.config
            )
        )

    def test_unconfirmed_and_tombstone_tracks_are_rejected(self):
        unconfirmed = [
            DiscTrack(1, (1.0, -0.20), confirmed=False),
            DiscTrack(2, (1.0, 0.20)),
        ]
        tombstone = [
            DiscTrack(1, (1.0, -0.20), phase="tombstone"),
            DiscTrack(2, (1.0, 0.20)),
        ]
        self.assertIsNone(
            select_gate_candidate(
                unconfirmed, self.path, self.robot, self.config
            )
        )
        self.assertIsNone(
            select_gate_candidate(
                tombstone, self.path, self.robot, self.config
            )
        )

    def test_stale_tracks_do_not_arm_gate(self):
        stale = [
            DiscTrack(1, (1.0, -0.20), last_seen_age_sec=0.41),
            DiscTrack(2, (1.0, 0.20), last_seen_age_sec=0.41),
        ]
        self.assertIsNone(
            select_gate_candidate(
                stale, self.path, self.robot, self.config
            )
        )

    def test_projected_width_matches_rectangle_geometry(self):
        width = projected_vehicle_width(
            math.radians(2.0), 0.165, 0.150
        )
        self.assertAlmostEqual(width, 0.311334, places=6)

        candidate = select_gate_candidate(
            self.gate_tracks(), self.path, self.robot, self.config
        )
        errors = gate_errors(
            RobotPose(1.0, 0.0, math.radians(2.0)),
            candidate.frame,
            self.config,
        )
        self.assertAlmostEqual(
            errors.predicted_clearance_m, 0.025333, places=6
        )


class NarrowGateStateMachineTests(unittest.TestCase):
    def setUp(self):
        self.config = NarrowGateConfig()
        self.machine = NarrowGateStateMachine(self.config)
        self.path = [(0.0, 0.0), (2.0, 0.0)]
        self.tracks = NarrowGateGeometryTests.gate_tracks()

    def update(self, time_sec, x, y=0.0, yaw=0.0, **kwargs):
        return self.machine.update(
            now_sec=time_sec,
            robot=RobotPose(x, y, yaw),
            tracks=kwargs.get("tracks", self.tracks),
            path=kwargs.get("path", self.path),
            navigation_active=kwargs.get("navigation_active", True),
            inputs_fresh=kwargs.get("inputs_fresh", True),
        )

    def test_normal_navigation_for_open_field(self):
        decision = self.machine.update(
            0.0,
            RobotPose(0.0, 0.0, 0.0),
            NarrowGateGeometryTests.gate_tracks(center_distance=0.60),
            self.path,
            True,
        )
        self.assertEqual(decision.state, STATE_NORMAL)
        self.assertFalse(decision.authority_requested)

    def test_candidate_must_persist_before_takeover(self):
        first = self.update(0.0, 0.0)
        self.assertEqual(first.state, STATE_ARMED)
        self.assertFalse(first.authority_requested)

        early = self.update(0.30, 0.50)
        self.assertEqual(early.state, STATE_ARMED)
        self.assertFalse(early.cancel_move_base)

        takeover = self.update(0.61, 0.50)
        self.assertEqual(takeover.state, STATE_ALIGN)
        self.assertTrue(takeover.authority_requested)
        self.assertTrue(takeover.cancel_move_base)

    def test_gap_hysteresis_does_not_drop_armed_pair(self):
        narrow = NarrowGateGeometryTests.gate_tracks(
            center_distance=0.54
        )
        wider = NarrowGateGeometryTests.gate_tracks(
            center_distance=0.56
        )
        first = self.machine.update(
            0.0, RobotPose(0.0, 0.0, 0.0), narrow, self.path, True
        )
        self.assertEqual(first.state, STATE_ARMED)
        second = self.machine.update(
            0.2, RobotPose(0.0, 0.0, 0.0), wider, self.path, True
        )
        self.assertEqual(second.state, STATE_ARMED)

    def test_geometry_jump_restarts_pair_confirmation(self):
        self.update(0.0, 0.0)
        shifted = NarrowGateGeometryTests.gate_tracks(x=1.03)
        after_jump = self.update(0.59, 0.49, tracks=shifted)
        self.assertEqual(after_jump.state, STATE_ARMED)
        self.assertAlmostEqual(after_jump.candidate_stable_sec, 0.0)

        still_confirming = self.update(0.80, 0.50, tracks=shifted)
        self.assertEqual(still_confirming.state, STATE_ARMED)
        self.assertFalse(still_confirming.cancel_move_base)

        takeover = self.update(1.20, 0.53, tracks=shifted)
        self.assertEqual(takeover.state, STATE_ALIGN)
        self.assertTrue(takeover.cancel_move_base)

    def test_complete_crossing_releases_and_requests_resume(self):
        self.update(0.0, 0.0)
        self.update(0.61, 0.50)

        aligning = self.update(0.70, 0.50)
        self.assertEqual(aligning.state, STATE_ALIGN)
        crossing = self.update(1.01, 0.50)
        self.assertEqual(crossing.state, STATE_CROSSING)

        at_gate = self.update(1.10, 1.00)
        self.assertEqual(at_gate.state, STATE_CROSSING)
        self.assertGreater(at_gate.command_body[0], 0.0)

        exit_clear = self.update(1.20, 1.27)
        self.assertEqual(exit_clear.state, STATE_EXIT_CLEAR)
        stopping = self.update(1.30, 1.36)
        self.assertEqual(stopping.state, STATE_STOP_RELEASE)
        self.assertEqual(stopping.command_body, (0.0, 0.0, 0.0))

        release = self.update(1.61, 1.36)
        self.assertEqual(release.state, STATE_NORMAL)
        self.assertFalse(release.authority_requested)
        self.assertTrue(release.resume_move_base)

    def test_lateral_error_fails_closed(self):
        self.update(0.0, 0.0)
        self.update(0.61, 0.50)
        self.update(0.70, 0.50)
        self.update(1.01, 0.50)
        failed = self.update(1.10, 0.80, y=0.03)
        self.assertEqual(failed.state, STATE_FAILED_STOP)
        self.assertTrue(failed.authority_requested)
        self.assertEqual(failed.command_body, (0.0, 0.0, 0.0))

    def test_alignment_can_correct_yaw_before_entering_gate(self):
        self.update(0.0, 0.0)
        takeover = self.update(
            0.61, 0.50, yaw=math.radians(10.0)
        )
        self.assertEqual(takeover.state, STATE_ALIGN)
        correcting = self.update(
            0.70, 0.50, yaw=math.radians(10.0)
        )
        self.assertEqual(correcting.state, STATE_ALIGN)
        self.assertLess(correcting.command_body[2], 0.0)

    def test_input_timeout_after_takeover_fails_closed(self):
        self.update(0.0, 0.0)
        self.update(0.61, 0.50)
        failed = self.update(0.70, 0.50, inputs_fresh=False)
        self.assertEqual(failed.state, STATE_FAILED_STOP)
        self.assertEqual(failed.reason, "gate_input_timeout")

    def test_reset_is_required_after_failure(self):
        self.update(0.0, 0.0)
        self.update(0.61, 0.50)
        self.update(0.70, 0.50, inputs_fresh=False)
        still_failed = self.update(1.0, 0.50)
        self.assertEqual(still_failed.state, STATE_FAILED_STOP)
        self.machine.reset()
        normal = self.machine.update(
            1.1,
            RobotPose(0.0, 0.0, 0.0),
            [],
            self.path,
            False,
        )
        self.assertEqual(normal.state, STATE_NORMAL)


if __name__ == "__main__":
    unittest.main()
