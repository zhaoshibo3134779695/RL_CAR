#!/usr/bin/env python3

import math
import unittest

from yellow_track_avoidance.narrow_gate import DiscTrack, RobotPose
from yellow_track_avoidance.passage_navigation import (
    BoundaryLines,
    PassageConfig,
    PassageStateMachine,
    SceneAssessment,
    SCENE_BLOCKED,
    SCENE_BYPASS,
    SCENE_GATE,
    SCENE_MULTI,
    SCENE_NORMAL,
    STATE_DEGRADED_HOLD,
    STATE_EXIT_RESUME,
    STATE_FAILED_STOP,
    STATE_GATE_ALIGN,
    STATE_NORMAL,
    STATE_RECOVERABLE_HOLD,
    STATE_SINGLE_AVOID,
    STATE_TAKEOVER_BRAKE,
    assess_scene,
    infer_parallel_boundary,
    odom_corridor_cache_valid,
    passage_attempt_key,
    point_in_polygon,
    register_passage_attempt,
    robot_boundary_clearance,
    swept_command_safety,
    route_tangent_at_point,
)


class PassageGeometryTests(unittest.TestCase):
    def setUp(self):
        self.path = ((0.0, 0.0), (3.0, 0.0))
        # BEV geometry starts in front of the camera/robot.  The planner must
        # conservatively extend this first measured cross-section to the base.
        self.boundary = BoundaryLines(
            left_world=((0.39, 0.455), (2.50, 0.455)),
            right_world=((0.39, -0.455), (2.50, -0.455)),
            yellow_line_width_m=0.030,
            quality=1.0,
            fresh=True,
        )
        self.robot = RobotPose(0.0, 0.0, 0.0)
        self.config = PassageConfig()

    def test_odom_corridor_cache_has_time_and_distance_bounds(self):
        self.assertTrue(odom_corridor_cache_valid(
            10.8, 10.0, (1.0, 2.0), RobotPose(1.04, 2.0, 0.0),
            1.2, 0.9, 0.05,
        ))
        self.assertFalse(odom_corridor_cache_valid(
            11.0, 10.0, (1.0, 2.0), RobotPose(1.04, 2.0, 0.0),
            1.2, 0.9, 0.05,
        ))
        self.assertFalse(odom_corridor_cache_valid(
            10.4, 10.0, (1.0, 2.0), RobotPose(1.051, 2.0, 0.0),
            1.2, 0.9, 0.05,
        ))

    def test_passage_attempt_is_latched_per_external_goal(self):
        attempts = set()
        assessment = SceneAssessment(
            SCENE_GATE, ("gate", 11, 12), "true_narrow_gate"
        )
        first = passage_attempt_key(4, assessment)
        self.assertFalse(register_passage_attempt(attempts, first))
        self.assertTrue(register_passage_attempt(attempts, first))
        next_goal = passage_attempt_key(5, assessment)
        self.assertFalse(register_passage_attempt(attempts, next_goal))

    def assess(self, tracks, boundary=None):
        return assess_scene(
            tracks,
            self.path,
            self.robot,
            self.boundary if boundary is None else boundary,
            self.config,
        )

    def test_empty_and_stale_tracks_keep_normal_navigation(self):
        self.assertEqual(self.assess([]).scene, SCENE_NORMAL)
        stale = DiscTrack(
            1,
            (1.0, 0.0),
            last_seen_age_sec=self.config.track_fresh_sec + 0.01,
        )
        self.assertEqual(self.assess([stale]).scene, SCENE_NORMAL)

    def test_disc_outside_route_and_road_is_ignored(self):
        assessment = self.assess([DiscTrack(1, (1.0, 0.70))])
        self.assertEqual(assessment.scene, SCENE_NORMAL)

    def test_single_disc_selects_larger_right_passage(self):
        assessment = self.assess([DiscTrack(1, (1.0, 0.10))])
        self.assertEqual(assessment.scene, SCENE_BYPASS)
        self.assertEqual(assessment.bypass_plan.selected_side, -1)
        self.assertGreater(
            assessment.bypass_plan.right_residual_m,
            assessment.bypass_plan.left_residual_m,
        )

    def test_center_disc_has_two_feasible_sides(self):
        assessment = self.assess([DiscTrack(1, (1.0, 0.0))])
        self.assertEqual(assessment.scene, SCENE_BYPASS)
        self.assertAlmostEqual(
            assessment.bypass_plan.left_residual_m, 0.025
        )
        self.assertAlmostEqual(
            assessment.bypass_plan.right_residual_m, 0.025
        )

    def test_irrelevant_second_disc_does_not_change_topology(self):
        assessment = self.assess(
            [
                DiscTrack(1, (1.0, 0.10)),
                DiscTrack(2, (1.0, 0.70)),
            ]
        )
        self.assertEqual(assessment.scene, SCENE_BYPASS)
        self.assertEqual(assessment.affecting_ids, (1,))

    def test_two_same_side_discs_use_multi_channel(self):
        assessment = self.assess(
            [
                DiscTrack(1, (0.90, 0.05)),
                DiscTrack(2, (1.15, 0.28)),
            ]
        )
        self.assertEqual(assessment.scene, SCENE_MULTI)
        self.assertEqual(
            set(assessment.bypass_plan.obstacle_ids), {1, 2}
        )

    def test_competition_pair_forces_all_three_layouts_between_discs(self):
        self.config.force_between_disc_pair = True
        layouts = (
            ((0.65, 0.28), (1.05, -0.28)),
            ((0.65, 0.20), (1.05, -0.32)),
            ((0.65, -0.28), (1.05, 0.28)),
        )
        obstacle_half_width = (
            self.config.vehicle_half_width_m
            + self.config.disc_radius_m
            + self.config.disc_clearance_margin_m
            + self.config.geometry_uncertainty_m
        )
        for index, positions in enumerate(layouts):
            with self.subTest(layout=index):
                assessment = self.assess(
                    [
                        DiscTrack(1, positions[0]),
                        DiscTrack(2, positions[1]),
                    ]
                )
                self.assertEqual(assessment.scene, SCENE_MULTI)
                self.assertEqual(
                    assessment.reason, "forced_between_disc_pair"
                )
                plan = assessment.bypass_plan
                for progress, lateral in positions:
                    reference = plan.lateral_at(progress)
                    if lateral == min(item[1] for item in positions):
                        self.assertGreaterEqual(
                            reference,
                            lateral + obstacle_half_width - 1e-6,
                        )
                    else:
                        self.assertLessEqual(
                            reference,
                            lateral - obstacle_half_width + 1e-6,
                        )

    def test_competition_mode_runs_only_geometrically_valid_disc_gate(self):
        self.config.competition_two_disc_mode = True
        self.config.vehicle_half_length_m = 0.170
        self.config.vehicle_half_width_m = 0.165
        self.config.minimum_predicted_gate_clearance_m = 0.010
        single = self.assess([DiscTrack(1, (1.0, 0.0))])
        self.assertEqual(single.scene, SCENE_NORMAL)
        self.assertEqual(
            single.reason, "competition_waiting_for_exact_pair"
        )

        pair = self.assess(
            [DiscTrack(1, (1.0, -0.25)), DiscTrack(2, (1.0, 0.25))]
        )
        self.assertEqual(pair.scene, SCENE_GATE)
        self.assertEqual(pair.reason, "competition_two_disc_gate")
        self.assertTrue(pair.admission_safety.safe)
        self.assertGreaterEqual(
            pair.admission_safety.minimum_disc_clearance_m, 0.010
        )
        self.assertAlmostEqual(
            pair.admission_safety.minimum_disc_clearance_m, 0.035
        )

        pair_with_unrelated_track = self.assess(
            [
                DiscTrack(1, (1.0, -0.25)),
                DiscTrack(2, (1.0, 0.25)),
                DiscTrack(3, (0.8, 0.70)),
            ]
        )
        self.assertEqual(pair_with_unrelated_track.scene, SCENE_GATE)
        self.assertEqual(
            pair_with_unrelated_track.gate_frame.pair_ids, (1, 2)
        )

    def test_competition_mode_rejects_full_sweep_over_yellow(self):
        self.config.competition_two_disc_mode = True
        self.config.vehicle_half_length_m = 0.170
        self.config.vehicle_half_width_m = 0.165
        self.config.gate_special_mode_max_gap_m = 0.80
        self.config.gate_release_gap_m = 0.85
        self.config.gate_longitudinal_mismatch_max_m = 0.50
        self.config.gate_normal_angle_tolerance_rad = math.radians(55.0)
        assessment = self.assess(
            [DiscTrack(1, (0.8, 0.28)), DiscTrack(2, (1.2, -0.28))]
        )
        self.assertEqual(assessment.scene, SCENE_BLOCKED)
        self.assertEqual(assessment.reason, "admission_yellow_clearance")
        self.assertFalse(assessment.admission_safety.safe)

    def test_current_car_rejects_40cm_gate_below_4cm_margin(self):
        self.config.competition_two_disc_mode = True
        self.config.vehicle_half_length_m = 0.170
        self.config.vehicle_half_width_m = 0.165
        self.config.minimum_predicted_gate_clearance_m = 0.040
        assessment = self.assess(
            [DiscTrack(1, (1.0, -0.25)), DiscTrack(2, (1.0, 0.25))]
        )
        self.assertEqual(assessment.scene, SCENE_BLOCKED)
        self.assertEqual(assessment.reason, "gate_too_narrow")

    def test_route_tangent_comes_from_local_sbpl_segment(self):
        path = ((0.0, 0.0), (0.8, 0.0), (1.3, 0.5), (2.0, 1.2))
        tangent = route_tangent_at_point((1.1, 0.3), path, 0.4)
        self.assertIsNotNone(tangent)
        self.assertAlmostEqual(tangent[0], math.sqrt(0.5), places=6)
        self.assertAlmostEqual(tangent[1], math.sqrt(0.5), places=6)

    def test_true_close_pair_uses_gate_controller(self):
        assessment = self.assess(
            [
                DiscTrack(11, (1.0, -0.231)),
                DiscTrack(12, (1.0, 0.231)),
            ]
        )
        self.assertEqual(assessment.scene, SCENE_GATE)
        self.assertAlmostEqual(
            assessment.gate_frame.edge_gap_m, 0.362
        )

    def test_formal_profile_disables_gate_special_case(self):
        self.config.enable_gate_special_mode = False
        assessment = self.assess(
            [
                DiscTrack(11, (1.0, -0.231)),
                DiscTrack(12, (1.0, 0.231)),
            ]
        )
        self.assertNotEqual(assessment.scene, SCENE_GATE)
        self.assertIsNone(assessment.gate_frame)

    def test_formal_profile_claims_disc_anywhere_inside_road(self):
        self.config.enable_gate_special_mode = False
        self.config.claim_all_road_discs = True
        self.config.vehicle_half_length_m = 0.170
        self.config.vehicle_half_width_m = 0.165
        assessment = self.assess([DiscTrack(1, (0.80, 0.35))])
        self.assertEqual(assessment.scene, SCENE_MULTI)
        self.assertEqual(assessment.affecting_ids, (1,))
        self.assertAlmostEqual(
            assessment.bypass_plan.lateral_at(0.80), 0.0
        )

    def test_multi_channel_exposes_read_only_diagnostics(self):
        self.config.enable_gate_special_mode = False
        self.config.vehicle_half_length_m = 0.170
        self.config.vehicle_half_width_m = 0.165
        diagnostics = {}
        assessment = assess_scene(
            [
                DiscTrack(1, (0.31, -0.24)),
                DiscTrack(2, (1.20, 0.24)),
            ],
            self.path,
            self.robot,
            self.boundary,
            self.config,
            planner_diagnostics=diagnostics,
        )
        self.assertEqual(assessment.scene, SCENE_MULTI)
        self.assertEqual(
            diagnostics["planner"],
            "free_interval_dynamic_programming",
        )
        self.assertFalse(diagnostics["force_between_pair"])
        self.assertEqual(diagnostics["failure_reason"], "")
        self.assertEqual(
            len(diagnostics["sample_s_m"]),
            len(diagnostics["free_intervals_m"]),
        )
        self.assertEqual(
            tuple(assessment.bypass_plan.reference_d_m),
            diagnostics["selected_lateral_m"],
        )

    def test_staggered_close_pair_uses_gate_controller(self):
        assessment = self.assess(
            [
                DiscTrack(8, (0.527, 0.230), last_seen_age_sec=1.50),
                DiscTrack(11, (0.742, -0.242), last_seen_age_sec=0.02),
            ]
        )
        self.assertEqual(assessment.scene, SCENE_GATE)
        self.assertAlmostEqual(
            assessment.gate_frame.edge_gap_m,
            0.41866077545925906,
        )
        self.assertAlmostEqual(
            math.degrees(assessment.gate_frame.yaw_ref),
            24.489667056909443,
        )

    def test_wide_pair_does_not_trigger_gate_mode(self):
        assessment = self.assess(
            [
                DiscTrack(11, (1.0, -0.280)),
                DiscTrack(12, (1.0, 0.280)),
            ]
        )
        self.assertNotEqual(assessment.scene, SCENE_GATE)

    def test_too_narrow_gate_is_latched_stop_candidate(self):
        assessment = self.assess(
            [
                DiscTrack(11, (1.0, -0.210)),
                DiscTrack(12, (1.0, 0.210)),
            ]
        )
        self.assertEqual(assessment.scene, SCENE_BLOCKED)
        self.assertEqual(assessment.reason, "gate_too_narrow")

    def test_single_disc_with_no_safe_side_is_blocked(self):
        boundary = BoundaryLines(
            left_world=((0.39, 0.405), (2.50, 0.405)),
            right_world=((0.39, -0.405), (2.50, -0.405)),
            yellow_line_width_m=0.030,
            quality=1.0,
            fresh=True,
        )
        assessment = self.assess(
            [DiscTrack(1, (1.0, 0.0))], boundary=boundary
        )
        self.assertEqual(assessment.scene, SCENE_BLOCKED)
        self.assertIn("no_safe", assessment.reason)

    def test_single_disc_reports_uncovered_near_field_boundary(self):
        boundary = BoundaryLines(
            left_world=((1.10, 0.455), (2.50, 0.455)),
            right_world=((1.10, -0.455), (2.50, -0.455)),
            yellow_line_width_m=0.030,
            quality=1.0,
            fresh=True,
        )
        assessment = self.assess(
            [DiscTrack(1, (1.0, 0.10))], boundary=boundary
        )
        self.assertEqual(assessment.scene, SCENE_BLOCKED)
        self.assertEqual(
            assessment.reason, "single_disc_boundary_uncovered"
        )


class PassageSafetyGeometryTests(unittest.TestCase):
    def setUp(self):
        self.path = ((0.0, 0.0), (3.0, 0.0))
        self.boundary = BoundaryLines(
            left_world=((0.40, 0.460), (2.50, 0.460)),
            right_world=((0.40, -0.460), (2.50, -0.460)),
            yellow_line_width_m=0.030,
            quality=1.0,
            fresh=True,
        )
        self.config = PassageConfig(
            vehicle_half_length_m=0.170,
            vehicle_half_width_m=0.165,
        )

    def test_activation_polygon_includes_edges_and_rejects_outside(self):
        polygon = (
            (-0.65, 0.85),
            (2.35, 0.85),
            (2.35, 1.95),
            (-0.65, 1.95),
        )
        self.assertTrue(point_in_polygon((2.0395, 1.5870), polygon))
        self.assertTrue(point_in_polygon((-0.60, 1.00), polygon))
        self.assertFalse(point_in_polygon((-0.66, 1.50), polygon))
        self.assertFalse(point_in_polygon((1.0, 0.84), polygon))

    def test_metric_right_boundary_is_inferred_from_physical_width(self):
        inferred = infer_parallel_boundary(
            ((0.55, 0.455), (1.55, 0.455)), -0.91
        )
        self.assertEqual(len(inferred), 2)
        self.assertAlmostEqual(inferred[0][0], 0.55)
        self.assertAlmostEqual(inferred[0][1], -0.455)
        self.assertAlmostEqual(inferred[1][1], -0.455)

    def test_metric_inference_handles_slanted_observed_line(self):
        observed = ((0.50, 0.50), (1.50, 0.30))
        inferred = infer_parallel_boundary(observed, -0.91)
        for source, shifted in zip(observed, inferred):
            self.assertAlmostEqual(
                math.hypot(
                    source[0] - shifted[0],
                    source[1] - shifted[1],
                ),
                0.91,
            )
            self.assertLess(shifted[1], source[1])

    def test_rectangle_clearance_uses_full_body_and_yellow_width(self):
        centered = robot_boundary_clearance(
            RobotPose(0.0, 0.0, 0.0),
            self.path,
            self.boundary,
            self.config,
        )
        self.assertAlmostEqual(centered, 0.280, places=6)
        touching = robot_boundary_clearance(
            RobotPose(0.0, 0.280, 0.0),
            self.path,
            self.boundary,
            self.config,
        )
        self.assertAlmostEqual(touching, 0.0, places=6)

    def test_swept_guard_rejects_command_that_will_cross_yellow(self):
        result = swept_command_safety(
            robot=RobotPose(0.0, 0.20, 0.0),
            command_body=(0.0, 0.30, 0.0),
            path=self.path,
            boundary=self.boundary,
            tracks=(),
            config=self.config,
        )
        self.assertFalse(result.safe)
        self.assertEqual(result.reason, "swept_yellow_clearance")
        self.assertLess(
            result.minimum_line_clearance_m,
            self.config.emergency_line_clearance_m,
        )

    def test_swept_guard_rejects_future_disc_contact(self):
        result = swept_command_safety(
            robot=RobotPose(0.0, 0.0, 0.0),
            command_body=(0.50, 0.0, 0.0),
            path=self.path,
            boundary=self.boundary,
            tracks=(DiscTrack(1, (0.25, 0.0)),),
            config=self.config,
        )
        self.assertFalse(result.safe)
        self.assertEqual(result.reason, "swept_disc_clearance")

    def test_swept_guard_accepts_centered_low_speed_motion(self):
        result = swept_command_safety(
            robot=RobotPose(0.0, 0.0, 0.0),
            command_body=(0.05, 0.0, 0.0),
            path=self.path,
            boundary=self.boundary,
            tracks=(),
            config=self.config,
        )
        self.assertTrue(result.safe)
        self.assertGreater(result.sample_count, 1)


class PassageStateMachineTests(unittest.TestCase):
    def setUp(self):
        self.config = PassageConfig(
            scene_confirm_sec=0.20,
            handoff_stop_hold_sec=0.10,
            exit_stop_hold_sec=0.10,
        )
        self.machine = PassageStateMachine(self.config)
        self.path = ((0.0, 0.0), (3.0, 0.0))
        self.boundary = BoundaryLines(
            left_world=((0.39, 0.455), (2.50, 0.455)),
            right_world=((0.39, -0.455), (2.50, -0.455)),
            yellow_line_width_m=0.030,
            quality=1.0,
            fresh=True,
        )
        self.single = [DiscTrack(1, (1.0, 0.10))]

    def update(
        self,
        now,
        x=0.0,
        y=0.0,
        yaw=0.0,
        tracks=None,
        navigation_active=True,
        linear_speed=0.0,
        angular_speed=0.0,
        fresh=True,
        mission_active=None,
        live_gate_tracks=(),
    ):
        return self.machine.update(
            now_sec=now,
            robot=RobotPose(x, y, yaw),
            tracks=self.single if tracks is None else tracks,
            path=self.path,
            boundary=self.boundary,
            navigation_active=navigation_active,
            measured_linear_speed_mps=linear_speed,
            measured_angular_speed_radps=angular_speed,
            critical_inputs_fresh=fresh,
            mission_active=mission_active,
            live_gate_tracks=live_gate_tracks,
        )

    def test_single_disc_confirm_cancel_brake_and_follow(self):
        first = self.update(0.0)
        self.assertEqual(first.state, STATE_NORMAL)
        self.assertFalse(first.cancel_move_base)

        takeover = self.update(0.21)
        self.assertEqual(takeover.state, STATE_TAKEOVER_BRAKE)
        self.assertTrue(takeover.cancel_move_base)
        self.assertTrue(takeover.authority_requested)

        braking = self.update(
            0.22, navigation_active=False
        )
        self.assertEqual(braking.state, STATE_TAKEOVER_BRAKE)
        active = self.update(
            0.33, navigation_active=False
        )
        self.assertEqual(active.state, STATE_SINGLE_AVOID)

        following = self.update(
            0.38, x=0.40, navigation_active=False
        )
        self.assertEqual(following.state, STATE_SINGLE_AVOID)
        self.assertGreater(following.command_body[0], 0.0)
        self.assertLess(following.command_body[1], 0.0)

    def test_recent_move_base_failure_can_still_trigger_takeover(self):
        first = self.update(
            0.0,
            navigation_active=False,
            mission_active=True,
        )
        self.assertEqual(first.assessment.scene, SCENE_BYPASS)
        takeover = self.update(
            0.21,
            navigation_active=False,
            mission_active=True,
        )
        self.assertEqual(takeover.state, STATE_TAKEOVER_BRAKE)
        self.assertTrue(takeover.cancel_move_base)

    def test_bypass_completion_stops_then_resumes_navigation(self):
        self.update(0.0)
        self.update(0.21)
        self.update(0.22, navigation_active=False)
        self.update(0.33, navigation_active=False)
        stopping = self.update(
            0.40, x=2.00, navigation_active=False
        )
        self.assertEqual(stopping.state, STATE_EXIT_RESUME)
        self.assertEqual(stopping.command_body, (0.0, 0.0, 0.0))

        resumed = self.update(
            0.51, x=2.00, navigation_active=False
        )
        self.assertEqual(resumed.state, STATE_NORMAL)
        self.assertTrue(resumed.resume_move_base)
        self.assertFalse(resumed.authority_requested)

    def test_input_loss_holds_then_fails_closed(self):
        self.update(0.0)
        self.update(0.21)
        hold = self.update(0.22, fresh=False)
        self.assertEqual(hold.state, STATE_DEGRADED_HOLD)
        self.assertEqual(hold.command_body, (0.0, 0.0, 0.0))

        failed = self.update(
            0.22 + self.config.degraded_hold_sec + 0.01,
            fresh=False,
        )
        self.assertEqual(failed.state, STATE_FAILED_STOP)
        self.assertTrue(failed.authority_requested)

    def test_finish_priority_resumes_goal_after_input_timeout(self):
        self.config.competition_finish_priority = True
        self.machine = PassageStateMachine(self.config)
        self.update(0.0)
        self.update(0.21)
        hold = self.update(0.22, fresh=False)
        self.assertEqual(hold.state, STATE_DEGRADED_HOLD)

        resumed = self.update(
            0.22 + self.config.degraded_hold_sec + 0.01,
            fresh=False,
            navigation_active=False,
        )
        self.assertEqual(resumed.state, STATE_NORMAL)
        self.assertTrue(resumed.resume_move_base)
        self.assertFalse(resumed.authority_requested)
        self.assertEqual(
            resumed.reason,
            "competition_resume_after:critical_input_timeout",
        )

    def test_finish_priority_uses_stopped_handoff_after_ack_timeout(self):
        self.config.competition_finish_priority = True
        self.config.takeover_ack_timeout_sec = 0.30
        self.machine = PassageStateMachine(self.config)
        self.update(0.0)
        self.update(0.21)
        waiting = self.update(0.40, navigation_active=True)
        self.assertEqual(waiting.state, STATE_TAKEOVER_BRAKE)
        stopped = self.update(0.52, navigation_active=True)
        self.assertEqual(stopped.state, STATE_TAKEOVER_BRAKE)
        active = self.update(0.63, navigation_active=True)
        self.assertEqual(active.state, STATE_SINGLE_AVOID)

    def test_finish_priority_releases_special_mode_without_progress(self):
        self.config.competition_finish_priority = True
        self.config.special_no_progress_timeout_sec = 0.50
        self.machine = PassageStateMachine(self.config)
        self.update(0.0)
        self.update(0.21)
        self.update(0.22, navigation_active=False)
        self.update(0.33, navigation_active=False)
        self.update(0.36, x=0.40, navigation_active=False)
        resumed = self.update(0.87, x=0.40, navigation_active=False)
        self.assertEqual(resumed.state, STATE_NORMAL)
        self.assertTrue(resumed.resume_move_base)
        self.assertEqual(
            resumed.reason,
            "competition_resume_after:special_no_progress_timeout",
        )

    def test_finish_priority_does_not_hold_on_blocked_scene(self):
        self.config.competition_finish_priority = True
        self.machine = PassageStateMachine(self.config)
        tracks = [
            DiscTrack(11, (1.0, -0.210)),
            DiscTrack(12, (1.0, 0.210)),
        ]
        self.update(0.0, x=0.45, tracks=tracks)
        continued = self.update(0.21, x=0.45, tracks=tracks)
        self.assertEqual(continued.state, STATE_NORMAL)
        self.assertFalse(continued.authority_requested)
        self.assertFalse(continued.cancel_move_base)
        self.assertTrue(
            continued.reason.startswith(
                "competition_continue_blocked_scene:"
            )
        )

    def test_gate_requires_stop_before_alignment(self):
        tracks = [
            DiscTrack(11, (1.0, -0.231)),
            DiscTrack(12, (1.0, 0.231)),
        ]
        self.update(0.0, x=0.45, tracks=tracks)
        takeover = self.update(0.21, x=0.45, tracks=tracks)
        self.assertEqual(takeover.state, STATE_TAKEOVER_BRAKE)
        self.assertTrue(takeover.cancel_move_base)

        self.update(
            0.22,
            x=0.50,
            tracks=tracks,
            navigation_active=False,
        )
        alignment = self.update(
            0.33,
            x=0.50,
            tracks=tracks,
            navigation_active=False,
        )
        self.assertEqual(alignment.state, STATE_GATE_ALIGN)

    def test_geometry_jump_restarts_confirmation(self):
        self.update(0.0)
        shifted = [DiscTrack(1, (1.04, 0.10))]
        decision = self.update(0.19, tracks=shifted)
        self.assertFalse(decision.cancel_move_base)
        self.assertAlmostEqual(decision.scene_stable_sec, 0.0)

    def test_blocked_candidate_cancels_dwa_then_replans_in_hold(self):
        tracks = [
            DiscTrack(11, (1.0, -0.210)),
            DiscTrack(12, (1.0, 0.210)),
        ]
        self.update(0.0, x=0.45, tracks=tracks)
        failed = self.update(0.21, x=0.45, tracks=tracks)
        self.assertEqual(failed.state, STATE_RECOVERABLE_HOLD)
        self.assertTrue(failed.cancel_move_base)
        self.assertTrue(failed.authority_requested)
        self.assertTrue(failed.reason.startswith("safe_hold_no_local_plan:"))
        self.assertEqual(
            {track.memory_id for track in self.machine.locked_obstacles},
            {11, 12},
        )
        changed = [
            DiscTrack(11, (1.0, -0.231)),
            DiscTrack(12, (1.0, 0.231)),
        ]
        confirming = self.update(
            1.0,
            x=0.45,
            tracks=changed,
            navigation_active=False,
        )
        self.assertEqual(confirming.state, STATE_RECOVERABLE_HOLD)
        self.assertEqual(
            confirming.reason, "failed_scene_recovery_confirming"
        )
        takeover = self.update(
            1.0 + self.config.scene_confirm_sec + 0.01,
            x=0.45,
            tracks=changed,
            navigation_active=False,
        )
        self.assertEqual(takeover.state, STATE_TAKEOVER_BRAKE)
        self.assertTrue(takeover.authority_requested)
        self.assertFalse(takeover.resume_move_base)
        self.assertEqual(
            takeover.reason,
            "safe_local_plan_reacquired",
        )

    def test_formal_free_interval_failure_never_returns_to_blind_dwa(self):
        self.config.competition_two_disc_mode = False
        self.config.enable_gate_special_mode = False
        self.config.claim_all_road_discs = True
        self.config.force_between_disc_pair = False
        self.config.competition_finish_priority = False
        self.machine = PassageStateMachine(self.config)
        tracks = [
            DiscTrack(11, (1.0, -0.210)),
            DiscTrack(12, (1.0, 0.210)),
        ]
        first = self.update(0.0, x=0.45, tracks=tracks)
        self.assertEqual(first.assessment.scene, SCENE_BLOCKED)
        self.assertEqual(
            first.assessment.reason,
            "multi_obstacle_no_free_interval",
        )
        hold = self.update(0.21, x=0.45, tracks=tracks)
        self.assertEqual(hold.state, STATE_RECOVERABLE_HOLD)
        self.assertTrue(hold.authority_requested)
        self.assertTrue(hold.cancel_move_base)
        self.assertEqual(hold.command_body, (0.0, 0.0, 0.0))
        self.assertEqual(
            hold.reason,
            "safe_hold_no_local_plan:multi_obstacle_no_free_interval",
        )

    def test_confirmed_clear_recoverable_hold_requests_goal_resume(self):
        tracks = [
            DiscTrack(11, (1.0, -0.210)),
            DiscTrack(12, (1.0, 0.210)),
        ]
        changed = [
            DiscTrack(11, (1.0, -0.70)),
            DiscTrack(12, (1.0, 0.70)),
        ]
        self.update(0.0, x=0.45, tracks=tracks)
        self.update(0.21, x=0.45, tracks=tracks)
        self.update(
            1.0,
            x=0.45,
            tracks=changed,
            navigation_active=False,
        )
        resumed = self.update(
            1.0 + self.config.scene_release_sec + 0.01,
            x=0.45,
            tracks=changed,
            navigation_active=False,
        )
        self.assertEqual(resumed.state, STATE_NORMAL)
        self.assertTrue(resumed.resume_move_base)
        self.assertEqual(
            resumed.reason, "recoverable_hold_cleared_resuming_goal"
        )

    def test_missing_tracks_do_not_auto_clear_scene_failure(self):
        tracks = [
            DiscTrack(11, (1.0, -0.210)),
            DiscTrack(12, (1.0, 0.210)),
        ]
        self.update(0.0, x=0.45, tracks=tracks)
        self.update(0.21, x=0.45, tracks=tracks)
        still_failed = self.update(
            2.0,
            tracks=[],
            navigation_active=False,
        )
        self.assertEqual(still_failed.state, STATE_RECOVERABLE_HOLD)
        self.assertEqual(still_failed.reason, "gate_too_narrow")

    def test_runtime_failure_still_requires_explicit_reset(self):
        self.machine.fail("direct_disc_clearance")
        still_failed = self.update(
            1.0,
            tracks=[],
            navigation_active=False,
        )
        self.assertEqual(still_failed.state, STATE_FAILED_STOP)
        self.machine.reset()
        normal = self.update(
            1.1,
            tracks=[],
            navigation_active=False,
        )
        self.assertEqual(normal.state, STATE_NORMAL)

    def test_competition_pair_is_frozen_after_takeover(self):
        self.config.competition_two_disc_mode = True
        self.config.competition_finish_priority = True
        self.config.vehicle_half_length_m = 0.170
        self.config.vehicle_half_width_m = 0.165
        self.config.minimum_predicted_gate_clearance_m = 0.010
        self.machine = PassageStateMachine(self.config)
        tracks = [
            DiscTrack(11, (1.0, -0.25)),
            DiscTrack(12, (1.0, 0.25)),
        ]
        self.update(0.0, x=0.45, tracks=tracks)
        takeover = self.update(0.21, x=0.45, tracks=tracks)
        self.assertEqual(takeover.state, STATE_TAKEOVER_BRAKE)
        locked = tuple(self.machine.locked_obstacles)

        self.update(0.22, tracks=[], navigation_active=False)
        alignment = self.update(
            0.33, x=0.50, tracks=[], navigation_active=False
        )
        self.assertEqual(alignment.state, STATE_GATE_ALIGN)
        self.assertEqual(tuple(self.machine.locked_obstacles), locked)

    def test_competition_gate_uses_fresh_visual_pair_after_takeover(self):
        self.config.competition_two_disc_mode = True
        self.config.competition_finish_priority = True
        self.config.vehicle_half_length_m = 0.170
        self.config.vehicle_half_width_m = 0.165
        self.config.minimum_predicted_gate_clearance_m = 0.010
        self.machine = PassageStateMachine(self.config)
        initial = [
            DiscTrack(11, (1.0, -0.25)),
            DiscTrack(12, (1.0, 0.25)),
        ]
        self.update(0.0, x=0.45, tracks=initial)
        self.update(0.21, x=0.45, tracks=initial)

        live = [
            DiscTrack(11, (1.08, -0.24), last_seen_age_sec=0.05),
            DiscTrack(12, (1.08, 0.26), last_seen_age_sec=0.05),
        ]
        self.update(
            0.22,
            x=0.50,
            tracks=initial,
            live_gate_tracks=live,
            navigation_active=False,
        )
        self.assertAlmostEqual(
            self.machine.locked_gate_frame.midpoint_xy[0], 1.08
        )
        self.assertAlmostEqual(
            self.machine.locked_gate_frame.midpoint_xy[1], 0.01
        )
        self.assertTrue(self.machine.gate_frame_live_updated)

    def test_competition_gate_rejects_stale_visual_pair(self):
        self.config.competition_two_disc_mode = True
        self.config.competition_finish_priority = True
        self.config.vehicle_half_length_m = 0.170
        self.config.vehicle_half_width_m = 0.165
        self.config.minimum_predicted_gate_clearance_m = 0.010
        self.machine = PassageStateMachine(self.config)
        initial = [
            DiscTrack(11, (1.0, -0.25)),
            DiscTrack(12, (1.0, 0.25)),
        ]
        self.update(0.0, x=0.45, tracks=initial)
        self.update(0.21, x=0.45, tracks=initial)

        stale = [
            DiscTrack(11, (1.08, -0.24), last_seen_age_sec=0.61),
            DiscTrack(12, (1.08, 0.26), last_seen_age_sec=0.61),
        ]
        self.update(
            0.22,
            x=0.50,
            tracks=initial,
            live_gate_tracks=stale,
            navigation_active=False,
        )
        self.assertEqual(
            self.machine.locked_gate_frame.midpoint_xy,
            (1.0, 0.0),
        )
        self.assertFalse(self.machine.gate_frame_live_updated)

    def test_40cm_gate_pauses_forward_motion_to_recenter(self):
        self.config.competition_two_disc_mode = True
        self.config.vehicle_half_length_m = 0.170
        self.config.vehicle_half_width_m = 0.165
        self.config.minimum_predicted_gate_clearance_m = 0.020
        self.config.gate_align_lateral_tolerance_m = 0.005
        tracks = [
            # 50 cm centre distance minus two 5 cm radii = 40 cm clear gap.
            DiscTrack(11, (1.0, -0.25)),
            DiscTrack(12, (1.0, 0.25)),
        ]
        assessment = assess_scene(
            tracks,
            self.path,
            RobotPose(0.0, 0.0, 0.0),
            self.boundary,
            self.config,
        )
        self.assertEqual(assessment.scene, SCENE_GATE)

        self.machine = PassageStateMachine(self.config)
        self.machine.locked_gate_frame = assessment.gate_frame
        command, errors, diagnostics = self.machine._gate_control(
            RobotPose(1.0, 0.006, 0.0),
            now_sec=1.0,
            forward_speed=0.030,
            align=False,
        )
        self.assertGreater(abs(errors.lateral_m), 0.005)
        self.assertGreaterEqual(diagnostics.predicted_clearance_m, 0.020)
        self.assertAlmostEqual(command[0], 0.0, places=9)
        self.assertLess(command[1], 0.0)

    def test_competition_pair_confirmation_stays_inside_nav_state(self):
        self.config.competition_two_disc_mode = True
        self.config.competition_finish_priority = True
        self.config.gate_takeover_distance_m = 0.70
        self.config.minimum_predicted_gate_clearance_m = 0.010
        self.machine = PassageStateMachine(self.config)
        tracks = [
            DiscTrack(11, (1.0, -0.25)),
            DiscTrack(12, (1.0, 0.25)),
        ]
        self.update(0.0, tracks=tracks)
        armed = self.update(0.21, tracks=tracks)
        self.assertEqual(armed.state, STATE_NORMAL)
        self.assertEqual(armed.reason, "special_mode_armed")
        self.assertFalse(armed.authority_requested)

    def test_competition_approach_uses_stage_timeout_not_translation_watchdog(self):
        self.config.competition_two_disc_mode = True
        self.config.competition_finish_priority = True
        self.config.vehicle_half_length_m = 0.170
        self.config.vehicle_half_width_m = 0.165
        self.config.approach_align_timeout_sec = 1.0
        self.config.special_no_progress_timeout_sec = 1.20
        self.config.minimum_predicted_gate_clearance_m = 0.010
        self.machine = PassageStateMachine(self.config)
        tracks = [
            DiscTrack(11, (1.0, -0.25)),
            DiscTrack(12, (1.0, 0.25)),
        ]
        self.update(0.0, x=0.45, tracks=tracks)
        self.update(0.21, x=0.45, tracks=tracks)
        self.update(0.22, tracks=[], navigation_active=False)
        self.update(0.33, x=0.30, tracks=[], navigation_active=False)
        still_aligning = self.update(
            0.70, x=0.30, tracks=[], navigation_active=False
        )
        self.assertEqual(still_aligning.state, STATE_GATE_ALIGN)
        timed_out = self.update(
            1.34, x=0.30, tracks=[], navigation_active=False
        )
        self.assertEqual(timed_out.state, STATE_NORMAL)
        self.assertTrue(timed_out.resume_move_base)
        self.assertEqual(
            timed_out.reason,
            "competition_resume_after:approach_align_timeout",
        )

    def test_competition_alignment_counts_yaw_improvement_as_progress(self):
        self.config.competition_two_disc_mode = True
        self.config.competition_finish_priority = True
        self.config.vehicle_half_length_m = 0.170
        self.config.vehicle_half_width_m = 0.165
        self.config.minimum_predicted_gate_clearance_m = 0.010
        self.config.special_no_progress_timeout_sec = 0.50
        self.machine = PassageStateMachine(self.config)
        tracks = [
            DiscTrack(11, (1.0, -0.25)),
            DiscTrack(12, (1.0, 0.25)),
        ]
        self.update(0.0, x=0.45, tracks=tracks)
        self.update(0.21, x=0.45, tracks=tracks)
        self.update(0.22, x=0.50, tracks=[], navigation_active=False)
        self.update(0.33, x=0.50, tracks=[], navigation_active=False)
        self.update(
            0.40,
            x=0.50,
            yaw=math.radians(10.0),
            tracks=[],
            navigation_active=False,
        )
        improving = self.update(
            0.80,
            x=0.50,
            yaw=math.radians(8.0),
            tracks=[],
            navigation_active=False,
        )
        self.assertEqual(improving.state, STATE_GATE_ALIGN)
        still_improving = self.update(
            1.20,
            x=0.50,
            yaw=math.radians(6.0),
            tracks=[],
            navigation_active=False,
        )
        self.assertEqual(still_improving.state, STATE_GATE_ALIGN)

    def test_competition_success_consumes_the_only_pair(self):
        self.config.competition_two_disc_mode = True
        self.machine = PassageStateMachine(self.config)
        self.machine.state = STATE_EXIT_RESUME
        self.machine.scene = SCENE_GATE
        self.machine.exit_since_sec = 1.0
        completed = self.update(
            1.0 + self.config.exit_stop_hold_sec + 0.01,
            tracks=[],
            navigation_active=False,
        )
        self.assertTrue(completed.resume_move_base)
        self.assertTrue(self.machine.pair_consumed)
        ignored = self.update(
            2.0,
            tracks=[
                DiscTrack(11, (1.0, -0.25)),
                DiscTrack(12, (1.0, 0.25)),
            ],
        )
        self.assertEqual(ignored.state, STATE_NORMAL)
        self.assertEqual(ignored.reason, "competition_pair_consumed")

    def test_competition_failure_has_cooldown_and_two_attempt_budget(self):
        self.config.competition_two_disc_mode = True
        self.config.competition_finish_priority = True
        self.config.special_retry_cooldown_sec = 2.0
        self.config.special_max_attempts = 2
        self.machine = PassageStateMachine(self.config)
        self.machine.special_attempt_count = 1
        first = self.machine._competition_fallback(5.0, "test_failure")
        self.assertTrue(first.resume_move_base)
        self.assertAlmostEqual(
            self.machine.special_retry_not_before_sec, 7.0
        )
        self.assertFalse(self.machine.pair_disabled)

        self.machine.special_attempt_count = 2
        self.machine._competition_fallback(8.0, "test_failure")
        self.assertTrue(self.machine.pair_disabled)


if __name__ == "__main__":
    unittest.main()
