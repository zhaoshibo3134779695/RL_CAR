#!/usr/bin/env python3
"""Core safety tests. Kept ROS-independent for ARM64 and desktop CI."""

import copy
import math
import os
import unittest

import cv2
import numpy as np

from yellow_track_avoidance.bev import BirdEyeTransformer, CameraUndistorter
from yellow_track_avoidance.config import calibration_ready, load_config
from yellow_track_avoidance.disc_memory import StableDiscMemory
from yellow_track_avoidance.model import BoundaryResult, DiskDetection, MotionState, PlanningResult, SafeCorridor, TrackedDisk
from yellow_track_avoidance.map_navigation import (
    build_reference_path,
    fuse_candidate_with_reference,
    local_to_map,
    load_yaml,
    validate_route,
)
from yellow_track_avoidance.navigation_control import (
    compute_navigation_command,
    lookahead_target,
)
from yellow_track_avoidance.pipeline import AvoidancePipeline
from yellow_track_avoidance.planning import LocalPathPlanner, build_safe_corridor, path_hits_circle
from yellow_track_avoidance.profile_boundary import NormalProfileBoundaryDetector
from yellow_track_avoidance.state_machine import AvoidanceStateMachine
from yellow_track_avoidance.vision import (
    BoundaryDetector,
    ColorSegmenter,
    DiskDetector,
    circularity,
    configured_detection_polygon,
    filter_yellow_orientation,
    merge_red_hsv,
)


PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def calibrated_config():
    config = load_config(os.path.join(PACKAGE_DIR, "config", "default.yaml"))
    config = copy.deepcopy(config)
    # Synthetic unit-test images use the full frame; live commissioning uses
    # the lower 40 percent and must not silently alter these fixtures.
    config["camera"].update({
        "roi_y_start_ratio": 0.0,
        "roi_y_end_ratio": 1.0,
        "yellow_roi_y_start_ratio": 0.0,
        "yellow_roi_y_end_ratio": 1.0,
    })
    config["bev"].update({
        "calibrated": True,
        "source_points": [[0.0, 799.0], [639.0, 799.0], [639.0, 0.0], [0.0, 0.0]],
        "output_width_px": 640,
        "output_height_px": 800,
        "real_width_m": 0.91,
        "real_length_m": 2.50,
    })
    config["camera_intrinsics"].update({
        "calibrated": True,
        "image_width": 640,
        "image_height": 480,
        "camera_matrix": [480.0, 0.0, 319.5, 0.0, 480.0, 239.5, 0.0, 0.0, 1.0],
        "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
        "rectified_camera_matrix": [480.0, 0.0, 319.5, 0.0, 480.0, 239.5, 0.0, 0.0, 1.0],
    })
    config["physical"]["robot_width_m"] = 0.20
    config["safety"]["boundary_margin_m"] = 0.03
    config["safety"]["obstacle_margin_m"] = 0.03
    config["safety"]["stop_is_latched"] = False
    config["safety"]["stop_recovery_frames"] = 1
    config["disk_detection"]["confirmation_frames"] = 1
    config["planner"]["maximum_curvature"] = 5.0
    config["control"]["enable_motion"] = False
    return config


def straight_boundary(config):
    mask = np.zeros((800, 640), dtype=np.uint8)
    cv2.line(mask, (2, 799), (2, 0), 255, 7)
    cv2.line(mask, (637, 799), (637, 0), 255, 7)
    return BoundaryDetector(config).detect(mask)


class CoreTests(unittest.TestCase):
    def test_00_current_car_profile_is_usb_and_motion_locked(self):
        config = load_config(
            os.path.join(PACKAGE_DIR, "config", "current_car.yaml")
        )
        self.assertEqual(config["ros"]["image_topic"], "/usb_cam/image_raw")
        self.assertEqual(config["camera"]["expected_encoding"], ["rgb8", "bgr8"])
        self.assertFalse(config["camera_intrinsics"]["calibrated"])
        self.assertFalse(config["bev"]["calibrated"])
        self.assertFalse(config["control"]["enable_motion"])
        self.assertFalse(calibration_ready(config))
        self.assertEqual(config["physical"]["robot_length_m"], 0.34)
        self.assertEqual(config["physical"]["robot_width_m"], 0.33)

    def test_00b_current_car_polygon_matches_accepted_roi(self):
        config = load_config(
            os.path.join(PACKAGE_DIR, "config", "current_car.yaml")
        )
        masks = ColorSegmenter(config).masks(
            np.zeros((480, 640, 3), dtype=np.uint8)
        )
        roi = masks["roi"]
        self.assertEqual(int(roi[400, 80]), 255)
        self.assertEqual(int(roi[300, 80]), 0)
        self.assertEqual(int(roi[300, 240]), 255)
        self.assertEqual(int(roi[300, 500]), 255)
        self.assertEqual(int(roi[200, 500]), 0)
        self.assertEqual(int(masks["yellow_roi"][300, 240]), 255)
        self.assertEqual(int(masks["yellow_roi"][300, 500]), 0)
        self.assertEqual(config["boundary"]["detection_mode"], "left_only")

    def test_00c_calibrated_profile_uses_left_yellow_full_disc_roi(self):
        config = load_config(
            os.path.join(
                PACKAGE_DIR,
                "config",
                "current_car_calibrated_640x480.yaml",
            )
        )
        masks = ColorSegmenter(config).masks(
            np.zeros((480, 640, 3), dtype=np.uint8)
        )
        self.assertEqual(int(masks["roi"][300, 500]), 255)
        self.assertEqual(int(masks["yellow_roi"][300, 200]), 255)
        self.assertEqual(int(masks["yellow_roi"][300, 500]), 0)
        self.assertEqual(config["boundary"]["detection_mode"], "left_only")

    def test_00ca_formal_corridor_width_uses_inner_edge_measurement(self):
        config = load_config(
            os.path.join(
                PACKAGE_DIR,
                "config",
                "current_car_recalibrated_640x480_20260809.yaml",
            )
        )
        clear_width = float(config["physical"]["track_gray_width_m"])
        line_width = float(config["physical"]["yellow_line_width_m"])
        self.assertAlmostEqual(clear_width, 0.91, places=9)
        self.assertAlmostEqual(clear_width + line_width, 0.94, places=9)

    def test_00d_left_only_fallback_never_reports_right_line(self):
        config = calibrated_config()
        config["boundary"]["detection_mode"] = "left_only"
        mask = np.zeros((800, 640), dtype=np.uint8)
        cv2.line(mask, (100, 799), (100, 0), 255, 9)
        cv2.line(mask, (540, 799), (540, 0), 255, 9)
        boundary = BoundaryDetector(config).detect(mask)
        self.assertTrue(boundary.left_detected)
        self.assertFalse(boundary.right_detected)
        self.assertTrue(np.all(np.isnan(boundary.right_px)))
        self.assertEqual(boundary.both_missing_frames, 0)
        self.assertGreater(boundary.quality, 0.5)

    def test_00e_left_only_profile_skips_right_detector(self):
        config = load_config(
            os.path.join(
                PACKAGE_DIR,
                "config",
                "current_car_calibrated_640x480.yaml",
            )
        )
        detector = NormalProfileBoundaryDetector(config)
        calls = []

        def fake_detect(side, _confidence, _opponent, _roi, rows):
            calls.append(side)
            if side != "left":
                self.fail("right profile must not run in left-only mode")
            result = detector._missing_side(rows)
            result.detected = True
            result.values_px[:] = 160.0
            result.quality = 0.82
            result.valid_sections = 20
            result.support_ratio = 0.8
            result.coverage_ratio = 0.8
            result.median_confidence = 0.8
            result.median_prominence = 8.0
            result.fit_rms_px = 1.0
            result.longest_missing_sections = 0
            result.observed_min_row_px = 260
            result.observed_max_row_px = 470
            result.slope_dx_dy = -0.2
            return result

        detector._detect_side = fake_detect
        zeros = np.zeros((480, 640), dtype=np.float32)
        boundary = detector.detect(zeros, zeros, zeros)
        self.assertEqual(calls, ["left"])
        self.assertTrue(boundary.left_detected)
        self.assertFalse(boundary.right_detected)
        self.assertAlmostEqual(boundary.quality, 0.82)
        self.assertEqual(boundary.both_missing_frames, 0)

    def test_01_red_hsv_ranges_are_merged(self):
        config = calibrated_config()
        hsv = np.zeros((1, 3, 3), dtype=np.uint8)
        hsv[0, 0] = (2, 255, 255)
        hsv[0, 1] = (175, 255, 255)
        hsv[0, 2] = (90, 255, 255)
        mask = merge_red_hsv(hsv, config)
        self.assertEqual(mask[0].tolist(), [255, 255, 0])

    def test_02_circularity_formula(self):
        radius = 10.0
        self.assertAlmostEqual(circularity(math.pi * radius * radius, 2.0 * math.pi * radius), 1.0)
        self.assertEqual(circularity(10.0, 0.0), 0.0)

    def test_03_red_ring_requires_white_center(self):
        config = calibrated_config()
        image = np.zeros((300, 300, 3), dtype=np.uint8)
        image[:] = (80, 80, 80)
        cv2.circle(image, (150, 150), 55, (0, 0, 255), -1)
        cv2.circle(image, (150, 150), 24, (255, 255, 255), -1)
        masks = ColorSegmenter(config).masks(image)
        detector = DiskDetector(config)
        metric = lambda points: np.asarray(points, dtype=np.float64) * 0.001
        rings = detector.detect(masks["red"], masks["white"], 1.0, 0.001, 0.001, metric)
        self.assertGreaterEqual(len(rings), 1)

        solid = np.zeros_like(image)
        solid[:] = (80, 80, 80)
        cv2.circle(solid, (150, 150), 55, (0, 0, 255), -1)
        solid_masks = ColorSegmenter(config).masks(solid)
        rejected = detector.detect(solid_masks["red"], solid_masks["white"], 1.0, 0.001, 0.001, metric)
        self.assertEqual(len(rejected), 0)

    def test_04_yellow_left_and_right_boundaries(self):
        config = calibrated_config()
        boundary = straight_boundary(config)
        self.assertTrue(boundary.left_detected)
        self.assertTrue(boundary.right_detected)
        self.assertTrue(np.all(boundary.left_px < boundary.right_px))

    def test_05_boundary_safe_corridor_accounts_for_robot_and_line(self):
        config = calibrated_config()
        transformer = BirdEyeTransformer(config)
        corridor = build_safe_corridor(straight_boundary(config), transformer, config)
        expected = 0.91 - 0.03 - 0.20 - 2.0 * 0.03
        self.assertTrue(corridor.valid)
        self.assertAlmostEqual(corridor.minimum_width_m, expected, delta=0.02)

    def test_06_path_circle_collision(self):
        points = np.column_stack([np.linspace(0.0, 2.0, 101), np.zeros(101)])
        self.assertTrue(path_hits_circle(points, (1.0, 0.0), 0.1))
        self.assertFalse(path_hits_circle(points, (1.0, 0.5), 0.1))

    def test_06b_path_collision_between_samples_is_not_missed(self):
        points = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
        self.assertTrue(path_hits_circle(points, (0.5, 0.05), 0.06))
        self.assertFalse(path_hits_circle(points, (0.5, 0.07), 0.06))

    def test_07_insufficient_left_and_right_space_stops(self):
        config = calibrated_config()
        machine = AvoidanceStateMachine(config)
        planning = PlanningResult([], None, False, -0.02, -0.01, "no_collision_free_candidate")
        decision = machine.update(1.0, True, True, 0, planning, [])
        self.assertEqual(decision.state, MotionState.STOP)
        self.assertEqual(decision.stop_reason, "no_collision_free_candidate")

    def test_08_two_disks_are_both_collision_checked(self):
        config = calibrated_config()
        x = np.linspace(0.0, 2.5, 80)
        corridor = SafeCorridor(
            forward_m=x,
            safe_left_m=np.full_like(x, 0.31),
            safe_right_m=np.full_like(x, -0.31),
            center_m=np.zeros_like(x),
            valid=True,
            minimum_width_m=0.62,
        )
        disks = [
            TrackedDisk(1, (0.95, 0.0), (320, 500), 0.10, 0.9, 1.0, 3),
            TrackedDisk(2, (1.45, -0.06), (340, 350), 0.10, 0.9, 1.0, 3),
        ]
        planner = LocalPathPlanner(config)
        result = planner.plan(corridor, disks)
        self.assertTrue(result.safe)
        expanded = planner.expanded_radius()
        for disk in disks:
            self.assertFalse(path_hits_circle(result.selected.points_m, disk.center_m, expanded))

    def test_08b_offline_curvature_limit_accepts_commandable_left_bypass(self):
        config = calibrated_config()
        config["physical"]["robot_width_m"] = 0.30
        config["safety"]["boundary_margin_m"] = 0.08
        config["safety"]["obstacle_margin_m"] = 0.10
        config["planner"]["maximum_curvature"] = 3.5
        x = np.linspace(0.0, 1.08, 40)
        corridor = SafeCorridor(
            forward_m=x,
            safe_left_m=np.full_like(x, 0.21),
            safe_right_m=np.full_like(x, -0.21),
            center_m=np.zeros_like(x),
            valid=True,
            minimum_width_m=0.42,
        )
        disk = TrackedDisk(1, (0.36, -0.275), (490, 610), 0.10, 0.95, 1.0, 3)
        result = LocalPathPlanner(config).plan(corridor, [disk])
        self.assertTrue(result.safe)
        self.assertEqual(result.selected.name, "left_0.16")
        self.assertLessEqual(result.selected.max_curvature, 3.5)

    def test_09_camera_timeout_forces_stop(self):
        pipeline = AvoidancePipeline(calibrated_config())
        result = pipeline.camera_timeout(10.0)
        self.assertEqual(result.state, MotionState.STOP)
        self.assertEqual(result.stop_reason, "camera_timeout")
        self.assertEqual(result.command.linear_x, 0.0)

    def test_10_sustained_double_boundary_loss_forces_stop(self):
        config = calibrated_config()
        machine = AvoidanceStateMachine(config)
        safe_path = np.column_stack([np.linspace(0.0, 1.0, 10), np.zeros(10)])
        from yellow_track_avoidance.model import PathCandidate
        candidate = PathCandidate("center", safe_path, 0.0, 0, True, 0.0)
        planning = PlanningResult([candidate], candidate, True, 0.2, 0.2)
        loss = int(config["boundary"]["both_side_stop_frames"])
        decision = machine.update(1.0, True, False, loss, planning, [])
        self.assertEqual(decision.state, MotionState.STOP)
        self.assertEqual(decision.stop_reason, "yellow_boundaries_lost")

    def test_11_intrinsics_are_part_of_the_metric_safety_gate(self):
        config = calibrated_config()
        config["camera_intrinsics"]["calibrated"] = False
        self.assertFalse(calibration_ready(config))

    def test_12_identity_intrinsic_remap_preserves_resolution(self):
        config = calibrated_config()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[120:240, 220:420] = (10, 120, 240)
        rectified = CameraUndistorter(config).undistort(image)
        self.assertEqual(rectified.shape, image.shape)
        self.assertTrue(np.array_equal(rectified, image))

    def test_12b_point_diagnostics_round_trip_full_image_coordinates(self):
        undistorter = CameraUndistorter(calibrated_config())
        distorted = np.asarray([[50.0, 80.0], [620.0, 350.0]])
        rectified = undistorter.undistort_points(distorted)
        recovered = undistorter.distort_points(rectified)
        self.assertTrue(np.allclose(recovered, distorted, atol=1e-6))

    def test_13_foreshortened_elliptical_disk_is_detected(self):
        config = calibrated_config()
        image = np.full((300, 400, 3), 120, dtype=np.uint8)
        cv2.ellipse(image, (210, 180), (62, 18), 8, 0, 360, (0, 0, 220), -1)
        cv2.ellipse(image, (210, 180), (29, 8), 8, 0, 360, (235, 235, 235), -1)
        masks = ColorSegmenter(config).masks(image)
        metric = lambda points: np.asarray(points, dtype=np.float64) * 0.001
        detector = DiskDetector(config)
        disks = detector.detect(
            masks["red"], masks["white"], 1.0, 0.001, 0.001, metric,
            image_bgr=image,
        )
        self.assertEqual(len(disks), 1)
        self.assertLess(min(disks[0].ellipse[1]) / max(disks[0].ellipse[1]), 0.40)
        self.assertGreater(disks[0].red_area_px, 0)
        self.assertGreater(disks[0].white_area_px, 0)
        self.assertLess(disks[0].red_white_center_distance_px, 2.0)
        self.assertFalse(disks[0].used_history)
        self.assertGreater(len(detector.last_candidates), 0)

    def test_13a_far_white_center_recovery_handles_four_pixel_disc(self):
        config = load_config(
            os.path.join(
                PACKAGE_DIR,
                "config",
                "current_car_recalibrated_640x480_20260809.yaml",
            )
        )
        image = np.full((300, 400, 3), 120, dtype=np.uint8)
        # A far disc is almost edge-on in the camera: the red annulus is only
        # two pixels high and therefore cannot satisfy the full-ring gate.
        cv2.ellipse(image, (210, 180), (11, 2), 0, 0, 360, (0, 0, 220), -1)
        cv2.ellipse(image, (210, 180), (7, 2), 0, 0, 360, (235, 235, 235), -1)
        masks = ColorSegmenter(config).masks(image)
        metric = lambda points: np.asarray(points, dtype=np.float64) * 0.001

        without_recovery = copy.deepcopy(config)
        without_recovery["disk_detection"]["far_white_center_recovery_enabled"] = False
        self.assertEqual(
            DiskDetector(without_recovery).detect(
                masks["red"], masks["white"], 1.0, 0.001, 0.001, metric,
                image_bgr=image,
            ),
            [],
        )

        detector = DiskDetector(config)
        recovered = detector.detect(
            masks["red"], masks["white"], 1.0, 0.001, 0.001, metric,
            image_bgr=image,
        )
        self.assertEqual(len(recovered), 1)
        accepted = [item for item in detector.last_candidates if item["accepted"]]
        self.assertEqual(len(accepted), 1)
        self.assertTrue(accepted[0]["far_white_center_recovery"])
        self.assertLess(
            accepted[0]["red_annulus_ratio"],
            float(config["disk_detection"]["minimum_red_annulus_ratio"]),
        )

    def test_13aa_far_recovery_rejects_separate_red_and_white_features(self):
        config = load_config(
            os.path.join(
                PACKAGE_DIR,
                "config",
                "current_car_recalibrated_640x480_20260809.yaml",
            )
        )
        image = np.full((300, 400, 3), 120, dtype=np.uint8)
        cv2.ellipse(image, (210, 180), (7, 2), 0, 0, 360, (235, 235, 235), -1)
        cv2.line(image, (180, 180), (194, 180), (0, 0, 220), 2)
        masks = ColorSegmenter(config).masks(image)
        metric = lambda points: np.asarray(points, dtype=np.float64) * 0.001
        detections = DiskDetector(config).detect(
            masks["red"], masks["white"], 1.0, 0.001, 0.001, metric,
            image_bgr=image,
        )
        self.assertEqual(detections, [])

    def test_13b_projected_metric_size_rejects_tiny_reflection(self):
        config = calibrated_config()
        config["disk_detection"]["minimum_projected_diameter_m"] = 0.055
        config["disk_detection"]["maximum_projected_diameter_m"] = 0.15
        pipeline = AvoidancePipeline(config)
        pipeline.disk_detector.last_candidates = [
            {"candidate_id": 7, "accepted": True, "reject_reason": ""}
        ]
        contour = np.asarray(
            [[315, 395], [325, 395], [328, 400], [325, 405], [315, 405]],
            dtype=np.int32,
        ).reshape(-1, 1, 2)
        reflection = DiskDetection(
            center_px=(320.0, 400.0),
            center_m=(0.0, 0.0),
            ellipse=((320.0, 400.0), (10.0, 10.0), 0.0),
            contour=contour,
            estimated_diameter_m=0.0,
            confidence=0.8,
            stamp_sec=1.0,
            candidate_id=7,
        )
        self.assertIsNone(pipeline._project_disk_to_bev(reflection))
        record = pipeline.disk_detector.last_candidates[0]
        self.assertFalse(record["accepted"])
        self.assertEqual(
            record["reject_reason"], "projected_diameter_too_small"
        )
        self.assertLess(
            record["projected_diameter_m"],
            config["disk_detection"]["minimum_projected_diameter_m"],
        )
        projection = record["metric_projection"]
        self.assertFalse(projection["roi_applied"])
        self.assertFalse(projection["resize_applied"])
        self.assertEqual(
            projection["detection_coordinate_system"],
            "undistorted_full_image_px",
        )
        self.assertEqual(
            projection["diameter_method"],
            "72_point_ellipse_same_homography_lateral_span",
        )
        self.assertEqual(projection["diameter_sample_count"], 72)

    def test_13c_visible_ring_ratio_corrects_only_reported_diameter(self):
        config = calibrated_config()
        config["disk_detection"][
            "visible_ring_to_physical_diameter_ratio"
        ] = 0.5
        config["disk_detection"]["minimum_projected_diameter_m"] = 0.0
        pipeline = AvoidancePipeline(config)
        pipeline.disk_detector.last_candidates = [
            {"candidate_id": 8, "accepted": True, "reject_reason": ""}
        ]
        contour = np.asarray(
            [[300, 390], [340, 390], [340, 410], [300, 410]],
            dtype=np.int32,
        ).reshape(-1, 1, 2)
        detection = DiskDetection(
            center_px=(320.0, 400.0),
            center_m=(0.0, 0.0),
            ellipse=((320.0, 400.0), (40.0, 20.0), 0.0),
            contour=contour,
            estimated_diameter_m=0.0,
            confidence=0.8,
            stamp_sec=1.0,
            candidate_id=8,
        )
        projected = pipeline._project_disk_to_bev(detection)
        self.assertIsNotNone(projected)
        diagnostics = pipeline.disk_detector.last_candidates[0][
            "metric_projection"
        ]
        self.assertEqual(
            diagnostics["diameter_method"],
            "72_point_visible_ring_lateral_span_with_physical_ratio",
        )
        self.assertAlmostEqual(
            projected.estimated_diameter_m,
            2.0 * diagnostics["estimated_visible_ring_diameter_m"],
        )
        self.assertAlmostEqual(
            diagnostics["estimated_physical_diameter_m"],
            projected.estimated_diameter_m,
        )

    def test_13d_geometric_mean_diameter_uses_both_ground_spans(self):
        config = calibrated_config()
        config["disk_detection"][
            "projected_diameter_mode"
        ] = "geometric_mean_ground_spans"
        config["disk_detection"][
            "visible_ring_to_physical_diameter_ratio"
        ] = 0.5
        config["disk_detection"]["minimum_projected_diameter_m"] = 0.0
        pipeline = AvoidancePipeline(config)
        pipeline.disk_detector.last_candidates = [
            {"candidate_id": 9, "accepted": True, "reject_reason": ""}
        ]
        contour = np.asarray(
            [[300, 390], [340, 390], [340, 410], [300, 410]],
            dtype=np.int32,
        ).reshape(-1, 1, 2)
        detection = DiskDetection(
            center_px=(320.0, 400.0),
            center_m=(0.0, 0.0),
            ellipse=((320.0, 400.0), (40.0, 20.0), 0.0),
            contour=contour,
            estimated_diameter_m=0.0,
            confidence=0.8,
            stamp_sec=1.0,
            candidate_id=9,
        )
        projected = pipeline._project_disk_to_bev(detection)
        self.assertIsNotNone(projected)
        diagnostics = pipeline.disk_detector.last_candidates[0][
            "metric_projection"
        ]
        self.assertEqual(
            diagnostics["diameter_method"],
            "72_point_visible_ring_geometric_mean_ground_spans_with_physical_ratio",
        )
        expected_visible = math.sqrt(
            diagnostics["estimated_visible_ring_forward_span_m"]
            * diagnostics["estimated_visible_ring_lateral_span_m"]
        )
        self.assertAlmostEqual(
            diagnostics["estimated_visible_ring_diameter_m"],
            expected_visible,
        )
        self.assertAlmostEqual(
            projected.estimated_diameter_m,
            2.0 * expected_visible,
        )

    def test_13e_current_camera_bev_includes_valid_close_ground_projection(self):
        config = load_config(
            os.path.join(
                PACKAGE_DIR,
                "config",
                "current_car_recalibrated_640x480_20260809.yaml",
            )
        )
        pipeline = AvoidancePipeline(config)
        pipeline.disk_detector.last_candidates = [
            {"candidate_id": 10, "accepted": True, "reject_reason": ""}
        ]
        contour = np.asarray(
            [[475, 330], [503, 330], [503, 414], [475, 414]],
            dtype=np.int32,
        ).reshape(-1, 1, 2)
        detection = DiskDetection(
            center_px=(489.0849, 372.0239),
            center_m=(0.0, 0.0),
            ellipse=((489.0849, 372.0239), (28.0295, 85.2621), 97.5318),
            contour=contour,
            estimated_diameter_m=0.0,
            confidence=0.97,
            stamp_sec=1.0,
            candidate_id=10,
        )
        projected = pipeline._project_disk_to_bev(detection)
        self.assertIsNotNone(projected)
        diagnostics = projected.projection_diagnostics
        self.assertTrue(diagnostics["inside_bev_output"])
        self.assertFalse(diagnostics["near_field_ground_extrapolated"])
        self.assertAlmostEqual(diagnostics["base_m"][0], 0.444, delta=0.015)
        self.assertAlmostEqual(projected.estimated_diameter_m, 0.098, delta=0.015)

        # Acceptance no longer depends on the legacy near-field exception:
        # the formal BEV canvas itself covers base-forward 0..2 m.
        config["disk_detection"]["allow_near_field_ground_extrapolation"] = False
        normal_pipeline = AvoidancePipeline(config)
        normal_pipeline.disk_detector.last_candidates = [
            {"candidate_id": 10, "accepted": True, "reject_reason": ""}
        ]
        self.assertIsNotNone(normal_pipeline._project_disk_to_bev(detection))

    def test_14_expanded_detection_includes_everything_below_side_seam(self):
        config = calibrated_config()
        config["bev"].update({
            "source_points": [
                [0.0, 312.0], [639.0, 312.0],
                [482.0, 259.0], [156.0, 263.0],
            ],
            "extend_to_image_bottom": True,
            "bottom_extension_output_fraction": 0.35,
        })
        transformer = BirdEyeTransformer(config)
        source = np.asarray([
            [0.0, 312.0], [639.0, 312.0],
            [0.0, 479.0], [639.0, 479.0],
        ])
        bev = transformer.original_to_bev(source)
        self.assertTrue(np.allclose(bev[:2, 1], transformer.destination_seam_y, atol=1.0))
        self.assertTrue(np.allclose(bev[2:, 1], 799.0, atol=1.0))
        restored = transformer.bev_to_original(bev)
        self.assertTrue(np.allclose(restored, source, atol=1.0))

    def test_14b_base_forward_output_bounds_preserve_calibrated_scale(self):
        config = calibrated_config()
        config["physical"]["local_forward_origin_offset_m"] = 0.50
        config["bev"].update({
            "real_length_m": 1.50,
            "output_base_forward_min_m": 0.0,
            "output_base_forward_max_m": 2.0,
        })
        transformer = BirdEyeTransformer(config)

        # Canvas bounds are expressed relative to base_link. The transformer's
        # metric API retains the existing local-ground convention because its
        # planning callers add the measured 0.50 m base offset.
        canvas_rows = np.asarray([[320.0, 799.0], [320.0, 0.0]])
        local = transformer.pixel_to_metric(canvas_rows)
        self.assertAlmostEqual(local[0, 0] + 0.50, 0.0, places=6)
        self.assertAlmostEqual(local[1, 0] + 0.50, 2.0, places=6)

        source = np.asarray(config["bev"]["source_points"], dtype=np.float64)
        projected = transformer.original_to_bev(source)
        source_local = transformer.pixel_to_metric(projected)
        self.assertTrue(np.allclose(source_local[:2, 0], 0.0, atol=1e-5))
        self.assertTrue(np.allclose(source_local[2:, 0], 1.50, atol=1e-5))
        restored = transformer.bev_to_original(projected)
        self.assertTrue(np.allclose(restored, source, atol=1e-3))

    def test_14c_lateral_origin_offset_is_signed_and_invertible(self):
        config = calibrated_config()
        config["physical"]["local_lateral_origin_offset_m"] = -0.0613
        transformer = BirdEyeTransformer(config)
        centre = np.asarray([[transformer.width_px * 0.5, 400.0]])
        metric = transformer.pixel_to_metric(centre)
        self.assertAlmostEqual(metric[0, 1], -0.0613, places=6)
        restored = transformer.metric_to_pixel(metric)
        self.assertTrue(np.allclose(restored, centre, atol=1e-6))

    def test_15_yellow_orientation_filter_rejects_horizontal_cross_line(self):
        config = calibrated_config()
        config["yellow_orientation_filter"] = {
            "enabled": True,
            "window_size_px": 25,
            "minimum_nonhorizontal_angle_deg": 10.0,
            "minimum_anisotropy": 0.20,
            "minimum_support_pixels": 18,
            "recovery_dilate_px": 3,
        }
        mask = np.zeros((240, 320), dtype=np.uint8)
        cv2.line(mask, (20, 55), (300, 55), 255, 7)
        cv2.line(mask, (75, 100), (75, 230), 255, 7)
        cv2.line(mask, (150, 225), (300, 165), 255, 7)
        filtered = filter_yellow_orientation(mask, config)

        horizontal_before = np.count_nonzero(mask[48:63, 120:280])
        horizontal_after = np.count_nonzero(filtered[48:63, 120:280])
        vertical_after = np.count_nonzero(filtered[105:225, 68:83])
        diagonal_after = np.count_nonzero(filtered[160:230, 140:310])
        self.assertLess(horizontal_after, 0.10 * horizontal_before)
        self.assertGreater(vertical_after, 600)
        self.assertGreater(diagonal_after, 600)

    def test_16_safe_corridor_does_not_extend_past_observed_yellow(self):
        config = calibrated_config()
        mask = np.zeros((800, 640), dtype=np.uint8)
        cv2.line(mask, (80, 520), (180, 220), 255, 7)
        cv2.line(mask, (560, 520), (460, 220), 255, 7)
        boundary = BoundaryDetector(config).detect(mask)
        observed = (
            (boundary.rows_px >= boundary.observed_min_row_px)
            & (boundary.rows_px <= boundary.observed_max_row_px)
        )
        corridor = build_safe_corridor(boundary, BirdEyeTransformer(config), config)
        self.assertGreater(boundary.observed_max_row_px, 0)
        self.assertLess(boundary.observed_max_row_px, 799)
        self.assertEqual(corridor.forward_m.size, int(np.count_nonzero(observed)))

    def test_17_raw_detection_uses_expanded_six_point_region(self):
        config = calibrated_config()
        config["camera"]["use_bev_polygon_for_detection"] = True
        config["bev"].update({
            "source_points": [
                [0.0, 312.0], [639.0, 312.0],
                [482.0, 259.0], [156.0, 263.0],
            ],
            "extend_to_image_bottom": True,
        })
        masks = ColorSegmenter(config).masks(
            np.full((480, 640, 3), 100, dtype=np.uint8)
        )
        self.assertEqual(int(masks["roi"][260, 100]), 0)
        self.assertEqual(int(masks["roi"][270, 320]), 255)
        self.assertEqual(int(masks["roi"][300, 10]), 0)
        self.assertEqual(int(masks["roi"][400, 10]), 255)
        self.assertTrue(np.array_equal(masks["roi"], masks["yellow_roi"]))

    def test_17b_raw_overlay_polygon_matches_live_detection_region(self):
        config = load_config(
            os.path.join(PACKAGE_DIR, "config", "calibrated.yaml")
        )
        pipeline = AvoidancePipeline(config)
        polygon = configured_detection_polygon(
            pipeline.segmenter.config, 640, 480
        )
        expected = np.asarray(
            [
                [78, 263],
                [560, 259],
                [639, 312],
                [639, 479],
                [0, 479],
                [0, 312],
            ],
            dtype=np.int32,
        )
        self.assertTrue(np.array_equal(polygon, expected))

    def test_18_nonfinite_disk_proposal_is_rejected(self):
        regions = DiskDetector._local_ellipse_regions(
            (480, 640),
            ((float("nan"), 200.0), (40.0, 20.0), 0.0),
            0.48, 1.08, 0.4,
        )
        self.assertIsNone(regions)

    def test_19_wrong_way_boundary_fit_is_rejected(self):
        config = calibrated_config()
        config["boundary"].update({
            "enforce_perspective_side_direction": True,
            "perspective_direction_slope_tolerance": 0.15,
            "use_previous_reference_center": False,
        })
        wrong = np.zeros((800, 640), dtype=np.uint8)
        cv2.line(wrong, (40, 220), (260, 700), 255, 7)
        cv2.line(wrong, (440, 700), (610, 220), 255, 7)
        boundary = BoundaryDetector(config).detect(wrong)
        self.assertFalse(boundary.left_detected)

    def test_20_perspective_row_assignment_ignores_outer_fragment(self):
        config = calibrated_config()
        config["boundary"].update({
            "row_assignment_mode": "perspective_sides",
            "perspective_center_exclusion_px": 70.0,
            "polynomial_degree": 1,
            "minimum_fit_points": 6,
            "enforce_perspective_side_direction": True,
            "perspective_direction_slope_tolerance": 0.15,
        })
        mask = np.zeros((800, 640), dtype=np.uint8)
        cv2.line(mask, (190, 200), (40, 700), 255, 7)
        cv2.line(mask, (450, 200), (600, 700), 255, 7)
        cv2.line(mask, (5, 650), (25, 700), 255, 7)
        boundary = BoundaryDetector(config).detect(mask)
        self.assertTrue(boundary.left_detected)
        self.assertTrue(boundary.right_detected)
        observed = (
            (boundary.rows_px >= boundary.observed_min_row_px)
            & (boundary.rows_px <= boundary.observed_max_row_px)
        )
        self.assertLess(np.polyfit(boundary.rows_px[observed], boundary.left_px[observed], 1)[0], 0.0)
        self.assertGreater(np.polyfit(boundary.rows_px[observed], boundary.right_px[observed], 1)[0], 0.0)

    def test_21_normal_profile_dp_prefers_side_lines_over_horizontal_line(self):
        config = load_config(os.path.join(PACKAGE_DIR, "config", "calibrated.yaml"))
        confidence = np.full((480, 640), 0.01, dtype=np.float32)
        opponent = np.zeros((480, 640), dtype=np.float32)
        roi = np.full((480, 640), 255, dtype=np.uint8)
        for first, second in (
            ((185, 268), (0, 340)),
            ((445, 260), (639, 330)),
        ):
            cv2.line(confidence, first, second, 0.92, 5)
            cv2.line(opponent, first, second, 55.0, 5)
        cv2.line(confidence, (0, 300), (639, 300), 0.98, 5)
        cv2.line(opponent, (0, 300), (639, 300), 55.0, 5)

        boundary = NormalProfileBoundaryDetector(config).detect(
            confidence, opponent, roi
        )
        self.assertTrue(boundary.left_detected)
        self.assertTrue(boundary.right_detected)
        paired = np.isfinite(boundary.left_px) & np.isfinite(boundary.right_px)
        self.assertGreater(np.count_nonzero(paired), 2)
        self.assertLess(
            np.polyfit(boundary.rows_px[paired], boundary.left_px[paired], 1)[0],
            0.0,
        )
        self.assertGreater(
            np.polyfit(boundary.rows_px[paired], boundary.right_px[paired], 1)[0],
            0.0,
        )

    def test_22_cr_cb_confidence_penalizes_green_carpet(self):
        config = load_config(os.path.join(PACKAGE_DIR, "config", "calibrated.yaml"))
        config["camera"]["use_bev_polygon_for_detection"] = False
        image = np.full((480, 640, 3), 120, dtype=np.uint8)
        image[300:360, 80:180] = (20, 190, 220)
        image[300:360, 260:360] = (45, 115, 65)
        masks = ColorSegmenter(config).masks(image)
        yellow = float(np.median(masks["yellow_confidence"][310:350, 90:170]))
        green = float(np.median(masks["yellow_confidence"][310:350, 270:350]))
        gray = float(np.median(masks["yellow_confidence"][380:420, 400:500]))
        self.assertGreater(yellow, green + 0.25)
        self.assertGreater(yellow, gray + 0.25)

    def test_23_disc_lane_filter_rejects_only_far_outside_center(self):
        config = calibrated_config()
        pipeline = AvoidancePipeline(config)
        rows = np.asarray([400, 350, 300, 250], dtype=np.int32)
        boundary = BoundaryResult(
            rows_px=rows,
            left_px=np.full(rows.shape, 100.0),
            right_px=np.full(rows.shape, 540.0),
            center_px=np.full(rows.shape, 320.0),
            left_detected=True,
            right_detected=True,
            used_history=False,
            both_missing_frames=0,
            lane_width_px=440.0,
            quality=1.0,
            observed_min_row_px=250,
            observed_max_row_px=400,
        )

        def detection(candidate_id, center):
            contour = np.asarray(
                [
                    [[center[0] - 5, center[1]]],
                    [[center[0], center[1] - 3]],
                    [[center[0] + 5, center[1]]],
                    [[center[0], center[1] + 3]],
                ],
                dtype=np.int32,
            )
            return DiskDetection(
                center_px=center,
                center_m=(0.0, 0.0),
                ellipse=(center, (20.0, 10.0), 0.0),
                contour=contour,
                estimated_diameter_m=0.0,
                confidence=0.9,
                stamp_sec=1.0,
                candidate_id=candidate_id,
            )

        pipeline.disk_detector.last_candidates = [
            {"candidate_id": 1, "accepted": True, "reject_reason": ""},
            {"candidate_id": 2, "accepted": True, "reject_reason": ""},
        ]
        inside = detection(1, (320.0, 350.0))
        outside = detection(2, (50.0, 350.0))
        kept = pipeline._apply_disk_lane_geometry(
            [inside, outside], boundary, (480, 640, 3)
        )
        self.assertEqual([item.candidate_id for item in kept], [1])
        self.assertTrue(inside.inside_lane)
        self.assertFalse(outside.inside_lane)
        self.assertEqual(
            pipeline.disk_detector.last_candidates[1]["reject_reason"],
            "center_outside_lane",
        )

    def test_24_disc_detection_never_reuses_previous_frame(self):
        config = calibrated_config()
        detector = DiskDetector(config)
        metric = lambda points: np.asarray(points, dtype=np.float64) * 0.001
        image = np.full((300, 400, 3), 110, dtype=np.uint8)
        cv2.ellipse(image, (210, 180), (62, 18), 8, 0, 360, (0, 0, 220), -1)
        cv2.ellipse(image, (210, 180), (29, 8), 8, 0, 360, (235, 235, 235), -1)
        masks = ColorSegmenter(config).masks(image)
        first = detector.detect(
            masks["red"], masks["white"], 1.0, 0.001, 0.001, metric,
            image_bgr=image,
        )
        self.assertEqual(len(first), 1)

        blank = np.full_like(image, 110)
        blank_masks = ColorSegmenter(config).masks(blank)
        second = detector.detect(
            blank_masks["red"], blank_masks["white"], 1.1, 0.001, 0.001, metric,
            image_bgr=blank,
        )
        self.assertEqual(second, [])
        self.assertFalse(any(item["accepted"] for item in detector.last_candidates))

    def test_25_red_object_and_separate_white_object_are_not_a_disc(self):
        config = calibrated_config()
        image = np.full((300, 400, 3), 110, dtype=np.uint8)
        cv2.rectangle(image, (80, 140), (160, 205), (0, 0, 230), -1)
        cv2.circle(image, (245, 172), 28, (235, 235, 235), -1)
        masks = ColorSegmenter(config).masks(image)
        metric = lambda points: np.asarray(points, dtype=np.float64) * 0.001
        detections = DiskDetector(config).detect(
            masks["red"], masks["white"], 1.0, 0.001, 0.001, metric,
            image_bgr=image,
        )
        self.assertEqual(detections, [])

    def test_26_raw_boundary_support_is_projected_without_extrapolation(self):
        config = calibrated_config()
        config["bev"].update({
            "source_points": [
                [0.0, 479.0], [639.0, 479.0],
                [639.0, 0.0], [0.0, 0.0],
            ],
            "output_height_px": 480,
            "extend_to_image_bottom": False,
        })
        pipeline = AvoidancePipeline(config)
        raw_rows = np.asarray([400, 350, 300], dtype=np.int32)
        raw = BoundaryResult(
            rows_px=raw_rows,
            left_px=np.full(raw_rows.shape, 100.0),
            right_px=np.full(raw_rows.shape, 540.0),
            center_px=np.full(raw_rows.shape, 320.0),
            left_detected=True,
            right_detected=True,
            used_history=False,
            both_missing_frames=0,
            lane_width_px=440.0,
            quality=0.93,
            observed_min_row_px=300,
            observed_max_row_px=400,
        )
        projected = pipeline._project_boundary_to_bev(raw)
        observed = np.isfinite(projected.left_px) & np.isfinite(projected.right_px)
        self.assertTrue(projected.left_detected)
        self.assertTrue(projected.right_detected)
        self.assertFalse(projected.used_history)
        self.assertGreaterEqual(projected.observed_min_row_px, 300)
        self.assertLessEqual(projected.observed_max_row_px, 400)
        self.assertTrue(np.allclose(projected.left_px[observed], 100.0, atol=0.1))
        self.assertTrue(np.allclose(projected.right_px[observed], 540.0, atol=0.1))
        self.assertAlmostEqual(projected.lane_width_px, 440.0, delta=0.2)

    def test_26b_single_boundary_support_survives_bev_projection(self):
        config = calibrated_config()
        config["bev"].update({
            "source_points": [
                [0.0, 479.0], [639.0, 479.0],
                [639.0, 0.0], [0.0, 0.0],
            ],
            "output_height_px": 480,
            "extend_to_image_bottom": False,
        })
        pipeline = AvoidancePipeline(config)
        raw_rows = np.asarray([400, 350, 300], dtype=np.int32)
        raw = BoundaryResult(
            rows_px=raw_rows,
            left_px=np.full(raw_rows.shape, np.nan),
            right_px=np.full(raw_rows.shape, 540.0),
            center_px=np.full(raw_rows.shape, np.nan),
            left_detected=False,
            right_detected=True,
            used_history=False,
            both_missing_frames=0,
            lane_width_px=0.0,
            quality=0.48,
            observed_min_row_px=300,
            observed_max_row_px=400,
        )
        projected = pipeline._project_boundary_to_bev(raw)
        right_observed = np.isfinite(projected.right_px)
        self.assertFalse(projected.left_detected)
        self.assertTrue(projected.right_detected)
        self.assertGreaterEqual(np.count_nonzero(right_observed), 2)
        self.assertGreaterEqual(projected.observed_min_row_px, 300)
        self.assertLessEqual(projected.observed_max_row_px, 400)
        self.assertAlmostEqual(projected.lane_width_px, 0.0)

    def test_27_calibrated_pipeline_can_skip_unsubscribed_debug_images(self):
        pipeline = AvoidancePipeline(calibrated_config())
        image = np.full((480, 640, 3), 110, dtype=np.uint8)
        result = pipeline.process(image, 1.0, debug_image_names=set())
        self.assertEqual(result.debug_images, {})
        self.assertTrue(result.diagnostics["calibrated"])
        self.assertFalse(result.diagnostics["motion_enabled"])

    def test_28_detect_only_pipeline_keeps_planning_disabled_and_stopped(self):
        pipeline = AvoidancePipeline(calibrated_config())
        image = np.full((480, 640, 3), 110, dtype=np.uint8)
        result = pipeline.process(
            image, 1.0, debug_image_names=set(), run_planning=False
        )
        self.assertEqual(result.state, MotionState.STOP)
        self.assertEqual(result.stop_reason, "planning_disabled")
        self.assertFalse(result.diagnostics["planning_enabled"])
        self.assertEqual(result.command.linear_x, 0.0)

    def test_28a_photometry_transition_can_preserve_far_red_evidence(self):
        pipeline = AvoidancePipeline(calibrated_config())
        evidence = np.full((8, 8), 255, dtype=np.uint8)
        pipeline._bev_far_red_history.append(evidence)

        pipeline.reset_perception_history(preserve_far_red_evidence=True)
        self.assertEqual(len(pipeline._bev_far_red_history), 1)

        pipeline.reset_perception_history(preserve_far_red_evidence=False)
        self.assertEqual(len(pipeline._bev_far_red_history), 0)

    def test_28b_vehicle_forward_origin_offset_prevents_false_close_stop(self):
        from yellow_track_avoidance.model import PathCandidate

        config = calibrated_config()
        config["physical"]["robot_width_m"] = 0.30
        points = np.column_stack(
            [np.linspace(0.0, 1.0, 10), np.zeros(10)]
        )
        candidate = PathCandidate(
            "center", points, 0.0, 0, True, 0.0
        )
        planning = PlanningResult(
            [candidate], candidate, True, 0.2, 0.2
        )
        obstacle = TrackedDisk(
            1, (0.22, -0.263), (487, 686), 0.10, 0.95, 1.0, 3
        )

        without_offset = AvoidanceStateMachine(config).update(
            1.0, True, True, 0, planning, [obstacle]
        )
        self.assertEqual(
            without_offset.stop_reason, "obstacle_too_close"
        )

        config["physical"]["local_forward_origin_offset_m"] = 0.39
        with_offset = AvoidanceStateMachine(config).update(
            1.0, True, True, 0, planning, [obstacle]
        )
        self.assertNotEqual(
            with_offset.stop_reason, "obstacle_too_close"
        )

    def test_29_map_route_is_right_to_left_and_switches_before_narrow_road(self):
        route_path = os.path.join(PACKAGE_DIR, "config", "track_7_15_route.yaml")
        route, _, _, _, _, path, _, report = validate_route(route_path)
        self.assertTrue(np.all(np.diff(path[:, 0]) < 0.0))
        self.assertAlmostEqual(path[0, 0], 4.15, places=6)
        self.assertAlmostEqual(path[-1, 0], 1.00, places=6)
        handoff = float(route["avoidance"]["handoff_x_m"])
        self.assertTrue(np.any(path[:, 0] > handoff))
        self.assertTrue(np.any(path[:, 0] <= handoff))
        self.assertTrue(report["avoidance_disabled_at_and_after_handoff"])

    def test_30_map_route_fits_current_and_migration_vehicles(self):
        route_path = os.path.join(PACKAGE_DIR, "config", "track_7_15_route.yaml")
        report = validate_route(route_path)[-1]
        self.assertEqual(report["map_size_m"], [5.0, 4.0])
        self.assertTrue(report["route_valid_for_all_recorded_vehicles"])
        self.assertTrue(
            report["vehicle_validation"]["current_vehicle"]["collision_free"]
        )
        self.assertTrue(
            report["vehicle_validation"]["migration_vehicle"]["collision_free"]
        )

    def test_31_local_camera_coordinates_transform_into_westbound_map_frame(self):
        transformed = local_to_map(
            np.asarray([[1.0, 0.20], [0.5, -0.10]]),
            (4.15, 3.49, math.pi),
        )
        self.assertTrue(
            np.allclose(transformed[0], [3.15, 3.29], atol=1e-9)
        )
        self.assertTrue(
            np.allclose(transformed[1], [3.65, 3.59], atol=1e-9)
        )

    def test_32_local_bypass_fuses_then_returns_to_static_map_route(self):
        route_path = os.path.join(PACKAGE_DIR, "config", "track_7_15_route.yaml")
        route = load_yaml(route_path)[0]
        reference = build_reference_path(route)
        local_x = np.asarray([0.0, 0.4, 0.8, 1.2])
        center = np.column_stack([local_x, np.zeros_like(local_x)])
        selected = np.column_stack([local_x, [0.0, 0.16, 0.08, 0.0]])
        fused, mode = fuse_candidate_with_reference(
            route,
            reference,
            (4.15, 3.49, math.pi),
            center,
            selected,
        )
        self.assertEqual(mode, "local_avoidance_fused")
        self.assertLess(np.min(fused[:, 1] - reference[:, 1]), -0.10)
        self.assertTrue(np.allclose(fused[-20:], reference[-20:], atol=1e-9))

    def test_33_disc_avoidance_is_disabled_after_map_handoff(self):
        route_path = os.path.join(PACKAGE_DIR, "config", "track_7_15_route.yaml")
        route = load_yaml(route_path)[0]
        reference = build_reference_path(route)
        local = np.asarray([[0.0, 0.0], [0.5, 0.2], [1.0, 0.0]])
        fused, mode = fuse_candidate_with_reference(
            route,
            reference,
            (2.50, 3.66, math.pi),
            np.column_stack([local[:, 0], np.zeros(3)]),
            local,
        )
        self.assertEqual(mode, "fixed_narrow_route")
        self.assertGreater(len(fused), 2)

    def test_34_navigation_lookahead_follows_right_to_left_route(self):
        path = np.asarray([[4.0, 3.5], [3.8, 3.5], [3.6, 3.5]])
        target, heading, nearest = lookahead_target(path, (4.0, 3.5), 0.30)
        self.assertEqual(nearest, 0)
        self.assertTrue(np.allclose(target, [3.7, 3.5], atol=1e-9))
        self.assertAlmostEqual(abs(heading), math.pi, places=9)

    def test_35_navigation_command_moves_forward_when_vehicle_faces_west(self):
        path = np.asarray([[4.0, 3.5], [3.5, 3.5], [3.0, 3.5]])
        command = compute_navigation_command(
            path, (4.0, 3.5, math.pi), 0.30, 0.05, 0.4,
            0.8, 1.2, 0.08, 0.30, math.radians(10), math.radians(25),
        )
        self.assertGreater(command["linear_x"], 0.0)
        self.assertAlmostEqual(command["linear_y"], 0.0, places=9)
        self.assertAlmostEqual(command["angular_z"], 0.0, places=9)

    def test_36_navigation_command_rotates_before_large_heading_error(self):
        path = np.asarray([[4.0, 3.5], [3.5, 3.5], [3.0, 3.5]])
        command = compute_navigation_command(
            path, (4.0, 3.5, 0.0), 0.30, 0.05, 0.4,
            0.8, 1.2, 0.08, 0.30, math.radians(10), math.radians(25),
        )
        self.assertAlmostEqual(command["linear_x"], 0.0, places=9)
        self.assertAlmostEqual(command["linear_y"], 0.0, places=9)
        self.assertAlmostEqual(abs(command["angular_z"]), 0.30, places=9)

    def test_37_fixed_disc_memory_merges_reacquired_track(self):
        memory = StableDiscMemory(8.0, 0.8, 0.18, 0.35)
        first = memory.observe(4, (2.00, 1.00), 1.0, True)
        second = memory.observe(9, (2.04, 0.98), 2.0, True)
        self.assertEqual(first.memory_id, second.memory_id)
        self.assertEqual(len(memory), 1)
        self.assertEqual(second.track_ids, {4, 9})
        self.assertTrue(np.allclose(second.fixed_xy, [2.014, 0.993]))

    def test_37b_scene_reset_clears_identity_and_metric_observations(self):
        memory = StableDiscMemory(8.0, 0.8, 0.18, 0.35)
        first = memory.observe(
            4,
            (2.0, 1.0),
            1.0,
            True,
            estimated_diameter_m=0.067,
            confidence=0.93,
        )
        self.assertEqual(first.estimated_diameter_m, 0.067)
        self.assertEqual(first.confidence, 0.93)
        memory.begin_navigation_epoch()
        memory.clear(reset_identity=True)
        self.assertEqual(len(memory), 0)
        self.assertEqual(memory.navigation_epoch, 0)
        after_reset = memory.observe(9, (3.0, 0.0), 2.0, True)
        self.assertEqual(after_reset.memory_id, 1)

    def test_38_fixed_disc_memory_uses_short_false_positive_timeout(self):
        memory = StableDiscMemory(8.0, 0.8, 0.18, 0.35)
        false_positive = memory.observe(
            1, (2.00, 1.00), 1.0, False
        )
        confirmed = memory.observe(2, (3.00, 1.00), 1.0, True)
        expired = memory.expire(1.81)
        self.assertIn(false_positive.memory_id, expired)
        self.assertNotIn(confirmed.memory_id, expired)
        self.assertEqual(
            [item.memory_id for item in memory.entries()],
            [confirmed.memory_id],
        )

    def test_39_confirmed_disc_cannot_expire_in_active_range(self):
        memory = StableDiscMemory(8.0, 0.8, 0.18, 0.35)
        confirmed = memory.observe(1, (1.00, 0.00), 1.0, True)
        self.assertIsNone(
            memory.stale_removal_reason(
                confirmed, 20.0, 0.50, 1.75
            )
        )
        self.assertEqual(
            memory.stale_removal_reason(
                confirmed, 20.0, 1.80, 1.75
            ),
            "distant_timeout",
        )
        self.assertIsNone(
            memory.stale_removal_reason(
                confirmed, 2.0, 0.25, 1.75
            )
        )

    def test_40_mission_lock_freezes_confirmed_world_position(self):
        memory = StableDiscMemory(
            8.0,
            0.8,
            0.18,
            0.35,
            freeze_confirmed_position=True,
        )
        locked = memory.observe(
            4,
            (2.00, 1.00),
            1.0,
            True,
            estimated_diameter_m=0.10,
        )
        reacquired = memory.observe(
            4,
            (2.04, 0.98),
            2.0,
            True,
            estimated_diameter_m=0.12,
        )
        self.assertEqual(reacquired.memory_id, locked.memory_id)
        self.assertEqual(reacquired.fixed_xy, (2.00, 1.00))
        self.assertEqual(reacquired.last_seen_sec, 2.0)
        self.assertAlmostEqual(reacquired.estimated_diameter_m, 0.107)

    def test_41_mission_lock_uses_confirmation_frame_before_freeze(self):
        memory = StableDiscMemory(
            8.0,
            0.8,
            0.18,
            0.50,
            freeze_confirmed_position=True,
        )
        memory.observe(4, (2.00, 1.00), 1.0, False)
        locked = memory.observe(4, (2.04, 0.98), 2.0, True)
        self.assertEqual(locked.fixed_xy, (2.02, 0.99))
        after_lock = memory.observe(4, (2.08, 0.96), 3.0, True)
        self.assertEqual(after_lock.fixed_xy, (2.02, 0.99))


if __name__ == "__main__":
    unittest.main()
