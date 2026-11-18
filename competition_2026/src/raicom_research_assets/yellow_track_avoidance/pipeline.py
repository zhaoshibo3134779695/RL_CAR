"""ROS-independent end-to-end perception, planning, and control pipeline."""

from collections import deque
from copy import deepcopy
import math
import time
from typing import Dict, List, Optional

import cv2
import numpy as np

from .bev import BirdEyeTransformer, CameraUndistorter
from .config import calibration_ready, intrinsic_calibration_ready
from .controller import PathController
from .model import (
    BoundaryResult,
    ControlCommand,
    DiskDetection,
    PipelineResult,
    PlanningResult,
)
from .planning import LocalPathPlanner, build_safe_corridor
from .profile_boundary import NormalProfileBoundaryDetector
from .state_machine import AvoidanceStateMachine
from .tracking import DiskTracker
from .vision import (
    BoundaryDetector,
    ColorSegmenter,
    DiskDetector,
)


class AvoidancePipeline:
    def __init__(self, config: Dict):
        self.config = config
        segmentation_config = config
        raw_points = config.get("camera", {}).get("detection_source_points")
        if raw_points is not None:
            segmentation_config = deepcopy(config)
            segmentation_config["camera"]["use_bev_polygon_for_detection"] = True
            segmentation_config["bev"]["source_points"] = raw_points
            segmentation_config["bev"]["extend_to_image_bottom"] = bool(
                config["camera"].get(
                    "detection_extend_to_image_bottom", True
                )
            )
        self.segmenter = ColorSegmenter(segmentation_config)
        self.boundary_detector = BoundaryDetector(config)
        self.profile_boundary_detector = (
            NormalProfileBoundaryDetector(config)
            if bool(config.get("normal_profile_detector", {}).get("enabled", False))
            else None
        )
        self.disk_detector = DiskDetector(config)
        fallback_frames = max(
            1,
            int(
                config.get("disk_detection", {}).get(
                    "bev_far_fallback_temporal_red_frames", 1
                )
            ),
        )
        self._bev_far_red_history = deque(maxlen=fallback_frames)
        self.tracker = DiskTracker(config)
        self.planner = LocalPathPlanner(config)
        self.state_machine = AvoidanceStateMachine(config)
        self.controller = PathController(config)
        self.undistorter: Optional[CameraUndistorter] = None
        self.transformer: Optional[BirdEyeTransformer] = None
        if intrinsic_calibration_ready(config):
            self.undistorter = CameraUndistorter(config)
        if calibration_ready(config):
            self.transformer = BirdEyeTransformer(config)

    @property
    def calibrated(self) -> bool:
        return self.transformer is not None

    def reset_stop(self) -> None:
        self.state_machine.request_reset()

    def reset_perception_history(self, preserve_far_red_evidence=False) -> None:
        """Discard observations collected before the commissioned image state.

        Startup photometry tuning is allowed to run the real detector for
        measurement, but those provisional frames must not contribute track
        confirmation after the selected exposure is latched.  The bounded
        five-frame far-disc red history may be retained across that exact
        transition: it contains image evidence at the newly selected fixed
        exposure, not a confirmed mission track, and lets the formal tracker
        reacquire a heavily foreshortened far disc immediately.  Explicit
        scene resets still clear it.
        """
        self.tracker.clear()
        self.tracker.next_id = 1
        self.disk_detector.last_candidates = []
        self.disk_detector.last_red_area_px = 0
        self.disk_detector.last_white_area_px = 0
        if not preserve_far_red_evidence:
            self._bev_far_red_history.clear()
        if self.profile_boundary_detector is not None:
            self.profile_boundary_detector.both_missing_frames = 0
            self.profile_boundary_detector.last_metrics = {}
        self.boundary_detector.previous = None
        self.boundary_detector.both_missing_frames = 0

    @staticmethod
    def _mask_bgr(mask: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _observed_boundary_mask(boundary: BoundaryResult) -> np.ndarray:
        if (
            boundary.observed_min_row_px < 0
            or boundary.observed_max_row_px < boundary.observed_min_row_px
        ):
            return np.zeros(boundary.rows_px.shape, dtype=bool)
        return (
            (boundary.rows_px >= int(boundary.observed_min_row_px))
            & (boundary.rows_px <= int(boundary.observed_max_row_px))
            & np.isfinite(boundary.left_px)
            & np.isfinite(boundary.right_px)
        )

    @staticmethod
    def _boundary_points(boundary: BoundaryResult, side: str) -> np.ndarray:
        values = boundary.left_px if side == "left" else boundary.right_px
        finite = np.isfinite(values) & np.isfinite(boundary.rows_px)
        if int(np.count_nonzero(finite)) < 2:
            return np.empty((0, 2), dtype=np.float64)
        points = np.column_stack(
            [values[finite], boundary.rows_px[finite].astype(np.float64)]
        )
        return points[np.argsort(points[:, 1])]

    @staticmethod
    def _point_to_polyline_distance(point, polyline: np.ndarray) -> float:
        if len(polyline) < 2:
            return float("nan")
        point_array = np.asarray(point, dtype=np.float64)
        starts = polyline[:-1]
        segments = polyline[1:] - starts
        lengths_squared = np.sum(segments * segments, axis=1)
        projection = np.sum((point_array - starts) * segments, axis=1) / np.maximum(
            lengths_squared, 1e-9
        )
        projection = np.clip(projection, 0.0, 1.0)
        nearest = starts + projection[:, None] * segments
        return float(np.min(np.linalg.norm(nearest - point_array, axis=1)))

    def _apply_disk_lane_geometry(self, detections, boundary, image_shape):
        """Attach current-frame lane metrics and reject outside candidates.

        The polygon closes the two observed yellow segments through the image
        bottom.  This represents the drivable image region after both yellow
        lines leave through the side edges without extrapolating either line.
        """
        if boundary is None or not (
            boundary.left_detected and boundary.right_detected
        ):
            return detections
        left = self._boundary_points(boundary, "left")
        right = self._boundary_points(boundary, "right")
        if len(left) < 2 or len(right) < 2:
            return detections
        height, width = image_shape[:2]
        polygon = np.vstack(
            [
                left,
                np.asarray([[0.0, height - 1.0], [width - 1.0, height - 1.0]]),
                right[::-1],
            ]
        ).astype(np.float32)
        margin = float(self.config["disk_detection"].get("lane_center_margin_px", 20.0))
        enabled = bool(self.config["disk_detection"].get("lane_filter_enabled", True))
        kept = []
        candidate_by_id = {
            int(item.get("candidate_id", -1)): item
            for item in self.disk_detector.last_candidates
        }
        for detection in detections:
            signed_distance = float(
                cv2.pointPolygonTest(
                    polygon.reshape(-1, 1, 2), detection.center_px, True
                )
            )
            detection.inside_lane = signed_distance >= 0.0
            detection.distance_to_left_line_px = self._point_to_polyline_distance(
                detection.center_px, left
            )
            detection.distance_to_right_line_px = self._point_to_polyline_distance(
                detection.center_px, right
            )
            record = candidate_by_id.get(detection.candidate_id)
            if record is not None:
                record.update(
                    {
                        "inside_lane": detection.inside_lane,
                        "lane_signed_distance_px": signed_distance,
                        "distance_to_left_line_px": detection.distance_to_left_line_px,
                        "distance_to_right_line_px": detection.distance_to_right_line_px,
                    }
                )
            if enabled and signed_distance < -margin:
                if record is not None:
                    record["accepted"] = False
                    record["reject_reason"] = "center_outside_lane"
                continue
            kept.append(detection)
        return kept

    def _draw_disc_overlay(self, image, disks, boundary):
        canvas = image.copy()
        if boundary is not None:
            for side in ("left", "right"):
                points = self._boundary_points(boundary, side)
                if len(points) >= 2:
                    cv2.polylines(
                        canvas,
                        [np.rint(points).astype(np.int32)],
                        False,
                        (0, 220, 255),
                        2,
                        cv2.LINE_AA,
                    )
        for index, disk in enumerate(disks):
            cv2.ellipse(canvas, disk.ellipse, (0, 255, 0), 2, cv2.LINE_AA)
            center = tuple(np.rint(disk.center_px).astype(int))
            cv2.drawMarker(
                canvas, center, (255, 255, 255), cv2.MARKER_CROSS, 10, 1
            )
            label = "D{} {:.2f} current".format(index, disk.confidence)
            cv2.putText(
                canvas,
                label,
                (center[0] + 6, center[1] - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        cv2.putText(
            canvas,
            "discs={} history=OFF motion=OFF".format(len(disks)),
            (12, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return canvas

    def _draw_disc_candidates(self, image):
        canvas = image.copy()
        candidates = self.disk_detector.last_candidates
        rejection_counts = {}
        for item in candidates:
            ellipse = item.get("ellipse")
            if ellipse is None:
                continue
            geometry = np.asarray(
                [ellipse[0][0], ellipse[0][1], ellipse[1][0], ellipse[1][1], ellipse[2]],
                dtype=np.float64,
            )
            if not np.all(np.isfinite(geometry)):
                continue
            accepted = bool(item.get("accepted", False))
            reason = str(item.get("reject_reason", ""))
            color = (0, 255, 0) if accepted else (
                (0, 180, 255) if reason == "duplicate_candidate" else (0, 0, 255)
            )
            cv2.ellipse(canvas, ellipse, color, 1, cv2.LINE_AA)
            center = (int(round(ellipse[0][0])), int(round(ellipse[0][1])))
            if accepted:
                cv2.putText(
                    canvas,
                    "{}:OK".format(int(item.get("candidate_id", -1))),
                    (center[0] + 3, center[1] + 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.36,
                    color,
                    1,
                    cv2.LINE_AA,
                )
            else:
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
        cv2.putText(
            canvas,
            "candidates={} accepted={}".format(
                len(candidates), sum(bool(item.get("accepted")) for item in candidates)
            ),
            (12, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        summary = " ".join(
            "{}={}".format(reason[:12], count)
            for reason, count in sorted(
                rejection_counts.items(), key=lambda item: (-item[1], item[0])
            )[:4]
        )
        cv2.putText(
            canvas,
            summary,
            (12, 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
        return canvas

    def _project_disk_to_bev(self, detection):
        """Project an original-image disk detection onto the metric ground plane.

        Detecting the thin red ring after a strong perspective warp destroys too
        much colour evidence.  Detection therefore stays in the undistorted
        camera image; only the accepted geometry is projected into the BEV.
        """
        candidate_record = next(
            (
                item
                for item in self.disk_detector.last_candidates
                if int(item.get("candidate_id", -1))
                == int(detection.candidate_id)
            ),
            None,
        )
        center_bev = self.transformer.original_to_bev([detection.center_px])[0]
        if not np.all(np.isfinite(center_bev)):
            if candidate_record is not None:
                candidate_record["accepted"] = False
                candidate_record["reject_reason"] = "metric_projection_nonfinite"
            return None

        center_m = self.transformer.pixel_to_metric(np.asarray([center_bev]))[0]
        forward_offset = float(
            self.config["physical"].get("local_forward_origin_offset_m", 0.0)
        )
        center_base_forward_m = float(center_m[0] + forward_offset)
        inside_bev = bool(
            0.0 <= center_bev[0] < self.transformer.width_px
            and 0.0 <= center_bev[1] < self.transformer.height_px
        )
        near_field_cfg = self.config["disk_detection"]
        near_field_extrapolated = bool(
            near_field_cfg.get("allow_near_field_ground_extrapolation", False)
            and 0.0 <= center_bev[0] < self.transformer.width_px
            and center_bev[1] >= self.transformer.height_px
            and float(
                near_field_cfg.get("near_field_minimum_forward_from_base_m", 0.0)
            )
            <= center_base_forward_m
            <= float(
                near_field_cfg.get(
                    "near_field_maximum_forward_from_base_m", forward_offset
                )
            )
        )
        if not inside_bev and not near_field_extrapolated:
            if candidate_record is not None:
                candidate_record["accepted"] = False
                candidate_record["reject_reason"] = "metric_projection_outside_valid_ground"
            return None

        (cx, cy), (axis_a, axis_b), angle_deg = detection.ellipse
        angles = np.linspace(0.0, 2.0 * math.pi, 72, endpoint=False)
        theta = math.radians(float(angle_deg))
        cos_theta, sin_theta = math.cos(theta), math.sin(theta)
        local_x = 0.5 * float(axis_a) * np.cos(angles)
        local_y = 0.5 * float(axis_b) * np.sin(angles)
        ellipse_points = np.column_stack(
            [
                float(cx) + local_x * cos_theta - local_y * sin_theta,
                float(cy) + local_x * sin_theta + local_y * cos_theta,
            ]
        )
        ellipse_bev = self.transformer.original_to_bev(ellipse_points)
        finite = np.all(np.isfinite(ellipse_bev), axis=1)
        ellipse_bev = ellipse_bev[finite]
        if len(ellipse_bev) < 5:
            return None
        if (
            np.max(np.abs(ellipse_bev[:, 0])) > 4.0 * self.transformer.width_px
            or np.max(np.abs(ellipse_bev[:, 1])) > 4.0 * self.transformer.height_px
        ):
            return None

        projected_contour = self.transformer.original_to_bev(
            detection.contour.reshape(-1, 2)
        )
        projected_contour = projected_contour[
            np.all(np.isfinite(projected_contour), axis=1)
        ]
        if len(projected_contour) < 3:
            projected_contour = ellipse_bev
        contour_bev = projected_contour.astype(np.float32).reshape(-1, 1, 2)
        try:
            fitted_bev = cv2.fitEllipse(
                ellipse_bev.astype(np.float32).reshape(-1, 1, 2)
            )
        except cv2.error:
            return None

        perimeter_m = self.transformer.pixel_to_metric(ellipse_bev)
        # Keep contour-size bias separate from the Homography: it must never
        # rescale otherwise-correct disc centres.  The legacy lateral span is
        # retained by default.  A calibrated profile may instead use the
        # geometric mean of both ground-plane spans; this is the equal-area
        # diameter and cancels small, opposite residual scale errors between
        # the forward and lateral axes.
        visible_forward_span_m = float(np.ptp(perimeter_m[:, 0]))
        visible_lateral_span_m = float(np.ptp(perimeter_m[:, 1]))
        diameter_mode = str(
            self.config["disk_detection"].get(
                "projected_diameter_mode", "lateral_span"
            )
        )
        if diameter_mode == "geometric_mean_ground_spans":
            visible_ring_diameter_m = math.sqrt(
                max(0.0, visible_forward_span_m * visible_lateral_span_m)
            )
        else:
            visible_ring_diameter_m = visible_lateral_span_m
        visible_ring_ratio = float(
            self.config["disk_detection"].get(
                "visible_ring_to_physical_diameter_ratio", 1.0
            )
        )
        diameter_m = visible_ring_diameter_m / visible_ring_ratio
        distorted_center = (
            np.asarray(detection.center_px, dtype=np.float64)
            if self.undistorter is None
            else self.undistorter.distort_points([detection.center_px])[0]
        )
        projection_diagnostics = {
            "input_resolution_px": [
                int(self.config["camera"]["expected_width"]),
                int(self.config["camera"]["expected_height"]),
            ],
            "roi_applied": False,
            "resize_applied": False,
            "detection_coordinate_system": (
                "undistorted_full_image_px"
                if self.undistorter is not None
                else "distorted_full_image_px"
            ),
            "distorted_full_image_px": [
                float(distorted_center[0]),
                float(distorted_center[1]),
            ],
            "roi_pixel": None,
            "resized_pixel": None,
            "undistorted_full_image_px": [
                float(detection.center_px[0]),
                float(detection.center_px[1]),
            ],
            "homography_source_is_undistorted": bool(
                self.config["bev"].get("source_is_undistorted", False)
            ),
            "homography_input_px": [
                float(detection.center_px[0]),
                float(detection.center_px[1]),
            ],
            "bev_pixel": [float(center_bev[0]), float(center_bev[1])],
            "bev_output_resolution_px": [
                int(self.transformer.width_px),
                int(self.transformer.height_px),
            ],
            "bev_lateral_m_per_pixel": float(
                self.transformer.m_per_px_x
            ),
            "bev_forward_m_per_pixel": float(
                self.transformer.m_per_px_y
            ),
            "ground_m": [float(center_m[0]), float(center_m[1])],
            "base_m": [
                float(center_m[0] + forward_offset),
                float(center_m[1]),
            ],
            "inside_bev_output": inside_bev,
            "near_field_ground_extrapolated": near_field_extrapolated,
            "diameter_method": (
                "72_point_visible_ring_geometric_mean_ground_spans_with_physical_ratio"
                if diameter_mode == "geometric_mean_ground_spans"
                and visible_ring_ratio != 1.0
                else "72_point_ellipse_same_homography_geometric_mean_ground_spans"
                if diameter_mode == "geometric_mean_ground_spans"
                else "72_point_visible_ring_lateral_span_with_physical_ratio"
                if visible_ring_ratio != 1.0
                else "72_point_ellipse_same_homography_lateral_span"
            ),
            "diameter_sample_count": int(len(perimeter_m)),
            "visible_ring_diameter_mode": diameter_mode,
            "visible_ring_to_physical_diameter_ratio": visible_ring_ratio,
            "estimated_visible_ring_diameter_m": visible_ring_diameter_m,
            "estimated_visible_ring_forward_span_m": visible_forward_span_m,
            "estimated_visible_ring_lateral_span_m": visible_lateral_span_m,
            "perimeter_ground_bounds_m": {
                "forward": [
                    float(np.min(perimeter_m[:, 0])),
                    float(np.max(perimeter_m[:, 0])),
                ],
                "lateral": [
                    float(np.min(perimeter_m[:, 1])),
                    float(np.max(perimeter_m[:, 1])),
                ],
            },
            "estimated_physical_diameter_m": diameter_m,
            "estimated_diameter_m": diameter_m,
        }
        minimum_diameter = float(
            self.config["disk_detection"].get(
                "minimum_projected_diameter_m", 0.0
            )
        )
        maximum_diameter = float(
            self.config["disk_detection"].get(
                "maximum_projected_diameter_m", float("inf")
            )
        )
        if candidate_record is not None:
            candidate_record["projected_diameter_m"] = diameter_m
            candidate_record["metric_projection"] = projection_diagnostics
        far_disc_recovery = bool(
            candidate_record
            and (
                candidate_record.get("far_white_center_recovery", False)
                or candidate_record.get("far_red_ring_recovery", False)
            )
        )
        if far_disc_recovery:
            maximum_diameter = float(
                self.config["disk_detection"].get(
                    "far_recovery_maximum_projected_diameter_m",
                    maximum_diameter,
                )
            )
        if diameter_m < minimum_diameter or diameter_m > maximum_diameter:
            if candidate_record is not None:
                candidate_record["accepted"] = False
                candidate_record["reject_reason"] = (
                    "projected_diameter_too_small"
                    if diameter_m < minimum_diameter
                    else "projected_diameter_too_large"
                )
            return None
        return type(detection)(
            center_px=(float(center_bev[0]), float(center_bev[1])),
            center_m=(float(center_m[0]), float(center_m[1])),
            ellipse=fitted_bev,
            contour=contour_bev,
            estimated_diameter_m=diameter_m,
            confidence=detection.confidence,
            stamp_sec=detection.stamp_sec,
            red_annulus_ratio=detection.red_annulus_ratio,
            white_center_ratio=detection.white_center_ratio,
            red_sector_coverage=detection.red_sector_coverage,
            white_center_contrast=detection.white_center_contrast,
            proposal_source=detection.proposal_source + "_original",
            candidate_id=detection.candidate_id,
            contour_area_px=detection.contour_area_px,
            red_area_px=detection.red_area_px,
            white_area_px=detection.white_area_px,
            red_white_center_distance_px=detection.red_white_center_distance_px,
            inside_lane=detection.inside_lane,
            distance_to_left_line_px=detection.distance_to_left_line_px,
            distance_to_right_line_px=detection.distance_to_right_line_px,
            projection_diagnostics=projection_diagnostics,
        )

    def _project_boundary_to_bev(self, boundary: BoundaryResult) -> BoundaryResult:
        """Project the accepted raw-image boundary support into the metric BEV.

        The raw normal-profile detector is the commissioned yellow-line
        detector.  Re-segmenting its warped mask would clip line centres at the
        BEV edges and would replace current-frame support with a second,
        unrelated fit.  This method preserves the exact observed support and
        only changes coordinates.
        """
        height = int(self.transformer.height_px)
        step = max(1, int(self.config["boundary"]["row_step_px"]))
        rows = np.arange(height - 1, -1, -step, dtype=np.int32)

        def project_side(values, detected):
            output = np.full(rows.shape, np.nan, dtype=np.float64)
            if not detected:
                return output, False
            valid = np.isfinite(values) & np.isfinite(boundary.rows_px)
            if boundary.observed_min_row_px >= 0:
                valid &= boundary.rows_px >= int(boundary.observed_min_row_px)
                valid &= boundary.rows_px <= int(boundary.observed_max_row_px)
            if int(np.count_nonzero(valid)) < 2:
                return output, False
            points = np.column_stack(
                [values[valid], boundary.rows_px[valid].astype(np.float64)]
            )
            projected = self.transformer.original_to_bev(points)
            finite = np.all(np.isfinite(projected), axis=1)
            projected = projected[finite]
            if len(projected) < 2:
                return output, False
            order = np.argsort(projected[:, 1])
            projected = projected[order]
            unique_y, unique_indices = np.unique(projected[:, 1], return_index=True)
            unique_x = projected[unique_indices, 0]
            if len(unique_y) < 2:
                return output, False
            supported = (rows >= int(math.ceil(max(0.0, float(unique_y[0]))))) & (
                rows <= int(math.floor(min(float(height - 1), float(unique_y[-1]))))
            )
            output[supported] = np.interp(
                rows[supported].astype(np.float64), unique_y, unique_x
            )
            return output, int(np.count_nonzero(supported)) >= 2

        left, left_detected = project_side(
            boundary.left_px, boundary.left_detected
        )
        right, right_detected = project_side(
            boundary.right_px, boundary.right_detected
        )
        paired = np.isfinite(left) & np.isfinite(right)
        # Keep the observed row support when only one physical yellow edge is
        # visible.  ros_node performs the metric-width reconstruction later;
        # using the left/right intersection here used to erase the surviving
        # side before that inference could run.
        observed = np.isfinite(left) | np.isfinite(right)
        center = np.full(rows.shape, np.nan, dtype=np.float64)
        center[paired] = 0.5 * (left[paired] + right[paired])
        if np.any(observed):
            observed_rows = rows[observed]
            observed_min = int(np.min(observed_rows))
            observed_max = int(np.max(observed_rows))
            lane_width = (
                float(np.median(right[paired] - left[paired]))
                if np.any(paired)
                else 0.0
            )
        else:
            observed_min = -1
            observed_max = -1
            lane_width = 0.0
        return BoundaryResult(
            rows_px=rows,
            left_px=left,
            right_px=right,
            center_px=center,
            left_detected=left_detected,
            right_detected=right_detected,
            used_history=bool(boundary.used_history),
            both_missing_frames=int(boundary.both_missing_frames),
            lane_width_px=lane_width,
            quality=float(boundary.quality),
            observed_min_row_px=observed_min,
            observed_max_row_px=observed_max,
        )

    def _merge_disk_views(self, detections):
        """Fuse original-image and BEV candidates without merging two disks."""
        threshold = float(
            self.config["disk_detection"].get("cross_view_dedup_distance_m", 0.06)
        )
        merged = []
        for detection in sorted(
            detections, key=lambda item: item.confidence, reverse=True
        ):
            if any(
                math.hypot(
                    detection.center_m[0] - kept.center_m[0],
                    detection.center_m[1] - kept.center_m[1],
                ) <= threshold
                for kept in merged
            ):
                continue
            merged.append(detection)
        return sorted(merged, key=lambda item: item.center_m[0])

    def _detect_metric_bev_far_disks(self, masks, working_image, stamp_sec):
        """Recover severely foreshortened far discs in calibrated ground space.

        At the far end of the Stage-1 view the 100 mm red/white disc is only
        one to three source-image pixels high.  Fitting an ellipse to those
        pixels is intrinsically intermittent.  The calibrated homography,
        however, gives us a stronger and physically meaningful test: a white
        centre at a valid ground location must have red evidence in the
        annulus of a 100 mm metric disc.

        This is intentionally a *far-range fallback*.  Normal and near discs
        continue through ``DiskDetector`` with all of its strict image-space
        checks.  A candidate here must independently satisfy white-component,
        red-annulus, multi-sector, metric-range and lateral-road constraints.
        Navigation still uses the commissioned fixed physical disc radius.
        """
        cfg = self.config["disk_detection"]
        if not bool(cfg.get("bev_far_fallback_enabled", False)):
            return []
        # Do not let a one-arc first frame seed a biased mission track.  Five
        # frames cost only about 0.5 s during stationary ACQUIRE and make the
        # two sides of the compressed red ring available before any fallback
        # position is emitted.
        if len(self._bev_far_red_history) < self._bev_far_red_history.maxlen:
            return []

        temporal_red = (
            np.maximum.reduce(tuple(self._bev_far_red_history))
            if self._bev_far_red_history
            else masks["red"]
        )
        red_bev = self.transformer.warp_mask(temporal_red)
        white_bev = self.transformer.warp_mask(masks["white"])
        value_bev = self.transformer.warp_image(
            cv2.cvtColor(working_image, cv2.COLOR_BGR2HSV)[:, :, 2]
        )
        close_x = max(
            1, int(cfg.get("bev_far_fallback_proposal_close_kernel_x", 27)) | 1
        )
        close_y = max(
            1, int(cfg.get("bev_far_fallback_proposal_close_kernel_y", 3)) | 1
        )
        proposal_red = cv2.morphologyEx(
            temporal_red,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (close_x, close_y)),
        )
        contours, _ = cv2.findContours(
            proposal_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        physical_diameter = float(self.config["physical"]["disk_diameter_m"])
        radius_m = 0.5 * physical_diameter
        radius_x = radius_m / max(self.transformer.m_per_px_x, 1e-9)
        radius_y = radius_m / max(self.transformer.m_per_px_y, 1e-9)
        annulus_inner = float(cfg.get("bev_far_fallback_annulus_inner_ratio", 0.45))
        annulus_outer = float(cfg.get("bev_far_fallback_annulus_outer_ratio", 1.35))
        center_ratio = float(cfg.get("bev_far_fallback_center_ratio", 0.65))
        minimum_area = float(
            cfg.get("bev_far_fallback_minimum_proposal_area_px", 2.0)
        )
        maximum_area = float(
            cfg.get("bev_far_fallback_maximum_proposal_area_px", 500.0)
        )
        minimum_width = int(cfg.get("bev_far_fallback_minimum_proposal_width_px", 8))
        maximum_width = int(cfg.get("bev_far_fallback_maximum_proposal_width_px", 65))
        maximum_height = int(cfg.get("bev_far_fallback_maximum_proposal_height_px", 18))
        minimum_red_pixels = int(cfg.get("bev_far_fallback_minimum_red_pixels", 30))
        minimum_white_ratio = float(
            cfg.get("bev_far_fallback_minimum_white_center_ratio", 0.18)
        )
        minimum_center_contrast = float(
            cfg.get("bev_far_fallback_minimum_center_contrast", 12.0)
        )
        sector_count = max(4, int(cfg.get("bev_far_fallback_sector_count", 8)))
        sector_pixel_threshold = max(
            1, int(cfg.get("bev_far_fallback_minimum_red_pixels_per_sector", 5))
        )
        minimum_supported_sectors = max(
            1, int(cfg.get("bev_far_fallback_minimum_supported_sectors", 2))
        )
        forward_offset = float(
            self.config["physical"].get("local_forward_origin_offset_m", 0.0)
        )
        minimum_base_forward = float(
            cfg.get("bev_far_fallback_minimum_forward_from_base_m", 0.90)
        )
        maximum_base_forward = float(
            cfg.get("bev_far_fallback_maximum_forward_from_base_m", 1.95)
        )
        maximum_lateral = float(
            cfg.get("bev_far_fallback_maximum_abs_lateral_m", 0.52)
        )

        candidates = []
        height, width = red_bev.shape[:2]
        for candidate_id, contour in enumerate(contours):
            proposal_area = float(cv2.contourArea(contour))
            if not minimum_area <= proposal_area <= maximum_area:
                continue
            _, _, proposal_width, proposal_height = cv2.boundingRect(contour)
            if not minimum_width <= proposal_width <= maximum_width:
                continue
            if proposal_height > maximum_height:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] <= 0.0:
                continue
            source_center_x = float(moments["m10"] / moments["m00"])
            source_center_y = float(moments["m01"] / moments["m00"])
            center_bev = self.transformer.original_to_bev(
                np.asarray([[source_center_x, source_center_y]], dtype=np.float64)
            )[0]
            if not np.all(np.isfinite(center_bev)):
                continue
            center_x = float(center_bev[0])
            center_y = float(center_bev[1])
            center_m = self.transformer.pixel_to_metric(
                np.asarray([[center_x, center_y]], dtype=np.float64)
            )[0]
            center_base_forward = float(center_m[0] + forward_offset)
            if not minimum_base_forward <= center_base_forward <= maximum_base_forward:
                continue
            if abs(float(center_m[1])) > maximum_lateral:
                continue

            extent_x = annulus_outer * radius_x + 2.0
            extent_y = annulus_outer * radius_y + 2.0
            x0 = max(0, int(math.floor(center_x - extent_x)))
            x1 = min(width, int(math.ceil(center_x + extent_x)) + 1)
            y0 = max(0, int(math.floor(center_y - extent_y)))
            y1 = min(height, int(math.ceil(center_y + extent_y)) + 1)
            if x0 >= x1 or y0 >= y1:
                continue
            yy, xx = np.indices((y1 - y0, x1 - x0), dtype=np.float32)
            normalized_x = (xx + x0 - center_x) / max(radius_x, 1e-6)
            normalized_y = (yy + y0 - center_y) / max(radius_y, 1e-6)
            radial = np.sqrt(normalized_x ** 2 + normalized_y ** 2)
            annulus = (radial >= annulus_inner) & (radial <= annulus_outer)
            center = radial <= center_ratio
            red_crop = red_bev[y0:y1, x0:x1] > 0
            white_crop = white_bev[y0:y1, x0:x1] > 0
            value_crop = value_bev[y0:y1, x0:x1]
            red_support = red_crop & annulus
            red_pixels = int(np.count_nonzero(red_support))
            white_pixels = int(np.count_nonzero(white_crop & center))
            white_ratio = float(white_pixels) / max(1, int(np.count_nonzero(center)))
            center_contrast = float(
                np.median(value_crop[center]) - np.median(value_crop[annulus])
            )
            if (
                red_pixels < minimum_red_pixels
                or white_ratio < minimum_white_ratio
                or center_contrast < minimum_center_contrast
            ):
                continue

            angular = (
                np.arctan2(normalized_y, normalized_x) + 2.0 * math.pi
            ) % (2.0 * math.pi)
            supported_sectors = 0
            for sector_index in range(sector_count):
                low = 2.0 * math.pi * sector_index / sector_count
                high = 2.0 * math.pi * (sector_index + 1) / sector_count
                sector_red = int(
                    np.count_nonzero(
                        red_support & (angular >= low) & (angular < high)
                    )
                )
                supported_sectors += sector_red >= sector_pixel_threshold
            if supported_sectors < minimum_supported_sectors:
                continue

            red_ratio = float(red_pixels) / max(1, int(np.count_nonzero(annulus)))
            sector_coverage = float(supported_sectors) / float(sector_count)
            confidence = float(
                np.clip(
                    0.50
                    + 0.22 * min(1.0, red_pixels / 180.0)
                    + 0.18 * min(1.0, white_ratio / 0.40)
                    + 0.05 * min(1.0, center_contrast / 30.0)
                    + 0.05 * min(1.0, supported_sectors / 4.0),
                    0.0,
                    0.95,
                )
            )
            ellipse = (
                (center_x, center_y),
                (2.0 * radius_x, 2.0 * radius_y),
                0.0,
            )
            ellipse_contour = cv2.ellipse2Poly(
                (int(round(center_x)), int(round(center_y))),
                (max(1, int(round(radius_x))), max(1, int(round(radius_y)))),
                0,
                0,
                360,
                6,
            ).reshape(-1, 1, 2)
            candidates.append(
                DiskDetection(
                    center_px=(center_x, center_y),
                    center_m=(float(center_m[0]), float(center_m[1])),
                    ellipse=ellipse,
                    contour=ellipse_contour,
                    estimated_diameter_m=physical_diameter,
                    confidence=confidence,
                    stamp_sec=stamp_sec,
                    red_annulus_ratio=red_ratio,
                    white_center_ratio=white_ratio,
                    red_sector_coverage=sector_coverage,
                    white_center_contrast=center_contrast,
                    proposal_source="bev_metric_white_red_fallback",
                    candidate_id=100000 + candidate_id,
                    contour_area_px=proposal_area,
                    red_area_px=red_pixels,
                    white_area_px=white_pixels,
                    red_white_center_distance_px=0.0,
                    inside_lane=True,
                    projection_diagnostics={
                        "detection_coordinate_system": "metric_bev_px",
                        "undistorted_full_image_px": [
                            source_center_x,
                            source_center_y,
                        ],
                        "ground_m": [float(center_m[0]), float(center_m[1])],
                        "base_m": [center_base_forward, float(center_m[1])],
                        "diameter_method": "known_100mm_metric_annulus_fallback",
                        "estimated_physical_diameter_m": physical_diameter,
                        "red_proposal_area_px": proposal_area,
                        "red_proposal_width_px": proposal_width,
                        "red_proposal_height_px": proposal_height,
                        "metric_annulus_red_pixels": red_pixels,
                        "metric_supported_red_sectors": supported_sectors,
                        "metric_center_contrast": center_contrast,
                    },
                )
            )

        maximum_candidates = max(
            1, int(cfg.get("bev_far_fallback_maximum_candidates", 2))
        )
        return sorted(
            candidates, key=lambda item: item.confidence, reverse=True
        )[:maximum_candidates]

    @staticmethod
    def _draw_raw_overlay(
        image, disk_roi_mask, yellow_roi_mask, raw_disks, boundary
    ):
        overlay = image.copy()
        for roi_mask, color in (
            (disk_roi_mask, (255, 0, 0)),
            (yellow_roi_mask, (0, 255, 0)),
        ):
            contours, _ = cv2.findContours(
                roi_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay, contours, -1, color, 3, cv2.LINE_AA)
        if boundary is not None:
            rows = boundary.rows_px.astype(np.int32)
            for detected, values in (
                (boundary.left_detected, boundary.left_px),
                (boundary.right_detected, boundary.right_px),
            ):
                if not detected:
                    continue
                visible = (
                    (rows >= int(boundary.observed_min_row_px))
                    & (rows <= int(boundary.observed_max_row_px))
                    & np.isfinite(values)
                )
                points = np.column_stack([values[visible], rows[visible]])
                if len(points) >= 2:
                    cv2.polylines(
                        overlay,
                        [np.rint(points).astype(np.int32)],
                        False,
                        (0, 255, 255),
                        3,
                        cv2.LINE_AA,
                    )
        for disk in raw_disks:
            cv2.drawContours(overlay, [disk.contour], -1, (0, 0, 255), 3)
            cv2.ellipse(overlay, disk.ellipse, (0, 0, 255), 2, cv2.LINE_AA)
        return overlay

    def _draw_bev_plan(self, bev_image, boundary, corridor, planning, disks):
        canvas = bev_image.copy()
        if boundary is not None:
            rows = boundary.rows_px.astype(np.int32)
            observed = self._observed_boundary_mask(boundary)
            for values, color in (
                (boundary.left_px, (0, 255, 255)),
                (boundary.right_px, (0, 255, 255)),
                (boundary.center_px, (255, 180, 0)),
            ):
                points = np.column_stack(
                    [values[observed], rows[observed]]
                ).astype(np.int32)
                if len(points) < 2:
                    continue
                cv2.polylines(canvas, [points], False, color, 2, cv2.LINE_AA)
        if corridor is not None and corridor.forward_m.size and self.transformer is not None:
            for lateral, color in (
                (corridor.safe_left_m, (0, 180, 0)),
                (corridor.safe_right_m, (0, 180, 0)),
            ):
                pixels = self.transformer.metric_to_pixel(np.column_stack([corridor.forward_m, lateral]))
                cv2.polylines(canvas, [np.rint(pixels).astype(np.int32)], False, color, 2, cv2.LINE_AA)
        if self.transformer is not None:
            expanded = self.planner.expanded_radius()
            radius_px = int(round(expanded / max(self.transformer.m_per_px_x, 1e-6)))
            for disk in disks:
                center = tuple(np.rint(self.transformer.metric_to_pixel(np.asarray([disk.center_m]))[0]).astype(int))
                cv2.circle(canvas, center, max(1, radius_px), (0, 80, 255), 2, cv2.LINE_AA)
                cv2.circle(canvas, center, 4, (255, 255, 255), -1, cv2.LINE_AA)
                forward_offset = float(
                    self.config["physical"].get(
                        "local_forward_origin_offset_m", 0.0
                    )
                )
                base_distance = float(
                    np.hypot(
                        disk.center_m[0] + forward_offset,
                        disk.center_m[1],
                    )
                )
                cv2.putText(
                    canvas,
                    "disk {} base {:.2f}m".format(
                        disk.track_id, base_distance
                    ),
                    (center[0] + 5, center[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            if planning is not None:
                for candidate in planning.candidates:
                    pixels = self.transformer.metric_to_pixel(candidate.points_m)
                    color = (80, 80, 80) if not candidate.collision_free else (170, 100, 0)
                    cv2.polylines(canvas, [np.rint(pixels).astype(np.int32)], False, color, 1, cv2.LINE_AA)
                if planning.selected is not None:
                    pixels = self.transformer.metric_to_pixel(planning.selected.points_m)
                    cv2.polylines(canvas, [np.rint(pixels).astype(np.int32)], False, (255, 0, 255), 4, cv2.LINE_AA)
        forward_offset = float(
            self.config["physical"].get(
                "local_forward_origin_offset_m", 0.0
            )
        )
        cv2.putText(
            canvas,
            "BEV bottom is {:.2f}m ahead of base_link".format(
                forward_offset
            ),
            (10, canvas.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return canvas

    def _uncalibrated_result(self, image, masks, stamp_sec):
        boundary_started = time.perf_counter()
        zeros_to_metric = lambda points: np.zeros((len(points), 2), dtype=np.float64)
        if self.profile_boundary_detector is not None:
            boundary = self.profile_boundary_detector.detect(
                masks["yellow_confidence"],
                masks["yellow_opponent"],
                masks["yellow_roi"],
            )
        else:
            boundary = self.boundary_detector.detect(masks["yellow"])
        boundary_finished = time.perf_counter()
        raw_disks = self.disk_detector.detect(
            masks["red"], masks["white"], stamp_sec, 0.0, 0.0, zeros_to_metric,
            image_bgr=image,
        )
        raw_disks = self._apply_disk_lane_geometry(
            raw_disks, boundary, image.shape
        )
        disk_finished = time.perf_counter()
        decision = self.state_machine.force_stop("calibration_incomplete", stamp_sec)
        debug = {
            "raw_overlay": self._draw_raw_overlay(
                image,
                masks["roi"],
                masks["yellow_roi"],
                raw_disks,
                boundary,
            ),
            "yellow_mask": self._mask_bgr(masks["yellow"]),
            "yellow_score": self._mask_bgr(masks["yellow_score"]),
            "red_mask": self._mask_bgr(masks["red"]),
            "disc_overlay": self._draw_disc_overlay(image, raw_disks, boundary),
            "disc_red_mask": self._mask_bgr(masks["red"]),
            "disc_white_mask": self._mask_bgr(masks["white"]),
            "disc_candidates": self._draw_disc_candidates(image),
            "bev": image.copy(),
            "plan": image.copy(),
        }
        debug_finished = time.perf_counter()
        diagnostics = {
            "calibrated": False,
            "intrinsics_calibrated": self.undistorter is not None,
            "boundary_quality": boundary.quality,
            "left_detected": boundary.left_detected,
            "right_detected": boundary.right_detected,
            "lane_width_px": boundary.lane_width_px,
            "boundary_used_history": boundary.used_history,
            "raw_disk_count": len(raw_disks),
            "disc_candidate_count": len(self.disk_detector.last_candidates),
            "disc_rejected_candidate_count": sum(
                not bool(item.get("accepted", False))
                for item in self.disk_detector.last_candidates
            ),
            "disc_red_mask_area_px": self.disk_detector.last_red_area_px,
            "disc_white_mask_area_px": self.disk_detector.last_white_area_px,
            "disc_used_history": False,
            "boundary_processing_time_ms": (
                boundary_finished - boundary_started
            ) * 1000.0,
            "disc_processing_time_ms": (
                disk_finished - boundary_finished
            ) * 1000.0,
            "debug_render_time_ms": (
                debug_finished - disk_finished
            ) * 1000.0,
            "motion_enabled": bool(self.config["control"]["enable_motion"]),
            "desired_command": {"linear_x": 0.0, "angular_z": 0.0},
        }
        if self.profile_boundary_detector is not None:
            diagnostics.update(self.profile_boundary_detector.last_metrics)
        return PipelineResult(
            stamp_sec=stamp_sec, state=decision.state, stop_reason=decision.stop_reason,
            boundary=boundary, raw_disks=raw_disks, tracked_disks=[], corridor=None,
            planning=None, command=ControlCommand(), debug_images=debug,
            diagnostics=diagnostics,
        )

    def process(
        self,
        image_bgr: np.ndarray,
        stamp_sec: float,
        debug_image_names=None,
        run_planning: bool = True,
    ) -> PipelineResult:
        requested_debug = (
            None if debug_image_names is None else set(debug_image_names)
        )

        def wants_debug(name: str) -> bool:
            return requested_debug is None or name in requested_debug

        processing_started = time.perf_counter()
        working_image = (
            self.undistorter.undistort(image_bgr)
            if self.undistorter is not None
            else image_bgr
        )
        undistort_finished = time.perf_counter()
        masks = self.segmenter.masks(working_image)
        # Keep only a very short startup history.  It is used solely by the
        # metric far-disc fallback and is cleared when commissioned camera
        # photometry is latched or the scene is reset.
        self._bev_far_red_history.append(masks["red"].copy())
        segmentation_finished = time.perf_counter()
        if self.transformer is None:
            result = self._uncalibrated_result(working_image, masks, stamp_sec)
            result.diagnostics["undistort_time_ms"] = (
                undistort_finished - processing_started
            ) * 1000.0
            result.diagnostics["segmentation_time_ms"] = (
                segmentation_finished - undistort_finished
            ) * 1000.0
            result.diagnostics["processing_time_ms"] = (
                time.perf_counter() - processing_started
            ) * 1000.0
            return result

        need_bev_image = wants_debug("bev") or wants_debug("plan")
        bev_image = (
            self.transformer.warp_image(working_image)
            if need_bev_image
            else None
        )
        boundary: Optional[BoundaryResult] = None
        raw_boundary: Optional[BoundaryResult] = None
        corridor = None
        planning: Optional[PlanningResult] = None
        original_disks = []
        raw_disks = []
        tracked_disks = []
        processing_error = ""
        bev_fallback_disks = []
        bev_fallback_time_ms = 0.0

        try:
            raw_boundary = (
                self.profile_boundary_detector.detect(
                    masks["yellow_confidence"],
                    masks["yellow_opponent"],
                    masks["yellow_roi"],
                )
                if self.profile_boundary_detector is not None
                else self.boundary_detector.detect(masks["yellow"])
            )
            boundary = self._project_boundary_to_bev(raw_boundary)
            original_disks = self.disk_detector.detect(
                masks["red"],
                masks["white"],
                stamp_sec,
                0.0,
                0.0,
                lambda points: np.zeros((len(points), 2), dtype=np.float64),
                image_bgr=working_image,
            )
            original_disks = self._apply_disk_lane_geometry(
                original_disks, raw_boundary, working_image.shape
            )
            projected_original = [
                projected
                for projected in (
                    self._project_disk_to_bev(detection)
                    for detection in original_disks
                )
                if projected is not None
            ]
            if len(projected_original) < 2:
                bev_fallback_started = time.perf_counter()
                bev_fallback_disks = self._detect_metric_bev_far_disks(
                    masks, working_image, stamp_sec
                )
                bev_fallback_time_ms = (
                    time.perf_counter() - bev_fallback_started
                ) * 1000.0
            raw_disks = self._merge_disk_views(
                projected_original + bev_fallback_disks
            )
            tracked_disks = self.tracker.update(raw_disks, stamp_sec)
            if run_planning:
                corridor = build_safe_corridor(
                    boundary, self.transformer, self.config
                )
                planning = self.planner.plan(
                    corridor, tracked_disks, self.state_machine.locked_direction
                )
                boundary_ok = (
                    corridor.valid
                    and boundary.quality
                    >= float(self.config["boundary"]["minimum_quality"])
                    and boundary.both_missing_frames
                    < int(self.config["boundary"]["both_side_stop_frames"])
                )
                decision = self.state_machine.update(
                    stamp_sec,
                    calibration_ok=True,
                    boundary_ok=boundary_ok,
                    boundary_loss_frames=boundary.both_missing_frames,
                    planning=planning,
                    obstacles=tracked_disks,
                )
                desired = self.controller.command(
                    planning, decision.state, stamp_sec
                )
                command = (
                    desired
                    if bool(self.config["control"]["enable_motion"])
                    else ControlCommand()
                )
            else:
                decision = self.state_machine.force_stop(
                    "planning_disabled", stamp_sec
                )
                self.controller.reset()
                desired = ControlCommand()
                command = ControlCommand()
        except Exception as exc:
            processing_error = "{}: {}".format(type(exc).__name__, exc)
            decision = self.state_machine.force_stop("image_processing_error", stamp_sec)
            self.controller.reset()
            desired = ControlCommand()
            command = ControlCommand()
            if planning is None:
                planning = PlanningResult([], None, False, 0.0, 0.0, "image_processing_error")

        debug = {}
        if wants_debug("raw_overlay"):
            debug["raw_overlay"] = self._draw_raw_overlay(
                working_image,
                masks["roi"],
                masks["yellow_roi"],
                original_disks,
                raw_boundary,
            )
        if wants_debug("yellow_mask"):
            debug["yellow_mask"] = self._mask_bgr(
                self.transformer.warp_mask(masks["yellow"])
            )
        if wants_debug("yellow_score"):
            debug["yellow_score"] = self._mask_bgr(
                self.transformer.warp_mask(masks["yellow_score"])
            )
        if wants_debug("red_mask"):
            debug["red_mask"] = self._mask_bgr(
                self.transformer.warp_mask(masks["red"])
            )
        if wants_debug("disc_overlay"):
            debug["disc_overlay"] = self._draw_disc_overlay(
                working_image, original_disks, raw_boundary
            )
        if wants_debug("disc_red_mask"):
            debug["disc_red_mask"] = self._mask_bgr(masks["red"])
        if wants_debug("disc_white_mask"):
            debug["disc_white_mask"] = self._mask_bgr(masks["white"])
        if wants_debug("disc_candidates"):
            debug["disc_candidates"] = self._draw_disc_candidates(working_image)
        if wants_debug("bev"):
            debug["bev"] = bev_image
        if wants_debug("plan"):
            debug["plan"] = self._draw_bev_plan(
                bev_image, boundary, corridor, planning, tracked_disks
            )
        diagnostics = {
            "calibrated": True,
            "intrinsics_calibrated": True,
            "boundary_quality": 0.0 if boundary is None else boundary.quality,
            "left_detected": False if boundary is None else boundary.left_detected,
            "right_detected": False if boundary is None else boundary.right_detected,
            "raw_disk_count": len(raw_disks),
            "confirmed_disk_count": len(tracked_disks),
            "safe_path": False if planning is None else planning.safe,
            "selected_path": None if planning is None or planning.selected is None else planning.selected.name,
            "left_space_m": 0.0 if planning is None else planning.left_space_m,
            "right_space_m": 0.0 if planning is None else planning.right_space_m,
            "motion_enabled": bool(self.config["control"]["enable_motion"]),
            "desired_command": {"linear_x": desired.linear_x, "angular_z": desired.angular_z},
            "output_command": {"linear_x": command.linear_x, "angular_z": command.angular_z},
            "processing_error": processing_error,
            "boundary_source": "projected_raw_profile",
            "boundary_used_history": False if boundary is None else bool(boundary.used_history),
            "disc_used_history": False,
            "bev_far_fallback_disk_count": len(bev_fallback_disks),
            "bev_far_fallback_time_ms": bev_fallback_time_ms,
            "planning_enabled": bool(run_planning),
            "local_forward_origin_offset_m": float(
                self.config["physical"].get(
                    "local_forward_origin_offset_m", 0.0
                )
            ),
            "local_lateral_origin_offset_m": float(
                self.config["physical"].get(
                    "local_lateral_origin_offset_m", 0.0
                )
            ),
        }
        if self.profile_boundary_detector is not None:
            diagnostics.update(self.profile_boundary_detector.last_metrics)
        result = PipelineResult(
            stamp_sec=stamp_sec, state=decision.state, stop_reason=decision.stop_reason,
            boundary=boundary, raw_disks=raw_disks, tracked_disks=tracked_disks,
            corridor=corridor, planning=planning, command=command,
            debug_images=debug, diagnostics=diagnostics,
        )
        result.diagnostics["processing_time_ms"] = (
            time.perf_counter() - processing_started
        ) * 1000.0
        return result

    def camera_timeout(self, now_sec: float) -> PipelineResult:
        decision = self.state_machine.force_stop("camera_timeout", now_sec)
        self.controller.reset()
        return PipelineResult(
            stamp_sec=now_sec, state=decision.state, stop_reason=decision.stop_reason,
            boundary=None, raw_disks=[], tracked_disks=[], corridor=None, planning=None,
            command=ControlCommand(), debug_images={},
            diagnostics={"calibrated": self.calibrated, "camera_timeout": True, "motion_enabled": bool(self.config["control"]["enable_motion"])},
        )
