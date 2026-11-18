"""OpenCV yellow-boundary and red-white disk perception."""

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .model import BoundaryResult, DiskDetection


def circularity(area: float, perimeter: float) -> float:
    if perimeter <= 0.0:
        return 0.0
    return float(4.0 * math.pi * area / (perimeter * perimeter))


def merge_red_hsv(hsv: np.ndarray, config: Dict) -> np.ndarray:
    cfg = config["red_hsv"]
    first = cv2.inRange(hsv, np.asarray(cfg["lower_1"], np.uint8), np.asarray(cfg["upper_1"], np.uint8))
    second = cv2.inRange(hsv, np.asarray(cfg["lower_2"], np.uint8), np.asarray(cfg["upper_2"], np.uint8))
    return cv2.bitwise_or(first, second)


def _odd_kernel(value: int) -> np.ndarray:
    size = max(1, int(value))
    if size % 2 == 0:
        size += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _clean_mask(mask: np.ndarray, open_size: int, close_size: int, open_iter: int, close_iter: int) -> np.ndarray:
    result = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _odd_kernel(open_size), iterations=max(0, int(open_iter)))
    return cv2.morphologyEx(result, cv2.MORPH_CLOSE, _odd_kernel(close_size), iterations=max(0, int(close_iter)))


def filter_yellow_orientation(mask: np.ndarray, config: Dict) -> np.ndarray:
    """Keep locally vertical/diagonal yellow strokes and reject cross-lines.

    A local structure tensor is estimated from foreground pixel coordinates.
    This works on curved boundaries because every small section is classified
    independently instead of assigning one direction to a whole contour.
    """
    cfg = config.get("yellow_orientation_filter", {})
    if not bool(cfg.get("enabled", False)) or not np.any(mask):
        return mask

    window = max(5, int(cfg.get("window_size_px", 25)))
    if window % 2 == 0:
        window += 1
    minimum_angle = float(cfg.get("minimum_nonhorizontal_angle_deg", 10.0))
    minimum_anisotropy = float(cfg.get("minimum_anisotropy", 0.20))
    minimum_support = max(3, int(cfg.get("minimum_support_pixels", 18)))

    foreground = (mask > 0).astype(np.float64)
    height, width = mask.shape[:2]
    yy, xx = np.mgrid[:height, :width]
    xx = xx.astype(np.float64)
    yy = yy.astype(np.float64)
    kernel = (window, window)

    def local_sum(values):
        return cv2.boxFilter(
            values,
            ddepth=cv2.CV_64F,
            ksize=kernel,
            normalize=False,
            borderType=cv2.BORDER_CONSTANT,
        )

    count = local_sum(foreground)
    safe_count = np.maximum(count, 1.0)
    mean_x = local_sum(foreground * xx) / safe_count
    mean_y = local_sum(foreground * yy) / safe_count
    covariance_xx = local_sum(foreground * xx * xx) / safe_count - mean_x * mean_x
    covariance_yy = local_sum(foreground * yy * yy) / safe_count - mean_y * mean_y
    covariance_xy = local_sum(foreground * xx * yy) / safe_count - mean_x * mean_y

    separation = np.sqrt(
        np.maximum(
            0.0,
            (covariance_xx - covariance_yy) ** 2 + 4.0 * covariance_xy ** 2,
        )
    )
    anisotropy = separation / np.maximum(covariance_xx + covariance_yy, 1e-6)
    angle = 0.5 * np.degrees(
        np.arctan2(2.0 * covariance_xy, covariance_xx - covariance_yy)
    )
    angle_from_horizontal = np.abs(angle)

    accepted = (
        (foreground > 0.0)
        & (count >= minimum_support)
        & (anisotropy >= minimum_anisotropy)
        & (angle_from_horizontal >= minimum_angle)
    )
    support = accepted.astype(np.uint8) * 255
    recovery_size = max(1, int(cfg.get("recovery_dilate_px", 3)))
    if recovery_size > 1:
        support = cv2.dilate(support, _odd_kernel(recovery_size), iterations=1)
    return cv2.bitwise_and(mask, support)


def configured_detection_polygon(
    config: Dict, width: int, height: int
) -> Optional[np.ndarray]:
    """Return the raw-image detection polygon, independent of BEV fitting."""
    camera = config.get("camera", {})
    bev = config.get("bev", {})
    if not bool(camera.get("use_bev_polygon_for_detection", False)):
        return None
    points = np.asarray(bev.get("source_points", []), dtype=np.float32)
    if points.shape != (4, 2):
        return None
    if bool(bev.get("extend_to_image_bottom", False)):
        points = np.asarray(
            [
                points[3],
                points[2],
                points[1],
                [float(width - 1), float(height - 1)],
                [0.0, float(height - 1)],
                points[0],
            ],
            dtype=np.float32,
        )
    points[:, 0] = np.clip(points[:, 0], 0.0, float(width - 1))
    points[:, 1] = np.clip(points[:, 1], 0.0, float(height - 1))
    return np.rint(points).astype(np.int32)


def normalized_polygon_mask(
    height: int, width: int, polygon
) -> np.ndarray:
    """Rasterize normalized [x, y] polygon coordinates into an image mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    if not polygon:
        return mask
    points = np.asarray(polygon, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
        return mask
    points[:, 0] = np.clip(points[:, 0], 0.0, 1.0) * max(0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0.0, 1.0) * max(0, height - 1)
    cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 255)
    return mask


class ColorSegmenter:
    def __init__(self, config: Dict):
        self.config = config

    @staticmethod
    def _sigmoid(value):
        array = np.asarray(value, dtype=np.float32)
        return 1.0 / (1.0 + np.exp(-np.clip(array, -20.0, 20.0)))

    def _yellow_confidence_v2(self, image_bgr, hsv, ycc, yellow_roi, opponent):
        """Continuous Cr-Cb confidence with green/specular penalties."""
        cfg = self.config["yellow_confidence_v2"]
        luminance, cr, cb = cv2.split(ycc.astype(np.float32))
        b, g, r = cv2.split(image_bgr.astype(np.float32))
        rgb_sum = r + g + b + 1.0
        green_excess = np.maximum(g - r, 0.0) / rgb_sum
        chroma = np.abs(cr - 128.0) + np.abs(cb - 128.0)

        p_absolute = self._sigmoid(
            (opponent - float(cfg["absolute_center"]))
            / max(float(cfg["absolute_scale"]), 1e-3)
        )
        p_green = self._sigmoid(
            (green_excess - float(cfg["green_excess_center"]))
            / max(float(cfg["green_excess_scale"]), 1e-4)
        )
        p_specular = self._sigmoid(
            (luminance - float(cfg["specular_luminance_center"]))
            / max(float(cfg["specular_luminance_scale"]), 1e-3)
        ) * self._sigmoid(
            (float(cfg["specular_chroma_center"]) - chroma)
            / max(float(cfg["specular_chroma_scale"]), 1e-3)
        )
        confidence = (
            p_absolute
            * (1.0 - float(cfg["green_penalty"]) * p_green)
            * (1.0 - float(cfg["specular_penalty"]) * p_specular)
        )

        hue = hsv[:, :, 0].astype(np.float32)
        saturation = hsv[:, :, 1].astype(np.float32)
        hue_low = float(cfg["hue_low"])
        hue_peak = float(cfg["hue_peak"])
        hue_high = float(cfg["hue_high"])
        hue_evidence = np.clip(
            np.minimum(
                (hue - hue_low) / max(hue_peak - hue_low, 1e-3),
                (hue_high - hue) / max(hue_high - hue_peak, 1e-3),
            ),
            0.0,
            1.0,
        )
        saturation_gate = self._sigmoid(
            (saturation - float(cfg["hue_bonus_min_saturation"]))
            / max(float(cfg["hue_bonus_saturation_scale"]), 1e-3)
        )
        bonus = float(cfg["hue_bonus_weight"]) * hue_evidence * saturation_gate
        confidence = confidence + (1.0 - confidence) * bonus
        confidence[yellow_roi == 0] = 0.0
        return np.clip(confidence, 0.0, 1.0).astype(np.float32)

    def _yellow_score(self, image_bgr, hsv, yellow_roi):
        cfg = self.config.get("yellow_scoring", {})
        if not bool(cfg.get("enabled", False)):
            yellow_cfg = self.config["yellow_hsv"]
            mask = cv2.inRange(
                hsv,
                np.asarray(yellow_cfg["lower"], np.uint8),
                np.asarray(yellow_cfg["upper"], np.uint8),
            )
            return mask.astype(np.float32) / 255.0

        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
        hue = hsv[:, :, 0].astype(np.float32)
        saturation = hsv[:, :, 1].astype(np.float32)
        value = hsv[:, :, 2].astype(np.float32)
        lab_b = lab[:, :, 2].astype(np.float32)
        cb = ycrcb[:, :, 2].astype(np.float32)

        hue_low = float(cfg["hue_low"])
        hue_peak = float(cfg["hue_peak"])
        hue_high = float(cfg["hue_high"])
        rising = (hue - hue_low) / max(1e-6, hue_peak - hue_low)
        falling = (hue_high - hue) / max(1e-6, hue_high - hue_peak)
        hue_evidence = np.clip(np.minimum(rising, falling), 0.0, 1.0)
        saturation_evidence = np.clip(
            saturation / max(1.0, float(cfg["saturation_full_scale"])), 0.0, 1.0
        )
        hsv_evidence = hue_evidence * saturation_evidence

        score = np.zeros(hue.shape, dtype=np.float32)
        roi_bool = yellow_roi > 0
        valid_rows = np.flatnonzero(np.any(roi_bool, axis=1))
        if valid_rows.size == 0:
            return score
        band_height = max(1, int(cfg["band_height_px"]))
        first_row = int(valid_rows[0])
        last_row = int(valid_rows[-1]) + 1
        ground_max_sat = float(cfg["ground_max_saturation"])
        ground_min_value = float(cfg["ground_min_value"])

        for y0 in range(first_row, last_row, band_height):
            y1 = min(last_row, y0 + band_height)
            band_roi = roi_bool[y0:y1]
            ground = (
                band_roi
                & (saturation[y0:y1] <= ground_max_sat)
                & (value[y0:y1] >= ground_min_value)
            )
            if int(np.count_nonzero(ground)) < 64:
                ground = band_roi & (value[y0:y1] >= ground_min_value)
            if int(np.count_nonzero(ground)) < 16:
                continue
            reference_b = lab_b[y0:y1][ground]
            reference_cb = cb[y0:y1][ground]
            median_b = float(np.median(reference_b))
            median_cb = float(np.median(reference_cb))
            mad_b = max(1.0, float(np.median(np.abs(reference_b - median_b))))
            mad_cb = max(1.0, float(np.median(np.abs(reference_cb - median_cb))))
            b_floor = max(float(cfg["lab_b_min_delta"]), float(cfg["lab_b_mad_k"]) * mad_b)
            cb_floor = max(float(cfg["cb_min_delta"]), float(cfg["cb_mad_k"]) * mad_cb)
            b_full = max(b_floor + 1.0, float(cfg["lab_b_full_delta"]))
            cb_full = max(cb_floor + 1.0, float(cfg["cb_full_delta"]))
            b_evidence = np.clip(
                (lab_b[y0:y1] - median_b - b_floor) / (b_full - b_floor), 0.0, 1.0
            )
            cb_evidence = np.clip(
                (median_cb - cb[y0:y1] - cb_floor) / (cb_full - cb_floor), 0.0, 1.0
            )
            band_score = (
                float(cfg["hsv_weight"]) * hsv_evidence[y0:y1]
                + float(cfg["lab_b_weight"]) * b_evidence
                + float(cfg["cb_weight"]) * cb_evidence
            )
            gate = (
                band_roi
                & (hue_evidence[y0:y1] >= float(cfg["minimum_hue_evidence"]))
                & (value[y0:y1] >= ground_min_value)
            )
            score[y0:y1][gate] = band_score[gate]
        return np.clip(score, 0.0, 1.0)

    def masks(self, image_bgr: np.ndarray) -> Dict[str, np.ndarray]:
        white_cfg = self.config["white_hsv"]
        morph = self.config["morphology"]
        height, width = image_bgr.shape[:2]
        camera = self.config["camera"]
        x0 = int(round(width * float(camera["roi_x_start_ratio"])))
        x1 = int(round(width * float(camera["roi_x_end_ratio"])))
        y0 = int(round(height * float(camera["roi_y_start_ratio"])))
        y1 = int(round(height * float(camera["roi_y_end_ratio"])))
        roi = np.zeros((height, width), dtype=np.uint8)
        roi[max(0, y0) : min(height, y1), max(0, x0) : min(width, x1)] = 255

        normalized_roi = camera.get("roi_polygon_normalized")
        if normalized_roi:
            roi = cv2.bitwise_and(
                roi,
                normalized_polygon_mask(height, width, normalized_roi),
            )

        detection_polygon = configured_detection_polygon(
            self.config, width, height
        )
        polygon_roi = None
        if detection_polygon is not None:
            polygon_roi = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(polygon_roi, [detection_polygon], 255)
            roi = cv2.bitwise_and(roi, polygon_roi)

        yellow_x0 = int(round(width * float(camera.get("yellow_roi_x_start_ratio", camera["roi_x_start_ratio"]))))
        yellow_x1 = int(round(width * float(camera.get("yellow_roi_x_end_ratio", camera["roi_x_end_ratio"]))))
        yellow_y0 = int(round(height * float(camera.get("yellow_roi_y_start_ratio", camera["roi_y_start_ratio"]))))
        yellow_y1 = int(round(height * float(camera.get("yellow_roi_y_end_ratio", camera["roi_y_end_ratio"]))))
        yellow_roi = np.zeros((height, width), dtype=np.uint8)
        yellow_roi[
            max(0, yellow_y0) : min(height, yellow_y1),
            max(0, yellow_x0) : min(width, yellow_x1),
        ] = 255
        normalized_yellow_roi = camera.get("yellow_roi_polygon_normalized")
        if normalized_yellow_roi:
            yellow_roi = cv2.bitwise_and(
                yellow_roi,
                normalized_polygon_mask(height, width, normalized_yellow_roi),
            )
        if polygon_roi is not None:
            yellow_roi = cv2.bitwise_and(yellow_roi, polygon_roi)

        # Colour conversion and all dense floating-point evidence are limited
        # to the configured detection region.  The upper half of this camera
        # image can never contribute a line or disk, so processing it only
        # wastes the 10 Hz frame budget.
        active_roi = cv2.bitwise_or(roi, yellow_roi)
        active_y, active_x = np.nonzero(active_roi)
        red = np.zeros((height, width), dtype=np.uint8)
        white = np.zeros((height, width), dtype=np.uint8)
        yellow = np.zeros((height, width), dtype=np.uint8)
        yellow_score = np.zeros((height, width), dtype=np.float32)
        yellow_opponent = np.zeros((height, width), dtype=np.float32)
        if active_y.size == 0:
            return {
                "yellow": yellow,
                "yellow_score": yellow,
                "yellow_confidence": yellow_score,
                "yellow_opponent": yellow_opponent,
                "red": red,
                "white": white,
                "roi": roi,
                "yellow_roi": yellow_roi,
            }

        crop_y0 = int(np.min(active_y))
        crop_y1 = int(np.max(active_y)) + 1
        crop_x0 = int(np.min(active_x))
        crop_x1 = int(np.max(active_x)) + 1
        image_crop = image_bgr[crop_y0:crop_y1, crop_x0:crop_x1]
        roi_crop = roi[crop_y0:crop_y1, crop_x0:crop_x1]
        yellow_roi_crop = yellow_roi[crop_y0:crop_y1, crop_x0:crop_x1]
        hsv = cv2.cvtColor(image_crop, cv2.COLOR_BGR2HSV)
        ycc = cv2.cvtColor(image_crop, cv2.COLOR_BGR2YCrCb)
        red_crop = merge_red_hsv(hsv, self.config)
        white_crop = cv2.inRange(
            hsv,
            np.asarray(white_cfg["lower"], np.uint8),
            np.asarray(white_cfg["upper"], np.uint8),
        )
        opponent_crop = (
            ycc[:, :, 1].astype(np.float32) - ycc[:, :, 2].astype(np.float32)
        )
        confidence_v2 = self.config.get("yellow_confidence_v2", {})
        if bool(confidence_v2.get("enabled", False)):
            score_crop = self._yellow_confidence_v2(
                image_crop, hsv, ycc, yellow_roi_crop, opponent_crop
            )
            threshold = float(confidence_v2.get("display_threshold", 0.4))
        else:
            score_crop = self._yellow_score(image_crop, hsv, yellow_roi_crop)
            threshold = float(self.config.get("yellow_scoring", {}).get("score_threshold", 0.5))
        yellow_crop = (
            (score_crop >= threshold) & (yellow_roi_crop > 0)
        ).astype(np.uint8) * 255
        red_crop = cv2.bitwise_and(red_crop, roi_crop)
        white_crop = cv2.bitwise_and(white_crop, roi_crop)
        yellow_crop = _clean_mask(
            yellow_crop,
            morph["yellow_open_kernel"], morph["yellow_close_kernel"],
            morph["yellow_open_iterations"], morph["yellow_close_iterations"],
        )
        red_crop = _clean_mask(
            red_crop,
            morph["red_open_kernel"], morph["red_close_kernel"],
            morph["red_open_iterations"], morph["red_close_iterations"],
        )
        yellow_crop = filter_yellow_orientation(yellow_crop, self.config)
        red_exclusion = int(
            self.config.get("yellow_orientation_filter", {}).get(
                "red_exclusion_dilate_px", 0
            )
        )
        if red_exclusion > 0 and np.any(red_crop):
            red_neighborhood = cv2.dilate(
                red_crop, _odd_kernel(red_exclusion), iterations=1
            )
            yellow_crop = cv2.bitwise_and(
                yellow_crop, cv2.bitwise_not(red_neighborhood)
            )
            score_crop = score_crop.copy()
            score_crop[red_neighborhood > 0] = 0.0
        white_crop = _clean_mask(
            white_crop,
            morph["white_open_kernel"], morph["white_close_kernel"],
            morph["white_open_iterations"], morph["white_close_iterations"],
        )
        crop = np.s_[crop_y0:crop_y1, crop_x0:crop_x1]
        red[crop] = red_crop
        white[crop] = white_crop
        yellow[crop] = yellow_crop
        yellow_score[crop] = score_crop
        yellow_opponent[crop] = opponent_crop
        return {
            "yellow": yellow,
            "yellow_score": np.rint(yellow_score * 255.0).astype(np.uint8),
            "yellow_confidence": yellow_score.astype(np.float32),
            "yellow_opponent": yellow_opponent.astype(np.float32),
            "red": red,
            "white": white,
            "roi": roi,
            "yellow_roi": yellow_roi,
        }


class BoundaryDetector:
    def __init__(self, config: Dict):
        self.cfg = config["boundary"]
        self.previous: Optional[BoundaryResult] = None
        self.lane_width_px = float(self.cfg["initial_lane_width_px"])
        self.both_missing_frames = 0

    @staticmethod
    def _clusters(xs: np.ndarray, minimum_gap: int) -> List[np.ndarray]:
        if xs.size == 0:
            return []
        ordered = np.sort(np.unique(xs))
        cuts = np.where(np.diff(ordered) > minimum_gap)[0] + 1
        return [cluster for cluster in np.split(ordered, cuts) if cluster.size]

    def _robust_fit(self, rows: np.ndarray, values: np.ndarray, sample_rows: np.ndarray) -> Optional[np.ndarray]:
        degree = int(self.cfg["polynomial_degree"])
        minimum = max(int(self.cfg["minimum_fit_points"]), degree + 1)
        keep = np.isfinite(values)
        if int(np.count_nonzero(keep)) < minimum:
            return None
        y = rows[keep].astype(np.float64)
        x = values[keep].astype(np.float64)
        coeff = None
        for _ in range(max(1, int(self.cfg["robust_fit_iterations"]))):
            if x.size < minimum:
                return None
            coeff = np.polyfit(y, x, degree)
            residual = np.abs(x - np.polyval(coeff, y))
            new_keep = residual <= float(self.cfg["fit_outlier_px"])
            if np.all(new_keep):
                break
            x, y = x[new_keep], y[new_keep]
        return None if coeff is None else np.polyval(coeff, sample_rows)

    def detect(self, yellow_mask: np.ndarray) -> BoundaryResult:
        height, width = yellow_mask.shape[:2]
        left_only = (
            str(self.cfg.get("detection_mode", "both")).strip().lower()
            == "left_only"
        )
        step = max(1, int(self.cfg["row_step_px"]))
        rows = np.arange(height - 1, -1, -step, dtype=np.int32)
        half_band = max(0, int(self.cfg["row_band_half_height_px"]))
        minimum_pixels = max(1, int(self.cfg["minimum_row_pixels"]))
        gap = max(1, int(self.cfg["minimum_boundary_gap_px"]))

        left_raw = np.full(rows.shape, np.nan, dtype=np.float64)
        right_raw = np.full(rows.shape, np.nan, dtype=np.float64)
        reference_center = width * 0.5
        if (
            bool(self.cfg.get("use_previous_reference_center", True))
            and self.previous is not None
            and self.previous.center_px.size
        ):
            previous_observed = (
                (self.previous.rows_px >= self.previous.observed_min_row_px)
                & (self.previous.rows_px <= self.previous.observed_max_row_px)
                & np.isfinite(self.previous.center_px)
            )
            if np.any(previous_observed):
                reference_center = float(np.median(self.previous.center_px[previous_observed]))

        max_jump = float(self.cfg["row_tracking_max_jump_px"])
        max_gap_rows = int(self.cfg["row_tracking_max_gap_rows"])
        velocity_alpha = float(self.cfg["row_tracking_velocity_alpha"])
        left_track = {"value": None, "velocity": 0.0, "gaps": 0}
        right_track = {"value": None, "velocity": 0.0, "gaps": 0}
        perspective_assignment = (
            str(self.cfg.get("row_assignment_mode", "tracked")).strip().lower()
            == "perspective_sides"
        )
        center_exclusion = float(
            self.cfg.get("perspective_center_exclusion_px", gap)
        )

        def assign(track, value):
            if track["value"] is not None:
                steps = track["gaps"] + 1
                measured_velocity = (value - track["value"]) / float(steps)
                track["velocity"] = (
                    (1.0 - velocity_alpha) * track["velocity"]
                    + velocity_alpha * measured_velocity
                )
            track["value"] = value
            track["gaps"] = 0

        def match(track, centers, excluded):
            if track["value"] is None:
                return None
            predicted = track["value"] + track["velocity"] * float(track["gaps"] + 1)
            candidates = [
                (abs(value - predicted), index, value)
                for index, value in enumerate(centers)
                if index not in excluded
            ]
            if not candidates:
                return None
            error, index, value = min(candidates)
            if error > max_jump:
                return None
            assign(track, value)
            return index

        for index, row in enumerate(rows):
            y0, y1 = max(0, row - half_band), min(height, row + half_band + 1)
            xs = np.nonzero(yellow_mask[y0:y1])[1]
            clusters = [c for c in self._clusters(xs, gap) if c.size >= minimum_pixels]
            centers = [float(np.median(cluster)) for cluster in clusters]

            if perspective_assignment:
                left_candidates = [
                    value for value in centers
                    if value < reference_center - 0.5 * center_exclusion
                ]
                right_candidates = [
                    value for value in centers
                    if value > reference_center + 0.5 * center_exclusion
                ]
                if left_candidates:
                    left_raw[index] = max(left_candidates)
                if right_candidates:
                    right_raw[index] = min(right_candidates)
                continue

            used = set()

            left_index = match(left_track, centers, used)
            if left_index is not None:
                used.add(left_index)
                left_raw[index] = centers[left_index]
            right_index = match(right_track, centers, used)
            if right_index is not None:
                used.add(right_index)
                right_raw[index] = centers[right_index]

            if left_track["value"] is None and right_track["value"] is None:
                pairs = []
                for left_candidate, left_value in enumerate(centers[:-1]):
                    for right_candidate in range(left_candidate + 1, len(centers)):
                        right_value = centers[right_candidate]
                        if right_value - left_value < gap:
                            continue
                        center_error = abs(0.5 * (left_value + right_value) - reference_center)
                        pairs.append((center_error, left_candidate, right_candidate))
                if pairs:
                    _, left_index, right_index = min(pairs)
                    assign(left_track, centers[left_index])
                    assign(right_track, centers[right_index])
                    used.update((left_index, right_index))
                    left_raw[index] = centers[left_index]
                    right_raw[index] = centers[right_index]
                elif centers:
                    nearest = min(range(len(centers)), key=lambda item: abs(centers[item] - reference_center))
                    if centers[nearest] < reference_center:
                        assign(left_track, centers[nearest])
                        left_raw[index] = centers[nearest]
                    else:
                        assign(right_track, centers[nearest])
                        right_raw[index] = centers[nearest]
                    used.add(nearest)

            if left_track["value"] is None and right_track["value"] is not None:
                candidates = [
                    item for item in range(len(centers))
                    if item not in used and centers[item] < right_track["value"] - gap
                ]
                if candidates:
                    left_index = min(
                        candidates,
                        key=lambda item: abs(
                            0.5 * (centers[item] + right_track["value"]) - reference_center
                        ),
                    )
                    assign(left_track, centers[left_index])
                    used.add(left_index)
                    left_raw[index] = centers[left_index]

            if right_track["value"] is None and left_track["value"] is not None:
                candidates = [
                    item for item in range(len(centers))
                    if item not in used and centers[item] > left_track["value"] + gap
                ]
                if candidates:
                    right_index = min(
                        candidates,
                        key=lambda item: abs(
                            0.5 * (left_track["value"] + centers[item]) - reference_center
                        ),
                    )
                    assign(right_track, centers[right_index])
                    used.add(right_index)
                    right_raw[index] = centers[right_index]

            for track, matched in (
                (left_track, np.isfinite(left_raw[index])),
                (right_track, np.isfinite(right_raw[index])),
            ):
                if track["value"] is not None and not matched:
                    track["gaps"] += 1
                    if track["gaps"] > max_gap_rows:
                        track.update(value=None, velocity=0.0, gaps=0)

        if left_only:
            # The right image half is deliberately outside the trusted yellow
            # ROI. Discard any fallback-detector fragment that happened to be
            # classified on the right; metric reconstruction happens later.
            right_raw[:] = np.nan

        left_count = int(np.count_nonzero(np.isfinite(left_raw)))
        right_count = int(np.count_nonzero(np.isfinite(right_raw)))
        left_detected = left_count >= int(self.cfg["minimum_fit_points"])
        right_detected = right_count >= int(self.cfg["minimum_fit_points"])

        paired = np.isfinite(left_raw) & np.isfinite(right_raw)
        if not left_only and np.any(paired):
            observed = float(np.median(right_raw[paired] - left_raw[paired]))
            if float(self.cfg["minimum_lane_width_px"]) <= observed <= float(self.cfg["maximum_lane_width_px"]):
                alpha = float(self.cfg["lane_width_alpha"])
                self.lane_width_px = observed if self.lane_width_px <= 0.0 else (1.0 - alpha) * self.lane_width_px + alpha * observed

        left_fit = self._robust_fit(rows, left_raw, rows) if left_detected else None
        right_fit = self._robust_fit(rows, right_raw, rows) if right_detected else None
        left_detected = left_fit is not None
        right_detected = right_fit is not None
        if bool(self.cfg.get("enforce_perspective_side_direction", False)):
            slope_tolerance = float(
                self.cfg.get("perspective_direction_slope_tolerance", 0.15)
            )

            def fitted_slope(fitted_values, raw_values):
                valid = np.isfinite(raw_values) & np.isfinite(fitted_values)
                if int(np.count_nonzero(valid)) < 2:
                    return 0.0
                return float(
                    np.polyfit(
                        rows[valid].astype(np.float64),
                        fitted_values[valid].astype(np.float64),
                        1,
                    )[0]
                )

            if left_fit is not None and fitted_slope(left_fit, left_raw) > slope_tolerance:
                left_fit = None
                left_detected = False
            if right_fit is not None and fitted_slope(right_fit, right_raw) < -slope_tolerance:
                right_fit = None
                right_detected = False
        used_history = False

        if not left_only:
            if left_fit is not None and right_fit is None and self.lane_width_px > 0.0:
                right_fit = left_fit + self.lane_width_px
                used_history = True
            elif right_fit is not None and left_fit is None and self.lane_width_px > 0.0:
                left_fit = right_fit - self.lane_width_px
                used_history = True

        if left_only and left_fit is not None:
            self.both_missing_frames = 0
            right_fit = np.full(rows.shape, np.nan, dtype=np.float64)
            right_detected = False
        elif left_fit is None or right_fit is None:
            self.both_missing_frames += 1
            if (
                not left_only
                and self.previous is not None
                and self.both_missing_frames <= int(self.cfg["single_side_hold_frames"])
            ):
                left_fit = self.previous.left_px.copy()
                right_fit = self.previous.right_px.copy()
                used_history = True
            else:
                left_fit = np.full(rows.shape, np.nan, dtype=np.float64)
                right_fit = np.full(rows.shape, np.nan, dtype=np.float64)
        else:
            self.both_missing_frames = 0

        left_fit = np.asarray(left_fit, dtype=np.float64)
        right_fit = np.asarray(right_fit, dtype=np.float64)
        if (
            not left_only
            and self.previous is not None
            and self.previous.rows_px.shape == rows.shape
            and np.all(np.isfinite(left_fit))
        ):
            alpha = float(self.cfg["smoothing_alpha"])
            left_fit = alpha * left_fit + (1.0 - alpha) * self.previous.left_px
            right_fit = alpha * right_fit + (1.0 - alpha) * self.previous.right_px

        left_fit = np.clip(left_fit, 0.0, width - 1.0)
        right_fit = np.clip(right_fit, 0.0, width - 1.0)
        center = (
            np.full(rows.shape, np.nan, dtype=np.float64)
            if left_only
            else 0.5 * (left_fit + right_fit)
        )
        observed_rows = rows[np.isfinite(left_raw) | np.isfinite(right_raw)]
        observed_min_row = int(np.min(observed_rows)) if observed_rows.size else -1
        observed_max_row = int(np.max(observed_rows)) if observed_rows.size else -1
        detected_rows = (
            left_count
            if left_only and left_detected
            else (
                min(left_count, right_count)
                if left_detected and right_detected
                else max(left_count, right_count) * 0.5
            )
        )
        quality = min(1.0, float(detected_rows) / max(1.0, float(rows.size)))
        if used_history:
            quality *= 0.65

        result = BoundaryResult(
            rows_px=rows, left_px=left_fit, right_px=right_fit, center_px=center,
            left_detected=left_detected, right_detected=right_detected,
            used_history=used_history, both_missing_frames=self.both_missing_frames,
            lane_width_px=self.lane_width_px, quality=quality,
            observed_min_row_px=observed_min_row,
            observed_max_row_px=observed_max_row,
        )
        if (
            not left_only
            and np.all(np.isfinite(left_fit))
            and np.all(np.isfinite(right_fit))
        ):
            self.previous = result
        return result


class DiskDetector:
    def __init__(self, config: Dict):
        self.cfg = config["disk_detection"]
        self.last_candidates: List[Dict[str, object]] = []
        self.last_red_area_px = 0
        self.last_white_area_px = 0

    @staticmethod
    def _mask_centroid(mask: np.ndarray, slices) -> Optional[Tuple[float, float]]:
        points = np.argwhere(mask)
        if points.size == 0:
            return None
        y_slice, x_slice = slices
        return (
            float(np.mean(points[:, 1]) + int(x_slice.start)),
            float(np.mean(points[:, 0]) + int(y_slice.start)),
        )

    def _record_rejection(self, record: Dict[str, object], reason: str) -> None:
        record["accepted"] = False
        record["reject_reason"] = str(reason)
        self.last_candidates.append(record)

    @staticmethod
    def _ellipse_regions(shape: Tuple[int, int], ellipse, inner_ratio: float, outer_ratio: float, center_ratio: float):
        height, width = shape
        (cx, cy), (major, minor), angle_deg = ellipse
        if major <= 0.0 or minor <= 0.0:
            empty = np.zeros(shape, dtype=np.uint8)
            return empty, empty
        yy, xx = np.indices((height, width), dtype=np.float32)
        theta = math.radians(angle_deg)
        dx, dy = xx - cx, yy - cy
        xr = dx * math.cos(theta) + dy * math.sin(theta)
        yr = -dx * math.sin(theta) + dy * math.cos(theta)
        radius = np.sqrt((xr / (major * 0.5)) ** 2 + (yr / (minor * 0.5)) ** 2)
        annulus = ((radius >= inner_ratio) & (radius <= outer_ratio)).astype(np.uint8)
        center = (radius <= center_ratio).astype(np.uint8)
        return annulus, center

    @staticmethod
    def _local_ellipse_regions(shape, ellipse, inner_ratio, outer_ratio, center_ratio):
        height, width = shape
        (cx, cy), (axis_a, axis_b), angle_deg = ellipse
        geometry = np.asarray(
            [cx, cy, axis_a, axis_b, angle_deg], dtype=np.float64
        )
        if not np.all(np.isfinite(geometry)) or axis_a <= 0.0 or axis_b <= 0.0:
            return None
        theta = math.radians(angle_deg)
        cos_theta, sin_theta = math.cos(theta), math.sin(theta)
        radius_x = 0.5 * outer_ratio * math.sqrt(
            (axis_a * cos_theta) ** 2 + (axis_b * sin_theta) ** 2
        )
        radius_y = 0.5 * outer_ratio * math.sqrt(
            (axis_a * sin_theta) ** 2 + (axis_b * cos_theta) ** 2
        )
        x0 = max(0, int(math.floor(cx - radius_x - 2.0)))
        x1 = min(width, int(math.ceil(cx + radius_x + 2.0)) + 1)
        y0 = max(0, int(math.floor(cy - radius_y - 2.0)))
        y1 = min(height, int(math.ceil(cy + radius_y + 2.0)) + 1)
        if x0 >= x1 or y0 >= y1:
            return None
        yy, xx = np.indices((y1 - y0, x1 - x0), dtype=np.float32)
        dx, dy = xx + x0 - cx, yy + y0 - cy
        xr = dx * cos_theta + dy * sin_theta
        yr = -dx * sin_theta + dy * cos_theta
        normalized_x = xr / max(axis_a * 0.5, 1e-6)
        normalized_y = yr / max(axis_b * 0.5, 1e-6)
        radius = np.sqrt(normalized_x ** 2 + normalized_y ** 2)
        angular = (np.arctan2(normalized_y, normalized_x) + 2.0 * math.pi) % (2.0 * math.pi)
        annulus = (radius >= inner_ratio) & (radius <= outer_ratio)
        center = radius <= center_ratio
        return (slice(y0, y1), slice(x0, x1)), annulus, center, angular

    @staticmethod
    def _fit_ellipse(contour):
        if len(contour) >= 5:
            return cv2.fitEllipse(contour)
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        return ((cx, cy), (2.0 * radius, 2.0 * radius), 0.0)

    @staticmethod
    def _scale_ellipse(ellipse, scale):
        return (
            (float(ellipse[0][0]), float(ellipse[0][1])),
            (float(ellipse[1][0]) * scale, float(ellipse[1][1]) * scale),
            float(ellipse[2]),
        )

    def _ring_sector_metrics(self, red_crop, annulus, angular):
        count = max(4, int(self.cfg["ring_sector_count"]))
        supported = []
        annulus_bool = annulus > 0
        for index in range(count):
            sector = (
                annulus_bool
                & (angular >= 2.0 * math.pi * index / count)
                & (angular < 2.0 * math.pi * (index + 1) / count)
            )
            pixels = int(np.count_nonzero(sector))
            red_ratio = float(np.count_nonzero((red_crop > 0) & sector)) / max(1, pixels)
            supported.append(red_ratio >= float(self.cfg["minimum_sector_red_ratio"]))
        coverage = float(np.mean(supported))
        doubled = supported + supported
        longest_gap = 0
        current_gap = 0
        for present in doubled:
            current_gap = 0 if present else current_gap + 1
            longest_gap = max(longest_gap, current_gap)
        longest_gap = min(longest_gap, count)
        return coverage, float(longest_gap) / float(count)

    def _proposals(self, red_mask, white_mask):
        proposals = []
        proposal_red = red_mask
        close_x = int(self.cfg.get("proposal_red_close_kernel_x", 1))
        close_y = int(self.cfg.get("proposal_red_close_kernel_y", 1))
        if close_x > 1 or close_y > 1:
            close_x = max(1, close_x | 1)
            close_y = max(1, close_y | 1)
            proposal_red = cv2.morphologyEx(
                red_mask,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (close_x, close_y)),
            )
        red_contours, hierarchy = cv2.findContours(
            proposal_red, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )
        hierarchy = None if hierarchy is None else hierarchy[0]
        for index, contour in enumerate(red_contours):
            if hierarchy is not None and bool(self.cfg.get("outer_contours_only", True)) and hierarchy[index][3] >= 0:
                continue
            area = float(cv2.contourArea(contour))
            if not float(self.cfg["minimum_contour_area_px"]) <= area <= float(self.cfg["maximum_contour_area_px"]):
                continue
            perimeter = float(cv2.arcLength(contour, True))
            if circularity(area, perimeter) < float(self.cfg["minimum_circularity"]):
                continue
            has_hole = bool(hierarchy is not None and hierarchy[index][2] >= 0)
            proposals.append((self._fit_ellipse(contour), contour, "red_ring", has_hole))

        white_contours, _ = cv2.findContours(
            white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        scales = [float(value) for value in self.cfg["white_outer_scale_hypotheses"]]
        for contour in white_contours:
            area = float(cv2.contourArea(contour))
            if not float(self.cfg["minimum_white_candidate_area_px"]) <= area <= float(self.cfg["maximum_white_candidate_area_px"]):
                continue
            if len(contour) < 5:
                continue
            base = self._fit_ellipse(contour)
            for scale in scales:
                proposals.append((self._scale_ellipse(base, scale), contour, "white_center", False))
        return proposals

    def detect(
        self,
        red_mask: np.ndarray,
        white_mask: np.ndarray,
        stamp_sec: float,
        m_per_px_x: float,
        m_per_px_y: float,
        pixel_to_metric,
        image_bgr: Optional[np.ndarray] = None,
    ) -> List[DiskDetection]:
        neutral_mask = None
        value_channel = None
        if image_bgr is not None:
            active = (red_mask > 0) | (white_mask > 0)
            active_y, active_x = np.nonzero(active)
            if active_y.size:
                y0 = int(np.min(active_y))
                y1 = int(np.max(active_y)) + 1
                x0 = int(np.min(active_x))
                x1 = int(np.max(active_x)) + 1
            else:
                y0, y1, x0, x1 = 0, 1, 0, 1
            image_crop = image_bgr[y0:y1, x0:x1]
            hsv = cv2.cvtColor(image_crop, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(image_crop, cv2.COLOR_BGR2LAB)
            tolerance = int(self.cfg["center_lab_neutral_tolerance"])
            neutral_crop = (
                (hsv[:, :, 1] <= int(self.cfg["center_max_saturation"]))
                & (np.abs(lab[:, :, 1].astype(np.int16) - 128) <= tolerance)
                & (np.abs(lab[:, :, 2].astype(np.int16) - 128) <= tolerance)
            )
            neutral_mask = np.zeros(red_mask.shape, dtype=bool)
            value_channel = np.zeros(red_mask.shape, dtype=np.float32)
            neutral_mask[y0:y1, x0:x1] = neutral_crop
            value_channel[y0:y1, x0:x1] = hsv[:, :, 2]

        self.last_candidates = []
        self.last_red_area_px = int(np.count_nonzero(red_mask))
        self.last_white_area_px = int(np.count_nonzero(white_mask))
        detections: List[DiskDetection] = []
        candidate_records: Dict[int, Dict[str, object]] = {}
        proposals = self._proposals(red_mask, white_mask)
        for candidate_id, (ellipse, contour, source, has_hole) in enumerate(proposals):
            contour_area = float(cv2.contourArea(contour))
            record: Dict[str, object] = {
                "candidate_id": candidate_id,
                "source": source,
                "ellipse": ellipse,
                "contour_area_px": contour_area,
                "has_hole": bool(has_hole),
                "accepted": False,
                "reject_reason": "not_evaluated",
                "confidence": 0.0,
            }
            candidate_records[candidate_id] = record
            geometry = np.asarray(
                [
                    ellipse[0][0], ellipse[0][1],
                    ellipse[1][0], ellipse[1][1], ellipse[2],
                ],
                dtype=np.float64,
            )
            if not np.all(np.isfinite(geometry)):
                self._record_rejection(record, "nonfinite_geometry")
                continue
            axes = ellipse[1]
            major, minor = max(axes), min(axes)
            axis_ratio = minor / max(major, 1e-6)
            record.update({
                "major_axis_px": float(major),
                "minor_axis_px": float(minor),
                "axis_ratio": float(axis_ratio),
            })
            if minor < float(self.cfg["minimum_minor_axis_px"]):
                self._record_rejection(record, "minor_axis_too_small")
                continue
            if major > float(self.cfg["maximum_major_axis_px"]):
                self._record_rejection(record, "major_axis_too_large")
                continue
            if axis_ratio < float(self.cfg["minimum_axis_ratio"]):
                self._record_rejection(record, "axis_ratio_too_small")
                continue

            if source == "white_center":
                already_covered = False
                for accepted in detections:
                    distance = math.hypot(
                        float(ellipse[0][0]) - accepted.center_px[0],
                        float(ellipse[0][1]) - accepted.center_px[1],
                    )
                    if distance <= max(
                        6.0,
                        float(self.cfg["dedup_center_fraction"])
                        * max(major, max(accepted.ellipse[1])),
                    ):
                        already_covered = True
                        break
                if already_covered:
                    self._record_rejection(record, "duplicate_candidate")
                    continue

                half_extent = int(math.ceil(0.55 * major)) + 2
                center_x = int(round(float(ellipse[0][0])))
                center_y = int(round(float(ellipse[0][1])))
                x0 = max(0, center_x - half_extent)
                x1 = min(red_mask.shape[1], center_x + half_extent + 1)
                y0 = max(0, center_y - half_extent)
                y1 = min(red_mask.shape[0], center_y + half_extent + 1)
                nearby_red = int(np.count_nonzero(red_mask[y0:y1, x0:x1]))
                record["nearby_red_area_px"] = nearby_red
                if nearby_red < int(
                    self.cfg.get("minimum_red_pixels_near_white_proposal", 5)
                ):
                    self._record_rejection(record, "no_red_near_white_center")
                    continue

            local_regions = self._local_ellipse_regions(
                red_mask.shape,
                ellipse,
                float(self.cfg["annulus_inner_ratio"]),
                float(self.cfg["annulus_outer_ratio"]),
                float(self.cfg["center_ratio"]),
            )
            if local_regions is None:
                self._record_rejection(record, "ellipse_outside_image")
                continue
            slices, annulus, center, angular = local_regions
            red_crop = red_mask[slices]
            white_crop = white_mask[slices]
            annulus_count = max(1, int(np.count_nonzero(annulus)))
            center_count = max(1, int(np.count_nonzero(center)))
            red_support = (red_crop > 0) & annulus
            white_support = (white_crop > 0) & center
            red_area_px = int(np.count_nonzero(red_support))
            white_area_px = int(np.count_nonzero(white_support))
            red_annulus = float(red_area_px) / annulus_count
            white_center = float(white_area_px) / center_count
            red_center = float(np.count_nonzero((red_crop > 0) & center)) / center_count
            sector_coverage, missing_sector_fraction = self._ring_sector_metrics(
                red_crop, annulus, angular
            )
            if neutral_mask is None:
                neutral_center = white_center
                center_contrast = 0.0
                center_evidence = white_support
            else:
                neutral_crop = neutral_mask[slices]
                value_crop = value_channel[slices]
                neutral_center = float(np.count_nonzero(neutral_crop & center)) / center_count
                center_evidence = ((white_crop > 0) | neutral_crop) & center
                center_values = value_crop[center]
                ring_values = value_crop[annulus]
                center_contrast = (
                    float(np.median(center_values) - np.median(ring_values))
                    if center_values.size and ring_values.size
                    else 0.0
                )
            red_centroid = self._mask_centroid(red_support, slices)
            white_centroid = self._mask_centroid(center_evidence, slices)
            center_distance = (
                math.hypot(
                    red_centroid[0] - white_centroid[0],
                    red_centroid[1] - white_centroid[1],
                )
                if red_centroid is not None and white_centroid is not None
                else float("inf")
            )
            record.update({
                "red_area_px": red_area_px,
                "white_area_px": white_area_px,
                "red_annulus_ratio": red_annulus,
                "white_center_ratio": white_center,
                "neutral_center_ratio": neutral_center,
                "red_center_ratio": red_center,
                "ring_sector_coverage": sector_coverage,
                "missing_sector_fraction": missing_sector_fraction,
                "center_brightness_contrast": center_contrast,
                "red_white_center_distance_px": (
                    float(center_distance) if np.isfinite(center_distance) else None
                ),
            })

            # At the far end of the commissioned Stage-1 view a physical disc
            # is only about four pixels high.  Perspective compression turns
            # the red annulus into a two-pixel horizontal band, so requiring a
            # complete annulus makes an otherwise correct white-centred disc
            # flicker.  Recover only the smallest white-centre hypothesis and
            # require all independent evidence (nearby red pixels, partial
            # ring continuity, neutral bright centre and concentricity).  This
            # is deliberately separate from the normal thresholds: it does
            # not make generic red or white objects easier to accept.
            far_recovery_geometry = (
                major >= float(self.cfg.get(
                    "far_white_center_recovery_minimum_major_axis_px", 12.0
                ))
                and major <= float(self.cfg.get(
                    "far_white_center_recovery_maximum_major_axis_px", 30.0
                ))
                and minor <= float(self.cfg.get(
                    "far_white_center_recovery_maximum_minor_axis_px", 8.0
                ))
                and axis_ratio >= float(self.cfg.get(
                    "far_white_center_recovery_minimum_axis_ratio", 0.12
                ))
                and axis_ratio <= float(self.cfg.get(
                    "far_white_center_recovery_maximum_axis_ratio", 0.40
                ))
            )
            far_white_center_recovery = bool(
                self.cfg.get("far_white_center_recovery_enabled", False)
            ) and (
                source == "white_center"
                and far_recovery_geometry
                and int(record.get("nearby_red_area_px", 0))
                >= int(
                    self.cfg.get(
                        "far_white_center_recovery_minimum_nearby_red_pixels", 12
                    )
                )
                and red_annulus
                >= float(
                    self.cfg.get(
                        "far_white_center_recovery_minimum_red_annulus_ratio", 0.16
                    )
                )
                and sector_coverage
                >= float(
                    self.cfg.get(
                        "far_white_center_recovery_minimum_sector_coverage", 0.35
                    )
                )
                and missing_sector_fraction
                <= float(
                    self.cfg.get(
                        "far_white_center_recovery_maximum_missing_sector_fraction",
                        0.60,
                    )
                )
                and neutral_center
                >= float(
                    self.cfg.get(
                        "far_white_center_recovery_minimum_neutral_center_ratio",
                        0.85,
                    )
                )
                and white_center
                >= float(
                    self.cfg.get(
                        "far_white_center_recovery_minimum_white_center_ratio", 0.75
                    )
                )
                and center_contrast
                >= float(
                    self.cfg.get(
                        "far_white_center_recovery_minimum_center_contrast", 15.0
                    )
                )
                and red_center
                <= float(
                    self.cfg.get(
                        "far_white_center_recovery_maximum_red_center_ratio", 0.40
                    )
                )
                and np.isfinite(center_distance)
                and center_distance
                <= float(
                    self.cfg.get(
                        "far_white_center_recovery_maximum_center_offset_px", 2.0
                    )
                )
            )
            record["far_white_center_recovery"] = far_white_center_recovery
            # The same far disc alternates between a white-centre proposal and
            # a partial red-contour proposal as exposure changes.  Recover the
            # latter only inside the strict far-disc size/shape envelope and
            # only when the neutral white centre remains tightly concentric.
            far_red_ring_recovery = bool(
                self.cfg.get("far_red_ring_recovery_enabled", False)
            ) and (
                source == "red_ring"
                and major >= float(self.cfg.get(
                    "far_red_ring_recovery_minimum_major_axis_px", 12.0
                ))
                and major <= float(self.cfg.get(
                    "far_red_ring_recovery_maximum_major_axis_px", 38.0
                ))
                and minor <= float(self.cfg.get(
                    "far_red_ring_recovery_maximum_minor_axis_px", 12.0
                ))
                and axis_ratio >= float(self.cfg.get(
                    "far_red_ring_recovery_minimum_axis_ratio", 0.12
                ))
                and axis_ratio <= float(self.cfg.get(
                    "far_red_ring_recovery_maximum_axis_ratio", 0.55
                ))
                and red_annulus >= float(self.cfg.get(
                    "far_red_ring_recovery_minimum_red_annulus_ratio", 0.40
                ))
                and sector_coverage >= float(self.cfg.get(
                    "far_red_ring_recovery_minimum_sector_coverage", 0.50
                ))
                and missing_sector_fraction <= float(self.cfg.get(
                    "far_red_ring_recovery_maximum_missing_sector_fraction", 0.50
                ))
                and neutral_center >= float(self.cfg.get(
                    "far_red_ring_recovery_minimum_neutral_center_ratio", 0.80
                ))
                and white_center >= float(self.cfg.get(
                    "far_red_ring_recovery_minimum_white_center_ratio", 0.55
                ))
                and center_contrast >= float(self.cfg.get(
                    "far_red_ring_recovery_minimum_center_contrast", 8.0
                ))
                and red_center <= float(self.cfg.get(
                    "far_red_ring_recovery_maximum_red_center_ratio", 0.80
                ))
                and np.isfinite(center_distance)
                and center_distance <= float(self.cfg.get(
                    "far_red_ring_recovery_maximum_center_offset_px", 2.5
                ))
            )
            record["far_red_ring_recovery"] = far_red_ring_recovery
            far_disc_recovery = (
                far_white_center_recovery or far_red_ring_recovery
            )
            if (
                red_annulus < float(self.cfg["minimum_red_annulus_ratio"])
                and not far_disc_recovery
            ):
                self._record_rejection(record, "red_annulus_too_weak")
                continue
            if (
                sector_coverage < float(self.cfg["minimum_ring_sector_coverage"])
                and not far_disc_recovery
            ):
                self._record_rejection(record, "ring_sector_coverage_too_low")
                continue
            if (
                missing_sector_fraction
                > float(self.cfg["maximum_missing_sector_fraction"])
                and not far_disc_recovery
            ):
                self._record_rejection(record, "ring_gap_too_large")
                continue
            if (
                neutral_center < float(self.cfg["minimum_neutral_center_ratio"])
                and not far_disc_recovery
            ):
                self._record_rejection(record, "neutral_center_too_weak")
                continue
            # At long range the white centre can have the same absolute value
            # as the grey cloth.  A neutral centre that is locally brighter
            # than its red annulus is still valid white-centre evidence.
            adaptive_white_center = white_center
            if image_bgr is not None and center_contrast >= float(
                self.cfg["minimum_center_brightness_contrast"]
            ):
                adaptive_white_center = max(adaptive_white_center, neutral_center)
            if (
                adaptive_white_center < float(self.cfg["minimum_white_center_ratio"])
                and not far_disc_recovery
            ):
                self._record_rejection(record, "white_center_too_weak")
                continue
            if (
                image_bgr is not None
                and center_contrast
                < float(self.cfg["minimum_center_brightness_contrast"])
                and not far_disc_recovery
            ):
                self._record_rejection(record, "center_contrast_too_low")
                continue
            strong_ring_recovery = bool(
                self.cfg.get("strong_ring_recovery_enabled", False)
            ) and (
                source == "red_ring"
                and red_center
                <= float(
                    self.cfg.get(
                        "strong_ring_recovery_maximum_red_center_ratio",
                        self.cfg["maximum_red_center_ratio"],
                    )
                )
                and red_annulus
                >= float(
                    self.cfg.get(
                        "strong_ring_recovery_minimum_red_annulus_ratio",
                        1.0,
                    )
                )
                and sector_coverage
                >= float(
                    self.cfg.get(
                        "strong_ring_recovery_minimum_sector_coverage",
                        1.0,
                    )
                )
                and neutral_center
                >= float(
                    self.cfg.get(
                        "strong_ring_recovery_minimum_neutral_center_ratio",
                        1.0,
                    )
                )
                and center_contrast
                >= float(
                    self.cfg.get(
                        "strong_ring_recovery_minimum_center_contrast",
                        float("inf"),
                    )
                )
                and center_distance
                <= float(
                    self.cfg.get(
                        "strong_ring_recovery_maximum_center_offset_px",
                        0.0,
                    )
                )
            )
            record["strong_ring_recovery"] = strong_ring_recovery
            if (
                red_center > float(self.cfg["maximum_red_center_ratio"])
                and not strong_ring_recovery
                and not far_disc_recovery
            ):
                self._record_rejection(record, "red_center_too_strong")
                continue
            maximum_center_offset = max(
                float(self.cfg.get("minimum_center_offset_limit_px", 4.0)),
                float(self.cfg.get("maximum_center_offset_fraction", 0.35)) * major,
            )
            record["maximum_center_offset_px"] = maximum_center_offset
            if not np.isfinite(center_distance) or center_distance > maximum_center_offset:
                self._record_rejection(record, "red_white_centers_misaligned")
                continue

            contrast_score = np.clip(
                (center_contrast - float(self.cfg["minimum_center_brightness_contrast"]))
                / max(
                    1.0,
                    float(self.cfg["full_center_brightness_contrast"])
                    - float(self.cfg["minimum_center_brightness_contrast"]),
                ),
                0.0,
                1.0,
            ) if image_bgr is not None else white_center
            axis_score = float(np.clip(
                (axis_ratio - float(self.cfg["minimum_axis_ratio"])) / 0.35, 0.0, 1.0
            ))
            confidence = float(
                0.28 * red_annulus
                + 0.27 * sector_coverage
                + 0.20 * neutral_center
                + 0.15 * max(float(contrast_score), white_center)
                + 0.10 * axis_score
            )
            if has_hole:
                confidence = min(1.0, confidence + float(self.cfg["hierarchy_bonus"]))
            record["confidence"] = confidence
            confidence_threshold = (
                float(
                    self.cfg.get(
                        "far_white_center_recovery_confidence_threshold", 0.50
                    )
                )
                if far_disc_recovery
                else float(self.cfg["confidence_threshold"])
            )
            if confidence < confidence_threshold:
                self._record_rejection(record, "confidence_too_low")
                continue

            center_px = (float(ellipse[0][0]), float(ellipse[0][1]))
            metric = pixel_to_metric(np.asarray([center_px], dtype=np.float64))[0]
            diameter_m = 0.5 * (major * m_per_px_x + minor * m_per_px_y)
            detections.append(
                DiskDetection(
                    center_px=center_px,
                    center_m=(float(metric[0]), float(metric[1])),
                    ellipse=ellipse,
                    contour=contour,
                    estimated_diameter_m=float(diameter_m),
                    confidence=confidence,
                    stamp_sec=stamp_sec,
                    red_annulus_ratio=red_annulus,
                    white_center_ratio=white_center,
                    red_sector_coverage=sector_coverage,
                    white_center_contrast=center_contrast,
                    proposal_source=source,
                    candidate_id=candidate_id,
                    contour_area_px=contour_area,
                    red_area_px=red_area_px,
                    white_area_px=white_area_px,
                    red_white_center_distance_px=float(center_distance),
                    used_history=False,
                )
            )
            record["accepted"] = True
            record["reject_reason"] = ""
            self.last_candidates.append(record)

        unique: List[DiskDetection] = []
        for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
            major = max(detection.ellipse[1])
            duplicate = False
            for kept in unique:
                distance = math.hypot(
                    detection.center_px[0] - kept.center_px[0],
                    detection.center_px[1] - kept.center_px[1],
                )
                if distance <= max(
                    6.0,
                    float(self.cfg["dedup_center_fraction"])
                    * max(major, max(kept.ellipse[1])),
                ):
                    duplicate = True
                    break
            if not duplicate:
                unique.append(detection)
            else:
                record = candidate_records.get(detection.candidate_id)
                if record is not None:
                    record["accepted"] = False
                    record["reject_reason"] = "duplicate_candidate"
        return sorted(unique, key=lambda item: item.center_px[1], reverse=True)
