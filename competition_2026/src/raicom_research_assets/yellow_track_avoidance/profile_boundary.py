"""Perspective-aware yellow-boundary detection from continuous confidence.

The detector samples one-dimensional profiles normal to nominal left/right
track boundaries.  A small dynamic program selects a smooth high-confidence
path, after which only current-frame supported points are fitted and exposed.
"""

from dataclasses import dataclass
import math
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .model import BoundaryResult


def _sigmoid(value):
    array = np.asarray(value, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-np.clip(array, -20.0, 20.0)))


def _longest_false_run(values: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in values.tolist():
        current = 0 if value else current + 1
        longest = max(longest, current)
    return longest


@dataclass
class _Profile:
    offsets: np.ndarray
    coordinates: np.ndarray
    base_confidence: np.ndarray
    score: np.ndarray
    prominence: np.ndarray
    valid_roi: np.ndarray


@dataclass
class _SideResult:
    detected: bool
    values_px: np.ndarray
    quality: float
    valid_sections: int
    support_ratio: float
    coverage_ratio: float
    median_confidence: float
    median_prominence: float
    fit_rms_px: float
    longest_missing_sections: int
    observed_min_row_px: int
    observed_max_row_px: int
    slope_dx_dy: float


class NormalProfileBoundaryDetector:
    """Detect left/right yellow lines without a hard full-image mask."""

    def __init__(self, config: Dict):
        self.config = config
        self.cfg = config["normal_profile_detector"]
        self.boundary_cfg = config["boundary"]
        self.both_missing_frames = 0
        self.last_metrics: Dict[str, object] = {}

    @staticmethod
    def _sample(image: np.ndarray, coordinates: np.ndarray, nearest=False) -> np.ndarray:
        map_x = coordinates[:, 0].astype(np.float32).reshape(1, -1)
        map_y = coordinates[:, 1].astype(np.float32).reshape(1, -1)
        interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
        sampled = cv2.remap(
            image,
            map_x,
            map_y,
            interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return sampled.reshape(-1).astype(np.float32)

    @staticmethod
    def _missing_side(rows: np.ndarray) -> _SideResult:
        return _SideResult(
            detected=False,
            values_px=np.full(rows.shape, np.nan, dtype=np.float64),
            quality=0.0,
            valid_sections=0,
            support_ratio=0.0,
            coverage_ratio=0.0,
            median_confidence=0.0,
            median_prominence=0.0,
            fit_rms_px=float("inf"),
            longest_missing_sections=int(rows.size),
            observed_min_row_px=-1,
            observed_max_row_px=-1,
            slope_dx_dy=float("nan"),
        )

    @staticmethod
    def _interp_profile(offsets, values, queries):
        return np.interp(
            queries,
            offsets,
            values,
            left=np.nan,
            right=np.nan,
        )

    def _profiles(
        self,
        side: str,
        confidence: np.ndarray,
        opponent: np.ndarray,
        roi: np.ndarray,
    ) -> List[_Profile]:
        side_cfg = self.cfg[side]
        far = np.asarray(side_cfg["far_point"], dtype=np.float32)
        near = np.asarray(side_cfg["near_point"], dtype=np.float32)
        tangent = near - far
        tangent /= max(float(np.linalg.norm(tangent)), 1e-6)
        normal = np.asarray([tangent[1], -tangent[0]], dtype=np.float32)
        if side == "left" and normal[0] < 0.0:
            normal *= -1.0
        if side == "right" and normal[0] > 0.0:
            normal *= -1.0

        count = max(4, int(self.cfg["normal_sections"]))
        step = max(1.0, float(self.cfg["normal_step_px"]))
        far_width = float(self.cfg["corridor_far_px"])
        near_width = float(self.cfg["corridor_near_px"])
        road_distances = np.asarray(
            self.cfg.get("road_reference_inside_distances_px", [8.0, 12.0, 16.0]),
            dtype=np.float32,
        )
        ridge_distances = np.asarray(
            self.cfg.get("ridge_distances_px", [3.0, 5.0, 8.0, 11.0]),
            dtype=np.float32,
        )
        profiles: List[_Profile] = []
        geometry = []
        for fraction in np.linspace(0.0, 1.0, count):
            center = far + float(fraction) * (near - far)
            width = (1.0 - fraction) * far_width + fraction * near_width
            offsets = np.arange(-width, width + 0.5 * step, step, dtype=np.float32)
            coordinates = center.reshape(1, 2) + offsets.reshape(-1, 1) * normal.reshape(1, 2)
            geometry.append((offsets, coordinates))

        # Remap every profile of one side in a single OpenCV call.  Calling
        # remap three times for each of 32 sections adds substantial Python/C++
        # transition overhead even though each individual profile is short.
        all_coordinates = np.concatenate(
            [coordinates for _, coordinates in geometry], axis=0
        )
        all_base = self._sample(confidence, all_coordinates)
        all_values = self._sample(opponent, all_coordinates)
        all_inside = self._sample(roi, all_coordinates, nearest=True) > 0.5

        cursor = 0
        for offsets, coordinates in geometry:
            next_cursor = cursor + len(offsets)
            base = all_base[cursor:next_cursor]
            values = all_values[cursor:next_cursor]
            inside = all_inside[cursor:next_cursor]
            cursor = next_cursor

            # All features below are evaluated for the whole profile at once.
            # The original scalar loop repeatedly called np.interp thousands of
            # times per frame and dominated runtime despite the tiny DP itself.
            road_queries = offsets[:, None] + road_distances[None, :]
            road_samples = self._interp_profile(offsets, values, road_queries)
            road_valid = np.isfinite(road_samples)
            road_for_median = np.where(road_valid, road_samples, np.inf)
            road_for_median.sort(axis=1)
            road_counts = np.sum(road_valid, axis=1)
            road_reference = np.zeros(offsets.shape, dtype=np.float32)
            for count_value in np.unique(road_counts):
                if count_value <= 0:
                    continue
                rows_with_count = road_counts == count_value
                middle = int(count_value) // 2
                if int(count_value) % 2:
                    road_reference[rows_with_count] = road_for_median[rows_with_count, middle]
                else:
                    road_reference[rows_with_count] = 0.5 * (
                        road_for_median[rows_with_count, middle - 1]
                        + road_for_median[rows_with_count, middle]
                    )
            local_delta = np.where(
                road_counts > 0,
                values - road_reference,
                -50.0,
            ).astype(np.float32)

            left_queries = offsets[:, None] - ridge_distances[None, :]
            right_queries = offsets[:, None] + ridge_distances[None, :]
            left_shoulders = self._interp_profile(offsets, values, left_queries)
            right_shoulders = self._interp_profile(offsets, values, right_queries)
            ridge_valid = np.isfinite(left_shoulders) & np.isfinite(right_shoulders)
            ridge_values = values[:, None] - 0.5 * (left_shoulders + right_shoulders)
            ridge_values = np.where(ridge_valid, ridge_values, -np.inf)
            has_ridge = np.any(ridge_valid, axis=1)
            prominence = np.where(
                has_ridge,
                np.max(ridge_values, axis=1),
                -50.0,
            ).astype(np.float32)

            inner = self._interp_profile(offsets, values, offsets + 5.0)
            inner_edge = np.where(
                np.isfinite(inner),
                values - inner,
                -50.0,
            ).astype(np.float32)

            p_local = _sigmoid(
                (local_delta - float(self.cfg["local_delta_center"]))
                / max(float(self.cfg["local_delta_scale"]), 1e-3)
            )
            p_ridge = _sigmoid(
                (prominence - float(self.cfg["ridge_center"]))
                / max(float(self.cfg["ridge_scale"]), 1e-3)
            )
            p_edge = _sigmoid(
                (inner_edge - float(self.cfg.get("inner_edge_center", 2.0)))
                / max(float(self.cfg.get("inner_edge_scale", 2.0)), 1e-3)
            )
            weights = self.cfg["profile_score_weights"]
            score = (
                float(weights["base_confidence"]) * base
                + float(weights["local_delta"]) * p_local
                + float(weights["ridge"]) * p_ridge
                + float(weights["inner_edge"]) * p_edge
            )
            score = np.where(inside, np.clip(score, 1e-4, 1.0), 1e-4)
            profiles.append(
                _Profile(
                    offsets=offsets,
                    coordinates=coordinates,
                    base_confidence=base,
                    score=score.astype(np.float32),
                    prominence=prominence,
                    valid_roi=inside,
                )
            )
        return profiles

    def _dynamic_path(self, profiles: List[_Profile]) -> List[int]:
        smooth_weight = float(self.cfg["dp_smooth_weight"])
        position_weight = float(self.cfg["dp_position_weight"])
        max_change = float(self.cfg["max_offset_change_per_section_px"])
        costs: List[np.ndarray] = []
        predecessors: List[np.ndarray] = []

        first = profiles[0]
        first_width = max(float(np.max(np.abs(first.offsets))), 1.0)
        first_cost = -np.log(first.score + 1e-6) + position_weight * (
            first.offsets / first_width
        ) ** 2
        costs.append(first_cost.astype(np.float64))
        predecessors.append(np.full(first.offsets.shape, -1, dtype=np.int32))

        for index in range(1, len(profiles)):
            previous = profiles[index - 1]
            current = profiles[index]
            difference = current.offsets[:, None] - previous.offsets[None, :]
            transition = smooth_weight * (difference / max(max_change, 1.0)) ** 2
            transition[np.abs(difference) > max_change] = np.inf
            total = transition + costs[-1][None, :]
            best_previous = np.argmin(total, axis=1).astype(np.int32)
            best_cost = total[np.arange(len(current.offsets)), best_previous]
            width = max(float(np.max(np.abs(current.offsets))), 1.0)
            unary = -np.log(current.score + 1e-6) + position_weight * (
                current.offsets / width
            ) ** 2
            costs.append(best_cost + unary)
            predecessors.append(best_previous)

        selected = [0] * len(profiles)
        selected[-1] = int(np.argmin(costs[-1]))
        for index in range(len(profiles) - 1, 0, -1):
            selected[index - 1] = int(predecessors[index][selected[index]])
        return selected

    @staticmethod
    def _weighted_line(points: np.ndarray, weights: np.ndarray):
        normalized = np.maximum(weights.astype(np.float64), 1e-6)
        normalized /= float(np.sum(normalized))
        center = np.sum(points.astype(np.float64) * normalized[:, None], axis=0)
        centered = points.astype(np.float64) - center.reshape(1, 2)
        covariance = (centered * normalized[:, None]).T @ centered
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        direction = eigenvectors[:, int(np.argmax(eigenvalues))]
        if direction[1] < 0.0:
            direction *= -1.0
        normal = np.asarray([-direction[1], direction[0]], dtype=np.float64)
        residuals = centered @ normal
        rms = math.sqrt(float(np.sum(normalized * residuals * residuals)))
        return center, direction, rms

    def _detect_side(
        self,
        side: str,
        confidence: np.ndarray,
        opponent: np.ndarray,
        roi: np.ndarray,
        rows: np.ndarray,
    ) -> _SideResult:
        profiles = self._profiles(side, confidence, opponent, roi)
        selected = self._dynamic_path(profiles)
        coordinates = np.asarray(
            [profiles[i].coordinates[selected[i]] for i in range(len(profiles))],
            dtype=np.float64,
        )
        selected_score = np.asarray(
            [profiles[i].score[selected[i]] for i in range(len(profiles))],
            dtype=np.float64,
        )
        selected_base = np.asarray(
            [profiles[i].base_confidence[selected[i]] for i in range(len(profiles))],
            dtype=np.float64,
        )
        selected_prominence = np.asarray(
            [profiles[i].prominence[selected[i]] for i in range(len(profiles))],
            dtype=np.float64,
        )
        selected_inside = np.asarray(
            [profiles[i].valid_roi[selected[i]] for i in range(len(profiles))],
            dtype=bool,
        )
        valid = (
            selected_inside
            & (selected_score >= float(self.cfg["minimum_section_confidence"]))
            & (selected_base >= float(self.cfg["minimum_base_confidence"]))
            & (selected_prominence >= float(self.cfg["minimum_section_prominence"]))
        )
        valid_count = int(np.count_nonzero(valid))
        support_ratio = float(valid_count) / float(max(1, len(valid)))
        valid_indices = np.flatnonzero(valid)
        coverage_ratio = (
            float(valid_indices[-1] - valid_indices[0] + 1) / float(len(valid))
            if valid_indices.size
            else 0.0
        )
        missing_run = (
            _longest_false_run(valid[valid_indices[0] : valid_indices[-1] + 1])
            if valid_indices.size
            else len(valid)
        )
        median_confidence = float(np.median(selected_score[valid])) if valid_count else 0.0
        median_prominence = float(np.median(selected_prominence[valid])) if valid_count else 0.0

        values = np.full(rows.shape, np.nan, dtype=np.float64)
        fit_rms = float("inf")
        slope = float("nan")
        observed_min = -1
        observed_max = -1
        geometry_ok = False
        if valid_count >= 2:
            fit_weights = selected_score[valid] * _sigmoid(
                (selected_prominence[valid] - float(self.cfg["minimum_section_prominence"]))
                / 2.0
            )
            center, direction, fit_rms = self._weighted_line(
                coordinates[valid], fit_weights
            )
            if abs(float(direction[1])) > 1e-5:
                slope = float(direction[0] / direction[1])
                tolerance = float(self.cfg["direction_slope_tolerance"])
                geometry_ok = slope <= tolerance if side == "left" else slope >= -tolerance
                observed_min = int(math.floor(float(np.min(coordinates[valid, 1]))))
                observed_max = int(math.ceil(float(np.max(coordinates[valid, 1]))))
                in_support = (rows >= observed_min) & (rows <= observed_max)
                values[in_support] = center[0] + (
                    rows[in_support].astype(np.float64) - center[1]
                ) * slope
                values[(values < 0.0) | (values >= confidence.shape[1])] = np.nan

        maximum_rms = float(self.cfg["maximum_fit_rms_px"])
        fit_quality = max(0.0, 1.0 - fit_rms / max(maximum_rms, 1e-6)) if np.isfinite(fit_rms) else 0.0
        confidence_quality = np.clip(
            (median_confidence - float(self.cfg["minimum_median_confidence"]))
            / max(1e-6, 1.0 - float(self.cfg["minimum_median_confidence"])),
            0.0,
            1.0,
        )
        prominence_quality = np.clip(
            median_prominence / max(2.0 * float(self.cfg["minimum_median_prominence"]), 1e-6),
            0.0,
            1.0,
        )
        quality = float(
            0.25 * support_ratio
            + 0.20 * confidence_quality
            + 0.15 * prominence_quality
            + 0.15 * fit_quality
            + 0.15 * coverage_ratio
            + 0.10 * float(geometry_ok)
        )
        detected = bool(
            valid_count >= int(self.cfg["minimum_valid_sections"])
            and support_ratio >= float(self.cfg["minimum_support_ratio"])
            and coverage_ratio >= float(self.cfg["minimum_arc_coverage_ratio"])
            and median_confidence >= float(self.cfg["minimum_median_confidence"])
            and median_prominence >= float(self.cfg["minimum_median_prominence"])
            and fit_rms <= maximum_rms
            and missing_run <= int(self.cfg["maximum_consecutive_missing_sections"])
            and geometry_ok
            and quality >= float(self.cfg["minimum_quality_score"])
        )
        if not detected:
            values[:] = np.nan

        return _SideResult(
            detected=detected,
            values_px=values,
            quality=quality,
            valid_sections=valid_count,
            support_ratio=support_ratio,
            coverage_ratio=coverage_ratio,
            median_confidence=median_confidence,
            median_prominence=median_prominence,
            fit_rms_px=fit_rms,
            longest_missing_sections=missing_run,
            observed_min_row_px=observed_min if detected else -1,
            observed_max_row_px=observed_max if detected else -1,
            slope_dx_dy=slope,
        )

    def detect(
        self,
        confidence: np.ndarray,
        opponent: np.ndarray,
        roi: np.ndarray,
    ) -> BoundaryResult:
        height, width = confidence.shape[:2]
        step = max(1, int(self.boundary_cfg["row_step_px"]))
        rows = np.arange(height - 1, -1, -step, dtype=np.int32)
        left_only = (
            str(self.boundary_cfg.get("detection_mode", "both"))
            .strip().lower() == "left_only"
        )
        left = self._detect_side("left", confidence, opponent, roi, rows)
        right = (
            self._missing_side(rows)
            if left_only
            else self._detect_side("right", confidence, opponent, roi, rows)
        )

        paired = np.isfinite(left.values_px) & np.isfinite(right.values_px)
        lane_width = (
            float(np.median(right.values_px[paired] - left.values_px[paired]))
            if np.any(paired)
            else 0.0
        )
        center = 0.5 * (left.values_px + right.values_px)
        observed_mins = [
            value for value in (left.observed_min_row_px, right.observed_min_row_px)
            if value >= 0
        ]
        observed_maxs = [
            value for value in (left.observed_max_row_px, right.observed_max_row_px)
            if value >= 0
        ]
        observed_min = min(observed_mins) if observed_mins else -1
        observed_max = max(observed_maxs) if observed_maxs else -1
        if left_only and left.detected:
            self.both_missing_frames = 0
            quality = left.quality
        elif left.detected and right.detected:
            self.both_missing_frames = 0
            quality = min(left.quality, right.quality)
        else:
            self.both_missing_frames += 1
            quality = 0.5 * max(left.quality, right.quality)

        self.last_metrics = {
            "profile_detector": True,
            "boundary_detection_mode": (
                "left_only" if left_only else "both"
            ),
            "left_quality": left.quality,
            "right_quality": right.quality,
            "left_valid_sections": left.valid_sections,
            "right_valid_sections": right.valid_sections,
            "left_support_ratio": left.support_ratio,
            "right_support_ratio": right.support_ratio,
            "left_median_confidence": left.median_confidence,
            "right_median_confidence": right.median_confidence,
            "left_median_prominence": left.median_prominence,
            "right_median_prominence": right.median_prominence,
            "left_fit_rms_px": left.fit_rms_px if np.isfinite(left.fit_rms_px) else None,
            "right_fit_rms_px": right.fit_rms_px if np.isfinite(right.fit_rms_px) else None,
            "left_slope_dx_dy": left.slope_dx_dy if np.isfinite(left.slope_dx_dy) else None,
            "right_slope_dx_dy": right.slope_dx_dy if np.isfinite(right.slope_dx_dy) else None,
            "left_longest_missing_sections": left.longest_missing_sections,
            "right_longest_missing_sections": right.longest_missing_sections,
        }
        return BoundaryResult(
            rows_px=rows,
            left_px=left.values_px,
            right_px=right.values_px,
            center_px=center,
            left_detected=left.detected,
            right_detected=right.detected,
            used_history=False,
            both_missing_frames=self.both_missing_frames,
            lane_width_px=lane_width,
            quality=quality,
            observed_min_row_px=observed_min,
            observed_max_row_px=observed_max,
        )
