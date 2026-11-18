"""Bird's-eye-view homography and pixel/metric coordinate conversion."""

from typing import Iterable, Tuple

import cv2
import numpy as np

from .config import ConfigError, calibration_ready, intrinsic_calibration_ready


class CameraUndistorter:
    """Resolution-checked OpenCV remap using checked-in camera intrinsics."""

    def __init__(self, config):
        if not intrinsic_calibration_ready(config):
            raise ConfigError("camera intrinsic calibration is incomplete")
        camera = config["camera_intrinsics"]
        self.width = int(camera["image_width"])
        self.height = int(camera["image_height"])
        self.camera_matrix = np.asarray(camera["camera_matrix"], dtype=np.float64).reshape(3, 3)
        self.distortion = np.asarray(camera["distortion_coefficients"], dtype=np.float64).reshape(-1)
        self.rectified_matrix = np.asarray(
            camera["rectified_camera_matrix"], dtype=np.float64
        ).reshape(3, 3)
        self.map_x, self.map_y = cv2.initUndistortRectifyMap(
            self.camera_matrix,
            self.distortion,
            None,
            self.rectified_matrix,
            (self.width, self.height),
            cv2.CV_32FC1,
        )

    def undistort(self, image: np.ndarray) -> np.ndarray:
        if image.ndim < 2 or image.shape[1] != self.width or image.shape[0] != self.height:
            raise ConfigError(
                "intrinsic calibration resolution {}x{} does not match image {}x{}".format(
                    self.width, self.height, image.shape[1], image.shape[0]
                )
            )
        return cv2.remap(
            image,
            self.map_x,
            self.map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

    def undistort_points(self, pixels: Iterable[Tuple[float, float]]) -> np.ndarray:
        """Map distorted full-image pixels into rectified full-image pixels."""
        points = np.asarray(list(pixels), dtype=np.float64).reshape(-1, 1, 2)
        if points.size == 0:
            return np.empty((0, 2), dtype=np.float64)
        return cv2.undistortPoints(
            points,
            self.camera_matrix,
            self.distortion,
            P=self.rectified_matrix,
        ).reshape(-1, 2)

    def distort_points(self, rectified_pixels: Iterable[Tuple[float, float]]) -> np.ndarray:
        """Map rectified full-image pixels back to raw distorted pixels."""
        pixels = np.asarray(list(rectified_pixels), dtype=np.float64).reshape(-1, 2)
        if len(pixels) == 0:
            return np.empty((0, 2), dtype=np.float64)
        homogeneous = np.column_stack(
            [pixels, np.ones(len(pixels), dtype=np.float64)]
        )
        normalized = homogeneous.dot(
            np.linalg.inv(self.rectified_matrix).T
        )
        normalized[:, 0] /= normalized[:, 2]
        normalized[:, 1] /= normalized[:, 2]
        object_points = np.column_stack(
            [
                normalized[:, 0],
                normalized[:, 1],
                np.ones(len(pixels), dtype=np.float64),
            ]
        )
        projected, _ = cv2.projectPoints(
            object_points,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            self.camera_matrix,
            self.distortion,
        )
        return projected.reshape(-1, 2)


class BirdEyeTransformer:
    """Maps the configured ground quadrilateral to a metric BEV image.

    Source point order is near-left, near-right, far-right, far-left. BEV
    metric coordinates use x forward and y left, with the robot at the bottom
    center of the output image.
    """

    def __init__(self, config):
        if not calibration_ready(config):
            raise ConfigError("homography or physical calibration is incomplete")
        bev = config["bev"]
        self.width_px = int(bev["output_width_px"])
        self.height_px = int(bev["output_height_px"])
        self.real_width_m = float(bev["real_width_m"])
        self.real_length_m = float(bev["real_length_m"])
        self.m_per_px_x = self.real_width_m / float(max(1, self.width_px - 1))
        # The calibrated ground quadrilateral defines metric scale and yaw,
        # but its lateral zero is the calibration canvas centre rather than
        # necessarily the vehicle centre.  Keep this signed translation
        # separate from the homography so a measured base_link offset moves
        # every visual feature (boundaries and discs) consistently.
        self.local_lateral_origin_offset_m = float(
            config.get("physical", {}).get(
                "local_lateral_origin_offset_m", 0.0
            )
        )

        # ``real_length_m`` is the calibrated distance between the near and
        # far source-point pairs. Enlarging that value would rescale every
        # previously verified metric measurement. Optional base-forward
        # output bounds instead render/crop the same ground homography on a
        # canvas expressed relative to base_link.
        forward_offset = float(
            config.get("physical", {}).get("local_forward_origin_offset_m", 0.0)
        )
        self.base_forward_min_m = float(
            bev.get("output_base_forward_min_m", forward_offset)
        )
        self.base_forward_max_m = float(
            bev.get(
                "output_base_forward_max_m",
                forward_offset + self.real_length_m,
            )
        )
        if self.base_forward_max_m <= self.base_forward_min_m:
            raise ConfigError(
                "bev output base-forward maximum must exceed its minimum"
            )
        self.metric_forward_min_m = self.base_forward_min_m - forward_offset
        self.metric_forward_max_m = self.base_forward_max_m - forward_offset
        self.output_forward_span_m = (
            self.metric_forward_max_m - self.metric_forward_min_m
        )
        self.m_per_px_y = self.output_forward_span_m / float(
            max(1, self.height_px - 1)
        )

        source = np.asarray(bev["source_points"], dtype=np.float32)
        self.extend_to_image_bottom = bool(
            bev.get("extend_to_image_bottom", False)
        )
        near_destination_y = self.metric_forward_max_m / self.m_per_px_y
        far_destination_y = (
            self.metric_forward_max_m - self.real_length_m
        ) / self.m_per_px_y
        destination = np.asarray(
            [
                [0.0, near_destination_y],
                [self.width_px - 1.0, near_destination_y],
                [self.width_px - 1.0, far_destination_y],
                [0.0, far_destination_y],
            ],
            dtype=np.float32,
        )
        if abs(float(cv2.contourArea(source))) < 1.0:
            raise ConfigError("homography source points are degenerate")
        self.source_polygon = source
        self.lower_matrix = None
        self.lower_inverse_matrix = None
        self.source_seam_y = None
        self.destination_seam_y = None

        if self.extend_to_image_bottom:
            if (
                "output_base_forward_min_m" in bev
                or "output_base_forward_max_m" in bev
            ):
                raise ConfigError(
                    "explicit base-forward BEV bounds cannot be combined with "
                    "the legacy piecewise image-bottom extension"
                )
            if bool(config["control"]["enable_motion"]):
                raise ConfigError(
                    "expanded image-bottom detection is detect-only; motion must remain disabled"
                )
            image_width = int(config["camera"]["expected_width"])
            image_height = int(config["camera"]["expected_height"])
            if image_width < 2 or image_height < 2:
                raise ConfigError("expanded detection requires a fixed camera resolution")
            bottom_fraction = float(
                bev.get("bottom_extension_output_fraction", 0.35)
            )
            if not 0.05 <= bottom_fraction <= 0.90:
                raise ConfigError(
                    "bev.bottom_extension_output_fraction must be in [0.05, 0.90]"
                )

            self.source_seam_y = float(np.mean(source[:2, 1]))
            self.destination_seam_y = float(self.height_px - 1) * (
                1.0 - bottom_fraction
            )
            upper_destination = np.asarray(
                [
                    [0.0, self.destination_seam_y],
                    [self.width_px - 1.0, self.destination_seam_y],
                    [self.width_px - 1.0, 0.0],
                    [0.0, 0.0],
                ],
                dtype=np.float32,
            )
            bottom_y = float(image_height - 1)
            right_x = float(image_width - 1)
            lower_source = np.asarray(
                [
                    [0.0, bottom_y],
                    [right_x, bottom_y],
                    [right_x, self.source_seam_y],
                    [0.0, self.source_seam_y],
                ],
                dtype=np.float32,
            )
            lower_destination = np.asarray(
                [
                    [0.0, self.height_px - 1.0],
                    [self.width_px - 1.0, self.height_px - 1.0],
                    [self.width_px - 1.0, self.destination_seam_y],
                    [0.0, self.destination_seam_y],
                ],
                dtype=np.float32,
            )
            self.matrix = cv2.getPerspectiveTransform(source, upper_destination)
            self.inverse_matrix = cv2.getPerspectiveTransform(
                upper_destination, source
            )
            self.lower_matrix = cv2.getPerspectiveTransform(
                lower_source, lower_destination
            )
            self.lower_inverse_matrix = cv2.getPerspectiveTransform(
                lower_destination, lower_source
            )
            self.source_polygon = np.asarray(
                [
                    source[3],
                    source[2],
                    source[1],
                    [right_x, bottom_y],
                    [0.0, bottom_y],
                    source[0],
                ],
                dtype=np.float32,
            )
        else:
            self.matrix = cv2.getPerspectiveTransform(source, destination)
            self.inverse_matrix = cv2.getPerspectiveTransform(destination, source)

    def warp_image(self, image: np.ndarray, interpolation=cv2.INTER_LINEAR) -> np.ndarray:
        upper = cv2.warpPerspective(
            image,
            self.matrix,
            (self.width_px, self.height_px),
            flags=interpolation,
            borderMode=cv2.BORDER_CONSTANT,
        )
        if not self.extend_to_image_bottom:
            return upper
        lower = cv2.warpPerspective(
            image,
            self.lower_matrix,
            (self.width_px, self.height_px),
            flags=interpolation,
            borderMode=cv2.BORDER_CONSTANT,
        )
        seam_row = int(round(self.destination_seam_y))
        upper[seam_row + 1 :] = lower[seam_row + 1 :]
        return upper

    def warp_mask(self, mask: np.ndarray) -> np.ndarray:
        return self.warp_image(mask, interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def transform_points(points: Iterable[Tuple[float, float]], matrix: np.ndarray) -> np.ndarray:
        array = np.asarray(list(points), dtype=np.float32).reshape(-1, 1, 2)
        if array.size == 0:
            return np.empty((0, 2), dtype=np.float32)
        return cv2.perspectiveTransform(array, matrix).reshape(-1, 2)

    def original_to_bev(self, points: Iterable[Tuple[float, float]]) -> np.ndarray:
        array = np.asarray(list(points), dtype=np.float32).reshape(-1, 2)
        if len(array) == 0:
            return np.empty((0, 2), dtype=np.float32)
        if not self.extend_to_image_bottom:
            return self.transform_points(array, self.matrix)
        result = np.empty_like(array)
        lower = array[:, 1] >= self.source_seam_y
        result[~lower] = self.transform_points(array[~lower], self.matrix)
        result[lower] = self.transform_points(array[lower], self.lower_matrix)
        return result

    def bev_to_original(self, points: Iterable[Tuple[float, float]]) -> np.ndarray:
        array = np.asarray(list(points), dtype=np.float32).reshape(-1, 2)
        if len(array) == 0:
            return np.empty((0, 2), dtype=np.float32)
        if not self.extend_to_image_bottom:
            return self.transform_points(array, self.inverse_matrix)
        result = np.empty_like(array)
        lower = array[:, 1] >= self.destination_seam_y
        result[~lower] = self.transform_points(array[~lower], self.inverse_matrix)
        result[lower] = self.transform_points(
            array[lower], self.lower_inverse_matrix
        )
        return result

    def pixel_to_metric(self, pixels: np.ndarray) -> np.ndarray:
        points = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
        metric = np.empty_like(points)
        metric[:, 0] = self.metric_forward_max_m - points[:, 1] * self.m_per_px_y
        metric[:, 1] = (
            (self.width_px * 0.5 - points[:, 0]) * self.m_per_px_x
            + self.local_lateral_origin_offset_m
        )
        return metric

    def metric_to_pixel(self, metric_points: np.ndarray) -> np.ndarray:
        points = np.asarray(metric_points, dtype=np.float64).reshape(-1, 2)
        pixels = np.empty_like(points)
        pixels[:, 0] = self.width_px * 0.5 - (
            points[:, 1] - self.local_lateral_origin_offset_m
        ) / self.m_per_px_x
        pixels[:, 1] = (
            self.metric_forward_max_m - points[:, 0]
        ) / self.m_per_px_y
        return pixels
