"""Configuration loading and strict safety validation."""

from copy import deepcopy
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml


class ConfigError(ValueError):
    pass


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge a vehicle profile over the shared defaults."""
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_mapping(config_path: Path, stack: List[Path]) -> Dict[str, Any]:
    if config_path in stack:
        chain = " -> ".join(str(item) for item in stack + [config_path])
        raise ConfigError("configuration inheritance cycle: {}".format(chain))
    if not config_path.is_file():
        raise ConfigError("configuration file does not exist: {}".format(config_path))
    with config_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ConfigError("configuration root must be a mapping")
    data = deepcopy(data)
    parent = data.pop("extends", None)
    if parent is None:
        return data
    if not isinstance(parent, str) or not parent.strip():
        raise ConfigError("extends must be a non-empty YAML file path")
    parent_path = Path(parent).expanduser()
    if not parent_path.is_absolute():
        parent_path = config_path.parent / parent_path
    parent_data = _load_mapping(parent_path.resolve(), stack + [config_path])
    return _deep_merge(parent_data, data)


def load_config(path: str) -> Dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    data = _load_mapping(config_path, [])
    data["_config_path"] = str(config_path)
    validate_config(data, require_calibration=False)
    return data


def _require(config: Dict[str, Any], dotted: str) -> Any:
    value: Any = config
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            raise ConfigError("missing configuration key: {}".format(dotted))
        value = value[key]
    return value


def _positive(value: Any, name: str, allow_zero: bool = False) -> None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise ConfigError("{} must be numeric".format(name))
    if numeric < 0.0 or (not allow_zero and numeric == 0.0):
        raise ConfigError("{} must be {}".format(name, "non-negative" if allow_zero else "positive"))


def _hsv_triplet(value: Iterable[Any], name: str) -> None:
    values = list(value)
    if len(values) != 3:
        raise ConfigError("{} must contain three HSV values".format(name))
    if not (0 <= int(values[0]) <= 179):
        raise ConfigError("{} hue must be in [0, 179]".format(name))
    if any(not 0 <= int(v) <= 255 for v in values[1:]):
        raise ConfigError("{} saturation/value must be in [0, 255]".format(name))


def validate_config(config: Dict[str, Any], require_calibration: bool = False) -> None:
    required = [
        "ros.image_topic",
        "ros.camera_info_topic",
        "ros.output_cmd_topic",
        "camera.roi_y_start_ratio",
        "camera.roi_y_end_ratio",
        "camera_intrinsics.calibrated",
        "camera_intrinsics.image_width",
        "camera_intrinsics.image_height",
        "camera_intrinsics.camera_matrix",
        "camera_intrinsics.distortion_coefficients",
        "camera_intrinsics.rectified_camera_matrix",
        "yellow_hsv.lower",
        "yellow_hsv.upper",
        "red_hsv.lower_1",
        "red_hsv.upper_1",
        "red_hsv.lower_2",
        "red_hsv.upper_2",
        "white_hsv.lower",
        "white_hsv.upper",
        "bev.output_width_px",
        "bev.output_height_px",
        "bev.real_width_m",
        "bev.real_length_m",
        "physical.robot_width_m",
        "physical.robot_length_m",
        "physical.disk_diameter_m",
        "physical.track_gray_width_m",
        "physical.yellow_line_width_m",
        "safety.boundary_margin_m",
        "safety.obstacle_margin_m",
        "control.enable_motion",
    ]
    for key in required:
        _require(config, key)

    for key in (
        "yellow_hsv.lower",
        "yellow_hsv.upper",
        "red_hsv.lower_1",
        "red_hsv.upper_1",
        "red_hsv.lower_2",
        "red_hsv.upper_2",
        "white_hsv.lower",
        "white_hsv.upper",
    ):
        _hsv_triplet(_require(config, key), key)

    for key in ("bev.output_width_px", "bev.output_height_px"):
        _positive(_require(config, key), key)
    for key in (
        "bev.real_width_m",
        "bev.real_length_m",
        "physical.robot_width_m",
        "physical.robot_length_m",
        "physical.disk_diameter_m",
        "physical.track_gray_width_m",
        "physical.yellow_line_width_m",
    ):
        _positive(_require(config, key), key, allow_zero=not require_calibration)
    if "local_forward_origin_offset_m" in config.get("physical", {}):
        _positive(
            config["physical"]["local_forward_origin_offset_m"],
            "physical.local_forward_origin_offset_m",
            allow_zero=True,
        )
    if "local_lateral_origin_offset_m" in config.get("physical", {}):
        try:
            lateral_offset = float(
                config["physical"]["local_lateral_origin_offset_m"]
            )
        except (TypeError, ValueError):
            raise ConfigError(
                "physical.local_lateral_origin_offset_m must be numeric"
            )
        if not math.isfinite(lateral_offset):
            raise ConfigError(
                "physical.local_lateral_origin_offset_m must be finite"
            )
    bev = config.get("bev", {})
    output_forward_min = bev.get("output_base_forward_min_m")
    output_forward_max = bev.get("output_base_forward_max_m")
    if (output_forward_min is None) != (output_forward_max is None):
        raise ConfigError(
            "bev.output_base_forward_min_m and "
            "bev.output_base_forward_max_m must be configured together"
        )
    if output_forward_min is not None:
        output_forward_min = float(output_forward_min)
        output_forward_max = float(output_forward_max)
        if not math.isfinite(output_forward_min) or not math.isfinite(output_forward_max):
            raise ConfigError("BEV base-forward output bounds must be finite")
        if output_forward_min < 0.0:
            raise ConfigError("BEV base-forward output minimum cannot be negative")
        if output_forward_max <= output_forward_min:
            raise ConfigError(
                "BEV base-forward output maximum must exceed its minimum"
            )
    ring_ratio = config.get("disk_detection", {}).get(
        "visible_ring_to_physical_diameter_ratio"
    )
    if ring_ratio is not None:
        ring_ratio = float(ring_ratio)
        if not 0.0 < ring_ratio <= 1.0:
            raise ConfigError(
                "disk_detection.visible_ring_to_physical_diameter_ratio "
                "must be in (0, 1]"
            )
    diameter_mode = str(
        config.get("disk_detection", {}).get(
            "projected_diameter_mode", "lateral_span"
        )
    )
    if diameter_mode not in (
        "lateral_span",
        "geometric_mean_ground_spans",
    ):
        raise ConfigError(
            "disk_detection.projected_diameter_mode must be "
            "'lateral_span' or 'geometric_mean_ground_spans'"
        )

    start = float(_require(config, "camera.roi_y_start_ratio"))
    end = float(_require(config, "camera.roi_y_end_ratio"))
    if not 0.0 <= start < end <= 1.0:
        raise ConfigError("camera ROI ratios must satisfy 0 <= start < end <= 1")

    for key in (
        "roi_polygon_normalized",
        "yellow_roi_polygon_normalized",
    ):
        polygon = config["camera"].get(key)
        if polygon is None:
            continue
        if (
            not isinstance(polygon, list)
            or len(polygon) < 3
            or any(not isinstance(point, list) or len(point) != 2 for point in polygon)
        ):
            raise ConfigError("camera.{} must contain at least three [x, y] points".format(key))
        try:
            coordinates = [float(value) for point in polygon for value in point]
        except (TypeError, ValueError):
            raise ConfigError("camera.{} coordinates must be numeric".format(key))
        if any(value < 0.0 or value > 1.0 for value in coordinates):
            raise ConfigError("camera.{} coordinates must be in [0, 1]".format(key))

    boundary_mode = str(
        config.get("boundary", {}).get("detection_mode", "both")
    ).strip().lower()
    if boundary_mode not in ("both", "left_only"):
        raise ConfigError(
            "boundary.detection_mode must be 'both' or 'left_only'"
        )

    points = _require(config, "bev.source_points")
    if not isinstance(points, list) or len(points) != 4 or any(len(p) != 2 for p in points):
        raise ConfigError("bev.source_points must contain four [x, y] points")
    calibrated = bool(_require(config, "bev.calibrated"))
    if require_calibration and not calibrated:
        raise ConfigError("BEV calibration is not marked valid")

    intrinsic_calibrated = bool(_require(config, "camera_intrinsics.calibrated"))
    for key in ("camera_intrinsics.camera_matrix", "camera_intrinsics.rectified_camera_matrix"):
        matrix = list(_require(config, key))
        if len(matrix) != 9:
            raise ConfigError("{} must contain nine values".format(key))
    distortion = list(_require(config, "camera_intrinsics.distortion_coefficients"))
    if len(distortion) < 4:
        raise ConfigError("camera_intrinsics.distortion_coefficients must contain at least four values")
    if require_calibration:
        if not intrinsic_calibrated:
            raise ConfigError("camera intrinsic calibration is not marked valid")
        camera_matrix = [float(v) for v in _require(config, "camera_intrinsics.camera_matrix")]
        rectified = [float(v) for v in _require(config, "camera_intrinsics.rectified_camera_matrix")]
        if camera_matrix[0] <= 0.0 or camera_matrix[4] <= 0.0 or rectified[0] <= 0.0 or rectified[4] <= 0.0:
            raise ConfigError("camera intrinsic focal lengths must be positive")
        if int(_require(config, "camera_intrinsics.image_width")) != int(_require(config, "camera.expected_width")):
            raise ConfigError("intrinsic image width does not match expected camera width")
        if int(_require(config, "camera_intrinsics.image_height")) != int(_require(config, "camera.expected_height")):
            raise ConfigError("intrinsic image height does not match expected camera height")


def calibration_ready(config: Dict[str, Any]) -> bool:
    try:
        validate_config(config, require_calibration=True)
    except ConfigError:
        return False
    return bool(config["bev"]["calibrated"] and config["camera_intrinsics"]["calibrated"])


def intrinsic_calibration_ready(config: Dict[str, Any]) -> bool:
    try:
        intrinsics = config["camera_intrinsics"]
        matrix = [float(v) for v in intrinsics["camera_matrix"]]
        rectified = [float(v) for v in intrinsics["rectified_camera_matrix"]]
        distortion = [float(v) for v in intrinsics["distortion_coefficients"]]
        return bool(
            intrinsics["calibrated"]
            and len(matrix) == 9
            and len(rectified) == 9
            and len(distortion) >= 4
            and matrix[0] > 0.0
            and matrix[4] > 0.0
            and rectified[0] > 0.0
            and rectified[4] > 0.0
        )
    except (KeyError, TypeError, ValueError):
        return False
