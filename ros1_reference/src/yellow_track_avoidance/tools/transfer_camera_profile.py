#!/usr/bin/env python3
"""Transfer a checked camera/perception profile to another sensor readout.

This is intended for two modes from the same fixed lens and sensor.  It
transforms calibrated pixel coordinates and pixel-domain detector thresholds
while leaving metric BEV dimensions and metric safety parameters unchanged.
An independently measured checkerboard calibration is still preferred.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import yaml


def scaled(value, factor):
    result = float(value) * factor
    if isinstance(value, int) and not isinstance(value, bool):
        return int(round(result))
    return type(value)(result)


def odd_scaled(value, factor):
    value = int(value)
    if value <= 1:
        return value
    result = max(1, int(round(value * factor)))
    return result if result % 2 else result + 1


def transform_point(point, scale_x, scale_y, offset_x, offset_y):
    return [
        float(point[0]) * scale_x + offset_x,
        float(point[1]) * scale_y + offset_y,
    ]


def scale_named_values(section, keys, factor):
    for key in keys:
        if key in section:
            section[key] = scaled(section[key], factor)


def scale_named_areas(section, keys, factor):
    for key in keys:
        if key in section:
            section[key] = scaled(section[key], factor * factor)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--scale-x", type=float, required=True)
    parser.add_argument("--scale-y", type=float, required=True)
    parser.add_argument("--offset-x", type=float, required=True)
    parser.add_argument("--offset-y", type=float, required=True)
    parser.add_argument(
        "--camera-matrix",
        type=float,
        nargs=9,
        required=True,
        metavar=("FX", "S", "CX", "Z0", "FY", "CY", "Z1", "Z2", "ONE"),
    )
    parser.add_argument("--distortion", type=float, nargs=5, required=True)
    parser.add_argument("--registration-mean-error-px", type=float, required=True)
    parser.add_argument("--registration-max-error-px", type=float, required=True)
    parser.add_argument("--source-image", required=True)
    parser.add_argument("--target-image", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.width < 2 or args.height < 2:
        raise SystemExit("target dimensions must be positive")
    if args.scale_x <= 0.0 or args.scale_y <= 0.0:
        raise SystemExit("profile scales must be positive")

    base_path = Path(args.base_config).expanduser().resolve()
    output_path = Path(args.output_config).expanduser().resolve()
    if base_path == output_path:
        raise SystemExit("output config must differ from the accepted base config")
    with base_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    camera = config["camera"]
    source_width = int(camera["expected_width"])
    source_height = int(camera["expected_height"])
    camera["expected_width"] = args.width
    camera["expected_height"] = args.height
    if "detection_source_points" in camera:
        camera["detection_source_points"] = [
            transform_point(
                point,
                args.scale_x,
                args.scale_y,
                args.offset_x,
                args.offset_y,
            )
            for point in camera["detection_source_points"]
        ]

    intrinsics = config["camera_intrinsics"]
    old_rectified = list(intrinsics["rectified_camera_matrix"])
    intrinsics["image_width"] = args.width
    intrinsics["image_height"] = args.height
    intrinsics["camera_matrix"] = [float(value) for value in args.camera_matrix]
    intrinsics["distortion_coefficients"] = [
        float(value) for value in args.distortion
    ]
    intrinsics["rectified_camera_matrix"] = [
        old_rectified[0] * args.scale_x,
        0.0,
        old_rectified[2] * args.scale_x + args.offset_x,
        0.0,
        old_rectified[4] * args.scale_y,
        old_rectified[5] * args.scale_y + args.offset_y,
        0.0,
        0.0,
        1.0,
    ]
    intrinsics["calibration_rms_px"] = float(
        args.registration_mean_error_px
    )

    bev = config["bev"]
    bev["source_points"] = [
        transform_point(
            point,
            args.scale_x,
            args.scale_y,
            args.offset_x,
            args.offset_y,
        )
        for point in bev["source_points"]
    ]

    scale = 0.5 * (args.scale_x + args.scale_y)
    area_scale = scale * scale

    scale_named_values(config["yellow_scoring"], ["band_height_px"], scale)
    scale_named_values(
        config["yellow_orientation_filter"],
        ["window_size_px", "recovery_dilate_px", "red_exclusion_dilate_px"],
        scale,
    )
    scale_named_areas(
        config["yellow_orientation_filter"],
        ["minimum_support_pixels"],
        scale,
    )

    morphology = config["morphology"]
    for key in (
        "yellow_open_kernel",
        "yellow_close_kernel",
        "red_open_kernel",
        "red_close_kernel",
        "white_open_kernel",
        "white_close_kernel",
    ):
        morphology[key] = odd_scaled(morphology[key], scale)

    scale_named_values(
        config["boundary"],
        [
            "row_step_px",
            "row_band_half_height_px",
            "minimum_row_pixels",
            "minimum_boundary_gap_px",
            "perspective_center_exclusion_px",
            "row_tracking_max_jump_px",
            "fit_outlier_px",
            "initial_lane_width_px",
            "minimum_lane_width_px",
            "maximum_lane_width_px",
        ],
        scale,
    )

    profile = config["normal_profile_detector"]
    scale_named_values(
        profile,
        [
            "normal_step_px",
            "corridor_far_px",
            "corridor_near_px",
            "max_offset_change_per_section_px",
            "maximum_fit_rms_px",
        ],
        scale,
    )
    for key in ("road_reference_inside_distances_px", "ridge_distances_px"):
        profile[key] = [float(value) * scale for value in profile[key]]
    for side in ("left", "right"):
        for point_name in ("far_point", "near_point"):
            profile[side][point_name] = transform_point(
                profile[side][point_name],
                args.scale_x,
                args.scale_y,
                args.offset_x,
                args.offset_y,
            )

    disk = config["disk_detection"]
    scale_named_areas(
        disk,
        [
            "minimum_contour_area_px",
            "maximum_contour_area_px",
            "minimum_white_candidate_area_px",
            "maximum_white_candidate_area_px",
            "minimum_red_pixels_near_white_proposal",
        ],
        scale,
    )
    scale_named_values(
        disk,
        [
            "minimum_minor_axis_px",
            "maximum_major_axis_px",
            "minimum_center_offset_limit_px",
            "lane_center_margin_px",
        ],
        scale,
    )
    disk["proposal_red_close_kernel_x"] = odd_scaled(
        disk["proposal_red_close_kernel_x"], args.scale_x
    )
    disk["proposal_red_close_kernel_y"] = odd_scaled(
        disk["proposal_red_close_kernel_y"], args.scale_y
    )

    scale_named_values(
        config["disc_acceptance"],
        ["center_std_near_px", "center_std_middle_px", "center_std_far_px"],
        scale,
    )

    metadata = config.setdefault("calibration_metadata", {})
    metadata["method"] = "registered_camera_profile_transfer"
    metadata["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["profile_transfer"] = {
        "base_config": str(base_path),
        "source_resolution": [source_width, source_height],
        "target_resolution": [args.width, args.height],
        "scale_x": args.scale_x,
        "scale_y": args.scale_y,
        "offset_x_px": args.offset_x,
        "offset_y_px": args.offset_y,
        "registration_mean_error_px": args.registration_mean_error_px,
        "registration_max_error_px": args.registration_max_error_px,
        "source_image": str(Path(args.source_image).expanduser().resolve()),
        "target_image": str(Path(args.target_image).expanduser().resolve()),
    }
    metadata["intrinsics_valid"] = True
    metadata["safety_valid"] = False
    metadata["validation_note"] = (
        "Profile registration is suitable for locked static accuracy testing; "
        "checkerboard and metric ground checks are still required before motion."
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, allow_unicode=True, sort_keys=False)
    print(output_path)
    print(
        "scale=({:.9f},{:.9f}) offset=({:.6f},{:.6f}) area_scale={:.6f}".format(
            args.scale_x,
            args.scale_y,
            args.offset_x,
            args.offset_y,
            area_scale,
        )
    )


if __name__ == "__main__":
    main()
