#!/usr/bin/env python3
"""Interactive four-point ground-plane homography calibration tool."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

from yellow_track_avoidance.bev import CameraUndistorter
from yellow_track_avoidance.config import intrinsic_calibration_ready, load_config


POINT_NAMES = ("near-left", "near-right", "far-right", "far-left")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="real camera track image")
    parser.add_argument("--config", required=True, help="base configuration YAML")
    parser.add_argument("--output-config", required=True, help="new calibrated YAML; base file is not overwritten")
    parser.add_argument("--output-bev", required=True, help="preview BEV image path")
    parser.add_argument("--real-width-m", type=float, default=0.0, help="width between clicked near/far points")
    parser.add_argument("--real-length-m", type=float, default=0.0, help="forward ground length represented by clicks")
    parser.add_argument(
        "--lateral-margin-m",
        type=float,
        default=0.05,
        help="mapped margin outside each clicked yellow centreline",
    )
    parser.add_argument("--display-scale", type=float, default=0.65)
    parser.add_argument(
        "--allow-uncalibrated-preview",
        action="store_true",
        help="allow a raw distorted preview; output remains safety-invalid",
    )
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit("could not read {}".format(image_path))
    config = load_config(args.config)
    intrinsics_valid = intrinsic_calibration_ready(config)
    if intrinsics_valid:
        image = CameraUndistorter(config).undistort(image)
    elif not args.allow_uncalibrated_preview:
        raise SystemExit(
            "camera intrinsics are not calibrated; run intrinsic_calibrator.py first "
            "or use --allow-uncalibrated-preview for a non-safety preview"
        )
    scale = float(args.display_scale)
    if scale <= 0.0:
        raise SystemExit("display scale must be positive")
    display = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    clicks = []

    def redraw():
        frame = display.copy()
        for index, point in enumerate(clicks):
            shown = (int(round(point[0] * scale)), int(round(point[1] * scale)))
            cv2.circle(frame, shown, 6, (0, 0, 255), -1)
            cv2.putText(frame, POINT_NAMES[index], (shown[0] + 8, shown[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        instruction = "click: {} | r reset | Enter save | q quit".format(
            POINT_NAMES[len(clicks)] if len(clicks) < 4 else "complete"
        )
        cv2.putText(frame, instruction, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 255, 20), 2, cv2.LINE_AA)
        cv2.imshow("yellow-track homography calibration", frame)

    def on_mouse(event, x, y, _flags, _userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
            clicks.append((float(x) / scale, float(y) / scale))
            redraw()

    cv2.namedWindow("yellow-track homography calibration", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("yellow-track homography calibration", on_mouse)
    redraw()
    while True:
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            cv2.destroyAllWindows()
            raise SystemExit("calibration cancelled")
        if key == ord("r"):
            clicks[:] = []
            redraw()
        if key in (10, 13) and len(clicks) == 4:
            break
    cv2.destroyAllWindows()

    width_m = float(args.real_width_m) or float(config["bev"]["real_width_m"])
    length_m = float(args.real_length_m) or float(config["bev"]["real_length_m"])
    if width_m <= 0.0:
        width_m = float(input("real width between clicked left/right points (m): ").strip())
    if length_m <= 0.0:
        length_m = float(input("real forward length between near/far points (m): ").strip())
    if width_m <= 0.0 or length_m <= 0.0:
        raise SystemExit("real width and length must be positive")
    margin_m = float(args.lateral_margin_m)
    if margin_m < 0.0:
        raise SystemExit("lateral margin must be non-negative")
    mapped_width_m = width_m + 2.0 * margin_m

    width_px = int(config["bev"]["output_width_px"])
    height_px = int(config["bev"]["output_height_px"])
    source = np.asarray(clicks, dtype=np.float32)
    margin_px = margin_m / mapped_width_m * float(width_px - 1)
    destination = np.asarray(
        [
            [margin_px, height_px - 1],
            [width_px - 1 - margin_px, height_px - 1],
            [width_px - 1 - margin_px, 0],
            [margin_px, 0],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source, destination)
    bev = cv2.warpPerspective(image, matrix, (width_px, height_px))

    config.pop("_config_path", None)
    config["bev"]["source_points"] = [[float(x), float(y)] for x, y in clicks]
    config["bev"]["real_width_m"] = mapped_width_m
    config["bev"]["real_length_m"] = length_m
    config["bev"]["calibrated"] = bool(intrinsics_valid)
    config["bev"]["source_is_undistorted"] = bool(intrinsics_valid)
    config["camera"]["expected_width"] = int(image.shape[1])
    config["camera"]["expected_height"] = int(image.shape[0])
    config["calibration_metadata"] = {
        "source_image": str(image_path),
        "point_order": list(POINT_NAMES),
        "yellow_center_width_m": width_m,
        "lateral_margin_each_side_m": margin_m,
        "meters_per_pixel_x": mapped_width_m / max(1, width_px - 1),
        "meters_per_pixel_y": length_m / max(1, height_px - 1),
        "intrinsics_valid": bool(intrinsics_valid),
        "safety_valid": bool(intrinsics_valid),
    }

    output_config = Path(args.output_config).expanduser().resolve()
    output_config.parent.mkdir(parents=True, exist_ok=True)
    with output_config.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, allow_unicode=True, sort_keys=False)
    output_bev = Path(args.output_bev).expanduser().resolve()
    output_bev.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_bev), bev)
    print("calibration: {}".format(output_config))
    print("BEV preview: {}".format(output_bev))
    print("meters/pixel: x={:.6f}, y={:.6f}".format(
        mapped_width_m / max(1, width_px - 1), length_m / max(1, height_px - 1)
    ))
    if not intrinsics_valid:
        print("WARNING: raw distorted preview only; bev.calibrated remains false")


if __name__ == "__main__":
    main()
