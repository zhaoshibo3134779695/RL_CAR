#!/usr/bin/env python3
"""Calibrate the fixed RGB camera from checkerboard images or a live ROS topic."""

import argparse
import glob
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

from yellow_track_avoidance.config import load_config


def detect_corners(image, board_size):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, board_size, flags)
    if not found:
        return None
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
        40,
        0.001,
    )
    return cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria)


def load_offline_images(pattern, board_size):
    accepted = []
    for filename in sorted(glob.glob(pattern)):
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        if image is None:
            print("skip unreadable: {}".format(filename))
            continue
        corners = detect_corners(image, board_size)
        print("{}: {}".format(filename, "accepted" if corners is not None else "no checkerboard"))
        if corners is not None:
            accepted.append((image, corners, filename))
    return accepted


def capture_live(topic, board_size, minimum_views, capture_dir):
    import rospy
    from cv_bridge import CvBridge
    from sensor_msgs.msg import Image

    bridge = CvBridge()
    latest = {"image": None}
    accepted = []

    def callback(message):
        latest["image"] = bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")

    rospy.init_node("yellow_track_intrinsic_capture", anonymous=True, disable_signals=True)
    subscriber = rospy.Subscriber(topic, Image, callback, queue_size=1, buff_size=2 ** 24)
    deadline = time.monotonic() + 8.0
    while latest["image"] is None and time.monotonic() < deadline:
        time.sleep(0.05)
    if latest["image"] is None:
        subscriber.unregister()
        raise RuntimeError("no image received from {}".format(topic))

    output_dir = Path(capture_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.namedWindow("yellow-track intrinsic calibration", cv2.WINDOW_NORMAL)
    print("Move the checkerboard across center/corners and vary tilt/distance.")
    print("SPACE capture | ENTER calibrate | r reset | q quit")
    while True:
        image = latest["image"].copy()
        corners = detect_corners(image, board_size)
        display = image.copy()
        if corners is not None:
            cv2.drawChessboardCorners(display, board_size, corners, True)
        cv2.putText(
            display,
            "views {}/{} | SPACE capture | ENTER calibrate | r reset | q quit".format(
                len(accepted), minimum_views
            ),
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0) if corners is not None else (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("yellow-track intrinsic calibration", display)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            subscriber.unregister()
            cv2.destroyAllWindows()
            raise RuntimeError("calibration cancelled")
        if key == ord("r"):
            accepted[:] = []
            print("captured views reset")
        if key == 32:
            if corners is None:
                print("capture rejected: checkerboard not detected")
                continue
            index = len(accepted)
            filename = output_dir / "checkerboard_{:03d}.png".format(index)
            cv2.imwrite(str(filename), image)
            accepted.append((image.copy(), corners.copy(), str(filename)))
            print("captured {}".format(filename))
        if key in (10, 13):
            if len(accepted) < minimum_views:
                print("need at least {} accepted views".format(minimum_views))
                continue
            break
    subscriber.unregister()
    cv2.destroyAllWindows()
    return accepted


def object_points(board_size, square_size_m):
    columns, rows = board_size
    points = np.zeros((columns * rows, 3), np.float32)
    points[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    points[:, :2] *= float(square_size_m)
    return points


def solve(views, board_size, square_size_m):
    image_size = (views[0][0].shape[1], views[0][0].shape[0])
    for image, _corners, filename in views:
        if (image.shape[1], image.shape[0]) != image_size:
            raise RuntimeError("mixed image resolution at {}".format(filename))
    object_template = object_points(board_size, square_size_m)
    object_sets = [object_template.copy() for _ in views]
    image_sets = [corners.astype(np.float32) for _image, corners, _name in views]
    rms, matrix, distortion, rotations, translations = cv2.calibrateCamera(
        object_sets, image_sets, image_size, None, None
    )

    errors = []
    for objects, observed, rotation, translation in zip(
        object_sets, image_sets, rotations, translations
    ):
        projected, _ = cv2.projectPoints(objects, rotation, translation, matrix, distortion)
        errors.append(float(np.sqrt(np.mean(np.sum((observed - projected) ** 2, axis=2)))))
    return rms, matrix, distortion.reshape(-1), rotations, translations, errors, image_size


def reject_outliers(views, solution, board_size, square_size_m, minimum_views):
    errors = np.asarray(solution[5], dtype=np.float64)
    median = float(np.median(errors))
    mad = float(np.median(np.abs(errors - median)))
    threshold = median + max(0.15, 2.5 * 1.4826 * mad)
    keep = errors <= threshold
    if int(np.sum(keep)) < minimum_views or bool(np.all(keep)):
        return views, solution, []
    rejected = [views[index][2] for index in range(len(views)) if not keep[index]]
    filtered = [view for index, view in enumerate(views) if keep[index]]
    return filtered, solve(filtered, board_size, square_size_m), rejected


def calibration_coverage(views, image_size):
    centers = np.asarray([np.mean(corners.reshape(-1, 2), axis=0) for _im, corners, _name in views])
    hull_areas = np.asarray(
        [abs(cv2.contourArea(cv2.convexHull(corners.reshape(-1, 2)))) for _im, corners, _name in views]
    )
    center_span = np.ptp(centers, axis=0) / np.asarray(image_size, dtype=np.float64)
    scale_ratio = float(np.max(hull_areas) / max(np.min(hull_areas), 1.0))
    return center_span, scale_ratio


def write_outputs(args, config, views, solution):
    rms, matrix, distortion, _rotations, _translations, errors, image_size = solution
    width, height = image_size
    rectified, _roi = cv2.getOptimalNewCameraMatrix(
        matrix, distortion, image_size, 1.0, image_size, centerPrincipalPoint=False
    )
    center_span, scale_ratio = calibration_coverage(views, image_size)
    if center_span[0] < 0.18 or center_span[1] < 0.18 or scale_ratio < 1.20:
        raise RuntimeError(
            "checkerboard coverage is insufficient: center_span=({:.3f},{:.3f}) scale_ratio={:.3f}".format(
                center_span[0], center_span[1], scale_ratio
            )
        )

    config.pop("_config_path", None)
    config["camera_intrinsics"].update(
        {
            "calibrated": True,
            "image_width": int(width),
            "image_height": int(height),
            "distortion_model": "plumb_bob",
            "camera_matrix": matrix.reshape(-1).astype(float).tolist(),
            "distortion_coefficients": distortion.astype(float).tolist(),
            "rectified_camera_matrix": rectified.reshape(-1).astype(float).tolist(),
            "calibration_rms_px": float(rms),
        }
    )
    config["camera"]["expected_width"] = int(width)
    config["camera"]["expected_height"] = int(height)
    config.setdefault("calibration_metadata", {})["intrinsics"] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "board_inner_corners": [int(args.board_cols), int(args.board_rows)],
        "square_size_m": float(args.square_size_m),
        "accepted_views": len(views),
        "mean_reprojection_error_px": float(np.mean(errors)),
        "maximum_reprojection_error_px": float(np.max(errors)),
        "center_span_ratio": center_span.astype(float).tolist(),
        "checkerboard_scale_ratio": scale_ratio,
        "source_images": [name for _image, _corners, name in views],
    }

    output_config = Path(args.output_config).expanduser().resolve()
    if output_config == Path(args.config).expanduser().resolve():
        raise RuntimeError("output config must differ from the base config")
    output_config.parent.mkdir(parents=True, exist_ok=True)
    with output_config.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, allow_unicode=True, sort_keys=False)

    if args.output_camera_info:
        projection = np.zeros((3, 4), dtype=np.float64)
        projection[:, :3] = rectified
        camera_info = {
            "image_width": int(width),
            "image_height": int(height),
            "camera_name": "yellow_track_rgb",
            "camera_matrix": {"rows": 3, "cols": 3, "data": matrix.reshape(-1).astype(float).tolist()},
            "distortion_model": "plumb_bob",
            "distortion_coefficients": {"rows": 1, "cols": len(distortion), "data": distortion.astype(float).tolist()},
            "rectification_matrix": {"rows": 3, "cols": 3, "data": np.eye(3).reshape(-1).tolist()},
            "projection_matrix": {"rows": 3, "cols": 4, "data": projection.reshape(-1).astype(float).tolist()},
        }
        camera_info_path = Path(args.output_camera_info).expanduser().resolve()
        camera_info_path.parent.mkdir(parents=True, exist_ok=True)
        with camera_info_path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(camera_info, stream, allow_unicode=True, sort_keys=False)

    if args.output_preview:
        preview = cv2.undistort(views[0][0], matrix, distortion, None, rectified)
        preview_path = Path(args.output_preview).expanduser().resolve()
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(preview_path), preview)
    print("intrinsic config: {}".format(output_config))
    print("RMS={:.4f}px mean={:.4f}px max={:.4f}px views={}".format(
        rms, float(np.mean(errors)), float(np.max(errors)), len(views)
    ))
    print("K={}".format(matrix.reshape(-1).tolist()))
    print("D={}".format(distortion.tolist()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="uncalibrated base YAML")
    parser.add_argument("--output-config", required=True, help="new intrinsic-calibrated YAML")
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--image-glob", help="quoted glob of saved checkerboard images")
    sources.add_argument("--ros-topic", help="live sensor_msgs/Image topic")
    parser.add_argument("--capture-dir", default="/tmp/yellow_track_intrinsic_frames")
    parser.add_argument("--output-camera-info", default="")
    parser.add_argument("--output-preview", default="")
    parser.add_argument("--board-cols", type=int, required=True, help="checkerboard inner corners across")
    parser.add_argument("--board-rows", type=int, required=True, help="checkerboard inner corners down")
    parser.add_argument("--square-size-m", type=float, required=True)
    parser.add_argument("--minimum-views", type=int, default=15)
    args = parser.parse_args()
    if args.board_cols < 3 or args.board_rows < 3:
        raise SystemExit("board rows/columns are inner-corner counts and must be >= 3")
    if args.square_size_m <= 0.0 or args.minimum_views < 8:
        raise SystemExit("square size must be positive and minimum views must be >= 8")

    board_size = (args.board_cols, args.board_rows)
    if args.image_glob:
        views = load_offline_images(args.image_glob, board_size)
    else:
        views = capture_live(args.ros_topic, board_size, args.minimum_views, args.capture_dir)
    if len(views) < args.minimum_views:
        raise SystemExit("only {} valid views; need at least {}".format(len(views), args.minimum_views))
    config = load_config(args.config)
    solution = solve(views, board_size, args.square_size_m)
    views, solution, rejected = reject_outliers(
        views,
        solution,
        board_size,
        args.square_size_m,
        args.minimum_views,
    )
    if rejected:
        print("rejected high-error views: {}".format(", ".join(rejected)))
    write_outputs(args, config, views, solution)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print("calibration failed: {}".format(exc), file=sys.stderr)
        sys.exit(1)
