#!/usr/bin/env python3
"""Rebase calibrated pixel geometry after replacing camera intrinsics.

Both configurations must describe the same physical camera pose and image
resolution.  Visible pixel points are carried through the complete old and
new lens models.  The ground homography is refitted from a dense set of raw
image samples so the empirically calibrated BEV metric geometry is retained
despite the changed distortion model.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml


def camera_model(config):
    intrinsics = config["camera_intrinsics"]
    camera_matrix = np.asarray(
        intrinsics["camera_matrix"], dtype=np.float64
    ).reshape(3, 3)
    distortion = np.asarray(
        intrinsics["distortion_coefficients"], dtype=np.float64
    ).reshape(-1)
    rectified_matrix = np.asarray(
        intrinsics["rectified_camera_matrix"], dtype=np.float64
    ).reshape(3, 3)
    if (
        camera_matrix[0, 0] <= 0.0
        or camera_matrix[1, 1] <= 0.0
        or rectified_matrix[0, 0] <= 0.0
        or rectified_matrix[1, 1] <= 0.0
    ):
        raise ValueError("invalid camera model")
    return camera_matrix, distortion, rectified_matrix


def undistort_raw_points(points, model):
    camera_matrix, distortion, rectified_matrix = model
    array = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.undistortPoints(
        array,
        camera_matrix,
        distortion,
        P=rectified_matrix,
    ).reshape(-1, 2)


def rectified_to_raw_points(points, model):
    camera_matrix, distortion, rectified_matrix = model
    array = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    inverse_rectified = np.linalg.inv(rectified_matrix)
    homogeneous = np.column_stack(
        (array, np.ones((array.shape[0],), dtype=np.float64))
    )
    rays = (inverse_rectified @ homogeneous.T).T
    object_points = np.column_stack(
        (
            rays[:, 0] / rays[:, 2],
            rays[:, 1] / rays[:, 2],
            np.ones((array.shape[0],), dtype=np.float64),
        )
    )
    projected, _ = cv2.projectPoints(
        object_points,
        np.zeros((3, 1), dtype=np.float64),
        np.zeros((3, 1), dtype=np.float64),
        camera_matrix,
        distortion,
    )
    return projected.reshape(-1, 2)


def transform_points(points, old_model, new_model):
    raw_points = rectified_to_raw_points(points, old_model)
    return undistort_raw_points(raw_points, new_model).tolist()


def fit_ground_homography(reference, old_model, new_model, image_size):
    width, height = image_size
    bev = reference["bev"]
    output_width = int(bev["output_width_px"])
    output_height = int(bev["output_height_px"])
    destination = np.asarray(
        [
            [0.0, output_height - 1.0],
            [output_width - 1.0, output_height - 1.0],
            [output_width - 1.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )
    old_source = np.asarray(bev["source_points"], dtype=np.float32)
    old_homography = cv2.getPerspectiveTransform(old_source, destination)

    # Only fit samples that belong to the calibrated BEV.  Extrapolated source
    # corners can lie well outside the image, where a polynomial lens model is
    # not physically meaningful.
    raw_points = np.asarray(
        [
            (x, y)
            for y in np.linspace(0.40 * (height - 1), height - 1, 61)
            for x in np.linspace(0.0, width - 1, 81)
        ],
        dtype=np.float64,
    )
    old_rectified = undistort_raw_points(raw_points, old_model)
    new_rectified = undistort_raw_points(raw_points, new_model)
    old_bev = cv2.perspectiveTransform(
        old_rectified.astype(np.float32).reshape(-1, 1, 2),
        old_homography,
    ).reshape(-1, 2)
    usable = (
        np.all(np.isfinite(old_bev), axis=1)
        & (old_bev[:, 0] >= 0.0)
        & (old_bev[:, 0] <= output_width - 1.0)
        & (old_bev[:, 1] >= 0.0)
        & (old_bev[:, 1] <= output_height - 1.0)
    )
    if int(np.count_nonzero(usable)) < 100:
        raise ValueError("too few common ground samples to rebase homography")

    new_homography, _ = cv2.findHomography(
        new_rectified[usable],
        old_bev[usable],
        method=0,
    )
    if new_homography is None or not np.all(np.isfinite(new_homography)):
        raise ValueError("ground homography fit failed")
    new_source = cv2.perspectiveTransform(
        destination.reshape(-1, 1, 2),
        np.linalg.inv(new_homography),
    ).reshape(-1, 2)

    predicted = cv2.perspectiveTransform(
        new_rectified[usable].astype(np.float32).reshape(-1, 1, 2),
        new_homography,
    ).reshape(-1, 2)
    delta = predicted - old_bev[usable]
    pixel_errors = np.linalg.norm(delta, axis=1)
    metric_errors = np.hypot(
        delta[:, 0] * float(bev["real_width_m"]) / float(output_width - 1),
        delta[:, 1] * float(bev["real_length_m"]) / float(output_height - 1),
    )
    fit = {
        "sample_count": int(np.count_nonzero(usable)),
        "mean_error_bev_px": float(np.mean(pixel_errors)),
        "p95_error_bev_px": float(np.percentile(pixel_errors, 95.0)),
        "max_error_bev_px": float(np.max(pixel_errors)),
        "mean_error_m": float(np.mean(metric_errors)),
        "p95_error_m": float(np.percentile(metric_errors, 95.0)),
        "max_error_m": float(np.max(metric_errors)),
    }
    return new_source.tolist(), fit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-config", required=True)
    parser.add_argument("--intrinsic-config", required=True)
    parser.add_argument("--output-config", required=True)
    args = parser.parse_args()

    reference_path = Path(args.reference_config).expanduser().resolve()
    intrinsic_path = Path(args.intrinsic_config).expanduser().resolve()
    output_path = Path(args.output_config).expanduser().resolve()
    if output_path in (reference_path, intrinsic_path):
        raise SystemExit("output config must be a new file")

    with reference_path.open("r", encoding="utf-8") as stream:
        reference = yaml.safe_load(stream)
    with intrinsic_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    old_camera = reference["camera"]
    new_camera = config["camera"]
    old_size = (
        int(old_camera["expected_width"]),
        int(old_camera["expected_height"]),
    )
    new_size = (
        int(new_camera["expected_width"]),
        int(new_camera["expected_height"]),
    )
    if old_size != new_size:
        raise SystemExit(
            "rebase requires the same image resolution: {} != {}".format(
                old_size, new_size
            )
        )

    old_model = camera_model(reference)
    new_model = camera_model(config)
    config["camera"]["detection_source_points"] = transform_points(
        reference["camera"]["detection_source_points"],
        old_model,
        new_model,
    )
    new_source, fit = fit_ground_homography(
        reference,
        old_model,
        new_model,
        new_size,
    )
    config["bev"]["source_points"] = new_source
    for side in ("left", "right"):
        for point_name in ("near_point", "far_point"):
            config["normal_profile_detector"][side][point_name] = (
                transform_points(
                    [reference["normal_profile_detector"][side][point_name]],
                    old_model,
                    new_model,
                )[0]
            )

    metadata = config.setdefault("calibration_metadata", {})
    metadata["method"] = (
        "checkerboard_intrinsics_with_full_lens_ground_homography_rebase"
    )
    metadata["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["rectified_geometry_rebase"] = {
        "reference_config": str(reference_path),
        "intrinsic_config": str(intrinsic_path),
        "mapping": "old_rectified_to_raw_to_new_rectified",
        "ground_fit": fit,
    }
    metadata["intrinsics_valid"] = True
    metadata["safety_valid"] = False
    metadata["validation_note"] = (
        "Ground geometry was rebased through both complete lens models; "
        "repeat metric disc and lane checks before enabling motion."
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, allow_unicode=True, sort_keys=False)
    print(output_path)
    print(
        "ground fit samples={} mean={:.4f} m p95={:.4f} m max={:.4f} m".format(
            fit["sample_count"],
            fit["mean_error_m"],
            fit["p95_error_m"],
            fit["max_error_m"],
        )
    )


if __name__ == "__main__":
    main()
