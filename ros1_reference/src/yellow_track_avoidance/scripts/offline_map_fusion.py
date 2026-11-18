#!/usr/bin/env python3
"""Fuse one camera frame with the static route and render dry-run previews."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from yellow_track_avoidance.config import load_config
from yellow_track_avoidance.map_navigation import (
    build_reference_path,
    draw_route_preview,
    fuse_candidate_with_reference,
    load_metric_map,
    load_yaml,
    local_to_map,
    metric_to_pixel,
    validate_vehicle_path,
)
from yellow_track_avoidance.pipeline import AvoidancePipeline
from yellow_track_avoidance.planning import LocalPathPlanner, polyline_clearance_to_point


def candidate_by_name(planning, name):
    return next((item for item in planning.candidates if item.name == name), None)


def fused_for_candidate(route, reference, pose, center, candidate):
    return fuse_candidate_with_reference(
        route, reference, pose, center.points_m, candidate.points_m
    )[0]


def render_scene(
    route,
    occupancy,
    resolution,
    origin,
    reference,
    pose,
    planning,
    obstacles,
    config,
    title,
):
    canvas = draw_route_preview(route, occupancy, resolution, origin, reference)
    center = candidate_by_name(planning, "center")
    selected = planning.selected
    if center is not None:
        for candidate in planning.candidates:
            fused = fused_for_candidate(route, reference, pose, center, candidate)
            if fused is None:
                continue
            distance = np.concatenate(
                [[0.0], np.cumsum(np.linalg.norm(np.diff(fused, axis=0), axis=1))]
            )
            local = fused[distance <= float(candidate.points_m[-1, 0]) + 1e-6]
            pixels = metric_to_pixel(local, occupancy, resolution, origin)
            color = (110, 110, 110)
            thickness = 1
            if candidate.collision_free:
                color = (220, 150, 0)
                thickness = 2
            if selected is not None and candidate.name == selected.name:
                color = (255, 0, 255)
                thickness = 4
            cv2.polylines(
                canvas,
                [np.rint(pixels).astype(np.int32)],
                False,
                color,
                thickness,
                cv2.LINE_AA,
            )

    local_centers = np.asarray([item.center_m for item in obstacles], dtype=np.float64).reshape(-1, 2)
    map_centers = local_to_map(
        local_centers,
        pose,
        float(route["fusion"]["local_forward_origin_offset_m"]),
    )
    physical = config["physical"]
    expanded = (
        0.5 * float(physical["disk_diameter_m"])
        + 0.5 * float(physical["robot_width_m"])
        + float(config["safety"]["obstacle_margin_m"])
        + float(config["planner"]["collision_sample_padding_m"])
    )
    expanded_px = int(round(expanded / resolution))
    physical_px = int(round(0.5 * float(physical["disk_diameter_m"]) / resolution))
    for index, point in enumerate(map_centers):
        pixel = tuple(np.rint(metric_to_pixel([point], occupancy, resolution, origin)[0]).astype(int))
        cv2.circle(canvas, pixel, expanded_px, (0, 80, 255), 2, cv2.LINE_AA)
        cv2.circle(canvas, pixel, max(2, physical_px), (0, 0, 220), -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            "D{}".format(index + 1),
            (pixel[0] + 5, pixel[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 180),
            1,
            cv2.LINE_AA,
        )
    state_text = "SAFE {}".format(selected.name) if selected is not None else "STOP NO SAFE PATH"
    state_color = (0, 140, 0) if selected is not None else (0, 0, 220)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, 27), (255, 255, 255), -1)
    cv2.putText(
        canvas,
        "{} | {}".format(title, state_text),
        (8, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        state_color,
        1,
        cv2.LINE_AA,
    )
    return canvas, map_centers, expanded


def fused_validation(route, route_path, reference, pose, planning, obstacles, config):
    if planning is None or planning.selected is None:
        return {"safe": False, "reason": "no_safe_local_candidate"}
    center = candidate_by_name(planning, "center")
    if center is None:
        return {"safe": False, "reason": "center_candidate_missing"}
    fused = fused_for_candidate(route, reference, pose, center, planning.selected)
    occupancy, resolution, origin, _, _ = load_metric_map(route, route_path)
    physical = config["physical"]
    footprint = validate_vehicle_path(
        occupancy,
        resolution,
        origin,
        fused,
        float(physical["robot_length_m"]),
        float(physical["robot_width_m"]),
        float(route["safety"]["map_boundary_margin_m"]),
    )
    map_centers = local_to_map(
        [item.center_m for item in obstacles], pose,
        float(route["fusion"]["local_forward_origin_offset_m"]),
    )
    expanded = (
        0.5 * float(physical["disk_diameter_m"])
        + 0.5 * float(physical["robot_width_m"])
        + float(config["safety"]["obstacle_margin_m"])
        + float(config["planner"]["collision_sample_padding_m"])
    )
    clearances = [polyline_clearance_to_point(fused, item) - expanded for item in map_centers]
    obstacle_clearance = min(clearances) if clearances else None
    safe = footprint["collision_free"] and (obstacle_clearance is None or obstacle_clearance >= 0.0)
    return {
        "safe": safe,
        "reason": "" if safe else "fused_map_collision",
        "selected": planning.selected.name,
        "map_footprint": footprint,
        "minimum_obstacle_clearance_m": obstacle_clearance,
        "fused_path_points": len(fused),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--route", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    image = cv2.imread(str(Path(args.image).expanduser()))
    if image is None:
        raise SystemExit("could not read image: {}".format(args.image))
    config = load_config(args.config)
    config["control"]["enable_motion"] = False
    pipeline = AvoidancePipeline(config)
    result = None
    repeats = max(3, int(config["disk_detection"]["confirmation_frames"]))
    for index in range(repeats):
        result = pipeline.process(
            image, index * 0.1, debug_image_names=set(), run_planning=True
        )
    route, route_path = load_yaml(args.route)
    occupancy, resolution, origin, _, _ = load_metric_map(route, route_path)
    reference = build_reference_path(route)
    start = route["start"]
    pose = (float(start["x_m"]), float(start["y_m"]), float(start["yaw_rad"]))
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    current_image, current_centers, _ = render_scene(
        route, occupancy, resolution, origin, reference, pose,
        result.planning, result.tracked_disks, config, "CURRENT 2-DISC FUSION",
    )
    cv2.imwrite(str(output / "fusion_current.png"), current_image)
    current_report = fused_validation(
        route, route_path, reference, pose, result.planning,
        result.tracked_disks, config,
    )

    demo_obstacles = result.tracked_disks[:1]
    demo_planning = LocalPathPlanner(config).plan(result.corridor, demo_obstacles)
    demo_image, demo_centers, _ = render_scene(
        route, occupancy, resolution, origin, reference, pose,
        demo_planning, demo_obstacles, config, "ONE-DISC VALIDATION",
    )
    cv2.imwrite(str(output / "fusion_single_disc_demo.png"), demo_image)
    demo_report = fused_validation(
        route, route_path, reference, pose, demo_planning, demo_obstacles, config,
    )
    report = {
        "motion_enabled": False,
        "pose_source": route["fusion"]["pose_source"],
        "robot_pose_map": list(pose),
        "current_scene": {
            "disc_count": len(result.tracked_disks),
            "map_disc_centers_m": current_centers.tolist(),
            "fusion": current_report,
        },
        "single_disc_validation": {
            "disc_count": len(demo_obstacles),
            "map_disc_centers_m": demo_centers.tolist(),
            "fusion": demo_report,
        },
    }
    with (output / "fusion_report.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
