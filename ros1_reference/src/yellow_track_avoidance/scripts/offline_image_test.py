#!/usr/bin/env python3
"""Run the complete core on one image without importing ROS or sending motion."""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from yellow_track_avoidance.config import load_config
from yellow_track_avoidance.pipeline import AvoidancePipeline


def _finite(value):
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _report(result):
    planning = result.planning
    return {
        "stamp": result.stamp_sec,
        "state": result.state.value,
        "stop_reason": result.stop_reason,
        "diagnostics": result.diagnostics,
        "boundaries": None if result.boundary is None else {
            "left_detected": result.boundary.left_detected,
            "right_detected": result.boundary.right_detected,
            "used_history": result.boundary.used_history,
            "quality": result.boundary.quality,
            "lane_width_px": result.boundary.lane_width_px,
            "left_px": [_finite(value) for value in result.boundary.left_px],
            "right_px": [_finite(value) for value in result.boundary.right_px],
            "rows_px": result.boundary.rows_px.tolist(),
        },
        "disks": [
            {
                "track_id": disk.track_id,
                "center_px": list(disk.center_px),
                "center_m": list(disk.center_m),
                "estimated_diameter_m": disk.estimated_diameter_m,
                "confidence": disk.confidence,
                "hits": disk.hits,
            }
            for disk in result.tracked_disks
        ],
        "planning": None if planning is None else {
            "safe": planning.safe,
            "reason": planning.reason,
            "selected": None if planning.selected is None else planning.selected.name,
            "left_space_m": _finite(planning.left_space_m),
            "right_space_m": _finite(planning.right_space_m),
            "candidates": [
                {
                    "name": candidate.name,
                    "collision_free": candidate.collision_free,
                    "score": _finite(candidate.score),
                    "reject_reason": candidate.reject_reason,
                    "minimum_boundary_clearance_m": _finite(candidate.minimum_boundary_clearance_m),
                    "minimum_obstacle_clearance_m": _finite(candidate.minimum_obstacle_clearance_m),
                    "max_curvature": candidate.max_curvature,
                    "points_m": candidate.points_m.tolist(),
                }
                for candidate in planning.candidates
            ],
        },
        "command": {
            "linear_x": result.command.linear_x,
            "linear_y": result.command.linear_y,
            "angular_z": result.command.angular_z,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="input image path")
    parser.add_argument("--config", required=True, help="calibrated YAML configuration")
    parser.add_argument("--output-dir", required=True, help="directory for debug images and result.json")
    parser.add_argument("--repeat-frames", type=int, default=0, help="repeat the image to exercise temporal confirmation")
    args = parser.parse_args()

    image = cv2.imread(str(Path(args.image).expanduser()))
    if image is None:
        raise SystemExit("could not read image: {}".format(args.image))
    config = load_config(args.config)
    config["control"]["enable_motion"] = False
    pipeline = AvoidancePipeline(config)
    pipeline.reset_stop()
    repeats = args.repeat_frames or max(
        int(config["disk_detection"]["confirmation_frames"]),
        int(config["safety"]["stop_recovery_frames"]),
    )
    result = None
    for index in range(max(1, repeats)):
        result = pipeline.process(image, index * 0.1)
    assert result is not None

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name, debug_image in result.debug_images.items():
        cv2.imwrite(str(output / (name + ".png")), debug_image)
    with (output / "result.json").open("w", encoding="utf-8") as stream:
        json.dump(_report(result), stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(_report(result), ensure_ascii=False, indent=2, allow_nan=False))
    print("debug output: {}".format(output))


if __name__ == "__main__":
    main()
