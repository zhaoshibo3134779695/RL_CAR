#!/usr/bin/env python3
"""Generate and validate the fixed-map route without commanding a vehicle."""

import argparse
import csv
import json
from pathlib import Path

import cv2

from yellow_track_avoidance.map_navigation import draw_route_preview, validate_route


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    route, _, occupancy, resolution, origin, path, headings, report = validate_route(
        args.route
    )
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    handoff_x = float(route["avoidance"]["handoff_x_m"])
    with (output / "reference_route.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["index", "x_m", "y_m", "yaw_rad", "disc_avoidance_enabled"])
        for index, ((x, y), yaw) in enumerate(zip(path, headings)):
            writer.writerow([index, x, y, yaw, bool(x > handoff_x)])
    with (output / "route_report.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    preview = draw_route_preview(route, occupancy, resolution, origin, path)
    if not cv2.imwrite(str(output / "route_preview.png"), preview):
        raise RuntimeError("failed to write route preview")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    print("offline navigation output: {}".format(output))


if __name__ == "__main__":
    main()
