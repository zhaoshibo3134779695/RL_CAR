#!/usr/bin/env python3
"""Scale an SBPL motion-primitive file to a different grid resolution."""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--resolution", type=float, required=True)
    parser.add_argument(
        "--add-holonomic",
        action="store_true",
        help="add short/long left and right translations at fixed heading",
    )
    parser.add_argument(
        "--preserve-metric",
        action="store_true",
        help=(
            "preserve primitive distances in metres by rescaling discrete "
            "end cells instead of metric intermediate poses"
        ),
    )
    return parser.parse_args()


def parse_primitives(lines):
    header = lines[:3]
    primitives = []
    index = 3
    while index < len(lines):
        primitive_id = int(lines[index].split(":", 1)[1])
        start_angle = int(lines[index + 1].split(":", 1)[1])
        end_pose = [
            int(value)
            for value in lines[index + 2].split(":", 1)[1].split()
        ]
        cost = int(lines[index + 3].split(":", 1)[1])
        pose_count = int(lines[index + 4].split(":", 1)[1])
        poses = [
            [float(value) for value in line.split()]
            for line in lines[index + 5:index + 5 + pose_count]
        ]
        primitives.append(
            {
                "id": primitive_id,
                "start": start_angle,
                "end": end_pose,
                "cost": cost,
                "poses": poses,
            }
        )
        index += 5 + pose_count
    return header, primitives


def holonomic_primitives(primitives, resolution):
    augmented = []
    start_angles = sorted({primitive["start"] for primitive in primitives})
    for start_angle in start_angles:
        group = [
            primitive
            for primitive in primitives
            if primitive["start"] == start_angle
        ]
        augmented.extend(group)
        by_id = {primitive["id"]: primitive for primitive in group}
        next_id = max(by_id) + 1
        for source_id in (0, 1):
            source = by_id[source_id]
            forward_x, forward_y = source["end"][:2]
            yaw = source["poses"][0][2]
            for lateral_x, lateral_y in (
                (-forward_y, forward_x),
                (forward_y, -forward_x),
            ):
                pose_count = len(source["poses"])
                poses = []
                for index in range(pose_count):
                    ratio = index / float(max(1, pose_count - 1))
                    poses.append(
                        [
                            ratio * lateral_x * resolution,
                            ratio * lateral_y * resolution,
                            yaw,
                        ]
                    )
                augmented.append(
                    {
                        "id": next_id,
                        "start": start_angle,
                        "end": [lateral_x, lateral_y, start_angle],
                        "cost": 2,
                        "poses": poses,
                    }
                )
                next_id += 1
    return augmented


def serialize(header, primitives, resolution):
    output = [
        "resolution_m: {:.6f}".format(resolution),
        header[1],
        "totalnumberofprimitives: {}".format(len(primitives)),
    ]
    for primitive in primitives:
        output.extend(
            [
                "primID: {}".format(primitive["id"]),
                "startangle_c: {}".format(primitive["start"]),
                "endpose_c: {} {} {}".format(*primitive["end"]),
                "additionalactioncostmult: {}".format(primitive["cost"]),
                "intermediateposes: {}".format(len(primitive["poses"])),
            ]
        )
        for x_m, y_m, yaw_rad in primitive["poses"]:
            output.append(
                "{:.6f} {:.6f} {:.6f}".format(x_m, y_m, yaw_rad)
            )
    return output


def main():
    args = parse_args()
    lines = args.source.read_text(encoding="utf-8").splitlines()
    if not lines or not lines[0].startswith("resolution_m:"):
        raise ValueError("source is not an SBPL motion-primitive file")
    source_resolution = float(lines[0].split(":", 1)[1])
    target_resolution = float(args.resolution)
    if source_resolution <= 0.0 or target_resolution <= 0.0:
        raise ValueError("resolutions must be positive")
    header, primitives = parse_primitives(lines)
    if args.preserve_metric:
        cell_scale = source_resolution / target_resolution
        for primitive in primitives:
            for index in (0, 1):
                scaled = primitive["end"][index] * cell_scale
                rounded = int(round(scaled))
                if abs(scaled - rounded) > 1.0e-9:
                    raise ValueError(
                        "target resolution cannot represent primitive end cell"
                    )
                primitive["end"][index] = rounded
    else:
        metric_scale = target_resolution / source_resolution
        for primitive in primitives:
            for pose in primitive["poses"]:
                pose[0] *= metric_scale
                pose[1] *= metric_scale
    if args.add_holonomic:
        primitives = holonomic_primitives(primitives, target_resolution)
    output = serialize(header, primitives, target_resolution)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_text(
        "\n".join(output) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
