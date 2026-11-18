#!/usr/bin/env python3
"""Generate centimeter-grid maps from the measured competition geometry."""

from pathlib import Path

import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
MAP_DIR = ROOT / "map"
CONFIG_DIR = ROOT / "config"

OLD_WIDTH = 96
OLD_HEIGHT = 78
OLD_RESOLUTION = 0.049743
OLD_ORIGIN = (-2.486458, -1.856923)

WIDTH = 475
HEIGHT = 380
RESOLUTION = 0.01
LINE_WIDTH = 3

# Pixel-center coordinates on the old sketch mapped to measured centimeters.
# A 51 cm center-to-center gap gives a 48 cm clear passage with 3 cm lines.
X_SOURCE = np.array([0.0, 11.0, 22.0, 33.0, 40.0, 44.0,
                     62.0, 73.0, 84.0, 95.0])
X_TARGET = np.array([1.0, 52.0, 108.0, 159.0, 205.0, 217.0,
                     320.0, 371.0, 422.0, 473.0])
Y_SOURCE = np.array([1.0, 12.0, 23.0, 33.0, 42.0, 53.0,
                     54.0, 62.0, 65.0, 67.0, 77.0])
Y_TARGET = np.array([1.0, 52.0, 103.0, 154.0, 205.0, 256.0,
                     265.0, 307.0, 322.0, 327.0, 378.0])

INITIAL_POSE = (-2.188, 1.296422513659279)


def project_x(value):
    return float(np.interp(value, X_SOURCE, X_TARGET))


def project_y(value):
    return float(np.interp(value, Y_SOURCE, Y_TARGET))


def project_image(source_name, output_name):
    source = cv2.imread(str(MAP_DIR / source_name), cv2.IMREAD_GRAYSCALE)
    if source is None or source.shape != (OLD_HEIGHT, OLD_WIDTH):
        raise RuntimeError("Unexpected source map: {}".format(source_name))

    occupied = source < 128
    skeleton = np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)
    ys, xs = np.nonzero(occupied)

    for y, x in zip(ys, xs):
        point = (int(round(project_x(x))), int(round(project_y(y))))
        skeleton[point[1], point[0]] = 0
        for dx, dy in ((1, 0), (0, 1), (1, 1), (-1, 1)):
            nx, ny = x + dx, y + dy
            if (0 <= nx < OLD_WIDTH and 0 <= ny < OLD_HEIGHT and
                    occupied[ny, nx]):
                neighbor = (
                    int(round(project_x(nx))),
                    int(round(project_y(ny))),
                )
                cv2.line(skeleton, point, neighbor, 0, 1, cv2.LINE_8)

    mask = (skeleton == 0).astype(np.uint8)
    kernel = np.ones((LINE_WIDTH, LINE_WIDTH), dtype=np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    output = np.where(mask != 0, 0, 255).astype(np.uint8)
    output_path = MAP_DIR / output_name
    if not cv2.imwrite(str(output_path), output):
        raise RuntimeError("Unable to write {}".format(output_path))


def old_world_to_source_pixel(x, y):
    column = (x - OLD_ORIGIN[0]) / OLD_RESOLUTION - 0.5
    row = OLD_HEIGHT - 0.5 - (y - OLD_ORIGIN[1]) / OLD_RESOLUTION
    return column, row


def calculate_origin():
    old_x, old_y = old_world_to_source_pixel(*INITIAL_POSE)
    new_x = project_x(old_x)
    new_y = project_y(old_y)
    origin_x = INITIAL_POSE[0] - (new_x + 0.5) * RESOLUTION
    origin_y = INITIAL_POSE[1] - (HEIGHT - new_y - 0.5) * RESOLUTION
    return origin_x, origin_y


def old_world_to_new_world(x, y, origin):
    old_x, old_y = old_world_to_source_pixel(x, y)
    new_x = project_x(old_x)
    new_y = project_y(old_y)
    world_x = origin[0] + (new_x + 0.5) * RESOLUTION
    world_y = origin[1] + (HEIGHT - new_y - 0.5) * RESOLUTION
    return world_x, world_y


def write_map_yaml(image_name, yaml_name, origin):
    data = {
        "image": str((MAP_DIR / image_name).resolve()),
        "resolution": RESOLUTION,
        "origin": [round(origin[0], 6), round(origin[1], 6), 0.0],
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.196,
    }
    with open(MAP_DIR / yaml_name, "w") as stream:
        yaml.safe_dump(data, stream, sort_keys=False)


def transform_waypoints(origin):
    source_path = CONFIG_DIR / "7.20nicepoint.yaml"
    with open(source_path) as stream:
        route = yaml.safe_load(stream)

    for waypoint in route["waypoints"]:
        position = waypoint["position"]
        x, y = old_world_to_new_world(position["x"], position["y"], origin)
        position["x"] = round(x, 12)
        position["y"] = round(y, 12)

    output_path = CONFIG_DIR / "7.20nicepoint_precise_0721.yaml"
    with open(output_path, "w") as stream:
        yaml.safe_dump(route, stream, sort_keys=False)


def validate_geometry():
    checks = {
        "outer_width": (X_TARGET[-1] - X_TARGET[0]) + LINE_WIDTH,
        "outer_height": (Y_TARGET[-1] - Y_TARGET[0]) + LINE_WIDTH,
        "top_horizontal": X_TARGET[4] - X_TARGET[2],
        "center_vertical": Y_TARGET[-1] - Y_TARGET[6],
        "bottom_right_horizontal": X_TARGET[-1] - X_TARGET[5],
        "left_lower_vertical": Y_TARGET[-1] - Y_TARGET[8],
        "start_clear_passage": X_TARGET[1] - X_TARGET[0] - LINE_WIDTH,
        "bottom_clear_passage": Y_TARGET[-1] - Y_TARGET[9] - LINE_WIDTH,
    }
    expected = {
        "outer_width": 475.0,
        "outer_height": 380.0,
        "top_horizontal": 97.0,
        "center_vertical": 113.0,
        "bottom_right_horizontal": 256.0,
        "left_lower_vertical": 56.0,
        "start_clear_passage": 48.0,
        "bottom_clear_passage": 48.0,
    }
    for name, value in checks.items():
        if abs(value - expected[name]) > 1e-9:
            raise RuntimeError("{} is {} cm, expected {} cm".format(
                name, value, expected[name]))
        print("{}: {:.0f} cm".format(name, value))


def main():
    validate_geometry()
    origin = calculate_origin()
    project_image("map7_aligned.png", "map7_precise_0721.png")
    project_image("7.15.png", "forbidden_precise_0721.png")
    write_map_yaml("map7_precise_0721.png", "map7_precise_0721.yaml", origin)
    write_map_yaml(
        "forbidden_precise_0721.png", "forbidden_precise_0721.yaml", origin
    )
    transform_waypoints(origin)
    print("origin: [{:.6f}, {:.6f}, 0]".format(*origin))


if __name__ == "__main__":
    main()
