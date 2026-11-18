#!/usr/bin/env python3
"""Generate 1 cm occupancy maps from the measured competition geometry."""

from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MAP_DIR = ROOT / "map"

# Source-pixel coordinates mapped to measured centimeters. The source topology
# is retained while the annotated dimensions become exact constraints.
X_SOURCE = np.array([0.0, 11.0, 22.0, 40.0, 44.0, 95.0])
X_CM = np.array([0.0, 51.0, 107.0, 204.0, 216.0, 475.0])
Y_SOURCE = np.array([0.0, 54.0, 65.0, 77.0])
Y_CM = np.array([0.0, 264.0, 321.0, 380.0])

PIXEL_OFFSET = 1
WIDTH = 478
HEIGHT = 383
LINE_WIDTH_PX = 3

# Rigid metric map dimensions.  The measured 4.75 x 3.80 m dimensions are
# outer-edge dimensions, so 3 cm lines have center spans of 4.72 x 3.77 m.
RIGID_WIDTH = 475
RIGID_HEIGHT = 380
RIGID_MIN_X = 0.0
RIGID_MAX_X = 95.0
RIGID_MIN_Y = 1.0
RIGID_MAX_Y = 77.0


def map_x(x):
    return int(round(np.interp(x, X_SOURCE, X_CM))) + PIXEL_OFFSET


def map_y(y):
    return int(round(np.interp(y, Y_SOURCE, Y_CM))) + PIXEL_OFFSET


def transform_map(source_name, output_name):
    source = cv2.imread(str(MAP_DIR / source_name), cv2.IMREAD_GRAYSCALE)
    if source is None:
        raise RuntimeError("Unable to read {}".format(source_name))

    occupied = source < 128
    skeleton = np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)
    ys, xs = np.nonzero(occupied)

    # Join neighboring source pixels after projection so long and diagonal
    # boundaries remain continuous at the higher resolution.
    for y, x in zip(ys, xs):
        p = (map_x(x), map_y(y))
        skeleton[p[1], p[0]] = 0
        for dx, dy in ((1, 0), (0, 1), (1, 1), (-1, 1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < source.shape[1] and 0 <= ny < source.shape[0] and occupied[ny, nx]:
                cv2.line(
                    skeleton,
                    p,
                    (map_x(nx), map_y(ny)),
                    0,
                    1,
                    lineType=cv2.LINE_8,
                )

    occupied_mask = (skeleton == 0).astype(np.uint8)
    kernel = np.ones((LINE_WIDTH_PX, LINE_WIDTH_PX), dtype=np.uint8)
    occupied_mask = cv2.dilate(occupied_mask, kernel, iterations=1)
    output = np.where(occupied_mask != 0, 0, 255).astype(np.uint8)

    if not cv2.imwrite(str(MAP_DIR / output_name), output):
        raise RuntimeError("Unable to write {}".format(output_name))


def rigid_x(x):
    center_span = RIGID_WIDTH - LINE_WIDTH_PX
    return int(round(1 + (x - RIGID_MIN_X) * center_span /
                     (RIGID_MAX_X - RIGID_MIN_X)))


def rigid_y(y):
    center_span = RIGID_HEIGHT - LINE_WIDTH_PX
    return int(round(1 + (y - RIGID_MIN_Y) * center_span /
                     (RIGID_MAX_Y - RIGID_MIN_Y)))


def transform_map_rigid(source_name, output_name):
    """Project the complete source with one global X/Y metric transform."""
    source = cv2.imread(str(MAP_DIR / source_name), cv2.IMREAD_GRAYSCALE)
    if source is None:
        raise RuntimeError("Unable to read {}".format(source_name))

    occupied = source < 128
    skeleton = np.full((RIGID_HEIGHT, RIGID_WIDTH), 255, dtype=np.uint8)
    ys, xs = np.nonzero(occupied)

    for y, x in zip(ys, xs):
        p = (rigid_x(x), rigid_y(y))
        skeleton[p[1], p[0]] = 0
        for dx, dy in ((1, 0), (0, 1), (1, 1), (-1, 1)):
            nx, ny = x + dx, y + dy
            if (0 <= nx < source.shape[1] and
                    RIGID_MIN_Y <= ny <= RIGID_MAX_Y and
                    occupied[ny, nx]):
                cv2.line(
                    skeleton,
                    p,
                    (rigid_x(nx), rigid_y(ny)),
                    0,
                    1,
                    lineType=cv2.LINE_8,
                )

    occupied_mask = (skeleton == 0).astype(np.uint8)
    kernel = np.ones((LINE_WIDTH_PX, LINE_WIDTH_PX), dtype=np.uint8)
    occupied_mask = cv2.dilate(occupied_mask, kernel, iterations=1)
    output = np.where(occupied_mask != 0, 0, 255).astype(np.uint8)

    if not cv2.imwrite(str(MAP_DIR / output_name), output):
        raise RuntimeError("Unable to write {}".format(output_name))


def main():
    transform_map("map7_aligned.png", "map7_competition_measured.png")
    transform_map("map630_cropped.png", "map630_competition_measured.png")
    transform_map_rigid("map7_aligned.png", "map7_competition_rigid.png")
    transform_map_rigid("map630_cropped.png", "map630_competition_rigid.png")


if __name__ == "__main__":
    main()
