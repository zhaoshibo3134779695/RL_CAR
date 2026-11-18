#!/usr/bin/env python3
"""Round internal orthogonal corners in the competition forbidden map."""

from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
MAP_DIR = ROOT / "map"
SOURCE = MAP_DIR / "map630_cropped_right_angle.png"
OUTPUT = MAP_DIR / "map630_cropped_rounded_10cm.png"

# Keep the start barrier and the corner that was deliberately squared earlier.
PRESERVED_CORNERS = {(11, 12), (50, 12)}
CARDINAL_DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1))


def find_orthogonal_corners(occupied):
    height, width = occupied.shape
    corners = []
    for y in range(2, height - 2):
        for x in range(2, width - 2):
            if not occupied[y, x] or (x, y) in PRESERVED_CORNERS:
                continue

            directions = []
            for index, (dx, dy) in enumerate(CARDINAL_DIRECTIONS):
                if occupied[y + dy, x + dx]:
                    directions.append(index)

            if len(directions) != 2 or (directions[0] - directions[1]) % 2 == 0:
                continue
            if not all(
                occupied[
                    y + 2 * CARDINAL_DIRECTIONS[index][1],
                    x + 2 * CARDINAL_DIRECTIONS[index][0],
                ]
                for index in directions
            ):
                continue
            corners.append((x, y, directions))
    return corners


def main():
    image = cv2.imread(str(SOURCE), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError("Unable to read {}".format(SOURCE))

    occupied = image < 100
    corners = find_orthogonal_corners(occupied)

    for x, y, directions in corners:
        first = CARDINAL_DIRECTIONS[directions[0]]
        second = CARDINAL_DIRECTIONS[directions[1]]

        # Replace the two-pixel square corner with a short diagonal transition.
        image[y, x] = 255
        image[y + first[1], x + first[0]] = 255
        image[y + second[1], x + second[0]] = 255
        image[y + first[1] + second[1], x + first[0] + second[0]] = 0

    if not cv2.imwrite(str(OUTPUT), image):
        raise RuntimeError("Unable to write {}".format(OUTPUT))
    print("Rounded {} internal corners in {}".format(len(corners), OUTPUT))


if __name__ == "__main__":
    main()
