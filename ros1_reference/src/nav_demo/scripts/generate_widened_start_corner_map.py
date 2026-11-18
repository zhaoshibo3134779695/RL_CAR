#!/usr/bin/env python3
"""Widen the two passages meeting at the lower-left RViz corner."""

from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MAP_DIR = ROOT / "map"
SOURCE = MAP_DIR / "map630_cropped_rounded_10cm.png"
OUTPUT = MAP_DIR / "map630_cropped_rounded_10cm_start_corner_widened.png"


def main():
    image = cv2.imread(str(SOURCE), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError("Unable to read {}".format(SOURCE))

    # RViz shows this source-image L rotated 90 degrees counterclockwise.
    # Move its vertical-looking side right and its lower side down by one cell.
    image[12, 11:51] = 255
    image[12:66, 11] = 255
    points = np.array(((50, 13), (10, 13), (10, 65)), np.int32)
    cv2.polylines(image, (points,), False, 0, 1, lineType=cv2.LINE_8)

    if not cv2.imwrite(str(OUTPUT), image):
        raise RuntimeError("Unable to write {}".format(OUTPUT))
    print("Wrote {}".format(OUTPUT))


if __name__ == "__main__":
    main()
