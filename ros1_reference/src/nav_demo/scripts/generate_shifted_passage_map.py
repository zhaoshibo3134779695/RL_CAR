#!/usr/bin/env python3
"""Shift the upper-left passage's short vertical wall right by one cell."""

from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MAP_DIR = ROOT / "map"
SOURCE = MAP_DIR / "map630_cropped_rounded_10cm.png"
OUTPUT = MAP_DIR / "map630_cropped_rounded_10cm_shifted.png"


def main():
    image = cv2.imread(str(SOURCE), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError("Unable to read {}".format(SOURCE))

    # Original wall: x=40, y=24..33. Move its main run to x=41 and
    # reconnect both rounded ends without leaving a gap in the barrier.
    image[24:34, 40:42] = 255
    points = np.array(((39, 23), (41, 24), (41, 33), (40, 34)), np.int32)
    cv2.polylines(image, (points,), False, 0, 1, lineType=cv2.LINE_8)

    if not cv2.imwrite(str(OUTPUT), image):
        raise RuntimeError("Unable to write {}".format(OUTPUT))
    print("Wrote {}".format(OUTPUT))


if __name__ == "__main__":
    main()
