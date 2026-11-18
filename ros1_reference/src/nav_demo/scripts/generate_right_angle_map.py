#!/usr/bin/env python3
"""Replace the upper-left stepped corner with a square corner."""

from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
MAP_DIR = ROOT / "map"
SOURCE = MAP_DIR / "map630_cropped.png"
OUTPUT = MAP_DIR / "map630_cropped_right_angle.png"


def main():
    image = cv2.imread(str(SOURCE), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError("Unable to read {}".format(SOURCE))

    # Remove only the stepped diagonal between the existing horizontal and
    # vertical walls, then reconnect both walls with a 90-degree corner.
    image[13:19, 44:51] = 255
    cv2.line(image, (44, 12), (50, 12), 0, 1, lineType=cv2.LINE_8)
    cv2.line(image, (50, 12), (50, 19), 0, 1, lineType=cv2.LINE_8)

    if not cv2.imwrite(str(OUTPUT), image):
        raise RuntimeError("Unable to write {}".format(OUTPUT))


if __name__ == "__main__":
    main()
