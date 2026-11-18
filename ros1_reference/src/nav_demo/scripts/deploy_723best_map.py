#!/usr/bin/env python3
"""Deploy the user-drawn 723best image without changing its geometry."""

from pathlib import Path
import shutil

import cv2
import yaml


ROOT = Path(__file__).resolve().parents[1]
MAP_DIR = ROOT / "map"
SOURCE = Path("/home/mowen/Desktop/723best.png")
SOURCE_COPY = MAP_DIR / "723best_source.png"
MAP_IMAGE = MAP_DIR / "723best_map.png"
MAP_YAML = MAP_DIR / "723best_map.yaml"

RESOLUTION = 0.005
ORIGIN = [-2.480489, -1.838045, 0.0]
INNER_WIDTH_METERS = 4.75
INNER_HEIGHT_METERS = 3.80
BOUNDARY_WIDTH_METERS = 0.03


def main():
    image = cv2.imread(str(SOURCE), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError("Unable to read {}".format(SOURCE))

    dark_y, dark_x = (image < 200).nonzero()
    if len(dark_x) == 0:
        raise RuntimeError("No map lines found in {}".format(SOURCE))

    left = int(dark_x.min())
    right = int(dark_x.max()) + 1
    top = int(dark_y.min())
    bottom = int(dark_y.max()) + 1
    cropped = image[top:bottom, left:right]

    expected_width = round(
        (INNER_WIDTH_METERS + 2 * BOUNDARY_WIDTH_METERS) / RESOLUTION
    )
    expected_height = round(
        (INNER_HEIGHT_METERS + 2 * BOUNDARY_WIDTH_METERS) / RESOLUTION
    )
    if cropped.shape[1] != expected_width:
        raise RuntimeError(
            "Expected map width {}, got {}".format(
                expected_width, cropped.shape[1]
            )
        )
    if cropped.shape[0] == expected_height + 1:
        # The exported source contains one extra blank row immediately below
        # the 3 cm top boundary. Removing it keeps all lower geometry fixed.
        boundary_pixels = round(BOUNDARY_WIDTH_METERS / RESOLUTION)
        cropped = cv2.vconcat(
            [
                cropped[:boundary_pixels],
                cropped[boundary_pixels + 1:],
            ]
        )
    elif cropped.shape[0] != expected_height:
        raise RuntimeError(
            "Expected map height {}, got {}".format(
                expected_height, cropped.shape[0]
            )
        )

    # Remove anti-aliasing only; line locations and shapes remain unchanged.
    _, binary = cv2.threshold(cropped, 127, 255, cv2.THRESH_BINARY)

    MAP_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(SOURCE), str(SOURCE_COPY))
    if not cv2.imwrite(str(MAP_IMAGE), binary):
        raise RuntimeError("Unable to write {}".format(MAP_IMAGE))

    metadata = {
        "image": MAP_IMAGE.name,
        "resolution": RESOLUTION,
        "origin": ORIGIN,
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.196,
    }
    with MAP_YAML.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(metadata, stream, sort_keys=False)

    print(
        "Deployed {}x{} map at {:.3f} m/pixel".format(
            binary.shape[1],
            binary.shape[0],
            RESOLUTION,
        )
    )
    print("Crop bounds: left={} top={} right={} bottom={}".format(
        left, top, right, bottom
    ))


if __name__ == "__main__":
    main()
